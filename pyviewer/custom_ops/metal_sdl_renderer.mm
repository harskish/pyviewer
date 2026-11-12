// Native texture views and GPU resource completion fences. No SDL headers or patched SDL APIs.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.h>
#include <objc/runtime.h>
#include <pybind11/numpy.h>

// Metal retains bound resources until the SDL command buffer completes. Tie
// the allocator's Tensor lifetime to the texture, not just to a Python wrapper.
@interface MPSStorageOwner : NSObject {
@public
    torch::Tensor tensor;
}
@end
@implementation MPSStorageOwner
@end

@interface SubmissionOwner : NSObject {
@public
    dispatch_semaphore_t completed;
}
@end
@implementation SubmissionOwner
- (void)dealloc { dispatch_semaphore_signal(completed); }
@end

static char storage_key, submission_key;

static void finish_mps_work() {
    auto* stream = at::mps::getCurrentMPSStream();
    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
    __block id<MTLCommandBuffer> completed = nil;
    // Python's MPS synchronize holds the GIL but bypasses the stream queue.
    // Commit under the GIL; release it only for the completed-handler wait.
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
        stream->endKernelCoalescing();
        [stream->commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            completed = buffer;
            dispatch_semaphore_signal(finished);
        }];
        stream->synchronize(at::mps::SyncType::COMMIT);
    });
    {
        pybind11::gil_scoped_release release;
        dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
    }
    TORCH_CHECK(completed.status != MTLCommandBufferStatusError,
                "MPS producer failed: ", completed.error.localizedDescription.UTF8String);
}

class TextureView {
    torch::Tensor data_;
    id<MTLTexture> texture_;
    bool copied_ = false;
public:
    TextureView(const torch::Tensor& input, uint64_t device_address) {
        TORCH_CHECK(input.device().is_mps() && input.dim() == 3 && input.size(2) == 4,
                    "Expected MPS HWC RGBA input");
        TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
                    "Expected float16 or float32");
        TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && device_address, "Invalid texture dimensions/device");
        @autoreleasepool {
            id<MTLDevice> device = (__bridge id<MTLDevice>)(void*)device_address;
            TORCH_CHECK(device.registryID == at::mps::getCurrentMPSStream()->device().registryID,
                        "SDL and MPS must use the same Metal device");
            const NSUInteger height = input.size(0), width = input.size(1);
            const NSUInteger pixel_bytes = input.element_size() * 4;
            const MTLPixelFormat format = input.scalar_type() == at::kHalf
                ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA32Float;
            const NSUInteger alignment = [device minimumLinearTextureAlignmentForPixelFormat:format];
            NSUInteger row_bytes = input.stride(0) * input.element_size();
            NSUInteger offset = input.storage_offset() * input.element_size();
            data_ = input;
            if (input.stride(2) != 1 || input.stride(1) != 4 || row_bytes < width * pixel_bytes ||
                    row_bytes % alignment || offset % alignment) {
                row_bytes = ((width * pixel_bytes + alignment - 1) / alignment) * alignment;
                data_ = torch::empty({(int64_t)height, (int64_t)(row_bytes / pixel_bytes), 4}, input.options());
                data_.slice(1, 0, width).copy_(input);
                offset = 0;
                copied_ = true;
            }
            id<MTLBuffer> buffer = __builtin_bit_cast(id<MTLBuffer>, data_.storage().data());
            MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
            descriptor.usage = MTLTextureUsageShaderRead;
            descriptor.storageMode = buffer.storageMode;
            descriptor.cpuCacheMode = buffer.cpuCacheMode;
            texture_ = [buffer newTextureWithDescriptor:descriptor offset:offset bytesPerRow:row_bytes];
            TORCH_CHECK(texture_, "Metal cannot create a texture view over this MPS allocation");
            MPSStorageOwner* owner = [MPSStorageOwner new];
            owner->tensor = data_;
            objc_setAssociatedObject(texture_, &storage_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            finish_mps_work();
        }
    }
    uint64_t pointer() const { return (uint64_t)(__bridge void*)texture_; }
    bool copied() const { return copied_; }
};

class FrameFence {
    dispatch_semaphore_t completed_;
public:
    FrameFence(dispatch_semaphore_t completed) : completed_(completed) {}
    void wait() {
        pybind11::gil_scoped_release release;
        dispatch_semaphore_wait(completed_, DISPATCH_TIME_FOREVER);
        // Multiple waits are safe.
        dispatch_semaphore_signal(completed_);
    }
};

class RenderContext {
    CAMetalLayer* layer_;
    id<MTLRenderPipelineState> pipeline_;
    id<MTLSamplerState> nearest_, linear_;
public:
    RenderContext(uint64_t layer_address) {
        TORCH_CHECK(layer_address, "SDL did not expose a Metal layer");
        @autoreleasepool {
            layer_ = (__bridge CAMetalLayer*)(void*)layer_address;
            NSString* source = @R"metal(
                #include <metal_stdlib>
                using namespace metal;
                struct Vertex { float2 pos; float2 uv; float4 color; };
                struct Options { float2 size; float white; uint hdr; uint encoding; float output_white; };
                struct Varying { float4 pos [[position]]; float2 uv; float4 color; };
                float3 srgb_to_linear(float3 x) {
                    return select(x/12.92f, pow(max((x+.055f)/1.055f,0.0f),float3(2.4f)), x>.04045f);
                }
                float3 linear_to_srgb(float3 x) {
                    return select(12.92f*x,1.055f*pow(max(x,0.0f),float3(1.0f/2.4f))-.055f,x>.0031308f);
                }
                vertex Varying image_vertex(uint i [[vertex_id]], constant Vertex* v [[buffer(0)]],
                                             constant Options& p [[buffer(1)]]) {
                    Varying out;
                    out.pos=float4(v[i].pos/p.size*float2(2,-2)+float2(-1,1),0,1);
                    out.uv=v[i].uv;
                    out.color=float4(srgb_to_linear(v[i].color.rgb),v[i].color.a);
                    return out;
                }
                fragment float4 image_fragment(Varying in [[stage_in]], texture2d<float> image [[texture(0)]],
                                               sampler sample_image [[sampler(0)]], constant Options& p [[buffer(1)]]) {
                    float4 value=image.sample(sample_image,in.uv);
                    if (p.encoding==1) value.rgb=srgb_to_linear(value.rgb);
                    else if (p.encoding==2) {
                        float3 a=pow(clamp(value.rgb,0.0f,1.0f),float3(32.0f/2523.0f));
                        float3 nits=10000.0f*pow(max(a-3424.0f/4096.0f,0.0f)/
                            max(2413.0f/128.0f-2392.0f/128.0f*a,1e-6f),float3(16384.0f/2610.0f));
                        value.rgb=float3x3(float3(1.660491,-.124550,-.018151),float3(-.587641,1.132900,-.100579),
                                           float3(-.072850,-.008349,1.118730))*nits/p.white;
                    }
                    value*=in.color;
                    if (p.hdr) value.rgb*=p.output_white;
                    else value.rgb=linear_to_srgb(value.rgb);
                    return value;
                }
            )metal";
            NSError* error = nil;
            id<MTLLibrary> library = [layer_.device newLibraryWithSource:source options:nil error:&error];
            TORCH_CHECK(library, "Metal image shader failed: ", error.localizedDescription.UTF8String);
            MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
            descriptor.vertexFunction = [library newFunctionWithName:@"image_vertex"];
            descriptor.fragmentFunction = [library newFunctionWithName:@"image_fragment"];
            auto attachment = descriptor.colorAttachments[0];
            attachment.pixelFormat = layer_.pixelFormat;
            attachment.blendingEnabled = YES;
            attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
            attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            pipeline_ = [layer_.device newRenderPipelineStateWithDescriptor:descriptor error:&error];
            TORCH_CHECK(pipeline_, "Metal image pipeline failed: ", error.localizedDescription.UTF8String);
            MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
            sampler.sAddressMode = sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
            sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterNearest;
            nearest_ = [layer_.device newSamplerStateWithDescriptor:sampler];
            sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterLinear;
            linear_ = [layer_.device newSamplerStateWithDescriptor:sampler];
        }
    }
    uint64_t device() const { return (uint64_t)(__bridge void*)layer_.device; }
    std::shared_ptr<FrameFence> draw(uint64_t encoder_address, const TextureView& view,
        pybind11::array_t<float, pybind11::array::c_style> vertices, int width, int height,
        int x, int y, int w, int h, bool nearest, bool hdr, int encoding, float white, float output_white) {
        TORCH_CHECK(encoder_address && vertices.ndim()==2 && vertices.shape(1)==8, "Invalid Metal image geometry");
        struct alignas(8) Options { float size[2]; float white; uint32_t hdr, encoding; float output_white; };
        @autoreleasepool {
            id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)(void*)encoder_address;
            id<MTLTexture> texture = (__bridge id<MTLTexture>)(void*)view.pointer();
            TORCH_CHECK(encoder.device.registryID == texture.device.registryID, "Metal device mismatch");
            Options options = {{float(width),float(height)},white,uint32_t(hdr),uint32_t(encoding),output_white};
            id<MTLBuffer> params = [layer_.device newBufferWithBytes:&options length:sizeof(options)
                options:MTLResourceStorageModeShared];
            SubmissionOwner* owner = [SubmissionOwner new];
            owner->completed = dispatch_semaphore_create(0);
            auto fence = std::make_shared<FrameFence>(owner->completed);
            objc_setAssociatedObject(params, &submission_key, owner, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            [encoder setRenderPipelineState:pipeline_];
            [encoder setViewport:MTLViewport{0,0,double(width),double(height),0,1}];
            [encoder setScissorRect:MTLScissorRect{NSUInteger(x),NSUInteger(y),NSUInteger(w),NSUInteger(h)}];
            if (vertices.nbytes() <= 4096) {
                [encoder setVertexBytes:vertices.data() length:vertices.nbytes() atIndex:0];
            } else {
                id<MTLBuffer> geometry = [layer_.device newBufferWithBytes:vertices.data()
                    length:vertices.nbytes() options:MTLResourceStorageModeShared];
                [encoder setVertexBuffer:geometry offset:0 atIndex:0];
            }
            [encoder setVertexBuffer:params offset:0 atIndex:1];
            [encoder setFragmentBuffer:params offset:0 atIndex:1];
            [encoder setFragmentTexture:texture atIndex:0];
            [encoder setFragmentSamplerState:nearest?nearest_:linear_ atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:vertices.shape(0)];
            return fence;
        }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<TextureView>(module, "TextureView")
        .def(pybind11::init<const torch::Tensor&, uint64_t>())
        .def_property_readonly("pointer", &TextureView::pointer)
        .def_property_readonly("copied", &TextureView::copied);
    pybind11::class_<FrameFence, std::shared_ptr<FrameFence>>(module, "FrameFence")
        .def("wait", &FrameFence::wait);
    pybind11::class_<RenderContext>(module, "RenderContext")
        .def(pybind11::init<uint64_t>())
        .def_property_readonly("device", &RenderContext::device)
        .def("draw", &RenderContext::draw);
}
