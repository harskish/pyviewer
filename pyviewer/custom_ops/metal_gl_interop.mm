#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>

using at::ScalarType;
using torch::indexing::Slice;
using namespace at::mps;

typedef struct {
    int             cvPixelFormat;
    MTLPixelFormat  mtlFormat;
    GLuint          glInternalFormat;
    GLuint          glFormat;
    GLuint          glType;
} TexFormatInfo;

// Table of equivalent formats across CoreVideo, Metal, and OpenGL
static const TexFormatInfo InteropFormatTable[] = {
    // Core Video Pixel Format,               Metal Pixel Format,            GL internalformat, GL format,   GL type
    // BGR
    { kCVPixelFormatType_32BGRA,              MTLPixelFormatBGRA8Unorm,      GL_RGBA,           GL_BGRA_EXT, GL_UNSIGNED_INT_8_8_8_8_REV },
    { kCVPixelFormatType_ARGB2101010LEPacked, MTLPixelFormatBGR10A2Unorm,    GL_RGB10_A2,       GL_BGRA,     GL_UNSIGNED_INT_2_10_10_10_REV },
    { kCVPixelFormatType_32BGRA,              MTLPixelFormatBGRA8Unorm_sRGB, GL_SRGB8_ALPHA8,   GL_BGRA,     GL_UNSIGNED_INT_8_8_8_8_REV },
    // RGB
    { kCVPixelFormatType_64RGBAHalf,          MTLPixelFormatRGBA16Float,     GL_RGBA,           GL_RGBA,     GL_HALF_FLOAT },
};

const TexFormatInfo *const textureFormatInfoFromMetalPixelFormat(MTLPixelFormat pixelFormat) {
    for(int i = 0; i < (sizeof(InteropFormatTable) / sizeof(TexFormatInfo)); i++) {
        if(pixelFormat == InteropFormatTable[i].mtlFormat) {
            return &InteropFormatTable[i];
        }
    }
    return NULL;
}

typedef struct {
    CGSize size;
    const TexFormatInfo *formatInfo;
    CVPixelBufferRef CVPixelBuffer;
    CVMetalTextureRef CVMTLTexture;
    CVOpenGLTextureCacheRef CVGLTextureCache;
    CVOpenGLTextureRef CVGLTexture;
    CGLPixelFormatObj CGLPixelFormat;
    CVMetalTextureCacheRef CVMTLTextureCache;
    MTLPixelFormat mtlFormat;
    id<MTLDevice> metalDevice;
    id<MTLTexture> metalTexture;
    NSOpenGLContext* openGLContext;
    GLuint openGLTexture;
} InteropTextureState;

static InteropTextureState state = {
    .size = {0.0, 0.0},  // { W, H }
    // other fields zero-initialized
};

static id<MTLLibrary> gInteropLib = nil;
static id<MTLRenderPipelineState> gInteropPipeline = nil;
static id<MTLSamplerState> gInteropSampler = nil;
static MTLPixelFormat gInteropPipelineFormat = MTLPixelFormatInvalid;

static void createGlTexture() {
    CVReturn cvret;
    cvret  = CVOpenGLTextureCacheCreate(
        kCFAllocatorDefault,
        nil,
        state.openGLContext.CGLContextObj,
        state.CGLPixelFormat,
        nil,
        &state.CVGLTextureCache);
    TORCH_CHECK(cvret == kCVReturnSuccess, "Failed to create OpenGL Texture Cache");

    cvret = CVOpenGLTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        state.CVGLTextureCache,
        state.CVPixelBuffer,
        nil,
        &state.CVGLTexture);
    TORCH_CHECK(cvret == kCVReturnSuccess, "Failed to create OpenGL Texture From Image");

    state.openGLTexture = CVOpenGLTextureGetName(state.CVGLTexture);
}

static void createMetalTexture() {
    CVReturn cvret;
    cvret = CVMetalTextureCacheCreate(
        kCFAllocatorDefault,
        nil,
        state.metalDevice,
        nil,
        &state.CVMTLTextureCache);
    TORCH_CHECK(cvret == kCVReturnSuccess, "Failed to create Metal texture cache");

    cvret = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        state.CVMTLTextureCache,
        state.CVPixelBuffer, nil,
        state.formatInfo->mtlFormat,
        state.size.width, state.size.height,
        0,
        &state.CVMTLTexture);
    TORCH_CHECK(cvret == kCVReturnSuccess, "Failed to create CoreVideo Metal texture from image");

    state.metalTexture = CVMetalTextureGetTexture(state.CVMTLTexture);
    TORCH_CHECK(state.metalTexture, "Failed to create Metal texture CoreVideo Metal Texture");
}

static void interopTextureRelease() {
    if (state.CVPixelBuffer) {
        CVPixelBufferRelease(state.CVPixelBuffer);
        state.CVPixelBuffer = NULL;
    }
    if (state.CVMTLTexture) {
        CFRelease(state.CVMTLTexture);
        state.CVMTLTexture = NULL;
    }
    if (state.CVGLTexture) {
        CFRelease(state.CVGLTexture);
        state.CVGLTexture = NULL;
    }
    if (state.CVGLTextureCache) {
        CFRelease(state.CVGLTextureCache);
        state.CVGLTextureCache = NULL;
    }
    if (state.CVMTLTextureCache) {
        CFRelease(state.CVMTLTextureCache);
        state.CVMTLTextureCache = NULL;
    }
    state.openGLTexture = 0;
    state.metalTexture = nil;
}

static void interopTextureInit(id<MTLDevice> metalDevice,
                                 NSOpenGLContext* glContext,
                                 MTLPixelFormat mtlPixelFormat,
                                 CGSize size) {
    state.mtlFormat = mtlPixelFormat;
    state.formatInfo = textureFormatInfoFromMetalPixelFormat(mtlPixelFormat);
    TORCH_CHECK(state.formatInfo, "Unsupported Metal format");

    state.size = size;
    state.metalDevice = metalDevice;
    state.openGLContext = glContext;
    state.CGLPixelFormat = glContext.pixelFormat.CGLPixelFormatObj;

    NSDictionary* cvBufferProperties = @{
        (__bridge NSString*)kCVPixelBufferOpenGLCompatibilityKey : @YES,
        (__bridge NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
    };

    CVReturn cvret = CVPixelBufferCreate(kCFAllocatorDefault,
                                         size.width, size.height,
                                         state.formatInfo->cvPixelFormat,
                                         (__bridge CFDictionaryRef)cvBufferProperties,
                                         &state.CVPixelBuffer);
    TORCH_CHECK(cvret == kCVReturnSuccess, "Failed to create CVPixelBuffer");

    createGlTexture();
    createMetalTexture();

    // For shader variant of blit
    if (!gInteropPipeline || gInteropPipelineFormat != mtlPixelFormat) {
        gInteropPipelineFormat = mtlPixelFormat;
        gInteropPipeline = nil; // Release old pipeline

        NSString *shaderPath = @"/Users/erik/code/pyviewer/pyviewer/custom_ops/shaders.metal";
        NSError *readErr = nil;
        NSString *shaderSrc = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&readErr];
        TORCH_CHECK(shaderSrc != nil, "Failed to read Metal shader file: ", readErr.localizedDescription);

        NSError *compileErr = nil;
        gInteropLib = [metalDevice newLibraryWithSource:shaderSrc options:nil error:&compileErr];
        TORCH_CHECK(gInteropLib != nil, "Failed to compile Metal shader: ", compileErr.localizedDescription);

        id<MTLFunction> vtxFn = [gInteropLib newFunctionWithName:@"vtx_passthrough"];
        id<MTLFunction> fragFn = [gInteropLib newFunctionWithName:@"frag_copy"];

        MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vtxFn;
        desc.fragmentFunction = fragFn;
        desc.colorAttachments[0].pixelFormat = mtlPixelFormat;
        gInteropPipeline = [metalDevice newRenderPipelineStateWithDescriptor:desc error:nil];

        if (!gInteropSampler) {
            MTLSamplerDescriptor *sampDesc = [[MTLSamplerDescriptor alloc] init];
            sampDesc.minFilter = MTLSamplerMinMagFilterNearest;
            sampDesc.magFilter = MTLSamplerMinMagFilterNearest;
            sampDesc.rAddressMode = MTLSamplerAddressModeClampToEdge; // default
            sampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge; // default
            sampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge; // default
            gInteropSampler = [metalDevice newSamplerStateWithDescriptor:sampDesc];
        }
    }
}

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void copy_rasterize(const torch::Tensor &inputHwc) {
    MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_queue_t serialQueue = stream->queue();

    // Figure out compatible formats
    // Floating point formats supported natively
    // Integer formats: only UInt8 supported
    MTLPixelFormat mtlFormat;
    ScalarType dtype = inputHwc.scalar_type();
    const bool isFp = isFloatingType(dtype);
    size_t nbits = c10::elementSize(dtype) * 8;
    
    torch::Tensor data = inputHwc;
    if (isFp && (nbits >= 32)) {
        // fp64 and fp32 => fp32
        data = data.toType(ScalarType::Float);
        mtlFormat = MTLPixelFormatRGBA32Float;
    } else if (isFp && (nbits <= 16)) {
        // fp16 and bfloat16 => fp16
        data = data.toType(ScalarType::Half);
        mtlFormat = MTLPixelFormatRGBA16Float;
    } else if (dtype == ScalarType::Byte) {
        // Integer data: only UInt8 for now
        mtlFormat = MTLPixelFormatRGBA8Unorm_sRGB;
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", dtype);
    }

    dispatch_sync(serialQueue, ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();

            id<MTLDevice> device = state.metalDevice;
            id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();

            const NSUInteger H = data.size(0);
            const NSUInteger W = data.size(1);
            MTLTextureDescriptor *srcDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtlFormat
                                                                                               width:W
                                                                                              height:H
                                                                                           mipmapped:NO];
            
            // Creating texture from linear buffer: some restrictions apply!
            // https://developer.apple.com/documentation/metal/mtlbuffer/maketexture(descriptor:offset:bytesperrow:)?language=objc
            srcDesc.usage = MTLTextureUsageShaderRead;
            id<MTLBuffer> srcBuffer = getMTLBufferStorage(data);
            NSUInteger bytesPerRow = W * 4 * c10::elementSize(data.scalar_type()); // RGBA
            id<MTLTexture> srcTexture = [srcBuffer newTextureWithDescriptor:srcDesc
                                                                     offset:0
                                                                bytesPerRow:bytesPerRow];

            MTLRenderPassDescriptor *rpDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            rpDesc.colorAttachments[0].texture = state.metalTexture;
            rpDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            rpDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:rpDesc];
            [encoder setRenderPipelineState:gInteropPipeline];
            [encoder setFragmentTexture:srcTexture atIndex:0];
            [encoder setFragmentSamplerState:gInteropSampler atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
            [encoder endEncoding];

            torch::mps::commit();
        }
    });
}

void copy_blit(const torch::Tensor &inputHwc) {
    const NSUInteger H = inputHwc.size(0);
    const NSUInteger W = inputHwc.size(1);

    // Blit approach: incoming data must be converted to compatible format
    // TODO: maybe bytesPerRow can be used to convert fp32 => fp16?
    TORCH_CHECK(isFloatingType(inputHwc.scalar_type()), "Blit: input tensor must be a floating point type"); // Float, Double, Half, Bfloat16 etc.
    TORCH_CHECK(state.mtlFormat == MTLPixelFormatRGBA16Float, "Blit: non-float internal mtl format");
    torch::Tensor inData = inputHwc.toType(ScalarType::Half);
    
    MPSStream *stream = at::mps::getCurrentMPSStream();

    // https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSStream.h#L69
    //dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
    dispatch_queue_t serialQueue = stream->queue(); // returns _serialQueue

    // Submits a block object for execution and returns after that block finishes executing.
    dispatch_sync(serialQueue, ^() {
        @autoreleasepool {
            stream->endKernelCoalescing(); // like MPSStream::fill / MPSStream::copy
            
            id<MTLBlitCommandEncoder> blitEncoder = [stream->commandBuffer() blitCommandEncoder];
            TORCH_CHECK(blitEncoder, "Failed to create blit command encoder");

            int bytesPerRow = W * 8; // fp16xRGBA = 4*16b = 8B
            id<MTLBuffer> buf = getMTLBufferStorage(inData);
            [blitEncoder copyFromBuffer:buf
                           sourceOffset:0
                      sourceBytesPerRow:bytesPerRow
                    sourceBytesPerImage:bytesPerRow * H
                             sourceSize:{W, H, 1}
                              toTexture:state.metalTexture // directly into interop texture
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:{0, 0, 0}];
            [blitEncoder endEncoding];
            
            // 1. http://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/api/src/mps.cpp#L27     # at::detail::getMPSHooks().commitStream();
            // 2. http://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSHooks.mm#L81  # at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT);
            // 3. http://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSStream.mm#L73 # endKernelCoalescing() => [commandBuffer() commitAndContinue] or [_commandBuffer commit]
            
            torch::mps::commit(); // SyncType::COMMIT
            //torch::mps::synchronize(); // SyncType::COMMIT_AND_WAIT

            //auto currBuffer = stream->commandBuffer();
            
            // NONE:                no commit to command buffer
            // COMMIT:              commit and flush the command buffer
            // COMMIT_AND_WAIT:     flush and wait for command buffer execution to finish
            // COMMIT_AND_CONTINUE: commit and continue with a new underlying command buffer
            // COMMIT_ADAPTIVE:     commit adaptively based on available memory
            
            //stream->synchronize(SyncType::COMMIT_AND_WAIT);
            //[currBuffer commitAndContinue];
            
            //[currBuffer waitUntilScheduled]; // hangs if using torch::mps::commit()
            //[commandBuffer waitUntilCompleted];
        }
    });
}

std::tuple<uint64_t, bool> toGlTex(const torch::Tensor &imgHwc) {
    TORCH_CHECK(imgHwc.device().is_mps(), "Input must be an MPS tensor");
    TORCH_CHECK(imgHwc.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(imgHwc.size(2) == 4, "Input must be RGBA");
    
    const ScalarType dtype = imgHwc.scalar_type();
    bool isFp = isFloatingType(dtype); // Float, Double, Half, Bfloat16 etc.
    TORCH_CHECK(isFp || dtype == ScalarType::Byte, "Only UInt8 or floating-point inputs currently supported");
    
    NSOpenGLContext* context = [NSOpenGLContext currentContext];
    TORCH_CHECK(context != nil, "No active OpenGL context");

    // Choice of shared Metal texture format quite limited (see InteropFormatTable)
    MTLPixelFormat mtlFmtShared = (isFp) ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm_sRGB;
    
    // Create new shared texture if dtype or shape changes
    bool updated = false;
    CGSize inSize = {CGFloat(imgHwc.size(1)), CGFloat(imgHwc.size(0))}; // W, H
    if (inSize.width != state.size.width || inSize.height != state.size.height || mtlFmtShared != state.mtlFormat) {
        interopTextureRelease();
        id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
        interopTextureInit(device, context, mtlFmtShared, inSize);
        updated = true;
    }

    // Blit approach: only BGR is suppored if incoming data is UInt
    // Shader approach: can take in RGB, convert to BGR during write
    
    bool useBlit = false; //isFp; // blit only supports RGB for float data
    if (useBlit)
        copy_blit(imgHwc);
    else
        copy_rasterize(imgHwc);
    
    // "In general, you need to execute glFlush() for Metal to get the results of the OpenGL rendering and you need to execute
    // [MTLCommandBuffer commit] and [MTLCommandBuffer waitUntilScheduled] for OpenGL to get the result of Metal rendering."
    //  - https://developer.apple.com/forums/thread/694201

    TORCH_CHECK(glGetError() == GL_NO_ERROR, "GL error flag was set");
    return { uint64_t(state.openGLTexture), updated };
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gl_tex_rect", &toGlTex, "Get OpenGL TEXTURE_RECTANGLE with contents of MPS tensor", py::arg("imgHwc"));
}
