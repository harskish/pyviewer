#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>

using at::ScalarType;
using torch::indexing::Slice;

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
}

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Interop texture state
static MTLPixelFormat curMtlFormat;

uint64_t toGlTex(const torch::Tensor &imgHwc) {
    TORCH_CHECK(imgHwc.device().is_mps(), "Input must be an MPS tensor");
    TORCH_CHECK(imgHwc.is_contiguous(), "Input must be contiguous");

    const int64_t H = imgHwc.size(0);
    const int64_t W = imgHwc.size(1);
    const int64_t C = imgHwc.size(2);
    TORCH_CHECK(C == 4, "Input must be RGBA");
    
    const ScalarType dtype = imgHwc.scalar_type();
    bool isSigned = isSignedType(dtype);
    bool isFp = isFloatingType(dtype); // Float, Double, Half, Bfloat16 etc.

    TORCH_CHECK(isFp || dtype == ScalarType::Byte, "Only UInt8 or floating-point inputs supported");
    
    NSOpenGLContext* context = [NSOpenGLContext currentContext];
    TORCH_CHECK(context != nil, "No active OpenGL context");

    CGSize inSize = {CGFloat(W), CGFloat(H)};
    torch::Tensor inData = imgHwc.toType((isFp) ? ScalarType::Half : ScalarType::Byte);
    
    // Metal-GL interop (via CoreVideo) only seems to support BGR-order UInt textures
    // => must swap order if input is non-fp
    // => TODO: probably faster to create explicit blit shader that does the swapping
    MTLPixelFormat inFmt = (isFp) ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm_sRGB;
    if (!isFp) {
        inData = torch::cat({
            inData.index({Slice(), Slice(), Slice(2, 3)}), // B
            inData.index({Slice(), Slice(), Slice(1, 2)}), // G
            inData.index({Slice(), Slice(), Slice(0, 1)}), // R
            inData.index({Slice(), Slice(), Slice(3, 4)}), // A
        }, 2);
    }
    
    TORCH_CHECK(inData.is_contiguous(), "Input non-contiguous after internal transformations");
    
    // Create new shared texture if dtype or shape changes
    if (inSize.width != state.size.width || inSize.height != state.size.height || inFmt != state.mtlFormat) {
        interopTextureRelease();
        
        if (isFp && dtype != ScalarType::Half)
            printf("WARNING: internally casting to fp16 (probably slow), consider passing fp16 directly\n");

        id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
        interopTextureInit(device, context, inFmt, inSize);
    }

    //CGLLockContext(context.CGLContextObj); // because rendering in separate display link thread
    
    // NB: MTLCommandBuffer != MPSCommandBuffer
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSStream.h#L69
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    // Submits a block object for execution and returns after that block finishes executing.
    dispatch_sync(serialQueue, ^{
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        TORCH_CHECK(blitEncoder, "Failed to create blit command encoder");

        MTLRegion region = {
            {0, 0, 0}, // Origin
            {NSUInteger(W), NSUInteger(H), 1}  // Size
        };
        int bytesPerRow = W * ((isFp) ? 8 : 4);
        id<MTLBuffer> buf = getMTLBufferStorage(inData);
        [blitEncoder copyFromBuffer:buf
                       sourceOffset:0
                  sourceBytesPerRow:bytesPerRow
                sourceBytesPerImage:bytesPerRow * H
                         sourceSize:region.size
                          toTexture:state.metalTexture // directly into interop texture
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:region.origin];
        [blitEncoder endEncoding];
        
        // 1. github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/api/src/mps.cpp#L27     # at::detail::getMPSHooks().commitStream();
        // 2. github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSHooks.mm#L81  # at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT);
        // 3. github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/mps/MPSStream.mm#L73 # endKernelCoalescing() => [commandBuffer() commitAndContinue] or [_commandBuffer commit]
        torch::mps::commit();
        
        //[commandBuffer commit];
        
        // In general, you need to execute glFlush() for Metal to get the results of the OpenGL rendering and you need to execute
        // [MTLCommandBuffer commit] and [MTLCommandBuffer waitUntilScheduled] for OpenGL to get the result of Metal rendering.
        // (https://developer.apple.com/forums/thread/694201)
        
        //[commandBuffer waitUntilScheduled]; // hangs if using torch::mps::commit()
        //[commandBuffer waitUntilCompleted];
    });
    
    //CGLUnlockContext(context.CGLContextObj);
    
    TORCH_CHECK(glGetError() == GL_NO_ERROR, "GL error flag was set");
    return uint64_t(state.openGLTexture);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gl_tex_rect", &toGlTex, "Get OpenGL TEXTURE_RECTANGLE with contents of MPS tensor", py::arg("imgHwc"));
}
