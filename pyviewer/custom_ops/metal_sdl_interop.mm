// GPU-only MPS buffer -> SDL-owned Metal texture copy.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <Metal/Metal.h>
#include <cstdint>

void upload_mps(const torch::Tensor& input, uint64_t texture_address) {
    TORCH_CHECK(input.device().is_mps(), "Expected an MPS tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf && input.is_contiguous(),
                "Expected contiguous float16 RGBA input");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 4, "Expected HWC RGBA input");
    TORCH_CHECK(texture_address, "SDL has not exposed a Metal texture");
    @autoreleasepool {
        id<MTLTexture> texture = (__bridge id<MTLTexture>)(void*)texture_address;
        auto* stream = at::mps::getCurrentMPSStream();
        TORCH_CHECK(texture.device.registryID == stream->device().registryID,
                    "PyTorch and SDL must use the same Metal device");
        TORCH_CHECK(texture.pixelFormat == MTLPixelFormatRGBA16Float,
                    "Expected an RGBA16Float Metal texture");
        const NSUInteger height = input.size(0), width = input.size(1);
        TORCH_CHECK(texture.width == width && texture.height == height, "Texture size mismatch");

        // Buffer-to-texture blits require aligned row pitch. Pad on the GPU,
        // retaining views' storage offsets when no padding is necessary.
        auto data = input;
        NSUInteger row_bytes = width * 8;
        NSUInteger offset = input.storage_offset() * input.element_size();
        if (row_bytes % 256 || offset % 256) {
            row_bytes = (row_bytes + 255) & ~NSUInteger(255);
            data = torch::empty({(int64_t)height, (int64_t)(row_bytes / 8), 4}, input.options());
            data.slice(1, 0, width).copy_(input);
            offset = 0;
        }
        id<MTLBuffer> source = __builtin_bit_cast(id<MTLBuffer>, data.storage().data());
        dispatch_semaphore_t finished = dispatch_semaphore_create(0);
        __block id<MTLCommandBuffer> completed = nil;
        // Keep the GIL while touching the stream: torch.mps.synchronize()
        // also holds it, but does not dispatch onto stream->queue().
        at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
            stream->endKernelCoalescing();
            id<MTLCommandBuffer> command = stream->commandBuffer();
            id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
            TORCH_CHECK(encoder, "Could not create Metal interop blit encoder");
            [encoder copyFromBuffer:source sourceOffset:offset sourceBytesPerRow:row_bytes
                sourceBytesPerImage:row_bytes * height sourceSize:MTLSizeMake(width, height, 1)
                toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            [encoder endEncoding];
            [command addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                completed = buffer;
                dispatch_semaphore_signal(finished);
            }];
            stream->synchronize(at::mps::SyncType::COMMIT);
        });
        // Wait for this blit without leaving the shared MPS command buffer
        // committed but still installed in the stream. A concurrent Python
        // synchronize could otherwise try to commit that buffer a second time.
        // The caller waits for prior SDL reads before writing the texture.
        {
            pybind11::gil_scoped_release release;
            dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
        }
        TORCH_CHECK(completed.status != MTLCommandBufferStatusError,
                    "Metal interop blit failed: ", completed.error.localizedDescription.UTF8String);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("upload_mps", &upload_mps);
}
