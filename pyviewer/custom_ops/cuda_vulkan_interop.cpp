// CUDA and Vulkan share a dedicated device-local buffer and binary semaphores.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>

static void checked(cudaError_t result) {
    TORCH_CHECK(result == cudaSuccess, "CUDA/Vulkan interop: ", cudaGetErrorString(result));
}

class CUDAVulkanBuffer {
    cudaExternalMemory_t memory_ = nullptr;
    cudaExternalSemaphore_t cuda_ready_ = nullptr, vulkan_ready_ = nullptr;
    void* pointer_ = nullptr;
    uint64_t bytes_;
    int device_;

    static cudaExternalSemaphore_t semaphore(int original) {
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        desc.handle.fd = dup(original);
        TORCH_CHECK(desc.handle.fd >= 0, "Could not duplicate external semaphore FD");
        cudaExternalSemaphore_t result = nullptr;
        auto error = cudaImportExternalSemaphore(&result, &desc);
        if (error != cudaSuccess) close(desc.handle.fd);
        checked(error);
        return result;
    }

    void cleanup() noexcept {
        // upload() completes the CUDA stream before returning; SDL waits its
        // render fence before releasing either API's references.
        if (pointer_) cudaFree(pointer_);
        if (memory_) cudaDestroyExternalMemory(memory_);
        if (cuda_ready_) cudaDestroyExternalSemaphore(cuda_ready_);
        if (vulkan_ready_) cudaDestroyExternalSemaphore(vulkan_ready_);
        pointer_ = nullptr;
        memory_ = nullptr;
        cuda_ready_ = vulkan_ready_ = nullptr;
    }

public:
    CUDAVulkanBuffer(int memory_fd, int cuda_fd, int vulkan_fd,
                     uint64_t allocation_size, uint64_t bytes, int device,
                     const std::string& vulkan_uuid) : bytes_(bytes), device_(device) {
        c10::cuda::CUDAGuard guard(device_);
        cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties, device_));
        std::ostringstream encoded;
        for (unsigned char value : properties.uuid.bytes)
            encoded << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
        TORCH_CHECK(encoded.str() == vulkan_uuid, "CUDA and SDL Vulkan selected different GPUs");
        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dup(memory_fd);
        TORCH_CHECK(desc.handle.fd >= 0, "Could not duplicate external memory FD");
        desc.size = allocation_size;
        desc.flags = cudaExternalMemoryDedicated;
        auto error = cudaImportExternalMemory(&memory_, &desc);
        if (error != cudaSuccess) close(desc.handle.fd);
        checked(error);
        try {
            cudaExternalMemoryBufferDesc mapping{};
            mapping.size = bytes_;
            checked(cudaExternalMemoryGetMappedBuffer(&pointer_, memory_, &mapping));
            cuda_ready_ = semaphore(cuda_fd);
            vulkan_ready_ = semaphore(vulkan_fd);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~CUDAVulkanBuffer() {
        c10::cuda::CUDAGuard guard(device_);
        cleanup();
    }

    void upload(const torch::Tensor& input) {
        TORCH_CHECK(input.is_cuda() && input.get_device() == device_, "CUDA device mismatch");
        TORCH_CHECK(input.scalar_type() == at::kHalf && input.is_contiguous(),
                    "Expected contiguous float16 RGBA input");
        TORCH_CHECK(input.dim() == 3 && input.size(2) == 4 && input.nbytes() == bytes_,
                    "CUDA/Vulkan upload shape mismatch");
        c10::cuda::CUDAGuard guard(device_);
        auto stream = c10::cuda::getCurrentCUDAStream(device_).stream();
        cudaExternalSemaphoreWaitParams wait{};
        checked(cudaWaitExternalSemaphoresAsync(&vulkan_ready_, &wait, 1, stream));
        checked(cudaMemcpyAsync(pointer_, input.data_ptr(), bytes_, cudaMemcpyDeviceToDevice, stream));
        cudaExternalSemaphoreSignalParams signal{};
        checked(cudaSignalExternalSemaphoresAsync(&cuda_ready_, &signal, 1, stream));
        // Complete this copy before letting the source tensor's storage go.
        checked(cudaStreamSynchronize(stream));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<CUDAVulkanBuffer>(module, "CUDAVulkanBuffer")
        .def(pybind11::init<int, int, int, uint64_t, uint64_t, int, const std::string&>())
        .def("upload", &CUDAVulkanBuffer::upload, pybind11::call_guard<pybind11::gil_scoped_release>());
}
