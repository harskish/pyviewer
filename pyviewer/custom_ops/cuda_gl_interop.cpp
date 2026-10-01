// Original code graciously provided by Pauli Kemppinen (github.com/msqrt)

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <cuda.h>
#include <cuda_runtime.h>
#ifdef WIN32
#include <windows.h>
#include <gl/GL.h>
#else
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>
#include <iostream>
using std::cout;

#define cudaErrors(...) do {\
    cudaError_t error = __VA_ARGS__;\
    if(error) {\
        cout << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << "\n";\
        cout << "while running " #__VA_ARGS__ "\n";\
    }\
} while(false)

uint64_t register_resource(const GLuint tex) {
    cudaGraphicsResource_t resource = nullptr;
    cudaErrors(cudaGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    return (uint64_t)resource;
}

void unregister_resource(uint64_t ptr) {
    auto resource = (cudaGraphicsResource_t)ptr;
    cudaErrors(cudaGraphicsUnregisterResource(resource));
}

void map_resource(uint64_t ptr, uint64_t stream_ptr) {
    auto resource = (cudaGraphicsResource_t)ptr;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaErrors(cudaGraphicsMapResources(1, &resource, stream));
}

void copy_to_resource(uint64_t data_ptr, int width, int height, int pitch, uint64_t ptr, uint64_t stream_ptr) {
    auto resource = (cudaGraphicsResource_t)ptr;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaArray_t array;
    cudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    if (stream_ptr) {
        cudaErrors(cudaMemcpy2DToArrayAsync(array, 0, 0, (const void*)data_ptr,
                                            width * pitch, width * pitch, height,
                                            cudaMemcpyDeviceToDevice, stream));
    } else {
        cudaErrors(cudaMemcpyToArray(array, 0, 0, (const void*)data_ptr,
                                     width * height * pitch, cudaMemcpyDeviceToDevice));
    }
}

void unmap_resource(uint64_t ptr, uint64_t stream_ptr) {
    auto resource = (cudaGraphicsResource_t)ptr;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaErrors(cudaGraphicsUnmapResources(1, &resource, stream));
}

void upload(uint64_t data_ptr, int width, int height, int pitch, uint64_t ptr, uint64_t stream_ptr) {
    map_resource(ptr, stream_ptr);
    copy_to_resource(data_ptr, width, height, pitch, ptr, stream_ptr);
    unmap_resource(ptr, stream_ptr);
}

PYBIND11_MODULE(cuda_gl_interop, m) {
    m.def("register", &register_resource, "register resource");
    m.def("unregister", &unregister_resource, "unregister resource");
    m.def("upload", &upload, "upload image data");
    m.def("map_resource", &map_resource, "map image for CUDA access");
    m.def("copy_to_resource", &copy_to_resource, "copy image on CUDA stream");
    m.def("unmap_resource", &unmap_resource, "release image to OpenGL");
}
