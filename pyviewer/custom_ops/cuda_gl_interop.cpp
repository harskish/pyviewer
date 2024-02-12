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

void upload(uint64_t data_ptr, int width, int height, int pitch, uint64_t ptr) {
    auto resource = (cudaGraphicsResource_t)ptr;
    cudaErrors(cudaGraphicsMapResources(1, &resource));
    cudaArray_t array;
    cudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    cudaErrors(cudaMemcpyToArray(array, 0, 0, (const void*)data_ptr, width * height * pitch, cudaMemcpyDeviceToDevice));
    cudaErrors(cudaGraphicsUnmapResources(1, &resource));
}

PYBIND11_MODULE(cuda_gl_interop, m) {
    m.def("register", &register_resource, "register resource");
    m.def("unregister", &unregister_resource, "unregister resource");
    m.def("upload", &upload, "upload image data");
}
