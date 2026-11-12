"""Native MPS/Metal and CUDA/Vulkan image copies, loaded only for GPU tensors.

Requires the SDL bridge in custom_ops/sdl3-interop.patch. SDL properties expose
native resources through that versioned bridge, never through private offsets.
"""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_MAIN_HANDLED", "1")
import sdl3 as sdl


@lru_cache(maxsize=2)
def extension(driver):
    from torch.utils.cpp_extension import load

    root = Path(__file__).with_name('custom_ops')
    metal = driver == 'metal'
    return load(
        name='pyviewer_mps_sdl' if metal else 'pyviewer_cuda_vulkan',
        sources=[str(root / ('metal_sdl_interop.mm' if metal else 'cuda_vulkan_interop.cpp'))],
        extra_cflags=['-O3', '-std=c++17'] + (['-fobjc-arc'] if metal else []),
        extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation'] if metal else [],
        with_cuda=not metal,
        verbose=False,
    )


def bridge_version(device):
    props = sdl.SDL_GetGPUDeviceProperties(device)
    return sdl.SDL_GetNumberProperty(props, b'pyviewer.interop.version', 0)


def supports_tensor(renderer, tensor):
    torch = sys.modules.get('torch')
    if torch is None or not torch.is_tensor(tensor) or tensor.device.type == 'cpu':
        return False
    driver = sdl.SDL_GetGPUDeviceDriver(renderer.device)
    return ((driver == b'metal' and tensor.device.type == 'mps') or
            (driver == b'vulkan' and sys.platform == 'linux' and tensor.device.type == 'cuda'))


class CUDATransfer:
    def __init__(self, renderer, width, height, device_index):
        self.renderer = renderer
        self.transfer = None
        self.mapping = None
        self.device_index = device_index
        module = extension('vulkan')
        props = sdl.SDL_GetGPUDeviceProperties(renderer.device)
        callback = sdl.SDL_GetPointerProperty(props, b'pyviewer.vulkan.create_transfer', None)
        context = sdl.SDL_GetPointerProperty(props, b'pyviewer.vulkan.context', None)
        if not callback or not context:
            raise RuntimeError('SDL has no Vulkan external-memory bridge')
        create = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)(callback)
        output = sdl.SDL_CreateProperties()
        fds = []
        try:
            nbytes = width * height * 8
            if nbytes > 0xffffffff:
                raise ValueError('SDL transfer buffers must be smaller than 4 GiB')
            pointer = create(context, nbytes, output)
            if not pointer:
                raise RuntimeError(sdl.SDL_GetError().decode())
            self.transfer = ctypes.cast(pointer, ctypes.POINTER(sdl.SDL_GPUTransferBuffer))
            for key in (b'pyviewer.memory_fd', b'pyviewer.cuda_ready_fd', b'pyviewer.vulkan_ready_fd'):
                fds.append(sdl.SDL_GetNumberProperty(output, key, -1))
            size = sdl.SDL_GetNumberProperty(output, b'pyviewer.allocation_size', 0)
            uuid = sdl.SDL_GetStringProperty(output, b'pyviewer.device_uuid', b'').decode()
            # The extension imports dup()s; this scope always closes the exports.
            self.mapping = module.CUDAVulkanBuffer(*fds, size, nbytes, device_index, uuid)
        except BaseException:
            self.release()
            raise
        finally:
            for fd in fds:
                if fd >= 0:
                    os.close(fd)
            sdl.SDL_DestroyProperties(output)

    def upload(self, tensor):
        if tensor.device.index != self.device_index:
            raise RuntimeError('A Vulkan interop texture cannot switch CUDA devices')
        self.mapping.upload(tensor)

    def release(self):
        self.mapping = None
        if self.transfer:
            sdl.SDL_ReleaseGPUTransferBuffer(self.renderer.device, self.transfer)
            self.transfer = None


def upload(texture, tensor, ready=None):
    import torch

    if ready is not None:
        with torch.cuda.device(tensor.device):
            ready.wait()
    data = tensor.detach()
    if not data.dtype.is_floating_point:
        data = data.to(torch.float32) / torch.iinfo(data.dtype).max
    data = data.to(torch.float16).contiguous()
    if data.device.type == 'mps':
        extension('metal').upload_mps(data, texture.native_metal_texture)
        return 'metal-mps'
    if texture.cuda_transfer is None:
        texture.cuda_transfer = CUDATransfer(texture.renderer, texture.width, texture.height, data.device.index)
    texture.cuda_transfer.upload(data)
    return 'vulkan-cuda'
