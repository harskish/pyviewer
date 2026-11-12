"""Native GPU image interop, loaded only for GPU tensors.

SDL_Renderer uses public Metal hooks and MPS texture views on stock SDL.
The SDL_Renderer GPU driver uses the bridge in custom_ops/sdl3-interop.patch. Native
resources use public hooks or that versioned bridge, never private offsets.
"""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_MAIN_HANDLED", "1")
import sdl3 as sdl


@lru_cache(maxsize=3)
def extension(driver):
    from torch.utils.cpp_extension import load

    root = Path(__file__).with_name('custom_ops')
    metal = driver in ('metal', 'metal-renderer')
    name = ('pyviewer_mps_sdl_renderer' if driver == 'metal-renderer' else
            'pyviewer_mps_sdl' if metal else 'pyviewer_cuda_vulkan')
    source = ('metal_sdl_renderer.mm' if driver == 'metal-renderer' else
              'metal_sdl_interop.mm' if metal else 'cuda_vulkan_interop.cpp')
    return load(
        name=name,
        sources=[str(root / source)],
        extra_cflags=['-O3', '-std=c++17'] + (['-fobjc-arc'] if metal else []),
        extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation', '-framework', 'QuartzCore'] if metal else [],
        with_cuda=not metal,
        verbose=False,
    )


def bridge_version(device):
    props = sdl.SDL_GetGPUDeviceProperties(device)
    return sdl.SDL_GetNumberProperty(props, b'pyviewer.interop.version', 0)


def checked(value):
    if not value:
        raise RuntimeError(sdl.SDL_GetError().decode())
    return value


def create_device(driver, debug, interop):
    """Create the device borrowed by SDL_Renderer's GPU driver."""
    formats = sdl.SDL_GPU_SHADERFORMAT_SPIRV | sdl.SDL_GPU_SHADERFORMAT_MSL
    if driver == 'vulkan' and sys.platform == 'linux' and interop != 'off':
        props = checked(sdl.SDL_CreateProperties())
        extensions = (ctypes.c_char_p * 2)(b'VK_KHR_external_memory_fd', b'VK_KHR_external_semaphore_fd')
        options = sdl.SDL_GPUVulkanOptions(vulkan_api_version=(1 << 22) | (1 << 12),
            device_extension_count=2, device_extension_names=extensions)
        try:
            checked(sdl.SDL_SetStringProperty(props, sdl.SDL_PROP_GPU_DEVICE_CREATE_NAME_STRING, driver.encode()))
            checked(sdl.SDL_SetBooleanProperty(props, sdl.SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug))
            checked(sdl.SDL_SetBooleanProperty(props, sdl.SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, True))
            checked(sdl.SDL_SetPointerProperty(props, sdl.SDL_PROP_GPU_DEVICE_CREATE_VULKAN_OPTIONS_POINTER, ctypes.addressof(options)))
            device = sdl.SDL_CreateGPUDeviceWithProperties(props)
            if device:
                return device
            if interop == 'required':
                return checked(device)
        finally:
            sdl.SDL_DestroyProperties(props)
    return checked(sdl.SDL_CreateGPUDevice(formats, debug, driver.encode()))


class InteropTexture:
    """FP16 upload resource borrowed by an SDL_Texture; performs no rendering."""
    def __init__(self, renderer, width, height):
        self.renderer = renderer
        self.width, self.height = width, height
        self.cuda_transfer = None
        self.texture = None
        self.native_metal_texture = None
        info = sdl.SDL_GPUTextureCreateInfo(type=sdl.SDL_GPU_TEXTURETYPE_2D,
            format=sdl.SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, usage=sdl.SDL_GPU_TEXTUREUSAGE_SAMPLER,
            width=width, height=height, layer_count_or_depth=1, num_levels=1,
            sample_count=sdl.SDL_GPU_SAMPLECOUNT_1)
        props = checked(sdl.SDL_CreateProperties())
        try:
            checked(sdl.SDL_SetBooleanProperty(props, b'pyviewer.interop', True))
            info.props = props
            self.texture = checked(sdl.SDL_CreateGPUTexture(renderer.device, ctypes.byref(info)))
            self.native_metal_texture = sdl.SDL_GetPointerProperty(props, b'pyviewer.metal.texture', None)
        finally:
            sdl.SDL_DestroyProperties(props)

    def upload(self, tensor, ready=None):
        self.renderer.wait_native()
        backend = upload(self, tensor, ready)
        if self.cuda_transfer is not None:
            command = checked(sdl.SDL_AcquireGPUCommandBuffer(self.renderer.device))
            copy = sdl.SDL_BeginGPUCopyPass(command)
            if not copy:
                sdl.SDL_CancelGPUCommandBuffer(command)
                checked(copy)
            source = sdl.SDL_GPUTextureTransferInfo(transfer_buffer=self.cuda_transfer.transfer,
                pixels_per_row=self.width, rows_per_layer=self.height)
            dest = sdl.SDL_GPUTextureRegion(texture=self.texture, w=self.width, h=self.height, d=1)
            sdl.SDL_UploadToGPUTexture(copy, ctypes.byref(source), ctypes.byref(dest), False)
            sdl.SDL_EndGPUCopyPass(copy)
            checked(sdl.SDL_SubmitGPUCommandBuffer(command))
        return backend

    def release(self):
        if self.texture:
            self.renderer.wait_native()
            if self.cuda_transfer is not None:
                self.cuda_transfer.release()
                self.cuda_transfer = None
            sdl.SDL_ReleaseGPUTexture(self.renderer.device, self.texture)
            self.texture = None
            self.native_metal_texture = None


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
