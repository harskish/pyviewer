"""ImGui geometry via stock SDL_Renderer, with native MPS and CUDA image uploads."""
import ctypes
import time
import warnings

import numpy as np
from .sdl3_backend import SDL3Platform
from . import sdl_torch_interop
from .sdl_torch_interop import extension
import sdl3 as sdl
from imgui_bundle import imgui


def checked(value):
    if not value:
        raise RuntimeError(sdl.SDL_GetError().decode())
    return value


def create_renderer(window, driver, hdr, swap_interval, *, gpu_device=None):
    if hdr not in ('auto', 'linear', 'sdr', True, False):
        raise ValueError('SDL_Renderer supports extended-linear HDR or SDR output')
    candidates = (['linear', 'sdr'] if hdr == 'auto' else ['linear'] if hdr is True else
                  ['sdr'] if hdr is False else [hdr])
    errors = []
    for mode in candidates:
        props = checked(sdl.SDL_CreateProperties())
        try:
            checked(sdl.SDL_SetPointerProperty(props, sdl.SDL_PROP_RENDERER_CREATE_WINDOW_POINTER,
                                              ctypes.cast(window, ctypes.c_void_p)))
            checked(sdl.SDL_SetStringProperty(props, sdl.SDL_PROP_RENDERER_CREATE_NAME_STRING, driver.encode()))
            if gpu_device:
                checked(sdl.SDL_SetPointerProperty(props, sdl.SDL_PROP_RENDERER_CREATE_GPU_DEVICE_POINTER,
                                                  ctypes.cast(gpu_device, ctypes.c_void_p)))
            colorspace = {'linear': sdl.SDL_COLORSPACE_SRGB_LINEAR,
                          'sdr': sdl.SDL_COLORSPACE_SRGB}[mode]
            checked(sdl.SDL_SetNumberProperty(props, sdl.SDL_PROP_RENDERER_CREATE_OUTPUT_COLORSPACE_NUMBER, colorspace))
            checked(sdl.SDL_SetNumberProperty(props, sdl.SDL_PROP_RENDERER_CREATE_PRESENT_VSYNC_NUMBER, swap_interval))
            renderer = sdl.SDL_CreateRendererWithProperties(props)
            if renderer:
                # SDL 3.4's GPU renderer ignores a failed swapchain-mode change.
                # Verify support rather than advertising HDR on an SDR swapchain.
                composition = {'linear': sdl.SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR,
                               'sdr': sdl.SDL_GPU_SWAPCHAINCOMPOSITION_SDR}[mode]
                if not gpu_device or sdl.SDL_WindowSupportsGPUSwapchainComposition(gpu_device, window, composition):
                    return renderer, mode
                sdl.SDL_DestroyRenderer(renderer)
                errors.append(f'{mode}: swapchain composition unavailable')
                continue
            errors.append(f'{mode}: {sdl.SDL_GetError().decode()}')
        finally:
            sdl.SDL_DestroyProperties(props)
    raise RuntimeError('SDL_Renderer creation failed: ' + '; '.join(errors))


class RendererTexture:
    def __init__(self, renderer, width, height, *, float_texture=True, srgb=False, pq=False, **_kwargs):
        self.renderer = renderer
        self.width, self.height = width, height
        self.dtype = np.float16 if float_texture else np.uint8
        self.srgb, self.pq = srgb, pq
        self.nearest = False
        self.texture = None
        self._view = None
        self._gpu_texture = None
        self.pending_tensor = self.pending_ready = None
        self.uploaded = False
        self._scale_mode = None
        self.id = renderer._next_texture_id
        renderer._next_texture_id += 1
        renderer.textures[self.id] = self
        self.ref = imgui.ImTextureRef(self.id)

    def _create(self, gpu_texture=None):
        props = checked(sdl.SDL_CreateProperties())
        try:
            fmt = (sdl.SDL_PIXELFORMAT_RGBA32 if self.dtype == np.uint8 else
                   sdl.SDL_PIXELFORMAT_RGBA64_FLOAT)
            colorspace = (sdl.SDL_COLORSPACE_HDR10 if self.pq else
                          sdl.SDL_COLORSPACE_SRGB if self.srgb else sdl.SDL_COLORSPACE_SRGB_LINEAR)
            for key, value in ((sdl.SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER, fmt),
                               (sdl.SDL_PROP_TEXTURE_CREATE_COLORSPACE_NUMBER, colorspace),
                               (sdl.SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER, self.width),
                               (sdl.SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER, self.height)):
                checked(sdl.SDL_SetNumberProperty(props, key, value))
            if gpu_texture is not None:
                checked(sdl.SDL_SetPointerProperty(props, sdl.SDL_PROP_TEXTURE_CREATE_GPU_TEXTURE_POINTER,
                                                  ctypes.cast(gpu_texture.texture, ctypes.c_void_p)))
            # Presentation must not apply another tone map to the ISP output.
            checked(sdl.SDL_SetFloatProperty(props, sdl.SDL_PROP_TEXTURE_CREATE_HDR_HEADROOM_FLOAT, 1.0))
            checked(sdl.SDL_SetFloatProperty(props, sdl.SDL_PROP_TEXTURE_CREATE_SDR_WHITE_POINT_FLOAT,
                                            self.renderer.sdr_white_nits if self.pq else 1.0))
            texture = checked(sdl.SDL_CreateTextureWithProperties(self.renderer.handle, props))
            checked(sdl.SDL_SetTextureBlendMode(texture, sdl.SDL_BLENDMODE_BLEND))
            return texture
        finally:
            sdl.SDL_DestroyProperties(props)

    def _replace(self, texture, view=None, gpu_texture=None):
        if self.texture:
            self.renderer.wait_native()
            sdl.SDL_DestroyTexture(self.texture)
        if self._gpu_texture is not None and self._gpu_texture is not gpu_texture:
            self._gpu_texture.release()
        self.texture, self._view, self._gpu_texture = texture, view, gpu_texture
        self._scale_mode = None

    def upload(self, rgba):
        data = np.ascontiguousarray(rgba, dtype=self.dtype)
        assert data.shape == (self.height, self.width, 4)
        if self.texture is None or self._view is not None or self._gpu_texture is not None:
            self._replace(self._create())
        checked(sdl.SDL_UpdateTexture(self.texture, None, data.ctypes.data, data.strides[0]))
        self.pending_tensor = self.pending_ready = None
        self.uploaded = True
        if self.id not in self.renderer._font_textures:
            self.renderer.last_upload_backend = 'cpu'
            self.renderer.last_upload_copied = True

    def upload_torch(self, rgba, ready=None):
        if self.renderer.interop_policy == 'off':
            return False
        supported = (sdl_torch_interop.supports_tensor(self.renderer, rgba)
                     if self.renderer.device else rgba.device.type == 'mps' and bool(self.renderer.metal_layer))
        if not supported:
            if self.renderer.interop_policy == 'required' and rgba.device.type != 'cpu':
                raise RuntimeError(f'No native SDL_Renderer interop for {rgba.device} and this driver')
            return False
        if not self.renderer.native_interop:
            self.renderer.interop_failure(RuntimeError('Native GPU interop unavailable; the GPU driver needs the SDL interop bridge'))
            return False
        assert rgba.shape == (self.height, self.width, 4)
        self.pending_tensor, self.pending_ready = rgba, ready
        self.uploaded = True
        return True

    def bind(self):
        mode = sdl.SDL_SCALEMODE_NEAREST if self.nearest else sdl.SDL_SCALEMODE_LINEAR
        if mode != self._scale_mode:
            checked(sdl.SDL_SetTextureScaleMode(self.texture, mode))
            self._scale_mode = mode
        return self.texture

    def release(self):
        self._replace(None)
        self.pending_tensor = self.pending_ready = None
        self.renderer.textures.pop(self.id, None)


class SDLRendererImguiRenderer(SDL3Platform):
    def __init__(self, window, handle, *, mode, sdr_white_nits=203.0, gpu_interop='auto', gpu_device=None, **kwargs):
        self.handle = handle
        self.device = gpu_device
        self.mode, self.sdr_white_nits = mode, sdr_white_nits
        self.sc_rgb_white = 1.0
        self.metal_layer = None if gpu_device else sdl.SDL_GetRenderMetalLayer(handle)
        self.metal_device = None
        self._context = None
        self.interop_policy = gpu_interop
        self.native_interop = gpu_interop != 'off' and (
            sdl_torch_interop.bridge_version(gpu_device) == 1 if gpu_device else bool(self.metal_layer))
        self.last_upload_backend = 'cpu'
        self.last_upload_copied = False
        self.last_native_upload_ms = 0.0
        self.last_presented = False
        self._module = self._last_fence = None
        self._queued = False
        self._interop_warning = False
        self.textures, self._font_textures = {}, {}
        self._next_texture_id = 1
        super().__init__(window, **kwargs)
        self.io.backend_flags |= (imgui.BackendFlags_.renderer_has_textures |
                                  imgui.BackendFlags_.renderer_has_vtx_offset)

    def wait_native(self):
        if self._queued:
            checked(sdl.SDL_RenderPresent(self.handle))
            self._queued = False
        if self.device:
            checked(sdl.SDL_WaitForGPUIdle(self.device))
        if self._last_fence is not None:
            self._last_fence.wait()
            self._last_fence = None

    def interop_failure(self, error):
        if self.interop_policy == 'required':
            raise RuntimeError(f'Native SDL_Renderer interop is required: {error}') from error
        if not self._interop_warning:
            warnings.warn(f'Native GPU upload unavailable; using CPU upload: {error}', RuntimeWarning, stacklevel=2)
            self._interop_warning = True
        self.native_interop = False

    def _upload_gpu_tensor(self, texture, tensor, ready):
        if texture._gpu_texture is None:
            native = sdl_torch_interop.InteropTexture(self, texture.width, texture.height)
            try:
                wrapped = texture._create(native)
            except BaseException:
                native.release()
                raise
            texture._replace(wrapped, gpu_texture=native)
        self.last_upload_backend = texture._gpu_texture.upload(tensor, ready)
        self.last_upload_copied = True

    def _prepare_native_uploads(self):
        pending = [t for t in self.textures.values() if t.pending_tensor is not None]
        for texture in pending:
            start = time.perf_counter()
            tensor = texture.pending_tensor
            try:
                import torch
                if self.device:
                    self._upload_gpu_tensor(texture, tensor, texture.pending_ready)
                else:
                    if self._module is None:
                        self._module = extension('metal-renderer')
                        self._context = self._module.RenderContext(self.metal_layer)
                        self.metal_device = self._context.device
                    data = tensor.detach()
                    if not data.dtype.is_floating_point:
                        data = data.float() / torch.iinfo(data.dtype).max
                    elif data.dtype not in (torch.float16, torch.float32):
                        data = data.float()
                    view = self._module.TextureView(data, self.metal_device)
                    texture._replace(None, view)
                    self.last_upload_copied = view.copied or data.dtype != tensor.dtype
                    self.last_upload_backend = 'metal-mps-view'
            except Exception as error:
                self.interop_failure(error)
                if texture.pending_ready is not None:
                    with torch.cuda.device(tensor.device):
                        texture.pending_ready.synchronize()
                image = tensor.detach()
                if not image.dtype.is_floating_point:
                    image = image.float() / torch.iinfo(image.dtype).max
                texture.upload(image.to(device='cpu', dtype=torch.float32).numpy())
            texture.pending_tensor = texture.pending_ready = None
            self.last_native_upload_ms = (time.perf_counter() - start) * 1000

    def _update_imgui_textures(self):
        for tex in imgui.get_platform_io().textures:
            if tex.status == imgui.ImTextureStatus.want_create:
                texture = RendererTexture(self, tex.width, tex.height, float_texture=False, srgb=True)
                self._font_textures[texture.id] = texture
                tex.set_tex_id(texture.id)
                texture.upload(tex.get_pixels_array().reshape(tex.height, tex.width, 4))
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_updates:
                self.textures[tex.tex_id].upload(tex.get_pixels_array().reshape(tex.height, tex.width, 4))
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_destroy:
                self._font_textures.pop(tex.tex_id).release()
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)

    def render(self, draw_data, *, present=True):
        self.last_presented = False
        self._update_imgui_textures()
        self._prepare_native_uploads()
        fx, fy = draw_data.framebuffer_scale
        px, py = draw_data.display_pos
        checked(sdl.SDL_SetRenderScale(self.handle, 1, 1))
        checked(sdl.SDL_SetRenderClipRect(self.handle, None))
        checked(sdl.SDL_SetRenderDrawColorFloat(self.handle, .08, .08, .08, 1))
        checked(sdl.SDL_RenderClear(self.handle))
        self._queued = True
        vertex_dtype = np.dtype({'names': ['pos', 'uv', 'color'],
            'formats': [('f4', 2), ('f4', 2), ('u1', 4)],
            'offsets': [imgui.VERTEX_BUFFER_POS_OFFSET, imgui.VERTEX_BUFFER_UV_OFFSET, imgui.VERTEX_BUFFER_COL_OFFSET],
            'itemsize': imgui.VERTEX_SIZE})
        for commands in draw_data.cmd_lists:
            count = commands.vtx_buffer.size()
            if not count:
                continue
            storage = (ctypes.c_ubyte * (count * imgui.VERTEX_SIZE)).from_address(commands.vtx_buffer.data_address())
            vertices = np.frombuffer(storage, dtype=vertex_dtype)
            positions = np.ascontiguousarray((vertices['pos'] - (px, py)) * (fx, fy), dtype=np.float32)
            colors = vertices['color'].astype(np.float32) / 255
            for cmd in commands.cmd_buffer:
                if not cmd.elem_count:
                    continue
                x1, y1, x2, y2 = cmd.clip_rect
                clip = sdl.SDL_Rect(max(0, int((x1-px)*fx)), max(0, int((y1-py)*fy)), 0, 0)
                clip.w, clip.h = int((x2-px)*fx)-clip.x, int((y2-py)*fy)-clip.y
                if clip.w <= 0 or clip.h <= 0:
                    continue
                checked(sdl.SDL_SetRenderClipRect(self.handle, ctypes.byref(clip)))
                texture = self.textures[cmd.tex_ref.get_tex_id()]
                offset = cmd.vtx_offset
                if texture._view is not None:
                    encoder = sdl.SDL_GetRenderMetalCommandEncoder(self.handle)
                    if not encoder:
                        continue  # Hidden/minimized windows may have no drawable.
                    index_type = ctypes.c_uint16 if imgui.INDEX_SIZE == 2 else ctypes.c_uint32
                    indices = np.ctypeslib.as_array((index_type * cmd.elem_count).from_address(
                        commands.idx_buffer.data_address() + cmd.idx_offset*imgui.INDEX_SIZE)).astype(np.int64) + offset
                    triangles = np.ascontiguousarray(np.concatenate((positions[indices], vertices['uv'][indices], colors[indices]),axis=1))
                    width, height = int(draw_data.display_size.x*fx), int(draw_data.display_size.y*fy)
                    clip.w, clip.h = min(clip.w,width-clip.x), min(clip.h,height-clip.y)
                    if clip.w > 0 and clip.h > 0:
                        self._last_fence = self._context.draw(encoder, texture._view, triangles, width, height,
                            clip.x,clip.y,clip.w,clip.h,texture.nearest,self.mode=='linear',
                            2 if texture.pq else int(texture.srgb),self.sdr_white_nits,self.sc_rgb_white)
                    continue
                checked(sdl.SDL_RenderGeometryRaw(self.handle, texture.bind(),
                    ctypes.cast(positions.ctypes.data + offset*8, ctypes.POINTER(ctypes.c_float)), 8,
                    ctypes.cast(colors.ctypes.data + offset*16, ctypes.POINTER(sdl.SDL_FColor)), 16,
                    ctypes.cast(vertices['uv'].ctypes.data + offset*imgui.VERTEX_SIZE,
                                ctypes.POINTER(ctypes.c_float)), imgui.VERTEX_SIZE,
                    count-offset, commands.idx_buffer.data_address()+cmd.idx_offset*imgui.INDEX_SIZE,
                    cmd.elem_count, imgui.INDEX_SIZE))
        checked(sdl.SDL_SetRenderClipRect(self.handle, None))
        if present:
            self.present()

    def present(self):
        checked(sdl.SDL_RenderPresent(self.handle))
        self._queued = False
        self.last_presented = True

    def read_pixels(self):
        """Read the current frame as linear float RGBA before present()."""
        surface = checked(sdl.SDL_RenderReadPixels(self.handle, None))
        try:
            data = surface.contents
            if data.format == sdl.SDL_PIXELFORMAT_RGBA64_FLOAT:
                dtype = np.float16
            elif data.format == sdl.SDL_PIXELFORMAT_RGBA128_FLOAT:
                dtype = np.float32
            else:
                converted = checked(sdl.SDL_ConvertSurface(surface, sdl.SDL_PIXELFORMAT_RGBA128_FLOAT))
                sdl.SDL_DestroySurface(surface)
                surface, data, dtype = converted, converted.contents, np.float32
            rows = np.frombuffer(ctypes.string_at(data.pixels, data.pitch*data.h), dtype=dtype)
            return rows.reshape(data.h, data.pitch // np.dtype(dtype).itemsize)[:, :data.w*4].reshape(data.h, data.w, 4).astype(np.float32)
        finally:
            sdl.SDL_DestroySurface(surface)

    def shutdown(self):
        self.wait_native()
        for tex in imgui.get_platform_io().textures:
            if tex.tex_id in self._font_textures:
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)
        for texture in list(self.textures.values()):
            texture.release()
        self._font_textures.clear()
        self.io.backend_flags &= ~(imgui.BackendFlags_.renderer_has_textures | imgui.BackendFlags_.renderer_has_vtx_offset)
        super().shutdown()
