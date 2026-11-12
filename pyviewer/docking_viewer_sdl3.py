"""SDL3 + imgui_bundle with Metal/Vulkan HDR presentation.

Uses Vulkan on Linux, Metal on macOS and D3D12 on Windows through SDL_Renderer.
Images are linear Rec.709 by default; Rec.2020/PQ input is also supported.
GPU tensors use native MPS/Metal or CUDA/Vulkan interop when available.
"""

import ctypes
from pathlib import Path
import sys
import time

import numpy as np

from .docking_viewer_py import PyDockingViewer, dockable  # re-export the shared decorator
from .sdl3_backend import window_sizes
from .sdl_renderer import SDLRendererImguiRenderer, RendererTexture, create_renderer
from .utils import PannableArea

import sdl3 as sdl
from imgui_bundle import imgui


class SDL3PannableArea(PannableArea):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wheel_pos = None
        self._hovered = False
        self.nearest = True
        self._draw_frame = -1

    def get_hdpi_scale(self):
        return tuple(imgui.get_io().display_framebuffer_scale)

    def set_interp_linear(self):
        self.nearest = False

    def set_interp_nearest(self):
        self.nearest = True

    @property
    def mouse_pos_abs(self):
        if self._wheel_pos is not None:
            return np.array(self._wheel_pos)
        return super().mouse_pos_abs

    def draw(self, texture, width, height, canvas_width, canvas_height):
        if min(canvas_width, canvas_height) <= 0:
            return
        self.canvas_w, self.canvas_h = canvas_width, canvas_height
        self.tex_w, self.tex_h = width, height
        self.output_pos_tl[:] = imgui.get_cursor_screen_pos()
        imgui.invisible_button('canvas##' + self.id, (canvas_width, canvas_height),
            imgui.ButtonFlags_.mouse_button_left | imgui.ButtonFlags_.mouse_button_right)
        self._hovered = imgui.is_item_hovered()
        self._draw_frame = imgui.get_frame_count()
        self.handle_pan()
        texture.nearest = self.nearest
        draw_list = imgui.get_window_draw_list()
        lo = tuple(self.output_pos_tl)
        hi = tuple(self.output_pos_tl + (canvas_width, canvas_height))
        draw_list.push_clip_rect(lo, hi, True)
        draw_list.add_rect_filled(lo, hi, imgui.color_convert_float4_to_u32(self.clear_color))
        # Existing numerical pan/zoom transforms; rasterization uses SDL3.
        uv = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        points = uv @ self.uv_to_screen_xform().T
        draw_list.add_image_quad(texture.ref, *(tuple(p[:2]) for p in points))
        draw_list.pop_clip_rect()

    def consume_wheel(self, x, y, pos):
        self._wheel_pos = pos
        try:
            if not (self._draw_frame == imgui.get_frame_count() and self._hovered and self.zoom_enabled and self.mouse_hovers_content() and self.tex_h > 0):
                return False
            self.mouse_wheel_callback(None, x, y)
            return True
        finally:
            self._wheel_pos = None


class SDL3DockingViewer(PyDockingViewer):
    """Same dockable UI API, with SDL3 window/events and HDR presentation.

    keydown accepts an SDL scancode or a name such as 'A' or 'ESCAPE'.
    hdr: 'auto' prefers extended-linear HDR then SDR; True/'linear' requires
    extended-linear HDR, False/'sdr' selects SDR explicitly.
    image_color_space: 'linear' (1.0 = SDR white), 'srgb', or 'rec2020_pq'
    (absolute ST 2084 encoded Rec.2020). Applies to floats and integers alike.
    sdr_white_nits: reference white for PQ input. Linear output uses SDL's
    reported SDR-white scale. Display capability is exposed through hdr_info.
    renderer_driver: SDL_Renderer driver; defaults to Metal on macOS and the
    SDL_GPU-backed renderer on Linux to retain CUDA/Vulkan interop, D3D12 on Windows.
    gpu_interop: 'auto' uses native GPU interop when available (CPU fallback with
    a warning), 'required' rejects unavailable GPU interop, 'off' uses CPU uploads.
    """

    def __init__(self, name, *, hdr='auto', image_color_space='linear', sdr_white_nits=203.0,
                 gpu_driver=None, debug_gpu=False, normalize=False, gpu_interop='auto',
                 renderer_driver=None, **kwargs):
        assert 'context_creation_api' not in kwargs, 'GLFW context_creation_api is not an SDL option'
        assert hdr in ('auto', 'linear', 'sdr', True, False), 'SDL_Renderer HDR output must be extended-linear'
        assert image_color_space in ('linear', 'srgb', 'rec2020_pq')
        assert sdr_white_nits > 0
        assert gpu_interop in ('auto', 'required', 'off')
        self.sdl_renderer = None
        self.gpu_interop = gpu_interop
        self._published_image = None
        self._hdr_request = hdr
        self.image_color_space = image_color_space
        self.sdr_white_nits = sdr_white_nits
        self.gpu_driver = gpu_driver or ('metal' if sys.platform == 'darwin' else 'vulkan')
        self.renderer_driver = renderer_driver or (
            'metal' if sys.platform == 'darwin' else 'direct3d12' if sys.platform == 'win32' else 'gpu')
        assert self.gpu_driver in ('metal', 'vulkan'), 'Native image interop targets Metal and Vulkan'
        self._debug_gpu = debug_gpu
        self._requested_normalize = normalize
        self._content_scale = 1.0
        self._should_close = False
        self.gpu_device = None
        self.hdr_info = {}
        self._drop_paths = None
        try:
            super().__init__(name, hdr=hdr not in (False, 'sdr'), normalize=normalize, **kwargs)
        except BaseException:
            self.stop_event.set()
            # Let the compute thread exit even if setup failed before its start signal.
            self.start_event.set()
            if self.window:
                if self.impl is not None:
                    self._cleanup()
                else:
                    self._destroy_platform()
            raise

    @property
    def _render_scale(self):
        return self.ui_scale * self._content_scale

    def _create_pan_handler(self):
        return SDL3PannableArea(force_mouse_capture=True)

    def _create_window(self):
        if sys.platform == 'darwin':
            # SDL requires this hint before initializing Cocoa application state.
            # Keep macOS's native trackpad inertia, matching the GLFW viewer.
            sdl.SDL_SetHint(sdl.SDL_HINT_MAC_SCROLL_MOMENTUM, b'1')
        assert sdl.SDL_Init(sdl.SDL_INIT_VIDEO | sdl.SDL_INIT_EVENTS), sdl.SDL_GetError()
        try:
            flags = sdl.SDL_WINDOW_RESIZABLE | sdl.SDL_WINDOW_HIGH_PIXEL_DENSITY
            if self._hidden:
                flags |= sdl.SDL_WINDOW_HIDDEN
            self.window = sdl.SDL_CreateWindow(self._window_title.encode(), *self._windowed_size, flags)
            assert self.window, sdl.SDL_GetError()
            if self.renderer_driver == 'gpu':
                from .sdl_torch_interop import create_device
                self.gpu_device = create_device(self.gpu_driver, self._debug_gpu, self.gpu_interop)
            self.sdl_renderer, self.hdr_mode = create_renderer(
                self.window, self.renderer_driver, self._hdr_request, self._swap_interval,
                gpu_device=self.gpu_device)
            self.hdr = self.hdr_mode != 'sdr'
            self.normalize = self._requested_normalize if not self.hdr else False
            self.supported_hdr_modes = {self.hdr_mode: True}
            self._update_hdr_info()
            print(f'SDL_Renderer: {self.renderer_driver}, presentation={self.hdr_mode}')
            return self.window
        except BaseException:
            self._destroy_platform()
            raise

    def _create_renderer(self):
        renderer = SDLRendererImguiRenderer(self.window, self.sdl_renderer, mode=self.hdr_mode,
            gpu_device=self.gpu_device,
            sdr_white_nits=self.sdr_white_nits, on_content_scale=self._set_content_scale,
            on_wheel=self.pan_handler.consume_wheel, gpu_interop=self.gpu_interop)
        renderer.sc_rgb_white = self.hdr_info['sdr_white_level']
        return renderer

    def _set_content_scale(self, scale):
        self._content_scale = scale
        self.scale_style_sizes()

    def _install_platform_callbacks(self):
        # SDL events are explicitly dispatched in _pump_events.
        pass

    def _update_hdr_info(self):
        props = sdl.SDL_GetRendererProperties(self.sdl_renderer)
        self.hdr_info = {
            'enabled': bool(sdl.SDL_GetBooleanProperty(props, sdl.SDL_PROP_RENDERER_HDR_ENABLED_BOOLEAN, False)),
            'sdr_white_level': float(sdl.SDL_GetFloatProperty(props, sdl.SDL_PROP_RENDERER_SDR_WHITE_POINT_FLOAT, 1.0)),
            'headroom': float(sdl.SDL_GetFloatProperty(props, sdl.SDL_PROP_RENDERER_HDR_HEADROOM_FLOAT, 1.0)),
        }
        if self.impl is not None:
            self.impl.sc_rgb_white = self.hdr_info['sdr_white_level']

    def _dispatch_event(self, event):
        kind = event.type
        if kind == sdl.SDL_EVENT_QUIT:
            self.shutdown()
            return
        if kind in (sdl.SDL_EVENT_DROP_BEGIN, sdl.SDL_EVENT_DROP_FILE, sdl.SDL_EVENT_DROP_COMPLETE):
            if event.drop.windowID != sdl.SDL_GetWindowID(self.window):
                return
            if kind == sdl.SDL_EVENT_DROP_BEGIN:
                self._drop_paths = []
            elif kind == sdl.SDL_EVENT_DROP_FILE and event.drop.data:
                path = Path(event.drop.data.decode())
                if self._drop_paths is None:
                    self.drag_and_drop_callback([path])
                else:
                    self._drop_paths.append(path)
            elif kind == sdl.SDL_EVENT_DROP_COMPLETE:
                paths, self._drop_paths = self._drop_paths, None
                if paths:
                    self.drag_and_drop_callback(paths)
            self.last_ui_active = time.monotonic()
            return
        if sdl.SDL_EVENT_WINDOW_FIRST <= kind <= sdl.SDL_EVENT_WINDOW_LAST:
            if event.window.windowID != sdl.SDL_GetWindowID(self.window):
                return
            if kind == sdl.SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                self.shutdown()
            if kind in (sdl.SDL_EVENT_WINDOW_HDR_STATE_CHANGED, sdl.SDL_EVENT_WINDOW_DISPLAY_CHANGED):
                self._update_hdr_info()
        if self.impl.process_event(event) or sdl.SDL_EVENT_WINDOW_FIRST <= kind <= sdl.SDL_EVENT_WINDOW_LAST:
            # Resize/expose/restore events should render promptly even while idle.
            self.last_ui_active = time.monotonic()

    def _pump_events(self, timeout=0):
        event = sdl.SDL_Event()
        if timeout > 0 and sdl.SDL_WaitEventTimeout(ctypes.byref(event), max(1, round(timeout * 1000))):
            self._dispatch_event(event)
        while sdl.SDL_PollEvent(ctypes.byref(event)):
            self._dispatch_event(event)

    def _get_window_sizes(self):
        size, pixels = window_sizes(self.window)
        if sdl.SDL_GetWindowFlags(self.window) & sdl.SDL_WINDOW_MINIMIZED:
            pixels = (0, 0)
        return size, pixels

    def _window_should_close(self):
        return self._should_close

    def shutdown(self):
        self._should_close = True

    def _apply_window_title(self):
        assert sdl.SDL_SetWindowTitle(self.window, self._window_title.encode()), sdl.SDL_GetError()

    def _swap_buffers(self):
        # The SDL renderer submits and presents in render().
        pass

    def _clear_frame(self):
        # The SDL renderer owns clearing; no OpenGL context is created.
        pass

    def get_texture(self, w, h, capacity=10):
        key = (w, h)
        if key not in self.texture_pool:
            self.texture_pool[key] = RendererTexture(self.impl, w, h,
                srgb=self.image_color_space == 'srgb', pq=self.image_color_space == 'rec2020_pq')
        self.texture_pool.move_to_end(key)
        if len(self.texture_pool) > capacity:
            _, old = self.texture_pool.popitem(last=False)
            old.release()
        return self.texture_pool[key]

    def update_image(self, *, img_hwc=None):
        super().update_image(img_hwc=img_hwc)
        ready = None
        image = self.image
        # Preserve the producer stream dependency across the compute/UI threads.
        if not isinstance(image, np.ndarray) and image.device.type == 'cuda':
            import torch
            with torch.cuda.device(image.device):
                ready = torch.cuda.Event()
                ready.record()
        self._published_image = image, self.img_dt, ready

    def _main_output_impl(self):
        published = self._published_image
        image, image_dt, ready = published if published is not None else (self.image, self.img_dt, None)
        if image is not None:
            height, width, _ = image.shape
            texture = self.get_texture(width, height)
            if image_dt > self.last_upload_dt or not texture.uploaded:
                start = time.monotonic()
                native = not isinstance(image, np.ndarray) and texture.upload_torch(image, ready)
                if not native:
                    if ready is not None:
                        ready.synchronize()
                    if not isinstance(image, np.ndarray):
                        image = image.detach().cpu().numpy()
                    if image.dtype.kind in 'ui':
                        image = image.astype(np.float32) / np.iinfo(image.dtype).max
                    assert image.dtype.kind == 'f'
                    texture.upload(image)
                self.last_upload_dt = image_dt
                self.tex_upload_ms = (time.monotonic() - start) * 1000
            canvas_width, canvas_height = map(int, imgui.get_content_region_avail())
            self.pan_handler.draw(texture, width, height, canvas_width, canvas_height)
        if self.impl.last_upload_backend in ('metal-mps', 'metal-mps-view', 'vulkan-cuda'):
            self.tex_upload_ms = self.impl.last_native_upload_ms
        self.draw_overlay(imgui.get_window_draw_list())

    def keydown(self, key):
        """Poll an SDL scancode; use keyhit(imgui.Key.*) for pressed events."""
        if isinstance(key, str):
            key = getattr(sdl, f'SDL_SCANCODE_{key.upper()}', None)
            if key is None:
                return False
        count = ctypes.c_int()
        keys = sdl.SDL_GetKeyboardState(ctypes.byref(count))
        assert 0 <= key < count.value
        return bool(keys[key])

    def enter_fullscreen(self):
        if not self._is_fullscreen:
            self._windowed_size = self._get_window_sizes()[0]
            assert sdl.SDL_SetWindowFullscreen(self.window, True), sdl.SDL_GetError()
            self._is_fullscreen = True

    def exit_fullscreen(self):
        if self._is_fullscreen:
            assert sdl.SDL_SetWindowFullscreen(self.window, False), sdl.SDL_GetError()
            self._is_fullscreen = False

    def _save_builtin_settings(self):
        # Persist the last windowed geometry when exiting in fullscreen.
        if self._is_fullscreen:
            self.exit_fullscreen()
            sdl.SDL_SyncWindow(self.window)
        super()._save_builtin_settings()

    def _destroy_platform(self):
        if imgui.get_current_context():
            imgui.destroy_context()
        if self.sdl_renderer:
            sdl.SDL_DestroyRenderer(self.sdl_renderer)
            self.sdl_renderer = None
        if self.gpu_device:
            sdl.SDL_DestroyGPUDevice(self.gpu_device)
            self.gpu_device = None
        if self.window:
            sdl.SDL_DestroyWindow(self.window)
            self.window = None
        sdl.SDL_QuitSubSystem(sdl.SDL_INIT_VIDEO | sdl.SDL_INIT_EVENTS)
