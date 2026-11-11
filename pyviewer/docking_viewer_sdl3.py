"""SDL_GPU + imgui_bundle prototype with HDR presentation.

Uses Vulkan on Linux/Windows and Metal on macOS, with stock PySDL3 bindings.
Floating-point images are linear Rec.709 by default, with 1.0 meaning SDR white.
Torch images use a CPU upload path; the GLFW viewer's GL interop is not used.
"""

import ctypes
from pathlib import Path
import sys
import time

import numpy as np

from .docking_viewer_py import PyDockingViewer, dockable  # re-export the shared decorator
from .sdl3_backend import window_sizes
from .sdl_gpu_renderer import SDLGPUImguiRenderer, GPUTexture, checked
from .utils import PannableArea

import sdl3 as sdl
from imgui_bundle import imgui


class SDL3PannableArea(PannableArea):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._wheel_pos = None

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
        self.handle_pan()
        draw_list = imgui.get_window_draw_list()
        lo = tuple(self.output_pos_tl)
        hi = tuple(self.output_pos_tl + (canvas_width, canvas_height))
        draw_list.push_clip_rect(lo, hi, True)
        draw_list.add_rect_filled(lo, hi, imgui.color_convert_float4_to_u32(self.clear_color))
        # Existing numerical pan/zoom transforms; rasterization is now SDL_GPU.
        uv = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        points = uv @ self.uv_to_screen_xform().T
        draw_list.add_image_quad(texture.ref, *(tuple(p[:2]) for p in points))
        draw_list.pop_clip_rect()

    def consume_wheel(self, x, y, pos):
        self._wheel_pos = pos
        try:
            if not (self.zoom_enabled and self.mouse_hovers_content() and self.tex_h > 0):
                return False
            self.mouse_wheel_callback(None, x, y)
            return True
        finally:
            self._wheel_pos = None


class SDL3DockingViewer(PyDockingViewer):
    """Same dockable UI API, with SDL_GPU window/events and HDR presentation.

    keydown accepts an SDL scancode or a name such as 'A' or 'ESCAPE'.
    hdr: 'auto' prefers linear HDR then PQ then SDR; True requires HDR;
    'linear'/'pq' require that mode, False/'sdr' selects SDR explicitly.
    image_color_space: 'linear' (1.0 = SDR white) or 'srgb', including floats.
    sdr_white_nits: reference white for PQ output. Linear output uses SDL's
    reported SDR-white scale. Display capability is exposed through hdr_info.
    """

    def __init__(self, name, *, hdr='auto', image_color_space='linear', sdr_white_nits=203.0,
                 gpu_driver=None, debug_gpu=False, normalize=False, **kwargs):
        assert 'context_creation_api' not in kwargs, 'GLFW context_creation_api is not an SDL option'
        assert hdr in ('auto', 'linear', 'pq', 'sdr', True, False)
        assert image_color_space in ('linear', 'srgb')
        assert sdr_white_nits > 0
        self._hdr_request = hdr
        self.image_color_space = image_color_space
        self.sdr_white_nits = sdr_white_nits
        self.gpu_driver = gpu_driver or ('metal' if sys.platform == 'darwin' else 'vulkan')
        assert self.gpu_driver in ('metal', 'vulkan'), 'Bundled shaders target Metal and Vulkan'
        self._debug_gpu = debug_gpu
        self._requested_normalize = normalize
        self._content_scale = 1.0
        self._should_close = False
        self.gpu_device = None
        self._window_claimed = False
        self.hdr_info = {}
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
        assert sdl.SDL_Init(sdl.SDL_INIT_VIDEO | sdl.SDL_INIT_EVENTS), sdl.SDL_GetError()
        try:
            flags = sdl.SDL_WINDOW_RESIZABLE | sdl.SDL_WINDOW_HIGH_PIXEL_DENSITY
            if self._hidden:
                flags |= sdl.SDL_WINDOW_HIDDEN
            self.window = sdl.SDL_CreateWindow(self._window_title.encode(), *self._windowed_size, flags)
            assert self.window, sdl.SDL_GetError()
            formats = sdl.SDL_GPU_SHADERFORMAT_SPIRV | sdl.SDL_GPU_SHADERFORMAT_MSL
            self.gpu_device = checked(sdl.SDL_CreateGPUDevice(formats, self._debug_gpu, self.gpu_driver.encode()))
            checked(sdl.SDL_ClaimWindowForGPUDevice(self.gpu_device, self.window))
            self._window_claimed = True
            self._configure_swapchain()
            self._update_hdr_info()
            return self.window
        except BaseException:
            self._destroy_platform()
            raise

    def _create_renderer(self):
        renderer = SDLGPUImguiRenderer(self.window, self.gpu_device, mode=self.hdr_mode,
            sdr_white_nits=self.sdr_white_nits, on_content_scale=self._set_content_scale,
            on_wheel=self.pan_handler.consume_wheel)
        renderer.sc_rgb_white = self.hdr_info['sdr_white_level']
        return renderer

    def _configure_swapchain(self):
        compositions = {'linear': sdl.SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR,
                        'pq': sdl.SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084,
                        'sdr': sdl.SDL_GPU_SWAPCHAINCOMPOSITION_SDR}
        request = self._hdr_request
        candidates = (['linear', 'pq', 'sdr'] if request == 'auto' else
                      ['linear', 'pq'] if request is True else
                      ['sdr'] if request is False else [request])
        supported = {name: bool(sdl.SDL_WindowSupportsGPUSwapchainComposition(
            self.gpu_device, self.window, composition)) for name, composition in compositions.items()}
        choices = [mode for mode in candidates if supported[mode]]
        assert choices, f'Requested HDR mode {request!r} is unsupported: {supported}'
        self.hdr_mode = choices[0]
        self.hdr = self.hdr_mode != 'sdr'
        self.normalize = self._requested_normalize if not self.hdr else False
        present = sdl.SDL_GPU_PRESENTMODE_VSYNC
        if self._swap_interval == 0:
            for option in (sdl.SDL_GPU_PRESENTMODE_IMMEDIATE, sdl.SDL_GPU_PRESENTMODE_MAILBOX):
                if sdl.SDL_WindowSupportsGPUPresentMode(self.gpu_device, self.window, option):
                    present = option
                    break
        checked(sdl.SDL_SetGPUSwapchainParameters(self.gpu_device, self.window,
                                                compositions[self.hdr_mode], present))
        self.supported_hdr_modes = supported
        print(f'SDL_GPU: {self.gpu_driver}, presentation={self.hdr_mode}, supported={supported}')

    def _set_content_scale(self, scale):
        self._content_scale = scale
        self.scale_style_sizes()

    def _install_platform_callbacks(self):
        # SDL events are explicitly dispatched in _pump_events.
        pass

    def _update_hdr_info(self):
        props = sdl.SDL_GetWindowProperties(self.window)
        self.hdr_info = {
            'enabled': bool(sdl.SDL_GetBooleanProperty(props, sdl.SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, False)),
            'sdr_white_level': float(sdl.SDL_GetFloatProperty(props, sdl.SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 1.0)),
            'headroom': float(sdl.SDL_GetFloatProperty(props, sdl.SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0)),
        }
        if self.impl is not None:
            self.impl.sc_rgb_white = self.hdr_info['sdr_white_level']

    def _dispatch_event(self, event):
        kind = event.type
        if kind == sdl.SDL_EVENT_QUIT:
            self.shutdown()
            return
        if kind == sdl.SDL_EVENT_DROP_FILE:
            if event.drop.windowID == sdl.SDL_GetWindowID(self.window):
                self.drag_and_drop_callback([Path(event.drop.data.decode())])
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
        # SDL_GPU submits and presents together in renderer.render().
        pass

    def _clear_frame(self):
        # SDL_GPU render passes own clearing; no OpenGL context is created.
        pass

    def get_texture(self, w, h, capacity=10):
        key = (w, h)
        if key not in self.texture_pool:
            self.texture_pool[key] = GPUTexture(self.impl, w, h,
                srgb=self.image_color_space == 'srgb')
        self.texture_pool.move_to_end(key)
        if len(self.texture_pool) > capacity:
            _, old = self.texture_pool.popitem(last=False)
            old.release()
        return self.texture_pool[key]

    def _main_output_impl(self):
        image, image_dt = self.image, self.img_dt
        if image is not None:
            height, width, _ = image.shape
            texture = self.get_texture(width, height)
            if image_dt > self.last_upload_dt:
                start = time.monotonic()
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
        self.draw_overlay(imgui.get_window_draw_list())

    def keydown(self, key):
        """Poll an SDL scancode; use keyhit(imgui.Key.*) for pressed events."""
        if isinstance(key, str):
            key = getattr(sdl, f'SDL_SCANCODE_{key.upper()}')
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
        if self._window_claimed:
            sdl.SDL_ReleaseWindowFromGPUDevice(self.gpu_device, self.window)
            self._window_claimed = False
        if self.gpu_device:
            sdl.SDL_DestroyGPUDevice(self.gpu_device)
            self.gpu_device = None
        if self.window:
            sdl.SDL_DestroyWindow(self.window)
            self.window = None
        sdl.SDL_QuitSubSystem(sdl.SDL_INIT_VIDEO | sdl.SDL_INIT_EVENTS)
