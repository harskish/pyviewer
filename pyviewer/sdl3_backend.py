"""SDL3 platform integration, based on imgui_bundle's Python SDL3 backend.

Reuses upstream's keyboard map, without initializing its OpenGL renderer.
Adds framebuffer/content scaling, five mouse buttons, and precise wheel input.
https://github.com/pthom/imgui_bundle/blob/v1.92.5/bindings/imgui_bundle/python_backends/sdl3_backend.py
"""

import ctypes
import math
import os

from .import_order import prepare_imgui_import

prepare_imgui_import()

# This backend owns its event loop; PySDL3 must not call __main__.main at exit.
os.environ.setdefault("SDL_MAIN_HANDLED", "1")
import sdl3 as sdl
from imgui_bundle import imgui
from imgui_bundle.python_backends.sdl3_backend import SDL3Renderer as UpstreamSDL3Renderer


def window_sizes(window):
    sizes = []
    for get_size in (sdl.SDL_GetWindowSize, sdl.SDL_GetWindowSizeInPixels):
        width, height = ctypes.c_int(), ctypes.c_int()
        assert get_size(window, ctypes.byref(width), ctypes.byref(height)), sdl.SDL_GetError()
        sizes.append((width.value, height.value))
    return tuple(sizes)


def dpi_scales(window_size, pixel_size, display_scale):
    """Return framebuffer density and UI scale in SDL window coordinates.

    Retina/Wayland: density supplies extra pixels. Windows/X11: content scale
    enlarges the UI in pixel-based window coordinates. Do not apply both twice.
    """
    density = tuple(p / w if w > 0 and p > 0 else 1.0
                    for w, p in zip(window_size, pixel_size))
    return density, display_scale / density[0]


def wheel_delta(event):
    """Preserve OS scroll direction and fractional trackpad motion.

    SDL's deltas already include the user's natural-scrolling preference.
    SDL_MOUSEWHEEL_FLIPPED describes that preference; reversing the deltas again
    inverts macOS trackpad zoom. Match ImGui's SDL3 backend: only negate X to
    convert SDL's right-positive horizontal axis to ImGui's left-positive axis.
    """
    return -float(event.wheel.x), float(event.wheel.y)


class SDL3Platform:
    _map_keys = UpstreamSDL3Renderer._map_keys

    def __init__(self, window, *, on_content_scale=None, on_wheel=None):
        self.window = window
        self.io = imgui.get_io()
        self.window_id = sdl.SDL_GetWindowID(window)
        self.on_content_scale = on_content_scale
        self.on_wheel = on_wheel
        self.content_scale = 1.0
        self._buttons_down = set()
        self._gui_time = None
        self._map_keys()
        platform = imgui.get_platform_io()
        platform.platform_get_clipboard_text_fn = lambda _ctx: sdl.SDL_GetClipboardText().decode()
        platform.platform_set_clipboard_text_fn = lambda _ctx, text: sdl.SDL_SetClipboardText(text.encode())
        sdl.SDL_StartTextInput(window)
        self.process_inputs()

    def process_event(self, event):
        kind = event.type
        # These SDL event structs share the windowID field's position.
        window_events = (
            sdl.SDL_EVENT_MOUSE_MOTION, sdl.SDL_EVENT_MOUSE_WHEEL,
            sdl.SDL_EVENT_MOUSE_BUTTON_DOWN, sdl.SDL_EVENT_MOUSE_BUTTON_UP,
            sdl.SDL_EVENT_KEY_DOWN, sdl.SDL_EVENT_KEY_UP, sdl.SDL_EVENT_TEXT_INPUT,
        )
        is_window_event = sdl.SDL_EVENT_WINDOW_FIRST <= kind <= sdl.SDL_EVENT_WINDOW_LAST
        if (kind in window_events or is_window_event) and event.window.windowID != self.window_id:
            return False

        io = self.io
        if kind == sdl.SDL_EVENT_MOUSE_MOTION:
            io.add_mouse_pos_event(event.motion.x, event.motion.y)
            return True

        if kind in (sdl.SDL_EVENT_MOUSE_BUTTON_DOWN, sdl.SDL_EVENT_MOUSE_BUTTON_UP):
            button = {
                sdl.SDL_BUTTON_LEFT: 0, sdl.SDL_BUTTON_RIGHT: 1,
                sdl.SDL_BUTTON_MIDDLE: 2, sdl.SDL_BUTTON_X1: 3, sdl.SDL_BUTTON_X2: 4,
            }.get(event.button.button)
            if button is None:
                return False
            down = kind == sdl.SDL_EVENT_MOUSE_BUTTON_DOWN
            io.add_mouse_pos_event(event.button.x, event.button.y)
            io.add_mouse_button_event(button, down)
            if down:
                self._buttons_down.add(button)
            else:
                self._buttons_down.discard(button)
            sdl.SDL_CaptureMouse(bool(self._buttons_down))
            return True

        if kind == sdl.SDL_EVENT_MOUSE_WHEEL:
            x, y = wheel_delta(event)
            pos = (event.wheel.mouse_x, event.wheel.mouse_y)
            io.add_mouse_pos_event(*pos)
            if self.on_wheel is None or not self.on_wheel(x, y, pos):
                io.add_mouse_wheel_event(x, y)
            return True

        if kind in (sdl.SDL_EVENT_WINDOW_FOCUS_GAINED, sdl.SDL_EVENT_WINDOW_FOCUS_LOST):
            focused = kind == sdl.SDL_EVENT_WINDOW_FOCUS_GAINED
            io.add_focus_event(focused)
            if not focused:
                self._buttons_down.clear()
                sdl.SDL_CaptureMouse(False)
            return True

        if kind == sdl.SDL_EVENT_WINDOW_MOUSE_LEAVE:
            if not self._buttons_down:
                io.add_mouse_pos_event(-3.4028235e38, -3.4028235e38)
            return True

        if kind in (sdl.SDL_EVENT_KEY_DOWN, sdl.SDL_EVENT_KEY_UP):
            key = self.key_map.get(event.key.scancode)
            if key is not None:
                io.add_key_event(key, kind == sdl.SDL_EVENT_KEY_DOWN)
            for flag, modifier in ((sdl.SDL_KMOD_CTRL, imgui.Key.mod_ctrl),
                                   (sdl.SDL_KMOD_SHIFT, imgui.Key.mod_shift),
                                   (sdl.SDL_KMOD_ALT, imgui.Key.mod_alt),
                                   (sdl.SDL_KMOD_GUI, imgui.Key.mod_super)):
                io.add_key_event(modifier, bool(event.key.mod & flag))
            return True
        if kind == sdl.SDL_EVENT_TEXT_INPUT:
            io.add_input_characters_utf8(event.text.text.decode())
            return True
        return False

    def process_inputs(self):
        size, pixels = window_sizes(self.window)
        display_scale = float(sdl.SDL_GetWindowDisplayScale(self.window))
        assert display_scale > 0, sdl.SDL_GetError()
        density, content_scale = dpi_scales(size, pixels, display_scale)
        self.io.display_size = size
        self.io.display_framebuffer_scale = density
        if not math.isclose(content_scale, self.content_scale):
            self.content_scale = content_scale
            if self.on_content_scale is not None:
                self.on_content_scale(content_scale)

        now = sdl.SDL_GetTicksNS() * 1e-9
        self.io.delta_time = max(1e-6, now - self._gui_time) if self._gui_time is not None else 1 / 60
        self._gui_time = now

    def shutdown(self):
        sdl.SDL_CaptureMouse(False)
        sdl.SDL_StopTextInput(self.window)
