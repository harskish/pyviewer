import sys
import json
from pathlib import Path
import numpy as np
import time
import threading
from typing import Union
from collections import OrderedDict
from functools import lru_cache, partial

from . import hdr_patch
from .egl_patch import patch
patch()

# Some callbacks broken if imported before imgui_bundle...??
if sys.platform == 'Darwin':
    assert 'glfw' not in sys.modules or 'imgui_bundle' in sys.modules, 'glfw should be imported after pyviewer'

from imgui_bundle import imgui  # type: ignore
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import glfw
import OpenGL.GL as gl

from .gl_viewer import _texture
from .toolbar_viewer import PannableArea
from .utils import normalize_image_data
from .imgui_themes import theme_deep_dark
from .easy_dict import EasyDict

# Torch import is slow
# => don't import unless already imported by calling code
if "torch" in sys.modules:
    import torch  # for syntax highlighting


def is_tensor(obj):
    if "torch" not in sys.modules:
        return False
    global torch
    torch = sys.modules['torch']  # store handle if imported after docking_viewer
    return torch.is_tensor(obj)


import importlib.util
if not importlib.util.find_spec("torch"):
    is_tensor = lambda obj: False


# Based on:
# https://traineq.org/ImGuiBundle/emscripten/bin/demo_docking.html
# https://github.com/pthom/imgui_bundle/blob/main/bindings/pyodide_web_demo/examples/demo_docking.py


def file_drop_callback_wrapper(window, paths, callback: callable, prev: callable):
    return callback([Path(p) for p in paths]) or prev(window, paths)


class _DockableWindowState:
    def __init__(self, label: str):
        self.label = label
        self.is_visible = True


_dockable_windows: list[tuple[int, int]] = []  # identified by (fun name, title)


def dockable(func=None, title=''):
    """
    Decorator indicating that generated UI elements
    should be placed in a dockable imgui window.
    """

    def wrapper(func):
        window_title = title or func.__name__
        _dockable_windows.append((func.__name__, window_title))
        setattr(func, '_layout_pos', 'Dummy')
        setattr(func, '_title', window_title)
        return func

    return wrapper if func is None else wrapper(func)


class PyDockingViewer:
    def __init__(
        self,
        name: str,
        normalize=False,
        with_implot=True,
        with_implot3d=False,
        with_node_editor=False,
        with_node_editor_config=None,
        with_tex_inspect=False,
        with_font_awesome=False,
    ):
        # Historical references from the previous hello_imgui/immapp backend:
        #  immapp.run() python stub:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/immapp/immapp_cpp.pyi#L162
        #  immapp.run() nanobind impl: https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/bindings/pybind_immapp_cpp.cpp#L158
        #  immapp.run() CPP impl:      https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/immapp/runner.cpp#L233
        #  hello_imgui.run() python stub:     https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/hello_imgui.pyi#L3416
        #  hello_imgui.run() nanobind impl:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/hello_imgui/bindings/pybind_hello_imgui.cpp#L1761
        #  hello_imgui.run() CPP impl:        https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui.cpp#L227

        # Start compute thread asap
        self.start_event: threading.Event = threading.Event()
        self.stop_event: threading.Event = threading.Event()
        compute_thread = threading.Thread(target=self.compute_loop, args=[], daemon=True)
        compute_thread.start()

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self.pan_handler = PannableArea(force_mouse_capture=True)
        self._ui_scale = 1.0
        self.ui_locked = True
        self.first_frame = True

        # Main image (output of self.compute())
        self.image: np.ndarray = None
        self.img_dt: float = 0
        self.img_shape: list = [3, 4, 4]  # CHW (to match ToolbarViewer)
        self.last_upload_dt: float = 0
        self.texture_pool: OrderedDict[tuple[int, int], _texture] = OrderedDict()
        self.state = EasyDict()
        self.tex_upload_ms = 0.0

        self.initial_font_size = 15
        self.default_font: imgui.ImFont = None
        self.code_font: imgui.ImFont = None
        self.fonts: list[imgui.ImFont] = []
        self.window: glfw._GLFWwindow = None
        self.impl: GlfwRenderer = None
        self._window_title = name
        self._orig_window_title = name
        self.load_font_awesome = with_font_awesome
        
        safe_name = name.lower().strip().replace(' ', '_')
        self._ini_path = f'{safe_name}.ini'
        self._prefs_path = Path(f'~/.config/{safe_name}.prefs.json').expanduser() # AppData on Windows?

        # For limiting OpenGL operations to UI thread
        self.ui_tid = threading.get_native_id()

        # Check if HDR mode has been turned on
        self.hdr = (hdr_patch.CUR_MODE == hdr_patch.Mode.PATCHED)

        # Normalize images before showing?
        self.normalize = normalize if not self.hdr else False

        # Optional addons are supported only partially in python backend
        self._implot_ctx = None
        if with_implot:
            try:
                from imgui_bundle import implot
                self._implot_ctx = implot
                implot.create_context()
            except Exception:
                print('Warning: failed to initialize implot context in python backend')

        if with_implot3d or with_node_editor or with_tex_inspect or with_node_editor_config is not None:
            print('Warning: some immapp add-ons are unavailable in python-only backend')

        self.show_app_menu = False
        self.show_view_menu = True
        self._dock_windows_by_label: dict[str, _DockableWindowState] = {}

        # Create window + imgui backend
        self.window = self.impl_glfw_init()
        imgui.create_context()
        self._setup_imgui_context(name)
        self.setup_theme()  # user-overridable
        self.scale_style_sizes()
        self.pan_handler.clear_color = imgui.get_style().color_(imgui.Col_.window_bg)

        self.load_fonts()
        self.impl = GlfwRenderer(self.window, attach_callbacks=True)

        # Set own glfw callbacks, chained with previous callback if present.
        def noop(*args, **kwargs):
            return False

        prev = glfw.set_drop_callback(self.window, None)
        glfw.set_drop_callback(
            self.window,
            partial(file_drop_callback_wrapper, callback=self.drag_and_drop_callback, prev=(prev or noop)),
        )
        self.pan_handler.set_callbacks(self.window)

        self._load_builtin_settings()
        self.load_settings()
        self.setup_state()
        self.start_event.set()

        self._main_loop()

    def impl_glfw_init(self):
        width, height = 1920, 1080
        window_name = self._window_title

        try:
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_WAYLAND)
        except Exception as e:
            pass

        
        # Wayland HDR: setup color management
        if self.hdr and sys.platform == 'linux':
            try:
                glfw.init_hint(hdr_patch.GLFW_WAYLAND_COLOR_MANAGEMENT, glfw.TRUE)
            except:
                raise RuntimeError('GLFW init hint failed, make sure glfw is imported before pyviewer/imgui_bundle')

        if not glfw.init():
            print('Could not initialize OpenGL context')
            sys.exit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        if self.hdr:
            glfw.window_hint(glfw.RED_BITS, 16)
            glfw.window_hint(glfw.GREEN_BITS, 16)
            glfw.window_hint(glfw.BLUE_BITS, 16)
            # KDE plasma: currently fails with floatbuffer
            #if hdr_patch.is_linux:
            #    glfw.window_hint(hdr_patch.GLFW_FLOATBUFFER, gl.GL_TRUE)

        window = glfw.create_window(int(width), int(height), window_name, None, None)
        if not window:
            glfw.terminate()
            print('Could not initialize Window')
            sys.exit(1)

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        return window

    def shutdown(self):
        if self.window:
            glfw.set_window_should_close(self.window, glfw.TRUE)
    
    def _setup_imgui_context(self, name: str):
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard

        # Python bindings may not expose io.ini_filename; load ini manually instead.
        try:
            if Path(self._ini_path).is_file():
                imgui.load_ini_settings_from_disk(self._ini_path)
        except Exception:
            print(f'Warning: failed to load imgui settings from "{self._ini_path}"')

    @property
    def ui_scale(self):
        """UI scale getter"""
        return self._ui_scale

    @ui_scale.setter
    def ui_scale(self, value: float):
        """UI scale setter"""
        self.set_ui_scale(value)

    def set_ui_scale(self, scale: float):
        if self._ui_scale == scale:
            return
        self._ui_scale = scale
        self.scale_style_sizes()

    # Includes keyboard (glfw.KEY_A) and mouse (glfw.MOUSE_BUTTON_LEFT)
    def keydown(self, key: Union[int, str]):
        if isinstance(key, str):
            key_name = f'KEY_{key.upper()}'
            key = getattr(glfw, key_name, None)
            if key is None:
                return False
        return glfw.get_key(self.window, key) == glfw.PRESS

    def keyhit(self, key: imgui.Key):
        return imgui.is_key_pressed(key, repeat=False)

    def scale_style_sizes(self):
        """More conservative alternative to imgui.get_style().scale_all_sizes()"""
        factor = self.ui_scale
        font_size = 9 * factor
        spacing = font_size * 0.3

        s = imgui.get_style()
        s.window_padding = [spacing, spacing]
        s.item_spacing = [spacing, spacing]
        s.item_inner_spacing = [spacing, spacing]
        s.columns_min_spacing = spacing
        s.indent_spacing = font_size
        s.scrollbar_size = font_size + 4

    def _draw_menu_wrapper(self):
        if imgui.begin_main_menu_bar():
            if self.show_app_menu:
                if imgui.begin_menu('App', True):
                    clicked_quit, _ = imgui.menu_item('Quit', 'Cmd+Q', False, True)
                    if clicked_quit:
                        glfw.set_window_should_close(self.window, True)
                    imgui.end_menu()

            if self.show_view_menu:
                if imgui.begin_menu('View', True):
                    for name in sorted(self._dock_windows_by_label.keys()):
                        item = self._dock_windows_by_label[name]
                        clicked, _ = imgui.menu_item(name, '', item.is_visible, True)
                        if clicked:
                            item.is_visible = not item.is_visible
                    imgui.end_menu()

            # User-provided items.
            self.draw_menu()

            # UI resize widget
            # Right-aligned button for locking / unlocking UI
            t = 'L' if self.ui_locked else 'U'
            c = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
            s = self.ui_scale

            # same_line() and negative sizes don't work within menu bar
            # => use an invisible button for spacing instead.
            max_x = imgui.get_window_width()
            cursor = imgui.get_cursor_pos()[0]

            # UI scale slider
            if not self.ui_locked:
                pad = max_x - cursor - 300 - 30 * s
                imgui.invisible_button('##hidden', size=(pad, 1))
                max_scale = 5.0
                min_scale = 0.1

                imgui.set_next_item_width(300)
                ch, val = imgui.slider_float('', s, min_scale, max_scale, format='%.2f')
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    (ch, val) = (True, 1.0)

                if ch:
                    self.set_ui_scale(val)
            else:
                pad = max_x - cursor - 25 * s
                imgui.invisible_button('##hidden', size=(pad, 1))

            imgui.push_style_color(imgui.Col_.text, (*c, 1))
            if imgui.button(t, size=(20 * s, 0)):
                self.ui_locked = not self.ui_locked
            imgui.pop_style_color()

            imgui.end_main_menu_bar()

    def _begin_dockspace(self):
        flags = (
            imgui.WindowFlags_.menu_bar
            | imgui.WindowFlags_.no_docking
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_nav_focus
        )

        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.pos)
        imgui.set_next_window_size(viewport.size)
        imgui.set_next_window_viewport(viewport.id_)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)

        open_flag = True
        imgui.begin('##MainDockSpaceHost', open_flag, flags)
        imgui.pop_style_var(2)

        dockspace_id = imgui.get_id('MainDockSpace')
        dock_flags = imgui.DockNodeFlags_.auto_hide_tab_bar
        imgui.dock_space(dockspace_id, (0.0, 0.0), dock_flags)

    def _draw_dockable_windows(self):
        fun_names, _ = zip(*_dockable_windows) if _dockable_windows else ([], [])
        # Use candidate names to avoid calling getattr on every attribute.
        candidates = [getattr(self, name) for name in dir(self) if name in fun_names]
        layout_funcs = [f for f in candidates if hasattr(f, '_title')]

        # Initialize window visibility map lazily so decorators still work across subclasses.
        for f in layout_funcs:
            if f._title not in self._dock_windows_by_label:
                self._dock_windows_by_label[f._title] = _DockableWindowState(f._title)

        for f in layout_funcs:
            state = self._dock_windows_by_label[f._title]
            if not state.is_visible:
                continue
            expanded, is_open = imgui.begin(f._title, True)
            state.is_visible = is_open
            if expanded:
                f()
            imgui.end()

    def _draw_status_bar_if_needed(self):
        if type(self).draw_status_bar == PyDockingViewer.draw_status_bar:
            return

        viewport = imgui.get_main_viewport()
        style = imgui.get_style()
        pad = style.window_padding[1] + style.item_spacing[1] + 24 * self.ui_scale
        h = max(24 * self.ui_scale, pad)

        imgui.set_next_window_pos((viewport.pos[0], viewport.pos[1] + viewport.size[1] - h))
        imgui.set_next_window_size((viewport.size[0], h))

        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_collapse
        )

        imgui.begin('##StatusBar', True, flags)
        self.draw_status_bar()
        imgui.end()

    def _main_loop(self):
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.impl.process_inputs()

                imgui.new_frame()
                imgui.push_font(self.default_font, self.initial_font_size * self.ui_scale)

                self.pre_new_frame()
                self._draw_menu_wrapper()
                self._begin_dockspace()
                self._draw_dockable_windows()
                self._draw_status_bar_if_needed()
                imgui.end()  # end dockspace host

                self.first_frame = False
                glfw.set_window_title(self.window, self._window_title)
                imgui.pop_font()

                gl.glClearColor(0.08, 0.08, 0.08, 1)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                imgui.render()
                self.impl.render(imgui.get_draw_data())
                glfw.swap_buffers(self.window)
        finally:
            self.stop_event.set()
            self.save_settings()
            self._save_builtin_settings()
            self._cleanup()

    def _load_builtin_settings(self):
        try:
            path = Path(self._prefs_path)
            path.parent.mkdir(exist_ok=True)
            if not path.is_file():
                return
            data = json.loads(path.read_text(encoding='utf-8'))
            self.ui_scale = float(data.get('ui_scale', self._ui_scale))
        except Exception:
            print(f'Warning: failed to load viewer settings from "{self._prefs_path}"')

    def _save_builtin_settings(self):
        try:
            data = {
                'ui_scale': float(self.ui_scale),
            }
            Path(self._prefs_path).write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:
            print(f'Warning: failed to save viewer settings to "{self._prefs_path}"')

    def _cleanup(self):
        try:
            imgui.save_ini_settings_to_disk(self._ini_path)
        except Exception:
            print(f'Warning: failed to save imgui settings to "{self._ini_path}"')

        for handle in self.texture_pool.values():
            del handle
        self.texture_pool.clear()

        if self.impl is not None:
            self.impl.shutdown()

        if self._implot_ctx is not None:
            try:
                self._implot_ctx.destroy_context()
            except Exception:
                pass

        glfw.terminate()

    def get_default_font_path(self):
        font = Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def get_mono_font_path(self):
        font = Path(__file__).parent / 'Hack-Regular.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def load_fonts(self):
        # Main and monospace font are loaded from local files so the port behaves
        # similarly to the original hello_imgui-based viewer.
        io = imgui.get_io()
        size = self.initial_font_size * self.ui_scale
        self.default_font = io.fonts.add_font_from_file_ttf(self.get_default_font_path(), size)
        self.code_font = io.fonts.add_font_from_file_ttf(self.get_mono_font_path(), size)

        if self.default_font is None:
            self.default_font = io.fonts.add_font_default()
        if self.code_font is None:
            self.code_font = self.default_font

        self.fonts = [self.default_font, self.code_font]

    def get_window(self, name: str):
        return self._dock_windows_by_label.get(name, None)

    def toggle_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = not w.is_visible

    def hide_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = False

    def show_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = True

    def toggle_menu(self):
        self.show_view_menu = not self.show_view_menu

    @lru_cache(maxsize=4)
    def alpha_ch_torch(self, h, w, maxval, dtype, device):
        """
        Get alpha channel for padding image data to rgba.
        Cached to speed up repeated padding of GPU tensors.
        """
        return maxval * torch.ones((h, w, 1), dtype=dtype, device=device)

    def update_image(self, *, img_hwc=None):
        is_np = isinstance(img_hwc, np.ndarray)
        assert is_np or is_tensor(img_hwc), 'Expected np.ndarray or torch.Tensor'

        is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
        is_signed = (img_hwc.dtype.kind == 'i') if is_np else img_hwc.dtype.is_signed
        dtype_bits = img_hwc.dtype.itemsize * 8
        h, w, _ = img_hwc.shape

        img_hwc = normalize_image_data(img_hwc, img_hwc.dtype) if self.normalize else img_hwc

        # RGBA texture uploads are much faster on some drivers
        if img_hwc.shape[-1] == 3:
            maxval = 1 if is_fp else 2 ** (dtype_bits - int(is_signed)) - 1
            if is_np:
                img_hwc = np.concatenate([img_hwc, maxval * np.ones((h, w, 1), dtype=img_hwc.dtype)], axis=-1)
            else:
                img_hwc = torch.cat([img_hwc, self.alpha_ch_torch(h, w, maxval, img_hwc.dtype, img_hwc.device)], dim=2)

        # Eventually uploaded by UI thread
        self.image = img_hwc
        self.img_shape = [self.image.shape[2], *self.image.shape[:2]]
        self.img_dt = time.monotonic()

    def get_texture(self, w: int, h: int, capacity=10):
        """Keep a pool of GL textures of different sizes"""
        key = (w, h)
        if key not in self.texture_pool:
            self.texture_pool[key] = _texture(gl.GL_NEAREST, gl.GL_NEAREST)

        self.texture_pool.move_to_end(key)
        if len(self.texture_pool) > capacity:
            self.texture_pool.popitem(last=False)

        return self.texture_pool[key]

    def _main_output_impl(self):
        # Need to do upload from main thread
        if self.image is not None and self.img_dt > self.last_upload_dt:
            gl.glFinish()
            t0 = time.monotonic()

            # Get GL texture of appropriate size.
            t_h, t_w, _ = self.image.shape
            tex_handle = self.get_texture(t_w, t_h)

            if isinstance(self.image, np.ndarray):
                tex_handle.upload_np(self.image)
            elif self.image.device.type == 'cuda':
                tex_handle.upload_torch(self.image)
            elif self.image.device.type == 'mps':
                tex_handle.upload_torch(self.image)
            else:
                tex_handle.upload_np(self.image.cpu().numpy())
            self.last_upload_dt = time.monotonic()
            self.tex_upload_ms = (self.last_upload_dt - t0) * 1000

        if self.image is not None:
            # Reallocate canvas if content region has changed.
            t_h, t_w, _ = self.image.shape
            c_w, c_h = map(int, imgui.get_content_region_avail())
            handle = self.get_texture(t_w, t_h)
            canvas_tex = self.pan_handler.draw_to_canvas(handle.tex, t_w, t_h, c_w, c_h, handle.type)
            imgui.image(canvas_tex, (c_w, c_h))

        draw_list = imgui.get_window_draw_list()
        self.draw_overlay(draw_list)

    @dockable(title='Main Output')
    def output(self):
        return self._main_output_impl()

    def compute_loop(self):
        print('Compute thread: waiting for start event')
        self.start_event.wait(timeout=None)
        print('Compute thread: start event received')

        while not self.stop_event.is_set():
            ret = self.compute()
            if ret is not None:
                self.update_image(img_hwc=ret)
            time.sleep(0.01)

        print('Compute thread: received stop event, exiting')

    def reset_window_title(self):
        self.set_window_title(self._orig_window_title, suffix=False)

    def set_window_title(self, title, suffix=False):
        """
        Set window title.
        If called from compute thread: updated later by UI thread.
        If called from UI thread: updated immediately.
        suffix: if True, append title as suffix to original.
        """
        if suffix:
            title = f'{self._orig_window_title} - {title}'
        self._window_title = title
        if threading.get_native_id() == self.ui_tid and self.window is not None:
            # If called from compute thread, this gets applied in the render loop.
            glfw.set_window_title(self.window, title)

    #########################
    # User provided callbacks
    #########################

    def draw_status_bar(self):
        pass

    def draw_menu(self):
        pass

    def draw_overlay(self, draw_list: imgui.ImDrawList):
        """Draw overlay on top of main output window"""
        pass

    # Perform computation, returning single np/torch image, or None
    def compute(self):
        pass

    # Program state init
    def setup_state(self):
        pass

    # Can be overridden
    def setup_theme(self):
        theme_deep_dark()

    def pre_new_frame(self):
        """
        Called each frame before drawing user windows.
        Good place to add transient UI state updates.
        """
        pass

    def load_settings(self):
        pass

    def save_settings(self):
        pass

    def drag_and_drop_callback(self, paths: list[Path]) -> bool:
        return False
