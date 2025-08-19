import abc
import functools
import sys
from pathlib import Path
import types
import typing
import numpy as np
import time
import threading
import time
from typing import Union
from functools import lru_cache

# Some callbacks broken if imported before imgui_bundle...??
assert 'glfw' not in sys.modules or 'imgui_bundle' in sys.modules, 'glfw should be imported after pyviewer'

from imgui_bundle.demos_python.demo_utils import demos_assets_folder
from imgui_bundle import hello_imgui, glfw_utils, imgui, immapp # type: ignore
from imgui_bundle.glfw_utils import glfw
import OpenGL.GL as gl

from .gl_viewer import _texture
from .toolbar_viewer import PannableArea
from .utils import normalize_image_data, float_flip_lsb
from .imgui_themes import theme_deep_dark
from .easy_dict import EasyDict

# Torch import is slow
# => don't import unless already imported by calling code
if "torch" in sys.modules:
    import torch # for syntax highlighting

def is_tensor(obj):
    return "torch" in sys.modules and torch.is_tensor(obj)

import importlib.util
if not importlib.util.find_spec("torch"):
    is_tensor = lambda obj: False

# Why not Python backend?
# - No Nanovg (https://github.com/pthom/imgui_bundle/issues/259#issuecomment-2391258789)
# - No fps throttling ("sleep mode" on inactivity)
# - Not well supported/tested in general
# - Probably won't get non-glfw backends
# - CPP backend sometimes releases GIL, should be slightly more performant

# Based on:
# https://traineq.org/ImGuiBundle/emscripten/bin/demo_docking.html
# https://github.com/pthom/imgui_bundle/blob/main/bindings/pyodide_web_demo/examples/demo_docking.py

def layout(pos: str):
    """
    [UNUSED] Decorator that specifies a dockable window.
    Arg: pos = initial window position (compass direction + C).
    """
    repl = {
        'T': 'N', 'B': 'S', 'L': 'W', 'R': 'E',
        'TL': 'NW', 'TR': 'NE', 'BL': 'SW', 'BR': 'SE',
        'M': 'C' # center/middle
    }
    valid = [*repl.keys(), *repl.values()]
    assert pos in valid, f'Invalid position specifier {pos}, options are: {valid}'
    pos = repl.get(pos, pos)

    def wrapper(func):
        setattr(func, '_layout_pos', pos)
        return func

    return wrapper

def file_drop_callback_wrapper(window: glfw._GLFWwindow, paths: list[Path], callback: typing.Callable[[list[Path]], bool], prev: typing.Callable):
    return callback([Path(p) for p in paths]) or prev(window, paths)

@lru_cache(maxsize=4)
def alpha_ch_torch(H, W, maxval, dtype, device):
    """
    Get alpha channel for padding image data to rgba.
    Cached to speed up repeated padding of GPU tensors.
    """
    return maxval * torch.ones((H, W, 1), dtype=dtype, device=device)

# ------------------------------------------------------------------------------------------------------------
# Docking window registry

_dockable_windows: list[tuple[str, str]] = [] # identified by (fun name, title)

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

# ------------------------------------------------------------------------------------------------------------
# Base class

class DockingViewerBase(metaclass=abc.ABCMeta):
    def __init__(self, runner_params: hello_imgui.RunnerParams, addons: immapp.AddOnsParams):
        immapp.run(runner_params, add_ons_params=addons)

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
            return # setter might be called repeatedly e.g. via imgui.slider
        old_ui_scale = self._ui_scale
        self._ui_scale = scale
        self.scale_style_sizes()

        # Rescale all fonts (even if they are merged)
        for font in self.fonts:
            font.font_size = font.font_size * self.ui_scale / old_ui_scale

        self.trigger_font_reload()

    # Includes keyboard (glfw.KEY_A) and mouse (glfw.MOUSE_BUTTON_LEFT)
    def keydown(self, key: Union[int, str]):
        raise NotImplementedError()

    def keyhit(self, key: imgui.Key):
        return imgui.is_key_pressed(key, repeat=False)

    def scale_style_sizes(self):
        """More conservative alternative to imgui.get_style().scale_all_sizes()"""
        factor = self.ui_scale
        font_size = 9 * factor #self._cur_font_size
        spacing = round(font_size * 0.3)

        s = imgui.get_style()
        s.window_padding        = [spacing, spacing]
        s.item_spacing          = [spacing, spacing]
        s.item_inner_spacing    = [spacing, spacing]
        s.columns_min_spacing   = spacing
        s.indent_spacing        = font_size
        s.scrollbar_size        = font_size + 4

    def _draw_menu_wrapper(self, runner_params: hello_imgui.RunnerParams):
        if self.show_app_menu:
            hello_imgui.show_app_menu(runner_params) # quit button
        if self.show_view_menu:
            hello_imgui.show_view_menu(runner_params) # status bar show/hide, docking layout reset, etc.

        # User-provided
        self.draw_menu()

        # UI resize widget

        # Right-aligned button for locking / unlocking UI
        T = 'L' if self.ui_locked else 'U'
        C = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
        s = self.ui_scale

        # same_line() and negative sizes don't work within menu bar
        # => use invisible button instead
        # Tested: 16.3.2025, imgui_bundle==1.6.2
        max_x = imgui.get_window_width()
        cursor = imgui.get_cursor_pos()[0]

        # UI scale slider
        if not self.ui_locked:
            pad = max_x - cursor - 300 - 30*s
            imgui.invisible_button('##hidden', size=(pad, 1))
            max_scale = 5
            min_scale = 0.1
            #max_scale = max(self.v._imgui_fonts.keys()) / self.v.default_font_size
            #min_scale = min(self.v._imgui_fonts.keys()) / self.v.default_font_size

            imgui.set_next_item_width(300) # size not dependent on s => prevents slider drift
            ch, val = imgui.slider_float('', s, min_scale, max_scale, format="%.1f")
            if imgui.is_item_hovered():
                if imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    (ch, val) = (True, 1.0)

            if ch:
                self.set_ui_scale(val)
        else:
            pad = max_x - cursor - 25*s
            imgui.invisible_button('##hidden', size=(pad, 1))

        imgui.push_style_color(imgui.Col_.text, (*C, 1))
        if imgui.button(T, size=(20*s, 0)):
            self.ui_locked = not self.ui_locked
        imgui.pop_style_color()

    def pre_new_frame(self):
        """
        Called each frame before ImGui::NewFrame().
        Good place to add new dockable windows etc.
        """
        pass

    def trigger_font_reload(self):
        """
        Trigger font reload after changing font.font_size.
        Used in conjunction with hello_imgui.load_font_dpi_responsive().
        """
        io = imgui.get_io()
        io.font_global_scale = float_flip_lsb(io.font_global_scale) # trigger reload

    def setup_layout_pos_based(self) -> hello_imgui.DockingParams:
        """
        Create initial docking layout based on @layout decorators.
        Only used initially, subsequent runs will reload previous layout.
        """
        splits = []
        def create_split(src: str, name: str, dir: imgui.Dir, ratio=0.25):
            split = hello_imgui.DockingSplit()
            split.initial_dock = src
            split.new_dock = name
            split.direction = dir
            split.ratio = ratio
            splits.append(split)
            return name

        def get_layout_dims(positions: set[str]) -> tuple[int, int]:
            all_pos_cat = ''.join(positions) # e.g. NWESW
            layout_width = 0
            layout_height = 0

            #  Figure out width
            if 'E' in all_pos_cat:
                layout_width += 1
            if 'W' in all_pos_cat:
                layout_width += 1
            if 'C' in all_pos_cat or 'N' in all_pos_cat or 'S' in all_pos_cat:
                layout_width += 1 # central or opposite to other horizontal pos

            # Figure out height
            if 'N' in all_pos_cat:
                layout_height += 1
            if 'S' in all_pos_cat:
                layout_height += 1
            if 'C' in all_pos_cat or 'E' in all_pos_cat or 'W' in all_pos_cat:
                layout_height += 1 # central or opposite to other vertical pos

            return (layout_width, layout_height)

        # Find all functions with @layout decorator
        all_funcs = [getattr(self, method_name) for method_name in dir(self) if callable(getattr(self, method_name))]
        layout_funcs = [func for func in all_funcs if hasattr(func, '_layout_pos')]

        # Create splits based on desired positions
        positions = set(f._layout_pos for f in layout_funcs)
        pos_to_dock_constr = {} # {NSEW...} => dock constructor
        main_dock = 'MainDockSpace'

        # Example layouts:
        #  _______________   _______________   _______________   _______________   _____________
        #  | NW | N | NE |   | NW | N | NE |   | NW |   | NE |   | NW | N | NE |   |   | N |   |
        #  |----|---|----|   |----|---|----|   |----| C |----|   |----|---|----|   | W |---| E |
        #  | W  | C |  E |   | W  | C |  E |   | W  |   |  E |   | W    |   E  |   |   | C |   |
        #  |----|---|----|   |-------------|   |-------------|   |-------------|   |-----------|
        #  | SW | S | SE |   |      S      |   |      S      |   |      S      |   |     S     |
        #  ---------------   ---------------   ---------------   ---------------   -------------

        # _______________          _______________
        # | NW |   | NE |          | NW |   | NE |
        # |----|---|----|  biggest |----|---|----|
        # | W  | C |  E |   first  | W  | C |  E |
        # |----|---|----|    =>    |-------------|
        # |    | S |    |          |      S      |
        # ---------------          ---------------

        assert get_layout_dims({'C'}) == (1, 1)
        assert get_layout_dims({'N', 'S'}) == (1, 2)
        assert get_layout_dims({'N', 'S', 'C'}) == (1, 3)
        assert get_layout_dims({'E', 'W', 'C'}) == (3, 1)
        assert get_layout_dims({'E', 'S', 'C'}) == (2, 2) # non-standard
        assert get_layout_dims({'E', 'NW', 'SW'}) == (2, 2)
        assert get_layout_dims({'N', 'S', 'E', 'W'}) == (3, 2)
        assert get_layout_dims({'NW', 'S', 'NE'}) == (3, 1)

        raise NotImplementedError('Not implemented')

    def get_default_font_path(self):
        font = Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def get_mono_font_path(self):
        font = Path(__file__).parent / 'Hack-Regular.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def load_fonts(self):
        # BUNDLED:
        # Akronim-Regular.ttf, DroidSans.ttf, Font_Awesome_6_Free-Solid-900.otf, Inconsolata-Medium.ttf,
        # NotoEmoji-Regular.ttf, entypo.ttf, fontawesome-webfont.ttf, Playbox/Playbox-FREE.otf,
        # Roboto/Roboto-Bold.ttf, Roboto/Roboto-BoldItalic.ttf, Roboto/Roboto-Regular.ttf, Roboto/Roboto-RegularItalic.ttf

        # Load the main font
        size = self.initial_font_size * self.ui_scale
        external = hello_imgui.FontLoadingParams(inside_assets=False)
        self.fonts.append(hello_imgui.load_font_dpi_responsive(self.get_default_font_path(), size, external))

        # Merge with Font Awesome 6 (fontawesome.com/search?o=r&ic=free&s=solid&ip=classic)
        # Both font handles must be kept around for resizing
        if self.load_font_awesome:
            # TODO: if playing with UI scale slider: due to non-default range, will eventually trigger
            # github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui_font.cpp#L71
            font_loading_params_title_icons = hello_imgui.FontLoadingParams(merge_to_last_font=True, use_full_glyph_range=True)
            self.fonts.append(hello_imgui.load_font_dpi_responsive(
                "fonts/Font_Awesome_6_Free-Solid-900.otf", size, font_loading_params_title_icons))

        self.code_font = hello_imgui.load_font_dpi_responsive(self.get_mono_font_path(), size, external)
        self.fonts.append(self.code_font)

    def get_window(self, name: str) -> hello_imgui.DockableWindow | None:
        windows = hello_imgui.get_runner_params().docking_params.dockable_windows
        for w in windows:
            if w.label == name:
                return w
        print(f'No DockableWindow with label "{name}"')
        return None

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
        params = hello_imgui.get_runner_params().imgui_window_params
        params.show_menu_bar = not params.show_menu_bar

    def reset_window_title(self):
        self._window_title = self._orig_window_title
    
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
        if threading.get_native_id() == self.ui_tid:
            # need glfw.make_context_current if calling from compute thread?
            glfw.set_window_title(self.window, title)

    # -----------------------------------------------------------------------------
    # Abstract methods
    
    @abc.abstractmethod
    def setup_layout(self) -> hello_imgui.DockingParams:
        pass

    @abc.abstractmethod
    def compute_loop(self) -> None:
        pass
    
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

    # Program state init
    def setup_state(self):
        pass

    # Can be overridden
    def setup_theme(self):
        theme_deep_dark()

    def load_settings(self):
        """Load settings using `hello_imgui.load_user_pref(k: str)`"""
        pass

    def save_settings(self):
        """Save settings using `hello_imgui.save_user_pref(k: str, v: str)`"""
        pass
    
    def compute(self):
        """Perform computation, returning single np/torch image, or None
        """
        pass

# ------------------------------------------------------------------------------------------------------------
# Single texture docking viewer

class DockingViewer(DockingViewerBase):
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
        # Immapp:
        #  immapp.run() python stub:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/immapp/immapp_cpp.pyi#L162
        #  immapp.run() nanobind impl: https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/bindings/pybind_immapp_cpp.cpp#L158
        #  immapp.run() CPP impl:      https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/immapp/runner.cpp#L233

        # Hello-Imgui:
        #  hello_imgui.run() python stub:     https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/hello_imgui.pyi#L3416
        #  hello_imgui.run() nanobind impl:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/hello_imgui/bindings/pybind_hello_imgui.cpp#L1761
        #  hello_imgui.run() CPP impl:        https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui.cpp#L227
        #  AbstractRunner::Run():             https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L124
        #  PreNewFrame call:                  https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1412
        #  SCOPED_RELEASE_GIL_ON_MAIN_THREAD: https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1443
        #  fnReloadFontsIfDpiScaleChanged:    https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L947

        # Start compute thread asap
        self.start_event: threading.Event = threading.Event()
        self.stop_event: threading.Event = threading.Event()
        compute_thread = threading.Thread(target=self.compute_loop, args=[], daemon=True)
        compute_thread.start()

        # Installed by pip, includes two fonts
        hello_imgui.set_assets_folder(demos_assets_folder())

        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = name
        runner_params.app_window_params.window_geometry.size = (1000, 900)
        runner_params.app_window_params.restore_previous_geometry = True
        runner_params.dpi_aware_params.only_use_font_dpi_responsive = True # automatically handle font scaling
        runner_params.fps_idling.fps_idle = 5.0

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self.pan_handler = PannableArea(force_mouse_capture=True)
        self._ui_scale = 1.0
        self.ui_locked = True # resize in progress?
        self.first_frame = True # e.g. imgui.set_scroll_size wonky on first frame

        # Main image (output of self.compute())
        self.image: np.ndarray = None
        self.img_dt: float = 0
        self.img_shape: list = [3, 4, 4] # CHW (to match ToolbarViewer)
        self.last_upload_dt: float = 0
        self.tex_handle: _texture = None # created after GL init
        self.state = EasyDict()
        self.tex_upload_ms = 0.0 # gl textture upload stats

        self.initial_font_size = 15
        self.fonts: list[hello_imgui.FontDpiResponsive] = []
        self.window: glfw._GLFWwindow = None
        self._window_title = name
        self._orig_window_title = name
        self.load_font_awesome = with_font_awesome

        # For limiting OpenGL operations to UI thread
        self.ui_tid = threading.get_native_id() # main thread

        # Check if HDR mode has been turned on
        from pyviewer import _macos_hdr_patch
        self.hdr = (_macos_hdr_patch.CUR_MODE == _macos_hdr_patch.Mode.PATCHED)

        # Normalize images before showing?
        self.normalize = normalize if not self.hdr else False

        def load_settings_cbk():
            try:
                self.ui_scale = float(hello_imgui.load_user_pref('ui_scale'))
            except:
                pass
            self.load_settings()

        def post_init_fun():
            #glfw.make_context_current(self.window)
            #glfw.swap_interval(1)  # Enable vsync
            self.tex_handle = _texture(gl.GL_NEAREST, gl.GL_NEAREST)
            load_settings_cbk()
            self.setup_state()
            self.start_event.set()

        def save_settings_cbk():
            hello_imgui.save_user_pref("ui_scale", f'{self.ui_scale}')
            self.save_settings()

        def before_exit():
            save_settings_cbk()
            self.stop_event.set()
            del self.tex_handle

        def add_backend_cbk(*args, **kwargs):
            # Set own glfw callbacks, will be chained by imgui
            self.window = glfw_utils.glfw_window_hello_imgui() # why not glfw.get_current_context()?
            self.pan_handler.set_callbacks(self.window)

        def setup_theme_cbk():
            self.setup_theme() # user-overridable
            self.scale_style_sizes()
            self.pan_handler.clear_color = imgui.get_style().color_(imgui.Col_.window_bg) # match theme_deep_dark

        def after_swap():
            self.first_frame = False
            glfw.set_window_title(self.window, self._window_title) # from main thread

        runner_params.callbacks.post_init = post_init_fun
        runner_params.callbacks.before_exit = before_exit
        runner_params.callbacks.post_init_add_platform_backend_callbacks = add_backend_cbk
        runner_params.callbacks.load_additional_fonts = self.load_fonts
        runner_params.callbacks.setup_imgui_style = setup_theme_cbk
        runner_params.callbacks.pre_new_frame = self.pre_new_frame
        runner_params.callbacks.after_swap = after_swap

        self.show_app_menu = False
        self.show_view_menu = True
        runner_params.imgui_window_params.show_menu_bar = True
        runner_params.imgui_window_params.show_menu_app = False # called manually
        runner_params.imgui_window_params.show_menu_view = False # called manually
        runner_params.callbacks.show_menus = lambda: self._draw_menu_wrapper(runner_params)

        # Status bar: fps etc.
        runner_params.imgui_window_params.remember_status_bar_settings = False # don't cache
        if type(self).draw_status_bar != DockingViewer.draw_status_bar: # overridden
            runner_params.imgui_window_params.show_status_bar = True # off by default
            runner_params.callbacks.show_status = self.draw_status_bar

        # Create "MainDockSpace"
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )

        # Allow splitting into separate windows?
        runner_params.imgui_window_params.enable_viewports = True

        # Docking layout
        runner_params.docking_params = self.setup_layout()
        runner_params.docking_params.main_dock_space_node_flags |= imgui.DockNodeFlags_.auto_hide_tab_bar

        # .ini for window and app state saving
        runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        runner_params.ini_filename = name.lower().strip().replace(' ', '_') + '.ini'
        ini_path = Path(hello_imgui.ini_folder_location(runner_params.ini_folder_type)) / runner_params.ini_filename
        print(f'INI path: {ini_path}')

        glfw.init()
        if self.hdr:
            glfw.window_hint(glfw.RED_BITS, 16)
            glfw.window_hint(glfw.GREEN_BITS, 16)
            glfw.window_hint(glfw.BLUE_BITS, 16)

        addons = immapp.AddOnsParams(
            with_implot=with_implot,
            with_implot3d=with_implot3d,
            with_node_editor=with_node_editor,
            with_node_editor_config=with_node_editor_config,
            with_tex_inspect=with_tex_inspect,
        )

        super().__init__(runner_params, addons)

    def setup_layout(self) -> hello_imgui.DockingParams:
        """
        Create initial dummy docking layout.
        Ignores @layout specifiers, creates horizontal split.
        Only used initially, subsequent runs will reload previous layout.
        """
        # Find all functions with @layout decorator
        # Use list of candidate names '_dockable_windows' to avoid calling getattr on properties before user state init
        # In case of multiple instances and name collisions: must check attribute _layout_pos as well
        fun_names, titles = zip(*_dockable_windows)
        candidates = [getattr(self, name) for name in dir(self) if name in fun_names]
        layout_funcs = [f for f in candidates if hasattr(f, '_title')]
        assert len(titles) == len(set(titles)), 'Titles must be unique'

        # Creating layout from position tags seems non-trivial, ambiguous, and only affects first startup anyway.
        print('Using dummy horizontal layout')

        # N-1 splits
        dock_names = ['MainDockSpace'] + [f'Dock{i}' for i in range(len(layout_funcs)-1)]
        splits = [hello_imgui.DockingSplit(i, n, imgui.Dir.right) for i, n in zip(dock_names[:-1], dock_names[1:])]
        windows = [hello_imgui.DockableWindow(f._title, d, f, can_be_closed_=True) for f, d in zip(layout_funcs, dock_names)]

        return hello_imgui.DockingParams(splits, windows)

    def update_image(self, *, img_hwc=None):
        is_np = isinstance(img_hwc, np.ndarray)
        assert is_np or is_tensor(img_hwc), 'Expected np.ndarray or torch.Tensor'

        is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
        is_signed = (img_hwc.dtype.kind == 'i') if is_np else img_hwc.dtype.is_signed
        dtype_bits = img_hwc.dtype.itemsize * 8
        H, W, C = img_hwc.shape

        # RGBA texture uploads are much faster on some drivers
        if img_hwc.shape[-1] == 3:
            maxval = 1 if is_fp else 2**(dtype_bits - int(is_signed)) - 1
            if is_np:
                img_hwc = np.concatenate([img_hwc, maxval * np.ones((H, W, 1), dtype=img_hwc.dtype)], axis=-1)
            else:
                img_hwc = torch.cat([img_hwc, alpha_ch_torch(H, W, maxval, img_hwc.dtype, img_hwc.device)], dim=2)

        img_hwc = normalize_image_data(img_hwc, img_hwc.dtype) if self.normalize else img_hwc

        # Eventually uploaded by UI thread
        self.image = img_hwc # if is_np else img_hwc.cpu().numpy()
        self.img_shape = [self.image.shape[2], *self.image.shape[:2]] # shape in chw format
        self.img_dt = time.monotonic()

    @dockable(title='Main Output')
    def output(self):
        # Need to do upload from main thread
        if self.img_dt > self.last_upload_dt:
            gl.glFinish()
            t0 = time.monotonic()
            if isinstance(self.image, np.ndarray):
                self.tex_handle.upload_np(self.image)
            elif self.image.device.type == 'cuda':
                self.tex_handle.upload_torch(self.image)
            elif self.image.device.type == 'mps':
                self.tex_handle.upload_torch(self.image)
            else:
                self.tex_handle.upload_np(self.image.cpu().numpy())
            self.last_upload_dt = time.monotonic()
            self.tex_upload_ms = (self.last_upload_dt - t0) * 1000  # upload time stats

        # Reallocate if window size has changed
        if self.image is not None:
            tH, tW, _ = self.image.shape
            cW, cH = map(int, imgui.get_content_region_avail())
            canvas_tex = self.pan_handler.draw_to_canvas(self.tex_handle.tex, tW, tH, cW, cH, self.tex_handle.type)
            imgui.image(canvas_tex, (cW, cH))

        draw_list = imgui.get_window_draw_list()
        self.draw_overlay(draw_list)

    def compute_loop(self) -> None:
        print('Compute thread: waiting for start event')
        self.start_event.wait(timeout=None)
        print('Compute thread: start event received')

        while not self.stop_event.is_set():
            ret = self.compute()
            if ret is not None:
                self.update_image(img_hwc=ret)
            time.sleep(0.01)

        print('Compute thread: received stop event, exiting')


# ------------------------------------------------------------------------------------------------------------
# Multi texture docking viewer

class TextureWindowUnit:
    """A class to manage 'single' window with texture
    """

    def __init__(self, name: str):
        self.name = name

        # Setup PannableArea for this window
        self._pannable_area = PannableArea(force_mouse_capture=True)

        # Setup OpenGL texture, lazily created
        self._texture: _texture | None = None

        # Buffer on RAM
        self.img_shape: list = [3, 4, 4]  # CHW (to match ToolbarViewer)
        self.image: np.ndarray | None = None

        # Timings
        self.img_dt: float = 0.0
        self.last_upload_dt: float = 0
        self.tex_upload_ms: float = 0.0  # time taken to upload texture in ms

        # Dynamically create a new independent copy of the window function
        orig_func = self.__class__.__dict__['window']
        copied_func = types.FunctionType(
            orig_func.__code__,
            orig_func.__globals__,
            name=orig_func.__name__,
            argdefs=orig_func.__defaults__,
            closure=orig_func.__closure__
        )
        dockable_window = dockable(copied_func, title=name)

        # Bind the copied function to this instance
        setattr(self, 'window', dockable_window.__get__(self, self.__class__))

    def generate_texture(self) -> None:
        """Lazily create OpenGL texture.
        """

        if self._texture is None:
            self._texture = _texture(gl.GL_NEAREST, gl.GL_NEAREST)

    def set_callbacks(self, glfw_window: glfw._GLFWwindow) -> None:
        """Set callbacks for PannableArea. This must be called after glfw window is created.

        Parameters
        ----------
        glfw_window: glfw._GLFWwindow
            GLFW window to set callbacks.
        """

        self._pannable_area.set_callbacks(glfw_window)

    def set_theme(self) -> None:
        """Set theme for PannableArea.
        """

        self._pannable_area.clear_color = imgui.get_style().color_(imgui.Col_.window_bg)  # match theme_deep_dark

    def dispose_texture(self) -> None:
        """Dispose OpenGL texture. This should be called when the window is closed or the texture is no longer needed.
        """

        if self._texture is not None:
            del self._texture
            self._texture = None

    def upload_texture(self) -> None:
        """Upload image data to OpenGL texture.
        """

        if self._texture is None:
            print('No OpenGL texture to be uploaded to')
            return

        if self.img_dt > self.last_upload_dt:
            t0 = time.monotonic()

            if isinstance(self.image, np.ndarray):
                self._texture.upload_np(self.image)
            elif self.image.device.type == 'cuda':
                self._texture.upload_torch(self.image)
            elif self.image.device.type == 'mps':
                # https://github.com/prabu-ram/Custom_PyTorch-Operations/blob/main/compiler.py
                self._texture.upload_mps(self.image)
            else:
                self._texture.upload_np(self.image.cpu().numpy())

            self.last_upload_dt = time.monotonic()

            self.tex_upload_ms = (self.last_upload_dt - t0) * 1000  # upload time stats

    def window(self):
        """Dockable window for displaying a texture.
        """

        # First upload texture if it has been set
        self.upload_texture()

        # Begin the ImGui window
        imgui.begin(self.name, True, imgui.WindowFlags_.no_collapse)

        if self.image is not None:
            tH, tW, _ = self.image.shape
            cW, cH = map(int, imgui.get_content_region_avail())

            # Draw the texture to the canvas
            canvas_tex = self._pannable_area.draw_to_canvas(self._texture.tex, tW, tH, cW, cH)
            imgui.image(canvas_tex, (cW, cH))

        imgui.end()


class TextureWindows(dict[str, TextureWindowUnit]):
    """A container class to manage multiple windows
    """

    def __init__(self, names: list[str], normalize: bool):
        windows = {str(name): TextureWindowUnit(name) for name in names}
        self.normalize = normalize
        super().__init__(windows)

    def generate_texture(self) -> None:
        """Generate OpenGL textures for all texture windows.
        """

        for window in self.values():
            window.generate_texture()

    def dispose_texture(self) -> None:
        """Dispose OpenGL textures for all texture windows.
        """

        for window in self.values():
            window.dispose_texture()

    def set_callbacks(self, glfw_window: glfw._GLFWwindow) -> None:
        """Set GLFW callbacks for all texture windows.

        Parameters
        ----------
        glfw_window: glfw._GLFWwindow
            GLFW window to set callbacks for .
        """

        for window in self.values():
            window.set_callbacks(glfw_window)

    def set_theme(self) -> None:
        """Set theme for all texture windows' PannableArea.
        """

        for window in self.values():
            window.set_theme()

    def update_image(self, **kwargs):
        """Update textures based on window names
        """

        kwargs = dict(kwargs)

        for key, img_hwc in kwargs.items():
            if img_hwc is None:
                continue

            assert key in self, f'An image data was assigned to a non-existing texture window: "{key}"'

            is_np = isinstance(img_hwc, np.ndarray)
            assert is_np or is_tensor(img_hwc), 'Expected np.ndarray or torch.Tensor'

            is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
            is_signed = (img_hwc.dtype.kind == 'i') if is_np else img_hwc.dtype.is_signed
            dtype_bits = img_hwc.dtype.itemsize * 8
            H, W, C = img_hwc.shape

            # RGBA texture uploads are much faster on some drivers
            if img_hwc.shape[-1] == 3:
                maxval = 1 if is_fp else 2**(dtype_bits - int(is_signed))
                if is_np:
                    img_hwc = np.concatenate([img_hwc, maxval * np.ones((H, W, 1), dtype=img_hwc.dtype)], axis=-1)
                else:
                    img_hwc = torch.cat([img_hwc, alpha_ch_torch(H, W, maxval, img_hwc.dtype, img_hwc.device)], dim=2)

            img_hwc = normalize_image_data(img_hwc, img_hwc.dtype) if self.normalize else img_hwc

            # Eventually uploaded by UI thread
            self[key].image = img_hwc
            self[key].img_shape = [img_hwc.shape[2], *img_hwc.shape[:2]]  # shape in chw format
            self[key].img_dt = time.monotonic()


class MultiTexturesDockingViewer(DockingViewerBase):
    def __init__(
        self,
        name: str,
        texture_names: list[str],
        normalize=False,
        enable_vsync=False,
        full_screen_mode=hello_imgui.FullScreenMode.no_full_screen,
        with_implot=True,
        with_implot3d=False,
        with_node_editor=False,
        with_node_editor_config=None,
        with_tex_inspect=False,
        with_font_awesome=False,
    ):
        """Docking viewer with multiple texture windows.

        Usage
        -----
        ```python
        class DemoViewer(MultiTexturesDockingViewer):
            def __init__(self, name):
                super().__init__(name, texture_names=['Window 0', 'Window 1'])

            def compute(self):
                img0 = ...
                img1 = ...
                
                return {
                    'Window 0': img0,
                    'Window 1': img1
                }
        ```
        """

        # Immapp:
        #  immapp.run() python stub:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/immapp/immapp_cpp.pyi#L162
        #  immapp.run() nanobind impl: https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/bindings/pybind_immapp_cpp.cpp#L158
        #  immapp.run() CPP impl:      https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/immapp/runner.cpp#L233

        # Hello-Imgui:
        #  hello_imgui.run() python stub:     https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/hello_imgui.pyi#L3416
        #  hello_imgui.run() nanobind impl:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/hello_imgui/bindings/pybind_hello_imgui.cpp#L1761
        #  hello_imgui.run() CPP impl:        https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui.cpp#L227
        #  AbstractRunner::Run():             https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L124
        #  PreNewFrame call:                  https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1412
        #  SCOPED_RELEASE_GIL_ON_MAIN_THREAD: https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1443
        #  fnReloadFontsIfDpiScaleChanged:    https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L947

        # Start compute thread asap
        self.start_event: threading.Event = threading.Event()
        self.stop_event: threading.Event = threading.Event()
        compute_thread = threading.Thread(target=self.compute_loop, args=[], daemon=True)
        compute_thread.start()

        # Installed by pip, includes two fonts
        hello_imgui.set_assets_folder(demos_assets_folder())

        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = name
        runner_params.app_window_params.window_geometry.size = (1000, 900)
        runner_params.app_window_params.window_geometry.full_screen_mode = full_screen_mode
        runner_params.app_window_params.restore_previous_geometry = True
        runner_params.dpi_aware_params.only_use_font_dpi_responsive = True  # automatically handle font scaling
        runner_params.fps_idling.fps_idle = 5.0

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self._ui_scale = 1.0
        self.ui_locked = True # resize in progress?
        self.first_frame = True # e.g. imgui.set_scroll_size wonky on first frame

        # Main image (output of self.compute())
        self.state = EasyDict()

        self.initial_font_size = 15
        self.fonts: list[hello_imgui.FontDpiResponsive] = []
        self.window: glfw._GLFWwindow = None
        self._window_title = name
        self._orig_window_title = name
        self.load_font_awesome = with_font_awesome
        self.enable_vsync = enable_vsync

        # For limiting OpenGL operations to UI thread
        self.ui_tid = threading.get_native_id() # main thread

        # Check if HDR mode has been turned on
        from pyviewer import _macos_hdr_patch
        self.hdr = (_macos_hdr_patch.CUR_MODE == _macos_hdr_patch.Mode.PATCHED)

        # Normalize images before showing?
        normalize = normalize if not self.hdr else False

        # Create texture windows
        self.texture_windows = TextureWindows(texture_names, normalize)  # for each dockable window

        def load_settings_cbk():
            try:
                self.ui_scale = float(hello_imgui.load_user_pref('ui_scale'))
            except:
                pass
            self.load_settings()

        def post_init_fun():
            glfw.make_context_current(self.window)

            if self.enable_vsync:
                glfw.swap_interval(1)  # Enable vsync

            # Create OpenGL textures
            self.texture_windows.generate_texture()

            # Setup
            load_settings_cbk()
            self.setup_state()
            self.start_event.set()

        def save_settings_cbk():
            hello_imgui.save_user_pref("ui_scale", f'{self.ui_scale}')
            self.save_settings()

        def before_exit():
            save_settings_cbk()
            self.stop_event.set()

            # Dispose all the OpenGL textures
            self.texture_windows.dispose_texture()

        def add_backend_cbk(*args, **kwargs):
            # Set own glfw callbacks, will be chained by imgui
            self.window = glfw_utils.glfw_window_hello_imgui()  # why not glfw.get_current_context()?
            self.texture_windows.set_callbacks(self.window)  # Set callbacks for PannableArea

            # Set callback for drop event
            def noop(*args, **kwargs):
                return False
            prev = glfw.set_drop_callback(self.window, None)
            glfw.set_drop_callback(self.window, functools.partial(file_drop_callback_wrapper, callback=self.drag_and_drop_callback, prev=(prev or noop)))

        def setup_theme_cbk():
            self.setup_theme()  # user-overridable
            self.scale_style_sizes()

            # Set theme for PannableArea
            self.texture_windows.set_theme()

        def after_swap():
            self.first_frame = False
            glfw.set_window_title(self.window, self._window_title)  # from main thread

        runner_params.callbacks.post_init = post_init_fun
        runner_params.callbacks.before_exit = before_exit
        runner_params.callbacks.post_init_add_platform_backend_callbacks = add_backend_cbk
        runner_params.callbacks.load_additional_fonts = self.load_fonts
        runner_params.callbacks.setup_imgui_style = setup_theme_cbk
        runner_params.callbacks.pre_new_frame = self.pre_new_frame
        runner_params.callbacks.after_swap = after_swap

        self.show_app_menu = False
        self.show_view_menu = True
        runner_params.imgui_window_params.show_menu_bar = True
        runner_params.imgui_window_params.show_menu_app = False  # called manually
        runner_params.imgui_window_params.show_menu_view = False  # called manually
        runner_params.callbacks.show_menus = lambda: self._draw_menu_wrapper(runner_params)

        # Status bar: fps etc.
        runner_params.imgui_window_params.remember_status_bar_settings = False  # don't cache
        if type(self).draw_status_bar != MultiTexturesDockingViewer.draw_status_bar:  # overridden
            runner_params.imgui_window_params.show_status_bar = True  # off by default
            runner_params.callbacks.show_status = self.draw_status_bar

        # Create "MainDockSpace"
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )

        # Allow splitting into separate windows?
        runner_params.imgui_window_params.enable_viewports = True

        # Docking layout
        runner_params.docking_params = self.setup_layout()
        runner_params.docking_params.main_dock_space_node_flags |= imgui.DockNodeFlags_.auto_hide_tab_bar

        # .ini for window and app state saving'
        runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        runner_params.ini_filename = name.lower().strip().replace(' ', '_') + '.ini'
        ini_path = Path(hello_imgui.ini_folder_location(runner_params.ini_folder_type)) / runner_params.ini_filename
        print(f'INI path: {ini_path}')

        glfw.init()

        if self.hdr:
            glfw.window_hint(glfw.RED_BITS, 16)
            glfw.window_hint(glfw.GREEN_BITS, 16)
            glfw.window_hint(glfw.BLUE_BITS, 16)

        addons = immapp.AddOnsParams(
            with_implot=with_implot,
            with_implot3d=with_implot3d,
            with_node_editor=with_node_editor,
            with_node_editor_config=with_node_editor_config,
            with_tex_inspect=with_tex_inspect,
        )

        super().__init__(runner_params, addons)

    def setup_layout(self) -> hello_imgui.DockingParams:
        """
        Create initial dummy docking layout.
        Ignores @layout specifiers, creates horizontal split.
        Only used initially, subsequent runs will reload previous layout.
        """

        # Find all functions with @layout decorator
        # Use list of candidate names '_dockable_windows' to avoid calling getattr on properties before user state init
        # In case of multiple instances and name collisions: must check attribute _layout_pos as well
        fun_names, titles = zip(*_dockable_windows)
        candidates = [getattr(self, name) for name in dir(self) if name in fun_names]

        # For texture windows
        for _, texture_window in self.texture_windows.items():
            for name in dir(texture_window):
                if name in fun_names:
                    candidates.append(getattr(texture_window, name))

        layout_funcs = [f for f in candidates if hasattr(f, '_title')]
        assert len(titles) == len(set(titles)), 'Titles must be unique'

        splits, windows = self.setup_docking_layout(layout_funcs)

        return hello_imgui.DockingParams(splits, windows)
    
    def compute_loop(self):
        print('Compute thread: waiting for start event')
        self.start_event.wait(timeout=None)
        print('Compute thread: start event received')

        while not self.stop_event.is_set():
            ret = self.compute()

            if ret is not None:
                self.texture_windows.update_image(**ret)

            time.sleep(0.01)

        print('Compute thread: received stop event, exiting')

    # --------------------------------------------------------------------------------------------
    # User provided callbacks

    # Can be overridden
    def drag_and_drop_callback(self, paths: list[Path]) -> bool:
        """Judge whether to accept the drad and drop event
        """
        return False

    # Can be overridden
    def setup_docking_layout(self, layout_funcs: list[typing.Callable[[None], None]]) -> tuple[list[hello_imgui.DockingSplit], list[hello_imgui.DockableWindow]]:
        """Setup an initial docking layout. This function can be overridden in your child class. As a default, all dockable windows are stacked horizontally with the same width.

        Parameters
        ----------
        layout_funcs : list[typing.Callable[[None], None]]
            Window functions defined with `@dockable` decorator

        Returns
        -------
        tuple[list[hello_imgui.DockingSplit], list[hello_imgui.DockableWindow]]
            A list of DockingSplit and a list of DockableWindow
        """

        dock_space_names = ['MainDockSpace'] + [f'Dock{i}' for i in range(len(layout_funcs) - 1)] # Pseudo names for dock spaces. 'MainDockSpace' must be included.
        n_windows = len(layout_funcs) # has equal length with dock_space_names

        # NOTE: Should be splitted like: (N-1) / N, (N-2) / (N-1), (N-3) / (N-2), ..., 3 / 4, 2 / 3, 1 / 2
        splits = [hello_imgui.DockingSplit(prev, cur, imgui.Dir.right, ratio_=(n_windows - i_dock - 1) / (n_windows - i_dock)) for i_dock, (prev, cur) in enumerate(zip(dock_space_names[:-1], dock_space_names[1:]))]
        windows = [hello_imgui.DockableWindow(f._title, d, f, can_be_closed_=True) for f, d in zip(layout_funcs, dock_space_names)]

        return splits, windows
