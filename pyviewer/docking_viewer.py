import sys
from pathlib import Path
import numpy as np
import time
import threading
import time
from typing import Union
import threading

# Some callbacks broken if imported before imgui_bundle...??
assert 'glfw' not in sys.modules or 'imgui_bundle' in sys.modules, 'glfw should be imported after pyviewer'

from imgui_bundle.demos_python.demo_utils import demos_assets_folder
from imgui_bundle import hello_imgui, glfw_utils, imgui, immapp # type: ignore
import glfw
import OpenGL.GL as gl

from .gl_viewer import _texture
from .toolbar_viewer import PannableArea
from .utils import normalize_image_data
from .imgui_themes import theme_deep_dark
from .easy_dict import EasyDict

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
        setattr(func, 'layout_pos', pos)
        return func
    
    return wrapper

def dockable(func):
    """
    Decorator indicating that generated UI elements
    should be placed in a dockable imgui window.
    """
    setattr(func, 'layout_pos', 'Dummy')
    return func

def file_drop_callback_wrapper(window, paths, callback: callable, prev: callable):
    return callback([Path(p) for p in paths]) or prev(window, paths)

def scroll_callback_wrapper(window, xoffset, yoffset, callback: callable, prev: callable):
    return callback(xoffset, yoffset) or prev(window, xoffset, yoffset)

class DockingViewer:
    #########################
    # User provided callbacks
    #########################

    # Draw toolbar, must be implemented
    def draw_toolbar(self):
        pass

    def draw_status_bar(self):
        pass

    # Draw overlays using main window draw list
    def draw_overlays(self, draw_list):
        pass

    def draw_menu(self):
        pass
    
    # One time compute thread initialization
    # Usually not needed
    def compute_thread_init(self):
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

    def load_settings(self):
        """Load settings using `hello_imgui.load_user_pref(k: str)`"""
        pass

    def save_settings(self):
        """Save settings using `hello_imgui.save_user_pref(k: str, v: str)`"""
        pass

    # Manual GLFW callbacks
    # Overrides the ones below
    def setup_callbacks(self, window):
        pass

    def drag_and_drop_callback(self, paths: list[Path]) -> bool:
        pass

    def scroll_callback(self, xoffset: float, yoffset: float) -> bool:
        pass

    # Cleanup before exit (Esc or window close)
    def shutdown(self):
        pass
    
    ################
    # Class internal
    ################
    
    # Includes keyboard (glfw.KEY_A) and mouse (glfw.MOUSE_BUTTON_LEFT)
    def keydown(self, key: Union[int, str]):
        """key: glfw keycode or str matching glfw keycode constant"""
        if isinstance(key, str):
            key = getattr(glfw, key.upper())
        return self.v.keydown(key)
    
    def keyhit(self, key: str):
        """key: glfw keycode or str matching glfw keycode constant"""
        if isinstance(key, str):
            key = getattr(glfw, key.upper())
        return self.v.keyhit(key)
        
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
    
    def set_ui_scale(self, scale: float):
        self.ui_scale = scale
        self.scale_style_sizes() #imgui.get_style().scale_all_sizes(delta)
        imgui.get_io().font_global_scale = scale * self.initial_font_scale # should reload fonts + rebuild font atlas for sharp results
    
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
        Good place to add new dockable windows, modify fonts, etc.
        """
        pass
    
    def __init__(
        self,
        name: str,
        with_markdown=False,
        with_markdown_options=None,
        with_implot=True,
        with_implot3d=False,
        with_node_editor=False,
        with_node_editor_config=None,
        with_tex_inspect=False,
    ):
        # DOCS/APIs:

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
        #runner_params.imgui_window_params.menu_app_title = "HLP"
        runner_params.app_window_params.window_geometry.size = (1000, 900)
        runner_params.app_window_params.restore_previous_geometry = True
        runner_params.dpi_aware_params.only_use_font_dpi_responsive = True # automatically handle font scaling

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self.pan_handler = PannableArea(force_mouse_capture=True)
        self.ui_scale = 1.0
        self.ui_locked = True # resize in progress?
        self.initial_font_scale = 1.0
        self.image: np.ndarray = None
        self.img_dt: float = 0
        self.last_upload_dt: float = 0
        self.tex_handle: _texture = None # created after GL init
        self.state = EasyDict()
        
        self.title_font: imgui.ImFont = None
        self.color_font: imgui.ImFont = None
        self.emoji_font: imgui.ImFont = None
        self.large_icon_font: imgui.ImFont = None
        self.window: glfw._GLFWwindow = None
        
        def load_settings_cbk():
            try:
                self.set_ui_scale(float(hello_imgui.load_user_pref('ui_scale')))
            except:
                pass
            self.load_settings()
        
        def post_init_fun():
            self.initial_font_scale = imgui.get_io().font_global_scale
            self.tex_handle = _texture(gl.GL_NEAREST, gl.GL_NEAREST)
            load_settings_cbk()
            self.setup_state()
            self.start_event.set()

        def before_exit():
            del self.tex_handle
            self.stop_event.set()
        
        def add_backend_cbk(*args, **kwargs):
            # Set own glfw callbacks, will be chained by imgui
            self.window = glfw_utils.glfw_window_hello_imgui() # why not glfw.get_current_context()?
            self.pan_handler.set_callbacks(self.window)

        def setup_theme_cbk():
            self.setup_theme() # user-overridable
            self.scale_style_sizes()
            self.pan_handler.clear_color = imgui.get_style().color_(imgui.Col_.window_bg) # match theme_deep_dark

        def save_settings_cbk():
            hello_imgui.save_user_pref("ui_scale", f'{self.ui_scale}')
            self.save_settings()

        runner_params.callbacks.post_init = post_init_fun
        runner_params.callbacks.before_exit = before_exit
        runner_params.callbacks.post_init_add_platform_backend_callbacks = add_backend_cbk
        runner_params.callbacks.load_additional_fonts = self.load_fonts
        runner_params.callbacks.setup_imgui_style = setup_theme_cbk
        runner_params.callbacks.before_exit = save_settings_cbk
        runner_params.callbacks.pre_new_frame = self.pre_new_frame

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

        glfw.init()  # needed by glfw_utils.glfw_window_hello_imgui
        addons = immapp.AddOnsParams(
            with_markdown=with_markdown,
            with_markdown_options=with_markdown_options,
            with_implot=with_implot,
            with_implot3d=with_implot3d,
            with_node_editor=with_node_editor,
            with_node_editor_config=with_node_editor_config,
            with_tex_inspect=with_tex_inspect,
        )
        immapp.run(runner_params, add_ons_params=addons)
    
    def setup_layout(self) -> hello_imgui.DockingParams:
        """
        Create initial dummy docking layout.
        Ignores @layout specifiers, creates horizontal split.
        Only used initially, subsequent runs will reload previous layout.
        """
        # Find all functions with @layout decorator
        all_funcs = [getattr(self, method_name) for method_name in dir(self) if callable(getattr(self, method_name))]
        layout_funcs = [func for func in all_funcs if hasattr(func, 'layout_pos')]

        # Creating layout from position tags seems non-trivial, ambiguous, and only affects first startup anyway.
        print('Using dummy horizontal layout')
        
        # N-1 splits
        dock_names = ['MainDockSpace'] + [f'Dock{i}' for i in range(len(layout_funcs)-1)]
        splits = [hello_imgui.DockingSplit(i, n, imgui.Dir.right) for i, n in zip(dock_names[:-1], dock_names[1:])]
        windows = [hello_imgui.DockableWindow(f.__name__, d, f, can_be_closed_=True) for f, d in zip(layout_funcs, dock_names)]

        return hello_imgui.DockingParams(splits, windows)
    
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
        layout_funcs = [func for func in all_funcs if hasattr(func, 'layout_pos')]

        # Create splits based on desired positions
        positions = set(f.layout_pos for f in layout_funcs)
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
    
    def load_fonts(self):
        hello_imgui.get_runner_params().callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
        hello_imgui.imgui_default_settings.load_default_font_with_font_awesome_icons()

        # Load the title font
        self.title_font = hello_imgui.load_font("fonts/DroidSans.ttf", 18.0)
        font_loading_params_title_icons = hello_imgui.FontLoadingParams()
        font_loading_params_title_icons.merge_to_last_font = True
        font_loading_params_title_icons.use_full_glyph_range = True
        self.title_font = hello_imgui.load_font(
            "fonts/Font_Awesome_6_Free-Solid-900.otf", 18.0, font_loading_params_title_icons)

        # Load the emoji font
        font_loading_params_emoji = hello_imgui.FontLoadingParams(use_full_glyph_range=True)
        self.emoji_font = hello_imgui.load_font(
            "fonts/NotoEmoji-Regular.ttf", 24.0, font_loading_params_emoji)

        # Load a large icon font
        font_loading_params_large_icon = hello_imgui.FontLoadingParams(use_full_glyph_range=True)
        self.large_icon_font = hello_imgui.load_font(
            "fonts/fontawesome-webfont.ttf", 24.0, font_loading_params_large_icon)

        # Load a colored font
        font_loading_params_color = hello_imgui.FontLoadingParams(load_color=True)
        self.color_font = hello_imgui.load_font(
            "fonts/Playbox/Playbox-FREE.otf", 24.0, font_loading_params_color)
    
    def update_image(self, arr):
        assert isinstance(arr, np.ndarray)
        
        # Eventually uploaded by UI thread
        self.image = normalize_image_data(arr, 'uint8')
        self.img_dt = time.monotonic()

    @dockable
    def draw_output(self):
        # Need to do upload from main thread
        if self.img_dt > self.last_upload_dt:
            self.tex_handle.upload_np(self.image)
            self.last_upload_dt = time.monotonic()
        
        # Reallocate if window size has changed
        if self.image is not None:
            tH, tW, _ = self.image.shape
            cW, cH = map(int, imgui.get_content_region_avail())
            canvas_tex = self.pan_handler.draw_to_canvas(self.tex_handle.tex, tW, tH, cW, cH)
            imgui.image(canvas_tex, (cW, cH))
    
    def compute_loop(self):
        print('Compute thread: waiting for start event')
        self.start_event.wait(timeout=None)
        print('Compute thread: start event received')

        while not self.stop_event.is_set():
            ret = self.compute()
            if ret is not None:
                self.update_image(ret)
            time.sleep(0.01)
        
        print('Compute thread: received stop event, exiting')

    def set_window_title(self, title):
        pass
