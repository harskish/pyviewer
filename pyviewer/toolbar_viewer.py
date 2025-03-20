from imgui_bundle import imgui
import glfw
import threading
import random
import string
import numpy as np
import time
from typing import Union
from pathlib import Path
from functools import partial
from threading import get_native_id

from . import gl_viewer
from .utils import imgui_item_width, begin_inline, PannableArea
from .easy_dict import EasyDict
from .params import ParamContainer, Param, draw_container

#----------------------------------------------------------------------------
# Helper class for UIs with toolbar on the left and output image on the right

def file_drop_callback_wrapper(window, paths, callback: callable, prev: callable):
    return callback([Path(p) for p in paths]) or prev(window, paths)

def scroll_callback_wrapper(window, xoffset, yoffset, callback: callable, prev: callable):
    return callback(xoffset, yoffset) or prev(window, xoffset, yoffset)

class ToolbarViewer:
    def __init__(self, name, pad_bottom=0, hidden=False, batch_mode=False):
        self.output_key = ''.join(random.choices(string.ascii_letters, k=20))
        self._user_pad_bottom = pad_bottom
        self.v = gl_viewer.viewer(name, hidden=hidden or batch_mode, swap_interval=1)
        self.menu_bar_height = self.v.font_size + 2*imgui.get_style().frame_padding.y
        
        # Window title can be used to show progress etc.
        self._orig_window_title = name
        self._window_title = name # updated from compute/UI thread, applied in render loop

        W, H = glfw.get_window_size(self.v._window)

        # Size of latest image data in texels, (C, H, W)
        self.img_shape = [3, 4, 4]
        # Position of drawn imgui.image element,
        # relative to glfw window top-left.
        # Initial value not exactly correct but good initial guess
        px, py = np.array(imgui.get_style().window_padding)
        
        # Output_pos is really "canvas_pos"
        self.output_pos_tl = np.array([self.toolbar_width + px, self.menu_bar_height + py], dtype=np.float32)
        self.output_pos_br = np.array([W - px, H - self.pad_bottom - py], dtype=np.float32)
        
        # Size in pixels of drawn imgui.image, (W, H)
        self.content_size_px = (1, 1) # size of centered content (image)
        self.ui_locked = True
        self.state = EasyDict()

        # Support image zoom and pan
        self.pan_handler = PannableArea()
        self.pan_handler.clear_color = (0.101, 0.101, 0.101, 1.00) # match theme_deep_dark
        self.pan_enabled = True
        
        # User nearest interpolation for sharpness by default
        self.v.set_interp_nearest()

        # For limiting OpenGL operations to UI thread
        self.ui_tid = get_native_id() # main thread

        # User-provided
        self.setup_state()
        
        # Batch mode: handle compute loop manually, don't start UI
        if not batch_mode:
            self.start_UI()
    
    def start_UI(self):
        compute_thread = threading.Thread(target=self._compute_loop, args=[])
        def init_callbacks(window):
            def noop(*args, **kwargs):
                return False
            prev = glfw.set_drop_callback(window, None)
            glfw.set_drop_callback(window,
                partial(file_drop_callback_wrapper, callback=self.drag_and_drop_callback, prev=(prev or noop)))
            prev = glfw.set_scroll_callback(window, None)
            glfw.set_scroll_callback(window,
                partial(scroll_callback_wrapper, callback=self.scroll_callback, prev=(prev or noop)))
            self.setup_callbacks(window)
            self.pan_handler.set_callbacks(window)
        self.v.start(self._ui_main, (compute_thread), init_callbacks)

    def cleanup(self):
        """Manually cleanup OpenGL resources (only needed in batch mode)"""
        self.v.gl_shutdown()

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
        if get_native_id() == self.ui_tid:
            # need glfw.make_context_current if calling from compute thread?
            glfw.set_window_title(self.v._window, title)

    # Extra user content below image
    @property
    def pad_bottom(self):
        return max(self._user_pad_bottom, int(round(24 * self.v.ui_scale)) + 6)

    @property
    def toolbar_width(self):
        return int(round(350 * self.v.ui_scale))

    @property
    def font_size(self):
        return self.v.font_size

    @property
    def ui_scale(self):
        return self.v.ui_scale

    @property
    def content_size(self):
        return np.array(self.content_size_px)

    @property
    def mouse_pos_abs(self):
        return np.array(imgui.get_mouse_pos())

    @property
    def mouse_pos_content_norm(self):
        return (self.mouse_pos_abs - self.output_pos_tl) / self.content_size
    
    @property
    def mouse_pos_img_norm(self):
        dims = self.output_pos_br - self.output_pos_tl
        if any(dims == 0):
            return np.array([-1, -1], dtype=np.float32) # no valid content
        return (self.mouse_pos_abs - self.output_pos_tl) / dims

    def _ui_main(self, v):
        # Update window title
        
        #curr = glfw.get_window_title(v._window) # imgui_bundle._glfw must contain symbol
        #if curr != self._window_title:
        glfw.set_window_title(self.v._window, self._window_title)
        
        self._toolbar_wrapper()
        self._draw_output()

    def _compute_loop(self):
        self.compute_thread_init()
        while not self.v.quit:
            img = self.compute()
            
            if img is not None:
                H, W, C = img.shape
                self.img_shape = [C, H, W]
                self.v.upload_image(self.output_key, img)
            else:
                time.sleep(1/60)
        self.shutdown()

    def _draw_output(self):
        v = self.v

        BOTTOM_PAD = self.pad_bottom
        W, H = glfw.get_window_size(v._window)
        imgui.set_next_window_size((W - self.toolbar_width, H - self.menu_bar_height - BOTTOM_PAD))
        imgui.set_next_window_pos((self.toolbar_width, self.menu_bar_height))

        s = v.ui_scale

        # Texture handle to main image
        # If image is missing: assume fullscreen plot, don't block mouse events
        tex_in = v._images.get(self.output_key)
        mouse_events = tex_in is None

        begin_inline('Output', inputs=mouse_events)
        
        # Calculate size of current (virtual) imgui.window
        cW, cH = map(int, imgui.get_content_region_avail())
        #rmin, rmax = imgui.get_window_content_region_min(), imgui.get_window_content_region_max()
        #cW, cH = [int(r-l) for l,r in zip(rmin, rmax)]
        
        # Compute size of image that fills smaller dimension
        # Bottom area for integer scaling buttons taken into account
        aspect = self.img_shape[2] / self.img_shape[1]
        out_width = min(cW, aspect*cH)
        self.content_size_px = (int(out_width), int(out_width / aspect)) # size of centered content (image)

        # Draw UI elements before image
        # TODO: sizes won't match if using together with image (not meant to be used simultaneously!)
        # TODO: not getting inputs currently
        self.draw_pre()
        
        # Create imgui.image from provided data
        if tex_in is not None:
            canvas_size = (cW, cH)
            tex = self.pan_handler.draw_to_canvas(tex_in.tex, tex_in.shape[1], tex_in.shape[0], *canvas_size, self.pan_enabled)
            imgui.image(tex, canvas_size) # uv0?

        # Imgui.image, i.e. (pannable) canvas, drawn above
        # => get position where it was drawn
        # => one dimension matches image content, centered along the other
        self.output_pos_tl[:] = imgui.get_item_rect_min()
        self.output_pos_br[:] = imgui.get_item_rect_max()
        
        # Size check: tested 15.1.2025, matches
        #dims = self.output_pos_br - self.output_pos_tl
        #if all(dims > 0) and tuple(dims) != (cW, cH):
        #    print('output_pos_{tl,br} mismatch')
        
        #imgui.get_window_draw_list().add_circle(*self.output_pos_tl, 3.0, imgui.get_color_u32_rgba(1,0,1,1), thickness=2) # Tested 15.1.2025, position matches
        #imgui.get_window_draw_list().add_circle(*self.output_pos_br, 3.0, imgui.get_color_u32_rgba(0,1,1,1), thickness=2) # Tested 15.1.2025, position matches

        self.draw_overlays(imgui.get_window_draw_list())

        imgui.end()

        # New window with inputs for bottom elements
        imgui.set_next_window_size((W - self.toolbar_width, BOTTOM_PAD))
        imgui.set_next_window_pos((self.toolbar_width, H - BOTTOM_PAD))
        begin_inline('Output below')
        
        # Equal spacing
        imgui.columns(2, 'outputBottom', borders=False)

        # Extra UI elements below output
        self.draw_output_extra()
        imgui.next_column()

        # Scaling buttons
        #button_region_width = imgui.get_content_region_available_width()
        button_region_width = imgui.get_content_region_avail()[0]

        sizes = ['0.5', '1', '2', '3', '4', '6']
        button_W = 40 * s
        pad_left = max(0, button_region_width - (button_W * len(sizes)))

        for i, s in enumerate(sizes):
            imgui.same_line(offset_from_start_x=pad_left+i*button_W)
            if imgui.button(f'{s}x', (button_W-3, 0)): # tiny pad
                if not self.pan_enabled:
                    # Resize window
                    resW = int(self.img_shape[2] * float(s))
                    resH = int(self.img_shape[1] * float(s))
                    glfw.set_window_size(v._window,
                        width=resW+W-cW, height=resH+H-cH+BOTTOM_PAD) # TODO probably wrong
                elif tex_in is not None: # do nothing if no image (fullscreen plot etc.)
                    self.pan_handler.set_output_scale(s)

        imgui.columns(1)

        imgui.end()

    def _toolbar_wrapper(self):
        if imgui.begin_main_menu_bar():
            self.menu_bar_height = imgui.get_window_height()

            # Right-aligned button for locking / unlocking UI
            T = 'L' if self.ui_locked else 'U'
            C = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
            s = self.v.ui_scale

            # Custom user menu items
            self.draw_menu()

            # same_line() and negative sizes don't work within menu bar
            # => use invisible button instead
            # Tested: 16.3.2025, imgui_bundle==1.6.2
            max_x = imgui.get_window_width()
            cursor = imgui.get_cursor_pos()[0]

            # UI scale slider
            if not self.ui_locked:
                pad = max_x - cursor - 300 - 30*s
                imgui.invisible_button('##hidden', size=(pad, 1))
                with imgui_item_width(300): # size not dependent on s => prevents slider drift
                    max_scale = max(self.v._imgui_fonts.keys()) / self.v.default_font_size
                    min_scale = min(self.v._imgui_fonts.keys()) / self.v.default_font_size
                    ch, val = imgui.slider_float('', s, min_scale, max_scale)
                if ch:
                    self.v.set_ui_scale(val)
            else:
                pad = max_x - cursor - 25*s
                imgui.invisible_button('##hidden', size=(pad, 1))
            
            imgui.push_style_color(imgui.Col_.text, (*C, 1))
            if imgui.button(T, size=(20*s, 0)):
                self.ui_locked = not self.ui_locked
            imgui.pop_style_color()
            imgui.end_main_menu_bar()
        
        # Constant width, height dynamic based on output window
        v = self.v
        _, H = glfw.get_window_size(v._window)
        imgui.set_next_window_size((self.toolbar_width, H - self.menu_bar_height))
        imgui.set_next_window_pos((0, self.menu_bar_height))

        # User callback
        begin_inline('toolbar')
        self._draw_toolbar_impl()
        imgui.end()

    def _draw_toolbar_impl(self):
        self.draw_toolbar_autoUI()
        self.draw_toolbar()

    # Only used by AutoUIViewer
    def draw_toolbar_autoUI(self, containers=None):
        pass

    def mouse_over_image(self):
        x, y = self.mouse_pos_img_norm
        return (0 <= x <= 1) and (0 <= y <= 1)

    def mouse_over_content(self):
        x, y = self.mouse_pos_content_norm
        return (0 <= x <= 1) and (0 <= y <= 1)

    # For updating image elsewhere than when returning from compute()
    # (e.g. in callbacks or from UI thread)
    def update_image(self, img_hwc):
        H, W = img_hwc.shape[0:2]
        C = 1 if img_hwc.ndim == 2 else img_hwc.shape[-1]
        self.img_shape = [C, H, W]
        self.v.upload_image(self.output_key, img_hwc)
    
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
    #------------------------
    # User-provided functions

    # Draw toolbar, must be implemented
    def draw_toolbar(self):
        pass

    # Draw extra UI elements below output image
    def draw_output_extra(self):
        if self._user_pad_bottom > 0:
            raise RuntimeError('Not implemented')
    
    # Draw UI elements before main image is drawn
    # Can be (ab)used to draw only plots by never
    # returning an image form compute()
    def draw_pre(self):
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

#----------------------------------------------
# Version that creates UI widgets automatically

class AutoUIViewer(ToolbarViewer):
    def draw_toolbar_autoUI(self, containers=None):
        if containers is None:
            containers = [self.state]
        
        for cont in containers:
            if not isinstance(cont, ParamContainer):
                continue
            draw_container(cont, reset_button=True)

#--------------
# Example usage

def main():
    import numpy as np
    
    class Test(ToolbarViewer):
        def setup_state(self):
            self.state.seed = 0
        
        def compute(self):
            rand = np.random.RandomState(seed=self.state.seed)
            img = rand.randn(256, 256, 3).astype(np.float32)
            return np.clip(img, 0, 1) # HWC
        
        def draw_toolbar(self):
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]

    _ = Test('test_viewer')
    print('Done')

if __name__ == '__main__':
    main()
