import imgui
import glfw
import threading
import random
import string
import numpy as np
import time

from . import gl_viewer
from .utils import imgui_item_width, begin_inline, PannableArea
from .easy_dict import EasyDict
from .params import ParamContainer, Param

#----------------------------------------------------------------------------
# Helper class for UIs with toolbar on the left and output image on the right

class ToolbarViewer:
    def __init__(self, name, pad_bottom=0, hidden=False, batch_mode=False):
        self.output_key = ''.join(random.choices(string.ascii_letters, k=20))
        self._user_pad_bottom = pad_bottom
        self.v = gl_viewer.viewer(name, hidden=hidden or batch_mode, swap_interval=1)
        self.menu_bar_height = self.v.font_size + 2*imgui.get_style().frame_padding.y

        W, H = glfw.get_window_size(self.v._window)

        # Size of latest image data in texels, (C, H, W)
        self.img_shape = [3, 4, 4]
        # Position of drawn imgui.image element,
        # relative to glfw window top-left.
        # Initial value not exactly correct but good initial guess
        px, py = np.array(imgui.get_style().window_padding)
        self.output_pos_tl = np.array([self.toolbar_width + px, self.menu_bar_height + py], dtype=np.float32)
        self.output_pos_br = np.array([W - px, H - self.pad_bottom - py], dtype=np.float32)
        # Size in pixels of drawn imgui.image, (W, H)
        self.content_size_px = (1, 1)
        self.ui_locked = True
        self.state = EasyDict()

        # Support image zoom and pan
        self.pan_handler = PannableArea()
        self.pan_handler.clear_color = (0.101, 0.101, 0.101, 1.00) # match theme_deep_dark
        self.pan_enabled = True
        
        # User-provided
        self.setup_state()

        # User nearest interpolation for sharpness by default
        self.v.set_interp_nearest()
        
        # Batch mode: handle compute loop manually, don't start UI
        if not batch_mode:
            self.start_UI()

    def start_UI(self):
        compute_thread = threading.Thread(target=self._compute_loop, args=[])
        def init_callbacks(window):
            self.setup_callbacks(window)
            self.pan_handler.set_callbacks(window)
        self.v.start(self._ui_main, (compute_thread), init_callbacks)

    # Extra user content below image
    @property
    def pad_bottom(self):
        return max(self._user_pad_bottom, int(round(20 * self.v.ui_scale)) + 6)

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
        return (self.mouse_pos_abs - self.output_area_tl) / self.content_size
    
    @property
    def mouse_pos_img_norm(self):
        dims = self.output_pos_br - self.output_pos_tl
        if any(dims == 0):
            return np.array([-1, -1], dtype=np.float32) # no valid content
        return (self.mouse_pos_abs - self.output_pos_tl) / dims

    def _ui_main(self, v):
        self._toolbar_wrapper()
        self._draw_output()

    def _compute_loop(self):
        while not self.v.quit:
            img = self.compute()
            
            if img is not None:
                H, W, C = img.shape
                self.img_shape = [C, H, W]
                self.v.upload_image(self.output_key, img)
            else:
                time.sleep(1/60)

    def _draw_output(self):
        v = self.v

        W, H = glfw.get_window_size(v._window)
        imgui.set_next_window_size(W - self.toolbar_width, H - self.menu_bar_height)
        imgui.set_next_window_position(self.toolbar_width, self.menu_bar_height)

        s = v.ui_scale

        begin_inline('Output')
        BOTTOM_PAD = self.pad_bottom
        
        # Calculate size of current (virtual) imgui.window
        rmin, rmax = imgui.get_window_content_region_min(), imgui.get_window_content_region_max()
        cW, cH = [int(r-l) for l,r in zip(rmin, rmax)]
        
        # Compute size of image that fills smaller dimension
        # Bottom area for integer scaling buttons taken into account
        aspect = self.img_shape[2] / self.img_shape[1]
        out_width = min(cW, aspect*(cH - BOTTOM_PAD))
        self.content_size_px = (int(out_width), int(out_width / aspect))
        
        # Create imgui.image from provided data
        if self.pan_enabled:
            tex_in = v._images.get(self.output_key)
            if tex_in:
                canvas_size = (cW, cH - BOTTOM_PAD)
                tex = self.pan_handler.draw_to_canvas(tex_in.tex, *self.content_size_px, *canvas_size)
                imgui.image(tex, *canvas_size)
        else:
            v.draw_image(self.output_key, width=out_width)

        # Imgui.image drawn above
        # => get position where it was drawn
        self.output_pos_tl[:] = imgui.get_item_rect_min()
        self.output_pos_br[:] = imgui.get_item_rect_max()

        # Equal spacing
        imgui.columns(2, 'outputBottom', border=False)

        # Extra UI elements below output
        self.draw_output_extra()
        imgui.next_column()

        # Scaling buttons, right-aligned within child
        imgui.begin_child('sizeButtons', width=0, height=0, border=False)
        child_w = imgui.get_content_region_available_width()

        sizes = ['0.5', '1', '2', '3', '4']    
        button_W = 40 * s
        pad_left = max(0, child_w - (button_W * len(sizes)))

        for i, s in enumerate(sizes):
            imgui.same_line(position=pad_left+i*button_W)
            if imgui.button(f'{s}x', width=button_W-4): # tiny pad
                resW = int(self.img_shape[2] * float(s))
                resH = int(self.img_shape[1] * float(s))
                glfw.set_window_size(v._window,
                    width=resW+W-cW, height=resH+H-cH+BOTTOM_PAD)
        imgui.end_child()

        imgui.columns(1)
        self.draw_overlays(imgui.get_window_draw_list())

        imgui.end()

    def _toolbar_wrapper(self):
        if imgui.begin_main_menu_bar():
            self.menu_bar_height = imgui.get_window_height()

            # Right-aligned button for locking / unlocking UI
            T = 'L' if self.ui_locked else 'U'
            C = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
            s = self.v.ui_scale

            imgui.text('') # needed for imgui.same_line

            # Custom user menu items
            self.draw_menu()

            # UI scale slider
            if not self.ui_locked:
                imgui.same_line(position=imgui.get_window_width()-300-25*s)
                with imgui_item_width(300): # size not dependent on s => prevents slider drift
                    max_scale = max(self.v._imgui_fonts.keys()) / self.v.default_font_size
                    min_scale = min(self.v._imgui_fonts.keys()) / self.v.default_font_size
                    ch, val = imgui.slider_float('', s, min_scale, max_scale)
                if ch:
                    self.v.set_ui_scale(val)

            imgui.same_line(position=imgui.get_window_width()-25*s)
            imgui.push_style_color(imgui.COLOR_TEXT, *C)
            if imgui.button(T, width=20*s):
                self.ui_locked = not self.ui_locked
            imgui.pop_style_color()
            imgui.end_main_menu_bar()

        
        # Constant width, height dynamic based on output window
        v = self.v
        _, H = glfw.get_window_size(v._window)
        imgui.set_next_window_size(self.toolbar_width, H - self.menu_bar_height)
        imgui.set_next_window_position(0, self.menu_bar_height)

        # User callback
        begin_inline('toolbar')
        self.draw_toolbar_autoUI()
        self.draw_toolbar()
        imgui.end()

    # Only used by AutoUIViewer
    def draw_toolbar_autoUI(self):
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
    
    #------------------------
    # User-provided functions

    # Draw toolbar, must be implemented
    def draw_toolbar(self):
        pass

    # Draw extra UI elements below output image
    def draw_output_extra(self):
        if self._user_pad_bottom > 0:
            raise RuntimeError('Not implemented')

    # Draw overlays using main window draw list
    def draw_overlays(self, draw_list):
        pass

    def draw_menu(self):
        pass
    
    # Perform computation, returning single np/torch image, or None
    def compute(self):
        pass

    # Program state init
    def setup_state(self):
        pass

    # GLFW callbacks
    def setup_callbacks(self, window):
        pass


#----------------------------------------------
# Version that creates UI widgets automatically

class AutoUIViewer(ToolbarViewer):
    def draw_toolbar_autoUI(self):
        if not isinstance(self.state, ParamContainer):
            return

        for _, p in self.state:
            if isinstance(p, Param):
                p.draw()
        
        # Draw below widgets
        if imgui.button('Reset'):
            for _, p in self.state:
                if isinstance(p, Param):
                    p.reset()

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
