from pathlib import Path
from threading import Thread
import multiprocessing as mp
import light_process as lp
import numpy as np
import random
import string
import time
import glfw
import imgui
import ctypes
import sys
import warnings
import array
from enum import Enum

# Check imgui version
import imgui
implot = getattr(imgui, 'plot', None)
assert implot, \
    'Pyviewer requires a custom version of imgui that comes bundled with implot (github.com/harskish/pyimgui).\n' + \
    'Please reinstall pyviewer to get the correct version.'

has_torch = False
try:
    import torch
    has_torch = True
except:
    pass

from . import gl_viewer
from .utils import begin_inline, normalize_image_data, PannableArea

class ImgShape(ctypes.Structure):
    _fields_ = [('h', ctypes.c_uint), ('w', ctypes.c_uint), ('c', ctypes.c_uint)]
class WindowSize(ctypes.Structure):
    _fields_ = [('w', ctypes.c_uint), ('h', ctypes.c_uint)]

class VizMode(Enum):
    IMAGE = 0
    PLOT_LINE = 1
    PLOT_LINE_DOT = 2
    PLOT_DOT = 3

class SingleImageViewer:
    def __init__(self, title, key=None, dtype='uint8', vsync=True, hidden=False, pannable=True):
        self.title = title
        self.key = key or ''.join(random.choices(string.ascii_letters, k=100))
        self.ui_process = None
        self.vsync = vsync

        # Images are copied to minimize critical section time.
        # With uint8 (~4x faster copies than float32), this is
        # faster than waiting for OpenGL upload (for some reason...)
        self.dtype = dtype
        
        # Shared resources for inter-process communication
        # One shared 8k rgb buffer allocated (max size), subset written to
        # Size does not affect performance, only memory usage
        self.max_size_img = (2*3840, 2*2160, 3)
        ctype = ctypes.c_uint8 if self.dtype == 'uint8' else ctypes.c_float
        self.shared_buffer_img = mp.Array(ctype, np.prod(self.max_size_img).item())

        # Plotting mode: max size 1M floats per axis
        self.max_size_plot = 1_000_000
        self.shared_buffer_plot = mp.Array(ctypes.c_float, 2*self.max_size_plot)
        
        # Non-scalar type: not updated in single transaction
        # Protected by shared_buffer's lock
        self.latest_shape = mp.Value(ImgShape, *(0,0,0), lock=False)
        self.latest_plot_len = mp.Value(ctypes.c_uint, 0, lock=False) # single axis
        
        # Current window size, protected by lock
        self.curr_window_size = mp.Value(WindowSize, *(0,0))
        
        # Scalar values updated atomically
        self.should_quit = mp.Value('i', 0)
        self.has_new_img = mp.Value('i', 0)
        
        # Image or plot
        self.viz_mode = mp.Value('i', VizMode.IMAGE.value)
        
        # For hiding/showing window
        self.hidden = mp.Value(ctypes.c_bool, hidden, lock=False)

        # For enabling/disabling mouse pan and zoom
        self.pan_enabled = mp.Value(ctypes.c_bool, pannable, lock=False)

        # Pausing (via pause key on keyboard) speeds up computation
        self.paused = mp.Value(ctypes.c_bool, False, lock=False)
        
        # For waiting until process has started
        self.started = mp.Value(ctypes.c_bool, False, lock=False)

        self._start()

    # Called from main thread, waits until viewer is visible
    def wait_for_startup(self, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.started.value:
            time.sleep(1/10)

    # Called from main thread, loops until viewer window is closed
    def wait_for_close(self):
        while self.ui_process.is_alive():
            time.sleep(0.2)

    @property
    def window_size(self):
        with self.curr_window_size.get_lock():
            return (self.curr_window_size.w, self.curr_window_size.h)

    def hide(self):
        self.hidden.value = True

    def show(self, sync=False):
        self.hidden.value = False
        if sync:
            self.wait_for_startup()

    def _start(self):
        self.started.value = False
        self.ui_process = lp.LightProcess(target=self.process_func) # won't import __main__
        self.ui_process.start()

    def restart(self):
        if self.ui_process.is_alive():
            self.ui_process.join()
        self._start()

    def close(self):
        self.should_quit.value = 1
        self.ui_process.join()

    def process_func(self):
        # Avoid double cuda init issues
        # (single image viewer cannot use GPU memory anyway)
        gl_viewer.has_pycuda = False
        gl_viewer.cuda_synchronize = lambda : None
        
        v = gl_viewer.viewer(self.title, swap_interval=int(self.vsync), hidden=self.hidden.value)
        v._window_hidden = self.hidden.value
        v.set_interp_nearest()
        v.pan_handler = PannableArea()
        compute_thread = Thread(target=self.compute, args=[v])

        def set_glfw_callbacks(window):
            self.window_size_callback(window, *glfw.get_window_size(window)) # call once to set defaults
            glfw.set_window_close_callback(window, self.window_close_callback)
            glfw.set_window_size_callback(window, self.window_size_callback)
            v.pan_handler.set_callbacks(window)
        v.start(self.ui, [compute_thread], set_glfw_callbacks)

    def window_close_callback(self, window):
        self.started.value = False

    def window_size_callback(self, window, w, h):
        with self.curr_window_size.get_lock():
            self.curr_window_size.w = w
            self.curr_window_size.h = h

    # Called from main thread
    def draw(self, img_hwc=None, img_chw=None, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        # Activate image mode
        self.viz_mode.value = VizMode.IMAGE.value
        
        if not has_torch:
            assert isinstance(img_hwc, (type(None), np.ndarray)), 'PyTorch not available, only numpy arrays supported'
            assert isinstance(img_chw, (type(None), np.ndarray)), 'PyTorch not available, only numpy arrays supported'

        assert img_hwc is not None or img_chw is not None, 'Must provide img_hwc or img_chw'
        assert img_hwc is None or img_chw is None, 'Cannot provide both img_hwc and img_chw'

        if has_torch and torch.is_tensor(img_hwc):
            img_hwc = img_hwc.detach().cpu().numpy()

        if has_torch and torch.is_tensor(img_chw):
            img_chw = img_chw.detach().cpu().numpy()

        # Convert chw to hwc, if provided
        # If grayscale: no conversion needed
        if img_chw is not None:
            img_hwc = img_chw if img_chw.ndim == 2 else np.transpose(img_chw, (1, 2, 0))
            img_chw = None

        # Convert data to valid range
        img_hwc = normalize_image_data(img_hwc)

        sz = np.prod(img_hwc.shape)
        assert sz <= np.prod(self.max_size_img), f'Image too large, max size {self.max_size_img}'

        # Synchronize
        with self.shared_buffer_img.get_lock():
            arr_np = np.frombuffer(self.shared_buffer_img.get_obj(), dtype=self.dtype)
            arr_np[:sz] = img_hwc.reshape(-1)
            self.latest_shape.h = img_hwc.shape[0]
            self.latest_shape.w = img_hwc.shape[1]
            self.latest_shape.c = img_hwc.shape[2]
            self.has_new_img.value = 1

    # Called from main thread
    def plot(self, y, *, x=None, ignore_pause=False):
        # Paused or closed
        if (self.paused.value and not ignore_pause) or not self.ui_process.is_alive():
            return

        # Activate plotting mode
        if self.viz_mode.value == VizMode.IMAGE.value:
            self.viz_mode.value = VizMode.PLOT_LINE.value
        
        assert x is not None or y is not None, 'Must provide data for x or y axis'

        if has_torch and torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        
        if has_torch and torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        
        # Flatten, fill missing data with linspace
        x = x.reshape(-1) if x is not None else np.linspace(0, 1, len(y.reshape(-1)))
        y = y.reshape(-1) if y is not None else np.linspace(0, 1, len(x.reshape(-1)))
        assert len(x) == len(y), 'X and Y length differs'

        sz = np.prod(x.shape)
        assert sz <= self.max_size_plot, f'Too much data, max size {self.max_size_plot} per axis'

        # Synchronize
        with self.shared_buffer_plot.get_lock():
            arr_np = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32')
            arr_np[0:sz] = x.astype(np.float32)
            arr_np[sz:2*sz] = y.astype(np.float32)
            self.latest_plot_len.value = sz

    # Called in loop from ui thread
    def ui(self, v):
        if self.should_quit.value == 1:
            glfw.set_window_should_close(v._window, True)
            return
        
        # Visibility changed
        if self.hidden.value != v._window_hidden:
            v._window_hidden = self.hidden.value
            if v._window_hidden:
                glfw.hide_window(v._window)
            else:
                glfw.show_window(v._window)

        if v.keyhit(glfw.KEY_PAUSE):
            self.paused.value = not self.paused.value
        if v.keyhit(glfw.KEY_M):
            self.viz_mode.value = (self.viz_mode.value + 1) % len(VizMode) # loop through modes

        imgui.set_next_window_size(*glfw.get_window_size(v._window))
        imgui.set_next_window_position(0, 0)
        begin_inline('Output')
        
        viz_mode = VizMode(self.viz_mode.value)

        # Draw provided image
        if viz_mode == VizMode.IMAGE:
            v.pan_handler.zoom_enabled = True
            if self.pan_enabled.value:
                tex_in = v._images.get(self.key)
                if tex_in:
                    tex_H, tex_W, _ = tex_in.shape
                    cW, cH = [int(r-l) for l,r in zip(
                        imgui.get_window_content_region_min(), imgui.get_window_content_region_max())]
                    scale = min(cW / tex_W, cH / tex_H)
                    out_res = (int(tex_W*scale), int(tex_H*scale))
                    tex = v.pan_handler.draw_to_canvas(tex_in.tex, *out_res, cW, cH)
                    imgui.image(tex, cW, cH)
            else:
                v.draw_image(self.key, width='fit')
        
        # Draw plot
        else:
            v.pan_handler.zoom_enabled = False
            x = y = None
            with self.shared_buffer_plot.get_lock():
                sz = self.latest_plot_len.value
                data = np.frombuffer(self.shared_buffer_plot.get_obj(), dtype='float32', count=2*sz).copy()
                x = data[0:sz]
                y = data[sz:]
            W, H = glfw.get_window_size(v._window)
            style = imgui.get_style()
            avail_h = H - 2*style.window_padding.y
            avail_w = W - 2*style.window_padding.x
            implot.begin_plot('', size=(avail_w, avail_h))
            arr_x = array.array('f', bytearray(x))
            arr_y = array.array('f', bytearray(y))
            if viz_mode in [VizMode.PLOT_LINE, VizMode.PLOT_LINE_DOT]:
                implot.plot_line2('', arr_x, arr_y, len(x))
            if viz_mode in [VizMode.PLOT_DOT, VizMode.PLOT_LINE_DOT]:
                implot.set_next_marker_style(size=2)
                implot.plot_scatter2('', arr_x, arr_y, len(x))
            implot.end_plot()

        if self.paused.value:
            imgui.push_font(v._imgui_fonts[30])
            dl = imgui.get_window_draw_list()
            dl.add_rect_filled(5, 8, 115, 43, imgui.get_color_u32_rgba(0,0,0,1))
            dl.add_text(20, 10, imgui.get_color_u32_rgba(1,1,1,1), 'PAUSED')
            imgui.pop_font()
            time.sleep(1/20) # <= real speedup of pausing comes from here

        imgui.end()

    # Called in loop from compute thread
    # Image upload loop, not needed for plotting
    def compute(self, v):
        self.started.value = True
        while not v.quit:
            if self.has_new_img.value == 1:
                with self.shared_buffer_img.get_lock():
                    shape = (self.latest_shape.h, self.latest_shape.w, self.latest_shape.c)
                    img = np.frombuffer(self.shared_buffer_img.get_obj(), dtype=self.dtype, count=np.prod(shape)).copy()
                    self.has_new_img.value = 0

                img = img.reshape(shape)
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                v.upload_image_np(self.key, img)
            elif self.paused.value:
                time.sleep(1/10) # paused
            else:
                time.sleep(1/80) # idle

# Suppress warning due to LightProcess
if not sys.warnoptions:  # allow overriding with `-W` option
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy',
        message="'pyviewer.single_image_viewer' found in sys.modules.*")

# Single global instance
# Removes need to pass variable around in code
# Just call draw() (optionally call init first)
inst: SingleImageViewer = None

def init(*args, sync=True, **kwargs):
    global inst

    if inst is None:
        inst = SingleImageViewer(*args, **kwargs)
        if sync:
            inst.wait_for_startup() # if calling from debugger: need to give process time to start

# No-op if already open, therwise (re)start
def show_window():
    # Elif to avoid immediate restart if first init
    if inst is None:
        init('SIV')
    elif not inst.started.value:
        inst.restart()
        inst.paused.value = False

def draw(*, img_hwc=None, img_chw=None, ignore_pause=False):
    init('SIV') # no-op if init already performed
    inst.draw(img_hwc, img_chw, ignore_pause)

def plot(y, *, x=None, ignore_pause=False):
    init('SIV') # no-op if init already performed
    inst.plot(x=x, y=y, ignore_pause=ignore_pause)