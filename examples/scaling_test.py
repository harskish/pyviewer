from pathlib import Path
import numpy as np
import pyviewer
from imgui_bundle import imgui
from enum import Enum
from pyviewer.utils import combo_box_vals
from functools import lru_cache
from io import BytesIO
from PIL import Image
import socket
import glfw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
matplotlib.use('Agg')

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

@lru_cache
def _build_siemens_star_unscaled(origin=(0, 0), radius=1, n=100, DPI=600, width=1024, height=1024):
    centres = np.linspace(0, 360, n+1)[:-1]
    step = (((360.0)/n)/4.0)
    patches = []
    for c in centres:
        patches.append(Wedge(origin, radius, c-step, c+step))
    
    fig, ax = plt.subplots(dpi=DPI, figsize=(10, 10))
    ax.add_collection(PatchCollection(patches, facecolors='k', edgecolors='none'))
    ax.text(0.75, 0.95, 'N=%d' % n, fontsize=18)
    plt.axis('equal')
    plt.axis([-1.03, 1.03, -1.03, 1.03])
    plt.tight_layout()

    io_buf = BytesIO()
    fig.savefig(io_buf, format='raw', dpi=DPI)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print('Siemens start shape:', img_arr.shape)

    return img_arr

@lru_cache
def build_siemens_star(origin=(0, 0), radius=1, n=100, DPI=600, width=1024, height=1024):
    img_arr = _build_siemens_star_unscaled(origin, radius, n, DPI)
    img = Image.fromarray(img_arr).convert('RGB').resize((width, height), resample=Image.LANCZOS)
    return np.array(img)

#siv.init('Async viewer', hidden=False)

class PATTERNS(Enum):
    GRID = 'Grid'
    STAR = 'Siemens star'

# auto_res, width, height, window_w, window_h, ui_scale, zoom
conf_debu = {
    0: dict(auto_res=False, width=1024, height=1024, win_w=1623, win_h=1110, ui_scale=1.667, zoom=1.00000000, tx=0.0000000, ty=0.00000000),
    1: dict(auto_res=False, width=1212, height=1212, win_w=2590, win_h=1298, ui_scale=1.667, zoom=1.00000000, tx=0.0000000, ty=0.00000000), # even height, lower tri artifacts, depends on quad pos
    2: dict(auto_res=False, width=1212, height=1212, win_w=1811, win_h=1298, ui_scale=1.667, zoom=1.00000000, tx=0.0000000, ty=0.00000000), # even square, works fine
    3: dict(auto_res=False, width=1212, height=1212, win_w=1812, win_h=1299, ui_scale=1.667, zoom=0.99917561, tx=0.0000000, ty=0.00000000), # odd-even square, fine initially but breaks on pan
    4: dict(auto_res=False, width=1213, height=1213, win_w=1812, win_h=1299, ui_scale=1.667, zoom=1.00000000, tx=0.0000000, ty=0.00000000), # odd-odd square, works fine
    5: dict(auto_res=False, width=1222, height=1222, win_w=2059, win_h=1308, ui_scale=1.667, zoom=1.00000000, tx=0.00000000, ty=0.00000000), # works fine
    6: dict(auto_res=False, width=1222, height=1222, win_w=2060, win_h=1308, ui_scale=1.667, zoom=1.00000000, tx=0.00000000, ty=0.00000000), # one px wider, breaks
}

conf_mbp = {
    0: dict(auto_res=False, width=1212, height=1212, win_w=1154, win_h=638, ui_scale=1.333, zoom=2.13756609, tx=0.00000000, ty=0.00000000),
}

default_conf = {
    0: dict(auto_res=True, width=512, height=512, win_w=1600, win_h=1024, ui_scale=1.667, zoom=1, tx=1.068e-05, ty=1.068e-05)
}

configs = { 'Debu': conf_debu, 'Eriks-MacBook-Pro.local': conf_mbp }.get(socket.gethostname(), default_conf)

from pyviewer.toolbar_viewer import ToolbarViewer
class Test(ToolbarViewer):
    def setup_state(self):
        self.auto_res = False  # res based on window size
        self.cuda = False
        self.width = 512
        self.height = 512
        self.pattern = PATTERNS.GRID
        self.tex_nearest = True
        self.cnv_nearest = True
        self.conf = 0
        self.load_config(configs[self.conf])
    
    def compute(self):
        if self.auto_res:
            self.width, self.height = [int(v) for v in self.content_size_px]
        
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        match self.pattern:
            case PATTERNS.GRID:
                img[::2, ::2, :] = 255
            case PATTERNS.STAR:
                img = build_siemens_star(width=self.width, height=self.height)
            case _:
                img[:, :, :] = np.array([255, 255, 0])

        #siv.draw(img_hwc=img)

        if self.cuda:
            import torch
            img = torch.from_numpy(img).cuda()

        return img
    
    def load_config(self, conf: dict):
        self.auto_res = conf['auto_res']
        self.width = conf['width']
        self.height = conf['height']
        glfw.set_window_size(self.v._window, conf['win_w'], conf['win_h'])
        self.v.set_ui_scale(conf['ui_scale'])
        self.pan_handler.zoom = conf['zoom']
        self.pan_handler.pan = (conf['tx'], conf['ty'])

    def draw_toolbar(self):
        self.pattern = combo_box_vals('Pattern', PATTERNS, self.pattern, to_str=lambda p: p.value)[1][0]
        self.auto_res = imgui.checkbox('Auto-res', self.auto_res)[1]
        self.pan_enabled = imgui.checkbox('Pan enabled', self.pan_enabled)[1]
        self.cuda = imgui.checkbox('CUDA', self.cuda)[1]
        self.width = imgui.slider_int('Width', self.width, 4, 2048*4)[1]
        self.height = imgui.slider_int('Height', self.height, 4, 2048*4)[1]
        ch, self.tex_nearest = imgui.checkbox('Tex nearest', self.tex_nearest)
        if ch:
            self.v.set_interp_nearest() if self.tex_nearest else self.v.set_interp_linear()
        ch, self.cnv_nearest = imgui.checkbox('Canvas nearest', self.cnv_nearest)
        if ch:
            self.pan_handler.set_interp_nearest() if self.cnv_nearest else self.pan_handler.set_interp_linear()
        self.pan_handler.zoom = imgui.slider_float('Zoom', self.pan_handler.zoom, 0, 10)[1]
        W, H = glfw.get_window_size(self.v._window)
        self.pan_handler.debug_mode = imgui.slider_int('Debug mode', self.pan_handler.debug_mode, 0, self.pan_handler.num_debug_modes - 1)[1]
        imgui.text(f'Window: {W}x{H}')
        imgui.text(f'Canvas: {self.pan_handler.canvas_w}x{self.pan_handler.canvas_h}')
        imgui.text(f'Content: {self.content_size_px[0]}x{self.content_size_px[1]}')
        imgui.text(f'Texture: {self.pan_handler.tex_w}x{self.pan_handler.tex_h}')
        imgui.text(f'Pan: ({self.pan_handler.pan[0]:.3f}, {self.pan_handler.pan[1]:.3f})')
        if imgui.button('Print'):
            print('{}: dict(auto_res=False, width={}, height={}, win_w={}, win_h={}, ui_scale={:.3f}, zoom={:.8f}, tx={:.8f}, ty={:.8f}),'.format(
                len(configs), self.width, self.height, W, H, self.ui_scale, self.pan_handler.zoom, *self.pan_handler.pan
            ))

        ch, self.conf = imgui.slider_int('Config', self.conf, 0, len(configs) - 1)
        if ch:
            self.load_config(configs[self.conf])

_ = Test('test_viewer')
#siv.inst.close()
print('Done')