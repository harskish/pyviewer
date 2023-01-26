from pathlib import Path
import numpy as np
import pyviewer
import imgui
from enum import Enum
from pyviewer import single_image_viewer as siv
from pyviewer.utils import combo_box_vals
from functools import lru_cache
from io import BytesIO
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
matplotlib.use('Agg')

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

@lru_cache
def build_siemens_star(origin=(0, 0), radius=1, n=100, DPI=400):
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
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print('Siemens start shape:', img_arr.shape)

    return img_arr

#siv.init('Async viewer', hidden=True)

class PATTERNS(Enum):
    GRID = 'Grid'
    STAR = 'Siemens star'

class Test(pyviewer.toolbar_viewer.ToolbarViewer):
    def setup_state(self):
        self.auto_res = True  # res based on window size
        self.width = 512
        self.height = 512
        self.pattern = PATTERNS.STAR
    
    def compute(self):
        if self.auto_res:
            self.width, self.height = [int(v) for v in self.content_size_px]
        
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        match self.pattern:
            case PATTERNS.GRID:
                img[::2, ::2, :] = 255
            case PATTERNS.STAR:
                img = Image.fromarray(build_siemens_star()) \
                    .resize((self.width, self.height), resample=Image.LANCZOS)
                img = np.array(img)
            case _:
                img[:, :, :] = np.array([255, 255, 0])

        return img
    
    def draw_toolbar(self):
        self.pattern = combo_box_vals('Pattern', PATTERNS, self.pattern, to_str=lambda p: p.value)[1]
        self.auto_res = imgui.checkbox('Auto-res', self.auto_res)[1]
        self.pan_enabled = imgui.checkbox('Use shader', self.pan_enabled)[1]
        self.width = imgui.slider_int('Width', self.width, 4, 2048*4)[1]
        self.height = imgui.slider_int('Height', self.height, 4, 2048*4)[1]

_ = Test('test_viewer')
#siv.inst.close()
print('Done')