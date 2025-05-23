from pathlib import Path
import numpy as np
import pyviewer # pip install -e .
from imgui_bundle import imgui, implot
from pyviewer import single_image_viewer as siv
from pyviewer.toolbar_viewer import ToolbarViewer

has_torch = False
try:
    import torch
    has_torch = True
except:
    pass

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

N = 50_000
x = np.linspace(0, 4*np.pi, N)
y = 2*np.cos(x)

siv.init('Async viewer', hidden=True)

class Test(ToolbarViewer):
    def setup_state(self):
        self.state.seed = 0
        self.state.img = None
    
    def compute(self):
        # Prime width, for testing GL_UNPACK_ALIGNMENT
        W = 257
        H = 199
        
        # Float gradient in [0, 1]
        l1 = np.linspace(0, 1, max(H, W), dtype=np.float32)
        l2 = np.linspace(1, 0, max(H, W), dtype=np.float32)
        grad_r = l1.reshape(-1, 1) * l1.reshape(1, -1)
        grad_g = l1.reshape(-1, 1) * l2.reshape(1, -1)
        grad_b = l2.reshape(-1, 1) * l1.reshape(1, -1)
        img = np.stack((grad_r, grad_b, grad_g), axis=-1) # [256, 256, 3]
        img = img[:H, :W, :]

        # Add noise
        rand = np.random.RandomState(seed=self.state.seed)
        img += 0.15 * rand.randn(*img.shape).astype(np.float32)

        # TEST: to uint8
        img = np.uint8(255*np.clip(img, 0, 1))

        # As torch tensor?
        # img = torch.from_numpy(img).to('mps')

        self.state.img = img
        return self.state.img
    
    def draw_toolbar(self):
        self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
        if implot.begin_plot('Plot'):
            implot.plot_line('2cos(x)', x, y)
            implot.end_plot()
        
        imgui.separator()
        imgui.text('Async viewer: separate process,\nwon\'t freeze if breakpoint is hit.')

        if siv.inst.hidden:
            if imgui.button('Open async viewer'):
                print('Opening...')
                siv.inst.show(sync=True)
        elif not siv.inst.started.value:
            if imgui.button('Reopen async viewer'):
                siv.inst.restart()
        elif imgui.button('Update async viewer'):
            siv.draw(img_hwc=self.state.img)

_ = Test('test_viewer')
siv.inst.close()
print('Done')