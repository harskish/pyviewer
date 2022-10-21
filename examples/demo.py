from pathlib import Path
import numpy as np
import array
import pyviewer # pip install -e .
import imgui

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

def toarr(a: np.ndarray):
    return array.array(a.dtype.char, a)

N = 50_000
x = np.linspace(0, 4*np.pi, N)
y = 2*np.cos(x)

x = toarr(x)
y = toarr(y)

class Test(pyviewer.toolbar_viewer.ToolbarViewer):
    def setup_state(self):
        self.state.seed = 0
    
    def compute(self):
        rand = np.random.RandomState(seed=self.state.seed)
        img = rand.randn(256, 256, 3).astype(np.float32)
        return np.clip(img, 0, 1) # HWC
    
    def draw_toolbar(self):
        self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
        if imgui.plot.begin_plot('Plot'):
            imgui.plot.plot_line2('2cos(x)', x, y, N)
            imgui.plot.end_plot()

_ = Test('test_viewer')
print('Done')
