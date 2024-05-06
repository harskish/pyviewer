from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from pathlib import Path
from copy import deepcopy
import numpy as np

@strict_dataclass
class State(ParamContainer):
    seed: Param = IntParam('Seed', 0, 0, 10)
    xform: Param = EnumSliderParam('Xform', np.sin,
        [np.sin, np.square, np.log], lambda f: f.__name__)
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = State()
        self.state_last = None
        self.cache = {}
        #self.v.set_interp_linear() # use bilinear instead of nearest
    
    def draw_pre(self):
        pass # fullscreen plotting (examples/plotting.py)

    def draw_overlays(self, draw_list):
        pass # draw on top of UI elements

    def draw_output_extra(self):
        pass # draw below main output (see `pad_bottom` in constructor)

    def draw_menu(self):
        pass

    def drag_and_drop_callback(self, paths: list[Path]):
        pass

    def compute(self):        
        if self.state_last != self.state:
            #self.init() # react to UI change
            self.state_last = deepcopy(self.state)
        key = str(self.state)
        if key not in self.cache:
            self.cache[key] = self.process(self.state)
        return self.cache[key]
    
    def process(self, state: State, res=96):
        noise = np.random.RandomState(state.seed).randn(res, res, 1)
        grad = np.linspace([0], [1], res) * np.linspace(0, 1, res)
        return state.xform(0.05 * noise + np.stack((grad, grad[::-1, :], grad[:, ::-1]), axis=-1))

if __name__ == '__main__':
    viewer = Viewer('AutoUI example')
