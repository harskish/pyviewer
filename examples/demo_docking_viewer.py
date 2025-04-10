from pyviewer.docking_viewer import DockingViewer, dockable
from imgui_bundle import imgui
import numpy as np

if __name__ == '__main__':
    class Test(DockingViewer):
        def setup_state(self):
            self.state.seed = 0
        
        def compute(self):
            rand = np.random.RandomState(seed=self.state.seed)
            img = rand.randn(256, 256, 3).astype(np.float32)
            return np.clip(img, 0, 1) # HWC
        
        @dockable
        def toolbar(self):
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]

    _ = Test('docking_viewer')
    print('Done')