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
            imgui.text(f'Dynamic font size: {self.fonts[0].font_size:.1f}')
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
            imgui.get_io().font_global_scale = imgui.slider_float('Font global scale', imgui.get_io().font_global_scale, 0.1, 5)[1]
            self.ui_scale = imgui.slider_float('UI scale', self.ui_scale, 0.1, 5.0)[1]

    _ = Test('docking_viewer')
    print('Done')