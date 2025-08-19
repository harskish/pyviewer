from pyviewer.docking_viewer import MultiTexturesDockingViewer, dockable, hello_imgui
from imgui_bundle import imgui
import numpy as np

if __name__ == '__main__':
    class Test(MultiTexturesDockingViewer):
        WINDOW_NAME_0 = 'Window 0'
        WINDOW_NAME_1 = 'Window 1'
        WINDOW_NAME_2 = 'Window 2'

        def __init__(self):
            super().__init__(
                'Test multiple textures docking viewer',
                texture_names=[
                    self.WINDOW_NAME_0,
                    self.WINDOW_NAME_1,
                    self.WINDOW_NAME_2,
                ],
                full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area,
            )

        def setup_state(self):
            self.state.seed_0 = 0
            self.state.seed_1 = 1
            self.state.seed_2 = 2

        def compute(self):
            rand_0 = np.random.RandomState(seed=self.state.seed_0)
            img_0 = rand_0.randn(256, 256, 3).astype(np.float32)

            rand_1 = np.random.RandomState(seed=self.state.seed_1)
            img_1 = rand_1.randn(256, 256, 3).astype(np.float32)

            rand_2 = np.random.RandomState(seed=self.state.seed_2)
            img_2 = rand_2.randn(256, 256, 3).astype(np.float32)

            return {
                self.WINDOW_NAME_0: np.clip(img_0, 0, 1),
                self.WINDOW_NAME_1: np.clip(img_1, 0, 1),
                self.WINDOW_NAME_2: np.clip(img_2, 0, 1),
            }

        @dockable
        def toolbar(self):
            imgui.text(f'Dynamic font size: {self.fonts[0].font_size:.1f}')
            self.state.seed_0 = imgui.slider_int('Seed 0', self.state.seed_0, 0, 1000)[1]
            self.state.seed_1 = imgui.slider_int('Seed 1', self.state.seed_1, 0, 1000)[1]
            self.state.seed_2 = imgui.slider_int('Seed 2', self.state.seed_2, 0, 1000)[1]
            imgui.get_io().font_global_scale = imgui.slider_float('Font global scale', imgui.get_io().font_global_scale, 0.1, 5)[1]
            self.ui_scale = imgui.slider_float('UI scale', self.ui_scale, 0.1, 5.0)[1]

    _ = Test()
    print('Done')