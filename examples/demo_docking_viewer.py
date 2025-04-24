from pyviewer.docking_viewer import DockingViewer, dockable
from imgui_bundle import imgui, hello_imgui # type: ignore
import numpy as np
from pathlib import Path
import struct

def float_flip_lsb(v: float) -> float:
    """Treat python float as 32bit, flip lsb, return"""
    binary = struct.unpack('!I', struct.pack('!f', v))[0] # '!' means big-endian
    binary ^= 1 # flip the least significant bit
    return struct.unpack('!f', struct.pack('!I', binary))[0]

if __name__ == '__main__':
    class Test(DockingViewer):
        def setup_state(self):
            self.state.seed = 0
            self.initial_font_size = 15
            self.dynamic_font = None

        def load_fonts(self):
            self.dynamic_font = hello_imgui.load_font_dpi_responsive("fonts/DroidSans.ttf", self.initial_font_size * self.ui_scale)

        def compute(self):
            rand = np.random.RandomState(seed=self.state.seed)
            img = rand.randn(256, 256, 3).astype(np.float32)
            return np.clip(img, 0, 1) # HWC
        
        def trigger_font_reload(self):
            """
            Trigger font reload after changing font.font_size.
            Used in conjunction with hello_imgui.load_font_dpi_responsive().
            """
            io = imgui.get_io()
            io.font_global_scale = float_flip_lsb(io.font_global_scale) # trigger reload
        
        def set_ui_scale(self, scale):
            self.ui_scale = scale
            self.scale_style_sizes()
            if self.dynamic_font is not None:
                self.dynamic_font.font_size = self.ui_scale * self.initial_font_size
                self.trigger_font_reload()

        @dockable
        def toolbar(self):
            imgui.text(f'Dynamic font size: {self.dynamic_font.font_size:.1f}')
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
            imgui.get_io().font_global_scale = imgui.slider_float('Font global scale', imgui.get_io().font_global_scale, 0.1, 5)[1]

    _ = Test('docking_viewer')
    print('Done')