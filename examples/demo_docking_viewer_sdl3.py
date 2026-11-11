"""SDL_GPU HDR viewer: python examples/demo_docking_viewer_sdl3.py --hdr linear

Use --hdr pq for HDR10/PQ, --hdr sdr for SDR, or auto (default) to prefer HDR.
The image is linear Rec.709; 1.0 is SDR white, with highlights up to 8.0.
"""
import argparse

from pyviewer.docking_viewer_sdl3 import SDL3DockingViewer, dockable
from imgui_bundle import imgui
import numpy as np


class Demo(SDL3DockingViewer):
    def __init__(self, *, frames=0, **kwargs):
        self._frame_limit = frames
        self._frames = 0
        super().__init__('SDL GPU HDR viewer', **kwargs)

    def setup_state(self):
        self.state.exposure = 1.0
        self.state.dropped = ''

    def compute(self):
        ramp = np.linspace(0, 8, 512, dtype=np.float32)[None, :, None]
        return np.broadcast_to(ramp * self.state.exposure, (256, 512, 3)).copy()

    @dockable(title='Controls')
    def controls(self):
        imgui.text(f'GPU: {self.gpu_driver}; presentation: {self.hdr_mode}')
        imgui.text(f'Display HDR: {self.hdr_info["enabled"]}; headroom: {self.hdr_info["headroom"]:.2f}')
        imgui.text(f'Framebuffer density: {tuple(imgui.get_io().display_framebuffer_scale)}')
        _, self.state.exposure = imgui.slider_float('Exposure', self.state.exposure, 0.1, 4.0)
        _, self.ui_scale = imgui.slider_float('UI scale', self.ui_scale, 0.5, 3.0)
        imgui.text('Drag to pan; wheel to zoom; right click to snap; double click to reset.')
        imgui.text('Alt+Enter: fullscreen; Escape: close.')
        imgui.text(self.state.dropped)

    def pre_new_frame(self):
        self._frames += 1
        if self._frame_limit and self._frames >= self._frame_limit:
            self.shutdown()

    def drag_and_drop_callback(self, paths):
        self.state.dropped = ', '.join(map(str, paths))
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdr', choices=('auto', 'linear', 'pq', 'sdr'), default='auto')
    parser.add_argument('--hidden', action='store_true')
    parser.add_argument('--frames', type=int, default=0, help='Exit after N frames (0: interactive)')
    parser.add_argument('--debug-gpu', action='store_true')
    args = vars(parser.parse_args())
    Demo(**args)
