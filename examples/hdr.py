from pyviewer import _macos_hdr_patch; _macos_hdr_patch.use_patched()
from pyviewer.docking_viewer import DockingViewer, dockable
from imgui_bundle import imgui
import numpy as np

# OpenGL is not color managed
# -> On MacOS, the framebuffer values will be treated as primaries of the display color space
# -> If using Display-P3, sRGB content will be overly saturated
# -> See: discourse.libsdl.org/t/correct-color-rendering-on-wide-color-gamut-displays-like-recent-ish-macs/26842

# Float gradient in [0, 1]
H, W = (512, 512)
l1 = np.linspace(0, 1, max(H, W), dtype=np.float32)
l2 = np.linspace(1, 0, max(H, W), dtype=np.float32)
grad_r = l1.reshape(-1, 1) * l1.reshape(1, -1)
grad_g = l1.reshape(-1, 1) * l2.reshape(1, -1)
grad_b = l2.reshape(-1, 1) * l1.reshape(1, -1)
img = np.stack((grad_r, grad_b, grad_g), axis=-1) # [256, 256, 3]
img = img[:H, :W, :]

class HDRViewer(DockingViewer):
    def setup_state(self):
        self.brightness = 1.0
        self.auto_bright = True
    
    def compute(self):
        self.update_image(img_hwc=self.brightness*img)

    @dockable
    def toolbar(self):
        vmax, vref, vcur = _macos_hdr_patch.get_edr_range(self.window, gamma=2.2)
        if self.auto_bright:
            self.brightness = vcur # current highest value that won't clip
        
        imgui.text(f'EDR headroom: cur={vcur:.2f}, ref={vref:.2f}, max{vmax:.2f}')
        self.brightness = imgui.slider_float('Brightness', self.brightness, 0.0, vmax)[1]
        self.auto_bright = imgui.checkbox('Automatic brightness', self.auto_bright)[1]

if __name__ == "__main__":
    viewer = HDRViewer('HDR toolbar viewer')
