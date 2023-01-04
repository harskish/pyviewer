from pathlib import Path
import numpy as np
import array
import pyviewer # pip install -e .
import imgui

has_torch = False
try:
    import torch
    has_torch = True
except:
    pass

# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

def toarr(a: np.ndarray):
    return array.array(a.dtype.char, a)

def demo():
    N = 50_000
    x = np.linspace(0, 4*np.pi, N)
    y = 2*np.cos(x)

    x = toarr(x)
    y = toarr(y)

    #siv = pyviewer.single_image_viewer.SingleImageViewer('Async viewer', hidden=True)

    class Test(pyviewer.toolbar_viewer.ToolbarViewer):
        def setup_state(self):
            self.state.seed = 0
            self.state.img = None
        
        def check_output(self, t):
            # Don't accidentally test numpy fallback
            if has_torch and torch.is_tensor(t):
                assert pyviewer.gl_viewer.has_pycuda, 'GL-compatible PyCUDA not installed'            
        
        def compute(self):
            # Prime width, for testing GL_UNPACK_ALIGNMENT
            W = 257
            H = 199
            
            # Float gradient in [0, 1]
            l1 = torch.linspace(0, 1, max(H, W), device='cuda')
            l2 = torch.linspace(1, 0, max(H, W), device='cuda')
            grad_r = l1.view(-1, 1) * l1.view(1, -1)
            grad_g = l1.view(-1, 1) * l2.view(1, -1)
            grad_b = l2.view(-1, 1) * l1.view(1, -1)
            img = torch.stack((grad_r, grad_g, grad_b), dim=-1) # [256, 256, 3]
            img = img[:H, :W, :]

            # Add noise
            torch.manual_seed(self.state.seed)
            img += 0.125 * torch.randn_like(img)

            # TEST: to uint8
            img = (255*torch.clamp(img, 0, 1)).byte()

            # As numpy
            #img = img.detach().cpu().numpy()

            self.check_output(img)
            self.state.img = img
            return self.state.img
        
        def draw_toolbar(self):
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
            if imgui.plot.begin_plot('Plot'):
                imgui.plot.plot_line2('2cos(x)', x, y, N)
                imgui.plot.end_plot()
            
            # imgui.separator()
            # imgui.text('Async viewer: separate process,\nwon\'t freeze if breakpoint is hit.')
            # if siv.hidden:
            #     if imgui.button('Open async viewer'):
            #         print('Opening...')
            #         siv.show(sync=True)
            # elif not siv.started.value:
            #     if imgui.button('Reopen async viewer'):
            #         siv.restart()
            # elif imgui.button('Update async viewer'):
            #     siv.draw(img_hwc=self.state.img)

    _ = Test('test_viewer')
    print('Done')

if __name__ == '__main__':
    # Async viewer requires __main__ guard
    demo()