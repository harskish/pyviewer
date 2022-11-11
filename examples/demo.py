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
        
        def compute(self):
            import torch
            
            rand = np.random.RandomState(seed=self.state.seed)
            img = rand.randn(256, 256, 3).astype(np.float32) # works (because np?)
            #img = torch.tensor(rand.randn(256, 256, 3).astype(np.float32), device='cuda') # OK, but squeezed
            #img = rand.randn(256, 256, 4).astype(np.float32) # works (because np?)
            #img = torch.tensor(rand.randn(256, 256, 4).astype(np.float32), device='cuda') # OK!
            
            # l1 = torch.linspace(0, 1, 256, device='cuda')
            # l2 = torch.linspace(1, 0, 256, device='cuda')
            # grad_r = l1.view(-1, 1) * l1.view(1, -1)
            # grad_g = l1.view(-1, 1) * l2.view(1, -1)
            # grad_b = l2.view(-1, 1) * l1.view(1, -1)
            # img = torch.stack((grad_r, grad_g, grad_b), dim=-1) # [256, 256, 3]

            # # TEST: to uint8
            # img = (255*torch.clamp(img, 0, 1)).byte()

            # # Add alpha?
            # #alpha = 1 if img.dtype.is_floating_point else 255
            # #img = torch.cat((img, alpha*torch.ones_like(img[:, :, :1])), dim=-1)

            self.state.img = np.clip(img, 0, 1) # HWC
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