import glfw
import array
from pyviewer.toolbar_viewer import AutoUIViewer
from pyviewer.params import *
from pyviewer import plot as implot

PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
    37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
]

def radical_inverse(b: int, i: int):
    exp = 1
    rev = 0
    while i > 0:
        exp = exp / b
        rev += exp * (i % b)
        i = i // b
    return rev

def halton(i: int, dim: int):
    return radical_inverse(PRIMES[dim], i)

# For lists: fast
# For np.array: pass bytearray(a) instead
def toarr(a: list):
    return array.array('f', a)

@strict_dataclass
class State(ParamContainer):
    N: Param = IntParam('Samples', 256, 1, 2048)
    dim1: Param = IntParam('Dimension 1', 0, 0, 20, buttons=True)
    dim2: Param = IntParam('Dimension 2', 1, 0, 20, buttons=True)
    
class Viewer(AutoUIViewer):
    def setup_state(self):
        self.state = State()
    
    def draw_pre(self):
        state = self.state
        W, H = glfw.get_window_size(self.v._window)
        style = imgui.get_style()
        avail_h = H - self.menu_bar_height - 2*style.window_padding.y
        avail_w = W - self.toolbar_width
        plot_side = min(avail_h, avail_w)
        xs = [halton(i, state.dim1) for i in range(state.N)]
        ys = [halton(i, state.dim2) for i in range(state.N)]
        implot.set_next_marker_style(size=6*self.ui_scale)
        implot.begin_plot('', size=(plot_side, plot_side))
        implot.plot_scatter2('Halton', toarr(xs), toarr(ys), len(xs))
        implot.end_plot()

if __name__ == '__main__':
    viewer = Viewer('Plotting example')
