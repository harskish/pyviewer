"""Exercise the macOS trackpad patches in dockable ImGui windows.

Run with: python examples/trackpad_widgets.py
Scroll over each plot, the text view, the list box, and the open combo.
"""

import numpy as np
from imgui_bundle import imgui, implot, implot3d

from pyviewer.docking_viewer_py import PyDockingViewer, dockable


class TrackpadWidgetsViewer(PyDockingViewer):
    def setup_state(self):
        # PyDockingViewer creates the ImPlot context and installs the macOS
        # patches. Its Python backend does not create an ImPlot3D context.
        self.plot3d_context = implot3d.create_context()
        self.x = np.linspace(0.0, 4.0 * np.pi, 400)
        self.y = np.sin(self.x)
        self.z = np.linspace(-1.0, 1.0, len(self.x))
        self.items = [f'Item {i:02d}' for i in range(60)]
        self.selected_item = 0
        self.combo_item = 0

    def _cleanup(self):
        try:
            implot3d.destroy_context(self.plot3d_context)
        finally:
            super()._cleanup()

    @dockable(title='ImPlot 2D')
    def plot_2d(self):
        imgui.text('Scroll over the plot to zoom around the cursor.')
        if implot.begin_plot('Sine wave', size=(480, 320)):
            implot.setup_axes('x', 'sin(x)')
            implot.plot_line('sin(x)', self.x, self.y)
            implot.end_plot()

    @dockable(title='ImPlot3D')
    def plot_3d(self):
        imgui.text('Scroll over the plot to zoom the 3D view.')
        if implot3d.begin_plot('Helix', size=(480, 320)):
            implot3d.setup_axes('x', 'y', 'z')
            implot3d.plot_line('helix', np.cos(self.x), self.y, self.z)
            implot3d.end_plot()

    @dockable(title='Scrollable views')
    def output(self):
        imgui.text('Scroll the window or the child view below.')
        if imgui.begin_child('Text child', size=(0, 220)):
            for i in range(50):
                imgui.text(f'Child row {i:02d}  -  trackpad scroll test')
        imgui.end_child()
        imgui.separator()
        for i in range(50):
            imgui.text(f'Window row {i:02d}  -  trackpad scroll test')

    @dockable(title='Lists and dropdown')
    def lists(self):
        imgui.text('Scroll the list box, then open and scroll the dropdown.')
        _, self.selected_item = imgui.list_box(
            'List box', self.selected_item, self.items, height_in_items=8
        )
        _, self.combo_item = imgui.combo('Dropdown', self.combo_item, self.items, 8)
        imgui.text(f'List: {self.items[self.selected_item]}')
        imgui.text(f'Dropdown: {self.items[self.combo_item]}')


if __name__ == '__main__':
    TrackpadWidgetsViewer('Trackpad widget scaling')
