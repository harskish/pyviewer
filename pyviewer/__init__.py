from . import single_image_viewer
from . import toolbar_viewer
from . import gl_viewer

# Check imgui version
import imgui
assert hasattr(imgui, 'plot'), \
    'Pyviewer requires a custom version of imgui that comes bundled with implot (github.com/harskish/pyimgui).\n' + \
    'Please reinstall pyviewer to get the correct version.'