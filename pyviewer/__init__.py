# Check imgui version
import imgui
plot = getattr(imgui, 'plot', None)
assert plot, \
    'Pyviewer requires a custom version of imgui that comes bundled with implot (github.com/harskish/pyimgui).\n' + \
    'Please reinstall pyviewer to get the correct version.'
