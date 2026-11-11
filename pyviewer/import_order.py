"""Keep GLFW and imgui_bundle loading in the required platform order."""

import sys


def prepare_imgui_import():
    """Load the first library before callers import imgui_bundle widgets."""
    if sys.platform == 'darwin':
        # Some callbacks break when GLFW loads before imgui_bundle on macOS.
        assert not ('glfw' in sys.modules and 'imgui_bundle' not in sys.modules), \
            'MacOS: must not import glfw before imgui_bundle'
        import imgui_bundle
    elif sys.platform == 'linux':
        # System GLFW supports Wayland; imgui_bundle's bundled GLFW does not.
        assert not ('imgui_bundle' in sys.modules and 'glfw' not in sys.modules), \
            'Linux: must not import imgui_bundle before glfw'
        import glfw
