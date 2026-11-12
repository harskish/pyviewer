import sys
import shutil
from importlib.util import find_spec
import platform
import os
from enum import Enum
from pathlib import Path

# Experimental HDR patch for MacOS and Linux (Wayland)
# Patched sources at: https://github.com/harskish/glfw/tree/tom94_plus_IMS212

# imgui-bundle currently uses pinned GLFW v3.3.8
# (https://github.com/pthom/imgui_bundle/tree/main/external/glfw)
# => hdr version based on pre-3.5.0, minor version mismatch here, hopefully works fine

# Locate the package without importing its native extension (which loads GLFW).
bundle_spec = find_spec('imgui_bundle')
assert bundle_spec is not None, 'Could not find imgui-bundle'
bundle_dir = Path(next(iter(bundle_spec.submodule_search_locations)))
is_macos = platform.system() == 'Darwin'
is_linux = platform.system() == 'Linux'
orig = bundle_dir / ('libglfw.3.dylib' if is_macos else 'libglfw.so.3')
backup = orig.with_name('backup_libglfw.3.dylib')
# Nix installs the patched library and this marker at build time.
preinstalled = orig.with_name('pyviewer-hdr-glfw').is_file()
patched_mac = Path(__file__).parent / 'libglfw.3.5.dylib'

# Wayland: just set env variable for glfw itself
patched_linux = Path(__file__).parent / 'libglfw.so.3.5'

GLFW_FLOATBUFFER = 0x00021011
GLFW_WAYLAND_COLOR_MANAGEMENT = 0x00026002

class Mode(int, Enum):
    UNKNOWN = 0
    ORIGINAL = 1
    PATCHED = 2

CUR_MODE = Mode.UNKNOWN

def use_patched():
    global CUR_MODE
    if is_macos and CUR_MODE != Mode.PATCHED:
        assert 'glfw' not in sys.modules, 'glfw already imported, cannot patch'
        assert 'imgui_bundle' not in sys.modules, 'imgui_bundle already imported, cannot patch'
        if not preinstalled:
            assert patched_mac.is_file(), 'Could not find patched libglfw'
            assert orig.is_file(), 'Could not find imgui-bundle\'s original libglfw'
            if not backup.is_file():
                shutil.copy2(orig, backup)
            shutil.copy2(patched_mac, orig)
        import_glfw()
        CUR_MODE = Mode.PATCHED
    elif is_linux and CUR_MODE != Mode.PATCHED:
        assert 'glfw' not in sys.modules, 'glfw already imported, cannot patch'
        import_glfw()
        CUR_MODE = Mode.PATCHED

def use_original():
    global CUR_MODE
    if is_macos and CUR_MODE != Mode.ORIGINAL:
        assert 'glfw' not in sys.modules, 'glfw already imported, cannot patch'
        # Preinstalled builds keep the HDR-capable library, also in SDR mode.
        if not preinstalled and backup.is_file():
            shutil.copy2(backup, orig)
        CUR_MODE = Mode.ORIGINAL
    elif is_linux and CUR_MODE != Mode.ORIGINAL:
        # Wayland: no-op
        CUR_MODE = Mode.ORIGINAL

def get_edr_range(glfw_window, gamma=1) -> tuple[float, float, float]:
    """
    Queries the EDR headroom of the monitor
    Returns:
        max_val: maximum rgb component value of monitor
        ref_val: reference maximum rgb component value (unused?)
        cur_val: current maximum displayable value without clipping
    """
    if CUR_MODE != Mode.PATCHED or not is_macos:
        return (1.0, 1.0, 1.0) # standard LDR range
    
    import glfw
    import ctypes
    from ctypes import c_float, c_void_p, cast, byref
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    monitor = glfw.get_window_monitor(glfw_window) or glfw.get_primary_monitor()
    handle = glfw._glfw
    assert isinstance(handle, ctypes.CDLL)
    rawfn = getattr(handle, 'glfwGetMonitorEDRRange')
    assert rawfn is not None, 'Could not get function pointer to glfwGetMonitorEDRRange'
    
    rawfn.argtypes = [
        c_void_p,  # actually GLFWMonitor*
        c_float_p,
        c_float_p,
        c_float_p,
    ]

    # Create c_float instances for the output values
    max_val = c_float()
    ref_val = c_float()
    cur_val = c_float()

    # Call the function
    _ = rawfn(
        cast(monitor, c_void_p),
        byref(max_val),
        byref(ref_val),
        byref(cur_val),
    )

    # Return raw floats
    return max_val.value**(1/gamma), ref_val.value**(1/gamma), cur_val.value**(1/gamma)


import ctypes
def import_glfw():
    patched = orig if preinstalled or is_macos else patched_linux if is_linux else None
    assert patched.is_file(), 'Could not find patched libglfw'
    os.environ['PYGLFW_LIBRARY'] = patched.as_posix()
    
    import glfw
    assert glfw._glfw._name == patched.as_posix(), 'Did not load patched library'

    from .utils import stdio_filter # utils import glfw
    with stdio_filter(err='WARNING: Wayland is not active'):
        import wayland # python-wayland
    
    from ctypes import cast
    win = ctypes.c_void_p
    lib = glfw._glfw
    
    def window_ptr(window):
        return cast(window, win)

    lib.glfwGetWindowSdrWhiteLevel.argtypes = [win]
    lib.glfwGetWindowSdrWhiteLevel.restype = ctypes.c_float
    setattr(glfw, 'glfwGetWindowSdrWhiteLevel', lambda window: lib.glfwGetWindowSdrWhiteLevel(window_ptr(window)))

    lib.glfwGetWindowMaxLuminance.argtypes = [win]
    lib.glfwGetWindowMaxLuminance.restype = ctypes.c_float
    setattr(glfw, 'glfwGetWindowMaxLuminance', lambda window: lib.glfwGetWindowMaxLuminance(window_ptr(window)))

    def get_transfer_function(window) -> str:
        transf = lib.glfwGetWindowTransfer(window_ptr(window))
        for t in wayland.wp_color_manager_v1.transfer_function:
            if t == transf:
                return t.name
        return f'unknown ({transf})'
    
    lib.glfwGetWindowTransfer.argtypes = [win]
    lib.glfwGetWindowTransfer.restype = ctypes.c_uint32
    setattr(glfw, 'glfwGetWindowTransfer', get_transfer_function)

    def get_primaries(window) -> str:
        prim = lib.glfwGetWindowPrimaries(window_ptr(window))
        for p in wayland.wp_color_manager_v1.primaries:
            if p == prim:
                return p.name
        return f'unknown ({prim})'
    
    lib.glfwGetWindowPrimaries.argtypes = [win]
    lib.glfwGetWindowPrimaries.restype = ctypes.c_uint32
    setattr(glfw, 'glfwGetWindowPrimaries', get_primaries)
