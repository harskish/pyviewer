import sys
import shutil
import site
import platform
from enum import Enum
from pathlib import Path

# Experimental MacOS EDR (HDR) patch
# Patched sources at: https://github.com/harskish/glfw

# imgui-bundle currently uses pinned GLFW v3.3.8
# (https://github.com/pthom/imgui_bundle/tree/main/external/glfw)
# => hdr version based on pre-3.5.0, minor version mismatch here, hopefully works fine
orig = Path(site.getsitepackages()[0]) / 'imgui_bundle' / 'libglfw.3.dylib'
backup = orig.with_name('backup_libglfw.3.dylib')
patched = Path(__file__).parent / 'libglfw.3.5.dylib'
is_macos = platform.system() == 'Darwin'

class Mode(int, Enum):
    UNKNOWN = 0
    ORIGINAL = 1
    PATCHED = 2

CUR_MODE = Mode.UNKNOWN

# Make copy of original
if is_macos and not backup.is_file():
    assert orig.is_file(), 'Could not find imgui-bundle\'s original libglfw'
    shutil.copy2(orig, backup)

def use_patched():
    global CUR_MODE
    if is_macos:
        assert 'glfw' not in sys.modules, 'glfw already imported, cannot patch'
        assert patched.is_file(), 'Could not find patched libglfw'
        shutil.copy2(patched, orig)
        CUR_MODE = Mode.PATCHED

def use_original():
    global CUR_MODE
    if is_macos:
        assert 'glfw' not in sys.modules, 'glfw already imported, cannot patch'
        assert backup.is_file(), 'Could not find backed-up libglfw'
        shutil.copy2(backup, orig)
        CUR_MODE = Mode.ORIGINAL

def get_edr_range(glfw_window) -> tuple[float, float, float]:
    """
    Queries the EDR headroom of the monitor
    Returns:
        max_val: maximum rgb component value of monitor
        ref_val: reference maximum rgb component value (unused?)
        cur_val: current maximum displayable value without clipping
    """
    if not is_macos or CUR_MODE != Mode.PATCHED:
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
    return max_val.value, ref_val.value, cur_val.value
