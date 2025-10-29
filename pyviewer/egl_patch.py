# EGL on NixOS and WSL seems to be buggy

import OpenGL.platform as gl_platform
assert getattr(gl_platform.PLATFORM, 'GL') is not None, 'Could not get raw GL platform (libGL probably not found)'

def is_egl():
    return 'EGLPlatform' in str(type(getattr(gl_platform, 'PLATFORM', None))) # as opposed to GLX (x11) or WGL (Windows)

_patched = not is_egl
def patch():
    global _patched
    if _patched:
        return

    # https://github.com/pyimgui/pyimgui/issues/318
    # https://github.com/pygame/pygame/issues/3110
    from OpenGL import contextdata, __version__ as pyogl_ver
    print('Applying EGL PyOpenGL monkey patch')
    if pyogl_ver != '3.1.7':
        print(f'Warning: only tested on 3.1.7, current version is {pyogl_ver}')
    def fixed( context = None ):
        if context is None:
            context = gl_platform.GetCurrentContext()
            if context == None:
                from OpenGL import error
                raise error.Error(
                    """Attempt to retrieve context when no valid context"""
                )
        return context
    contextdata.getContext = fixed
    _patched = True