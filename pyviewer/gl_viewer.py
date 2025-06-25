# Imgui viewer that supports separate ui and compute threads, image uploads from torch tensors.
# Original by Pauli Kemppinen (https://github.com/msqrt)
# Modified by Erik Härkönen

from functools import lru_cache
import numpy as np
import multiprocessing as mp
from pathlib import Path
import threading
from typing import Dict
import sys
from sys import platform
import ctypes
import time
from contextlib import contextmanager, nullcontext
from platform import uname

# Some callbacks broken if imported before imgui_bundle...??
assert 'glfw' not in sys.modules or 'imgui_bundle' in sys.modules, 'glfw should be imported after pyviewer'

from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from .imgui_themes import *

#from .utils import normalize_image_data

import glfw
glfw.ERROR_REPORTING = 'raise' # make sure errors don't get swallowed
import OpenGL.GL as gl

@lru_cache
def get_cuda_synchronize():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return lambda : None
    
def cuda_synchronize():
    sync_fun = get_cuda_synchronize()
    sync_fun()

@lru_cache
def get_cuda_plugin():
    try:
        print('Setting up CUDA PT plugin')
        from . import custom_ops
        pt_plugin = custom_ops.get_plugin('cuda_gl_interop', 'cuda_gl_interop.cpp', Path(__file__).parent / './custom_ops', unsafe_load_prebuilt=True)
        return pt_plugin
    except Exception as e:
        print('Failed to build CUDA-GL plugin:', e)
        print('Images will be uploaded from RAM.')
        return None
    
@lru_cache
def get_mps_plugin():
    try:
        print('Setting up Metal PT plugin')
        from . import custom_ops
        mtl_plugin = custom_ops.get_plugin('MetalGLInterop', 'metal_gl_interop.mm',
            Path(__file__).parent.parent/'pyviewer/custom_ops', cuda=False)
        return mtl_plugin
    except Exception as e:
        print('Failed to build Metal-GL plugin:', e)
        print('Images will be uploaded from RAM.')
        return None

class PTExtMapper:
    tex: int
    resource: int # uint64_t
    
    def __init__(self, tex: int) -> None:
        self.tex = tex
        self.resource = get_cuda_plugin().register(tex)
    
    def unregister(self) -> None:
        get_cuda_plugin().unregister(self.resource)

    def upload(self, ptr: int, W: int, H: int, N: int) -> None:
        get_cuda_plugin().upload(ptr, W, H, N, self.resource) # map, copy, unmap

MIPMAP_MODES = [gl.GL_NEAREST_MIPMAP_NEAREST, gl.GL_LINEAR_MIPMAP_NEAREST, gl.GL_NEAREST_MIPMAP_LINEAR, gl.GL_LINEAR_MIPMAP_LINEAR]
class _texture:
    '''
    This class maps torch tensors to gl textures without a CPU roundtrip.
    '''
    def __init__(self, min_filter=gl.GL_LINEAR, mag_filter=gl.GL_LINEAR):
        # Can be shared between py and c++
        self.tex_2d = gl.glGenTextures(1)
        self.tex = self.tex_2d # currently active
        self.type = gl.GL_TEXTURE_2D
        gl.glBindTexture(self.type, self.tex) # need to bind to modify
        # sets repeat and filtering parameters; change the second value of any tuple to change the value
        for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_MIRRORED_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_MIRRORED_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, min_filter), (gl.GL_TEXTURE_MAG_FILTER, mag_filter)):
            gl.glTexParameteri(self.type, *params)
        self.min_filter = min_filter
        self.mag_filter = mag_filter
        self.mapper = None  # torch extension cudaGraphicsResource_t
        self.is_fp = True
        self.shape = [0,0]

    # be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
    def __del__(self):
        if gl is not None:
            gl.glDeleteTextures(1, [self.tex])

    @property
    def needs_mipmap(self):
        return self.min_filter in MIPMAP_MODES or self.mag_filter in MIPMAP_MODES

    def set_interp(self, key, val):
        gl.glBindTexture(self.type, self.tex)
        gl.glTexParameteri(self.type, key, val)
        gl.glBindTexture(self.type, 0)
        if key == gl.GL_TEXTURE_MIN_FILTER:
            self.min_filter = val
        if key == gl.GL_TEXTURE_MAG_FILTER:
            self.mag_filter = val

    def generate_mipmaps(self):
        if self.type == gl.GL_TEXTURE_2D:
            prev = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, prev)

    def upload_np(self, image: np.ndarray):
        self.upload_iterable(image, image.shape, image.dtype.name)
    
    def upload_iterable(self, data, shape, dtype_str: str):
        has_alpha = shape[2] == 4
        is_fp = dtype_str in ['float32', 'float16']

        # Needed when alternating between CUDA and numpy
        if self.mapper is not None:
            self.mapper.unregister()
            self.mapper = None

        # See upload_ptr() for description of the formats
        internal_fmt = gl.GL_RGBA # if has_alpha else gl.GL_RGB # how OGL stores data
        incoming_fmt = gl.GL_RGBA if has_alpha else gl.GL_RGB # incoming channel format
        incoming_dtype = {
            'float32': gl.GL_FLOAT,
            'float16': gl.GL_HALF_FLOAT,
            'uint8': gl.GL_UNSIGNED_BYTE,
            'uint16': gl.GL_UNSIGNED_SHORT,
        }[dtype_str]

        # RGB UINT8 data: no guarantee of 4-byte row alignment
        # (F32 or RGBA alignment always divisible by 4)
        alignment = 4 if (is_fp or has_alpha) else 1
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment) # default: 4 bytes
        
        # Make sure tex2d is active
        self.tex = self.tex_2d
        self.type = gl.GL_TEXTURE_2D
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        if shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.is_fp != is_fp:
            # Reallocate
            self.shape = shape
            self.is_fp = is_fp
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, # GLenum target
                0,                # GLint level
                internal_fmt,     # GLint internalformat
                shape[1],         # GLsizei width
                shape[0],         # GLsizei height
                0,                # GLint border
                incoming_fmt,     # GLenum format
                incoming_dtype,   # GLenum type
                data,             # const void * data
            )
        else:
            # Overwrite
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,   # GLenum target
                0,                  # GLint level
                0,                  # GLint xoffset
                0,                  # GLint yoffset
                shape[1],           # GLsizei width
                shape[0],           # GLsizei height
                incoming_fmt,       # GLenum format
                incoming_dtype,     # GLenum type
                data,               # const void * pixels
            )
        if self.needs_mipmap:
            self.generate_mipmaps()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_mps(self, img):
        plugin = get_mps_plugin()
        if plugin is None:
            return self.upload_np(img.detach().cpu().numpy())
    
        self.type = gl.GL_TEXTURE_RECTANGLE # Metal-GL interop doesn't support TEXTURE_2D
        self.tex, ch = plugin.gl_tex_rect(img)
        if ch:
            self.set_interp(gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
            self.set_interp(gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)
    
    def upload_torch(self, img):
        import torch
        assert img.ndim == 3, "Please provide a HWC tensor"
        assert img.shape[2] < min(img.shape[0], img.shape[1]), "Please provide a HWC tensor"
        if img.device.type == 'mps':
            return self.upload_mps(img)
        
        if get_cuda_plugin() is None:
            return self.upload_np(img.detach().cpu().numpy())
        
        assert img.dtype in [torch.float32, torch.uint8], 'CUDA interop: only fp32 and u8 supported'
        
        # OpenGL stores RGBA-strided data always
        # Must add alpha for gpu memcopy to work
        if img.shape[2] == 3:
            alpha = 255 if img.dtype == torch.uint8 else 1.0
            img = torch.cat((img, alpha*torch.ones_like(img[:, :, :1])), dim=-1)

        self.type = gl.GL_TEXTURE_2D
        img = img.contiguous()
        self.upload_ptr(img.data_ptr(), img.shape, img.dtype.is_floating_point)
        if self.needs_mipmap:
            self.generate_mipmaps()

    # Copy from cuda pointer
    def upload_ptr(self, ptr, shape, is_fp32):
        assert get_cuda_plugin() is not None, 'PT plugin needed for pointer upload'
        has_alpha = shape[-1] == 4

        # Reallocate if shape changed or data type changed from np to torch
        # Must also reallocate before mipmap (re)generation (https://stackoverflow.com/a/20359917)
        if self.needs_mipmap or shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.is_fp != is_fp32 or self.mapper is None:
            self.shape = shape
            self.is_fp = is_fp32
            if self.mapper is not None:
                self.mapper.unregister()
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

            # Internal format specifies how OpenGL stores the data
            #   RGB32F: data stored as 32bit floats internally, shader sees normal ieee floats
            #   RGB8: data stored in unsigned normalized format, shader sees equally spaced floats in [0, 1] ("compressed float")
            #   RGBA8UI: data stored in unsigned integer format, shader sees ints
            
            
            """
            And if you are interested, most GPUs like chunks of 4 bytes.
            In other words, GL_RGBA or GL_BGRA is preferred when each component is a byte.
            GL_RGB and GL_BGR is considered bizarre since most GPUs, most CPUs and any other kind of chip don't handle 24 bits.
            This means, the driver converts your GL_RGB or GL_BGR to what the GPU prefers, which typically is RGBA/BGRA.
            (https://www.khronos.org/opengl/wiki/Common_Mistakes)
            """
            # => just use RGBA for compatibility
            internal_fmt = gl.GL_RGBA32F if is_fp32 else gl.GL_RGBA8
            assert has_alpha, 'ptr upload needs alpha channel'

            # Incoming channel format and dtype
            incoming_fmt = gl.GL_RGBA if has_alpha else gl.GL_RGB
            incoming_dtype = gl.GL_FLOAT if is_fp32 else gl.GL_UNSIGNED_BYTE # fp32 or u8

            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_fmt, shape[1], shape[0], 0, incoming_fmt, incoming_dtype, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self.mapper = PTExtMapper(int(self.tex))
        
        # Cast to python integer type
        ptr_int = int(ptr)
        assert ptr_int == ptr, 'Device pointer overflow'

        N = 4 if is_fp32 else 1
        H, W, C = shape
        assert C == 4, 'Input data must be RGBA' # OpenGL always stores alpha channel

        self.mapper.upload(ptr_int, W, H, C*N)
        cuda_synchronize()

class _editable:
    def __init__(self, name, ui_code = '', run_code = ''):
        self.name = name
        self.ui_code = ui_code if len(ui_code)>0 else 'imgui.begin(\'Test\')\nimgui.text(\'Example\')#your code here!\nimgui.end()'
        self.tentative_ui_code = self.ui_code
        self.run_code = run_code
        self.run_exception = ''
        self.ui_exception = ''
        self.ui_code_visible = False
    def try_execute(self, string, **kwargs):
        try:
            for key, value in kwargs.items():
                locals()[key] = value
            exec(string)
        except Exception as e: # while generally a bad idea, here we truly want to skip any potential error to not disrupt the worker threads
            return 'Exception: ' + str(e)
        return ''
    def loop(self, v):
        imgui.begin(self.name)
        
        self.run_code = imgui.input_text_multiline('run code', self.run_code, 2048)[1]
        if len(self.run_exception)>0:
            imgui.text(self.run_exception)

        _, self.ui_code_visible = imgui.checkbox('Show UI code', self.ui_code_visible)
        if self.ui_code_visible:
            self.tentative_ui_code = imgui.input_text_multiline('ui code', self.tentative_ui_code, 2048)[1]
            if imgui.button('Apply UI code'):
                self.ui_code = self.tentative_ui_code
            if len(self.ui_exception)>0:
                imgui.text(self.ui_exception)
                
        imgui.end()

        self.ui_exception = self.try_execute(self.ui_code, v=v)

    def run(self, **kwargs):
        self.run_exception = self.try_execute(self.run_code, **kwargs)


class viewer:
    def __init__(self, title, inifile=None, swap_interval=0, hidden=False):
        self.quit = False

        self._images = {}
        self._editables = {}
        self.tex_interp_mode_min = gl.GL_LINEAR
        self.tex_interp_mode_mag = gl.GL_LINEAR
        self.default_font_size = 15
        self.renderer = None

        fname = inifile or "".join(c for c in title.lower() if c.isalnum())
        self._inifile = Path(fname).with_suffix('.ini')

        if not glfw.init():
            raise RuntimeError('GLFW init failed')
        
        try:
            with open(self._inifile, 'r') as file:
                self._width, self._height = [max(int(i), 1) for i in file.readline().split()]
                self.window_pos = [max(int(i), 0) for i in file.readline().split()]
                start_maximized = int(file.readline().rstrip())
                self.ui_scale = float(file.readline().rstrip())
                self.fullscreen = bool(int(file.readline().rstrip()))
                key = file.readline().rstrip()
                while key is not None and len(key)>0:
                    code = [None, None]
                    for i in range(2):
                        lines = int(file.readline().rstrip())
                        code[i] = '\n'.join((file.readline().rstrip() for _ in range(lines)))
                    self._editables[key] = _editable(key, code[0], code[1])
                    key = file.readline().rstrip()
        except Exception as e:
            self._width, self._height = 1280, 720
            self.window_pos = (50, 50)
            self.ui_scale = 1.0
            self.fullscreen = False
            start_maximized = 0

        glfw.window_hint(glfw.MAXIMIZED, start_maximized)
        glfw.window_hint(glfw.VISIBLE, not hidden)
        
        # MacOS, WSL require forward-compatible core profile
        is_wsl = 'microsoft-standard' in uname().release
        if 'darwin' in platform or is_wsl:
            glsl_version = "#version 150"
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

            # Enable EDR
            glfw.window_hint(glfw.RED_BITS, 16)
            glfw.window_hint(glfw.GREEN_BITS, 16)
            glfw.window_hint(glfw.BLUE_BITS, 16)
        else:
            # GL 3.0 + GLSL 130
            glsl_version = "#version 130"
            # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
            # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) # // 3.2+ only
            # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        if is_wsl:
            # https://github.com/pyimgui/pyimgui/issues/318
            # https://github.com/pygame/pygame/issues/3110
            from OpenGL import contextdata, platform as gl_platform
            print('Applying WSL PyOpenGL monkey patch (only tested on 3.1.7)')
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
        
        from py.io import StdCaptureFD # type: ignore
        capture = StdCaptureFD(out=False, in_=False)
        try:
            if self.fullscreen:
                monitor = glfw.get_monitors()[0]
                params = glfw.get_video_mode(monitor)
                self._window = glfw.create_window(params.size.width, params.size.height, title, monitor, None)
            else:
                self._window = glfw.create_window(self._width, self._height, title, None, None)
        except glfw.GLFWError as e:
            stdout, stderr = capture.reset()
            print(stdout)
            print(stderr)
            print(f'Window creation failed:\n{e}')
            if 'MESA-LOADER' in stderr and is_wsl:
                print('MESA loader errors on WSL!')
                print('Make sure required symlink exists: `sudo ln -s /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri`')
                print('If MESA complains about GLIBCXX, then also run `conda install -c conda-forge gcc=12.1.0`')
            import sys
            sys.exit(1)
        else: # try-except else branch: no error
            capture.reset()
        
        if not self._window:
            raise RuntimeError('Could not create window')

        glfw.set_window_pos(self._window, *self.window_pos)
        glfw.make_context_current(self._window)
        # print('GL context:', gl.glGetString(gl.GL_VERSION).decode('utf8'))

        glfw.swap_interval(swap_interval) # should increase on high refresh rate monitors
        #glfw.make_context_current(None) # TODO: why?

        self._imgui_context = imgui.create_context()
        self._implot_context = implot.create_context()
        implot.set_imgui_context(self._imgui_context)

        # Transfer window address to imgui.backends.glfw_init_for_opengl
        window_address = ctypes.cast(self._window, ctypes.c_void_p).value
        assert window_address is not None
        imgui.backends.glfw_init_for_opengl(window_address, True)
        imgui.backends.opengl3_init(glsl_version)

        #self.hdpi_factor = 1 / glfw.get_monitor_content_scale(glfw.get_primary_monitor())[0]
        #imgui.get_io().font_global_scale = self.hdpi_factor
        font = self.get_default_font()

        # MPLUSRounded1c-Medium.ttf: no content for sizes >35
        # Apple M1, WSL have have low texture count limits
        # Too many fonts => GLFWRenderer.refresh_font_texture() will be slow
        font_sizes = range(8, 36, 1) if 'win32' in platform else range(9, 36, 2) # make sure to include 15 (default font size)
        font_sizes = [int(s) for s in font_sizes]
        handle = imgui.get_io().fonts
        glyph_range = None #handle.get_glyph_ranges_chinese_full() # NB: full Chinese range super slow
        self._imgui_fonts = {
            size: handle.add_font_from_file_ttf(font, size, glyph_ranges_as_int_list=glyph_range) for size in font_sizes
        }

        # TODO: add scale field to font?
        # github.com/harskish/pyplotgui/blob/dev/version-2.0/imgui/core.pyx#L2344

        self._context_lock = mp.Lock()
        self._context_tid = None # id of thread in critical section
        self.set_ui_scale(self.ui_scale)

    def get_default_font(self):
        font = Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return str(font)

    @contextmanager
    def lock(self, strict=True):
        # Prevent double locks, e.g. when
        # calling upload_image() from UI thread
        tid = threading.get_ident()
        if self._context_tid == tid:
            yield self._context_lock
            return
        
        context_manager = None
        
        try:
            self._context_lock.acquire()
            self._context_tid = tid
            glfw.make_context_current(self._window)
            context_manager = self._context_lock
        except glfw.GLFWError as e:
            reason = {65544: 'No monitor found'}.get(e.error_code, 'unknown')
            print(f'{str(e)} (code 0x{e.error_code:x}: "{reason}")')
            context_manager = nullcontext
            if strict:
                raise e
        finally:
            yield context_manager

            # Cleanup after caller is done
            glfw.make_context_current(None)
            self._context_tid = None
            self._context_lock.release()

    # Scales fonts and sliders/etc
    def set_ui_scale(self, scale):
        k = self.default_font_size
        self.set_font_size(k*scale)
        self.ui_scale = self.font_size / k

    def set_interp(self, min, mag, update_existing=True):
        self.tex_interp_mode_min = min
        self.tex_interp_mode_mag = mag
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, self.tex_interp_mode_min)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, self.tex_interp_mode_mag)
                if tex.needs_mipmap:
                    tex.generate_mipmaps()
    
    def set_interp_linear(self, update_existing=True):
        self.set_interp(gl.GL_LINEAR, gl.GL_LINEAR, update_existing)

    def set_interp_nearest(self, update_existing=True):
        self.set_interp(gl.GL_NEAREST, gl.GL_NEAREST, update_existing)

    def editable(self, name, **kwargs):
        if name not in self._editables:
            self._editables[name] = _editable(name)
        self._editables[name].run(**kwargs)

    # Includes keyboard (glfw.KEY_A) and mouse (glfw.MOUSE_BUTTON_LEFT)
    def keydown(self, key):
        return key in self._pressed_keys

    def keyhit(self, key):
        if key in self._hit_keys:
            self._hit_keys.remove(key)
            return True
        return False

    def draw_texture(self, handle, tex_W, tex_H, scale=1, width=None, pad_h=0, pad_v=0):
        if width == 'fill':
            scale = imgui.get_window_content_region_width() / tex_W
        elif width == 'fit':
            cW, cH = [r-l for l,r in zip(
                imgui.get_window_content_region_min(), imgui.get_window_content_region_max())]
            scale = min((cW-pad_h)/tex_W, (cH-pad_v)/tex_H)
        elif width is not None:
            scale = width / tex_W
        imgui.image(handle, tex_W*scale, tex_H*scale)

    def draw_image(self, name, scale=1, width=None, pad_h=0, pad_v=0):
        if name in self._images:
            img = self._images[name]
            self.draw_texture(img.tex, img.shape[1], img.shape[0], scale, width, pad_h, pad_v)

    def close(self):
        glfw.set_window_should_close(self._window, True)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.3) # 0.4

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.set_fullscreen(self.fullscreen)

    def set_fullscreen(self, value):
        monitor = glfw.get_monitors()[0]
        params = glfw.get_video_mode(monitor)
        if value:
            # Save size and pos
            self._width, self._height = glfw.get_window_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)
            glfw.set_window_monitor(self._window, monitor, \
                0, 0, params.size.width, params.size.height, params.refresh_rate)
        else:
            # Restore previous size and pos
            posy = max(10, self.window_pos[1]) # title bar at least partially visible
            glfw.set_window_monitor(self._window, None, \
                self.window_pos[0], posy, self._width, self._height, params.refresh_rate)

    @lru_cache(maxsize=1) # only run if params change
    def set_default_style(self, color_scheme='dark', spacing=9, indent=23, scrollbar=27):
        #theme_custom()        
        #theme_dark_overshifted()
        #theme_ps()
        theme_deep_dark()
        #theme_contrast()
        
        # Overrides based on UI scale / font size
        s = imgui.get_style()
        s.window_padding        = [spacing, spacing]
        s.item_spacing          = [spacing, spacing]
        s.item_inner_spacing    = [spacing, spacing]
        s.columns_min_spacing   = spacing
        s.indent_spacing        = indent
        s.scrollbar_size        = scrollbar

        c0 = s.color_(imgui.Col_.menu_bar_bg)
        c1 = s.color_(imgui.Col_.frame_bg)
        s.set_color_(imgui.Col_.popup_bg, c0 * 0.7 + c1 * 0.3) # force alpha one?

    def gl_shutdown(self):
        """
        Cleanup OpenGL resources. Called on window close
        or manually if start() was never called (batch mode)
        """
        glfw.make_context_current(self._window)
        del self._images
        self._images: Dict[str, _texture] = {}
        glfw.make_context_current(None)
        glfw.destroy_window(self._window)
    
    def start(self, loopfunc, workers = (), glfw_init_callback = None):
        self._pressed_keys = set()
        self._hit_keys = set()
        
        with self.lock():
            t0 = time.monotonic()
            self.renderer = GlfwRenderer(self._window)
            print(f'GlfwRenderer init (incl. font atlas) took: {time.monotonic()-t0:.2f}s')

        def on_key(window, key, scan, pressed, mods):
            if pressed:
                if key not in self._pressed_keys:
                    self._hit_keys.add(key)
                self._pressed_keys.add(key)
            else:
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key) # check seems to be needed over RDP sometimes
            if key != glfw.KEY_ESCAPE: # imgui erases text with escape (??)
                self.renderer.keyboard_callback(window, key, scan, pressed, mods)

        def on_mouse_button(window, key, action, mods):
            if action == glfw.PRESS:
                if key not in self._pressed_keys:
                    self._hit_keys.add(key)
                self._pressed_keys.add(key)
            else:
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key)
            self.renderer.mouse_button_callback(window, key, action, mods)

        glfw.set_key_callback(self._window, on_key)
        glfw.set_mouse_button_callback(self._window, on_mouse_button)

        # For settings custom callbacks etc.
        if glfw_init_callback is not None:
            glfw_init_callback(self._window)

        # allow single thread object
        if not hasattr(workers, '__len__'):
            workers = (workers,)

        for i in range(len(workers)):
            workers[i].start()

        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self.renderer.process_inputs()

            if self.keyhit(glfw.KEY_ESCAPE):
                glfw.set_window_should_close(self._window, 1)

            with self.lock(strict=False) as l:
                if l == nullcontext:
                    continue
                
                # If adding fonts, before new_frame:
                #self.renderer.refresh_font_texture()
                
                imgui.new_frame()

                # Tero viewer:
                imgui.push_font(self._imgui_fonts[self._cur_font_size])
                self.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)
        
                loopfunc(self)

                for key in self._editables:
                    self._editables[key].loop(self)

                imgui.pop_font()

                imgui.render()
                
                gl.glClearColor(0, 0, 0, 1)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                self.renderer.render(imgui.get_draw_data())
                
                # TODO: compute thread has to wait until sync is done
                # and lock is released if calling upload_image()?
                glfw.swap_buffers(self._window)
        
        # Update size and pos
        if not self.fullscreen:
            self._width, self._height = glfw.get_framebuffer_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)

        with open(self._inifile, 'w') as file:
            file.write('{} {}\n'.format(self._width, self._height))
            file.write('{} {}\n'.format(*self.window_pos))
            file.write('{}\n'.format(glfw.get_window_attrib(self._window, glfw.MAXIMIZED)))
            file.write('{}\n'.format(self.ui_scale))
            file.write('{}\n'.format(int(self.fullscreen)))
            for k, e in self._editables.items():
                file.write(k+'\n')
                for code in (e.ui_code, e.run_code):
                    lines = code.split('\n')
                    file.write(str(len(lines))+'\n')
                    for line in lines:
                        file.write(line+'\n')

        with self.lock():
            self.quit = True

        for i in range(len(workers)):
            workers[i].join()
            
        self.gl_shutdown()

    def upload_image(self, name, data):
        if isinstance(data, np.ndarray):
            return self.upload_image_np(name, data)
        
        import torch
        assert torch.is_tensor(data), 'Unknown image type (expected np.ndarray or torch.Tensor)'
        if data.device.type == 'cpu':
            return self.upload_image_np(name, data.numpy())
        else:
            return self.upload_image_torch(name, data)

    # Upload image from PyTorch tensor
    def upload_image_torch(self, name, tensor):
        import torch
        assert isinstance(tensor, torch.Tensor)

        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            cuda_synchronize()
            if not self.quit:
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode_min, self.tex_interp_mode_mag)
                self._images[name].upload_torch(tensor)

    def upload_image_np(self, name, data):
        assert isinstance(data, np.ndarray)

        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            if not self.quit:
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode_min, self.tex_interp_mode_mag)
                self._images[name].upload_np(data)
