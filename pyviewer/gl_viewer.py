# Imgui viewer that supports separate ui and compute threads, image uploads from torch tensors.
# Original by Pauli Kemppinen (https://github.com/msqrt)
# Modified by Erik Härkönen

import numpy as np
import multiprocessing as mp
from pathlib import Path
from threading import get_ident
from typing import Dict
from sys import platform
from contextlib import contextmanager, nullcontext
from platform import uname

import imgui.core
import imgui.plot as implot
from imgui.integrations.glfw import GlfwRenderer
from .imgui_themes import theme_deep_dark
from .utils import normalize_image_data

import glfw
glfw.ERROR_REPORTING = 'raise' # make sure errors don't get swallowed
import OpenGL.GL as gl

has_torch = False
try:
    import torch
    has_torch = True
except:
    pass

cuda_synchronize = lambda : None
if has_torch and torch.cuda.is_available():
    cuda_synchronize = torch.cuda.synchronize

has_pycuda = False
try:
    import pycuda
    import pycuda.gl as cuda_gl
    import pycuda.tools
    has_pycuda = True
except Exception:
    pass

class _texture:
    '''
    This class maps torch tensors to gl textures without a CPU roundtrip.
    '''
    def __init__(self, min_mag_filter=gl.GL_LINEAR):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex) # need to bind to modify
        # sets repeat and filtering parameters; change the second value of any tuple to change the value
        for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, min_mag_filter), (gl.GL_TEXTURE_MAG_FILTER, min_mag_filter)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, *params)
        self.mapper = None
        self.is_fp = True
        self.shape = [0,0]

    # be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
    def __del__(self):
        gl.glDeleteTextures(1, [self.tex])
        if self.mapper is not None:
            self.mapper.unregister()

    def set_interp(self, key, val):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, key, val)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_np(self, image):
        shape = image.shape
        is_fp = image.dtype.kind == 'f'
        has_alpha = image.shape[2] == 4
        
        # See upload_ptr() for description of the formats
        internal_fmt = gl.GL_RGB32F if is_fp else gl.GL_RGB8 # how OGL stores data
        incoming_fmt = gl.GL_RGBA if has_alpha else gl.GL_RGB  # incoming channel format
        incoming_dtype = gl.GL_FLOAT if is_fp else gl.GL_UNSIGNED_BYTE # incoming dtype

        # RGB UINT8 data: no guarantee of 4-byte row alignment
        # (F32 or RGBA alignment always divisible by 4)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1) # default: 4 bytes

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
                image,            # const void * data
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
                image,              # const void * pixels
            )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_torch(self, img):
        assert img.device.type == "cuda", "Please provide a CUDA tensor"
        assert img.ndim == 3, "Please provide a HWC tensor"
        assert img.shape[2] < min(img.shape[0], img.shape[1]), "Please provide a HWC tensor"
        assert img.dtype in [torch.float32, torch.uint8], 'Only fp32 and u8 supported'

        if not has_pycuda:
            return self.upload_np(img.detach().cpu().numpy())
        
        # OpenGL stores RGBA-strided data always
        # Must add alpha for gpu memcopy to work
        if img.shape[2] == 3:
            alpha = 255 if img.dtype == torch.uint8 else 1.0
            img = torch.cat((img, alpha*torch.ones_like(img[:, :, :1])), dim=-1)

        img = img.contiguous()
        self.upload_ptr(img.data_ptr(), img.shape, img.dtype.is_floating_point)

    # Copy from cuda pointer
    def upload_ptr(self, ptr, shape, is_fp32):
        assert has_pycuda, 'PyCUDA-GL not available, cannot upload using raw pointer'
        has_alpha = shape[-1] == 4

        # reallocate if shape changed or data type changed from np to torch
        if shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.is_fp != is_fp32 or self.mapper is None:
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
            self.mapper = cuda_gl.RegisteredImage(int(self.tex), gl.GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.WRITE_DISCARD)
        
        # map texture to cuda ptr
        tex_data = self.mapper.map()
        tex_arr = tex_data.array(0, 0)

        # Cast to python integer type
        ptr_int = int(ptr)
        assert ptr_int == ptr, 'Device pointer overflow'

        N = 4 if is_fp32 else 1
        H, W, C = shape
        assert C == 4, 'Input data must be RGBA' # OpenGL always stores alpha channel
        
        # copy from torch tensor to mapped gl texture (avoid cpu roundtrip)
        # https://documen.tician.de/pycuda/driver.html#pycuda.driver.Memcpy2D
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(ptr_int)
        cpy.set_dst_array(tex_arr)
        cpy.width_in_bytes = W*C*N  # Number of bytes to copy for each row in the transfer
        cpy.src_pitch = W*C*N       # Size of a row in bytes at the origin of the copy
        cpy.dst_pitch = W*C*N       # Size of a row in bytes at the destination of the copy
        cpy.height = H
        cpy(aligned=False)

        # cleanup
        tex_data.unmap()
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
        self.tex_interp_mode = gl.GL_LINEAR
        self.default_font_size = 15
        
        fname = inifile or "".join(c for c in title.lower() if c.isalnum())
        self._inifile = Path(fname).with_suffix('.ini')

        if not glfw.init():
            raise RuntimeError('GLFW init failed')

        if not has_pycuda:
            print('PyCUDA with GL support not available, images will be uploaded from RAM.')
        
        try:
            with open(self._inifile, 'r') as file:
                self._width, self._height = [int(i) for i in file.readline().split()]
                self.window_pos = [int(i) for i in file.readline().split()]
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
        if 'darwin' in platform or 'microsoft-standard' in uname().release:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        if self.fullscreen:
            monitor = glfw.get_monitors()[0]
            params = glfw.get_video_mode(monitor)
            self._window = glfw.create_window(params.size.width, params.size.height, title, monitor, None)
        else:
            self._window = glfw.create_window(self._width, self._height, title, None, None)
        
        if not self._window:
            raise RuntimeError('Could not create window')

        glfw.set_window_pos(self._window, *self.window_pos)
        glfw.make_context_current(self._window)
        print('GL context:', gl.glGetString(gl.GL_VERSION).decode('utf8'))

        self._cuda_context = None
        if has_pycuda:
            pycuda.driver.init()
            self._cuda_context = pycuda.gl.make_context(pycuda.driver.Device(0))
        glfw.swap_interval(swap_interval) # should increase on high refresh rate monitors
        glfw.make_context_current(None)

        self._imgui_context = imgui.create_context()
        self._implot_context = implot.create_context()
        implot.set_imgui_context(self._imgui_context)
        #implot.get_style().anti_aliased_lines = True # turn global AA on

        font = self.get_default_font()

        # MPLUSRounded1c-Medium.tff: no content for sizes >35
        # Apple M1, WSL have have low texture count limits
        font_sizes = range(8, 36, 1) if 'win32' in platform else range(8, 36, 2)
        font_sizes = [int(s) for s in font_sizes]
        handle = imgui.get_io().fonts
        self._imgui_fonts = {
            size: handle.add_font_from_file_ttf(font, size,
                glyph_ranges=handle.get_glyph_ranges_chinese_full()) for size in font_sizes
        }

        self._context_lock = mp.Lock()
        self._context_tid = None # id of thread in critical section

    def get_default_font(self):
        return str(Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf')
    
    def push_context(self):
        if has_pycuda:
            self._cuda_context.push()
    
    def pop_context(self):
        if has_pycuda:
            self._cuda_context.pop()

    @contextmanager
    def lock(self, strict=True):
        # Prevent double locks, e.g. when
        # calling upload_image() from UI thread
        tid = get_ident()
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

    def set_interp_linear(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.tex_interp_mode = gl.GL_LINEAR

    def set_interp_nearest(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        self.tex_interp_mode = gl.GL_NEAREST

    def editable(self, name, **kwargs):
        if name not in self._editables:
            self._editables[name] = _editable(name)
        self._editables[name].run(**kwargs)

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

        c0 = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        c1 = s.colors[imgui.COLOR_FRAME_BACKGROUND]
        s.colors[imgui.COLOR_POPUP_BACKGROUND] = [x * 0.7 + y * 0.3 for x, y in zip(c0, c1)][:3] + [1]

    def start(self, loopfunc, workers = (), glfw_init_callback = None):
        # allow single thread object
        if not hasattr(workers, '__len__'):
            workers = (workers,)

        for i in range(len(workers)):
            workers[i].start()

        self.set_ui_scale(self.ui_scale)        
        
        with self.lock():
            impl = GlfwRenderer(self._window)
        
        self._pressed_keys = set()
        self._hit_keys = set()

        def on_key(window, key, scan, pressed, mods):
            if pressed:
                if key not in self._pressed_keys:
                    self._hit_keys.add(key)
                self._pressed_keys.add(key)
            else:
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key) # check seems to be needed over RDP sometimes
            if key != glfw.KEY_ESCAPE: # imgui erases text with escape (??)
                impl.keyboard_callback(window, key, scan, pressed, mods)

        glfw.set_key_callback(self._window, on_key)

        # For settings custom callbacks etc.
        if glfw_init_callback is not None:
            glfw_init_callback(self._window)

        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            impl.process_inputs()

            if self.keyhit(glfw.KEY_ESCAPE):
                glfw.set_window_should_close(self._window, 1)

            with self.lock(strict=False) as l:
                if l == nullcontext:
                    continue

                # Breaks on MacOS. Needed?
                #imgui.get_io().display_size = glfw.get_framebuffer_size(self._window)
                
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

                impl.render(imgui.get_draw_data())
                
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
            
        glfw.make_context_current(self._window)
        del self._images
        self._images: Dict[str, _texture] = {}
        glfw.make_context_current(None)

        glfw.destroy_window(self._window)
        self.pop_context()

    def upload_image(self, name, data):
        if has_torch and torch.is_tensor(data):
            if data.device.type in ['mps', 'cpu']:
                # would require gl-metal interop or metal UI backend
                return self.upload_image_np(name, data.cpu().numpy())
            else:
                return self.upload_image_torch(name, data)
        else:
            return self.upload_image_np(name, data)

    # Upload image from PyTorch tensor
    def upload_image_torch(self, name, tensor):
        assert isinstance(tensor, torch.Tensor)
        tensor = normalize_image_data(tensor, 'uint8')

        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            cuda_synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_torch(tensor)
                self.pop_context()

    def upload_image_np(self, name, data):
        assert isinstance(data, np.ndarray)
        data = normalize_image_data(data, 'uint8')

        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            cuda_synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_np(data)
                self.pop_context()
