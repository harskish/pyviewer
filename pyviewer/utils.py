import numpy as np
import imgui
import contextlib
from io import BytesIO
from pathlib import Path
import os
import glfw
import random
import string
from textwrap import dedent

import OpenGL.GL as gl
import ctypes

# ImGui widget that wraps arbitrary object
# and allows mouse pand & zoom controls
class PannableArea():
    def __init__(self, set_callbacks=False, glfw_window=False) -> None:  # draw_content: callable, 
        self.prev_cbk: callable = lambda : None  # for chaining
        self.output_pos_tl = np.zeros(2, dtype=np.float32)
        self.id = ''.join(random.choices(string.ascii_letters, k=20))
        self.is_panning = False
        self.pan = (0, 0)
        self.pan_start = (0, 0)
        self.pan_delta = (0, 0)
        self.zoom: float = 1.0

        # Canvas onto which resmapled image is drawn
        self.canvas_tex = None
        self.canvas_fb = None
        self.canvas_w = 0
        self.canvas_h = 0

        if set_callbacks:
            assert glfw_window, 'Must provide glfw window for callback setting'
            self.set_callbacks(glfw_window)

    def resize_canvas(self, W, H):
        if self.canvas_w == W and self.canvas_h == H:
            return
        
        #print(f'PannableArea: resizing to {W}x{H}')
        self.canvas_w = W
        self.canvas_h = H

        last_texture = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.canvas_tex)

        # Reallocate, id stays the same
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,     # GLenum target
            0,                    # GLint level
            gl.GL_RGBA,           # GLint internalformat
            W,                    # GLsizei width
            H,                    # GLsizei height
            0,                    # GLint border
            gl.GL_RGBA,           # GLenum format
            gl.GL_UNSIGNED_BYTE,  # GLenum type
            None                  # const void * data
        )

        # Restore state
        gl.glBindTexture(gl.GL_TEXTURE_2D, last_texture)

    def init_gl(self, canvas_width, canvas_height):
        self.canvas_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.canvas_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.resize_canvas(canvas_width, canvas_height)

        # Framebuffer for offscreen rendering
        self.canvas_fb = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.canvas_fb)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, self.canvas_tex, 0)
        gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Could not create framebuffer for offscreen rendering')

        self._shader_handle = gl.glCreateProgram()
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        
        gl.glShaderSource(vertex_shader, dedent(
            """
            #version 330
            uniform mat3 xform;
            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 texcoord;
            out vec2 v_texcoord;

            void main()
            {
                v_texcoord = texcoord;
                vec3 posxy = xform * vec3(position, 1.0);
                gl_Position = vec4(posxy.xy, 0.0, 1.0);
            }"""
        ))

        gl.glShaderSource(fragment_shader, dedent(
            """
            #version 330
            uniform sampler2D tex;
            in vec2 v_texcoord;
            out vec4 color;

            void main()
            {
                color = texture(tex, v_texcoord);
            }"""
        ))

        for prog in [vertex_shader, fragment_shader]:
            gl.glCompileShader(prog)
            log = gl.glGetShaderInfoLog(prog)
            if log:
                print(log)

        gl.glAttachShader(self._shader_handle, vertex_shader)
        gl.glAttachShader(self._shader_handle, fragment_shader)

        gl.glLinkProgram(self._shader_handle)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        self._attrib_location_pos = gl.glGetAttribLocation(self._shader_handle, "position")
        self._attrib_location_texcoord = gl.glGetAttribLocation(self._shader_handle, "texcoord")
        self._attrib_location_tex = gl.glGetUniformLocation(self._shader_handle, "tex")
        self._attrib_location_xform = gl.glGetUniformLocation(self._shader_handle, "xform")

        self._vao_handle = gl.glGenVertexArrays(1)  # bundles one or more VBOs
        self._vbo_handle = gl.glGenBuffers(1)       # per-vertex information to be interpolated (bound as GL_ARRAY_BUFFER)
        self._elements_handle = gl.glGenBuffers(1)  # buffer of indices into vbo (bound as GL_ELEMENT_ARRAY_BUFFER)
        
        gl.glBindVertexArray(self._vao_handle)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_handle)
        gl.glEnableVertexAttribArray(self._attrib_location_pos)
        gl.glEnableVertexAttribArray(self._attrib_location_texcoord)

        size_float = 4
        vertex_size = 4 * size_float # 2*pos + 2*uv
        gl.glVertexAttribPointer(self._attrib_location_pos, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_size, ctypes.c_void_p(0*size_float))
        gl.glVertexAttribPointer(self._attrib_location_texcoord, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_size, ctypes.c_void_p(2*size_float))

        # Two static tris that form a quad
        # OpenGL 3.3: just stick to glBufferData
        # OpenGL 4.5+: could use glNamedBufferData (dynamic) or glNamedBufferStorage (static)

        # Vertex data:            strip         position           UV coord
        vertices = np.array([  #  order    (-1, +1)   (+1, +1)   (0,0)  (1,0)
            -1, +1, 0, 0,      #  1----3        +-------+           +----+
            -1, -1, 0, 1,      #  |  / |        |  NDC  |           |    |
            +1, +1, 1, 0,      #  | /  |        | SPACE |           |    |
            +1, -1, 1, 1,      #  2----4        +-------+           +----+
        ], dtype=np.float32)   #           (-1, -1)   (+1, -1)   (0,1)  (1,1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_handle)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 4 * vertex_size, vertices, gl.GL_STATIC_DRAW)

    def draw_to_canvas(self, texture_in, W, H):
        last_texture = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        last_array_buffer = gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)
        last_vertex_array = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_framebuffer = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
        last_clear_color = gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE)
        last_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

        if self.canvas_fb is None:
            self.init_gl(W, H)

        # Reallocate if window size has changed
        self.resize_canvas(W, H)

        # Update pan & zoom state
        self.handle_pan()

        #gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glClearColor(0, 0, 0, 1)

        xform = self.get_transform(1, 1)
        xform[1, 1] *= -1  # flip y
        xform[0:2, 2] *= 2 # adapt to larger [-1, 1] NDC range
        xform = np.transpose(xform)

        gl.glUseProgram(self._shader_handle)
        gl.glUniform1i(self._attrib_location_tex, 0) # slot 0
        gl.glUniformMatrix3fv(self._attrib_location_xform, 1, gl.GL_FALSE, xform)
        gl.glBindVertexArray(self._vao_handle)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.canvas_fb)

        # Run shader
        gl.glViewport(0, 0, self.canvas_w, self.canvas_h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_in) # slot 0
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        # restore state
        gl.glBindTexture(gl.GL_TEXTURE_2D, last_texture)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, last_array_buffer)
        gl.glBindVertexArray(last_vertex_array)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, last_framebuffer)
        gl.glClearColor(*last_clear_color)
        gl.glViewport(*last_viewport)

        return self.canvas_tex

    def set_callbacks(self, glfw_window):
        self.prev_cbk = glfw.set_scroll_callback(glfw_window, self.mouse_wheel_callback)

    def get_transform(self, W, H):        
        M = np.eye(3, dtype=np.float32)
        M[0, 2] += (self.pan[0]+self.pan_delta[0])*W
        M[1, 2] += (self.pan[1]+self.pan_delta[1])*H
        M *= self.zoom
        return M

    # Handle pan action
    def handle_pan(self):
        self.output_pos_tl[:] = imgui.get_item_rect_min()
        xy = np.array(self.mouse_pos_img_norm)
        
        # Figure out what part of image is currently visible
        box = np.array([
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]).reshape(1, 2, 3)
        M = np.linalg.inv(self.get_transform(1, 1))
        box = (box @ M)[0, 0:2, 0:2]
        a, b = (box[0] * (1 - xy) + box[1] * xy).tolist()

        if imgui.is_mouse_clicked(0) and self.mouse_hovers_content():
            self.is_panning = True
            self.pan_start = (a, b)
        if self.is_panning and imgui.is_mouse_down(0):
            self.pan_delta = (a - self.pan_start[0], b - self.pan_start[1])
        if self.is_panning and imgui.is_mouse_released(0):
            self.pan = tuple(s+d for s,d in zip(self.pan, self.pan_delta))
            self.pan_start = self.pan_delta = (0, 0)
            self.is_panning = False
        if imgui.is_mouse_double_clicked(0) and self.mouse_hovers_content():  # Reset view
            self.pan = self.pan_start = self.pan_delta = (0, 0)
            self.zoom = 1.0
            self.is_panning = False
    
    @property
    def mouse_pos_abs(self):
        return np.array(imgui.get_mouse_pos())

    @property
    def mouse_pos_img_norm(self):
        dims = np.array((self.canvas_w, self.canvas_h))
        if any(dims == 0):
            return np.array([-1, -1], dtype=np.float32) # no valid content
        return (self.mouse_pos_abs - self.output_pos_tl) / dims

    def mouse_hovers_content(self):
        x, y = self.mouse_pos_img_norm
        return (0 <= x <= 1) and (0 <= y <= 1)
    
    def mouse_wheel_callback(self, window, x, y) -> None:
        if self.mouse_hovers_content():
            self.zoom = max(1e-2, (0.85**np.sign(-y)) * self.zoom)
        else:
            self.prev_cbk(window, x, y) # scroll imgui lists etc.

# Dataclass that enforces type annotation
# Enables compare-by-value
def strict_dataclass(cls, *args, **kwargs):
    annotations = cls.__dict__.get('__annotations__', {})
    for name in dir(cls):
        if name.startswith('__'):
            continue
        if name not in annotations:
            raise RuntimeError(f'Unannotated field: {name}')
    
    # Write updated
    setattr(cls, '__annotations__', annotations)
        
    from dataclasses import dataclass
    return dataclass(cls, *args, **kwargs)

# with-block for item id
@contextlib.contextmanager
def imgui_id(id: str):
    imgui.push_id(id)
    yield
    imgui.pop_id()

# with-block for item width
@contextlib.contextmanager
def imgui_item_width(size):
    imgui.push_item_width(size)
    yield
    imgui.pop_item_width()

# Full screen imgui window
def begin_inline(name):
    with imgui.styled(imgui.STYLE_WINDOW_ROUNDING, 0):
        imgui.begin(name,
            flags = \
                imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_SAVED_SETTINGS
        )

# Recursive getattr
def rgetattr(obj, key, default=None):
    head = obj
    while '.' in key:
        bot, key = key.split('.', maxsplit=1)
        head = getattr(head, bot, {})
    return getattr(head, key, default)

# Combo box that returns value, not index
def combo_box_vals(title, values, current, height_in_items=-1, to_str=str):
    values = list(values)
    curr_idx = 0 if current not in values else values.index(current)
    changed, ind = imgui.combo(title, curr_idx, [to_str(v) for v in values], height_in_items)
    return changed, values[ind]

# Imgui slider that can switch between int and float formatting at runtime
def slider_dynamic(title, v, min, max, width=0.0):
    scale_fmt = '%.2f' if np.modf(v)[0] > 0 else '%.0f' # dynamically change from ints to floats
    with imgui_item_width(width):
        return imgui.slider_float(title, v, min, max, format=scale_fmt)

# Int2 slider that prevents overlap
def slider_range(v1, v2, vmin, vmax, push=False, title='', width=0.0):
    with imgui_item_width(width):
        s, e = imgui.slider_int2(title, v1, v2, vmin, vmax)[1]

    if push:
        return (min(s, e), max(s, e))
    elif s != v1:
        return (min(s, e), e)
    elif e != v2:
        return (s, max(s, e))
    else:
        return (s, e)

# Shape batch as square if possible
def get_grid_dims(B):
    if B == 0:
        return (0, 0)
    
    S = int(B**0.5 + 0.5)
    while B % S != 0:
        S -= 1
    return (B // S, S) # (W, H)

def reshape_grid_np(img_batch):
    if isinstance(img_batch, list):
        img_batch = np.concatenate(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = np.reshape(img_batch, [rows, cols, C, H, W])
    img_batch = np.transpose(img_batch, [0, 3, 1, 4, 2])
    img_batch = np.reshape(img_batch, [rows * H, cols * W, C])

    return img_batch

def reshape_grid_torch(img_batch):
    import torch
    if isinstance(img_batch, list):
        img_batch = torch.cat(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = img_batch.reshape(rows, cols, C, H, W)
    img_batch = img_batch.permute(0, 3, 1, 4, 2)
    img_batch = img_batch.reshape(rows * H, cols * W, C)

    return img_batch

def reshape_grid(batch):
    return reshape_grid_np(batch) if isinstance(batch, np.ndarray) else reshape_grid_torch(batch)

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_latent(B, n_dims=512, seed=None):
    seeds = sample_seeds(B, base=seed)
    return seeds_to_latents(seeds, n_dims)

def seeds_to_latents(seeds, n_dims=512):
    latents = np.zeros((len(seeds), n_dims), dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(n_dims)
    
    return latents

# File copy with progress bar
# For slow network drives etc.
def copy_with_progress(pth_from, pth_to):
    from tqdm import tqdm
    os.makedirs(pth_to.parent, exist_ok=True)
    size = int(os.path.getsize(pth_from))
    fin = open(pth_from, 'rb')
    fout = open(pth_to, 'ab')

    try:
        with tqdm(ncols=80, total=size, bar_format=pth_from.name + ' {l_bar}{bar} | Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.close()

# File open with progress bar
# For slow network drives etc.
# Supports context manager
def open_prog(pth, mode):
    from tqdm import tqdm
    size = int(os.path.getsize(pth))
    fin = open(pth, 'rb')

    assert mode == 'rb', 'Only rb supported'
    fout = BytesIO()

    try:
        with tqdm(ncols=80, total=size, bar_format=Path(pth).name + ' {l_bar}{bar}| Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.seek(0)

    return fout

# Convert input image to valid range for showing
# Output converted to target dtype *after* scaling
#   => should not affect quality that much
def normalize_image_data(img_hwc, target_dtype='uint8'):    
    is_np = isinstance(img_hwc, np.ndarray)
    is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
    
    # Valid ranges for RGB data
    maxval = 1 if is_fp else 255
    minval = 0
    
    # If outside of range: normalize to [0, 1]
    if img_hwc.max() > maxval or img_hwc.min() < minval:
        img_hwc = img_hwc.astype(np.float32) if is_np else img_hwc.float()
        img_hwc -= img_hwc.min() # min is negative
        img_hwc /= img_hwc.max()
        is_fp = True
        maxval = 1
    
    # At this point, data will be:
    #  i) fp32, in [0, 1]
    # ii) uint8, in [0, 255]

    # Convert to target dtype
    if target_dtype == 'uint8':
        img_hwc = img_hwc * 255 if is_fp else img_hwc
        img_hwc = np.uint8(img_hwc) if is_np else img_hwc.byte()
    else:
        img_hwc = img_hwc.astype(np.float32) if is_np else img_hwc.float()
        img_hwc = img_hwc / maxval

    # (H, W) to (H, W, 1)
    if img_hwc.ndim == 2:
        img_hwc = img_hwc[..., None]

    # Use at most 3 channels
    img_hwc = img_hwc[:, :, :3]

    return img_hwc