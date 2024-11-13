import numpy as np
import imgui
import contextlib
from io import BytesIO
from pathlib import Path
from functools import partial
import shutil
import os
import glfw
import random
import string
import time
from textwrap import dedent
from functools import wraps
from platform import system

import OpenGL.GL as gl
import ctypes

# ImGui widget that wraps arbitrary object and allows mouse pand & zoom controls.
# Does not transform texture coordinates; instead transforms a single textured quad.
# Panning produces no temporal aliasing: the pan follows the mouse, which moves in one pixel increments.
class PannableArea():
    def __init__(self, set_callbacks=False, glfw_window=False) -> None:  # draw_content: callable, 
        self.prev_cbk: callable = lambda : None  # for chaining
        self.output_pos_tl = np.zeros(2, dtype=np.float32)
        self.id = ''.join(random.choices(string.ascii_letters, k=20))
        self.is_panning = False # currently panning?
        self.pan_enabled = True
        self.zoom_enabled = True
        # Pan magnitude: image edges at +-0.5 at unit scale
        self.pan = (0, 0)
        self.pan_delta = (0, 0)
        self.pan_start = (0, 0)
        self.zoom: float = 1.0
        self.clear_color = (0, 0, 0, 1) # in [0, 1]

        # Canvas onto which resmapled image is drawn
        self.canvas_tex = None
        self.canvas_fb = None
        self.canvas_w = 0
        self.canvas_h = 0
        
        # Size of image on screen (not texture dims)
        self.tex_h = 0
        self.tex_w = 0

        if set_callbacks:
            assert glfw_window, 'Must provide glfw window for callback setting'
            self.set_callbacks(glfw_window)

        # Precopute a few transformations
        center = np.array([
            2, 0, -1,
            0, 2, -1,
            0, 0,  1,
        ], dtype=np.float32).reshape(3, 3)
        self.uv_to_ndc = np.diag([1, -1, 1]) @ center
        self.ndc_to_uv = np.linalg.inv(self.uv_to_ndc)

    def resize_canvas(self, W, H):
        if self.canvas_w == W and self.canvas_h == H:
            return
        
        if H <= 0 or W <= 0:
            return # minimized
        
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
        # Canvas always rendered at native scale, interp. should be irrelevant
        self.canvas_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.canvas_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
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
                v_texcoord = texcoord; // texel coords untouched
                vec3 pos = xform * vec3(position, 1.0);
                gl_Position = vec4(pos.xy, 0.0, 1.0); // quad itself transformed
            }
            """
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
            }
            """
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

    # tW, tH = input texture size
    # cW, cH = canvas size
    def draw_to_canvas(self, texture_in, tW, tH, cW, cH, pan_enabled=True):
        last_texture = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        last_array_buffer = gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)
        last_vertex_array = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        last_framebuffer = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
        last_clear_color = gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE)
        last_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

        if self.canvas_fb is None:
            self.init_gl(cW, cH)

        # Reallocate if window size has changed
        self.resize_canvas(cW, cH)

        # Keep track of content position
        # Cannot use `get_item_rect_min`, since imgui.image hasn't been drawn yet
        rmin = imgui.get_window_content_region_min()
        wmin = imgui.get_window_position()
        self.output_pos_tl[:] = (wmin.x + rmin.x, wmin.y + rmin.y)
        self.tex_w = tW
        self.tex_h = tH
        self.pan_enabled = pan_enabled

        # Update pan & zoom state
        if self.pan_enabled:
            self.handle_pan()

        #gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glClearColor(*self.clear_color)
        
        # Transform textured quad based on pan and zoom
        # GL expects [..., tX, tY, 1]
        xform = self.get_quad_transform()
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
    
    def get_transform(self):
        """
        Get current image transform.
        Standard mathematical conventions: x-axis right, y-axis up.
        Origin at (0, 0), unit scale translation (edge to edge has magnitude 1.0f).
        """
        M = np.eye(3, dtype=np.float32)
        M[0, 2] = (self.pan[0]+self.pan_delta[0])
        M[1, 2] = (self.pan[1]+self.pan_delta[1])
        M = np.diag((self.zoom, self.zoom, 1)) @ M # zoom relative to viewport center (post-translation)
        return M

    def get_transform_ndc(self):
        """
        Get current image transform.
        Standard mathematical conventions: x-axis right, y-axis up.
        Origin at (0, 0), ndc scale translation (edge to edge has magnitude 2.0f).
        """
        M = np.eye(3, dtype=np.float32)
        M[0, 2] = (self.pan[0]+self.pan_delta[0])*2
        M[1, 2] = (self.pan[1]+self.pan_delta[1])*2
        M = np.diag((self.zoom, self.zoom, 1)) @ M # zoom relative to viewport center (post-translation)
        return M
    
    def get_transform_scaled(self, W, H):
        """
        Get current image transform.
        Origin at (W/2, H/2), should be applied to coordinates in [0, W]x[0, H]
        """
        M = self.get_transform_ndc()

        # NDC to absolute (inputs in [O,W]x[0,H])
        S = np.array([
            W/2,   0, W/2,
              0, H/2, H/2,
              0,   0,   1,
        ]).astype(np.float32).reshape(3, 3)
        M = S @ M @ np.linalg.inv(S)

        return M
    
    def get_transform_01(self):
        """
        Get current image transform.
        Origin at (0.5, 0.5), should be applied to coordinates in [0, 1]
        """
        return self.get_transform_scaled(1, 1)
    
    def get_quad_scale_corr(self):
        """Get quad vs. image aspect ratio scale correction"""
        # Image size is computed to fill smaller canvas dimension
        # Account for this scaling to prevent image stretching
        # => squish quad to same aspect ratio as image
        if self.tex_h == 0 or self.tex_h == 0:
            return np.diag([1, 1, 1])
        aspect = self.tex_w / self.tex_h
        out_width = min(self.canvas_w, aspect*self.canvas_h)
        final_size = (out_width, out_width / aspect)
        corr = np.diag([final_size[0]/self.canvas_w, final_size[1]/self.canvas_h, 1])
        return corr

    def get_quad_transform(self, flip_y=True):
        """
        Get the matrix that transforms the textured quad before rendering.
        Takes content and window aspect ratios into account.
        Quad is in ndc space when this transformation is applied.
        """
        # Get current pan/zoom xform
        # Normal mathematical conventions (y-up)
        xform = self.get_transform_ndc() if self.pan_enabled else np.eye(3)
        
        # Flipping: not for fixing texture access; want to flip whole y axis (quad position included)
        flip = np.diag([1, -1, 1]) if flip_y else np.eye(3)

        corr = self.get_quad_scale_corr()
        xform = flip @ xform @ corr
        
        return xform
    
    # Set transform state from np matrix
    # Assuming M contains only zoom & translation
    def set_transform(self, M, W=1, H=1):
        #assert not self.is_panning
        assert M.shape == (3, 3)
        self.zoom = np.sqrt(M[0, 0] * M[1, 1])
        self.pan = (
            M[0, 2] / W / self.zoom,
            M[1, 2] / H / self.zoom,
        )

    def get_visible_box_canvas(self):
        """
        Returns top-left and bottom-right coords of currently visible canvas (!= image) region.
        Full untransformed canvas matches the unit box [0, 1]^2.
        The origin of the uv-space is in the top-left.
        """
        box = np.array([
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]).reshape(1, 2, 3)
        M = np.linalg.inv(self.get_transform_01()) # scaled for inputs in [0, 1]
        box = (box @ M)[0, 0:2, 0:2] # TODO: missing transpose, superfluous inverse?
        return (box[0], box[1]) # tl, br
    
    def get_visible_box_image(self):
        """
        Returns top-left and bottom-right coords of currently visible image (!= canvas) region.
        Full untransformed image matches the unit box [0, 1]^2.
        The origin of the uv-space is in the top-left.
        """
        # Spaces:
        # uv: x right, y down, range [0, 1]
        # ndc: x right, y up, range [-1, 1] (OpenGL style)

        # Image's uv range [0, 1] (y flipped) to ncd range [-1, 1]

        # Transform that transforms quad (i.e. image)
        # Y flipping disabled => normal y-up transformation
        ndc_to_viewport = self.get_quad_transform(flip_y=False)
        
        # Visible box: full ndc viewport 
        # tl in ndc is (-1, 1)
        # br in ndc is (1, -1)
        box_viewport = np.array([
            -1,  1,
             1, -1,
             1,  1
        ], dtype=np.float32).reshape(3, 2) # 2 column vectors
        
        # Invert transformations to get viewport box in image's uv space
        uv_to_viewport = ndc_to_viewport @ self.uv_to_ndc
        viewport_to_uv = np.linalg.inv(uv_to_viewport)
        box_uv = viewport_to_uv @ box_viewport # column vectors
        
        # Real quad coords (not visible part)
        TL, BR = (box_uv[:2, 0], box_uv[:2, 1])
        #slow_print(f'x: [{TL[0]:.2f} -> {BR[0]:.2f}], y: [{TL[1]:.2f} -> {BR[1]:.2f}]')
        
        return (TL, BR)
    
    def uv_to_screen_xform(self):
        """
        Matrix that transforms image uvs to absolute screen positions (after pan and zoom).
        Both spaces have TL origin, y axis down.
        """        
        
        # Step 1: map from uv space to ndc space

        # Step 2: apply aspect ratio and user transform
        xform = self.get_quad_transform(flip_y=False)

        # Step 3: back to uv space

        # Step 4: uv to screen pos
        W, H = self.canvas_w, self.canvas_h
        px, py = self.output_pos_tl
        uv_to_screen = np.array([
            W, 0, px,
            0, H, py,
            0, 0,  1,
        ]).reshape(3, 3)

        return (uv_to_screen @ self.ndc_to_uv @ xform @ self.uv_to_ndc)

    def screen_to_uv_xform(self):
        """Matrix that transforms absolute screen positions (e.g. mouse pos) to image uvs."""
        return np.linalg.inv(self.uv_to_screen_xform)

    def get_hovered_uv_canvas(self):
        """
        UVs of currently hovered canvas point, relative to top-left.
        Takes transformation into account.
        """
        tl, br = self.get_visible_box_canvas() # coords in [0, 1]^2
        xy = np.array(self.mouse_pos_canvas_norm)
        u, v = (tl * (1 - xy) + br * xy).tolist()
        return (u, v)
    
    def get_hovered_uv_image(self):
        """
        UVs of currently hovered image point, relative to top-left texel.
        Takes current transformation into account, can be outside of [0, 1].
        """
        tl, br = self.get_visible_box_image() # uv's in [0, 1]
        xy = np.array(self.mouse_pos_canvas_norm)
        u, v = (tl * (1 - xy) + br * xy).tolist()
        return (u, v)
    
    def reset_xform(self):
        self.pan = self.pan_start = self.pan_delta = (0, 0)
        self.zoom = 1.0
        self.is_panning = False
    
    # Handle pan action
    def handle_pan(self):
        if imgui.is_mouse_clicked(0) and self.mouse_hovers_content():
            u, v = self.get_hovered_uv_canvas()
            self.pan_start = (u, 1 - v) # convert to y-up
            self.is_panning = True
        if self.is_panning and imgui.is_mouse_down(0):
            u, v = self.get_hovered_uv_canvas()
            self.pan_delta = (u - self.pan_start[0], (1 - v) - self.pan_start[1]) # convert to y-up
        if self.is_panning and imgui.is_mouse_released(0):
            self.pan = tuple(s+d for s,d in zip(self.pan, self.pan_delta))
            self.pan_start = self.pan_delta = (0, 0)
            self.is_panning = False
        if imgui.is_mouse_double_clicked(0) and self.mouse_hovers_content(): # reset view
            self.reset_xform()
    
    @property
    def mouse_pos_abs(self):
        return np.array(imgui.get_mouse_pos())

    @property
    def mouse_pos_canvas_norm(self):
        """
        Normalized mouse position on canvas, in [0, 1] relative to top-left.
        Does not consider current transformation.
        """
        dims = np.array((self.canvas_w, self.canvas_h))
        if any(dims == 0):
            return np.array([-1, -1], dtype=np.float32) # no valid content
        return (self.mouse_pos_abs - self.output_pos_tl) / dims

    def mouse_hovers_content(self):
        x, y = self.mouse_pos_canvas_norm
        return (0 <= x <= 1) and (0 <= y <= 1)
    
    def mouse_wheel_callback(self, window, x, y) -> None:
        if not (self.mouse_hovers_content() and self.zoom_enabled):
            return self.prev_cbk(window, x, y) # scroll imgui lists etc.
        
        # MacOS trackpad needs separate speed
        speed = 0.035 if system() == 'Darwin' else 0.15
        scale_fill_h = max(1, (self.tex_w / self.tex_h) * (self.canvas_h / self.canvas_w))
        zoom_max = 0.5 * scale_fill_h * self.tex_h # canvas height convers 2 pixels
        zoom_min = 50 / min(self.canvas_h, self.canvas_w) # canvas ~50x50 pixels
        new_zoom = min(zoom_max, max(zoom_min, self.zoom * (1.0 + y * speed)))

        # Zoom relative to mouse cursor
        # => keep same UV under cursor after zooming
        center_uv = 0.5 * sum(self.get_visible_box_canvas())
        mouse_pos = np.array(self.get_hovered_uv_canvas())
        dt = mouse_pos - center_uv # center to mouse cursor
        dm = dt * ((self.zoom / new_zoom) - 1) # movement of cursor UV due to zoom
        self.pan = np.array(self.pan) + dm * np.array([1, -1]) # to y up
        self.zoom = new_zoom

# Dataclass that enforces type annotation
# Enables compare-by-value
def strict_dataclass(cls=None, *args, ignore_attr=(), **kwargs):
    def wrap(cls):
        # Needed if subclassing a strict_dataclass
        new = [ignore_attr] if isinstance(ignore_attr, str) else list(ignore_attr)
        ignore = getattr(cls, '__ignore_attr__', set()).union(new)
        setattr(cls, '__ignore_attr__', ignore)
        
        annotations = cls.__dict__.get('__annotations__', {})
        for name in dir(cls):
            if name.startswith('__'):
                continue
            if name in ignore:
                continue # for functions etc.
            if name not in annotations:
                raise RuntimeError(f'Unannotated field: {name}')
    
        # Write updated
        setattr(cls, '__annotations__', annotations)
            
        from dataclasses import dataclass
        return dataclass(cls, *args, **kwargs)

    # See if we're being called as @strict_dataclass or @strict_dataclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @strict_dataclass without parens.
    return wrap(cls)

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
# Idx needed if there are duplicates in the allowed values
def combo_box_vals(title, values, current=None, idx=None, height_in_items=-1, to_str: callable = str):
    values = list(values)
    idx = values.index(current) if idx is None else idx
    changed, new_idx = imgui.combo(title, idx, [to_str(v) for v in values], height_in_items)
    return changed, (values[new_idx], new_idx)

# Slider that cycles through predefined values
# Abusing format to draw current enum value onto slider
def enum_slider(title, values, current, to_str: callable = str):
    values = list(values)
    curr_idx = 0 if current not in values else values.index(current)
    changed, idx = imgui.slider_int(title, curr_idx, 0, len(values)-1, format=to_str(current))
    return changed, values[idx]

# Imgui slider that can switch between int and float formatting at runtime
def slider_dynamic(title, v, min, max, width=0.0):
    scale_fmt = '%.2f' if np.modf(v)[0] > 0 else '%.0f' # dynamically change from ints to floats
    with imgui_item_width(width):
        return imgui.slider_float(title, v, min, max, format=scale_fmt)

# Int2 slider that prevents overlap
def slider_range_int(v1, v2, vmin, vmax, push=False, title='', width=0.0):
    with imgui_item_width(width):
        ch, (s, e) = imgui.slider_int2(title, v1, v2, vmin, vmax)

    if push:
        return ch, (min(s, e), max(s, e))
    elif s != v1:
        return ch, (min(s, e), e)
    elif e != v2:
        return ch, (s, max(s, e))
    else:
        return ch, (s, e)
    
# Float2 slider that prevents overlap
def slider_range_float(v1, v2, vmin, vmax, push=False, title='', width=0.0):
    with imgui_item_width(width):
        ch, (s, e) = imgui.slider_float2(title, v1, v2, vmin, vmax)

    if push:
        return ch, (min(s, e), max(s, e))
    elif s != v1:
        return ch, (min(s, e), e)
    elif e != v2:
        return ch, (s, max(s, e))
    else:
        return ch, (s, e)

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

def reshape_grid(img_nchw):
    return reshape_grid_np(img_nchw) if isinstance(img_nchw, np.ndarray) else reshape_grid_torch(img_nchw)

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
def copy_with_progress(pth_from: Path, pth_to: Path):
    os.makedirs(pth_to.parent, exist_ok=True)
    return shutil.copyfileobj(open_prog(pth_from, 'rb', pth_to.name), open(pth_to, 'ab'))

def open_prog(pth, mode, name=None, progress_cbk=None):
    """
    File open with progress bar, for slow network drives etc.
    Supports context manager.
    progress_cbk: callback with interface def()
    """
    from tqdm import tqdm
    
    assert mode in ['r', 'rb'], 'Only r and rb supported'
    total_size = int(os.path.getsize(pth))
    label = Path(pth).name if name is None else name
    pbar = tqdm(ncols=80, total=total_size, bar_format=f'{label} {{l_bar}}{{bar}}| {{elapsed}}<{{remaining}}')
    handle = open(pth, mode)

    def update(size):
        pbar.update(size)
        if progress_cbk is not None:
            progress_cbk(pbar.n, total_size)
        if pbar.n == pbar.total:
            pbar.refresh()
            pbar.close()

    def read(size, orig=None):
        update(size)
        return orig(size)
    handle.read = partial(read, orig=handle.read)

    def readinto(memory, orig=None):
        update(memory.nbytes)
        return orig(memory)
    handle.readinto = partial(readinto, orig=handle.readinto)

    def close(orig=None):
        update(pbar.total - pbar.n) # set to 100%
        return orig()
    handle.close = partial(close, orig=handle.close)
    
    return handle

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

def rate_limit(T, default=None, delay=0):
    """
    Do something (expensive) at most every T seconds.
    Args:
      T: minimum interval between evaluations, in seconds.
      default: default return value in case function wasn't called.
      delay: initial delay, in seconds.
    """
    def wrapper(f):
        @wraps(f)
        def aux(*args, **kwargs):
            if time.monotonic() - aux.latest >= T:
                retval = f(*args, **kwargs)
                aux.latest = time.monotonic()
                return retval
            else:
                return default
        aux.latest = time.monotonic() - T + delay
        return aux
    return wrapper

@rate_limit(T=0.5)
def slow_print(*args, **kwargs):
    print(*args, **kwargs)

def lazy_print(s: str):
    if s != getattr(lazy_print, "prev", None):
        print(s)
        setattr(lazy_print, "prev", s)

def resolve_lnk(p: Path):
    """
    Resolve Windows path containing shortcuts (.lnk).
    Supports three cases:
    1. Target is a .lnk
    2. Path to target contains a .lnk
    3. Both of the above
    """
    if system() != 'Windows':
        return p

    p = Path(p).absolute()
    if p.exists() and p.suffix != '.lnk':
        return p

    root, *parents = [p, *p.parents][::-1]
    
    for p in parents:
        root = root / p.name
        lnk = root.with_suffix('.lnk')
        if root.exists() and root.suffix != '.lnk':
            continue
        elif lnk.is_file():
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            root = Path(shell.CreateShortCut(str(lnk)).Targetpath)
        elif root.is_symlink():
            raise RuntimeError('Unhandled link type')
    
    return root

import inspect
def __LINE__():
    return str(inspect.currentframe().f_back.f_lineno)