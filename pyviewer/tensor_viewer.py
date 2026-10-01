"""A debugger-friendly, threaded viewer for tensors larger than GPU textures.

Only a strided view of the visible slice is materialized. The source is retained
by reference; callers must coordinate writes if they need a consistent frame.
"""

from dataclasses import dataclass
from math import ceil, floor
from threading import Event, Lock, Thread, current_thread

import numpy as np
from imgui_bundle import imgui
import glfw
import OpenGL.GL as gl

from . import egl_patch, gl_viewer
from .utils import PannableArea, begin_inline


@dataclass(frozen=True)
class SampleRegion:
    y: slice
    x: slice
    # Source pixel edges occupied by the sampled texture (x0, y0, x1, y1).
    bounds: tuple[int, int, int, int]


def sample_region(shape, visible_box, resolution):
    """Plan positive-stride slices, bounded by output resolution in each axis.

    visible_box is an unclipped (top-left, bottom-right) pair in image UVs.
    The regular integer stride approximates nearest sampling when zoomed out;
    zooming in uses stride one. No operation depends on source storage size.
    """
    height, width = shape
    out_w, out_h = resolution
    if min(height, width, out_w, out_h) <= 0:
        return None
    tl, br = visible_box
    x0 = max(0, min(width, floor(float(tl[0]) * width)))
    y0 = max(0, min(height, floor(float(tl[1]) * height)))
    x1 = max(0, min(width, ceil(float(br[0]) * width)))
    y1 = max(0, min(height, ceil(float(br[1]) * height)))
    if x1 <= x0 or y1 <= y0:
        return None
    # Use the full visible span, including space outside the image, so sampling
    # density follows screen pixels even when most of the canvas is empty.
    sx = max(1, ceil((float(br[0]) - float(tl[0])) * width / out_w))
    sy = max(1, ceil((float(br[1]) - float(tl[1])) * height / out_h))
    # Anchor the lattice at the image origin to reduce shimmer while panning.
    x0, y0 = (x0 // sx) * sx, (y0 // sy) * sy
    # Alignment can add a sample. Increase the stride rather than dropping the
    # far edge of the visible image to stay within the output budget.
    if ceil((x1 - x0) / sx) > out_w:
        sx = max(sx + 1, ceil((x1 - x0) / out_w))
    if ceil((y1 - y0) / sy) > out_h:
        sy = max(sy + 1, ceil((y1 - y0) / out_h))
    nx, ny = ceil((x1 - x0) / sx), ceil((y1 - y0) / sy)
    return SampleRegion(slice(y0, y1, sy), slice(x0, x1, sx),
                        (x0, y0, min(width, x0 + nx * sx), min(height, y0 + ny * sy)))


def tensor_plane(tensor, x_dim, y_dim, indices, region, channel_dim=None, channel=0):
    """Return a 2D torch view, without copying, even for noncontiguous inputs."""
    key = [int(i) for i in indices]
    key[x_dim], key[y_dim] = region.x, region.y
    if channel_dim is not None:
        key[channel_dim] = int(channel)
    plane = tensor[tuple(key)]
    # Basic indexing preserves the source dimension order and storage strides.
    return plane.transpose(0, 1) if x_dim < y_dim else plane


def display_image(planes, normalize):
    """Copy only sampled planes to CPU, then convert to uploadable RGB."""
    import torch

    # Stack/convert AFTER slicing: temporary storage stays viewport-sized.
    with torch.no_grad():
        sampled = torch.stack(planes, dim=-1).to(dtype=torch.float32).cpu().numpy()
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        lo, hi = float(sampled.min()), float(sampled.max())
        sampled -= lo
        if hi > lo:
            sampled /= hi - lo
    elif planes[0].dtype == torch.uint8:
        sampled /= 255.0
    np.clip(sampled, 0, 1, out=sampled)
    if sampled.shape[-1] == 1:
        sampled = np.repeat(sampled, 3, axis=-1)
    return np.ascontiguousarray(sampled)


class TensorPannableArea(PannableArea):
    """Pan in source coordinates and upload just the visible strided slice.

    Unlike PannableArea, this draws a cropped texture directly into ImGui's
    canvas. There is no full-source texture or intermediate GL framebuffer.
    """

    def __init__(self):
        super().__init__(force_mouse_capture=True)
        self.max_texture_size = None
        self.sample_shape = None

    def get_visible_box_canvas(self):
        # The base implementation's row-vector multiplication loses translation.
        pan = np.asarray(self.pan) + np.asarray(self.pan_delta)
        center = np.array([0.5 - pan[0], 0.5 + pan[1]])
        half = 0.5 / self.zoom
        return center - half, center + half

    def get_transform_ndc(self):
        # Large source coordinates require more than float32 precision when
        # zoomed in to individual pixels.
        pan = np.asarray(self.pan, dtype=np.float64) + self.pan_delta
        return np.array([[self.zoom, 0, 2 * self.zoom * pan[0]],
                         [0, self.zoom, 2 * self.zoom * pan[1]],
                         [0, 0, 1]], dtype=np.float64)

    def handle_pan(self):
        if not self.force_mouse and imgui.get_io().want_capture_mouse:
            return
        mouse = self.mouse_pos_abs
        if imgui.is_mouse_clicked(0) and self.mouse_hovers_content():
            self.pan_start = mouse.copy()
            self.is_panning = True
        if self.is_panning:
            if imgui.is_mouse_down(0) or imgui.is_mouse_released(0):
                # Use screen displacement, not coordinates transformed by the
                # pan being updated. Otherwise the previous delta feeds back
                # into this frame and causes oscillation and half-speed drags.
                delta = (mouse - self.pan_start) / (np.array([self.canvas_w, self.canvas_h]) * self.zoom)
                self.pan = np.asarray(self.pan) + delta * (1, -1)
                self.pan_start = mouse.copy()
            if not imgui.is_mouse_down(0):
                self.is_panning = False
                self.pan_start = (0, 0)
        if self.reset_enabled and imgui.is_mouse_double_clicked(0) and self.mouse_hovers_content():
            self.reset_xform()
        if self.snap_enabled and imgui.is_mouse_clicked(1) and self.mouse_hovers_content():
            self.snap_nearest_fractional_scale()

    def draw_tensor(self, v, tensor, selection, normalize):
        cW, cH = [max(0, int(n)) for n in imgui.get_content_region_avail()]
        self.canvas_w, self.canvas_h = cW, cH
        self.output_pos_tl[:] = imgui.get_cursor_screen_pos()
        self.tex_w, self.tex_h = tensor.shape[selection.x_dim], tensor.shape[selection.y_dim]
        if min(cW, cH) <= 0:
            return
        self.handle_pan()
        if self.max_texture_size is None:
            self.max_texture_size = int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE))
        hdpi_x, hdpi_y = imgui.get_io().display_framebuffer_scale
        budget = (min(self.max_texture_size, max(1, ceil(cW * hdpi_x))),
                  min(self.max_texture_size, max(1, ceil(cH * hdpi_y))))
        region = sample_region((self.tex_h, self.tex_w), self.get_visible_box_image(), budget)
        draw_list = imgui.get_window_draw_list()
        pos = tuple(self.output_pos_tl)
        end = (pos[0] + cW, pos[1] + cH)
        draw_list.add_rect_filled(pos, end, imgui.color_convert_float4_to_u32(self.clear_color))
        if region is not None:
            channels = selection.channels if selection.channel_dim is not None else (0,)
            planes = [tensor_plane(tensor, selection.x_dim, selection.y_dim,
                                   selection.indices, region, selection.channel_dim, ch)
                      for ch in channels]
            image = display_image(planes, normalize)
            self.sample_shape = image.shape
            v.upload_image_np('tensor', image)
            x0, y0, x1, y1 = region.bounds
            transform = self.uv_to_screen_xform()
            tl = transform @ (x0 / self.tex_w, y0 / self.tex_h, 1)
            br = transform @ (x1 / self.tex_w, y1 / self.tex_h, 1)
            draw_list.push_clip_rect(pos, end, True)
            # Crop the final partial sampling cell rather than stretching all
            # cells to fit the image boundary.
            uv_max = ((x1 - x0) / (image.shape[1] * region.x.step),
                      (y1 - y0) / (image.shape[0] * region.y.step))
            draw_list.add_image(imgui.ImTextureRef(v._images['tensor'].tex),
                                tuple(tl[:2]), tuple(br[:2]), (0, 0), uv_max)
            draw_list.pop_clip_rect()
        else:
            self.sample_shape = None
        imgui.dummy((cW, cH))


@dataclass
class TensorSelection:
    x_dim: int
    y_dim: int
    channel_dim: int | None
    indices: list[int]
    channels: tuple[int, int, int]

    def validate(self, shape):
        dims = [self.x_dim, self.y_dim]
        if self.channel_dim is not None:
            dims.append(self.channel_dim)
        if any(d < 0 or d >= len(shape) for d in dims) or len(set(dims)) != len(dims):
            raise ValueError('X, Y and channel dimensions must be distinct valid dimensions')
        if len(self.indices) != len(shape) or any(i < 0 or i >= n for i, n in zip(self.indices, shape)):
            raise ValueError('Provide one valid slice index per tensor dimension')
        if len(self.channels) != 3:
            raise ValueError('Provide three channel indices (R, G, B)')
        if self.channel_dim is not None and any(c < 0 or c >= shape[self.channel_dim] for c in self.channels):
            raise ValueError('Channel indices are out of range')


class TensorViewer:
    """Single-process viewer; the entire UI runs in one untraced thread.

    Use one active instance at a time: GLFW/ImGui have process-global state.
    GLFW windows in background threads are supported on Linux/Windows; macOS
    requires window/event handling on the main thread.
    """

    def __init__(self, title='Tensor viewer', *, normalize=True, vsync=True, hidden=False):
        self.title, self.normalize, self.vsync = title, normalize, vsync
        self._hidden = hidden
        self.ui_locked = True
        self._lock = Lock()
        self._pending = None
        self._tensor = None
        self._selection = None
        self._quit = Event()
        self._started = Event()
        self._finished = Event()
        self._error = None
        self._window_size = (0, 0)
        self.pan_handler = TensorPannableArea()
        self.thread = Thread(target=self.start_thread, name='pyviewer-tensor', daemon=True)
        self.thread.pydev_do_not_trace = True
        self.thread.start()

    @property
    def window_size(self):
        return self._window_size

    def wait_for_startup(self, timeout=10):
        if not self._started.wait(timeout):
            self._raise_error()
            raise TimeoutError(f'Tensor viewer did not start within {timeout}s')
        self._raise_error()

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError('Tensor viewer thread failed') from self._error

    def wait_for_close(self):
        self.thread.join()
        self._raise_error()

    def close(self):
        self._quit.set()
        if current_thread() is not self.thread:
            self.wait_for_close()

    def hide(self):
        self._hidden = True

    def show(self):
        self._hidden = False

    def draw(self, tensor, *, x_dim=None, y_dim=None, channel_dim=None,
             indices=None, channels=None):
        """Publish a tensor reference without scanning or copying its storage.

        Defaults to a grayscale slice with the last two dimensions as Y/X.
        UI selections persist across draw calls with the same shape. Supplying
        any selection argument resets the selection; omit them to preserve it.
        Normalization uses only sampled values, so its range may change on pan.
        CUDA callers should finish producing the tensor before publishing it.
        """
        import torch

        self._raise_error()
        if self._finished.is_set():
            raise RuntimeError('Tensor viewer is closed')
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError('Expected a dense strided torch.Tensor')
        if tensor.ndim < 2 or any(n == 0 for n in tensor.shape):
            raise ValueError('Expected at least two nonempty dimensions')
        if tensor.is_complex() or tensor.is_quantized:
            raise TypeError('Complex and quantized tensors are not supported')
        supplied = any(a is not None for a in (x_dim, y_dim, channel_dim, indices, channels))
        selection = TensorSelection(
            tensor.ndim - 1 if x_dim is None else x_dim,
            tensor.ndim - 2 if y_dim is None else y_dim,
            channel_dim, list(indices) if indices is not None else [0] * tensor.ndim,
            tuple(channels) if channels is not None else (0, 0, 0))
        selection.validate(tensor.shape)
        # Never hold the publication lock while sampling or operating on storage.
        with self._lock:
            self._pending = (tensor.detach(), selection, supplied)

    def start_thread(self):
        v = None
        try:
            context_api = glfw.EGL_CONTEXT_API if egl_patch.is_egl() else glfw.NATIVE_CONTEXT_API
            v = gl_viewer.viewer(self.title, swap_interval=int(self.vsync), hidden=self._hidden,
                                 context_creation_api=context_api)
            v.set_interp_nearest()
            v.pan_handler = self.pan_handler
            window_hidden = self._hidden

            def ui(viewer):
                nonlocal window_hidden
                if self._quit.is_set():
                    glfw.set_window_should_close(viewer._window, True)
                    return
                if self._hidden != window_hidden:
                    window_hidden = self._hidden
                    (glfw.hide_window if window_hidden else glfw.show_window)(viewer._window)
                self._window_size = glfw.get_window_size(viewer._window)
                self._ui(viewer)

            def callbacks(window):
                self.pan_handler.set_callbacks(window)
                self._started.set()

            v.start(ui, glfw_init_callback=callbacks)
        except BaseException as error:
            self._error = error
            if v is not None and not v.quit:
                v.gl_shutdown()
        finally:
            self._tensor = self._pending = None
            self._finished.set()
            self._started.set()  # Wake startup waiters on failure as well.

    def _ui(self, v):
        # A debugger may suspend the producer while it holds this lock. Keep
        # rendering the previous reference rather than waiting for publication.
        if self._lock.acquire(blocking=False):
            try:
                pending, self._pending = self._pending, None
            finally:
                self._lock.release()
            if pending is not None:
                tensor, selection, supplied = pending
                if self._tensor is None or tensor.shape != self._tensor.shape or supplied:
                    self._selection = selection
                    self.pan_handler.reset_xform()
                self._tensor = tensor
        menu_height = self._menu_bar(v)
        imgui.set_next_window_pos((0, menu_height))
        imgui.set_next_window_size((self._window_size[0], max(0, self._window_size[1] - menu_height)))
        begin_inline('Tensor viewer', inputs=True)
        try:
            if self._tensor is None:
                imgui.text('Call draw(tensor) to select a tensor.')
                return
            self._controls()
            self.pan_handler.draw_tensor(v, self._tensor, self._selection, self.normalize)
        finally:
            imgui.end()

    def _menu_bar(self, v):
        height = 0
        if imgui.begin_main_menu_bar():
            height = imgui.get_window_height()
            scale = v.ui_scale
            available = imgui.get_window_width() - imgui.get_cursor_pos()[0]
            slider_width = 300 if not self.ui_locked else 0
            padding = available - slider_width - (30 if not self.ui_locked else 25) * scale
            if padding > 0:
                imgui.invisible_button('##scale_padding', (padding, 1))
            if not self.ui_locked:
                # Fixed width prevents slider drift when changing the scale.
                imgui.set_next_item_width(300)
                changed, value = imgui.slider_float('##ui_scale', scale, 0.1, 4.0)
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed, value = True, 1.0
                if changed:
                    v.set_ui_scale(value)
            color = (0.8, 0.0, 0.0, 1) if self.ui_locked else (0.0, 1.0, 0.0, 1)
            imgui.push_style_color(imgui.Col_.text, color)
            if imgui.button('L' if self.ui_locked else 'U', (20 * scale, 0)):
                self.ui_locked = not self.ui_locked
            imgui.pop_style_color()
            imgui.end_main_menu_bar()
        return height

    def _controls(self):
        tensor, s = self._tensor, self._selection
        imgui.text(f'{tuple(tensor.shape)} | {tensor.dtype} | {tensor.device}')
        if imgui.collapsing_header('Dimensions and channels', imgui.TreeNodeFlags_.default_open):
            changed = False
            for axis in ('x_dim', 'y_dim', 'channel_dim'):
                imgui.text({'x_dim': 'X', 'y_dim': 'Y', 'channel_dim': 'Channel'}[axis])
                if axis == 'channel_dim':
                    imgui.same_line()
                    if imgui.radio_button('Grayscale', s.channel_dim is None):
                        s.channel_dim = None
                        changed = True
                for dim, size in enumerate(tensor.shape):
                    imgui.same_line()
                    other = next((other for other in ('x_dim', 'y_dim', 'channel_dim')
                                  if other != axis and getattr(s, other) == dim), None)
                    # X/Y clicks swap dimensions, including on 2D tensors.
                    # A new channel axis needs a non-spatial dimension.
                    imgui.begin_disabled(axis == 'channel_dim' and s.channel_dim is None and other is not None)
                    if imgui.radio_button(f'{dim} ({size})##{axis}', getattr(s, axis) == dim):
                        if other is not None:
                            setattr(s, other, getattr(s, axis))
                        setattr(s, axis, dim)
                        if s.channel_dim is not None:
                            s.channels = tuple(min(i, tensor.shape[s.channel_dim] - 1) for i in s.channels)
                        changed = True
                    imgui.end_disabled()
            for dim, size in enumerate(tensor.shape):
                if dim not in (s.x_dim, s.y_dim, s.channel_dim):
                    _, s.indices[dim] = imgui.slider_int(f'Index dim {dim}', s.indices[dim], 0, size - 1)
            if s.channel_dim is not None:
                channels = list(s.channels)
                for i, name in enumerate(('R', 'G', 'B')):
                    _, channels[i] = imgui.slider_int(f'{name} index', channels[i], 0, tensor.shape[s.channel_dim] - 1)
                s.channels = tuple(channels)
            if changed:
                self.pan_handler.reset_xform()
        _, self.normalize = imgui.checkbox('Normalize visible samples', self.normalize)
        imgui.same_line()
        if imgui.button('Reset view'):
            self.pan_handler.reset_xform()
        imgui.separator()


inst: TensorViewer | None = None


def init(title='Tensor viewer', *, sync=True, **kwargs):
    global inst
    if inst is None or inst._finished.is_set():
        inst = TensorViewer(title, **kwargs)
    if sync:
        inst.wait_for_startup()
    return inst


def draw(tensor, **kwargs):
    """Open the global viewer and publish a tensor reference."""
    init().draw(tensor, **kwargs)


def close():
    global inst
    if inst is not None:
        inst.close()
        inst = None
