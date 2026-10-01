"""A debugger-friendly, threaded viewer for tensors larger than GPU textures.

Only a strided view of the visible slice is materialized. The source is retained
by reference; callers must coordinate writes if they need a consistent frame.
"""

from dataclasses import dataclass
from collections import deque
from ctypes import c_bool
from math import ceil, floor
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from time import monotonic, perf_counter

import numpy as np
from imgui_bundle import imgui
import glfw
import OpenGL.GL as gl

from . import egl_patch
from .docking_viewer_py import PyDockingViewer, dockable
from .gl_viewer import _texture
from .utils import PannableArea, get_grid_dims


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


def sampled_image(planes):
    """Pack viewport samples on their source device."""
    import torch

    with torch.no_grad():
        sampled = torch.stack(planes, dim=-1).to(dtype=torch.float32)
        if sampled.is_cuda:
            return torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.nan_to_num(sampled.numpy(), nan=0.0, posinf=0.0, neginf=0.0)


def display_image(planes, normalize, limits=None, sampled=None):
    """Convert viewport-sized sampled planes to uploadable RGB."""
    import torch

    sampled = sampled_image(planes) if sampled is None else sampled
    if isinstance(sampled, torch.Tensor):
        with torch.no_grad():
            if normalize:
                lo, hi = limits if limits is not None else (sampled.min(), sampled.max())
                scale = torch.where(hi > lo, hi - lo, torch.ones_like(hi))
                sampled = (sampled - lo) / scale
            elif planes[0].dtype == torch.uint8:
                sampled = sampled / 255.0
            sampled = sampled.clamp(0, 1)
            if sampled.shape[-1] == 1:
                sampled = sampled.expand(*sampled.shape[:-1], 3)
            return sampled.contiguous()
    if normalize:
        lo, hi = limits if limits is not None else (float(sampled.min()), float(sampled.max()))
        sampled -= lo
        if hi > lo:
            sampled /= hi - lo
    elif planes[0].dtype == torch.uint8:
        sampled /= 255.0
    np.clip(sampled, 0, 1, out=sampled)
    if sampled.shape[-1] == 1:
        sampled = np.repeat(sampled, 3, axis=-1)
    return np.ascontiguousarray(sampled)


class CudaFrameTimer:
    """CUDA stream timestamps, read only after the final event completes."""

    upload_stages = ('Alpha', 'Texture', 'Map', 'Copy', 'Unmap', 'Mipmap', 'Fallback')

    def __init__(self, stream):
        self.stream = stream
        self.marks = []
        self.cpu_intervals = {}

    def add_cpu(self, name, milliseconds):
        self.cpu_intervals[name] = self.cpu_intervals.get(name, 0.0) + milliseconds

    def mark(self, name):
        import torch

        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self.marks.append((name, event))

    def ready(self):
        return self.marks[-1][1].query()

    def results(self):
        durations = {}
        for (_, start), (name, end) in zip(self.marks, self.marks[1:]):
            durations[name] = durations.get(name, 0.0) + start.elapsed_time(end)
        durations['Upload'] = sum(durations.get(name, 0.0) for name in self.upload_stages)
        durations['Total'] = self.marks[0][1].elapsed_time(self.marks[-1][1])
        return durations


class TensorPannableArea(PannableArea):
    """Pan in source coordinates and upload just the visible strided slice.

    Unlike PannableArea, this draws a cropped texture directly into ImGui's
    canvas. There is no full-source texture or intermediate GL framebuffer.
    """

    def __init__(self):
        super().__init__(force_mouse_capture=True)
        self.max_texture_size = None
        self.sample_shape = None
        self._grid_texture_count = 0

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

    def draw_tensor(self, v, tensor, selection, normalize, cuda_stream=None, gpu_timer=None):
        # Grid coordinates are virtual: each visible tile slices the original tensor.
        # Only viewport-sized samples are copied for texture upload.
        cW, cH = [max(0, int(n)) for n in imgui.get_content_region_avail()]
        self.canvas_w, self.canvas_h = cW, cH
        self.output_pos_tl[:] = imgui.get_cursor_screen_pos()
        tile_w, tile_h = tensor.shape[selection.x_dim], tensor.shape[selection.y_dim]
        if selection.grid_dim is None:
            cols, rows = 1, 1
        else:
            cols, rows = selection.grid_shape or get_grid_dims(tensor.shape[selection.grid_dim])
        self.tex_w, self.tex_h = cols * tile_w, rows * tile_h
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
        count = 0
        if region is not None:
            visible_tl, visible_br = self.get_visible_box_image()
            transform = self.uv_to_screen_xform()
            tiles = []
            draw_list.push_clip_rect(pos, end, True)
            for row in range(region.y.start // tile_h, (region.y.stop - 1) // tile_h + 1):
                for col in range(region.x.start // tile_w, (region.x.stop - 1) // tile_w + 1):
                    grid_index = selection.grid_start + row * cols + col
                    if selection.grid_dim is not None and grid_index >= tensor.shape[selection.grid_dim]:
                        continue
                    tile_region = sample_region(
                        (tile_h, tile_w),
                        ((visible_tl[0] * cols - col, visible_tl[1] * rows - row),
                         (visible_br[0] * cols - col, visible_br[1] * rows - row)),
                        budget)
                    if tile_region is None:
                        continue
                    indices = selection.indices.copy()
                    if selection.grid_dim is not None:
                        indices[selection.grid_dim] = grid_index
                    channels = selection.channels if selection.channel_dim is not None else (0,)
                    planes = [tensor_plane(tensor, selection.x_dim, selection.y_dim,
                                           indices, tile_region, selection.channel_dim, ch)
                              for ch in channels]
                    tiles.append((row, col, tile_region, planes, sampled_image(planes)))
            if gpu_timer is not None:
                gpu_timer.mark('Sample')
            limits = None
            if normalize and tiles:
                if tensor.is_cuda:
                    import torch
                    limits = (torch.stack([sampled.min() for *_, sampled in tiles]).min(),
                              torch.stack([sampled.max() for *_, sampled in tiles]).max())
                else:
                    limits = (min(float(sampled.min()) for *_, sampled in tiles),
                              max(float(sampled.max()) for *_, sampled in tiles))
            if gpu_timer is not None:
                gpu_timer.mark('Range')
            for row, col, tile_region, planes, sampled in tiles:
                image = display_image(planes, normalize, limits, sampled)
                if gpu_timer is not None:
                    gpu_timer.mark('Convert')
                self.sample_shape = image.shape
                name = f'tensor-grid-{count}' if selection.grid_dim is not None else 'tensor'
                if tensor.is_cuda:
                    capacity = (min(tile_h, budget[1]), min(tile_w, budget[0]))
                    v.upload_image_torch(name, image, stream=cuda_stream,
                                         timer=gpu_timer, capacity=capacity)
                else:
                    v.upload_image_np(name, image)
                x0, y0, x1, y1 = tile_region.bounds
                x0 += col * tile_w
                x1 += col * tile_w
                y0 += row * tile_h
                y1 += row * tile_h
                tl = transform @ (x0 / self.tex_w, y0 / self.tex_h, 1)
                br = transform @ (x1 / self.tex_w, y1 / self.tex_h, 1)
                # Crop the final partial sampling cell at the tile edge.
                tex_h, tex_w = v._images[name].shape[:2]
                uv_max = ((x1 - x0) / (tex_w * tile_region.x.step),
                          (y1 - y0) / (tex_h * tile_region.y.step))
                draw_list.add_image(imgui.ImTextureRef(v._images[name].tex),
                                    tuple(tl[:2]), tuple(br[:2]), (0, 0), uv_max)
                count += 1
                if gpu_timer is not None:
                    gpu_timer.mark('Other')
            draw_list.pop_clip_rect()
        if count == 0:
            self.sample_shape = None
        kept = count if selection.grid_dim is not None else 0
        for slot in range(kept, self._grid_texture_count):
            v._images.pop(f'tensor-grid-{slot}').release()
        self._grid_texture_count = kept
        imgui.dummy((cW, cH))


@dataclass
class TensorSelection:
    x_dim: int
    y_dim: int
    channel_dim: int | None
    indices: list[int]
    channels: tuple[int, int, int]
    grid_dim: int | None = None
    grid_shape: tuple[int, int] | None = None
    grid_start: int = 0

    def validate(self, shape):
        dims = [self.x_dim, self.y_dim]
        if self.channel_dim is not None:
            dims.append(self.channel_dim)
        if self.grid_dim is not None:
            dims.append(self.grid_dim)
        if any(d < 0 or d >= len(shape) for d in dims) or len(set(dims)) != len(dims):
            raise ValueError('X, Y, channel and grid dimensions must be distinct valid dimensions')
        if len(self.indices) != len(shape) or any(i < 0 or i >= n for i, n in zip(self.indices, shape)):
            raise ValueError('Provide one valid slice index per tensor dimension')
        if len(self.channels) != 3:
            raise ValueError('Provide three channel indices (R, G, B)')
        if self.channel_dim is not None and any(c < 0 or c >= shape[self.channel_dim] for c in self.channels):
            raise ValueError('Channel indices are out of range')


class _TensorDockingViewer(PyDockingViewer):
    """Docking UI for a TensorViewer; tensors stay owned by its publisher."""

    def __init__(self, owner):
        self.owner = owner
        self._images = {}
        self._dock_layout_ready = False
        context_api = glfw.EGL_CONTEXT_API if egl_patch.is_egl() else glfw.NATIVE_CONTEXT_API
        super().__init__(owner.title, normalize=False, with_implot=False,
                         swap_interval=int(owner.vsync), hidden=owner._hidden,
                         context_creation_api=context_api)

    def compute_loop(self):
        # TensorViewer.draw() publishes references; there is no compute worker.
        pass

    def setup_state(self):
        self.pan_handler = self.owner.pan_handler
        self.pan_handler.set_callbacks(self.window)
        self.idle_timeout_s = float('inf')
        self.owner._started.set()

    def pre_new_frame(self):
        self.owner._pre_new_frame(self)

    def _begin_dockspace(self):
        super()._begin_dockspace()
        if self._dock_layout_ready:
            return
        self._dock_layout_ready = True
        if Path(self._ini_path).is_file():
            return
        dock_id = imgui.get_id('MainDockSpace')
        builder = imgui.internal
        builder.dock_builder_remove_node(dock_id)
        builder.dock_builder_add_node(dock_id)
        builder.dock_builder_set_node_size(dock_id, imgui.get_main_viewport().size)
        side = builder.dock_builder_split_node(dock_id, imgui.Dir.right, 0.26)
        bottom = builder.dock_builder_split_node(side.id_at_dir, imgui.Dir.down, 0.35)
        builder.dock_builder_dock_window('Tensor Image', side.id_at_opposite_dir)
        builder.dock_builder_dock_window('Axes', bottom.id_at_opposite_dir)
        builder.dock_builder_dock_window('GPU Timings', bottom.id_at_dir)
        builder.dock_builder_finish(dock_id)

    @dockable(title='Tensor Image')
    def output(self):
        self.owner._image_ui(self)

    @dockable(title='Axes')
    def axes(self):
        if self.owner._tensor is None:
            imgui.text('Call draw(tensor) to select a tensor.')
        else:
            self.owner._controls()

    @dockable(title='GPU Timings')
    def timings(self):
        self.owner._timings_window()

    def upload_image_torch(self, name, tensor, stream=None, timer=None, capacity=None):
        if name not in self._images:
            start = perf_counter()
            self._images[name] = _texture(gl.GL_NEAREST, gl.GL_NEAREST)
            if timer is not None:
                timer.add_cpu('Texture', (perf_counter() - start) * 1000)
                timer.mark('Texture')
        self._images[name].upload_torch(tensor, stream=stream, timer=timer,
                                        capacity=capacity)

    def upload_image_np(self, name, image):
        if name not in self._images:
            self._images[name] = _texture(gl.GL_NEAREST, gl.GL_NEAREST)
        self._images[name].upload_np(image)

    def _cleanup(self):
        for texture in self._images.values():
            texture.release()
        self._images.clear()
        super()._cleanup()


class TensorViewer:
    """Single-process viewer; the entire UI runs in one untraced thread.

    Use one active instance at a time: GLFW/ImGui have process-global state.
    GLFW windows in background threads are supported on Linux/Windows; macOS
    requires window/event handling on the main thread.
    P/Pause toggles updates; N/Space accepts one next tensor while paused.
    """

    def __init__(self, title='Tensor viewer', *, normalize=True, vsync=True, hidden=False,
                 paused=False, next=False):
        self.title, self.normalize, self.vsync = title, normalize, vsync
        self.paused = c_bool(paused)
        self.next = c_bool(next)
        self._hidden = hidden
        self._lock = Lock()
        self._pending = None
        self._tensor = None
        self._ready_event = None
        self._cuda_streams = {}
        self._timing_pending = deque(maxlen=64)
        self._timing_latest = None
        self._timing_history = deque()
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

    def draw(self, tensor, *, x_dim=None, y_dim=None, channel_dim=None, grid_dim=None,
             indices=None, channels=None, ignore_pause=False):
        """Publish a tensor reference without scanning or copying its storage.

        Defaults to a grayscale slice with the last two dimensions as Y/X.
        grid_dim unwraps one other dimension into a spatial grid.
        UI selections persist across draw calls with the same shape. Supplying
        any selection argument resets the selection; omit them to preserve it.
        Normalization uses only sampled values, so its range may change on pan.
        CUDA callers should finish producing the tensor before publishing it.
        """
        import torch

        self._raise_error()
        if self._finished.is_set():
            raise RuntimeError('Tensor viewer is closed')
        if self.paused.value and not ignore_pause and not self.next.value:
            return
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError('Expected a dense strided torch.Tensor')
        if tensor.ndim < 2 or any(n == 0 for n in tensor.shape):
            raise ValueError('Expected at least two nonempty dimensions')
        if tensor.is_complex() or tensor.is_quantized:
            raise TypeError('Complex and quantized tensors are not supported')
        supplied = any(a is not None for a in (x_dim, y_dim, channel_dim, grid_dim, indices, channels))
        selection = TensorSelection(
            tensor.ndim - 1 if x_dim is None else x_dim,
            tensor.ndim - 2 if y_dim is None else y_dim,
            channel_dim, list(indices) if indices is not None else [0] * tensor.ndim,
            tuple(channels) if channels is not None else (0, 0, 0), grid_dim)
        selection.validate(tensor.shape)
        # Never hold the publication lock while sampling or operating on storage.
        ready_event = None
        if tensor.is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(tensor.device))
        with self._lock:
            self._pending = (tensor.detach(), selection, supplied, ready_event)
            self.next.value = False

    def start_thread(self):
        try:
            _TensorDockingViewer(self)
        except BaseException as error:
            self._error = error
        finally:
            self._tensor = self._pending = None
            self._finished.set()
            self._started.set()  # Wake startup waiters on failure as well.

    def _pre_new_frame(self, v):
        if self._quit.is_set():
            glfw.set_window_should_close(v.window, True)
            return
        if imgui.is_key_pressed(imgui.Key.p, repeat=False) or imgui.is_key_pressed(imgui.Key.pause, repeat=False):
            self.paused.value = not self.paused.value
        if imgui.is_key_pressed(imgui.Key.n, repeat=False) or imgui.is_key_pressed(imgui.Key.space, repeat=True):
            self.next.value = True
        if self._hidden != v._hidden:
            v._hidden = self._hidden
            (glfw.hide_window if self._hidden else glfw.show_window)(v.window)
        self._window_size = glfw.get_window_size(v.window)
        # A debugger may suspend the producer while it holds this lock. Keep
        # rendering the previous reference rather than waiting for publication.
        if self._lock.acquire(blocking=False):
            try:
                pending, self._pending = self._pending, None
            finally:
                self._lock.release()
            if pending is not None:
                tensor, selection, supplied, ready_event = pending
                if self._tensor is None or tensor.shape != self._tensor.shape or supplied:
                    self._selection = selection
                    self.pan_handler.reset_xform()
                self._tensor = tensor
                self._ready_event = ready_event

    def _image_ui(self, v):
        canvas_pos = imgui.get_cursor_screen_pos()
        if self._tensor is None:
            imgui.text('Call draw(tensor) to select a tensor.')
            self._draw_status_overlay(v, canvas_pos)
            return
        if self._tensor.is_cuda:
            import torch
            device = self._tensor.device
            stream = self._cuda_streams.get(device)
            if stream is None:
                stream = self._cuda_streams[device] = torch.cuda.Stream(device=device)
            if self._ready_event is not None:
                stream.wait_event(self._ready_event)
                self._ready_event = None
            with torch.cuda.stream(stream):
                timer = CudaFrameTimer(stream)
                timer.mark('Start')
                cpu_start = perf_counter()
                self.pan_handler.draw_tensor(v, self._tensor, self._selection,
                                             self.normalize, cuda_stream=stream, gpu_timer=timer)
                self._tensor.record_stream(stream)
                cpu_ms = (perf_counter() - cpu_start) * 1000
                timer.mark('Other')
                self._timing_pending.append((timer, cpu_ms))
        else:
            self.pan_handler.draw_tensor(v, self._tensor, self._selection, self.normalize)
        self._draw_status_overlay(v, canvas_pos)

    def _draw_status_overlay(self, v, canvas_pos):
        if not (self.paused.value or self.next.value):
            return
        label = 'NEXT' if self.next.value else 'PAUSED'
        scale = v.ui_scale
        imgui.push_font(v.default_font, 31 * scale)
        try:
            text_w, text_h = imgui.calc_text_size(label)
            x, y = canvas_pos[0] + 5 * scale, canvas_pos[1] + 8 * scale
            draw_list = imgui.get_window_draw_list()
            draw_list.add_rect_filled((x, y),
                                      (x + text_w + 30 * scale, y + text_h + 6 * scale),
                                      imgui.color_convert_float4_to_u32((0, 0, 0, 1)))
            color = (0.8, 0.8, 0.8, 1) if monotonic() - v.last_ui_active > 2 else (1, 1, 1, 1)
            draw_list.add_text((x + 15 * scale, y + 2 * scale),
                               imgui.color_convert_float4_to_u32(color), label)
        finally:
            imgui.pop_font()

    def _timings_window(self):
        now = perf_counter()
        while self._timing_pending and self._timing_pending[0][0].ready():
            timer, cpu_ms = self._timing_pending.popleft()
            self._timing_latest = (timer.results(), cpu_ms, timer.cpu_intervals.copy())
            self._timing_history.append((now, *self._timing_latest))
        while self._timing_history and self._timing_history[0][0] < now - 5:
            self._timing_history.popleft()

        if self._timing_latest is None:
            imgui.text('Waiting for a CUDA frame')
            return
        times, cpu_ms, cpu_parts = self._timing_latest
        imgui.text('CUDA intervals: latest / max 5 s')
        for name in ('Sample', 'Range', 'Convert', *CudaFrameTimer.upload_stages,
                     'Upload', 'Other', 'Total'):
            if name in ('Mipmap', 'Fallback') and name not in times:
                continue
            maximum = max((sample.get(name, 0.0) for _, sample, _, _ in self._timing_history),
                          default=None)
            max_text = f'{maximum:.2f}' if maximum is not None else '--'
            imgui.text(f'{name}: {times.get(name, 0.0):.2f} / {max_text} ms')
        imgui.separator()
        imgui.text('CPU calls: latest / max 5 s')
        for name in CudaFrameTimer.upload_stages:
            if name not in cpu_parts:
                continue
            maximum = max((parts.get(name, 0.0) for _, _, _, parts in self._timing_history),
                          default=None)
            imgui.text(f'{name}: {cpu_parts[name]:.2f} / {maximum:.2f} ms')
        maximum = max((cpu for _, _, cpu, _ in self._timing_history), default=None)
        max_text = f'{maximum:.2f}' if maximum is not None else '--'
        imgui.text(f'CPU submit: {cpu_ms:.2f} / {max_text} ms')
        imgui.text(f'Pending frames: {len(self._timing_pending)}')
        imgui.text('CUDA stream time; includes launch gaps')
        imgui.text('OpenGL rendering is not timed')

    def _controls(self):
        tensor, s = self._tensor, self._selection
        imgui.text(f'{tuple(tensor.shape)} | {tensor.dtype} | {tensor.device}')
        if imgui.collapsing_header('Dimensions', imgui.TreeNodeFlags_.default_open):
            changed = False
            for axis in ('x_dim', 'y_dim', 'channel_dim', 'grid_dim'):
                imgui.text({'x_dim': 'X', 'y_dim': 'Y', 'channel_dim': 'C', 'grid_dim': 'G'}[axis])
                for dim, size in enumerate(tensor.shape):
                    imgui.same_line()
                    other = next((other for other in ('x_dim', 'y_dim', 'channel_dim', 'grid_dim')
                                  if other != axis and getattr(s, other) == dim), None)
                    # X/Y clicks swap dimensions, including on 2D tensors.
                    # A new channel/grid axis needs an unused dimension.
                    imgui.begin_disabled(axis in ('channel_dim', 'grid_dim') and getattr(s, axis) is None and other is not None)
                    if imgui.radio_button(f'{dim} ({size})##{axis}', getattr(s, axis) == dim):
                        old_grid_dim = s.grid_dim
                        if other is not None:
                            setattr(s, other, getattr(s, axis))
                        setattr(s, axis, dim)
                        if s.grid_dim != old_grid_dim:
                            s.grid_shape = None
                            s.grid_start = 0
                        if s.channel_dim is not None:
                            s.channels = tuple(min(i, tensor.shape[s.channel_dim] - 1) for i in s.channels)
                        changed = True
                    imgui.end_disabled()
                if axis in ('channel_dim', 'grid_dim'):
                    imgui.same_line()
                    label = 'Grayscale' if axis == 'channel_dim' else 'No grid'
                    if imgui.radio_button(label, getattr(s, axis) is None):
                        setattr(s, axis, None)
                        if axis == 'grid_dim':
                            s.grid_shape = None
                            s.grid_start = 0
                        changed = True
            if s.grid_dim is not None:
                length = tensor.shape[s.grid_dim]
                cols, rows = s.grid_shape or get_grid_dims(length)
                cols_changed, cols = imgui.slider_int('Grid columns', cols, 1, length // rows)
                rows_changed, rows = imgui.slider_int('Grid rows', rows, 1, length // cols)
                s.grid_shape = (cols, rows)
                page_size = cols * rows
                last_page = (length - 1) // page_size
                page = min(s.grid_start // page_size, last_page)
                if last_page:
                    _, page = imgui.slider_int('Grid page', page, 0, last_page)
                s.grid_start = page * page_size
                changed |= cols_changed or rows_changed
            for dim, size in enumerate(tensor.shape):
                if dim not in (s.x_dim, s.y_dim, s.channel_dim, s.grid_dim):
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
