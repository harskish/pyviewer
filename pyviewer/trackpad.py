"""macOS trackpad scrolling for Python ImGui and ImPlot viewers."""

import sys
from contextlib import contextmanager
from functools import wraps
from math import expm1, log1p

from imgui_bundle import imgui, implot, implot3d


# Independent sensitivity multipliers
IMGUI_SCROLL_SCALE = 1.0
IMPLOT_SCROLL_SCALE = 1.0
IMPLOT3D_SCROLL_SCALE = 1.0


class ImGuiScrollScalePatch:
    def __init__(self, scale=None):
        if scale is None:
            scale = 0.100 * IMGUI_SCROLL_SCALE
        if scale <= 0:
            raise ValueError('scale must be positive')
        self.scale = scale
        self._original = None
        self._patched = None

    def install(self):
        """Scale wheel events before ImGui distributes them to windows."""
        if self._patched is not None:
            return self
        self._original = imgui.IO.add_mouse_wheel_event

        @wraps(self._original)
        def add_mouse_wheel_event(io, xoffset, yoffset):
            self._original(io, xoffset * self.scale, yoffset * self.scale)

        self._patched = add_mouse_wheel_event
        imgui.IO.add_mouse_wheel_event = add_mouse_wheel_event
        return self

    def uninstall(self):
        """Restore the original ImGui input method."""
        if self._patched is None:
            return
        if imgui.IO.add_mouse_wheel_event is self._patched:
            imgui.IO.add_mouse_wheel_event = self._original
        self._patched = None
        self._original = None
        global _installed_scroll_patch
        if _installed_scroll_patch is self:
            _installed_scroll_patch = None

    @contextmanager
    def original_wheel(self, scale=1.0):
        """Temporarily expose raw wheel input with a widget's own multiplier."""
        io = imgui.get_io()
        scaled_x, scaled_y = io.mouse_wheel_h, io.mouse_wheel
        io.mouse_wheel_h = scaled_x / self.scale * scale
        io.mouse_wheel = scaled_y / self.scale * scale
        try:
            yield
        finally:
            io.mouse_wheel_h = scaled_x
            io.mouse_wheel = scaled_y


_installed_scroll_patch = None


def install_imgui_scroll_scale(scale=None):
    """Install once; return the patch (and its scale) for ImPlot compensation."""
    global _installed_scroll_patch
    if _installed_scroll_patch is None:
        _installed_scroll_patch = ImGuiScrollScalePatch(scale).install()
    return _installed_scroll_patch


class ImPlotTrackpadZoom:
    def __init__(self, enabled=True, zoom_per_scroll=None):
        if zoom_per_scroll is None:
            zoom_per_scroll = expm1(log1p(0.035) * 1.2 * IMPLOT_SCROLL_SCALE)
        if zoom_per_scroll <= 0:
            raise ValueError('zoom_per_scroll must be positive')
        self.enabled = enabled
        self.zoom_per_scroll = zoom_per_scroll
        self._rect = None
        self._limits = None
        self._scroll = 0.0
        self._cursor = None

    def install(self):
        """Disable ImPlot's sign-only wheel zoom in the current context.

        ImPlot's input map is shared by all plots in the context, so every
        scrollable plot should use its own instance of this helper.
        """
        if self.enabled:
            implot.get_input_map().zoom_rate = 0.0

    def on_scroll(self, yoffset, cursor_pos):
        """Queue a raw vertical scroll delta; return True if consumed."""
        if not self.enabled or not yoffset or self._rect is None:
            return False
        x, y = cursor_pos
        left, top, right, bottom = self._rect
        if not (left <= x <= right and top <= y <= bottom):
            return False
        self._scroll += yoffset
        self._cursor = (x, y)
        return True

    def before_plot(self):
        """Apply queued scroll to the next plot's axes before BeginPlot."""
        if not self.enabled or not self._scroll or self._limits is None:
            return
        left, top, right, bottom = self._rect
        if right <= left or bottom <= top:
            self.cancel()
            return
        x, y = self._cursor
        tx = (x - left) / (right - left)
        ty = (y - top) / (bottom - top)
        x_min, x_max, y_min, y_max = self._limits
        # Compound fractional events so the trackpad's momentum tail counts.
        scale = (1.0 + self.zoom_per_scroll) ** self._scroll
        new_width = (x_max - x_min) / scale
        new_height = (y_max - y_min) / scale
        anchor_x = x_min + tx * (x_max - x_min)
        anchor_y = y_max - ty * (y_max - y_min)
        implot.set_next_axis_limits(implot.ImAxis_.x1,
                                    anchor_x - tx * new_width,
                                    anchor_x + (1 - tx) * new_width,
                                    implot.Cond_.always)
        implot.set_next_axis_limits(implot.ImAxis_.y1,
                                    anchor_y - (1 - ty) * new_height,
                                    anchor_y + ty * new_height,
                                    implot.Cond_.always)
        self.cancel()

    def after_plot(self):
        """Remember the current plot rectangle and axis limits before EndPlot."""
        if not self.enabled:
            return
        pos, size = implot.get_plot_pos(), implot.get_plot_size()
        self._rect = (pos.x, pos.y, pos.x + size.x, pos.y + size.y)
        limits = implot.get_plot_limits()
        self._limits = (limits.x.min, limits.x.max, limits.y.min, limits.y.max)

    def cancel(self):
        """Discard pending scroll events."""
        self._scroll = 0.0
        self._cursor = None

    def clear(self):
        """Forget a plot that is no longer displayed."""
        self.cancel()
        self._rect = None
        self._limits = None


class ImPlotTrackpadZoomPatch:
    """Patch the Python ImPlot module for all plots in the current context."""

    def __init__(self, zoom_per_scroll=None, wheel_scale=1.0):
        if zoom_per_scroll is None:
            zoom_per_scroll = expm1(log1p(0.035) * 1.2 * IMPLOT_SCROLL_SCALE)
        if zoom_per_scroll <= 0:
            raise ValueError('zoom_per_scroll must be positive')
        if wheel_scale <= 0:
            raise ValueError('wheel_scale must be positive')
        # ImGui may scale wheel input for dropdowns. Undo that scaling exactly
        # so the plot keeps the same zoom for each original trackpad delta.
        self.zoom_per_scroll = expm1(log1p(zoom_per_scroll) / wheel_scale)
        self._plots = {}
        self._active = []
        self._original_begin = implot.begin_plot
        self._original_end = implot.end_plot
        self._original_rate = None
        self._patched_begin = None
        self._patched_end = None

    def install(self):
        if self._patched_begin is not None:
            return self
        self._original_rate = implot.get_input_map().zoom_rate
        implot.get_input_map().zoom_rate = 0.0

        @wraps(self._original_begin)
        def begin_plot(title_id, *args, **kwargs):
            # Reapply for projects that switch between ImPlot contexts.
            implot.get_input_map().zoom_rate = 0.0
            # Keep the context wrapper alive: its Python id can change between
            # frames when the caller does not retain create_context()'s result.
            plot_id = (imgui.get_current_context(), imgui.get_id(title_id))
            if plot_id not in self._plots:
                self._plots[plot_id] = ImPlotTrackpadZoom(zoom_per_scroll=self.zoom_per_scroll)
            zoom = self._plots[plot_id]
            wheel = imgui.get_io().mouse_wheel
            mouse = imgui.get_mouse_pos()
            zoom.on_scroll(wheel, (mouse.x, mouse.y))
            zoom.before_plot()
            opened = self._original_begin(title_id, *args, **kwargs)
            if opened:
                self._active.append(zoom)
            return opened

        @wraps(self._original_end)
        def end_plot():
            if self._active:
                self._active.pop().after_plot()
            self._original_end()

        self._patched_begin = begin_plot
        self._patched_end = end_plot
        implot.begin_plot = begin_plot
        implot.end_plot = end_plot
        return self

    def uninstall(self):
        """Restore the original Python functions and ImPlot wheel rate."""
        if self._patched_begin is None:
            return
        if implot.begin_plot is self._patched_begin:
            implot.begin_plot = self._original_begin
        if implot.end_plot is self._patched_end:
            implot.end_plot = self._original_end
        implot.get_input_map().zoom_rate = self._original_rate
        self._patched_begin = None
        self._patched_end = None
        self._plots.clear()
        self._active.clear()
        global _installed_plot_patch
        if _installed_plot_patch is self:
            _installed_plot_patch = None


_installed_plot_patch = None


def install_implot_trackpad_zoom(zoom_per_scroll=None, wheel_scale=1.0):
    """Install once after context creation; compensate for any ImGui wheel scale."""
    global _installed_plot_patch
    if _installed_plot_patch is None:
        _installed_plot_patch = ImPlotTrackpadZoomPatch(zoom_per_scroll, wheel_scale).install()
    return _installed_plot_patch



class ImPlot3DWheelPatch:
    """Apply ImPlot3D's own wheel sensitivity during Python plots."""

    def __init__(self, scroll_patch):
        self.scroll_patch = scroll_patch
        self._original_begin = implot3d.begin_plot
        self._original_end = implot3d.end_plot
        self._patched_begin = None
        self._patched_end = None
        self._active = []

    def install(self):
        if self._patched_begin is not None:
            return self

        @wraps(self._original_begin)
        def begin_plot(*args, **kwargs):
            wheel = self.scroll_patch.original_wheel(scale=0.6 * IMPLOT3D_SCROLL_SCALE)
            wheel.__enter__()
            try:
                opened = self._original_begin(*args, **kwargs)
            except BaseException:
                wheel.__exit__(None, None, None)
                raise
            if opened:
                self._active.append(wheel)
            else:
                wheel.__exit__(None, None, None)
            return opened

        @wraps(self._original_end)
        def end_plot(*args, **kwargs):
            wheel = self._active.pop() if self._active else None
            try:
                return self._original_end(*args, **kwargs)
            finally:
                if wheel is not None:
                    wheel.__exit__(None, None, None)

        self._patched_begin = begin_plot
        self._patched_end = end_plot
        implot3d.begin_plot = begin_plot
        implot3d.end_plot = end_plot
        return self

    def uninstall(self):
        if self._patched_begin is None:
            return
        if implot3d.begin_plot is self._patched_begin:
            implot3d.begin_plot = self._original_begin
        if implot3d.end_plot is self._patched_end:
            implot3d.end_plot = self._original_end
        self._patched_begin = None
        self._patched_end = None
        self._active.clear()
        global _installed_plot3d_patch
        if _installed_plot3d_patch is self:
            _installed_plot3d_patch = None


_installed_plot3d_patch = None


def install_implot3d_wheel_compensation(scroll_patch):
    """Install once, including before an ImPlot3D context exists."""
    global _installed_plot3d_patch
    if _installed_plot3d_patch is None:
        _installed_plot3d_patch = ImPlot3DWheelPatch(scroll_patch).install()
    return _installed_plot3d_patch


def install_trackpad_patches(scroll_scale=None):
    """Install the Python backend fixes on macOS; return the scroll patch."""
    if sys.platform != "darwin":
        return None
    scroll_patch = install_imgui_scroll_scale(scroll_scale)
    install_implot3d_wheel_compensation(scroll_patch)
    if implot.get_current_context() is not None:
        install_implot_trackpad_zoom(wheel_scale=scroll_patch.scale)
    return scroll_patch
