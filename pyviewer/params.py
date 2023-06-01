import imgui
from pyviewer.utils import enum_slider, combo_box_vals, slider_range_int, slider_range_float, strict_dataclass
from typing import Tuple

""" Small param wrappers for automatically creating UI widgets """

class Param:
    def __init__(self, type, label: str, default_val, tooltip: str = None) -> None:
        self.type = type
        self.label = label
        self.default = default_val
        self.value = default_val
        self.tooltip = tooltip
    
    def draw_tooltip(self):
        if self.tooltip and imgui.is_item_hovered():
            imgui.set_tooltip(self.tooltip)
    
    def draw_widget(self):
        raise NotImplementedError()
    
    def reset(self):
        self.value = self.default

    def draw(self):
        _, self.value = self.draw_widget()
        self.draw_tooltip()

class _RangeParam(Param):
    def __init__(self, type, label, default_val, minval, maxval, tooltip: str = None) -> None:
        super().__init__(type, label, default_val, tooltip)
        self.min = minval
        self.max = maxval

class _Range2Param(Param):
    def __init__(self, type, label, default_val, minval, maxval, overlap: bool = True, tooltip: str = None) -> None:
        super().__init__(type, label, default_val, tooltip)
        self.min = minval
        self.max = maxval
        self.overlap = overlap  # allow v2 to go below v1?

class EnumParam(Param):
    def __init__(self, label, default_val, valid_vals=(), tooltip: str = None) -> None:
        super().__init__(type(default_val), label, default_val, tooltip)
        self.opts = list(valid_vals)

    def draw_widget(self):
        return combo_box_vals(self.label, self.opts, self.value)

class EnumSliderParam(Param):
    def __init__(self, label, default_val, valid_vals=(), tooltip: str = None) -> None:
        super().__init__(type(default_val), label, default_val, tooltip)
        self.opts = list(valid_vals)

    def draw_widget(self):
        return enum_slider(self.label, self.opts, self.value)

class BoolParam(Param):
    def __init__(self, label, default_val: bool, tooltip: str = None) -> None:
        super().__init__(bool, label, default_val, tooltip)
    
    def draw_widget(self):
        return imgui.checkbox(self.label, self.value)

class IntParam(_RangeParam):
    def __init__(self, label, default_val: int, minval, maxval, buttons=False, tooltip: str = None) -> None:
        super().__init__(int, label, default_val, minval, maxval, tooltip)
        self.buttons = buttons
    
    def draw_widget(self):
        changed, val = imgui.slider_int(self.label, self.value, self.min, self.max)
        if self.buttons:
            imgui.same_line()
            if imgui.button(f'<##{self.label}'):
                changed, val = (True, max(self.min, min(self.max, val - 1)))
            imgui.same_line()
            if imgui.button(f'>##{self.label}'):
                changed, val = (True, max(self.min, min(self.max, val + 1)))

        return changed, val

class Int2Param(_Range2Param):
    def __init__(self, label, default_val: Tuple[int], minval, maxval, overlap: bool = True, tooltip: str = None) -> None:
        super().__init__(Tuple[int], label, default_val, minval, maxval, overlap, tooltip)
    
    def draw_widget(self):
        if not self.overlap:
            return slider_range_int(*self.value, self.min, self.max, title=self.label)
        else:
            return imgui.slider_int2(self.label, *self.value, self.min, self.max)

class FloatParam(_RangeParam):
    def __init__(self, label, default_val: float, minval, maxval, tooltip: str = None) -> None:
        super().__init__(float, label, default_val, minval, maxval, tooltip)
    
    def draw_widget(self):
        return imgui.slider_float(self.label, self.value, self.min, self.max)
    
class Float2Param(_Range2Param):
    def __init__(self, label, default_val: Tuple[float], minval, maxval, overlap: bool = True, tooltip: str = None) -> None:
        super().__init__(Tuple[float], label, default_val, minval, maxval, overlap, tooltip)
    
    def draw_widget(self):
        if not self.overlap:
            return slider_range_float(*self.value, self.min, self.max, title=self.label)
        else:
            return imgui.slider_float2(self.label, *self.value, self.min, self.max)
    
##########################################
# Container that exposes raw values

@strict_dataclass
class ParamContainer:
    def __iter__(self):
        for attr, _ in self.__dataclass_fields__.items():
            yield attr, super().__getattribute__(attr)
    
    def __getattribute__(self, __name: str):
        obj = super().__getattribute__(__name)
        if isinstance(obj, Param):
            return obj.value
        else:
            return obj
    
    def __setattr__(self, __name: str, __value) -> None:
        obj = super().__getattribute__(__name)
        if isinstance(obj, Param) and not isinstance(__value, Param):
            obj.value = __value
        else:
            super().__setattr__(__name, __value)