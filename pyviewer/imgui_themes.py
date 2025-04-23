from imgui_bundle import imgui
import re

""" Convenience consts and functions to make copy-pasting C++ style configs easier """

# https://github.com/pthom/litgen/blob/48f2f521/src/codemanip/code_utils.py#L228
def to_snake_case(name: str) -> str:
    if "re1" not in to_snake_case.__dict__:
        to_snake_case.re1 = re.compile("(.)([A-Z][a-z]+)")  # type: ignore
        to_snake_case.re2 = re.compile("__([A-Z])")  # type: ignore
        to_snake_case.re3 = re.compile("([a-z0-9])([A-Z])")  # type: ignore

    name = to_snake_case.re1.sub(r"\1_\2", name)  # type: ignore
    name = to_snake_case.re2.sub(r"_\1", name)  # type: ignore
    name = to_snake_case.re3.sub(r"\1_\2", name)  # type: ignore

    return name.lower()

# Color preview:
# 1. install vscode extension 'json-color-token'
# 2. set value "jsonColorToken.languages": ["json", "jsonc", "python"]
def color(hex: str):
    hex = hex.lower().lstrip('#')
    if len(hex) == 6:
        return [int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]
    else:
        return [int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6)]
    
def color_uint(hex: str):
    floats = color(hex)
    return [int(f * 255) for f in floats]

def set_color(s: imgui.Style, name: str, value):
    name = name.replace('ImGuiCol_', '')
    name = to_snake_case(name)
    col = getattr(imgui.Col_, name, None)
    if col is None:
        print(f'Ignoring unknown style var "{name}"')
    else:
        s.set_color_(col, value)

# Photoshop style by Derydoca from ImThemes (https://github.com/Patitotective/ImThemes/releases)
def theme_ps():
    s = imgui.get_style()

    def set(name, value):
        name = to_snake_case(name)
        assert hasattr(s, name), f"Style has no attribute {name}"
        setattr(s, name, value)

    set('Alpha', 1.0)
    #set('DisabledAlpha', 0.6000000238418579)
    set('WindowPadding', (8.0, 8.0))
    set('WindowRounding', 4.0)
    set('WindowBorderSize', 1.0)
    set('WindowMinSize', (32.0, 32.0))
    set('WindowTitleAlign', (0.0, 0.5))
    #set('WindowMenuButtonPosition', imgui.DIRECTION_LEFT)
    set('ChildRounding', 4.0)
    set('ChildBorderSize', 1.0)
    set('PopupRounding', 2.0)
    set('PopupBorderSize', 1.0)
    set('FramePadding', (4.0, 3.0))
    set('FrameRounding', 2.0)
    set('FrameBorderSize', 1.0)
    set('ItemSpacing', (8.0, 4.0))
    set('ItemInnerSpacing', (4.0, 4.0))
    #set('CellPadding', (4.0, 2.0))
    set('IndentSpacing', 21.0)
    set('ColumnsMinSpacing', 6.0)
    set('ScrollbarSize', 13.0)
    set('ScrollbarRounding', 12.0)
    set('GrabMinSize', 7.0)
    set('GrabRounding', 0.0)
    #set('TabRounding', 0.0)
    #set('TabBorderSize', 1.0)
    #set('TabMinWidthForCloseButton', 0.0)
    #set('ColorButtonPosition', imgui.DIRECTION_RIGHT)
    set('ButtonTextAlign', (0.5, 0.5))
    #set('SelectableTextAlign', (0.0, 0.0))
    
    set_color(s, 'ImGuiCol_Text',                  (1.0, 1.0, 1.0, 1.0))
    set_color(s, 'ImGuiCol_TextDisabled',          (0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0))
    set_color(s, 'ImGuiCol_WindowBg',              (0.1764705926179886, 0.1764705926179886, 0.1764705926179886, 1.0))
    set_color(s, 'ImGuiCol_ChildBg',               (0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 0.0))
    set_color(s, 'ImGuiCol_PopupBg',               (0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0))
    set_color(s, 'ImGuiCol_Border',                (0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0))
    set_color(s, 'ImGuiCol_BorderShadow',          (0.0, 0.0, 0.0, 0.0))
    set_color(s, 'ImGuiCol_FrameBg',               (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0))
    set_color(s, 'ImGuiCol_FrameBgHovered',        (0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 1.0))
    set_color(s, 'ImGuiCol_FrameBgActive',         (0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 1.0))
    set_color(s, 'ImGuiCol_TitleBg',               (0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0))
    set_color(s, 'ImGuiCol_TitleBgActive',         (0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0))
    set_color(s, 'ImGuiCol_TitleBgCollapsed',      (0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0))
    set_color(s, 'ImGuiCol_MenuBarBg',             (0.1921568661928177, 0.1921568661928177, 0.1921568661928177, 1.0))
    set_color(s, 'ImGuiCol_ScrollbarBg',           (0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0))
    set_color(s, 'ImGuiCol_ScrollbarGrab',         (0.2745098173618317, 0.2745098173618317, 0.2745098173618317, 1.0))
    set_color(s, 'ImGuiCol_ScrollbarGrabHovered',  (0.2980392277240753, 0.2980392277240753, 0.2980392277240753, 1.0))
    set_color(s, 'ImGuiCol_ScrollbarGrabActive',   (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_CheckMark',             (1.0, 1.0, 1.0, 1.0))
    set_color(s, 'ImGuiCol_SliderGrab',            (0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0))
    set_color(s, 'ImGuiCol_SliderGrabActive',      (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_Button',                (1.0, 1.0, 1.0, 0.0))
    set_color(s, 'ImGuiCol_ButtonHovered',         (1.0, 1.0, 1.0, 0.1560000032186508))
    set_color(s, 'ImGuiCol_ButtonActive',          (1.0, 1.0, 1.0, 0.3910000026226044))
    set_color(s, 'ImGuiCol_Header',                (0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0))
    set_color(s, 'ImGuiCol_HeaderHovered',         (0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0))
    set_color(s, 'ImGuiCol_HeaderActive',          (0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0))
    set_color(s, 'ImGuiCol_Separator',             (0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0))
    set_color(s, 'ImGuiCol_SeparatorHovered',      (0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0))
    set_color(s, 'ImGuiCol_SeparatorActive',       (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_ResizeGrip',            (1.0, 1.0, 1.0, 0.25))
    set_color(s, 'ImGuiCol_ResizeGripHovered',     (1.0, 1.0, 1.0, 0.6700000166893005))
    set_color(s, 'ImGuiCol_ResizeGripActive',      (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_PlotLines',             (0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0))
    set_color(s, 'ImGuiCol_PlotLinesHovered',      (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_PlotHistogram',         (0.5843137502670288, 0.5843137502670288, 0.5843137502670288, 1.0))
    set_color(s, 'ImGuiCol_PlotHistogramHovered',  (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_TextSelectedBg',        (1.0, 1.0, 1.0, 0.1560000032186508))
    set_color(s, 'ImGuiCol_DragDropTarget',        (1.0, 0.3882353007793427, 0.0, 1.0))
    #set_color(s, 'ImGuiCol_NavHighlight',          (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_NavWindowingHighlight', (1.0, 0.3882353007793427, 0.0, 1.0))
    set_color(s, 'ImGuiCol_NavWindowingDimBg',     (0.0, 0.0, 0.0, 0.5860000252723694))
    set_color(s, 'ImGuiCol_ModalWindowDimBg',      (0.0, 0.0, 0.0, 0.5860000252723694))

# https://github.com/ocornut/imgui/issues/707#issuecomment-917151020
def theme_deep_dark():
    s = imgui.get_style()

    def set(s, name, value):
        name = to_snake_case(name)
        assert hasattr(s, name), f"Style has no attribute {name}"
        setattr(s, name, value)

    # set(s, 'WindowPadding',     imgui.Vec2(8.00, 8.00))
    # set(s, 'FramePadding',      imgui.Vec2(5.00, 2.00))
    # #set(s, 'CellPadding',       imgui.Vec2(6.00, 6.00))
    # set(s, 'ItemSpacing',       imgui.Vec2(6.00, 6.00))
    # set(s, 'ItemInnerSpacing',  imgui.Vec2(6.00, 6.00))
    # set(s, 'TouchExtraPadding', imgui.Vec2(0.00, 0.00))
    # set(s, 'IndentSpacing',     25)
    # set(s, 'ScrollbarSize',     15)
    #set(s, 'GrabMinSize',       10)
    set(s, 'WindowBorderSize',  1)
    set(s, 'ChildBorderSize',   1)
    set(s, 'PopupBorderSize',   1)
    set(s, 'FrameBorderSize',   1)
    # #set(s, 'TabBorderSize',     1)
    # set(s, 'WindowRounding',    7)
    set(s, 'ChildRounding',     4)
    set(s, 'FrameRounding',     3)
    set(s, 'PopupRounding',     4)
    set(s, 'ScrollbarRounding', 5)
    set(s, 'GrabRounding',      3)
    # #set(s, 'LogSliderDeadzone', 4)
    # #set(s, 'TabRounding',       4)
    
    set_color(s, 'ImGuiCol_Text',                   (1.00, 1.00, 1.00, 1.00))
    set_color(s, 'ImGuiCol_TextDisabled',           (0.50, 0.50, 0.50, 1.00))
    set_color(s, 'ImGuiCol_WindowBg',               (0.10, 0.10, 0.10, 1.00))
    set_color(s, 'ImGuiCol_ChildBg',                (0.00, 0.00, 0.00, 0.00))
    set_color(s, 'ImGuiCol_PopupBg',                (0.19, 0.19, 0.19, 0.92))
    set_color(s, 'ImGuiCol_Border',                 (0.19, 0.19, 0.19, 0.29))
    set_color(s, 'ImGuiCol_BorderShadow',           (0.00, 0.00, 0.00, 0.24))
    set_color(s, 'ImGuiCol_FrameBg',                (0.05, 0.05, 0.05, 0.54))
    set_color(s, 'ImGuiCol_FrameBgHovered',         (0.19, 0.19, 0.19, 0.54))
    set_color(s, 'ImGuiCol_FrameBgActive',          (0.20, 0.22, 0.23, 1.00))
    set_color(s, 'ImGuiCol_TitleBg',                (0.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_TitleBgActive',          (0.06, 0.06, 0.06, 1.00))
    set_color(s, 'ImGuiCol_TitleBgCollapsed',       (0.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_MenuBarBg',              (0.14, 0.14, 0.14, 1.00))
    set_color(s, 'ImGuiCol_ScrollbarBg',            (0.05, 0.05, 0.05, 0.54))
    set_color(s, 'ImGuiCol_ScrollbarGrab',          (0.34, 0.34, 0.34, 0.54))
    set_color(s, 'ImGuiCol_ScrollbarGrabHovered',   (0.40, 0.40, 0.40, 0.54))
    set_color(s, 'ImGuiCol_ScrollbarGrabActive',    (0.56, 0.56, 0.56, 0.54))
    set_color(s, 'ImGuiCol_CheckMark',              (0.33, 0.67, 0.86, 1.00))
    set_color(s, 'ImGuiCol_SliderGrab',             (0.34, 0.34, 0.34, 0.54))
    set_color(s, 'ImGuiCol_SliderGrabActive',       (0.56, 0.56, 0.56, 0.54))
    set_color(s, 'ImGuiCol_Button',                 (0.05, 0.05, 0.05, 0.54))
    set_color(s, 'ImGuiCol_ButtonHovered',          (0.19, 0.19, 0.19, 0.54))
    set_color(s, 'ImGuiCol_ButtonActive',           (0.20, 0.22, 0.23, 1.00))
    set_color(s, 'ImGuiCol_Header',                 (0.00, 0.00, 0.00, 0.52))
    set_color(s, 'ImGuiCol_HeaderHovered',          (0.00, 0.00, 0.00, 0.36))
    set_color(s, 'ImGuiCol_HeaderActive',           (0.20, 0.22, 0.23, 0.33))
    set_color(s, 'ImGuiCol_Separator',              (0.28, 0.28, 0.28, 0.29))
    set_color(s, 'ImGuiCol_SeparatorHovered',       (0.44, 0.44, 0.44, 0.29))
    set_color(s, 'ImGuiCol_SeparatorActive',        (0.40, 0.44, 0.47, 1.00))
    set_color(s, 'ImGuiCol_ResizeGrip',             (0.28, 0.28, 0.28, 0.29))
    set_color(s, 'ImGuiCol_ResizeGripHovered',      (0.44, 0.44, 0.44, 0.29))
    set_color(s, 'ImGuiCol_ResizeGripActive',       (0.40, 0.44, 0.47, 1.00))
    set_color(s, 'ImGuiCol_Tab',                    (0.00, 0.00, 0.00, 0.52))
    set_color(s, 'ImGuiCol_TabHovered',             (0.14, 0.14, 0.14, 1.00))
    #set_color(s, 'ImGuiCol_TabActive',              (0.20, 0.20, 0.20, 0.36))
    #set_color(s, 'ImGuiCol_TabUnfocused',           (0.00, 0.00, 0.00, 0.52))
    #set_color(s, 'ImGuiCol_TabUnfocusedActive',     (0.14, 0.14, 0.14, 1.00))
    set_color(s, 'ImGuiCol_DockingPreview',         (0.33, 0.67, 0.86, 1.00))
    set_color(s, 'ImGuiCol_DockingEmptyBg',         (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_PlotLines',              (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_PlotLinesHovered',       (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_PlotHistogram',          (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_PlotHistogramHovered',   (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_TableHeaderBg',          (0.00, 0.00, 0.00, 0.52))
    set_color(s, 'ImGuiCol_TableBorderStrong',      (0.00, 0.00, 0.00, 0.52))
    set_color(s, 'ImGuiCol_TableBorderLight',       (0.28, 0.28, 0.28, 0.29))
    set_color(s, 'ImGuiCol_TableRowBg',             (0.00, 0.00, 0.00, 0.00))
    set_color(s, 'ImGuiCol_TableRowBgAlt',          (1.00, 1.00, 1.00, 0.06))
    set_color(s, 'ImGuiCol_TextSelectedBg',         (0.20, 0.22, 0.23, 1.00))
    set_color(s, 'ImGuiCol_DragDropTarget',         (0.33, 0.67, 0.86, 1.00))
    #set_color(s, 'ImGuiCol_NavHighlight',           (1.00, 0.00, 0.00, 1.00))
    set_color(s, 'ImGuiCol_NavWindowingHighlight',  (1.00, 0.00, 0.00, 0.70))
    set_color(s, 'ImGuiCol_NavWindowingDimBg',      (1.00, 0.00, 0.00, 0.20))
    set_color(s, 'ImGuiCol_ModalWindowDimBg',       (1.00, 0.00, 0.00, 0.35))

    # Docking-related
    s.window_padding = (3, 3)
    s.tab_rounding = 0
    s.set_color_(imgui.Col_.tab_dimmed_selected, (39/255, 44/255, 54/255, 1))
    s.set_color_(imgui.Col_.tab_selected, (39/255, 44/255, 54/255, 1))
    s.set_color_(imgui.Col_.tab, (39/255, 44/255, 54/255, 1))
    s.set_color_(imgui.Col_.title_bg, (15/255, 15/255, 15/255, 1))

# Deep-dark with ps-style borders
def theme_contrast():
    theme_deep_dark()
    s = imgui.get_style()
    s.set_color_(imgui.Col_.border, (0.26, 0.26, 0.26, 1.0))
    s.set_color_(imgui.Col_.border_shadow, (0.0, 0.0, 0.0, 0.0))
    s.child_border_size = 1.0
    s.popup_border_size = 1.0
    s.frame_border_size = 1.0

# https://github.com/ocornut/imgui/issues/707#issuecomment-678611331
def theme_dark_overshifted():
    s = imgui.get_style()
    
    set_color(s, 'ImGuiCol_Text',                  (1.00, 1.00, 1.00, 1.00))
    set_color(s, 'ImGuiCol_TextDisabled',          (0.50, 0.50, 0.50, 1.00))
    set_color(s, 'ImGuiCol_WindowBg',              (0.13, 0.14, 0.15, 1.00))
    set_color(s, 'ImGuiCol_ChildBg',               (0.13, 0.14, 0.15, 1.00))
    set_color(s, 'ImGuiCol_PopupBg',               (0.13, 0.14, 0.15, 1.00))
    set_color(s, 'ImGuiCol_Border',                (0.43, 0.43, 0.50, 0.50))
    set_color(s, 'ImGuiCol_BorderShadow',          (0.00, 0.00, 0.00, 0.00))
    set_color(s, 'ImGuiCol_FrameBg',               (0.25, 0.25, 0.25, 1.00))
    set_color(s, 'ImGuiCol_FrameBgHovered',        (0.38, 0.38, 0.38, 1.00))
    set_color(s, 'ImGuiCol_FrameBgActive',         (0.67, 0.67, 0.67, 0.39))
    set_color(s, 'ImGuiCol_TitleBg',               (0.08, 0.08, 0.09, 1.00))
    set_color(s, 'ImGuiCol_TitleBgActive',         (0.08, 0.08, 0.09, 1.00))
    set_color(s, 'ImGuiCol_TitleBgCollapsed',      (0.00, 0.00, 0.00, 0.51))
    set_color(s, 'ImGuiCol_MenuBarBg',             (0.14, 0.14, 0.14, 1.00))
    set_color(s, 'ImGuiCol_ScrollbarBg',           (0.02, 0.02, 0.02, 0.53))
    set_color(s, 'ImGuiCol_ScrollbarGrab',         (0.31, 0.31, 0.31, 1.00))
    set_color(s, 'ImGuiCol_ScrollbarGrabHovered',  (0.41, 0.41, 0.41, 1.00))
    set_color(s, 'ImGuiCol_ScrollbarGrabActive',   (0.51, 0.51, 0.51, 1.00))
    set_color(s, 'ImGuiCol_CheckMark',             (0.11, 0.64, 0.92, 1.00))
    set_color(s, 'ImGuiCol_SliderGrab',            (0.11, 0.64, 0.92, 1.00))
    set_color(s, 'ImGuiCol_SliderGrabActive',      (0.08, 0.50, 0.72, 1.00))
    set_color(s, 'ImGuiCol_Button',                (0.25, 0.25, 0.25, 1.00))
    set_color(s, 'ImGuiCol_ButtonHovered',         (0.38, 0.38, 0.38, 1.00))
    set_color(s, 'ImGuiCol_ButtonActive',          (0.67, 0.67, 0.67, 0.39))
    set_color(s, 'ImGuiCol_Header',                (0.22, 0.22, 0.22, 1.00))
    set_color(s, 'ImGuiCol_HeaderHovered',         (0.25, 0.25, 0.25, 1.00))
    set_color(s, 'ImGuiCol_HeaderActive',          (0.67, 0.67, 0.67, 0.39))
    set_color(s, 'ImGuiCol_Separator',             s.color_(imgui.Col_.border))
    set_color(s, 'ImGuiCol_SeparatorHovered',      (0.41, 0.42, 0.44, 1.00))
    set_color(s, 'ImGuiCol_SeparatorActive',       (0.26, 0.59, 0.98, 0.95))
    set_color(s, 'ImGuiCol_ResizeGrip',            (0.00, 0.00, 0.00, 0.00))
    set_color(s, 'ImGuiCol_ResizeGripHovered',     (0.29, 0.30, 0.31, 0.67))
    set_color(s, 'ImGuiCol_ResizeGripActive',      (0.26, 0.59, 0.98, 0.95))
    #set_color(s, 'ImGuiCol_Tab',                   (0.08, 0.08, 0.09, 0.83))
    #set_color(s, 'ImGuiCol_TabHovered',            (0.33, 0.34, 0.36, 0.83))
    #set_color(s, 'ImGuiCol_TabActive',             (0.23, 0.23, 0.24, 1.00))
    #set_color(s, 'ImGuiCol_TabUnfocused',          (0.08, 0.08, 0.09, 1.00))
    #set_color(s, 'ImGuiCol_TabUnfocusedActive',    (0.13, 0.14, 0.15, 1.00))
    #set_color(s, 'ImGuiCol_DockingPreview',        (0.26, 0.59, 0.98, 0.70))
    #set_color(s, 'ImGuiCol_DockingEmptyBg',        (0.20, 0.20, 0.20, 1.00))
    set_color(s, 'ImGuiCol_PlotLines',             (0.61, 0.61, 0.61, 1.00))
    set_color(s, 'ImGuiCol_PlotLinesHovered',      (1.00, 0.43, 0.35, 1.00))
    set_color(s, 'ImGuiCol_PlotHistogram',         (0.90, 0.70, 0.00, 1.00))
    set_color(s, 'ImGuiCol_PlotHistogramHovered',  (1.00, 0.60, 0.00, 1.00))
    set_color(s, 'ImGuiCol_TextSelectedBg',        (0.26, 0.59, 0.98, 0.35))
    set_color(s, 'ImGuiCol_DragDropTarget',        (0.11, 0.64, 0.92, 1.00))
    #set_color(s, 'ImGuiCol_NavHighlight',          (0.26, 0.59, 0.98, 1.00))
    set_color(s, 'ImGuiCol_NavWindowingHighlight', (1.00, 1.00, 1.00, 0.70))
    set_color(s, 'ImGuiCol_NavWindowingDimBg',     (0.80, 0.80, 0.80, 0.20))
    set_color(s, 'ImGuiCol_ModalWindowDimBg',      (0.80, 0.80, 0.80, 0.35))

# Based on style_colors_dark()
def theme_custom():
    # Values from: https://github.com/ocornut/imgui/blob/v1.65/imgui_draw.cpp#L170

    s = imgui.get_style()
    s.frame_padding         = [4, 3]
    s.window_border_size    = 1
    s.child_border_size     = 0
    s.popup_border_size     = 0
    s.frame_border_size     = 0
    s.window_rounding       = 0
    s.child_rounding        = 0
    s.popup_rounding        = 0
    s.frame_rounding        = 0
    s.scrollbar_rounding    = 0
    s.grab_rounding         = 0

    # Converted with matplotlib.colors.to_hex(color, keep_alpha=True)
    set_color(s, 'ImGuiCol_Text',                  color('#ffffffff'))
    set_color(s, 'ImGuiCol_TextDisabled',          color('#808080ff'))
    set_color(s, 'ImGuiCol_WindowBg',              color('#0f0f0ff0'))
    set_color(s, 'ImGuiCol_ChildBg',               color('#ffffff00'))
    set_color(s, 'ImGuiCol_PopupBg',               color('#141414f0'))
    set_color(s, 'ImGuiCol_Border',                color('#6e6e8080'))
    set_color(s, 'ImGuiCol_BorderShadow',          color('#00000000'))
    set_color(s, 'ImGuiCol_FrameBg',               color('#294a7a8a'))
    set_color(s, 'ImGuiCol_FrameBgHovered',        color('#4296fa66'))
    set_color(s, 'ImGuiCol_FrameBgActive',         color('#4296faab'))
    set_color(s, 'ImGuiCol_TitleBg',               color('#0a0a0aff'))
    set_color(s, 'ImGuiCol_TitleBgActive',         color('#294a7aff'))
    set_color(s, 'ImGuiCol_TitleBgCollapsed',      color('#00000082'))
    set_color(s, 'ImGuiCol_MenuBarBg',             color('#242424ff'))
    set_color(s, 'ImGuiCol_ScrollbarBg',           color('#05050587'))
    set_color(s, 'ImGuiCol_ScrollbarGrab',         color('#4f4f4fff'))
    set_color(s, 'ImGuiCol_ScrollbarGrabHovered',  color('#696969ff'))
    set_color(s, 'ImGuiCol_ScrollbarGrabActive',   color('#828282ff'))
    set_color(s, 'ImGuiCol_CheckMark',             color('#4296faff'))
    set_color(s, 'ImGuiCol_SliderGrab',            color('#3d85e0ff'))
    set_color(s, 'ImGuiCol_SliderGrabActive',      color('#4296faff'))
    set_color(s, 'ImGuiCol_Button',                color('#4296fa66'))
    set_color(s, 'ImGuiCol_ButtonHovered',         color('#4296faff'))
    set_color(s, 'ImGuiCol_ButtonActive',          color('#0f87faff'))
    set_color(s, 'ImGuiCol_Header',                color('#4296fa4f'))
    set_color(s, 'ImGuiCol_HeaderHovered',         color('#4296facc'))
    set_color(s, 'ImGuiCol_HeaderActive',          color('#4296faff'))
    set_color(s, 'ImGuiCol_Separator',             s.color_(imgui.Col_.border))
    set_color(s, 'ImGuiCol_SeparatorHovered',      color('#1a66bfc7'))
    set_color(s, 'ImGuiCol_SeparatorActive',       color('#1a66bfff'))
    set_color(s, 'ImGuiCol_ResizeGrip',            color('#4296fa40'))
    set_color(s, 'ImGuiCol_ResizeGripHovered',     color('#4296faab'))
    set_color(s, 'ImGuiCol_ResizeGripActive',      color('#4296faf2'))
    set_color(s, 'ImGuiCol_PlotLines',             color('#9c9c9cff'))
    set_color(s, 'ImGuiCol_PlotLinesHovered',      color('#ff6e59ff'))
    set_color(s, 'ImGuiCol_PlotHistogram',         color('#e6b200ff'))
    set_color(s, 'ImGuiCol_PlotHistogramHovered',  color('#ff9900ff'))
    set_color(s, 'ImGuiCol_TextSelectedBg',        color('#4296fa59'))
    set_color(s, 'ImGuiCol_DragDropTarget',        color('#BABA267D'))
    #set_color(s, 'ImGuiCol_NavHighlight',          color('#4296faff'))
    set_color(s, 'ImGuiCol_NavWindowingHighlight', color('#ffffffb2'))
    set_color(s, 'ImGuiCol_NavWindowingDimBg',     color('#cccccc33'))
    set_color(s, 'ImGuiCol_ModalWindowDimBg',      color('#cccccc59'))