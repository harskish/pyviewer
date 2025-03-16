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

# ==== Color identifiers for styling ====
# https://github.com/pyimgui/pyimgui/blob/1.4.0/imgui/core.pyx#L193
# ImGuiCol_Text = imgui.Col_.text
# ImGuiCol_TextDisabled = imgui.Col_.text_disabled
# ImGuiCol_WindowBg = imgui.Col_.window_bg
# ImGuiCol_ChildBg = imgui.Col_.child_bg
# ImGuiCol_PopupBg = imgui.Col_.popup_bg
# ImGuiCol_Border = imgui.Col_.border
# ImGuiCol_BorderShadow = imgui.Col_.border_shadow
# ImGuiCol_FrameBg = imgui.Col_.frame_bg
# ImGuiCol_FrameBgHovered = imgui.Col_.frame_bg_hovered
# ImGuiCol_FrameBgActive = imgui.Col_.frame_bg_active
# ImGuiCol_TitleBg = imgui.Col_.title_bg
# ImGuiCol_TitleBgActive = imgui.Col_.title_bg_active
# ImGuiCol_TitleBgCollapsed = imgui.Col_.title_bg_collapsed
# ImGuiCol_MenuBarBg = imgui.Col_.menu_bar_bg
# ImGuiCol_ScrollbarBg = imgui.Col_.scrollbar_bg
# ImGuiCol_ScrollbarGrab = imgui.Col_.scrollbar_grab
# ImGuiCol_ScrollbarGrabHovered = imgui.Col_.scrollbar_grab_hovered
# ImGuiCol_ScrollbarGrabActive = imgui.Col_.scrollbar_grab_active
# ImGuiCol_CheckMark = imgui.Col_.check_mark
# ImGuiCol_SliderGrab = imgui.Col_.slider_grab
# ImGuiCol_SliderGrabActive = imgui.Col_.slider_grab_active
# ImGuiCol_Button = imgui.Col_.button
# ImGuiCol_ButtonHovered = imgui.Col_.button_hovered
# ImGuiCol_ButtonActive = imgui.Col_.button_active
# ImGuiCol_Header = imgui.Col_.header
# ImGuiCol_HeaderHovered = imgui.Col_.header_hovered
# ImGuiCol_HeaderActive = imgui.Col_.header_active
# ImGuiCol_Separator = imgui.Col_.separator
# ImGuiCol_SeparatorHovered = imgui.Col_.separator_hovered
# ImGuiCol_SeparatorActive = imgui.Col_.separator_active
# ImGuiCol_ResizeGrip = imgui.Col_.resize_grip
# ImGuiCol_ResizeGripHovered = imgui.Col_.resize_grip_hovered
# ImGuiCol_ResizeGripActive = imgui.Col_.resize_grip_active
# ImGuiCol_PlotLines = imgui.Col_.plot_lines
# ImGuiCol_PlotLinesHovered = imgui.Col_.plot_lines_hovered
# ImGuiCol_PlotHistogram = imgui.Col_.plot_histogram
# ImGuiCol_PlotHistogramHovered = imgui.Col_.plot_histogram_hovered
# ImGuiCol_TextSelectedBg = imgui.Col_.text_selected_bg
# ImGuiCol_DragDropTarget = imgui.Col_.drag_drop_target
# ImGuiCol_NavHighlight = imgui.Col_.nav_cursor # TODO: chekc
# ImGuiCol_NavWindowingHighlight = imgui.Col_.nav_windowing_highlight
# ImGuiCol_NavWindowingDimBg = imgui.Col_.nav_windowing_dim_bg
# ImGuiCol_ModalWindowDimBg = imgui.Col_.modal_window_dim_bg
# ImGuiCol_COUNT = imgui.Col_.count
# ImGuiCol_Tab = imgui.Col_.tab
# ImGuiCol_TabHovered = imgui.Col_.tab_hovered
# #ImGuiCol_TabActive = imgui.Col_.tab_dimmed_selected # ??
# #ImGuiCol_TabUnfocused = imgui.Col_.tab_dimmed # ??
# #ImGuiCol_TabUnfocusedActive = imgui.Col_.tab_dimmed_selected # ??
# ImGuiCol_DockingPreview = imgui.Col_.docking_preview
# ImGuiCol_DockingEmptyBg = imgui.Col_.docking_empty_bg
# ImGuiCol_TableHeaderBg = imgui.Col_.table_header_bg
# ImGuiCol_TableBorderStrong = imgui.Col_.table_border_strong
# ImGuiCol_TableBorderLight = imgui.Col_.table_border_light
# ImGuiCol_TableRowBg = imgui.Col_.table_row_bg
# ImGuiCol_TableRowBgAlt = imgui.Col_.table_row_bg_alt

# style consts in v1.4.0
# Alpha = to_snake_case('Alpha') # = 'alpha'
# AntiAliasedFill = to_snake_case('AntiAliasedFill') # = 'anti_aliased_fill'
# AntiAliasedLines = to_snake_case('AntiAliasedLines') # = 'anti_aliased_lines'
# ButtonTextAlign = to_snake_case('ButtonTextAlign') # = 'button_text_align'
# ChildBorderSize = to_snake_case('ChildBorderSize') # = 'child_border_size'
# ChildRounding = to_snake_case('ChildRounding') # = 'child_rounding'
# Color = to_snake_case('Color') # = 'color'
# ColumnsMinSpacing = to_snake_case('ColumnsMinSpacing') # = 'columns_min_spacing'
# CurveTessellationTolerance = to_snake_case('CurveTessellationTolerance') # = 'curve_tessellation_tolerance'
# DisplaySafeAreaPadding = to_snake_case('DisplaySafeAreaPadding') # = 'display_safe_area_padding'
# DisplayWindowPadding = to_snake_case('DisplayWindowPadding') # = 'display_window_padding'
# FrameBorderSize = to_snake_case('FrameBorderSize') # = 'frame_border_size'
# FramePadding = to_snake_case('FramePadding') # = 'frame_padding'
# FrameRounding = to_snake_case('FrameRounding') # = 'frame_rounding'
# GrabMinSize = to_snake_case('GrabMinSize') # = 'grab_min_size'
# GrabRounding = to_snake_case('GrabRounding') # = 'grab_rounding'
# IndentSpacing = to_snake_case('IndentSpacing') # = 'indent_spacing'
# ItemInnerSpacing = to_snake_case('ItemInnerSpacing') # = 'item_inner_spacing'
# ItemSpacing = to_snake_case('ItemSpacing') # = 'item_spacing'
# MouseCursorScale = to_snake_case('MouseCursorScale') # = 'mouse_cursor_scale'
# PopupBorderSize = to_snake_case('PopupBorderSize') # = 'popup_border_size'
# PopupRounding = to_snake_case('PopupRounding') # = 'popup_rounding'
# ScrollbarRounding = to_snake_case('ScrollbarRounding') # = 'scrollbar_rounding'
# ScrollbarSize = to_snake_case('ScrollbarSize') # = 'scrollbar_size'
# TouchExtraPadding = to_snake_case('TouchExtraPadding') # = 'touch_extra_padding'
# WindowBorderSize = to_snake_case('WindowBorderSize') # = 'window_border_size'
# WindowMinSize = to_snake_case('WindowMinSize') # = 'window_min_size'
# WindowPadding = to_snake_case('WindowPadding') # = 'window_padding'
# WindowRounding = to_snake_case('WindowRounding') # = 'window_rounding'
# WindowTitleAlign = to_snake_case('WindowTitleAlign') # = 'window_title_align'
# DisabledAlpha = ''
# WindowMenuButtonPosition = ''
# CellPadding = ''
# TabRounding = ''
# TabBorderSize = ''
# TabMinWidthForCloseButton = ''
# ColorButtonPosition = ''
# ButtonTextAlign = ''
# SelectableTextAlign = ''
# LogSliderDeadzone = ''

def color(hex):
    hex = hex.lstrip('#')
    rgba = (int(hex[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))
    return imgui.Vec4(*rgba)

# Photoshop style by Derydoca from ImThemes (https://github.com/Patitotective/ImThemes/releases)
def theme_ps():
    s = imgui.get_style()

    def set(name, value):
        name = to_snake_case(name)
        assert hasattr(s, name), f"Style has no attribute {name}"
        setattr(s, name, value)

    set(s, 'Alpha', 1.0)
    #set(s, 'DisabledAlpha', 0.6000000238418579)
    set(s, 'WindowPadding', imgui.Vec2(8.0, 8.0))
    set(s, 'WindowRounding', 4.0)
    set(s, 'WindowBorderSize', 1.0)
    set(s, 'WindowMinSize', imgui.Vec2(32.0, 32.0))
    set(s, 'WindowTitleAlign', imgui.Vec2(0.0, 0.5))
    #set(s, 'WindowMenuButtonPosition', imgui.DIRECTION_LEFT)
    set(s, 'ChildRounding', 4.0)
    set(s, 'ChildBorderSize', 1.0)
    set(s, 'PopupRounding', 2.0)
    set(s, 'PopupBorderSize', 1.0)
    set(s, 'FramePadding', imgui.Vec2(4.0, 3.0))
    set(s, 'FrameRounding', 2.0)
    set(s, 'FrameBorderSize', 1.0)
    set(s, 'ItemSpacing', imgui.Vec2(8.0, 4.0))
    set(s, 'ItemInnerSpacing', imgui.Vec2(4.0, 4.0))
    #set(s, 'CellPadding', imgui.Vec2(4.0, 2.0))
    set(s, 'IndentSpacing', 21.0)
    set(s, 'ColumnsMinSpacing', 6.0)
    set(s, 'ScrollbarSize', 13.0)
    set(s, 'ScrollbarRounding', 12.0)
    set(s, 'GrabMinSize', 7.0)
    set(s, 'GrabRounding', 0.0)
    #set(s, 'TabRounding', 0.0)
    #set(s, 'TabBorderSize', 1.0)
    #set(s, 'TabMinWidthForCloseButton', 0.0)
    #set(s, 'ColorButtonPosition', imgui.DIRECTION_RIGHT)
    set(s, 'ButtonTextAlign', imgui.Vec2(0.5, 0.5))
    #set(s, 'SelectableTextAlign', imgui.Vec2(0.0, 0.0))
    
    s.colors[ImGuiCol_Text] =                  imgui.Vec4(1.0, 1.0, 1.0, 1.0)
    s.colors[ImGuiCol_TextDisabled] =          imgui.Vec4(0.4980392158031464, 0.4980392158031464, 0.4980392158031464, 1.0)
    s.colors[ImGuiCol_WindowBg] =              imgui.Vec4(0.1764705926179886, 0.1764705926179886, 0.1764705926179886, 1.0)
    s.colors[ImGuiCol_ChildBg] =               imgui.Vec4(0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 0.0)
    s.colors[ImGuiCol_PopupBg] =               imgui.Vec4(0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0)
    s.colors[ImGuiCol_Border] =                imgui.Vec4(0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0)
    s.colors[ImGuiCol_BorderShadow] =          imgui.Vec4(0.0, 0.0, 0.0, 0.0)
    s.colors[ImGuiCol_FrameBg] =               imgui.Vec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
    s.colors[ImGuiCol_FrameBgHovered] =        imgui.Vec4(0.2000000029802322, 0.2000000029802322, 0.2000000029802322, 1.0)
    s.colors[ImGuiCol_FrameBgActive] =         imgui.Vec4(0.2784313857555389, 0.2784313857555389, 0.2784313857555389, 1.0)
    s.colors[ImGuiCol_TitleBg] =               imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_TitleBgActive] =         imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_TitleBgCollapsed] =      imgui.Vec4(0.1450980454683304, 0.1450980454683304, 0.1450980454683304, 1.0)
    s.colors[ImGuiCol_MenuBarBg] =             imgui.Vec4(0.1921568661928177, 0.1921568661928177, 0.1921568661928177, 1.0)
    s.colors[ImGuiCol_ScrollbarBg] =           imgui.Vec4(0.1568627506494522, 0.1568627506494522, 0.1568627506494522, 1.0)
    s.colors[ImGuiCol_ScrollbarGrab] =         imgui.Vec4(0.2745098173618317, 0.2745098173618317, 0.2745098173618317, 1.0)
    s.colors[ImGuiCol_ScrollbarGrabHovered] =  imgui.Vec4(0.2980392277240753, 0.2980392277240753, 0.2980392277240753, 1.0)
    s.colors[ImGuiCol_ScrollbarGrabActive] =   imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_CheckMark] =             imgui.Vec4(1.0, 1.0, 1.0, 1.0)
    s.colors[ImGuiCol_SliderGrab] =            imgui.Vec4(0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0)
    s.colors[ImGuiCol_SliderGrabActive] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_Button] =                imgui.Vec4(1.0, 1.0, 1.0, 0.0)
    s.colors[ImGuiCol_ButtonHovered] =         imgui.Vec4(1.0, 1.0, 1.0, 0.1560000032186508)
    s.colors[ImGuiCol_ButtonActive] =          imgui.Vec4(1.0, 1.0, 1.0, 0.3910000026226044)
    s.colors[ImGuiCol_Header] =                imgui.Vec4(0.3098039329051971, 0.3098039329051971, 0.3098039329051971, 1.0)
    s.colors[ImGuiCol_HeaderHovered] =         imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_HeaderActive] =          imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_Separator] =             imgui.Vec4(0.2627451121807098, 0.2627451121807098, 0.2627451121807098, 1.0)
    s.colors[ImGuiCol_SeparatorHovered] =      imgui.Vec4(0.3882353007793427, 0.3882353007793427, 0.3882353007793427, 1.0)
    s.colors[ImGuiCol_SeparatorActive] =       imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_ResizeGrip] =            imgui.Vec4(1.0, 1.0, 1.0, 0.25)
    s.colors[ImGuiCol_ResizeGripHovered] =     imgui.Vec4(1.0, 1.0, 1.0, 0.6700000166893005)
    s.colors[ImGuiCol_ResizeGripActive] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_PlotLines] =             imgui.Vec4(0.4666666686534882, 0.4666666686534882, 0.4666666686534882, 1.0)
    s.colors[ImGuiCol_PlotLinesHovered] =      imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_PlotHistogram] =         imgui.Vec4(0.5843137502670288, 0.5843137502670288, 0.5843137502670288, 1.0)
    s.colors[ImGuiCol_PlotHistogramHovered] =  imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_TextSelectedBg] =        imgui.Vec4(1.0, 1.0, 1.0, 0.1560000032186508)
    s.colors[ImGuiCol_DragDropTarget] =        imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavHighlight] =          imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavWindowingHighlight] = imgui.Vec4(1.0, 0.3882353007793427, 0.0, 1.0)
    s.colors[ImGuiCol_NavWindowingDimBg] =     imgui.Vec4(0.0, 0.0, 0.0, 0.5860000252723694)
    s.colors[ImGuiCol_ModalWindowDimBg] =      imgui.Vec4(0.0, 0.0, 0.0, 0.5860000252723694)

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
    
    #from imgui_bundle/external/bindings_generation/external_library.py

    def set_color(name: str, value):
        name = name.replace('ImGuiCol_', '')
        name = to_snake_case(name)
        col = getattr(imgui.Col_, name, None)
        if col is None:
            print(f'Ignoring unknown style var "{name}"')
        else:
            s.set_color_(col, value)
    
    set_color('ImGuiCol_Text',                   (1.00, 1.00, 1.00, 1.00))
    set_color('ImGuiCol_TextDisabled',           (0.50, 0.50, 0.50, 1.00))
    set_color('ImGuiCol_WindowBg',               (0.10, 0.10, 0.10, 1.00))
    set_color('ImGuiCol_ChildBg',                (0.00, 0.00, 0.00, 0.00))
    set_color('ImGuiCol_PopupBg',                (0.19, 0.19, 0.19, 0.92))
    set_color('ImGuiCol_Border',                 (0.19, 0.19, 0.19, 0.29))
    set_color('ImGuiCol_BorderShadow',           (0.00, 0.00, 0.00, 0.24))
    set_color('ImGuiCol_FrameBg',                (0.05, 0.05, 0.05, 0.54))
    set_color('ImGuiCol_FrameBgHovered',         (0.19, 0.19, 0.19, 0.54))
    set_color('ImGuiCol_FrameBgActive',          (0.20, 0.22, 0.23, 1.00))
    set_color('ImGuiCol_TitleBg',                (0.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_TitleBgActive',          (0.06, 0.06, 0.06, 1.00))
    set_color('ImGuiCol_TitleBgCollapsed',       (0.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_MenuBarBg',              (0.14, 0.14, 0.14, 1.00))
    set_color('ImGuiCol_ScrollbarBg',            (0.05, 0.05, 0.05, 0.54))
    set_color('ImGuiCol_ScrollbarGrab',          (0.34, 0.34, 0.34, 0.54))
    set_color('ImGuiCol_ScrollbarGrabHovered',   (0.40, 0.40, 0.40, 0.54))
    set_color('ImGuiCol_ScrollbarGrabActive',    (0.56, 0.56, 0.56, 0.54))
    set_color('ImGuiCol_CheckMark',              (0.33, 0.67, 0.86, 1.00))
    set_color('ImGuiCol_SliderGrab',             (0.34, 0.34, 0.34, 0.54))
    set_color('ImGuiCol_SliderGrabActive',       (0.56, 0.56, 0.56, 0.54))
    set_color('ImGuiCol_Button',                 (0.05, 0.05, 0.05, 0.54))
    set_color('ImGuiCol_ButtonHovered',          (0.19, 0.19, 0.19, 0.54))
    set_color('ImGuiCol_ButtonActive',           (0.20, 0.22, 0.23, 1.00))
    set_color('ImGuiCol_Header',                 (0.00, 0.00, 0.00, 0.52))
    set_color('ImGuiCol_HeaderHovered',          (0.00, 0.00, 0.00, 0.36))
    set_color('ImGuiCol_HeaderActive',           (0.20, 0.22, 0.23, 0.33))
    set_color('ImGuiCol_Separator',              (0.28, 0.28, 0.28, 0.29))
    set_color('ImGuiCol_SeparatorHovered',       (0.44, 0.44, 0.44, 0.29))
    set_color('ImGuiCol_SeparatorActive',        (0.40, 0.44, 0.47, 1.00))
    set_color('ImGuiCol_ResizeGrip',             (0.28, 0.28, 0.28, 0.29))
    set_color('ImGuiCol_ResizeGripHovered',      (0.44, 0.44, 0.44, 0.29))
    set_color('ImGuiCol_ResizeGripActive',       (0.40, 0.44, 0.47, 1.00))
    set_color('ImGuiCol_Tab',                    (0.00, 0.00, 0.00, 0.52))
    set_color('ImGuiCol_TabHovered',             (0.14, 0.14, 0.14, 1.00))
    set_color('ImGuiCol_TabActive',              (0.20, 0.20, 0.20, 0.36))
    set_color('ImGuiCol_TabUnfocused',           (0.00, 0.00, 0.00, 0.52))
    set_color('ImGuiCol_TabUnfocusedActive',     (0.14, 0.14, 0.14, 1.00))
    set_color('ImGuiCol_DockingPreview',         (0.33, 0.67, 0.86, 1.00))
    set_color('ImGuiCol_DockingEmptyBg',         (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_PlotLines',              (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_PlotLinesHovered',       (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_PlotHistogram',          (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_PlotHistogramHovered',   (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_TableHeaderBg',          (0.00, 0.00, 0.00, 0.52))
    set_color('ImGuiCol_TableBorderStrong',      (0.00, 0.00, 0.00, 0.52))
    set_color('ImGuiCol_TableBorderLight',       (0.28, 0.28, 0.28, 0.29))
    set_color('ImGuiCol_TableRowBg',             (0.00, 0.00, 0.00, 0.00))
    set_color('ImGuiCol_TableRowBgAlt',          (1.00, 1.00, 1.00, 0.06))
    set_color('ImGuiCol_TextSelectedBg',         (0.20, 0.22, 0.23, 1.00))
    set_color('ImGuiCol_DragDropTarget',         (0.33, 0.67, 0.86, 1.00))
    #set_color('ImGuiCol_NavHighlight',           (1.00, 0.00, 0.00, 1.00))
    set_color('ImGuiCol_NavWindowingHighlight',  (1.00, 0.00, 0.00, 0.70))
    set_color('ImGuiCol_NavWindowingDimBg',      (1.00, 0.00, 0.00, 0.20))
    set_color('ImGuiCol_ModalWindowDimBg',       (1.00, 0.00, 0.00, 0.35))

# Deep-dark with ps-style borders
def theme_contrast():
    theme_deep_dark()
    s = imgui.get_style()
    s.colors[ImGuiCol_Border] = imgui.Vec4(0.26, 0.26, 0.26, 1.0)
    s.colors[ImGuiCol_BorderShadow] = imgui.Vec4(0.0, 0.0, 0.0, 0.0)
    setattr(s, ChildBorderSize, 1.0)
    setattr(s, PopupBorderSize, 1.0)
    setattr(s, FrameBorderSize, 1.0)

# https://github.com/ocornut/imgui/issues/707#issuecomment-678611331
def theme_dark_overshifted():
    s = imgui.get_style()
    
    s.colors[ImGuiCol_Text]                  = imgui.Vec4(1.00, 1.00, 1.00, 1.00)
    s.colors[ImGuiCol_TextDisabled]          = imgui.Vec4(0.50, 0.50, 0.50, 1.00)
    s.colors[ImGuiCol_WindowBg]              = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_ChildBg]               = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_PopupBg]               = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    s.colors[ImGuiCol_Border]                = imgui.Vec4(0.43, 0.43, 0.50, 0.50)
    s.colors[ImGuiCol_BorderShadow]          = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    s.colors[ImGuiCol_FrameBg]               = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_FrameBgHovered]        = imgui.Vec4(0.38, 0.38, 0.38, 1.00)
    s.colors[ImGuiCol_FrameBgActive]         = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_TitleBg]               = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    s.colors[ImGuiCol_TitleBgActive]         = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    s.colors[ImGuiCol_TitleBgCollapsed]      = imgui.Vec4(0.00, 0.00, 0.00, 0.51)
    s.colors[ImGuiCol_MenuBarBg]             = imgui.Vec4(0.14, 0.14, 0.14, 1.00)
    s.colors[ImGuiCol_ScrollbarBg]           = imgui.Vec4(0.02, 0.02, 0.02, 0.53)
    s.colors[ImGuiCol_ScrollbarGrab]         = imgui.Vec4(0.31, 0.31, 0.31, 1.00)
    s.colors[ImGuiCol_ScrollbarGrabHovered]  = imgui.Vec4(0.41, 0.41, 0.41, 1.00)
    s.colors[ImGuiCol_ScrollbarGrabActive]   = imgui.Vec4(0.51, 0.51, 0.51, 1.00)
    s.colors[ImGuiCol_CheckMark]             = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_SliderGrab]            = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_SliderGrabActive]      = imgui.Vec4(0.08, 0.50, 0.72, 1.00)
    s.colors[ImGuiCol_Button]                = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_ButtonHovered]         = imgui.Vec4(0.38, 0.38, 0.38, 1.00)
    s.colors[ImGuiCol_ButtonActive]          = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_Header]                = imgui.Vec4(0.22, 0.22, 0.22, 1.00)
    s.colors[ImGuiCol_HeaderHovered]         = imgui.Vec4(0.25, 0.25, 0.25, 1.00)
    s.colors[ImGuiCol_HeaderActive]          = imgui.Vec4(0.67, 0.67, 0.67, 0.39)
    s.colors[ImGuiCol_Separator]             = s.colors[ImGuiCol_Border]
    s.colors[ImGuiCol_SeparatorHovered]      = imgui.Vec4(0.41, 0.42, 0.44, 1.00)
    s.colors[ImGuiCol_SeparatorActive]       = imgui.Vec4(0.26, 0.59, 0.98, 0.95)
    s.colors[ImGuiCol_ResizeGrip]            = imgui.Vec4(0.00, 0.00, 0.00, 0.00)
    s.colors[ImGuiCol_ResizeGripHovered]     = imgui.Vec4(0.29, 0.30, 0.31, 0.67)
    s.colors[ImGuiCol_ResizeGripActive]      = imgui.Vec4(0.26, 0.59, 0.98, 0.95)
    #s.colors[ImGuiCol_Tab]                   = imgui.Vec4(0.08, 0.08, 0.09, 0.83)
    #s.colors[ImGuiCol_TabHovered]            = imgui.Vec4(0.33, 0.34, 0.36, 0.83)
    #s.colors[ImGuiCol_TabActive]             = imgui.Vec4(0.23, 0.23, 0.24, 1.00)
    #s.colors[ImGuiCol_TabUnfocused]          = imgui.Vec4(0.08, 0.08, 0.09, 1.00)
    #s.colors[ImGuiCol_TabUnfocusedActive]    = imgui.Vec4(0.13, 0.14, 0.15, 1.00)
    #s.colors[ImGuiCol_DockingPreview]        = imgui.Vec4(0.26, 0.59, 0.98, 0.70)
    #s.colors[ImGuiCol_DockingEmptyBg]        = imgui.Vec4(0.20, 0.20, 0.20, 1.00)
    s.colors[ImGuiCol_PlotLines]             = imgui.Vec4(0.61, 0.61, 0.61, 1.00)
    s.colors[ImGuiCol_PlotLinesHovered]      = imgui.Vec4(1.00, 0.43, 0.35, 1.00)
    s.colors[ImGuiCol_PlotHistogram]         = imgui.Vec4(0.90, 0.70, 0.00, 1.00)
    s.colors[ImGuiCol_PlotHistogramHovered]  = imgui.Vec4(1.00, 0.60, 0.00, 1.00)
    s.colors[ImGuiCol_TextSelectedBg]        = imgui.Vec4(0.26, 0.59, 0.98, 0.35)
    s.colors[ImGuiCol_DragDropTarget]        = imgui.Vec4(0.11, 0.64, 0.92, 1.00)
    s.colors[ImGuiCol_NavHighlight]          = imgui.Vec4(0.26, 0.59, 0.98, 1.00)
    s.colors[ImGuiCol_NavWindowingHighlight] = imgui.Vec4(1.00, 1.00, 1.00, 0.70)
    s.colors[ImGuiCol_NavWindowingDimBg]     = imgui.Vec4(0.80, 0.80, 0.80, 0.20)
    s.colors[ImGuiCol_ModalWindowDimBg]      = imgui.Vec4(0.80, 0.80, 0.80, 0.35)

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

    # Color preview: install vscode extension 'json-color-token', set value "jsonColorToken.languages": ["json", "jsonc", "python"]
    # Converted with matplotlib.colors.to_hex(color, keep_alpha=True)
    s.colors[ImGuiCol_Text]                  = color('#ffffffff')
    s.colors[ImGuiCol_TextDisabled]          = color('#808080ff')
    s.colors[ImGuiCol_WindowBg]              = color('#0f0f0ff0')
    s.colors[ImGuiCol_ChildBg]               = color('#ffffff00')
    s.colors[ImGuiCol_PopupBg]               = color('#141414f0')
    s.colors[ImGuiCol_Border]                = color('#6e6e8080')
    s.colors[ImGuiCol_BorderShadow]          = color('#00000000')
    s.colors[ImGuiCol_FrameBg]               = color('#294a7a8a')
    s.colors[ImGuiCol_FrameBgHovered]        = color('#4296fa66')
    s.colors[ImGuiCol_FrameBgActive]         = color('#4296faab')
    s.colors[ImGuiCol_TitleBg]               = color('#0a0a0aff')
    s.colors[ImGuiCol_TitleBgActive]         = color('#294a7aff')
    s.colors[ImGuiCol_TitleBgCollapsed]      = color('#00000082')
    s.colors[ImGuiCol_MenuBarBg]             = color('#242424ff')
    s.colors[ImGuiCol_ScrollbarBg]           = color('#05050587')
    s.colors[ImGuiCol_ScrollbarGrab]         = color('#4f4f4fff')
    s.colors[ImGuiCol_ScrollbarGrabHovered]  = color('#696969ff')
    s.colors[ImGuiCol_ScrollbarGrabActive]   = color('#828282ff')
    s.colors[ImGuiCol_CheckMark]             = color('#4296faff')
    s.colors[ImGuiCol_SliderGrab]            = color('#3d85e0ff')
    s.colors[ImGuiCol_SliderGrabActive]      = color('#4296faff')
    s.colors[ImGuiCol_Button]                = color('#4296fa66')
    s.colors[ImGuiCol_ButtonHovered]         = color('#4296faff')
    s.colors[ImGuiCol_ButtonActive]          = color('#0f87faff')
    s.colors[ImGuiCol_Header]                = color('#4296fa4f')
    s.colors[ImGuiCol_HeaderHovered]         = color('#4296facc')
    s.colors[ImGuiCol_HeaderActive]          = color('#4296faff')
    s.colors[ImGuiCol_Separator]             = s.colors[imgui.COLOR_BORDER]
    s.colors[ImGuiCol_SeparatorHovered]      = color('#1a66bfc7')
    s.colors[ImGuiCol_SeparatorActive]       = color('#1a66bfff')
    s.colors[ImGuiCol_ResizeGrip]            = color('#4296fa40')
    s.colors[ImGuiCol_ResizeGripHovered]     = color('#4296faab')
    s.colors[ImGuiCol_ResizeGripActive]      = color('#4296faf2')
    s.colors[ImGuiCol_PlotLines]             = color('#9c9c9cff')
    s.colors[ImGuiCol_PlotLinesHovered]      = color('#ff6e59ff')
    s.colors[ImGuiCol_PlotHistogram]         = color('#e6b200ff')
    s.colors[ImGuiCol_PlotHistogramHovered]  = color('#ff9900ff')
    s.colors[ImGuiCol_TextSelectedBg]        = color('#4296fa59')
    s.colors[ImGuiCol_DragDropTarget]        = color('#BABA267D')
    s.colors[ImGuiCol_NavHighlight]          = color('#4296faff')
    s.colors[ImGuiCol_NavWindowingHighlight] = color('#ffffffb2')
    s.colors[ImGuiCol_NavWindowingDimBg]     = color('#cccccc33')
    s.colors[ImGuiCol_ModalWindowDimBg]      = color('#cccccc59')