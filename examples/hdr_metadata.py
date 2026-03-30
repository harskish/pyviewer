from pyviewer import hdr_patch; hdr_patch.use_patched()
import glfw
import sys

def create_window_with_fallbacks(title: str, hdr=True):
    # No floatbuffer on X11
    # https://github.com/Tom94/nanogui-1/blob/f4f8b8b42b302cd0bfb91c592ea35f560e371ec4/src/screen.cpp#L157
    is_linux = sys.platform.startswith('linux')
    is_wayland = glfw.get_platform() == glfw.PLATFORM_WAYLAND

    float_buffer = hdr
    if is_linux and not is_wayland:
        float_buffer = False

    stencil_buffer = True
    depth_buffer = True

    depth_bits = stencil_bits = 0
    color_bits = 8
    
    if depth_buffer:
        depth_bits = 32
    if stencil_buffer:
        depth_bits = 24
        stencil_bits = 8
    if float_buffer:
        color_bits = 16

    glfw.window_hint(glfw.RED_BITS, color_bits)
    glfw.window_hint(glfw.GREEN_BITS, color_bits)
    glfw.window_hint(glfw.BLUE_BITS, color_bits)
    glfw.window_hint(glfw.ALPHA_BITS, color_bits)
    glfw.window_hint(glfw.STENCIL_BITS, stencil_bits)
    glfw.window_hint(glfw.DEPTH_BITS, depth_bits)
    glfw.window_hint(hdr_patch.GLFW_FLOATBUFFER, float_buffer)

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    glfw.window_hint(glfw.MAXIMIZED, glfw.FALSE)
    glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

    glfw.window_hint_string(glfw.X11_CLASS_NAME, title)
    glfw.window_hint_string(glfw.X11_INSTANCE_NAME, title)
    glfw.window_hint_string(glfw.WAYLAND_APP_ID, title)

    window = None

    for _ in range(2):
        window = glfw.create_window(640, 640, title, None, None)
        if window:
            break
        
        if float_buffer:
            float_buffer = False
            glfw.window_hint(hdr_patch.GLFW_FLOATBUFFER, glfw.FALSE)
            glfw.window_hint(glfw.RED_BITS, 10)
            glfw.window_hint(glfw.GREEN_BITS, 10)
            glfw.window_hint(glfw.BLUE_BITS, 10)
            glfw.window_hint(glfw.ALPHA_BITS, 2)

    if not window:
        raise RuntimeError('Could not create an OpenGL context')

    return window

if __name__ == "__main__":
    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_WAYLAND) # => sdr white >80, max lum 1400, pq, bt2020
    #glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11) # => sdr white 80, max lum 0, gamma22, srgb, 8bit
    glfw.init_hint(hdr_patch.GLFW_WAYLAND_COLOR_MANAGEMENT, glfw.TRUE)
    if not glfw.init():
        raise RuntimeError('GLFW init failed')

    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2) # https://github.com/Tom94/nanogui-1/blob/f4f8b8b42b302cd0bfb91c592ea35f560e371ec4/include/nanogui/screen.h#L93
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    w = create_window_with_fallbacks('HDR meta')
    if not w:
        raise RuntimeError('Window creation failed')
    
    glfw.make_context_current(w)
    glfw.show_window(w)

    # TEV:
    # - nanogui init: https://github.com/Tom94/tev/blob/v2.10.0/src/main.cpp#L754
    #   -> calls glfwInitHint(GLFW_WAYLAND_COLOR_MANAGEMENT, GLFW_TRUE): https://github.com/Tom94/nanogui-1/blob/f4f8b8b42b302cd0bfb91c592ea35f560e371ec4/src/common.cpp#L50
    #   -> calls glfwInit()
    # - New ImageViewer: https://github.com/Tom94/tev/blob/v2.10.0/src/main.cpp#L834
    #   - https://github.com/Tom94/tev/blob/v2.10.0/src/ImageViewer.cpp#L94
    #   - Screen{size, caption="tev", resizable=true, maximize, fullscreen=false, depth_buffer=true, stencil_buffer=true, floatBuffer}
    #     - nanogui::Screen::init(): https://github.com/Tom94/nanogui-1/blob/f4f8b8b42b302cd0bfb91c592ea35f560e371ec4/src/screen.cpp#L148
    #      - RED_BITS etc.
    # - nanogui::run():  https://github.com/Tom94/tev/blob/v2.10.0/src/main.cpp#L915
    #   -> main loop: https://github.com/Tom94/nanogui-1/blob/f4f8b8b42b302cd0bfb91c592ea35f560e371ec4/src/common.cpp#L96
    #     -> screen->draw_all();
    
    # - inits with 10-bit gamma22
    # - switches to pq after a while (https://github.com/Tom94/tev/blob/42090b18f708397c9164c2ebd5e2be85c7b3d2cc/src/ImageViewer.cpp#L1410)

    frame = 0
    while not glfw.window_should_close(w):
        glfw.poll_events()
        glfw.swap_buffers(w)
        if frame % 100 == 0:
            vmax, vref, vcur = hdr_patch.get_edr_range(w, gamma=2.2)
            print(f'EDR headroom: cur={vcur:.2f}, ref={vref:.2f}, max{vmax:.2f}')
            print(f'SDR white: {glfw.glfwGetWindowSdrWhiteLevel(w)}')
            print(f'Max luminance: {glfw.glfwGetWindowMaxLuminance(w)}')
            print(f'Transfer function: {glfw.glfwGetWindowTransfer(w)}')
            print(f'Primaries: {glfw.glfwGetWindowPrimaries(w)}')
            print(f'RED bits: {glfw.get_window_attrib(w, glfw.RED_BITS)}')
            print()

        frame += 1

    glfw.destroy_window(w)
    glfw.terminate()

    
