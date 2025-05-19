from pathlib import Path
from pyviewer import custom_ops
from pyviewer.params import enum_slider
import numpy as np
from functools import lru_cache
import torch
import time

# https://github.com/python-pillow/Pillow/issues/8959#issuecomment-2883368474
from PIL import Image, __version__ as PIL_VER
if [int(v) for v in PIL_VER.split('.')] < [11, 3, 0]:
    import pillow_avif

plugin = custom_ops.get_plugin('MetalGLInterop', 'metal_gl_interop.mm',
    Path(__file__).parent.parent/'pyviewer/custom_ops', cuda=False)

if __name__ == "__main__":
    from pyviewer.docking_viewer import DockingViewer, dockable
    from imgui_bundle import imgui
    import OpenGL.GL as gl

    class Viewer(DockingViewer):
        def setup_state(self):            
            assert plugin is not None

            self.scale = 1.0
            self.use_fp = False
            img = self.get_input(self.scale, self.use_fp)
            
            # Store the new texture handle
            self.tex_handle.upload_np(img.cpu().numpy())
            self.tex_handle_2d = self.tex_handle.tex
            self.tex_handle_rect, _ = plugin.gl_tex_rect(img)
            self.set_interp()

            self.brightness = 1.0
            self.use_interop = True # cpu round-trip or interop?
            self.dt_upload_ms = 0.0
            self.pan_cnv_render_ms = 0.0
            self.update = True
            
            self.last_render = time.monotonic()
            self.fps_ema = 0

        def set_interp(self):
            gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, self.tex_handle_rect)
            gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        
        @lru_cache
        def get_input(self, scale=1.0, fp=True):
            """Get scaled input image as cached GPU tensor"""
            img = Image.open(Path(__file__).parent / '../docs/testimg1.avif').convert('RGBA') # 6000x4000
            img = img.resize([int(d*scale) for d in img.size])
            arr = np.array(img, dtype=np.float16) / 255.0 if fp else np.array(img) # HWC
            print(arr.shape)
            return torch.tensor(arr, device='mps')
        
        @dockable(title='Test Output')
        def output(self):
            # Compute
            orig = self.get_input(self.scale, self.use_fp)
            curr = None
            if self.use_fp:
                curr = orig * torch.tensor([self.brightness, self.brightness, self.brightness, 1], dtype=orig.dtype, device='mps')
            else:
                curr = orig + int((self.brightness - 1) * 10) * torch.tensor([[[1, 1, 1, 0]]], dtype=torch.uint8, device='mps')
                curr = curr.clip(0, 255)
            
            # Draw
            gl.glFinish() # impacts upload perf! (hidden barrier in interop kernel?)
            torch.mps.synchronize()
            t0 = time.monotonic_ns()
            dt_ns = 0.0
            if self.update:
                if self.use_interop:
                    self.tex_handle_rect, ch = plugin.gl_tex_rect(curr)
                    if ch:
                        self.set_interp()
                else:
                    self.tex_handle.upload_np(curr.cpu().numpy())
                    self.tex_handle_2d = self.tex_handle.tex
                torch.mps.synchronize()
                dt_ns = time.monotonic_ns() - t0
            
            self.dt_upload_ms = 0.95 * self.dt_upload_ms + 0.05 * (1e-6 * dt_ns)
            tex = self.tex_handle_rect if self.use_interop else self.tex_handle_2d
            
            tH, tW, _ = curr.shape
            cW, cH = map(int, imgui.get_content_region_avail())
            tex_type = gl.GL_TEXTURE_RECTANGLE if self.use_interop else gl.GL_TEXTURE_2D
            gl.glFinish()
            t0 = time.monotonic_ns()
            canvas_tex = self.pan_handler.draw_to_canvas(tex, tW, tH, cW, cH, in_type=tex_type)
            gl.glFinish()
            dt_ns = time.monotonic_ns() - t0
            self.pan_cnv_render_ms = 0.95 * self.pan_cnv_render_ms + 0.05 * (1e-6 * dt_ns)
            imgui.image(canvas_tex, (cW, cH))

            #target_fps = 80
            #time.sleep(max(0, (1 / target_fps) - (time.monotonic() - self.last_render)))
            now = time.monotonic()
            dt_render = now - self.last_render
            self.last_render = now
            self.fps_ema = 0.95 * self.fps_ema + 0.05 * (1 / dt_render)
        
        @dockable
        def toolbar(self):
            imgui.text(f'FPS: {self.fps_ema:.0f}')
            imgui.text(f'Upload time: {self.dt_upload_ms:.1f}ms')
            imgui.text(f'Canvas render: {self.pan_cnv_render_ms:.1f}ms')
            self.scale = enum_slider('Resolution scale', [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5], self.scale)[1]
            self.update = imgui.checkbox('Update', self.update)[1]
            self.use_interop = imgui.checkbox('Use interop', self.use_interop)[1]
            self.brightness = imgui.slider_float('Brightness', self.brightness, 0.0, 2.5)[1]
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.brightness = 1.0
            self.use_fp = imgui.checkbox('Use floating-point', self.use_fp)[1]
    
    viewer = Viewer('mtl-interop test')
    print('Done')

    
