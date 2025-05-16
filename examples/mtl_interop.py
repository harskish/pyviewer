from pathlib import Path
from pyviewer import custom_ops
import numpy as np
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

            img = Image.open(Path(__file__).parent / '../docs/testimg1.avif').convert('RGBA') # 6000x4000
            img = np.array(img) # HWC, uint8
            img_fp = img.astype(np.float16) / 255.0 # HWC
            
            self.use_fp = True
            self.img_orig_uint = torch.tensor(img, device='mps')
            self.img_orig_fp = torch.tensor(img_fp, device='mps')
            img = self.img_orig_fp if self.use_fp else self.img_orig_uint
            
            # Seems like 4-8 bytes are added per color channel (plane):
            # https://stackoverflow.com/a/40815699
            H, W, C = img.shape
            print(f'Expected stats: width={W}, height={H}, bytes={H*W*C*1}, bytesPerRow={W*C*1}(+{C*8})')
            
            handle = plugin.gl_tex_rect(img)
            gl.glBindTexture(gl.GL_TEXTURE_RECTANGLE, handle)
            gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_RECTANGLE, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            
            # Store the new texture handle
            self.tex_handle_rect = handle
            self.tex_handle.upload_np(img.cpu().numpy())
            self.tex_handle_2d = self.tex_handle.tex

            self.out_scale = 1.0
            self.use_interop = True # cpu round-trip or interop?
            self.dt_upload_ms = 0.0
            self.update = True
            
            self.last_render = time.monotonic()
            self.fps_ema = 0
        
        @dockable(title='Test Output')
        def output(self):
            # Compute
            curr = None
            if self.use_fp:
                dtype = self.img_orig_fp.dtype
                curr = self.img_orig_fp * torch.tensor([self.out_scale, self.out_scale, self.out_scale, 1], dtype=dtype, device='mps')
            else:
                curr = self.img_orig_uint + int(self.out_scale * 10) * torch.tensor([[[1, 1, 1, 0]]], dtype=torch.uint8, device='mps')
                curr = curr.clip(0, 255)
            
            # Draw
            torch.mps.synchronize()
            t0 = time.monotonic_ns()
            dt_ns = 0.0
            if self.update:
                if self.use_interop:
                    self.tex_handle_rect = plugin.gl_tex_rect(curr)
                else:
                    self.tex_handle.upload_np(curr.cpu().numpy())
                    self.tex_handle_2d = self.tex_handle.tex
                dt_ns = time.monotonic_ns() - t0
            
            self.dt_upload_ms = 0.95 * self.dt_upload_ms + 0.05 * (1e-6 * dt_ns)
            tex = self.tex_handle_rect if self.use_interop else self.tex_handle_2d
            
            tH, tW, _ = curr.shape
            cW, cH = map(int, imgui.get_content_region_avail())
            tex_type = gl.GL_TEXTURE_RECTANGLE if self.use_interop else gl.GL_TEXTURE_2D
            canvas_tex = self.pan_handler.draw_to_canvas(tex, tW, tH, cW, cH, in_type=tex_type)
            imgui.image(canvas_tex, (cW, cH))

            target_fps = 80
            time.sleep(max(0, (1 / target_fps) - (time.monotonic() - self.last_render)))
            now = time.monotonic()
            dt_render = now - self.last_render
            self.last_render = now
            self.fps_ema = 0.95 * self.fps_ema + 0.05 * (1 / dt_render)
    
        #def compute(self):
        #    #self.img = self.out_scale * self.img_orig
        #    time.sleep(1/100)
        
        @dockable
        def toolbar(self):
            imgui.text(f'FPS: {self.fps_ema:.0f}')
            imgui.text(f'Upload time: {self.dt_upload_ms:.1f}ms')
            self.update = imgui.checkbox('Update', self.update)[1]
            self.use_interop = imgui.checkbox('Use interop', self.use_interop)[1]
            self.out_scale = imgui.slider_float('Brightness', self.out_scale, 0.0, 2.5)[1]
            self.use_fp = imgui.checkbox('Use floating-point', self.use_fp)[1]
    
    viewer = Viewer('mtl-interop test')
    print('Done')

    
