from pyviewer.docking_viewer import DockingViewer, dockable
from pyviewer.utils import reshape_grid_np
from imgui_bundle import imgui
import numpy as np
from matplotlib import cm
import OpenGL.GL as gl
from functools import lru_cache

# Example that demonstrates an image gallery layout
# Based on the loss functions described in: https://fangel.github.io/packing-images-in-a-grid

class Test(DockingViewer):
    def setup_state(self):
        self.state.seed = 0
        self.state.num_img = 12
        self.state.scale = True
        self.grid_ar = 1.0
        self.imgs: list[np.ndarray] = []

    def compute(self):
        np.random.seed(self.state.seed)

        # Use a matplotlib colormap to pick distinct solid colors for each image
        cmap = cm.get_cmap('tab20')
        indices = np.arange(cmap.N)
        np.random.shuffle(indices)

        imgs = []
        for i in range(self.state.num_img):
            AR = 0.5 + np.random.rand() # aspect ratio in [0.5, 1.5]
            H = 180
            W = int(AR * H)
            idx = indices[i % len(indices)]
            color = cmap(idx)[:3]  # RGB in [0,1]
            col = np.array(color, dtype=np.float32).reshape(3, 1, 1)
            img = col * np.ones((1, 3, H, W), dtype=np.float32)
            imgs.append(img)

        self.imgs = imgs
        return None
    
    @dockable
    def image_grid(self):
        N = len(self.imgs)
        if N == 0:
            return
        
        cW = imgui.get_content_region_avail()[0] # canvas width
        pad = imgui.get_style().item_spacing[0]

        # For viz
        cW -= 100
        
        @lru_cache
        def P(i, s):
            width_sum = sum(img.shape[-1] for img in self.imgs[i:i+s])
            return np.abs(cW - width_sum - (s - 1) * pad)
        
        @lru_cache
        def C(i, s):
            if i + s < N:
                return min(
                    C(i, s+1), # add one more image
                    P(i, s) + C(i+s, 1) # finish row, add new one with single image
                )
            return P(i, s) # no more images => cost of row with given range

        rows = []
        i = 0
        while i < N:
            costs = np.array([C(i, s) for s in range(1, N-i+1)] + [np.inf])
            best_cnt = np.where(costs[1:] > costs[:-1])[0][0] + 1
            rows.append(self.imgs[i:i+best_cnt])
            i += best_cnt

        for row in rows:
            row_width = sum(img.shape[-1] for img in row) + (len(row) - 1) * pad
            scale = (cW / row_width) if self.state.scale else 1.0
            
            for j, img in enumerate(row):
                N, Ch, H, W = img.shape
                assert Ch == 3, 'RGBA not supported'
                assert img.dtype == np.float32
                
                tex_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

                data_hwc = np.transpose(img[0], (1, 2, 0))
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, # GLenum target
                    0,                # GLint level
                    gl.GL_RGBA,       # GLint internalformat
                    W,                # GLsizei width
                    H,                # GLsizei height
                    0,                # GLint border
                    gl.GL_RGB,        # GLenum format (incoming)
                    gl.GL_FLOAT,      # GLenum type (incoming)
                    data_hwc,         # const void * data
                )
                
                if j != 0: # not first (new row)
                    imgui.same_line()
                
                tex_ref = imgui.ImTextureRef(tex_id)
                imgui.image(tex_ref, (scale * W, scale * H), uv0=(0, 0), uv1=(1, 1))

    @dockable
    def toolbar(self):
        self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
        self.state.num_img = imgui.slider_int('Images', self.state.num_img, 1, 50)[1]
        self.state.scale = imgui.checkbox('Scale', self.state.scale)[1]

if __name__ == '__main__':
    _ = Test('Gallery')
    print('Done')