import numpy as np
import torch
import imgui
import contextlib
from io import BytesIO
from pathlib import Path
import os
import glfw
import random
import string

import kornia
print('TODO: remove kornia dependency')

# ImGui widget that wraps arbitrary object
# and allows mouse pand & zoom controls
class PannableArea():
    def __init__(self, set_callbacks=False, glfw_window=False) -> None:  # draw_content: callable, 
        self.prev_cbk: callable = lambda : None  # for chaining
        self.output_rect_tl = np.zeros(2, dtype=np.float32)
        self.content_size_px = (1, 1)
        self.id = ''.join(random.choices(string.ascii_letters, k=20))
        self.pan = (0, 0)
        self.pan_start = (0, 0)
        self.pan_delta = (0, 0)
        self.zoom: float = 1.0
        #self.xform: np.ndarray = np.eye(3)

        if set_callbacks:
            assert glfw_window, 'Must provide glfw window for callback setting'
            self.set_callbacks(glfw_window)

    def zoom_and_pan(self, img_hwc):
        import kornia
        H, W, _ = img_hwc.shape
        total = self.get_transform(W, H).to(img_hwc.device)
        cW, cH = self.content_size_px
        pixel_size = max(cW / W, cH / H) * self.zoom
        mode = 'nearest' if pixel_size > 4 else 'bilinear'
        transformed = kornia.geometry.warp_affine(
            img_hwc.permute(2, 0, 1).unsqueeze(0), total[:, :2, :3], dsize=(H, W), mode=mode, align_corners=True)
        return transformed.squeeze().permute(1, 2, 0) # back to hwc

    def set_callbacks(self, glfw_window):
        self.prev_cbk = glfw.set_scroll_callback(glfw_window, self.mouse_wheel_callback)

    def get_transform(self, W, H, top_left=(0, 0)):
        tr = torch.tensor([
            (self.pan[0]+self.pan_delta[0])*W,
            (self.pan[1]+self.pan_delta[1])*H
        ], dtype=torch.float32).reshape(1, 2)
        center = torch.tensor(
            [top_left[0] - tr[0, 0] + W/2, top_left[1] - tr[0, 1] + H/2], dtype=torch.float32).reshape(1, 2) # for rotation and zoom
        angle = torch.tensor([0.0], dtype=torch.float32)
        scale = torch.tensor(2*[self.zoom], dtype=torch.float32).reshape(1, 2)
        
        _M = kornia.geometry.transform.get_affine_matrix2d(tr, center, scale, angle)
        #M = torch.tensor(np.diag(self.zoom, self.zoom, 1.0))
        #M[0:2, 2] += tr

        return _M

    # Content wrapped in with handler
    def __enter__(self):
        # Create container
        imgui.set_next_window_size(*imgui.get_window_size())
        imgui.set_next_window_position(*imgui.get_window_position())
        begin_inline(f'pannable_content##{self.id}')
        
        # Draw enclosed content
        return self

    def __exit__(self, *args, **kwargs):
        # Keep track of content bounds
        rmin = imgui.get_window_content_region_min()
        rmax = imgui.get_window_content_region_max()
        self.content_size_px = tuple([int(r-l) for l,r in zip(rmin, rmax)])
        self.output_rect_tl[:] = imgui.get_item_rect_min()

        # Handle pan action
        xy = torch.tensor(self.mouse_pos_content_norm) # normalized coords
        
        # Figure out what part of image is currently visible
        box = torch.tensor([
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]).view(1, 2, 3)
        M = torch.linalg.inv(self.get_transform(1, 1))
        box = (box @ M)[0, 0:2, 0:2]
        a, b = (box[0] * (1 - xy) + box[1] * xy).tolist()

        if imgui.is_mouse_clicked(0): # left mouse down
            self.pan_start = (a, b)
        if imgui.is_mouse_down(0):
            self.pan_delta = (a - self.pan_start[0], b - self.pan_start[1])
        if imgui.is_mouse_released(0): # left mouse up
            self.pan = tuple(s+d for s,d in zip(self.pan, self.pan_delta))
            self.pan_start = self.pan_delta = (0, 0)
        
        # Close container
        imgui.end()
    
    @property
    def content_size(self):
        return np.array(self.content_size_px)

    @property
    def mouse_pos_abs(self):
        return np.array(imgui.get_mouse_pos())

    @property
    def mouse_pos_content_norm(self):
        return (self.mouse_pos_abs - self.output_rect_tl) / self.content_size

    def mouse_hovers_content(self):
        x, y = self.mouse_pos_content_norm
        return (0 <= x <= 1) and (0 <= y <= 1)
    
    def mouse_wheel_callback(self, window, x, y) -> None:
        if self.mouse_hovers_content():
            self.zoom = max(1e-2, (0.85**np.sign(-y)) * self.zoom)
        else:
            self.prev_scroll_callback(window, x, y) # scroll imgui lists etc.

# Dataclass that enforces type annotation
# Enables compare-by-value
def strict_dataclass(cls, *args, **kwargs):
    annotations = cls.__dict__.get('__annotations__', {})
    for name in dir(cls):
        if name.startswith('__'):
            continue
        if name not in annotations:
            raise RuntimeError(f'Unannotated field: {name}')
    
    # Write updated
    setattr(cls, '__annotations__', annotations)
        
    from dataclasses import dataclass
    return dataclass(cls, *args, **kwargs)

# with-block for item id
@contextlib.contextmanager
def imgui_id(id: str):
    imgui.push_id(id)
    yield
    imgui.pop_id()

# with-block for item width
@contextlib.contextmanager
def imgui_item_width(size):
    imgui.push_item_width(size)
    yield
    imgui.pop_item_width()

# Full screen imgui window
def begin_inline(name):
    with imgui.styled(imgui.STYLE_WINDOW_ROUNDING, 0):
        imgui.begin(name,
            flags = \
                imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_SAVED_SETTINGS
        )

# Recursive getattr
def rgetattr(obj, key, default=None):
    head = obj
    while '.' in key:
        bot, key = key.split('.', maxsplit=1)
        head = getattr(head, bot, {})
    return getattr(head, key, default)

# Combo box that returns value, not index
def combo_box_vals(title, values, current, height_in_items=-1, to_str=str):
    values = list(values)
    curr_idx = 0 if current not in values else values.index(current)
    changed, ind = imgui.combo(title, curr_idx, [to_str(v) for v in values], height_in_items)
    return changed, values[ind]

# Imgui slider that can switch between int and float formatting at runtime
def slider_dynamic(title, v, min, max, width=0.0):
    scale_fmt = '%.2f' if np.modf(v)[0] > 0 else '%.0f' # dynamically change from ints to floats
    with imgui_item_width(width):
        return imgui.slider_float(title, v, min, max, format=scale_fmt)

# Int2 slider that prevents overlap
def slider_range(v1, v2, vmin, vmax, push=False, title='', width=0.0):
    with imgui_item_width(width):
        s, e = imgui.slider_int2(title, v1, v2, vmin, vmax)[1]

    if push:
        return (min(s, e), max(s, e))
    elif s != v1:
        return (min(s, e), e)
    elif e != v2:
        return (s, max(s, e))
    else:
        return (s, e)

# Shape batch as square if possible
def get_grid_dims(B):
    if B == 0:
        return (0, 0)
    
    S = int(B**0.5 + 0.5)
    while B % S != 0:
        S -= 1
    return (B // S, S) # (W, H)

def reshape_grid_np(img_batch):
    if isinstance(img_batch, list):
        img_batch = np.concatenate(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = np.reshape(img_batch, [rows, cols, C, H, W])
    img_batch = np.transpose(img_batch, [0, 3, 1, 4, 2])
    img_batch = np.reshape(img_batch, [rows * H, cols * W, C])

    return img_batch

def reshape_grid_torch(img_batch):
    if isinstance(img_batch, list):
        img_batch = torch.cat(img_batch, axis=0) # along batch dim
    
    B, C, H, W = img_batch.shape
    cols, rows = get_grid_dims(B)

    img_batch = img_batch.reshape(rows, cols, C, H, W)
    img_batch = img_batch.permute(0, 3, 1, 4, 2)
    img_batch = img_batch.reshape(rows * H, cols * W, C)

    return img_batch

def reshape_grid(batch):
    return reshape_grid_torch(batch) if torch.is_tensor(batch) else reshape_grid_np(batch)

def sample_seeds(N, base=None):
    if base is None:
        base = np.random.randint(np.iinfo(np.int32).max - N)
    return [(base + s) for s in range(N)]

def sample_latent(B, n_dims=512, seed=None):
    seeds = sample_seeds(B, base=seed)
    return seeds_to_latents(seeds, n_dims)

def seeds_to_latents(seeds, n_dims=512):
    latents = np.zeros((len(seeds), n_dims), dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        latents[i] = rng.standard_normal(n_dims)
    
    return latents

# File copy with progress bar
# For slow network drives etc.
def copy_with_progress(pth_from, pth_to):
    from tqdm import tqdm
    os.makedirs(pth_to.parent, exist_ok=True)
    size = int(os.path.getsize(pth_from))
    fin = open(pth_from, 'rb')
    fout = open(pth_to, 'ab')

    try:
        with tqdm(ncols=80, total=size, bar_format=pth_from.name + ' {l_bar}{bar} | Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.close()

# File open with progress bar
# For slow network drives etc.
# Supports context manager
def open_prog(pth, mode):
    from tqdm import tqdm
    size = int(os.path.getsize(pth))
    fin = open(pth, 'rb')

    assert mode == 'rb', 'Only rb supported'
    fout = BytesIO()

    try:
        with tqdm(ncols=80, total=size, bar_format=Path(pth).name + ' {l_bar}{bar}| Remaining: {remaining}') as pbar:
            while True:
                buf = fin.read(4*2**20) # 4 MiB
                if len(buf) == 0:
                    break
                fout.write(buf)
                pbar.update(len(buf))
    except Exception as e:
        print(f'File copy failed: {e}')
    finally:
        fin.close()
        fout.seek(0)

    return fout

# Convert input image to valid range for showing
# Output converted to target dtype *after* scaling
#   => should not affect quality that much
def normalize_image_data(img_hwc, target_dtype='uint8'):    
    is_np = isinstance(img_hwc, np.ndarray)
    is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
    
    # Valid ranges for RGB data
    maxval = 1 if is_fp else 255
    minval = 0
    
    # If outside of range: normalize to [0, 1]
    if img_hwc.max() > maxval or img_hwc.min() < minval:
        img_hwc = img_hwc.astype(np.float32) if is_np else img_hwc.float()
        img_hwc -= img_hwc.min() # min is negative
        img_hwc /= img_hwc.max()
        is_fp = True
        maxval = 1
    
    # At this point, data will be:
    #  i) fp32, in [0, 1]
    # ii) uint8, in [0, 255]

    # Convert to target dtype
    if target_dtype == 'uint8':
        img_hwc = img_hwc * 255 if is_fp else img_hwc
        img_hwc = np.uint8(img_hwc) if is_np else img_hwc.byte()
    else:
        img_hwc = img_hwc.astype(np.float32) if is_np else img_hwc.float()
        img_hwc = img_hwc / maxval

    # (H, W) to (H, W, 1)
    if img_hwc.ndim == 2:
        img_hwc = img_hwc[..., None]

    # Use at most 3 channels
    img_hwc = img_hwc[:, :, :3]

    return img_hwc