import numpy as np
import torch
import imgui
import contextlib
from io import BytesIO
from pathlib import Path
import os

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