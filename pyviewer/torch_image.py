"""GPU-friendly image packing. Torch is imported only when a helper is used."""
from functools import lru_cache
import math


@lru_cache(maxsize=1)
def _packing_library():
    import torch

    source = '''
        #include <metal_stdlib>
        using namespace metal;
        kernel void pack_TYPE(device const TYPE* input, device TYPE4* output,
                              constant uint64_t& channels, constant float& gamma,
                              uint i [[thread_position_in_grid]]) {
            TYPE3 rgb(input[channels*i], input[channels*i+1], input[channels*i+2]);
            if (gamma > 0) {
                rgb = select(rgb, TYPE3(0), rgb < TYPE3(0));
                rgb = TYPE3(precise::pow(float3(rgb), float3(gamma)));
            }
            output[i] = TYPE4(rgb, channels == 4 ? input[channels*i+3] : TYPE(1));
        }
    '''
    return torch.mps.compile_shader('\n'.join(source.replace('TYPE', t) for t in ('half', 'float')))


def pack_rgba(image, *, gamma=None):
    """Pack HWC RGB/RGBA Torch tensors, preserving device, dtype and RGB range.

    RGB inputs receive opaque alpha (1 for floats, the dtype maximum for
    integers). Already-packed RGBA is returned unchanged. An optional positive
    ``gamma`` decodes nonnegative RGB to linear while retaining existing alpha.
    Contiguous FP16/FP32 MPS images fuse conversion and packing into one pass;
    other devices/layouts use Torch operations. Input pixels are never modified.
    """
    import torch
    import torch.nn.functional as F

    if (not torch.is_tensor(image) or image.ndim != 3 or image.shape[-1] not in (3, 4)
            or image.dtype == torch.bool or image.is_complex()):
        raise ValueError('Expected an HWC RGB/RGBA tensor with a floating or integer dtype')
    if gamma is not None:
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0 or not image.dtype.is_floating_point:
            raise ValueError('Gamma decoding requires a positive finite gamma and floating input')
    channels = image.shape[-1]
    if channels == 4 and gamma is None:
        return image
    if (image.device.type == 'mps' and image.is_contiguous() and image.numel()
            and image.dtype in (torch.float16, torch.float32)
            and hasattr(torch.mps, 'compile_shader')):
        output = torch.empty((*image.shape[:2], 4), dtype=image.dtype, device=image.device)
        library = _packing_library()
        kernel = library.pack_half if image.dtype == torch.float16 else library.pack_float
        kernel(image, output, channels, 0.0 if gamma is None else gamma,
               threads=image.shape[0] * image.shape[1])
        return output
    rgb = image[..., :3]
    if gamma is not None:
        rgb = rgb.clamp_min(0).pow(gamma)
    if channels == 4:
        return torch.cat((rgb, image[..., 3:]), dim=-1)
    alpha = 1 if image.dtype.is_floating_point else torch.iinfo(image.dtype).max
    if image.dtype.is_floating_point:
        return F.pad(rgb, (0, 1), value=alpha)
    # Retain support for integer MPS dtypes without relying on constant_pad_nd.
    return torch.cat((rgb, torch.full((*image.shape[:2], 1), alpha,
                                     dtype=image.dtype, device=image.device)), dim=-1)
