"""Continuously display a batch of directional RGB ramps."""

from pathlib import Path
import sys
import time

# Running this file directly should use the neighboring pyviewer source.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pyviewer import tensor_viewer as tv


if __name__ == '__main__':
    B, height, width = 96, 2000, 2400  # ~28GiB
    x = torch.linspace(-1, 1, width, device='cuda')[None, None, None, :]
    y = torch.linspace(-1, 1, height, device='cuda')[None, None, :, None]
    tv.init(normalize=False)
    first = True
    while True:
        if not tv.inst.paused.value or tv.inst.next.value:
            # Give each RGB channel in each batch image its own direction.
            angles = torch.rand(B, 3, device='cuda') * (2 * torch.pi)
            dx = angles.cos()[..., None, None]
            dy = angles.sin()[..., None, None]
            tensor = 0.5 + 0.5 * (dx * x + dy * y) / (dx.abs() + dy.abs()) + 0.06 * torch.randn(B, 1, height, width, device='cuda')
            if first:
                tv.draw(tensor, x_dim=3, y_dim=2, channel_dim=1, channels=(0, 1, 2))
                first = False
            else:
                tv.draw(tensor)
        time.sleep(0.1)
