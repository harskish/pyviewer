"""Explore an NCHW tensor; set a debugger breakpoint after draw()."""

import torch
from pyviewer import tensor_viewer as tv


if __name__ == '__main__':
    # Allocate 32 GiB of uniform noise; the viewer samples only visible values.
    tensor = torch.rand(2, 4, 16384, 65536, device='cuda')
    tv.init(normalize=False)
    tv.draw(tensor, x_dim=3, y_dim=2, channel_dim=1, channels=(0, 1, 2))
    tv.inst.wait_for_close()
