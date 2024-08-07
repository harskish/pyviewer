# PyViewer

![Toolbar Viewer](docs/screenshot.jpg)

Pyviewer is a python library for easily visualizing NumPy arrays and PyTorch tensors.

## Components

### single_image_viewer.py

A viewer for showing single fullscreen images or line plots without other UI elements. Runs in a ***separate process*** and remains interactive even if the main process is suspended (e.g. in a debugger). Great for interactively looking at intermediate values of complex ML/CG/CV pipelines.

Usage:
```
from pyviewer import single_image_viewer as siv
siv.draw(img_chw=np.random.randn(3,64,64))
siv.plot(np.sin(2*np.pi*np.linspace(0, 1, 10_000)))
```

### toolbar_viewer.py
A viewer that shows ImGui UI elemets on the left, and a large image on the right. Runs in the main process, but supports visualizing torch tensors directly from GPU memory (unlike single_image_viewer).

## Other features
* Bundles a [custom build](https://github.com/harskish/pyplotgui) of PyImGui with plotting support (via ImPlot)
* Dynamically rescalable user interface
* Window resizing to integer multiple of content resolution

## Installation
`pip install pyviewer`

## Usage
See `examples/demo.py` for a usage example.
