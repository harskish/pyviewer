"""Rebuild bundled Vulkan and Metal shader assets.

Run: nix shell nixpkgs#glslang nixpkgs#spirv-cross -c python pyviewer/sdl_gpu_shaders/build.py
Shader compilation is only needed after editing shader sources, never at runtime.
"""
from pathlib import Path
import subprocess

root = Path(__file__).parent
for name in ('imgui.vert', 'imgui.frag', 'present.vert', 'present.frag'):
    source = root / name
    spirv = root / (name + '.spv')
    subprocess.run(['glslangValidator', '-V', str(source), '-o', str(spirv)], check=True)
    subprocess.run(['spirv-cross', str(spirv), '--msl', '--msl-version', '20100',
                    '--output', str(root / (name + '.msl'))], check=True)
