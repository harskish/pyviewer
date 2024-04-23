# Code graciously provided by Pauli Kemppinen (github.com/msqrt)

import os
import torch
import torch.utils.cpp_extension as cpp

import importlib
from functools import cache
from typing import Union

@cache
def get_plugin(
    plugin_name: str,
    source_files: Union[tuple, str],
    source_folder: str = '.',
    ldflags: tuple = None,
    cuda: bool = True, # can turn off if not needed
    verbose=True
):
    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        lib_dir = os.path.dirname(__file__) + r"."
        def find_cl_path():
            import glob
            for maybe_x86 in ["", " (x86)"]:
                for edition in ['Community', 'Enterprise', 'Professional', 'BuildTools']:
                    paths = sorted(glob.glob(f"C:/Program Files{maybe_x86}/Microsoft Visual Studio/*/{edition}/VC/Tools/MSVC/*/bin/Hostx64/x64"), reverse=True)
                    if paths:
                        return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path


    # Linker options.
    if os.name == 'posix':
        cflags = ['-O3', '-std=c++17']
        ldflags = ['-lGL', '-lGLEW', '-lEGL'] if ldflags is None else ldflags
    elif os.name == 'nt':
        libs = ['user32', 'opengl32']
        ldflags = (['/LIBPATH:' + lib_dir] + ['/DEFAULTLIB:' + x for x in libs]) if ldflags is None else ldflags
        cflags = ['/O2', '/DWIN32', '/std:c++17','/permissive-', '/w']


    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    if os.name == "nt":
        try:
            lock_fn = os.path.join(cpp._get_build_directory(plugin_name, False), 'lock')
            if os.path.exists(lock_fn):
                print("warning: Lock file exists in build directory: '%s', removing" % lock_fn)
                os.remove(lock_fn)
        except:
            pass
    
    # Compile and load.
    original = os.getcwd()
    os.chdir(source_folder)
    
    try:
        cpp.load(verbose=verbose, name=plugin_name, extra_cflags=cflags, extra_cuda_cflags=['-O2'], sources=source_files, extra_ldflags=ldflags, with_cuda=cuda)
    finally:
        os.chdir(original)
    return importlib.import_module(plugin_name)
