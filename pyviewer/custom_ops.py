# Code graciously provided by Pauli Kemppinen (github.com/msqrt)

import os
import sys
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
    unsafe_load_prebuilt: bool = False,
    verbose=True
):
    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        lib_dir = os.path.dirname(__file__) + r"."  # TODO: this seems incorrect
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

        cuda_path = os.getenv('CUDA_PATH')
        if cuda:
            if cuda_path is None:
                print('Please set environment variable CUDA_PATH')
            else:
                # As of Python 3.8, cwd and $PATH are no longer searched for DLLs
                os.add_dll_directory(os.path.join(cuda_path, 'bin'))

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

    # Custom build dir, avoids expensive torch.utils.cpp_extension import (for _get_build_directory())
    # Enables quicker loads if compiled module exists.
    import torch
    cu_str = ('cpu' if torch.version.cuda is None else f'cu{torch.version.cuda.replace(".", "")}')
    python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
    ext_dir = os.path.expanduser(f'~/.cache/torch_extensions/{python_version}_{cu_str}')
    build_dir = os.path.join(ext_dir, plugin_name)
    os.environ['TORCH_EXTENSIONS_DIR'] = ext_dir

    # Try to load prebuilt, continue to build on failure
    # Might load an old version if the sources have changed
    if unsafe_load_prebuilt:
        prev_path = [*sys.path]
        try:
            sys.path.append(build_dir)
            return importlib.import_module(plugin_name) # .pyd on Windows
        except ImportError as e:
            if e.msg.startswith('DLL load failed') and cuda:
                print('DLL load failed, make sure all DLLs required by module are available (NB: $PATH and cwd are not searched)')
        except Exception as e:
            pass # could not load cached, proceed to compilation step
        finally:
            sys.path = prev_path

    # Make sure build dir is honored
    import torch.utils.cpp_extension as cpp
    _build_dir = cpp._get_build_directory(plugin_name, False)
    assert _build_dir == build_dir, 'Torch extension build dir mismatch'

    if os.name == "nt":
        try:
            # Try to detect if a stray lock file is left in cache directory and show a warning.
            # This sometimes happens on Windows if the build is interrupted at just the right moment.
            lock_fn = os.path.join(build_dir, 'lock')
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
    except ImportError as e:
        if e.msg.startswith('DLL load failed') and cuda:
            # Debug with:
            # dumpbin.exe /dependents cuda_gl_interop.pyd
            # python -m dlldiag deps cuda_gl_interop.pyd
            # Dependencies.exe -chain cuda_gl_interop.pyd -depth 1
            print('DLL load failed, make sure all DLLs required by module are available (NB: $PATH and cwd are not searched)')
    except Exception as e:
        print('Custom op compilation failed:', e)
    finally:
        os.chdir(original)
    return importlib.import_module(plugin_name)
