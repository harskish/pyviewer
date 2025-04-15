#!/usr/bin/env python

from distutils.core import setup
from pathlib import Path

import sys
if 'develop' in sys.argv:
    print("WARNING: 'python setup.py develop' won't install \
        dependencies correctly, please use 'pip install -e .' instead.")

# Version bump workflow:
# Increment version number (together with other changes)
# git commit
# git tag -a "vX.X.X" -m "vX.X.X"  (only annotated tags show up in GitHub)
# git push --follow-tags
    
# In case of mistake:
# git tag --delete tagname
# git push --delete origin tagname

# Test package_data changes by manually starting
# the publish workflow and checking the "Print whl contents" step,
# or with: `rm -r build\ && pip wheel . && unzip -l pyviewer-*.whl`
# (not same environment, but good first step)

# Normally we would parse the _version.py sources to get the version string.
# Pyviewer's __init__.py imports nothing but the version string, however, so this is safe.
from pyviewer._version import __version__ as version_str

setup(name='pyviewer',
    version=version_str,
    description='Interactyive python viewers',
    author='Erik Härkönen',
    author_email='erik.harkonen@hotmail.com',
    url='https://github.com/harskish/pyviewer',
    packages=['pyviewer'], # name of importable thing
    setup_requires=['wheel'],
    install_requires=[
        'glfw==2.8.0',
        'numpy',
        'pyopengl==3.1.7',
        #'pyplotgui',  # custom imgui+implot package
        'imgui-bundle==1.6.2', # imgui + implot + many others
        'setuptools<=72.1.0', # github.com/pytorch/pytorch/issues/136541
        'light-process==0.0.7',
        'py==1.11.0', # for capturing c++ extension stdout/stderr, part of pytest
        'ninja', # for custom OP
    ],
    include_package_data=True,
    package_data={
        'pyviewer': [
            '*.ttf',            # embedded fonts
            'custom_ops/*.cpp', # custom CUDA op
        ],
    },
    long_description=Path('README.md').read_text()
        .replace('docs/screenshot.jpg', 'https://github.com/harskish/pyviewer/raw/master/docs/screenshot.jpg'),
    long_description_content_type="text/markdown",
    license='CC BY-NC-SA 4.0',
)
