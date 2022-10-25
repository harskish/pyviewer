#!/usr/bin/env python

from distutils.core import setup

import sys
if 'develop' in sys.argv:
    print("WARNING: 'python setup.py develop' won't install \
        dependencies correctly, please use 'pip install -e .' instead.")

setup(name='pyviewer',
    version='1.1',
    description='Interactyive python viewers',
    author='Erik Härkönen',
    author_email='erik.harkonen@hotmail.com',
    url='github.com/harskish/',
    packages=['pyviewer'], # name of importable thing
    setup_requires=['wheel'],
    install_requires=[
        'glfw',
        'numpy',
        'imgui@git+https://github.com/harskish/pyimgui.git@dev/version-2.0#egg=pyimgui',
        'pyopengl',
    ],
    include_package_data=True,
    package_data={
        '': ['*.ttf'] # embedded fonts
    },
)
