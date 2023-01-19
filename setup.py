#!/usr/bin/env python

from distutils.core import setup

import sys
if 'develop' in sys.argv:
    print("WARNING: 'python setup.py develop' won't install \
        dependencies correctly, please use 'pip install -e .' instead.")

setup(name='pyviewer',
    version='1.2.1',
    description='Interactyive python viewers',
    author='Erik Härkönen',
    author_email='erik.harkonen@hotmail.com',
    url='github.com/harskish/',
    packages=['pyviewer'], # name of importable thing
    setup_requires=['wheel'],
    install_requires=[
        'glfw>2.0.0',
        'numpy',
        'pyopengl>3.0.0',
        'pyplotgui',  # custom imgui+implot package
        'light-process==0.0.7',
    ],
    include_package_data=True,
    package_data={
        '': ['*.ttf'] # embedded fonts
    },
)
