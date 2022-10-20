#!/usr/bin/env python

from distutils.core import setup

setup(name='pyviewer',
    version='1.0',
    description='Interactyive python viewers',
    author='Erik Härkönen',
    author_email='erik.harkonen@hotmail.com',
    url='github.com/harskish/',
    #py_modules=['toolbar_viewer'], # individual .py files
    packages=['pyviewer'], # name of importable thing
    package_requires=[
        'imgui[glfw]',
    ],
    #dependency_links=[
    #    'https://github.com/harskish/imviz/tree/prs#egg=package-1.0'
    #]
    include_package_data=True,
    package_data={
        '': ['*.ttf'] # embedded fonts
    },
)
