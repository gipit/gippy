#!/usr/bin/env python
################################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    Copyright (C) 2014 Matthew A Hanson
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>
################################################################################

"""
setup for GIP and gippy
"""

import os
import shutil
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from copy import deepcopy
import numpy


class GIPinstall(install):
    def run(self):
        os.system('cd GIP; make; cd ..')
        shutil.copy('GIP/bin/Release/libgip.so', '/usr/lib/')
        install.run(self)


class GIPdevelop(develop):
    def initialize_options(self):
        gippy_module.runtime_library_dirs = [os.path.abspath('GIP/bin/Release')]
        develop.initialize_options(self)

    def run(self):
        os.system('cd GIP; make; cd ..')
        develop.run(self)

gippy_module = Extension(
    name='_gippy',
    sources=['gippy.i'],
    swig_opts=['-c++', '-w509', '-IGIP'],  # '-keyword'],
    include_dirs=['GIP', numpy.get_include()],
    libraries=['gip', 'gdal', 'boost_system', 'boost_filesystem'],  # ,'X11'],
    library_dirs=['GIP/bin/Release'],
    extra_compile_args=['-fPIC'],  # , '-std=c++0x'],
)

setup(
    name='gippy',
    version='0.9.0',
    description='Geospatial Image Processing for Python',
    author='Matthew Hanson',
    author_email='mhanson@appliedgeosolutions.com',
    license='GPLv2',
    ext_modules=[gippy_module],
    py_modules=['gippy'],
    #install_requires = ['','numpy'],
    cmdclass={
        "develop": GIPdevelop,
        "install": GIPinstall
    }
)
