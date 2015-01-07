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
import glob
from setuptools import setup, Extension, find_packages
#from setuptools.extension import Library
from setuptools.command.install import install
from setuptools.command.develop import develop
#from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
import numpy
from gippy.version import __version__


def add_reg(filename):
    """ Add gdal init function and version to the SWIG generated module file """
    f = open(filename, 'a')
    f.write('gip_gdalinit()\n')
    f.close()


class gippy_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        for m in modules:
            m.library_dirs.append(os.path.join(self.build_lib, os.path.dirname(m.name)))

#class gippy_bdist_egg(bdist_egg):
#    def run(self):
#        self.distribution.ext_modules = [gip_module_static, gippy_module]
#        self.run_command('build_ext')
#        add_reg('gippy/gippy.py')
#        bdist_egg.run(self)


class gippy_install(install):
    def finalize_options(self):
        install.finalize_options(self)
        for m in modules:
            print 'module', m, os.path.join(self.install_lib, os.path.dirname(m.name))
            m.runtime_library_dirs.append(os.path.join(self.install_lib, os.path.dirname(m.name)))

    def run(self):
        # ensure swig extension built before packaging
        self.run_command('build_ext')
        install.run(self)

        add_reg(os.path.join(self.install_lib, os.path.dirname(modules[1].name), 'gippy.py'))


class gippy_develop(develop):
    def finalize_options(self):
        develop.finalize_options(self)
        modules[1].runtime_library_dirs.append(os.path.abspath('./'))

    def run(self):
        develop.run(self)
        add_reg(os.path.join(os.path.dirname(modules[1].name), 'gippy.py'))

# Static library
#gip_module_static = Library(
#    name='gip',
#    sources=glob.glob('GIP/*.cpp'),
#    include_dirs=['GIP'],
#    language='c++',
#    extra_compile_args=['-std=c++0x', '-O3'],
#)

# Dynamic shared library
gip_module = Extension(
    name='gippy/libgip',
    sources=glob.glob('GIP/*.cpp'),
    include_dirs=['GIP'],
    language='c++',
    extra_compile_args=['-std=c++0x', '-O3'],
)

modules = [gip_module]
names = ['gippy', 'tests', 'algorithms']
for n in names:
    modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[os.path.join('gippy', n + '.i')],
            swig_opts=['-c++', '-w509', '-IGIP'],  # '-keyword'],,
            include_dirs=['GIP', numpy.get_include()],
            libraries=['gip', 'gdal', 'boost_system', 'boost_filesystem'],  # ,'X11'],
            extra_compile_args=['-fPIC', '-std=c++0x']
        )
    )


setup(
    name='gippy',
    version=__version__,
    description='Geospatial Image Processing for Python',
    author='Matthew Hanson',
    author_email='matt.a.hanson@gmail.com',
    license='GPLv2',
    ext_modules=modules,
    packages=find_packages(),
    #packages=['gippy', 'gippy.tests'],
    #package_dir={'gippy': 'gippy'},
    #py_modules=['gippy', 'gippy.tests'],
    #install_requires = ['numpy'],
    cmdclass={
        "develop": gippy_develop,
        "install": gippy_install,
        #"bdist_egg": gippy_bdist_egg,
        "build_ext": gippy_build_ext,
    }
)
