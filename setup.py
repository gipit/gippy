#!/usr/bin/env python
################################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    AUTHOR: Matthew Hanson
#    EMAIL:  matt.a.hanson@gmail.com
#
#    Copyright (C) 2015 Applied Geosolutions
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
################################################################################
"""
setup for GIP and gippy
"""

import os
import glob
from setuptools import setup, Extension, find_packages

from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
import numpy
import imp

__version__ = imp.load_source('gippy.version', 'gippy/version.py').__version__


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
    extra_compile_args=['-std=c++11', '-O3', '-DBOOST_LOG_DYN_LINK'],
)

modules = [gip_module]
names = ['gippy', 'tests', 'algorithms']
for n in names:
    modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[os.path.join('gippy', n + '.i')],
            swig_opts=['-c++', '-w509', '-IGIP'],  # '-keyword'],,
            #swig_opts=['-c++', '-w509', '-IGIP', '-Igippy/gdal/python', '-Igippy/gdal/python/docs'],  # '-keyword'],,
            include_dirs=['GIP', numpy.get_include(), '/usr/include/gdal'],
            libraries=['gip', 'gdal', 'boost_system', 'boost_filesystem', 'boost_log', 'pthread'],  # ,'X11'],
            extra_compile_args=['-fPIC', '-std=c++11', '-DBOOST_LOG_DYN_LINK']
        )
    )


setup(
    name='gippy',
    version=__version__,
    description='Geospatial Image Processing for Python',
    author='Matthew Hanson',
    author_email='matt.a.hanson@gmail.com',
    license='Apache v2.0',
    ext_modules=modules,
    packages=find_packages(),
    cmdclass={
        "develop": gippy_develop,
        "install": gippy_install,
        #"bdist_egg": gippy_bdist_egg,
        "build_ext": gippy_build_ext,
    }
)
