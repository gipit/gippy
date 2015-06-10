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
from setuptools.command.bdist_egg import bdist_egg
from wheel.bdist_wheel import bdist_wheel
import numpy
import imp

__version__ = imp.load_source('gippy.version', 'gippy/version.py').__version__


class gippy_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # ensure that swig modules can find libgip
        for m in swig_modules:
            m.library_dirs.append(os.path.join(self.build_lib, os.path.dirname(m.name)))


class gippy_develop(develop):
    def finalize_options(self):
        develop.finalize_options(self)
        for m in swig_modules:
            m.runtime_library_dirs.append(os.path.abspath('./'))
    def run(self):
        self.run_command('build_ext')
        develop.run(self)


class gippy_install(install):
    def finalize_options(self):
        install.finalize_options(self)
        add_runtime_library_dirs(self.install_lib)
    def run(self):
        # ensure swig extension built before packaging
        self.run_command('build_ext')
        install.run(self)


class gippy_bdist_egg(bdist_egg):
    def run(self):
        self.distribution.ext_modules = [gip_module] + swig_modules
        self.run_command('build_ext')
        bdist_egg.run(self)


class gippy_bdist_wheel(bdist_wheel):
    def run(self):
        self.distribution.ext_modules = [gip_module] + swig_modules
        self.run_command('build_ext')
        bdist_wheel.run(self)


def add_runtime_library_dirs(path):
    for m in swig_modules:
        m.runtime_library_dirs.append(os.path.join(path, os.path.dirname(m.name)))


# libgip - dynamic shared library
gip_module = Extension(
    name='gippy/libgip',
    sources=glob.glob('GIP/*.cpp'),
    include_dirs=['GIP'],
    language='c++',
    extra_compile_args=['-std=c++11', '-O3', '-DBOOST_LOG_DYN_LINK'],
)


swig_modules = []
for n in ['gippy', 'algorithms', 'tests']:
    swig_modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[os.path.join('gippy', n + '.i')],
            swig_opts=['-c++', '-w509', '-IGIP'],  # '-keyword'],,
            include_dirs=['GIP', numpy.get_include(), '/usr/include/gdal'],
            libraries=['gip', 'gdal', 'boost_system', 'boost_filesystem', 'boost_log', 'pthread'],  # ,'X11'],
            extra_compile_args=['-fPIC', '-std=c++11', '-O3', '-DBOOST_LOG_DYN_LINK']
        )
    )


setup(
    name='gippy',
    version=__version__,
    description='Geospatial Image Processing for Python',
    author='Matthew Hanson',
    author_email='matt.a.hanson@gmail.com',
    license='Apache v2.0',
    #platform_tag='linux_x86_64',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
    ext_modules=[gip_module] + swig_modules,
    packages=['gippy'],
    cmdclass={
        "build_ext": gippy_build_ext,
        "develop": gippy_develop,
        "install": gippy_install,
        "bdist_egg": gippy_bdist_egg,
        "bdist_wheel": gippy_bdist_wheel,
    }
)
