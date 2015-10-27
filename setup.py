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
import sys
import glob
import re
import subprocess
import logging
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_egg import bdist_egg
from distutils import sysconfig

from wheel.bdist_wheel import bdist_wheel
import numpy
import imp

logging.basicConfig()
log = logging.getLogger(__file__)
__version__ = imp.load_source('gippy.version', 'gippy/version.py').__version__


class CConfig(object):
    """Interface to config options from any utility"""

    def __init__(self, cmd):
        self.cmd = cmd
        self.get_include()
        self.get_libs()

    def get(self, option):
        try:
            stdout, stderr = subprocess.Popen(
                [self.cmd, option],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        except OSError:
            # e.g., [Errno 2] No such file or directory
            raise OSError("Could not find script")
        if stderr and not stdout:
            raise ValueError(stderr.strip())
        if sys.version_info[0] >= 3:
            result = stdout.decode('ascii').strip()
        else:
            result = stdout.strip()
        log.debug('%s %s: %r', self.cmd, option, result)
        return result

    def get_include(self):
        self.include = []
        for item in self.get('--cflags').split():
            if item.startswith("-I"):
                self.include.extend(item[2:].split(":"))
        return self.include

    def get_libs(self):
        self.libs = []
        self.lib_dirs = []
        self.extra_link_args = []
        for item in self.get('--libs').split():
            if item.startswith("-L"):
                self.lib_dirs.extend(item[2:].split(":"))
            elif item.startswith("-l"):
                self.libs.append(item[2:])
            else:
                # e.g. -framework GEOS
                self.extra_link_args.append(item)

    def version(self):
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', self.get('--version').strip())
        return tuple(map(int, match.groups()))


class gippy_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # ensure that swig modules can find libgip
        for m in swig_modules:
            build_dir = os.path.join(self.build_lib, os.path.dirname(m.name))
            m.library_dirs.append(build_dir)
            m.runtime_library_dirs.append(os.path.abspath('./'))
            m.runtime_library_dirs.append(build_dir)

class gippy_develop(develop):
    def run(self):
        self.run_command('build_ext')
        develop.run(self)


class gippy_install(install):
    def run(self):
        # ensure swig extension built before packaging
        self.run_command('build_ext')
        install.run(self)


class gippy_bdist_egg(bdist_egg):
    def run(self):
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_egg.run(self)


class gippy_bdist_wheel(bdist_wheel):
    def run(self):
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_wheel.run(self)


def add_runtime_library_dirs(path):
    for m in swig_modules:
        m.runtime_library_dirs.append(os.path.join(path, os.path.dirname(m.name)))

# GDAL config parameters
gdal_config = CConfig(os.environ.get('GDAL_CONFIG', 'gdal-config'))

extra_compile_args = ['-fPIC', '-O3', '-std=c++11', '-DBOOST_LOG_DYN_LINK']

extra_link_args = ['-Wl,--export-dynamic'] + gdal_config.extra_link_args

if sys.platform == 'darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

    ldshared = sysconfig.get_config_var('LDSHARED')

    sysconfig._config_vars['LDSHARED'] = re.sub(
        ' +', ' ',
        ldshared.replace('-bundle', '-dynamiclib')
    )

    extra_compile_args.append('-mmacosx-version-min=10.8')
    extra_compile_args.append('-Wno-absolute-value')
    # silence warning coming from boost python macros which
    # would is hard to silence via pragma
    extra_compile_args.append('-Wno-parentheses-equality')
    extra_link_args.append('-mmacosx-version-min=10.8')

gip_module =  Extension(
    name=os.path.join("gippy", "libgip"),
    sources=glob.glob('GIP/*.cpp'),
    include_dirs=['GIP', numpy.get_include()] + gdal_config.include,
    library_dirs=gdal_config.lib_dirs,
    libraries=[
        'boost_system', 'boost_filesystem',
        'boost_log', 'pthread'
    ] + gdal_config.libs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

swig_modules = []
for n in ['gippy', 'algorithms', 'tests']:
    swig_modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[os.path.join('gippy', n + '.i')],
            swig_opts=['-c++', '-w509', '-IGIP'],  # '-keyword'],,
            include_dirs=['GIP', numpy.get_include()] + gdal_config.include,
            library_dirs=gdal_config.lib_dirs,
            libraries=[
                'gip', 'boost_system', 'boost_filesystem',
                'boost_log', 'pthread'
            ] + gdal_config.libs,  # ,'X11'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    )


setup(
    name='gippy',
    version=__version__,
    description='Geospatial Image Processing for Python',
    author='Matthew Hanson',
    author_email='matt.a.hanson@gmail.com',
    license='Apache v2.0',
    # platform_tag='linux_x86_64',
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
        "develop": develop,
        "install": install,
        "bdist_egg": gippy_bdist_egg,
        "bdist_wheel": gippy_bdist_wheel,
    }
)
