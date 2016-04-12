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
from numpy import get_include as numpy_get_include
from imp import load_source

# setup imports
from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_egg import bdist_egg
from distutils import sysconfig
from wheel.bdist_wheel import bdist_wheel

__version__ = load_source('gippy.version', 'gippy/version.py').__version__

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))


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
        # log.info('%s %s: %r', self.cmd, option, result)
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


class _build_ext(build_ext):
    def finalize_options(self):
        log.debug('_build_ext finalize_options') # %s %s' % (install_dir, self.distribution))
        build_ext.finalize_options(self)

        #print self.distribution  
        #mock = _install(self.distribution)
        #mock.finalize_options()
        #install_dir = os.path.join(mock.install_lib, "gippy")

        # Workaround for OSX because rpath doesn't work there, is to build
        # in final module directory. This requires `pip uninstall gippy`
        # before re-installing
        #if sys.platform == 'darwin':
        #    self.build_lib = mock.install_lib

        # ensure that swig modules can find libgip
        # extensions seems to be referenced, adding it to the first one updates all swig_modules
        swig_modules[0].library_dirs.append(os.path.join(self.build_lib, 'gippy'))
        for m in swig_modules:
            log.debug('%s library_dirs: %s' % (m.name, ' '.join(m.library_dirs)))
            #if sys.platform != 'darwin':
            #    m.library_dirs.append(os.path.join(self.build_lib, "gippy"))

# ensure swig extension built before packaging

class _develop(develop):
    def finalize_options(self):
        log.debug('_develop finalize_options')
        print self.build_directory
        develop.finalize_options(self)
        add_runtime_library_dirs(os.path.abspath('./'))

    def run(self):
        log.debug('_develop run')
        self.run_command('build_ext')
        develop.run(self)


class _install(install):
    def finalize_options(self):
        install.finalize_options(self)
        log.debug('_install finalize_options')
        print self.install_lib, self.build_lib
        if sys.platform == 'darwin':
            self.build_lib = self.install_lib
        add_runtime_library_dirs(os.path.join(self.install_lib, 'gippy'))

    def run(self):
        log.debug('_install run')
        self.run_command('build_ext')
        install.run(self)


class _bdist_egg(bdist_egg):
    def run(self):
        log.debug('_bdist_egg run')
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_egg.run(self)


class _bdist_wheel(bdist_wheel):
    def run(self):
        log.debug('_bdist_wheel run')
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_wheel.run(self)


def add_runtime_library_dirs(path):
    for m in swig_modules:
        m.runtime_library_dirs.append(path)
        log.debug('%s runtime_library_dirs: %s' % (m.name, ' '.join(m.runtime_library_dirs)))    


# GDAL config parameters
gdal_config = CConfig(os.environ.get('GDAL_CONFIG', 'gdal-config'))

extra_compile_args = ['-fPIC', '-O3', '-std=c++11']

extra_link_args = gdal_config.extra_link_args

lib_dirs = gdal_config.lib_dirs + ['./']

if sys.platform == 'darwin':
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

    ldshared = sysconfig.get_config_var('LDSHARED')

    sysconfig._config_vars['LDSHARED'] = re.sub(
        ' +', ' ',
        ldshared.replace('-bundle', '-dynamiclib')
    )

    extra_compile_args.append('-mmacosx-version-min=10.8')
    # silence various warnings
    extra_compile_args.append('-Wno-absolute-value')
    extra_compile_args.append('-Wno-shift-negative-value')
    extra_compile_args.append('-Wno-parentheses-equality')
    extra_link_args.append('-mmacosx-version-min=10.8')
else:
    # Remove the "-Wstrict-prototypes" compiler option that swig adds, which isn't valid for C++.
    cfg_vars = sysconfig.get_config_vars()
    for key, value in cfg_vars.items():
        if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

gip_module = Extension(
    name=os.path.join("gippy", "libgip"),
    sources=glob.glob('GIP/*.cpp'),
    include_dirs=['GIP', numpy_get_include()] + gdal_config.include,
    library_dirs=lib_dirs,
    libraries=[
        'pthread'
    ] + gdal_config.libs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

swig_modules = []
for n in ['gippy', 'algorithms']:
    swig_modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[os.path.join('gippy', n + '.i')],
            swig_opts=['-c++', '-w509', '-IGIP', '-fcompact', '-fvirtual'],  # '-keyword'],,
            include_dirs=['GIP', numpy_get_include()] + gdal_config.include,
            library_dirs=lib_dirs,
            libraries=[
                'gip', 'pthread'
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
        "build_ext": _build_ext,
        "develop": _develop,
        "install": _install,
        "bdist_egg": _bdist_egg,
        "bdist_wheel": _bdist_wheel,
    }
)
