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
import shutil
from numpy import get_include as numpy_get_include
from imp import load_source

# setup imports
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_py import build_py

#from wheel.bdist_wheel import bdist_wheel
from distutils import sysconfig

__version__ = load_source('gippy.version', 'gippy/version.py').__version__

# get the dependencies and installs
with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

with open('requirements-dev.txt') as fid:
    test_requires = [l.strip() for l in fid.readlines() if l]

# logging
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig()
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
    # builds the external modules: libgip.so, _gippy.so, _algorithms.so
    def run(self):
        log.debug('_build_ext run')
        for m in swig_modules:
            # know where to find libgip for linking
            m.library_dirs.append(os.path.join(self.build_lib, 'gippy'))
            # on linux add to rpath
            if sys.platform != 'darwin':
                m.runtime_library_dirs.append('$ORIGIN')

        # in python3 the created .so files have names tagged to version, see PEP 3147, 3149
        if sysconfig.get_config_var('SOABI') is not None:
            # create link
            libpath = os.path.join(self.build_lib, 'gippy')
            if not os.path.exists(libpath):
                os.makedirs(libpath)
            link = os.path.join(self.build_lib, gip_module._full_name + '.so')
            os.symlink(os.path.basename(gip_module._file_name), link)

        # build the extensionss
        super().run()

        # for mac update runtime library location. Use otool -L to see shared libs in a .so
        if sys.platform == 'darwin':
            old_name = os.path.join(self.build_lib, gip_module._file_name)
            new_name = '@loader_path/libgip.so'  # %s' % os.path.basename(gip_module._file_name)
            cmd0 = 'install_name_tool -change %s %s' % (old_name, new_name)
            for m in swig_modules:
                cmd = '%s %s' % (cmd0, os.path.join(os.path.abspath(self.build_lib), m._file_name))
                log.debug(cmd)
                out = subprocess.check_output(cmd.split(' '))
                # also update top level .so files 'develop' likes to create (still not sure why)
                if os.path.exists(os.path.basename(m._file_name)):
                    cmd = '%s %s' % (cmd0, os.path.basename(m._file_name))
                    log.debug(cmd)
                    out = subprocess.check_output(cmd.split(' '))
                log.debug(out)


class _develop(develop):
    def run(self):
        log.debug('_develop run')
        super().run()
        # move lib files into gippy directory
        [shutil.move(f, 'gippy/') for f in glob.glob('*.so')]
        if sysconfig.get_config_var('SOABI') is not None:
            # rename libgip if Python 3.2+
            os.rename(gip_module._file_name, 'gippy/libgip.so')


class _build_py(build_py):
    def run(self):
        log.debug('_build_py run')
        self.run_command("build_ext")
        return super().run()


#class _bdist_wheel(bdist_wheel):
    # binary wheel
#    def run(self):
#        log.debug('_bdist_wheel run')
#        self.distribution.ext_modules = [gip_module] + swig_modules
#        self.run_command('build_ext')
#        bdist_wheel.run(self)


# GDAL config parameters
gdal_config = CConfig(os.environ.get('GDAL_CONFIG', 'gdal-config'))

extra_compile_args = ['-fPIC', '-O3', '-std=c++11']

if gdal_config.version()[0] == 2:
    extra_compile_args.append('-D GDAL2')

extra_link_args = gdal_config.extra_link_args

# not sure if current directory is necessary here
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
    extra_link_args.append('-mmacosx-version-min=10.8')
    # silence various warnings
    extra_compile_args.append('-Wno-absolute-value')
    extra_compile_args.append('-Wno-shift-negative-value')
    extra_compile_args.append('-Wno-parentheses-equality')
    extra_compile_args.append('-Wno-deprecated-declarations')
else:
    # Remove the "-Wstrict-prototypes" compiler option that swig adds, which isn't valid for C++.
    cfg_vars = sysconfig.get_config_vars()
    for key, value in cfg_vars.items():
        if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")
    extra_compile_args.append('-Wno-maybe-uninitialized')

# the libgip.so module containing all the C++ code
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

# the swig .so modules containing the C++ code that wraps libgip.so
swig_modules = []
for n in ['gippy', 'algorithms']:
    src = os.path.join('gippy', n + '.i')
    cppsrc = os.path.join('gippy', n + '_wrap.cpp')
    src = cppsrc if os.path.exists(cppsrc) else src
    swig_modules.append(
        Extension(
            name=os.path.join('gippy', '_' + n),
            sources=[src],
            swig_opts=['-c++', '-w509', '-w511', '-w315', '-IGIP', '-fcompact', '-fvirtual', '-keyword'],
            include_dirs=['GIP', numpy_get_include()] + gdal_config.include,
            library_dirs=lib_dirs,
            libraries=[
                'gip', 'pthread'
            ] + gdal_config.libs,
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    ext_modules=[gip_module] + swig_modules,
    packages=['gippy'],
    install_requires=install_requires,
    test_suite='nose.collector',
    tests_require=test_requires,
    cmdclass={
        "build_ext": _build_ext,
        "develop": _develop,
        "build_py": _build_py,
    }
)
