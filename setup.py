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
from setuptools.command.bdist_egg import bdist_egg
from distutils import sysconfig
from wheel.bdist_wheel import bdist_wheel

__version__ = load_source('gippy.version', 'gippy/version.py').__version__

# get the dependencies and installs
with open('requirements.txt') as fid:
    install_requires = [l.strip() for l in fid.readlines() if l]

with open('requirements-dev.txt') as fid:
    test_requires = [l.strip() for l in fid.readlines() if l]

# logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig()
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


class _develop(develop):
    # development installation (editable links via pip install -e)
    def finalize_options(self):
        log.debug('_develop finalize_options')
        develop.finalize_options(self)
        # TODO - remove the libs appended by _install which is called
        # for some reason during the develop.finalize_options(self) call
        # and which generates a warning (low priority)
        add_runtime_library_dirs(os.path.abspath('./'))

    def run(self):
        # for some reason we must get build_dir this way, which is available
        # to the install class, but not to the develop class (but install
        # options are called by develop options so global can be set there)
        global build_dir
        log.debug('_develop run')
        # build extension before packaging (maybe not needed for dev?)
        self.run_command('build_ext')
        develop.run(self)
        if sys.platform == 'darwin':
            # change the link path set in the library
            update_lib_path_mac(
                os.path.join(build_dir, gip_module._file_name),
                os.path.join(os.path.abspath(build_dir), gip_module._file_name)
            )


class _install(install):
    def finalize_options(self):
        global build_dir
        install.finalize_options(self)
        log.debug('_install finalize_options')
        build_dir = self.build_lib
        # know where to find libgip for linking
        swig_modules[0].library_dirs.append(os.path.join(self.build_lib, 'gippy'))
        for m in swig_modules:
            log.debug('%s library_dirs: %s' % (m.name, ' '.join(m.library_dirs)))
        # add libgip to runtime
        libpath = os.path.join(self.install_lib, 'gippy')
        add_runtime_library_dirs(libpath)

    def run(self):
        log.debug('_install run')
        # ensure extension built before packaging
        self.run_command('build_ext')
        install.run(self)
        if sys.platform == 'darwin':
            # change the link path to point to the install dir
            update_lib_path_mac(
                os.path.join(self.build_lib, gip_module._file_name),
                os.path.join(self.install_lib, gip_module._file_name),
                self.install_lib
            )


class _bdist_egg(bdist_egg):
    def run(self):
        log.debug('_bdist_egg run')
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_egg.run(self)


class _bdist_wheel(bdist_wheel):
    def run(self):
        global build_dir
        log.debug('_bdist_wheel run')
        self.distribution.ext_modules = swig_modules
        self.run_command('build_ext')
        bdist_wheel.run(self)
        if sys.platform == 'darwin':
            # change the link path set in the library
            update_lib_path_mac(
                os.path.join(build_dir, gip_module._file_name),
                os.path.join(os.path.abspath(build_dir), gip_module._file_name)
            )        


def add_runtime_library_dirs(path):
    path = os.path.abspath(path)
    if sys.platform != 'darwin':
        for m in swig_modules:
            m.runtime_library_dirs.append(path)
            log.debug('%s runtime_library_dirs: %s' % (m.name, ' '.join(m.runtime_library_dirs)))


# use 'otool -L filename.so' to see the linked libraries in an
# extension. This function updates swig .so files with absolute
# pathnames since clang insists on only using relative (it ignores rpath)
def update_lib_path_mac(oldpath, newpath, modpath=None):
    for m in swig_modules:
        if modpath is None:
            fin = os.path.basename(m._file_name)
        else:
            fin = os.path.join(modpath, m._file_name)
        cmd = [
            'install_name_tool',
            '-change',
            oldpath,
            newpath,
            fin
        ]
        out = subprocess.check_output(cmd)
        log.debug(out)


# GDAL config parameters
gdal_config = CConfig(os.environ.get('GDAL_CONFIG', 'gdal-config'))

extra_compile_args = ['-fPIC', '-O3', '-std=c++11']

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
    #cppsrc = os.path.join('gippy', n + '_wrap.cpp')
    #src = cppsrc if  os.path.exists(cppsrc) else src
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

# global so the build dir can be passed from install class options to
# develop class, which doesn't have access to build dir location normally
global build_dir


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
    install_requires=install_requires,
    test_suite='nose.collector',
    tests_require=test_requires,
    cmdclass={
        "develop": _develop,
        "install": _install,
        "bdist_egg": _bdist_egg,
        "bdist_wheel": _bdist_wheel,
    }
)
