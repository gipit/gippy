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
from .version import __version__
from gippy import init, DataType, GeoImage, GeoVector, Options

# register GDAL and OGR formats
init()

def mac_update():
    """ update search path on mac """
    import sys
    if sys.platform == 'darwin':
        import os
        from subprocess import check_output
        lib = 'libgip.so'
        path = os.path.dirname(__file__)
        for f in ['_gippy.so', '_algorithms.so']:
            fin = os.path.join(path, f)
    	    cmd = ['install_name_tool', '-change', lib, os.path.join(path, lib), fin]
    	    check_output(cmd)

mac_update()

# cleanup functions
del gippy
del version
del init
del mac_update

