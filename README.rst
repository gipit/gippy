GIPPY
=====

.. image:: https://circleci.com/gh/gipit/gippy.svg?style=svg&circle-token=fb40999b03328dc633a4d654f680eb5e1a6f3a2d
    :target: https://circleci.com/gh/gipit/gippy


Gippy is a Python library for image processing of geospatial raster data. The core of the library is implemented as a C++ library, libgip, with Python bindings automatically generated with `swig <http://www.swig.org/>`_. Gippy encapsulates the functionality of `GDAL <http://www.gdal.org/>`_ and `CImg <http://cimg.eu/>`_ that automatically handles issues common to geospatial data, such as handling of nodata values and chunking up of very large images by saving chains of functions and only processing the image in pieces upon a read request. In addition to providing a library of image processing functions and algorithms, Gippy can also be used as a simpler interface to GDAL for the opening, creating, reading and writing of geospatial raster files in Python.

See the full `documentation <https://gippy.readthedocs.io>`_.


Authors and Contributors
++++++++++++++++++++++++

- Matthew Hanson
- Ian Cooke
- Alireza Jazayeri


.. code::

    **GIPPY**: Geospatial Image Processing for Python

    AUTHOR: Matthew Hanson
    EMAIL:  matt.a.hanson@gmail.com

    Copyright (C) 2015 Applied Geosolutions
    EMAIL: oss@appliedgeosolutions.com

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


