
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

# Authors and Contributors

    Matthew Hanson
    Ian Cooke

# GIPPY

GIPPY is a library of python bindings to a C++ library called GIPS. GIPS is built on top of GDAL and CImg, an image processing template library. GIPPY provides a similar, yet simpler interface than GDAL for opening, creating, and reading geospatial raster files. Convenience functions have been added to make common tasks achievable with fewer lines of code.

Most notably GIPPY adds image processing functionality on top of GDAL for easier automation of processing functions. The main objects in the GIPPY library are the GeoRaster, which is a single raster band, and a GeoImage, which is a collection of GeoRaster objects (possibly from different files).  GeoImage and GeoRaster objects support various processing operations (e.g., +, -, log, abs) that can be chained together and saved as a processing chain.  The processing does not actually occur until the file is read (frequently followed by a write to a new file).  Reading may also occur in chunks, thereby facilitating the processing of very large files.

# Testing
At the moment, our testing is predominantly python 2.7 on Linux (Ubuntu 14.04LTS) with some work being done to support OS X.
