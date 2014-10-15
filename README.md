
    **GIPPY**: Geospatial Image Processing for Python

    Copyright (C) 2014 Matthew A Hanson

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>

# GIPPY

GIPPY is made up of python bindings to a C++ library called GIPS. GIPS is built on top of GDAL and an image processing template library called CImg. GIPPY provides a similar, yet simpler interface than GDAL for opening, creating, and reading geospatial raster files. Convenience functions have been added to make common tasks achievable with fewer lines of code.

Most notably GIPPY adds image processing functionality on top of GDAL for easier automation of processing functions. The main objects in the GIPPY library are the GeoRaster, which is a single raster band, and a GeoImage, which is a collection of GeoRaster objects (possibly from different files).  GeoImage and GeoRaster objects support various processing operations (e.g., +, -, log, abs) that can be chained together and saved as a processing chain.  The processing does not actually occur until the file is read (frequently followed by a write to a new file).  Reading may also occur in chunks, thereby facilitating the processing of very large files.

# Installation

GIPPY can be installed directly from PyPi using pip, or can be installed from a clone of the repository.
There are a few dependencies that must be installed first. These notes are for Ubuntu.

    1) Install the UbuntuGIS Repository:
    $ sudo apt-get install python-software-properties
    $ sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    $ sudo apt-get update

    2) Installed required dependencies
    $ sudo apt-get install python-setuptools python-numpy python-gdal g++ libgdal1-dev gdal-bin libboost-dev swig2.0 swig

    3) Install GIPPY (sudo not required if installing to virtual environment)
    $ sudo pip install gippy
    -or-
    $ git clone http://github.com/matthewhanson/gippy.git
    $ cd gippy
    $ sudo ./setup.py install

# Quickstart

The two main classes in GIPPY are GeoImage and GeoRaster.  A GeoRaster is a single raster band, analagous to GDALRasterBand.  A GeoImage is a collection of GeoRaster objects, similar to GDALDataset however the GeoRaster objects that it contains could be from different locations (different files).

Open existing images

    from gippy import GeoImage

    # Open up image read-only
    image = GeoImage('test.tif')

    # Open up image with write permissions
    image = GeoImage('test.tif', True)

    # Open up multiple files as a single image where numbands = numfiles x numbands
    image = GeoImage(['test1.tif', 'test2.tif', 'test3.tif'])

Creating new images

    import gippy

    # Create new 1000x1000 single-band byte image 
    image = gippy.GeoImage('test.tif', 1000, 1000, 1, gippy.GDT_Byte)

    # Create new image with same properties (size, metadata, SRS) as existing gimage GeoImage
    image = gippy.GeoImage('test.tif', gimage)

    # As above but with different datatype and 4 bands
    image = gippy.GeoImage('test.tif', gimage, gippy.GDT_Int16, 4)



## Development Note

For developing GIPS, it is recommended that you use a python virtual environment 
This allows multiple users on the same system to independently develop without 
collisions. If you are in a virtual environment (ve), install or develop will install
to the ve instead of the system