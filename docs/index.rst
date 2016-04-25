.. GIPPY documentation master file, created by
   sphinx-quickstart on Mon Apr 25 13:24:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GIPPY
=====

.. image:: https://travis-ci.org/gipit/gippy.svg?branch=develop
    :target: https://travis-ci.org/gipit/gippy

GIPPY is a library of python bindings to a C++ library called GIPS.

GIPS is built on top of GDAL and CImg, an image processing template library. GIPPY provides a similar, yet simpler interface than GDAL for opening, creating, and reading geospatial raster files. Convenience functions have been added to make common tasks achievable with fewer lines of code.

Most notably GIPPY adds image processing functionality on top of GDAL for easier automation of processing functions.

The main objects in the GIPPY library are the GeoRaster, which is a single raster band, and a GeoImage, which is a collection of GeoRaster objects (possibly from different files).

GeoImage and GeoRaster objects support various processing operations (e.g., +, -, log, abs) that can be chained together and saved as a processing chain.  The processing does not actually occur until the file is read (frequently followed by a write to a new file). Reading may also occur in chunks, thereby facilitating the processing of very large files.

Installation
++++++++++++

GIPPY can be installed directly from PyPi using pip, or can be installed from a clone of the repository.
There are a few dependencies that must be installed first. These notes are for Ubuntu.

1. Install the UbuntuGIS Repository:

.. code::

    $ sudo apt-get install python-software-properties
    $ sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    $ sudo apt-get update


2. Installed required dependencies

.. code::

    $ sudo apt-get install python-dev python-setuptools python-numpy python-gdal g++ libgdal1-dev gdal-bin swig2.0 swig


3. Install pip (if not installed)

.. code::

    $ sudo easy_install pip


4. Install GIPPY (sudo not required if installing to virtual environment)

.. code::

    $ sudo pip install gippy
    -or-
    $ git clone http://github.com/matthewhanson/gippy.git
    $ cd gippy
    $ sudo ./setup.py install


Indices and tables
==================

* :ref:`search`

