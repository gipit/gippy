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


Table of Contents
+++++++++++++++++

.. toctree::
   :maxdepth: 3

   install
   quickstart


Indices and tables
==================

* :ref:`search`

