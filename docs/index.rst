.. GIPPY documentation master file, created by
   sphinx-quickstart on Mon Apr 25 13:24:38 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GIPPY
+++++

.. image:: https://travis-ci.org/gipit/gippy.svg?branch=develop
    :target: https://travis-ci.org/gipit/gippy

Gippy is a Python library for image processing of geospatial raster data. Gippy includes a C++ library, libgip, that encapsulates the functionality of `GDAL <http://www.gdal.org/>`_ and `CImg <http://cimg.eu/>`_, with Python bindings generated with `SWIG <http://www.swig.org/>`_. Gippy handles issues common to geospatial data, such as handling of nodata values and chunking up of very large images by saving chains of functions and processing image in pieces when finally read.  It is reasonably lightweight, requiring only GDAL system libraries and a reasonably modern C++ compiler.

Gippy uses the GDAL C++ API, but provides a simplified object-oriented and Pythonic interface to using data. Gippy provides another level of abstraction where an image is a collection of raster bands and contains additional metadata important for processing.  A GeoImage can contain bands from different files, and images can be opened from multiple files, combined, raster bands removed, reordered, then saved as a new file. In Gippy, the focus is on manipulation and processing of a collection of raster bands for a given region of interest.

CImg is a C++ template library for image processing. As a template library, it's source code is included with Gippy and requires no additional installation. CImg allows images to be used as objects, with a variety of member mathematical functions, very much similar to NumPy. Gippy encapsulates this functionality, allowing raster bands to be used as objects, where they may be multiplied by a constant, scaled, or have some other mathematical operator applied. All of these functions are chained together and then, upon a read request, CImg is used on each chunk in turn and the processing chain applied.

Features
========

Band number abstraction
-----------------------

When a GeoImage is opened or created bandnames can be assigned using the bandnames keyword to open, or setting them after creation. Bands can be referenced via band index or the band name and can be iterated over as a collection. This allows algorithms to written to target specific types of bands (e.g., red, nir, lwir). No more having to worry about what band numbers are what for your data, just write a custom open function for your data and set the band names when you open it, and use band names in the rest of your code.

Nodata
------

While typically not found in traditional image processing applications, "no data" values are extremely common in geospatial data. Projected rasters will contain nodata values outside the data boundary, but included within the bounding box in projected space. Additionally, nodata values may be used to mask out invalid pixels (such as determine from a quality band or a cloud detection algorithm), or may be due to data missing due to sensor issues (e.g., strips of nodata in Landsat7 due to the broken scan line corrector mirrors). Gippy propogates any nodata value through the chain and creates output where nodata pixels stay as nodata pixels (even if the actual nodata value is changed when writing an output file).

Process chains
--------------

Gippy supports the chaining together of processing operations (e.g., +, -, log, abs) on raster bands. These operations are stored as C++ lambda functions in the GeoRaster object and are applied each time the data is read from disk (if user requested or due to a save).

Image Chunking
--------------

Gippy is most often used to procss input data to create a new data file, such as a GeoTIFF.  When using a Gippy algorithm or save() function, the image is automatically chunked up into pieces no larger than a default chunk size and the processing applied in pieces. Chunks can be used within Python code as well, so piecewise processing can be done in custom algorithms.

Algorithms
----------

Gippy includes an algorithm module, which is a collection of functions that operate on GeoImage objects. There are currently only a handful of functions, but this will be expanded upon in future versions. Gippy algorithms take in one or more images, parameters unique to the algorithm, and an output filename and output file options. Currently the algorithms module includes indices() which can calculate a variety of indices (e.g., NDVI, NDWI, LSWI, SATVI) in one pass, acca() and fmask() cloud detection algorithms for Landsat7, cookie_cutter() for mosaicking together scenes, linear_transform() for applying basis vectors to spectral data, pansharpening, and a rxd(), a multispectral anomaly detection algorithm. 


Table of Contents
+++++++++++++++++

.. toctree::
   :maxdepth: 3

   install
   quickstart
   algorithms


Indices and tables
==================

* :ref:`search`

