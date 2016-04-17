#!/usr/bin/env python

import os
import numpy as np
import gippy
import unittest
from datetime import datetime
import gippy.algorithms as alg
from utils import get_test_image

"""
Included are some tests for doing processing in NumPy instead of Gippy,
for doing speed comparisons. To see the durations of each test use:
    $ nosetests test --with-timer -v
"""


class GeoRasterTests(unittest.TestCase):
    """ Speed tests vs NumPy """

    def setUp(self):
        """ Configure options """
        gippy.Options.SetVerbose(1)
        gippy.Options.SetChunkSize(128.0)

    def test_sqrt(self):
        """ Test sqrt using gippy """
        geoimg = get_test_image()
        for band in geoimg:
            vals = band.sqrt().Read()
            # check against numpy

    def test_stats(self):
        """ Test statistics speed using gippy """
        geoimg = get_test_image()
        for band in geoimg:
            stats = band.Stats()
            # check against numpy

    def test_ndvi(self):
        """ Test NDVI using gippy """
        geoimg = get_test_image()
        fout = os.path.splitext(geoimg.Filename())[0] + '_gippy_ndvi'
        alg.Indices(geoimg, {'ndvi': fout})
        geoimg = None

    def test_ndvi_numpy(self):
        """ Test NDVI separately using numpy for speed comparison """
        geoimg = get_test_image()
        nodata = geoimg[0].NoData()
        red = geoimg['RED'].Read().astype('double')
        nir = geoimg['NIR'].Read().astype('double')
        ndvi = np.zeros(red.shape) + nodata
        inds = np.logical_and(red != nodata, nir != nodata)
        ndvi[inds] = (nir[inds] - red[inds])/(nir[inds] + red[inds])
        fout = os.path.splitext(geoimg.Filename())[0] + '_numpy_ndvi'
        geoimgout = gippy.GeoImage.create_from(fout, geoimg, 1, "float64")
        geoimgout[0].Write(ndvi)
        geoimgout = None
        geoimg = None
