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
        gippy.Options.set_verbose(2)
        gippy.Options.set_chunksize(128.0)

    def test_sqrt(self):
        """ Calculate sqrt of image """
        geoimg = get_test_image()
        for band in geoimg:
            vals = band.sqrt().read()
            mask = band.data_mask() == 1
            # check against numpy
            arr = band.read()


    def test_stats(self):
        """ Calculate statistics using gippy """
        geoimg = get_test_image()
        for band in geoimg:
            stats = band.stats()
            mask = band.data_mask() == 1
            # check against numpy
            arr = band.read()
            self.assertAlmostEqual(arr[mask].min(), stats[0])
            self.assertAlmostEqual(arr[mask].max(), stats[1])
            self.assertAlmostEqual(arr[mask].mean(), stats[2], places=2)

    def test_ndvi(self):
        """ Test NDVI using gippy """
        geoimg = get_test_image()
        fout = os.path.splitext(geoimg.filename())[0] + '_gippy_ndvi.tif'
        alg.indices(geoimg, {'ndvi': fout})
        geoimg = None
        os.remove(fout)

    def test_ndvi_numpy(self):
        """ Test NDVI separately using numpy for speed comparison """
        geoimg = get_test_image()
        nodata = geoimg[0].nodata()
        red = geoimg['RED'].read().astype('double')
        nir = geoimg['NIR'].read().astype('double')
        ndvi = np.zeros(red.shape) + nodata
        inds = np.logical_and(red != nodata, nir != nodata)
        ndvi[inds] = (nir[inds] - red[inds])/(nir[inds] + red[inds])
        fout = os.path.splitext(geoimg.filename())[0] + '_numpy_ndvi.tif'
        geoimgout = gippy.GeoImage.create_from(geoimg, fout, dtype="float64")
        geoimgout[0].write(ndvi)
        geoimgout = None
        geoimg = None
        os.remove(fout)

    def test_scale(self):
        """ Scale image to byte range """
        geoimg = get_test_image()
        for band in geoimg:
            band = band.autoscale(minout=1, maxout=255) #, percent=2.0)
            self.assertTrue(band.min() == 1)
            self.assertTrue(band.max() == 255)
