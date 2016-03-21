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
        gippy.Options.SetVerbose(3)
        gippy.Options.SetChunkSize(1024.0)

    def test_sqrt(self):
        """ Test sqrt using gippy """
        geoimg = get_test_image()
        for band in geoimg:
            vals = band.sqrt().Read()
            print 'gippy', band.Basename(), vals.shape

    def test_sqrt_numpy(self):
        """ Test sqrt using numpy for speed comparison """
        geoimg = get_test_image()
        for band in geoimg:
            vals = np.sqrt(band.Read())
            print 'numpy', band.Basename(), vals.shape

    def test_stats(self):
        """ Test statistics speed using gippy """
        geoimg = get_test_image()
        for band in geoimg:
            stats = band.Stats()
            print '%s: %s' % (band.Basename(), ', '.join(map(str, stats)))

    def test_stats_numpy(self):
        """ Test statistics speed using numpy for speed comparison """
        geoimg = get_test_image()
        for band in geoimg:
            nodata = band.NoDataValue()
            img = band.Read()
            subimg = img[img != nodata]
            stats = [subimg.min(), subimg.max(), subimg.mean(), subimg.std()]
            print '%s: %s' % (band.Basename(), ', '.join(map(str, stats)))

    def test_ndvi(self):
        """ Test NDVI using gippy """
        geoimg = get_test_image()
        fout = os.path.splitext(geoimg.Filename())[0] + '_gippy_ndvi'
        alg.Indices(geoimg, {'ndvi': fout})
        geoimg = None

    def test_ndvi_numpy(self):
        """ Test NDVI using numpy for speed comparison """
        geoimg = get_test_image()
        nodata = geoimg[0].NoDataValue()
        red = geoimg['RED'].Read().astype('double')
        nir = geoimg['NIR'].Read().astype('double')
        start = datetime.now()
        ndvi = np.ones(red.shape)
        #inds = red != nodata
        #ndvi[inds] = (nir[inds] - red[inds])/(nir[inds] + red[inds])
        ndvi = np.true_divide(nir - red, nir + red)
        ndvi[red == nodata] = nodata
        print 'calc and processed in %s' % (datetime.now() - start)
        fout = os.path.splitext(geoimg.Filename())[0] + '_numpy_ndvi'
        geoimgout = gippy.GeoImage(fout, geoimg, gippy.DataType("Float64"), 1)
        geoimgout[0].Write(ndvi)
        geoimgout = None
        geoimg = None
