#!/usr/bin/env python

import os
import numpy as np
import gippy as gp
import unittest
import gippy.test as gpt
# from nose.tools import raises

"""
Included are some tests for doing processing in NumPy instead of Gippy,
for doing speed comparisons. To see the durations of each test use:
    $ nosetests test --with-timer -v
"""


class GeoRasterTests(unittest.TestCase):
    """ Speed tests vs NumPy """

    def setUp(self):
        """ Configure options """
        gp.Options.set_verbose(1)
        gp.Options.set_chunksize(256.0)

    def test_size(self):
        """ Retrieve size and dimension in pixels """
        # note that xsize and ysize are redefined in GeoRaster from
        # GeoResource, thus it is tested again
        geoimg = gp.GeoImage.create(xsz=500, ysz=1500)
        self.assertEqual(geoimg.xsize(), 500)
        self.assertEqual(geoimg.ysize(), 1500)
        self.assertEqual(geoimg.size(), 1500*500)

    def test_type(self):
        """ Set datatype on create and verify """
        geoimg = gp.GeoImage.create(dtype='uint32')
        self.assertEqual(geoimg.type().string(), 'uint32')

    def test_naming(self):
        """ Get basename and desription """
        fout = 'test-image.tif'
        bname = os.path.splitext(fout)[0]
        bandnames = ['red', 'green', 'blue']
        geoimg = gp.GeoImage.create(fout, nb=3)
        geoimg.set_bandnames(bandnames)
        for i in range(0, 3):
            self.assertEqual(geoimg[i].description(), bandnames[i])
            self.assertEqual(geoimg[i].basename(), '%s[%s]' % (bname, i))
        os.remove(fout)
        # TODO - test color

    def test_gain_and_offset(self):
        """ Set and retrieve gain and offset """
        fout = 'test-gainoffset.tif'
        gains = [2.0, 3.0]
        offsets = [4.0, 5.0]
        geoimg = gp.GeoImage.create(fout, nb=2)
        geoimg[0].set_gain(gains[0])
        geoimg[1].set_gain(gains[1])
        geoimg[0].set_offset(offsets[0])
        geoimg[1].set_offset(offsets[1])
        # check persistance
        geoimg = None
        geoimg = gp.GeoImage(fout)
        for i in range(0, 2):
            self.assertEqual(geoimg[i].gain(), gains[i])
            self.assertEqual(geoimg[i].offset(), offsets[i])
        os.remove(fout)

    def test_nodata(self):
        """ Set nodata and retrieve """
        fout = 'test-nodata.tif'
        geoimg = gp.GeoImage.create(fout, xsz=100, ysz=100)
        geoimg.set_nodata(1)
        self.assertEqual(geoimg[0].nodata(), 1)
        geoimg = None
        geoimg = gp.GeoImage(fout)
        self.assertEqual(geoimg[0].nodata(), 1)
        # check that entire array is nan
        arr = np.where(geoimg.read() == np.nan)
        self.assertEqual(len(arr[0]), 0)
        self.assertEqual(len(arr[1]), 0)
        os.remove(fout)

    def test_bandmeta(self):
        """ Set metadata on band and retrieve """
        fout = 'test-meta.tif'
        geoimg = gp.GeoImage.create(fout, xsz=100, ysz=100)
        geoimg[0].add_bandmeta('TESTKEY', 'TESTVALUE')
        geoimg = None
        geoimg = gp.GeoImage(fout)
        self.assertEqual(geoimg[0].bandmeta('TESTKEY'), 'TESTVALUE')
        os.remove(fout)

    # TODO - test masking

    def test_stats(self):
        """ Calculate statistics using gippy """
        geoimg = gpt.get_test_image()
        for band in geoimg:
            stats = band.stats()
            mask = band.data_mask() == 1
            # check against numpy
            arr = band.read()
            self.assertAlmostEqual(arr[mask].min(), stats[0])
            self.assertAlmostEqual(arr[mask].max(), stats[1])
            self.assertAlmostEqual(arr[mask].mean(), stats[2], places=2)

    def test_scale(self):
        """ Scale image to byte range """
        geoimg = gpt.get_test_image()
        for band in geoimg:
            band = band.autoscale(minout=1, maxout=255, percent=2.0)
            self.assertTrue(band.min() == 1)
            self.assertTrue(band.max() == 255)

    def test_histogram(self):
        """ Calculate histogram of blank data """
        geoimg = gp.GeoImage.create(xsz=10, ysz=10, nb=2)
        arr = np.arange(10).reshape(1, 10) + 1
        for i in range(9):
            arr = np.append(arr, arr, axis=0)
        geoimg[0].write(arr.astype('uint8'))
        hist = geoimg[0].histogram(bins=10, normalize=False)
        self.assertEqual(hist[0], 10)
        self.assertEqual(hist.sum(), geoimg.size())
        hist = geoimg[0].histogram(bins=10)
        self.assertAlmostEqual(hist.sum(), 1.0)
        self.assertAlmostEqual(hist[0], 0.1)
        hist = geoimg[0].histogram(bins=10, normalize=False, cumulative=True)
        self.assertAlmostEqual(hist[-1], geoimg.size())

    def test_real_histogram(self):
        """ Calculate histogram of real data """
        geoimg = gpt.get_test_image()
        hist = geoimg[0].histogram(normalize=False)
        self.assertEqual(len(hist), 100)
        self.assertEqual(hist.sum(), geoimg.size())

    def test_sqrt(self):
        """ Calculate sqrt of image """
        geoimg = gpt.get_test_image().select(['red', 'green', 'swir1', 'nir'])
        for band in geoimg:
            vals = band.sqrt().read()
            mask = band.data_mask() == 1
            # check against numpy
            arr = band.read()
            self.assertTrue((vals[mask] == np.sqrt(arr[mask])).any())

    # TODO - test processing functions
    # Test filters
    def test_laplacian(self):
        """ Test with laplacian filter """
        geoimg = gp.GeoImage.create(xsz=10, ysz=10)
        arr = geoimg.read()
        arr[:, 0:6] = 1
        geoimg[0].write(arr)
        arrout = geoimg[0].laplacian().read()
        self.assertEqual(arrout[0, 5], -1.)
        self.assertEqual(arrout[0, 6], 1.)

    def test_convolve(self):
        """ Convolve an image with a 3x3 kernel """
        geoimg = gp.GeoImage.create(xsz=10, ysz=10)
        arr = geoimg.read() + 1
        geoimg[0].write(arr)
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        arrout = geoimg[0].convolve(kernel, boundary=False).read()
        self.assertEqual(arrout[0, 0], 4)
        self.assertEqual(arrout[5, 5], 9)
        self.assertEqual(arrout[5, 0], 6)

    def test_skeletonize(self):
        """ Skeletonize a binary imager """
        geoimg = gp.GeoImage.create(xsz=10, ysz=10)
        arr = geoimg.read()
        arr[3:8, :] = 1
        geoimg[0].write(arr)
        arrout = geoimg[0].skeletonize().read()

    def test_write(self):
        """ Write arrays of different datatype """
        geoimg = gp.GeoImage.create(xsz=100, ysz=100, dtype='uint8')
        arr = np.ones((100, 100)).astype('uint8')
        geoimg[0].write(arr)
        self.assertTrue(np.array_equal(arr, geoimg[0].read()))
        arr = np.ones((100, 100)).astype('float32')
        geoimg[0].write(arr)
        self.assertTrue(np.array_equal(arr, geoimg[0].read()))

    """
    def test_invalid_args(self):
        # Check that invalid arguments throw error
        geoimg = gippy.GeoImage.create(xsz=100, ysz=100, dtype='uint8')
        try:
            geoimg[0].write('invalid arg')
            geoimg[0].write([1.0, 1.0])
            self.assertTrue(False)
        except:
            pass
    """
