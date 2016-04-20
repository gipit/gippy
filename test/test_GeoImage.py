#!/usr/bin/env python

import os
import numpy as np
import gippy as gp
import unittest
from datetime import datetime
import gippy.algorithms as alg
from utils import get_test_image
from nose.tools import set_trace


class GeoImageTests(unittest.TestCase):

    prefix = 'test-'

    def setUp(self):
        """ Configure options """
        gp.Options.set_verbose(3)
        gp.Options.set_chunksize(4.0)

    def create_image(self, filename='', size=(1, 1000, 1000), dtype='uint8', temp=False):
        return gp.GeoImage.create(filename,
                                  xsz=size[1], ysz=size[2], nb=size[0],
                                  bbox=np.array([0.0, 0.0, 1.0, 1.0]),
                                  dtype=dtype, temp=temp)

    def test_open(self):
        """ Open existing image """
        geoimg = get_test_image()
        self.assertTrue(geoimg.xsize() > 0)
        self.assertTrue(geoimg.ysize() > 0)

    def test_create(self):
        """ Create single band image """
        fout = 'test.tif'
        geoimg = self.create_image(fout)
        self.assertTrue(geoimg.xsize() == 1000)
        self.assertTrue(geoimg.ysize() == 1000)
        self.assertTrue(os.path.exists(fout))
        # test resolution
        res = geoimg.resolution()
        self.assertEqual(res.x(), 1.0/geoimg.xsize())
        self.assertEqual(res.y(), -1.0/geoimg.ysize())
        os.remove(fout)

    def test_create_multiband(self):
        """ Create an RGB image """
        fout = 'test_3band.tif'
        geoimg = self.create_image(fout, (3, 1000, 1000))
        geoimg.set_bandnames(['green', 'red', 'blue'])
        # test selection of bands
        geoimg2 = geoimg.select(["red"])
        self.assertTrue(geoimg2.nbands() == 1)
        self.assertTrue(geoimg["red"].description() == "red")
        geoimg = None
        geoimg2 = None
        os.remove(fout)

    def test_create_temp_file(self):
        """ Create a temp file that is deleted when last reference gone """
        fout = self.prefix + '_temp.tif'
        geoimg = self.create_image(fout, size=(5, 1000, 1000), temp=True)
        self.assertTrue(os.path.exists(fout))
        # keep a band
        band = geoimg[1]
        geoimg = None
        # band still references file
        self.assertTrue(os.path.exists(fout))
        band = None
        # file should now have been deleted
        self.assertFalse(os.path.exists(fout))

    def test_create_autoname_temp(self):
        """ Create temp file with auto-generated filename """
        geoimg = self.create_image(size=(5, 1000, 1000))
        fname = geoimg.filename()
        self.assertTrue(os.path.exists(fname))
        geoimg = None
        self.assertFalse(os.path.exists(fname))

    def test_autoscale(self):
        """ Auto scale each band in image """
        geoimg = get_test_image()
        for band in geoimg:
            self.assertTrue(band.min() != 1.0)
            self.assertTrue(band.max() != 255.0)
        geoimg2 = geoimg.autoscale(minout=1.0, maxout=255.0)
        for band in geoimg2:
            self.assertTrue(band.min() == 1)
            self.assertTrue(band.max() == 255)

    def test_save(self):
        """ Save image as new image with different datatype """
        fname = 'test-byte.tif'
        geoimg = get_test_image().autoscale(1.0, 255.0).save(fname, 'uint8')
        geoimg = None
        geoimg = gp.GeoImage(fname)
        self.assertEqual(geoimg.type().string(), 'uint8')
        self.assertEqual(geoimg[0].min(), 1.0)
        self.assertEqual(geoimg[0].max(), 255.0)
