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
        gp.Options.SetVerbose(3)
        gp.Options.SetChunkSize(4.0)

    def create_image(self, filename='', size=(1, 1000, 1000), dtype='uint8', temp=False):
        return gp.GeoImage.create(filename, xsize=size[1], ysize=size[2], bsize=size[0],
                                  dtype=dtype, temp=temp)

    def test_open(self):
        """ Open existing image """
        geoimg = get_test_image()
        self.assertTrue(geoimg.XSize() > 0)
        self.assertTrue(geoimg.YSize() > 0)

    def test_create(self):
        """ Create single band image """
        fout = 'test.tif'
        geoimg = self.create_image(fout)
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(os.path.exists(fout))
        os.remove(fout)

    def test_create_multiband(self):
        """ Create an RGB image """
        fout = 'test_3band.tif'
        geoimg = self.create_image(fout, (3, 1000, 1000))
        geoimg.SetBandNames(['green', 'red', 'blue'])
        # test selection of bands
        geoimg2 = geoimg.select(["red"])
        self.assertTrue(geoimg2.NumBands() == 1)
        self.assertTrue(geoimg["red"].Description() == "red")
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
        fname = geoimg.Filename()
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
