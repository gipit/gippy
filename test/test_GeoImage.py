#!/usr/bin/env python

import os
import numpy as np
import gippy
import unittest
from datetime import datetime
import gippy.algorithms as alg
from utils import get_test_image
from nose.tools import set_trace


class GeoImageTests(unittest.TestCase):

    prefix = 'test-'

    def setUp(self):
        """ Configure options """
        gippy.Options.SetVerbose(4)
        gippy.Options.SetChunkSize(4.0)

    def create_image(self, filename, size=(1, 1000, 1000), dtype='UInt8'):
        geoimg = gippy.GeoImage(filename, size[1], size[2], size[0], gippy.DataType(dtype))
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(geoimg.XSize() == 1000)
        return geoimg

    def test_open(self):
        """ Test opening of an image """
        geoimg = get_test_image()
        self.assertTrue(geoimg.XSize() > 0)
        self.assertTrue(geoimg.YSize() > 0)

    def test_create(self):
        """ Test creation of image """
        fout = 'test.tif'
        geoimg = self.create_image(fout)
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(os.path.exists(fout))
        os.remove(fout)

    def test_create_multiband(self):
        """ Test creation of an RGB image """
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
