#!/usr/bin/env python

import os
import numpy as np
import gippy
import unittest
from datetime import datetime
import gippy.algorithms as alg
from utils import get_test_image
from nose.tools import set_trace


class GeoRasterTests(unittest.TestCase):

    def setUp(self):
        """ Configure options """
        gippy.Options.SetVerbose(4)
        gippy.Options.SetChunkSize(4.0)

    def test_open(self):
        """ Test opening of an image """
        geoimg = get_test_image()
        self.assertTrue(geoimg.XSize() > 0)
        self.assertTrue(geoimg.YSize() > 0)

    def test_create(self):
        """ Test creation of image """
        fout = 'test.tif'
        geoimg = gippy.GeoImage(fout, 1000, 1000, 1, gippy.DataType("UInt8"))
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(geoimg.XSize() == 1000)
        os.remove(fout)

    def test_create_multiband(self):
        """ Test creation of an RGB image """
        fout = 'test_3band.tif'
        geoimg = gippy.GeoImage(fout, 1000, 1000, 3, gippy.DataType("UInt8"))
        geoimg.SetBandNames(['green', 'red', 'blue'])
        # need to test something here, add gdalinfo util to parse bits
