#!/usr/bin/env python

import os
import numpy as np
import gippy
import unittest
from datetime import datetime
import gippy.algorithms as alg
from utils import get_test_image


class GeoRasterTests(unittest.TestCase):

    def setUp(self):
        """ Configure options """
        gippy.Options.SetVerbose(3)
        gippy.Options.SetChunkSize(1024.0)

    def test_open(self):
        """ Test opening of an image """
        geoimg = get_test_image()
        self.assertTrue(geoimg.XSize() > 0)
        self.assertTrue(geoimg.YSize() > 0)

    def test_create(self):
        """ Test creation of image """
        fout = 'test.tif'
        geoimg = gippy.GeoImage(fout, 1000, 1000, gippy.GDT_Byte, 1)
        self.assertTrue(geoimg.XSize() == 1000)
        self.assertTrue(geoimg.XSize() == 1000)
        os.remove(fout)
