#!/usr/bin/env python

import os
import gippy.algorithms as alg
import unittest
from utils import get_test_image


class GeoAlgorithmsTests(unittest.TestCase):

    def test_rxd(self):
        geoimg = get_test_image()
        rxd = alg.rxd(geoimg)
        self.assertEqual(rxd.bandnames()[0], "RXD")
        self.assertEqual(rxd.xsize(), geoimg.xsize())
        self.assertEqual(rxd.ysize(), geoimg.ysize())
        self.assertEqual(rxd.nbands(), 1)
        fname = rxd.filename()
        rxd = None
        self.assertTrue(os.path.exists(fname))
