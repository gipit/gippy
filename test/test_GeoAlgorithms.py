#!/usr/bin/env python

import os
import gippy.algorithms as alg
import unittest
from utils import get_test_image


class GeoAlgorithmsTests(unittest.TestCase):

    def test_rxd(self):
        """ Test RX Detector algorithm """
        geoimg = get_test_image().select(['red', 'green', 'blue'])
        rxd = alg.rxd(geoimg)
        self.assertEqual(rxd.bandnames()[0], "RXD")
        self.assertEqual(rxd.xsize(), geoimg.xsize())
        self.assertEqual(rxd.ysize(), geoimg.ysize())
        self.assertEqual(rxd.nbands(), 1)
        fname = rxd.filename()
        rxd = None
        self.assertFalse(os.path.exists(fname))

    def test_pansharpen(self):
        """ Test pan-sharpening algorithm """
        geoimg = get_test_image().select(['red', 'green', 'blue', 'nir'])
        panimg = get_test_image(bands=['pan'])
        fout = 'test-pansharpen.tif'
        imgout = alg.pansharp_brovey(geoimg, panimg, filename=fout)
