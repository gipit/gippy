#!/usr/bin/env python

import os
import unittest
import numpy as np
import gippy as gp
import gippy.algorithms as alg
import gippy.test as gpt


class GeoAlgorithmsTests(unittest.TestCase):

    def test_rxd(self):
        """ Test RX Detector algorithm """
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue'])
        fout = 'test-rxd.tif'
        rxd = alg.rxd(geoimg, filename=fout)
        self.assertEqual(rxd.bandnames()[0], "RXD")
        self.assertEqual(rxd.xsize(), geoimg.xsize())
        self.assertEqual(rxd.ysize(), geoimg.ysize())
        self.assertEqual(rxd.nbands(), 1)
        rxd = None
        os.remove(fout)

    def test_pansharpen(self):
        """ Test pan-sharpening algorithm """
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue', 'nir'])
        panimg = gpt.get_test_image(bands=['pan'])
        fout = 'test-pansharpen.tif'
        imgout = alg.pansharp_brovey(geoimg, panimg, filename=fout)
        self.assertAlmostEqual(imgout.resolution().x(), panimg.resolution().x(), places=1)
        self.assertAlmostEqual(imgout.resolution().y(), panimg.resolution().y(), places=1)
        self.assertEqual(imgout.nbands(), 4)
        os.remove(fout)
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue'])
        imgout = alg.pansharp_brovey(geoimg, panimg, filename=fout)
        self.assertEqual(imgout.nbands(), 3)
        self.assertAlmostEqual(imgout.resolution().x(), panimg.resolution().x(), places=1)
        self.assertAlmostEqual(imgout.resolution().y(), panimg.resolution().y(), places=1)
        os.remove(fout)

    def test_cookiecutter(self):
        """ Test creating mosaic from multiple images """
        bbox1 = np.array([0.0, 0.0, 1.0, 1.0])
        geoimg1 = gp.GeoImage.create(xsz=1000, ysz=1000, bbox=bbox1)
        bbox2 = np.array([1.0, 0.0, 1.0, 1.0])
        geoimg2 = gp.GeoImage.create(xsz=1000, ysz=1000, bbox=bbox2)
        res = geoimg1.resolution()
        imgout = alg.cookie_cutter([geoimg1, geoimg2], xres=res.x(), yres=res.y())
        ext = imgout.extent()
        self.assertEqual(ext.x0(), 0.0)
        self.assertEqual(ext.y0(), 0.0)
        self.assertEqual(ext.width(), 2.0)
        self.assertEqual(ext.height(), 1.0)

    def test_cookiecutter_real(self):
        """ Test cookie cutter on real image """
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue'])
        vpath = os.path.join(os.path.dirname(__file__), 'vectors')
        # test with feature of different projection
        feature = gp.GeoVector(os.path.join(vpath, 'aoi1_epsg4326.shp'))
        extin = feature.extent()
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=0.0003, yres=0.0003)
        extout = imgout.extent()
        self.assertAlmostEqual(extout.x0(), extin.x0())
        self.assertAlmostEqual(extout.y0(), extin.y0())
        self.assertAlmostEqual(extout.x1(), extin.x1())
        self.assertAlmostEqual(extout.y1(), extin.y1())
        # test with different projection
        feature = gp.GeoVector(os.path.join(vpath, 'aoi1_epsg32416.shp'))
        extin = feature.extent()
        # test extent matches feature
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=30.0, yres=30.0)
        extout = imgout.extent()
        self.assertAlmostEqual(extout.x0(), extin.x0())
        self.assertAlmostEqual(extout.y0(), extin.y0())
        self.assertAlmostEqual(extout.x1(), extin.x1())
        self.assertAlmostEqual(extout.y1(), extin.y1())
        # test cropping
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=30.0, yres=30.0, crop=True)
        extout = imgout.extent()
        self.assertTrue(extout.x0 >= extin.x0())
        self.assertTrue(extout.y0() >= extin.y0())
        self.assertTrue(extout.x1() <= extin.x1())
        self.assertTrue(extout.y1() <= extin.y1())
