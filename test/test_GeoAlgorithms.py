#!/usr/bin/env python

import os
import unittest
from copy import deepcopy
import numpy as np
import gippy as gp
import gippy.algorithms as alg
import gippy.test as gpt


class GeoAlgorithmsTests(unittest.TestCase):

    def setUp(self):
        gp.Options.set_verbose(1)

    def test_rxd(self):
        """ RX anamoly detector """
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
        """ Pansharpen multispectral image with panchromatic image """
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
        """ Create mosaic from multiple images (cookie cutter) """
        bbox1 = np.array([0.0, 0.0, 1.0, 1.0])
        geoimg1 = gp.GeoImage.create(xsz=1000, ysz=1000, bbox=bbox1)
        bbox2 = np.array([1.0, 0.0, 1.0, 1.0])
        geoimg2 = gp.GeoImage.create(xsz=1000, ysz=1000, bbox=bbox2)
        res = geoimg1.resolution()
        imgout = alg.cookie_cutter([geoimg1, geoimg2], xres=res.x(), yres=res.y())
        ext = imgout.extent()
        # This appears to be accurate to 7 decimal places.
        # Is something getting converted from a double to a float somewhere?
        self.assertAlmostEqual(ext.x0(), 0.0)
        self.assertAlmostEqual(ext.y0(), 0.0)
        self.assertAlmostEqual(ext.width(), 2.0, places=6)
        self.assertAlmostEqual(ext.height(), 1.0)  # ''
        self.assertAlmostEqual(imgout.resolution().x(), res.x())
        self.assertAlmostEqual(imgout.resolution().y(), res.y())

    def test_cookiecutter_gain(self):
        """ Cookie cutter on int image with floating point gain """
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        geoimg = gp.GeoImage.create(xsz=1000, ysz=1000, bbox=bbox, dtype='int16')
        geoimg.set_gain(0.0001)
        arr = np.zeros((1000,1000))
        arr[0:500,:] = 0.0002
        geoimg.write(deepcopy(arr))
        res = geoimg.resolution()
        imgout = alg.cookie_cutter([geoimg], xres=res.x(), yres=res.y())
        np.testing.assert_array_almost_equal(arr, imgout.read())

    def test_cookiecutter_real(self):
        """ Cookie cutter on single real image """
        geoimg = gpt.get_test_image().select(['red']) #, 'green', 'blue'])
        iext = geoimg.extent()
        vpath = os.path.join(os.path.dirname(__file__), 'vectors')
        # test with feature of different projection
        feature = gp.GeoVector(os.path.join(vpath, 'aoi1_epsg4326.shp'))
        extin = feature.extent()
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=0.0003, yres=0.0003)
        extout = imgout.extent()
        self.assertAlmostEqual(extout.x0() + 0.00015, extin.x0())
        self.assertAlmostEqual(extout.y0() + 0.00015, extin.y0())
        # cookie cutter will never add more than a pixel and a half in width
        self.assertTrue(extout.x1() - extin.x1() < 0.0045)
        self.assertTrue(extout.y1() - extin.y1() < 0.0045)
        self.assertAlmostEqual(imgout.resolution().x(),  0.0003)
        self.assertAlmostEqual(imgout.resolution().y(), -0.0003)

    def test_cookiecutter_real_reproj(self):
        """ Test with different projection """
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue'])
        vpath = os.path.join(os.path.dirname(__file__), 'vectors')
        feature = gp.GeoVector(os.path.join(vpath, 'aoi1_epsg32416.shp'))
        extin = feature.extent()
        # test extent matches feature
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=30.0, yres=30.0)
        extout = imgout.extent()
        self.assertAlmostEqual(extout.x0() + 15, extin.x0())
        self.assertAlmostEqual(extout.y0() + 15, extin.y0())
        # cookie cutter will never add more than a pixel and a half in width
        self.assertTrue(extout.x1() - extin.x1() < 45.0)
        self.assertTrue(extout.y1() - extin.y1() < 45.0)
        self.assertEqual(imgout.resolution().x(),  30.0)
        self.assertEqual(imgout.resolution().y(), -30.0)

    def test_cookiecutter_real_crop(self):
        """ Test cookie cutter with cropping """
        geoimg = gpt.get_test_image().select(['red', 'green', 'blue'])
        vpath = os.path.join(os.path.dirname(__file__), 'vectors')
        feature = gp.GeoVector(os.path.join(vpath, 'aoi1_epsg32416.shp'))
        imgout = alg.cookie_cutter([geoimg], feature=feature[0], xres=30.0, yres=30.0, crop=True)
        extin = feature.extent()
        extout = imgout.extent()
        self.assertTrue(extout.x0() + 15 >= extin.x0())  # half pixel shift
        self.assertTrue(extout.y0() + 15 >= extin.y0())  # half pixel shift
        # cookie cutter will never add more than a pixel and a half in width
        self.assertTrue(extout.x1() - extin.x1() < 45.0)
        self.assertTrue(extout.y1() - extin.y1() < 45.0)

    def test_ndvi(self):
        """ Calculate NDVI using gippy and apply colortable """
        geoimg = gpt.get_test_image()
        fout = 'test-ndvi.tif'
        imgout = alg.indices(geoimg, ['ndvi'])
        # add colorramp
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        white = np.array([255, 255, 255])
        imgout[0] = imgout[0].scale(-1.0, 1.0, 1, 255)
        imgout = imgout.save(fout, dtype='byte')
        # add color ramp for negative values
        imgout[0].add_colortable(red, white, value1=0, value2=128)
        # add color ramp for positive values
        imgout[0].add_colortable(white, green, value1=128, value2=255)
        # TODO - actually test something here
        os.remove(fout)

    def test_ndvi_numpy(self):
        """ Calculate NDVI using numpy (for speed comparison) """
        geoimg = gpt.get_test_image()
        nodata = geoimg[0].nodata()
        red = geoimg['RED'].read().astype('double')
        nir = geoimg['NIR'].read().astype('double')
        ndvi = np.zeros(red.shape) + nodata
        inds = np.logical_and(red != nodata, nir != nodata)
        ndvi[inds] = (nir[inds] - red[inds])/(nir[inds] + red[inds])
        fout = 'test-ndvi2.tif'
        geoimgout = gp.GeoImage.create_from(geoimg, fout, dtype="float64")
        geoimgout[0].write(ndvi)
        geoimgout = None
        geoimg = None
        os.remove(fout)
