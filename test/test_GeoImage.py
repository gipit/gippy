#!/usr/bin/env python

import os
from copy import deepcopy
import numpy as np
from numpy.testing import assert_array_equal
import gippy as gp
import unittest
import gippy.test as gpt


class GeoImageTests(unittest.TestCase):

    prefix = 'test-'

    def setUp(self):
        """ Configure options """
        gp.Options.set_verbose(1)
        gp.Options.set_chunksize(4.0)

    def test0_open(self):
        """ Open existing image """
        geoimg = gpt.get_test_image()
        self.assertEqual(geoimg.xsize(), 627)
        self.assertEqual(geoimg.ysize(), 603)

    def test1_create(self):
        """ Create single band image """
        fout = 'test.tif'
        geoimg = gp.GeoImage.create(fout, xsz=1000, ysz=1000)
        self.assertTrue(geoimg.xsize() == 1000)
        self.assertTrue(geoimg.ysize() == 1000)
        self.assertTrue(os.path.exists(fout))
        # test resolution
        res = geoimg.resolution()
        self.assertEqual(res.x(), 1.0/geoimg.xsize())
        self.assertEqual(res.y(), -1.0/geoimg.ysize())
        os.remove(fout)

    def test_read(self):
        """ Read multiband image """
        geoimg = gpt.get_test_image()
        arr = geoimg.read()
        self.assertEqual(geoimg.nbands(), arr.shape[0])
        # make sure x, y dimensions are same when reading single bands
        self.assertEqual(arr.shape[1:3], geoimg[0].read().shape)

    def test_read_random_pixels(self):
        """ Read random pixels """
        geoimg = gpt.get_test_image()
        arr = geoimg.read_random_pixels(1000)

    def test_uint16_read(self):
        """ read uint16 makes uint16 array """
        fout = 'test.tif'
        geoimg = gp.GeoImage.create(fout, dtype='uint16')
        self.assertTrue(os.path.exists(fout))
        arr = geoimg.read()
        self.assertEqual(str(arr.dtype), geoimg.type().string())
        os.remove(fout)

    def test_loop_through_bands(self):
        """ Check that GeoImage is iterable """
        geoimg = gpt.get_test_image()
        for band in geoimg:
            self.assertEqual(band.xsize(), geoimg.xsize())

    def test_select(self):
        """ Selection of bands from GeoImage """
        img1 = gpt.get_test_image()
        img2 = img1.select(['red', 'green', 'blue'])
        self.assertTrue(np.array_equal(img1['red'].read(), img2[0].read()))
        self.assertTrue(np.array_equal(img1['green'].read(), img2[1].read()))
        self.assertTrue(np.array_equal(img1['blue'].read(), img2[2].read()))

    def test_persistent_metadata(self):
        """ Writing metadata and check for persistence after reopening """
        fout = 'test-meta.tif'
        geoimg = gp.GeoImage.create(fout, xsz=1000, ysz=1000, nb=3)
        geoimg.set_bandnames(['red', 'green', 'blue'])
        geoimg.set_nodata(7)
        self.assertEqual(geoimg.bandnames()[0], 'red')
        geoimg = None
        # reopen
        geoimg = gp.GeoImage(fout)
        self.assertEqual(geoimg[0].nodata(), 7)
        self.assertEqual(list(geoimg.bandnames()), ['red', 'green', 'blue'])
        geoimg = None
        os.remove(fout)

    def test_create_image_with_gain(self):
        """ Create int image with floating point gain """
        fout = 'test-gain.tif'
        geoimg = gp.GeoImage.create(fout, xsz=1000, ysz=1000, dtype='int16')
        geoimg.set_gain(0.0001)
        arr = np.zeros((1000,1000)) + 0.0001
        arr[0:500,:] = 0.0002
        geoimg[0].write(deepcopy(arr))
        arrout = geoimg[0].read()
        np.testing.assert_array_equal(arr, arrout)
        os.remove(fout)

    def test_create_multiband(self):
        """ Create an RGB image """
        fout = 'test_3band.tif'
        geoimg = gp.GeoImage.create(fout, xsz=1000, ysz=1000, nb=3)
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
        geoimg = gp.GeoImage.create(fout, xsz=1000, ysz=1000, nb=5, temp=True)
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
        geoimg = gp.GeoImage.create(xsz=1000, ysz=1000, nb=3)
        fout = geoimg.filename()
        self.assertTrue(os.path.exists(fout))
        geoimg = None
        self.assertFalse(os.path.exists(fout))

    def test_autoscale(self):
        """ Auto scale each band in image """
        geoimg = gpt.get_test_image()
        for band in geoimg:
            self.assertTrue(band.min() != 1.0)
            self.assertTrue(band.max() != 255.0)
        geoimg2 = geoimg.autoscale(minout=1.0, maxout=255.0)
        for band in geoimg2:
            self.assertTrue(band.min() == 1)
            self.assertTrue(band.max() == 255)

    def test_overviews(self):
        """ Add overviews to an image """
        fout = 'test-overviews.tif'
        geoimg = gp.GeoImage.create(filename=fout, xsz=1000, ysz=1000, nb=2)
        fout = geoimg.filename()
        # add overviews
        geoimg.add_overviews() 
        # clear overviews
        geoimg.add_overviews(levels=[])
        self.assertFalse(os.path.exists(fout + '.ovr'))
        geoimg = None
        geoimg = gp.GeoImage(fout, False)
        geoimg.add_overviews()
        self.assertTrue(os.path.exists(fout + '.ovr'))
        os.remove(fout)
        os.remove(fout + '.ovr')

    def test_save(self):
        """ Save image as new image with different datatype """
        fout = 'test-byte.tif'
        geoimg = gpt.get_test_image().autoscale(1.0, 255.0).save(fout, 'uint8')
        geoimg = None
        geoimg = gp.GeoImage(fout)
        self.assertEqual(geoimg.type().string(), 'uint8')
        self.assertEqual(geoimg[0].min(), 1.0)
        self.assertEqual(geoimg[0].max(), 255.0)
        os.remove(fout)

    def test_save_with_gain(self):
        """ Save image with a gain, which should copy through """
        geoimg = gpt.get_test_image().select([2])
        geoimg.set_gain(0.0001)
        fout = 'test-savegain.tif'
        imgout = geoimg.save(fout)
        assert_array_equal(imgout.read(), geoimg.read())
        os.remove(fout)

    def test_warp(self):
        """ Warping image into another (blank) image """
        bbox = np.array([0.0, 0.0, 1.0, 1.0])
        # default image in EPSG:4326 that spans 1 degree
        geoimg = gp.GeoImage.create(xsz=1000, ysz=1000, nb=3, proj='EPSG:4326', bbox=bbox)
        # 3857, set resolution to 100 meters
        imgout = geoimg.warp(proj='EPSG:3857', xres=100.0, yres=100.0)
        self.assertTrue(os.path.exists(imgout.filename()))
        self.assertEqual(imgout.xsize(), 1114)
        self.assertEqual(imgout.ysize(), 1114)
        self.assertAlmostEqual(np.ceil(imgout.resolution().x()), 100.0)

    def test_real_warp(self):
        """ Warp real image to another projection """
        geoimg = gpt.get_test_image()
        fout = 'test-realwarp.tif'
        imgout = geoimg.warp(fout, proj='EPSG:4326', xres=0.0003, yres=0.0003)
        self.assertEqual(imgout.xsize(), 653)
        self.assertEqual(imgout.ysize(), 547)
        os.remove(fout)

    def test_warp_into(self):
        """ Warp real image into an existing image """
        geoimg = gpt.get_test_image().select([1])
        ext = geoimg.extent()
        bbox = np.array([ext.x0(), ext.y0(), ext.width(), ext.height()])
        imgout = gp.GeoImage.create('', geoimg.xsize(), geoimg.ysize(), 1, geoimg.srs(),
                                    bbox, geoimg.type().string());
        geoimg.warp_into(imgout)
        self.assertEqual(imgout.read().sum(), geoimg.read().sum())

