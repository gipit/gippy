#!/usr/bin/env python

import os
import gippy as gp
import unittest
import gippy.test as gpt

"""
    GeoResource is a virtual base class, and thus cannot be instanatiated on it's own.
    It is created with GeoImage or GeoRaster.  This doesn't test GeoResource on it's own,
    rather, it uses GeoImage, but tests only the functions of the base GeoResource class
"""


class GeoResourceTests(unittest.TestCase):

    def setUp(self):
        """ Configure options """
        gp.Options.set_verbose(1)
        gp.Options.set_chunksize(4.0)

    def test_filename(self):
        """ Check filename, basename, extension functions """
        fout = 'test.tif'
        geoimg = gp.GeoImage.create(fout)
        self.assertTrue(os.path.exists(fout))
        self.assertEqual(geoimg.filename(), fout)
        self.assertEqual(geoimg.basename(), 'test')
        self.assertEqual(geoimg.extension(), 'tif')
        os.remove(fout)

    def test_format(self):
        """ Get and set file format and verify """
        dformat = gp.Options.defaultformat()
        gp.Options.set_defaultformat('GTiff')
        geoimg = gp.GeoImage.create()
        self.assertEqual(geoimg.extension(), 'tif')
        # change default format
        gp.Options.set_defaultformat('MEM')
        geoimg = gp.GeoImage.create()
        self.assertEqual(geoimg.extension(), '')
        # create by specifying format to create
        geoimg = gp.GeoImage.create(format='GTiff')
        self.assertEqual(geoimg.extension(), 'tif')
        # change back to original format
        gp.Options.set_defaultformat(dformat)

    def test_size(self):
        """ Retrieve size and dimension in pixels """
        geoimg = gp.GeoImage.create(xsz=500, ysz=1500)
        self.assertEqual(geoimg.xsize(), 500)
        self.assertEqual(geoimg.ysize(), 1500)
        self.assertEqual(geoimg.size(), 1500*500)

    def test_coordinates(self):
        """ Validate coordinates of pixel locations """
        geoimg = gp.GeoImage.create(xsz=1000, ysz=1000)
        pt = geoimg.geoloc(0, 0)
        self.assertEqual(pt.x(), 0.0)
        self.assertEqual(pt.y(), 1.0)
        # this is out of bounds WRT pixels, but is coordinate of lower right corner
        # of the last pixel whereas geoloc(999, 999) is top left corner of last pixel
        pt = geoimg.geoloc(1000, 1000)
        self.assertEqual(pt.x(), 1.0)
        self.assertEqual(pt.y(), 0.0)
        self.assertEqual(geoimg.minxy().x(), 0.0)
        self.assertEqual(geoimg.minxy().x(), 0.0)
        self.assertEqual(geoimg.maxxy().x(), 1.0)
        self.assertEqual(geoimg.maxxy().x(), 1.0)
        # test resolution
        self.assertEqual(geoimg.resolution().x(), 1.0/1000.0)
        self.assertEqual(geoimg.resolution().y(), -1.0/1000.0)
        # test extent
        extent = geoimg.extent()
        self.assertEqual(extent.x0(), 0.0)
        self.assertEqual(extent.y0(), 0.0)
        self.assertEqual(extent.x1(), 1.0)
        self.assertEqual(extent.y1(), 1.0)

    def test_affine(self):
        """ Compare affine vs coordinates and resolution """
        geoimg = gp.GeoImage.create(xsz=100, ysz=100)
        aff = geoimg.affine()
        self.assertEqual(len(aff), 6)
        self.assertEqual(aff[0], geoimg.minxy().x())
        self.assertEqual(aff[1], geoimg.resolution().x())
        self.assertEqual(aff[2], 0.0)
        self.assertEqual(aff[3], geoimg.maxxy().y())
        self.assertEqual(aff[4], 0.0)
        self.assertEqual(aff[5], geoimg.resolution().y())
        # test with real image
        geoimg = gpt.get_test_image()
        aff = geoimg.affine()
        self.assertEqual(len(aff), 6)
        self.assertEqual(aff[0], geoimg.minxy().x())
        self.assertEqual(aff[1], geoimg.resolution().x())
        self.assertEqual(aff[2], 0.0)
        self.assertEqual(aff[3], geoimg.maxxy().y())
        self.assertEqual(aff[4], 0.0)
        self.assertEqual(aff[5], geoimg.resolution().y())

    def test_spatialreference(self):
        """ Set spatial reference """
        geoimg = gp.GeoImage.create(xsz=100, ysz=100)
        prj = geoimg.srs()
        geoimg.set_srs('EPSG:4326')
        # set it to default above, so they should be same
        self.assertEqual(geoimg.srs(), prj)

    def test_chunks(self):
        """ Chunk image to create list of rects """
        geoimg = gp.GeoImage.create(xsz=2000, ysz=2000)
        # test with one chunk
        chunks = geoimg.chunks(numchunks=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].width(), geoimg.xsize())
        self.assertEqual(chunks[0].height(), geoimg.ysize())
        # test with 11 chunks
        chunks = geoimg.chunks(numchunks=11)
        # test height of chunks is the same (except last one)
        self.assertEqual(len(chunks), 11)
        for i in range(1, len(chunks)-1):
            self.assertEqual(chunks[i].height(), chunks[i-1].height())
            self.assertEqual(chunks[i].width(), geoimg.xsize())
            self.assertEqual(chunks[i].y0(), chunks[i-1].y1())
        self.assertNotEqual(chunks[-1].height(), chunks[-2].height())
        # test with 10 chunks
        chunks = geoimg.chunks(numchunks=10)
        # test height of chunks is the same (including last one)
        self.assertEqual(len(chunks), 10)
        for i in range(1, len(chunks)):
            self.assertEqual(chunks[i].height(), chunks[i-1].height())
            self.assertEqual(chunks[i].width(), geoimg.xsize())
            self.assertEqual(chunks[i].y0(), chunks[i-1].y1())


    def test_meta(self):
        """ Set and retrieve metadata """
        geoimg = gp.GeoImage.create(xsz=1000, ysz=1000)
        meta = geoimg.meta()
        self.assertEqual(len(meta), 0)
        # set single metadata item
        geoimg.add_meta('testkey', 'testvalue')
        self.assertEqual(geoimg.meta('testkey'), 'testvalue')
        md = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
        geoimg.add_meta(md)
        md['testkey'] = 'testvalue'
        md2 = geoimg.meta()
        for m in md:
            self.assertEqual(geoimg.meta(m), md[m])
            self.assertEqual(md2[m], md[m])
