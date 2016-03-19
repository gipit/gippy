#!/usr/bin/env python

import os
import gippy
import unittest
import gippy.algorithms as alg

gippy.Options.SetVerbose(3)


class SpeedTests(unittest.TestCase):
    """ Speed tests vs NumPy """

    dirname = '/home/mhanson/landsat/downloads'
    sid = 'LC80080672015244LGN00'

    def read_image(self):
        bname = os.path.join(self.dirname, self.sid, self.sid)
        fnames = [bname + b + '.TIF' for b in ['_B4', '_B5']]
        geoimg = gippy.GeoImage(fnames)
        geoimg.SetBandName("RED", 1)
        geoimg.SetBandName("NIR", 2)
        geoimg.SetNoData(0)
        return geoimg

    def test_numpy_stats(self):
        """ Test statistics speed using numpy """
        geoimg = self.read_image()
        for band in geoimg:
            nodata = band.NoDataValue()
            img = band.Read()
            subimg = img[img != nodata]
            stats = [subimg.min(), subimg.max(), subimg.mean()]
            print stats

    def test_gippy_stats(self):
        """ Test statistics speed using gippy """
        geoimg = self.read_image()
        nodata = geoimg[0].NoDataValue()
        for band in geoimg:
            stats = band.Stats()
            print stats

    def test_numpy_ndvi(self):
        """ Test NDVI using numpy """
        geoimg = self.read_image()
        nodata = geoimg[0].NoDataValue()
        red = geoimg['RED'].Read().astype('double')
        nir = geoimg['NIR'].Read().astype('double')
        ndvi = (nir - red)/(nir + red)
        ndvi[red == nodata] = nodata
        fout = os.path.splitext(geoimg.Filename())[0] + '_numpy_ndvi'
        geoimgout = gippy.GeoImage(fout, geoimg, gippy.GDT_Float64, 1)
        geoimgout[0].Write(ndvi)
        geoimgout = None


    def test_gippy_ndvi(self):
        """ Test NDVI using gippy """
        geoimg = self.read_image()
        gippy.Options.SetChunkSize(16.0)
        fout = os.path.splitext(geoimg.Filename())[0] + '_gippy_ndvi'
        ndvi = alg.Indices(geoimg, {'ndvi': fout})




    # TODO cleanup

''
