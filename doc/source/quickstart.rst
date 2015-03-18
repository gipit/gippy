# Quickstart

The two main classes in GIPPY are GeoImage and GeoRaster.  A GeoRaster is a single raster band, analagous to GDALRasterBand.  A GeoImage is a collection of GeoRaster objects, similar to GDALDataset however the GeoRaster objects that it contains could be from different locations (different files).

Open existing images

    from gippy import GeoImage

    # Open up image read-only
    image = GeoImage('test.tif')

    # Open up image with write permissions
    image = GeoImage('test.tif', True)

    # Open up multiple files as a single image where numbands = numfiles x numbands
    image = GeoImage(['test1.tif', 'test2.tif', 'test3.tif'])

Creating new images

    import gippy

    # Create new 1000x1000 single-band byte image 
    image = gippy.GeoImage('test.tif', 1000, 1000, 1, gippy.GDT_Byte)

    # Create new image with same properties (size, metadata, SRS) as existing gimage GeoImage
    image = gippy.GeoImage('test.tif', gimage)

    # As above but with different datatype and 4 bands
    image = gippy.GeoImage('test.tif', gimage, gippy.GDT_Int16, 4)