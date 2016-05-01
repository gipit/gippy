Algorithms
++++++++++

Gippy includes an algorithm module, which is a collection of functions that operate on GeoImage objects. There are currently only a handful of functions, but this will be expanded upon in future versions. Gippy algorithms take in one or more images, parameters unique to the algorithm, and an output filename and output file options. Currently the algorithms module includes indices() which can calculate a variety of indices (e.g., NDVI, NDWI, LSWI, SATVI) in one pass, acca() and fmask() cloud detection algorithms for Landsat7, cookie_cutter() for mosaicking together scenes, linear_transform() for applying basis vectors to spectral data, pansharpening, and a rxd(), a multispectral anomaly detection algorithm. 

.. code::

    from gippy.test import get_test_image
    import gippy.algorithms as alg

    geoimg = get_test_image()

    index_image = alg.indices(geoimg, products=['ndvi', 'ndwi'])

    print(index_image.bandnames())

    > ['ndvi', 'ndwi']

acca(geoimg, filename)
    The ACCA algorithm operates on a GeoImage of Landsat7 data and must contains bands for Red, Green, NIR, SWIR1, and LWIR. Because it utilizes empirically derived constants it is currently only suitable for Landsat7.

fmask(geoimg, filename)
    Like ACCA, Fmask is currently only suitable for Landsat7 data. It requires bands for Blue, Red, Gree, NIR, SWIR1, SWIR2, and LWIR.

cookie_cutter([geoimgs], filename, feature, crop, proj, xres, yres, interpolation)
    The cookie_cutter algorithm takes in a list of GeoImage objects, and optionally a feature representing the region of interest. A mosaic will be created from all input images and cutting to the feature. If the crop keyword is set to True, the extent of the output will be the intesection of the feature and all the images, otherwise, it will be the extent of the feature.

indices(geoimg, products, filename)
    This calculates the desired indices given by the products list, and creates a single file with one index for each band. Indices currently supported: NDVI, EVI, LSWI, NDSI, NDWI, SATVI, MSAVI2.

linear_transform(geoimg, coef, filename)
    This applies the matrix of coefficients to create a new series of bands. The coef matrix must be of size #bands x #bands and will create the band outputs that are linear combinations of the inputs. Useful for transforming images into principal components by providing the Eigenvectors.

pansharpen(geoimg, panimg, weights, filename)
    This performs a Brovey pansharpening algorithm on the input image. First geoimg is upsampled and panimg shifted so that they both cover the same footprint. Pansharpening then uses the panimg values, and the weights (if provided) to get the pansharpened multispectral input. The input image must contain 3 or 4 bands, and if provided weights must be the same length.

rxd(geoimg, filename)
    The RX Detector algorithm is a simple anomaoly detector that reduces a multiband image into a single band that indicates how the spectral signature of pixels deviates from an average.    




