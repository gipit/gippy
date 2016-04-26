import gippy
from stestdata import TestData


# TODO - download landsat test image if not already
def get_test_image(sensor='landsat8', name='', bands=[]):
    """ Get test image from sat-testdata """
    t = TestData(sensor)

    if name == '':
        name = t.names[0]

    dat = t.examples[name].values()
    if len(bands) > 0:
        dat = [d for d in dat if d['band_type'] in bands]
    else:
        # filter out pan band
        dat = [d for d in dat if d['band_type'] != 'pan']

    filenames = [d['path'] for d in dat]
    bandnames = [d['band_type'] for d in dat]

    geoimg = gippy.GeoImage(filenames)
    geoimg.set_bandnames(bandnames)
    geoimg.set_nodata(0)
    return geoimg


"""
#arr = np.array([[1, 2], [3, 4], [5, 6]])
arr = np.array([1, 2, 3, 4, 5, 6])

VerboseOut('Original Array')
print arr, arr.dtype, '\n'

VerboseOut('Array w/o casting')
outarr = gippy.test(arr)
print outarr, outarr.dtype, '\n'

dtypes = ['uint8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'float32', 'float64']
for dt in dtypes:
    VerboseOut('Array as %s' % dt)
    outarr = gippy.test(arr.astype(dt))
    print outarr, outarr.dtype, '\n'
"""
