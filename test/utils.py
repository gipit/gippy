import gippy
from stestdata import TestData


# TODO - download landsat test image if not already
def get_test_image():
    """ get test image """
    t = TestData('landsat8')

    fnames = [None, None]
    for k, v in t.examples[t.names[0]].iteritems():
        if v['band_type'] == 'red':
            fnames[0] = v['path']

        if v['band_type'] == 'nir':
            fnames[1] = v['path']

    geoimg = gippy.GeoImage(fnames)
    geoimg.set_bandname('RED', 1)
    geoimg.set_bandname('NIR', 2)
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
