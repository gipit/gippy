
import os
import glob
import gippy

# TODO - download landsat test image if not already
def get_test_image():
    """ get test image """
    # look in samples directory
    #dirs = [d for d in os.listdir(os.path.join(path, 'samples')) if os.path.isdir(d)]
    bname = os.path.join(os.path.dirname(__file__), 'samples/landsat8/test')
    bands = [4, 5]
    fnames = ['%s_B%s.tif' % (bname, b) for b in bands]
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
