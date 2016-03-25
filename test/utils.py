
import os
import gippy

# TODO - download landsat test image if not already
def get_test_image():
    """ get test image """
    sid = 'LC80080672015244LGN00'
    path = os.path.dirname(__file__)
    fname = os.path.join(path, sid, sid)
    bands = [4, 5]
    fnames = ['%s_B%s.TIF' % (fname, b) for b in bands]
    geoimg = gippy.GeoImage(fnames)
    geoimg.SetBandName('RED', 1)
    geoimg.SetBandName('NIR', 2)
    geoimg.SetNoData(0)
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