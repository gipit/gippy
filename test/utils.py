
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
    return geoimg
