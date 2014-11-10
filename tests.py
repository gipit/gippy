################################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    Copyright (C) 2014 Matthew A Hanson
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>
################################################################################


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