#!/usr/bin/env python
################################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    Copyright (C) 2015 Applied Geosolutions
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
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