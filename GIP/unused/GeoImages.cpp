/*##############################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    AUTHOR: Matthew Hanson
#    EMAIL:  matt.a.hanson@gmail.com
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
##############################################################################*/

#include <gip/GeoImages.h>
#include <gip/geometry.h>

namespace gip {
    using std::vector;

    // find transformed union of all raster bounding boxes
    BoundingBox GeoImages::extent(std::string srs) const {
        BoundingBox ext;
        vector< BoundingBox > extents;
        for (vector<GeoImage>::const_iterator i=_GeoImages.begin(); i!=_GeoImages.end(); i++) {
            ext = i->extent();
            ext.transform(i->srs(), srs);
            extents.push_back(ext);
        }
        return union_all<double>(extents);
    }

}
