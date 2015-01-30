/*##############################################################################
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
##############################################################################*/

#ifndef GIP_GEOFEATURE_H
#define GIP_GEOFEATURE_H

#include <gdal/ogrsf_frmts.h>
#include <gdal/ogr_feature.h>

namespace gip {

    class GeoFeature {
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoFeature() {}
        //! New feature constructor
        explicit GeoFeature(OGRFeature* feature) {
            _OGRFeature = feature;
        }
        ~GeoFeature() {
            OGRFeature::DestroyFeature(_OGRFeature);
        }

        OGRGeometry* Geometry() const {
            return _OGRFeature->GetGeometryRef();
        }

    protected:
        OGRFeature* _OGRFeature;

    }; // class GeoFeature
} // namespace gip

#endif