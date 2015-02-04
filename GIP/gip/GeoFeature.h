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
#include <boost/shared_ptr.hpp>

#include <gip/GeoVectorResource.h>

namespace gip {

    class GeoFeature : public GeoVectorResource {
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoFeature() 
            : GeoVectorResource(), _Feature() {}
        //! New feature constructor
        explicit GeoFeature(const GeoVectorResource& vector, boost::shared_ptr<OGRFeature> feature) 
            : GeoVectorResource(vector) {
            _Feature = feature; //.reset(feature, OGRFeature::DestroyFeature);
        }
        //! Copy constructor
        GeoFeature(const GeoFeature& feature) 
            : GeoVectorResource(feature), _Feature(feature._Feature) {}
        //! Assignment operator
        GeoFeature& operator=(const GeoFeature& feature) {
            if (this == &feature) return *this;
            GeoVectorResource::operator=(feature);
            _Feature = feature._Feature;
            return *this;
        }
        ~GeoFeature() {
            if (Options::Verbose() > 4) {
                std::cout << "~GeoFeature (use_count = " << _Feature.use_count() << ")" << std::endl;
            }
        }

        OGRGeometry* Geometry() const {
            return _Feature->GetGeometryRef();
        }

        // output operator
        void print() const {
            _Feature->DumpReadable(NULL);
        }

    protected:
        boost::shared_ptr<OGRFeature> _Feature;

    }; // class GeoFeature
} // namespace gip

#endif
