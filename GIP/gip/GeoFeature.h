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
        /*explicit GeoFeature(const GeoVectorResource& vector, boost::shared_ptr<OGRFeature> feature) 
            : GeoVectorResource(vector) {
            _Feature = feature;
            if (Options::Verbose() > 4) use_counts("constructor");
        }*/
        //! Open feature constructor
        explicit GeoFeature(const GeoVectorResource& vector, int index)
            : GeoVectorResource(vector) {
            //if (!_Layer.TestCapability(OLCFastSetNextByIndex))
            //    std::cout << "using slow method of accessing feature" << std::endl;
            // Is this a race condition ?
            if (index == 0)
                _Layer->ResetReading();
            else
                _Layer->SetNextByIndex(index-1);
            _Feature.reset(_Layer->GetNextFeature(), OGRFeature::DestroyFeature);
        }
        //! Copy constructor
        GeoFeature(const GeoFeature& feature) 
            : GeoVectorResource(feature), _Feature(feature._Feature) {
            //if (Options::Verbose() > 4) use_count("copy constructor");
        }
        //! Assignment operator
        GeoFeature& operator=(const GeoFeature& feature) {
            if (this == &feature) return *this;
            GeoVectorResource::operator=(feature);
            _Feature = feature._Feature;
            //if (Options::Verbose() > 4) use_count("assignment");
            return *this;
        }
        ~GeoFeature() {
            //if (Options::Verbose() > 4) use_count("destructor");
        }

        //! \name Geospatial information
        //Rect<double> Extent() const {}

        OGRGeometry* Geometry() const {
            return _Feature->GetGeometryRef();
        }

        // output operator
        //void print() const {
        //    _Feature->DumpReadable(NULL);
        //}

        void use_count(std::string s="") const {
            GeoVectorResource::use_count(s);
            std::cout << "\tFeature use_count: " << _Feature.use_count() << std::endl;
        }

    protected:
        boost::shared_ptr<OGRFeature> _Feature;

    }; // class GeoFeature
} // namespace gip

#endif
