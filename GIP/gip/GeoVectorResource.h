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

#ifndef GIP_GEOVECTORRESOURCE_H
#define GIP_GEOVECTORRESOURCE_H

#include <memory>
#include <string>

#include <ogrsf_frmts.h>

#include <gip/geometry.h>

namespace gip {

    class GeoVectorResource {
    public:

        //! \name Constructors/Destructors
        //! Default constructor
        GeoVectorResource() : _OGRDataSource() {}
        //! Open existing layer from source
        GeoVectorResource(std::string, std::string layer="");
        //! Create new file vector layer
        //GeoVector(std::string filename, OGRwkbGeometryType dtype);

        //! Copy constructor
        GeoVectorResource(const GeoVectorResource& vector);
        //! Assignment operator
        GeoVectorResource& operator=(const GeoVectorResource& vector);
        //! Destructor
        ~GeoVectorResource();

        //! \name Resource information
        std::string filename() const;
        //! Basename, or short name of filename
        std::string basename() const;
        //! File format of dataset
        //std::string Format() const;
        std::string layer_name() const;

        // Geospatial
        //! Return spatial reference system as an OGRSpatialReference
        std::string srs() const;

        //! Return spatial reference system as WKT
        //std::string Projection() const;

        //! Get bounding box in projected units
        BoundingBox extent() const;

        //! Get number of features
        unsigned long int nfeatures() const {
            return _Layer->GetFeatureCount();
        }

        std::string primary_key() const {
            return _PrimaryKey;
        }

        //! Get list of attributes
        std::vector<std::string> attributes() const {
            std::vector<std::string> atts;
            OGRFeatureDefn* att = _Layer->GetLayerDefn();
            for (int i=0; i<att->GetFieldCount(); i++) {
                atts.push_back(std::string(att->GetFieldDefn(i)->GetNameRef()));
            }
            return atts;
        }

    protected:

        //! Filename to dataset
        std::string _Filename;

        //! Underlying OGRDataSource
        std::shared_ptr<OGRDataSource> _OGRDataSource;

        // OGRLayer - ptr linked to dataset
        OGRLayer* _Layer;

        std::string _PrimaryKey;

        void use_count(std::string s = "") const {
            std::cout << basename() << " GeoVectorResource " << s 
                << " use_count = " << _OGRDataSource.use_count() << std::endl;
        }

    private:
        void OpenLayer(std::string layer="");

    }; // class GeoVectorResource

} // namespace gip

#endif
