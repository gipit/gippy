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

#ifndef GIP_GEOFEATURE_H
#define GIP_GEOFEATURE_H

#include <memory>

#include <ogrsf_frmts.h>
#include <ogr_feature.h>
#include <cpl_error.h>
#include <ogr_api.h>

#include <gip/GeoVectorResource.h>

namespace gip {

    class GeoFeature : public GeoVectorResource {
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoFeature() 
            : GeoVectorResource(), _Feature() {}
        //! New feature constructor
        /*explicit GeoFeature(const GeoVectorResource& vector, std::shared_ptr<OGRFeature> feature) 
            : GeoVectorResource(vector) {
            _Feature = feature;
            if (Options::verbose() > 4) use_counts("constructor");
        }*/
        //! Constructor to open specific feature in a vector
        explicit GeoFeature(std::string filename, std::string layer, long int fid)
            : GeoVectorResource(filename, layer) {
            OpenFeature(fid);
        }
        //! Open feature constructor
        // TODO - is this used?
        explicit GeoFeature(const GeoVectorResource& vector, long int fid)
            : GeoVectorResource(vector) {
            OpenFeature(fid);
        }
        //! Open feature constructor
        explicit GeoFeature(const GeoVectorResource& vector, OGRFeature* feature)
            : GeoVectorResource(vector) {
            _Feature.reset(feature, OGRFeature::DestroyFeature);
        }
        //! Copy constructor
        GeoFeature(const GeoFeature& feature) 
            : GeoVectorResource(feature), _Feature(feature._Feature) {
            //if (Options::verbose() > 4) use_count("copy constructor");
        }
        //! Assignment operator
        GeoFeature& operator=(const GeoFeature& feature) {
            if (this == &feature) return *this;
            GeoVectorResource::operator=(feature);
            _Feature = feature._Feature;
            //if (Options::verbose() > 4) use_count("assignment");
            return *this;
        }
        ~GeoFeature() {
            //if (Options::verbose() > 4) use_count("destructor");
        }

        bool valid() const {
            return _Feature ? true : false;
        }

        //! Get value for the PrimaryKey
        std::string value() const {
            if (_PrimaryKey == "")
                return to_string(fid());
            else
                return (*this)[_PrimaryKey];
        }

        //! Return basename made up of layer name and feature name
        std::string basename() const {
            return layer_name() + "-" + value();
        }

        //! \name Geospatial information
        BoundingBox extent() const {
            OGREnvelope ext;
            _Feature->GetGeometryRef()->getEnvelope(&ext);
            return BoundingBox(
                Point<double>(ext.MinX, ext.MinY),
                Point<double>(ext.MaxX, ext.MaxY)
            );
        }

        //! Get geometry in Well Known Text format
        std::string geometry() const {
            return wkt_geometry();
        }

        //! Get geometry in Well Known Text format
        std::string wkt_geometry() const {
            char* wkt(NULL);
            ogr_geometry()->exportToWkt(&wkt);
            return std::string(wkt);
        }

        //! Get geometry as JSON
        std::string json_geometry() const {
            //char* json(NULL);
            return ogr_geometry()->exportToJson();
            //return std::string(json);
        }

        //! Get feature as geometry, warp to SRS if provided
        OGRGeometry* ogr_geometry(OGRSpatialReference* srs = NULL) const {
            OGRGeometry* geom = _Feature->GetGeometryRef();
            if (srs != NULL) {
                geom->transformTo(srs);
            }
            return geom;
        }

        long int fid() const {
            return _Feature->GetFID();
        }

        //! Get attribute by name
        std::string operator[](std::string name) {
            return static_cast<const GeoFeature&>(*this)[name];
        }
        //! Get attribute by name
        std::string operator[](std::string name) const {
            // Find index of field
            OGRFeatureDefn* atts = _Layer->GetLayerDefn();
            int i = atts->GetFieldIndex(name.c_str());
            if (i == -1) {
                throw std::out_of_range("No such attribute " + name);
            }
            return _Feature->GetFieldAsString(i);
        }

    protected:
        std::shared_ptr<OGRFeature> _Feature;

        void use_count(std::string s="") const {
            GeoVectorResource::use_count(s);
            std::cout << "\tFeature use_count: " << _Feature.use_count() << std::endl;
        }

    private:
        void OpenFeature(long int fid) {
            //if (!_Layer.TestCapability(OLCFastSetNextByIndex))
            //    std::cout << "using slow method of accessing feature" << std::endl;
            _Feature.reset(_Layer->GetFeature(fid), OGRFeature::DestroyFeature);
            if (CPLGetLastErrorNo() > 0) {
                throw std::out_of_range (CPLGetLastErrorMsg()); 
            }
        }

    }; // class GeoFeature
} // namespace gip

#endif
