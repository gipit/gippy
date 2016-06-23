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

#ifndef GIP_GEOVECTOR_H
#define GIP_GEOVECTOR_H

#include <string>
#include <algorithm>

#include <ogrsf_frmts.h>

#include <gip/geometry.h>
#include <gip/GeoVectorResource.h>
#include <gip/GeoFeature.h>

namespace gip {

    class GeoVector : public GeoVectorResource {
    public:

        //! \name Constructors/Destructors
        //! Default constructor
        GeoVector() 
            : GeoVectorResource() {}
        //! Open existing layer from source
        GeoVector(std::string filename, std::string layer="")
            : GeoVectorResource(filename, layer) {
        }

        //! Copy constructor
        GeoVector(const GeoVector& vector)
            : GeoVectorResource(vector) {
        }
        //! Assignment operator
        GeoVector& operator=(const GeoVector& vector) {
            if (this == &vector) return *this;
            GeoVectorResource::operator=(vector);
            return *this;
        }
        //! Destructor
        ~GeoVector() {
            //if (Options::verbose() > 4) use_counts("destructor");
        }

        void set_primary_key(std::string key="") {
            if (key == "") {
                _PrimaryKey = "";
                return;
            }
            std::vector<std::string> atts = attributes();
            if (std::find(atts.begin(), atts.end(), key) != atts.end()) {
                _PrimaryKey = key;
                // for now, don't check
                /*
                std::vector<std::string> vals = Values(key);
                unsigned int sz(vals.size());
                std::sort(vals.begin(), vals.end());
                vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
                if (sz == vals.size())
                    _PrimaryKey = key;
                else
                    throw std::runtime_error("Attribute " + key + " is not unique.");
                */
            } else
                throw std::out_of_range("No such attribute " + key);
        }

        // Feature Indexing
        //! Get feature (0-based index)
        GeoFeature operator[](unsigned int index) { 
            // Call const version
            return static_cast<const GeoVector&>(*this)[index];
        }
        //! Get feature (0-based index), const version
        const GeoFeature operator[](unsigned int index) const {
            if (index >= nfeatures())
                throw std::out_of_range("No feature " + to_string(index));
            _Layer->ResetReading();
            _Layer->SetNextByIndex(index);
            return GeoFeature(*this, _Layer->GetNextFeature());
        }
        //! Get feature using primary key
        GeoFeature operator[](std::string val) {
            return static_cast<const GeoVector&>(*this)[val];
        }
        //! Get feature using primary key (default to FID)
        const GeoFeature operator[](std::string val) const {
            if (_PrimaryKey == "") 
                return GeoFeature(*this, std::stol(val));
            else {
                std::vector<GeoFeature> query = where(primary_key(), val);
                if (query.size() == 0)
                    throw std::out_of_range("No feature with " + primary_key() + " = " + val);
                return query[0];
            }
        }

        //! Get value of this attribute for all features
        std::vector<std::string> values(std::string attr) {
            std::vector<std::string> vals;
            _Layer->ResetReading();
            GeoFeature f;
            for (unsigned int i=0; i<nfeatures(); i++) {
                f = GeoFeature(*this, _Layer->GetNextFeature());
                vals.push_back(f[attr]);
            }
            return vals;
        }

        //! Get all features whose "attribute" is equal to "val"
        std::vector<GeoFeature> where(std::string attr, std::string val) const {
            std::vector<GeoFeature> matches;
            _Layer->ResetReading();
            GeoFeature f;
            for (unsigned int i=0; i<nfeatures(); i++) {
                f = GeoFeature(*this, _Layer->GetNextFeature());
                if (f[attr] == val)
                    matches.push_back(f);
            }
            return matches;
        }

        std::vector<GeoFeature> where(std::string sql) const {
            std::vector<GeoFeature> matches;
            _Layer->SetAttributeFilter(sql.c_str());
            _Layer->ResetReading();
            OGRFeature* f;
            while ((f = _Layer->GetNextFeature())) {
                matches.push_back(GeoFeature(*this, f));
            }
            return matches;
        }

        //! Calculate intersection between passed in feature and features in this layer
        std::map<std::string, std::string> intersections(GeoFeature feat) {
            // transform passed in feature to native
            OGRSpatialReference* srs = _Layer->GetSpatialRef();
            OGRGeometry* geom = feat.ogr_geometry(srs);
            _Layer->SetSpatialFilter(geom);
            _Layer->ResetReading();
            OGRFeature* f;

            std::map<std::string, std::string> geoms;
            std::vector<std::string> areas;
            OGRGeometry* intersect;
            char* wkt;
            while ((f = _Layer->GetNextFeature())) {
                if (f->GetGeometryRef()->Overlaps(geom)) {
                    intersect = f->GetGeometryRef()->Intersection(geom);
                    intersect->exportToWkt(&wkt);
                    geoms[to_string(f->GetFID())] = wkt;
                    //return std::string(wkt);

                    //std::cout << "Area = " << intersect->Area() << std::endl;
                    //areas.push_back(intersect->exportToWkt());
                    //areas.push_back(intersect->exportToGEOS()->getArea());
                }
            }
            _Layer->SetSpatialFilter(NULL);
            return geoms;
        }

    protected:

    }; // class GeoVector

} // namespace gip

#endif
