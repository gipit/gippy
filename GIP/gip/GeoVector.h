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

#include <gdal/ogrsf_frmts.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

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
            //if (Options::Verbose() > 4) use_counts("destructor");
        }

        void SetPrimaryKey(std::string key="") {
            if (key == "") {
                _PrimaryKey = "";
                return;
            }
            std::vector<std::string> atts = Attributes();
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
            if (index >= size())
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
                std::vector<GeoFeature> query = where(PrimaryKey(), val);
                if (query.size() == 0)
                    throw std::out_of_range("No feature with " + PrimaryKey() + " = " + val);
                return query[0];
            }
        }

        //! Get value of this attribute for all features
        std::vector<std::string> Values(std::string attr) {
            std::vector<std::string> vals;
            _Layer->ResetReading();
            GeoFeature f;
            for (unsigned int i=0; i<size(); i++) {
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
            for (unsigned int i=0; i<size(); i++) {
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

    protected:

    }; // class GeoVector

} // namespace gip

#endif
