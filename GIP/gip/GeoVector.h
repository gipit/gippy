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
        //! Create new file vector layer
        //GeoVector(std::string filename, OGRwkbGeometryType dtype);

        //! Copy constructor
        GeoVector(const GeoVector& vector)
            : GeoVectorResource(vector) {
            //_Features = vector._Features;
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


        // Features
        //! Get feature (0-based index)
        GeoFeature operator[](unsigned int index) { 
            // Call const version
            return static_cast<const GeoVector&>(*this)[index];
        }
        //GeoFeature operator[](int index) { return GeoFeature(*this, _Features[index]); }
        //! Get feature, const version
        const GeoFeature operator[](unsigned int index) const {
            if (index < size())
                return GeoFeature(*this, index);
            throw std::out_of_range("No feature " + to_string(index));
        }
        //const GeoFeature operator[](int index) const { return GeoFeature(*this, _Features[index]); }

        //! Combine into single geometry - Should be freed with OGRGeometryFactory::destroyGeometry() after use.
        /*OGRGeometry* Union() const {
            OGRGeometry* site = OGRGeometryFactory::createGeometry( wkbMultiPolygon );
            return site;
        }*/

        std::vector<GeoFeature> where(std::string attr, std::string val) {
            std::vector<GeoFeature> matches;
            GeoFeature f;
            for (unsigned int i=0; i<size(); i++) {
                f = GeoFeature(*this, i);
                if (f[attr] == val)
                    matches.push_back(f);
            }
            return matches;
        }

        void use_counts(std::string s="") const {
            std::cout << Basename() << " GeoVector " << s << " use_counts" << std::endl;
            //for (unsigned int i=0; i<_Features.size(); i++)
            //   std::cout << "\tFeature " << i << ": use_count = " << _Features[i].use_count() << std::endl;
        }

    protected:
        // OGRFeature
        //std::vector<GeoFeature> _Features;
        //std::vector< boost::shared_ptr<OGRFeature> > _Features;

    }; // class GeoVector

} // namespace gip

#endif
