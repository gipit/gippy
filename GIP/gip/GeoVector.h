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

#ifndef GIP_GEOVECTOR_H
#define GIP_GEOVECTOR_H

#include <string>

#include <gdal/ogrsf_frmts.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <gip/geometry.h>
#include <gip/GeoFeature.h>

namespace gip {

    class GeoVector {
    public:

        //! \name Constructors/Destructors
        //! Default constructor
        GeoVector() : _OGRDataSource() {}
        //! Open existing layer from source
        GeoVector(std::string, std::string layer="");
        //! Create new file vector layer
        //GeoVector(std::string filename, OGRwkbGeometryType dtype);

        //! Copy constructor
        GeoVector(const GeoVector& vector);
        //! Assignment operator
        GeoVector& operator=(const GeoVector& vector);
        //! Destructor
        ~GeoVector();

        //! \name Resource information
        std::string Filename() const;
        //! Get path (boost filesystem path)
        boost::filesystem::path Path() const;
        //! Basename, or short name of filename
        std::string Basename() const;
        //! File format of dataset
        //std::string Format() const;

        // Geospatial
        OGRSpatialReference SRS() const;

        std::string Projection() const;

        Rect<double> Extent() const;

        // Features
        //! Get feature (0-based index)
        GeoFeature& operator[](int index) { return _Features[index]; }
        //! Get feature, const version
        const GeoFeature& operator[](int index) const { return _Features[index]; }

        //! Combine into single geometry - Should be freed with OGRGeometryFactory::destroyGeometry() after use.
        OGRGeometry* Union() const {
            OGRGeometry* site = OGRGeometryFactory::createGeometry( wkbMultiPolygon );

            return site;
        }

    protected:

        //! Filename to dataset
        boost::filesystem::path _Filename;

        //! Underlying OGRDataSource
        boost::shared_ptr<OGRDataSource> _OGRDataSource;

        // OGRLayer
        OGRLayer* _OGRLayer;

        //! OGRFeature
        std::vector< GeoFeature > _Features;

    private:
        void OpenLayer(std::string layer="");

    }; // class GeoVector

} // namespace gip

#endif
