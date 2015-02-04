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

#ifndef GIP_GEOVECTORRESOURCE_H
#define GIP_GEOVECTORRESOURCE_H

#include <string>

#include <gdal/ogrsf_frmts.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

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

    protected:

        //! Filename to dataset
        boost::filesystem::path _Filename;

        //! Underlying OGRDataSource
        boost::shared_ptr<OGRDataSource> _OGRDataSource;

        // OGRLayer - ptr linked to dataset
        OGRLayer* _Layer;

    private:
        void OpenLayer(std::string layer="");

    }; // class GeoVectorResource

} // namespace gip

#endif
