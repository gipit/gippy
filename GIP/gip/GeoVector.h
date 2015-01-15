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

namespace gip {

    class GeoVector {
    public:

        //! \name Constructors/Destructors
        //! Default constructor
        GeoVector() : _OGRDataSource() {}
        //! Open existing source
        GeoVector(std::string);
        //! Create new file on disk
        GeoVector(std::string filename, OGRwkbGeometryType dtype);

        //! Copy constructor
        GeoVector(const GeoVector& vector);
        //! Assignment operator
        GeoVector& operator=(const GeoVector& vector);
        //! Destructor
        ~GeoVector() {}

        //! \name Data Information
        //! Get number of layers
        //int NumLayers() const { return _OGRDataSource.GetLayerCount(); }

    protected:

        //! Filename to dataset
        boost::filesystem::path _Filename;

        //! Underlying OGRDataSource
        boost::shared_ptr<OGRDataSource> _OGRDataSource;

    }; // class GeoVector

} // namespace gip

#endif
