/*##############################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    Copyright (C) 2014 Matthew A Hanson
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>
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
