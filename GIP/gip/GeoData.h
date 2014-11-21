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

#ifndef GIP_GEODATA_H
#define GIP_GEODATA_H

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <gip/GeoResource.h>
#include <gdal/gdal_priv.h>
#include <gip/gip_CImg.h>
#include <gip/Utils.h>
#include <gip/geometry.h>

namespace gip {

    class GeoData : public GeoResource {
    public:

        //! \name Constructors/Destructor
        //! Default constructor
        GeoData() : GeoResource(), _GDALDataset() {}
        //! Open existing file
        GeoData(string, bool=false);
        //! Create new file on disk
        GeoData(int, int, int, GDALDataType, string, dictionary = dictionary());
        //! Copy constructor
        GeoData(const GeoData&);
        //! Assignment copy
        GeoData& operator=(const GeoData&);
        //! Destructor
        ~GeoData();

        //! \name File Information
        //! File format of dataset
        std::string Format() const { return _GDALDataset->GetDriver()->GetDescription(); }
        //! Return data type
        virtual GDALDataType DataType() const { return GDT_Unknown; }
        //! Return size of data type (in bytes)
        //int DataTypeSize() const;

        //! Get GDALDataset object - use cautiously
        GDALDataset* GetGDALDataset() const { return _GDALDataset.get(); }

        //! \name Spatial Information
        //! X Size of image/raster, in pixels
        unsigned int XSize() const { return _GDALDataset->GetRasterXSize(); }
        //! Y Size of image/raster, in pixels
        unsigned int YSize() const { return _GDALDataset->GetRasterYSize(); }

        //! Return projection definition in Well Known Text format
        string Projection() const {
            return _GDALDataset->GetProjectionRef();
        }
        CImg<double> Affine() const {
            double affine[6];
            _GDALDataset->GetGeoTransform(affine);
            return CImg<double>(&affine[0], 6);
        }

        //! Copy coordinate system
        GeoData& CopyCoordinateSystem(const GeoData&);

    protected:
        //! Underlying GDALDataset of this file
        boost::shared_ptr<GDALDataset> _GDALDataset;

    }; //class GeoData

} // namespace gip

#endif
