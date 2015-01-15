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
        //! Set projection
        GeoResource& SetProjection(string proj) {
            _GDALDataset->SetProjection(proj.c_str());
            return *this;
        }
        //! Get affine
        CImg<double> Affine() const {
            double affine[6];
            _GDALDataset->GetGeoTransform(affine);
            return CImg<double>(&affine[0], 6);
        }
        //! Set affine
        GeoResource& SetAffine(CImg<double> affine) {
            _GDALDataset->SetGeoTransform(affine.data());
            return *this;
        }

    protected:
        //! Underlying GDALDataset of this file
        boost::shared_ptr<GDALDataset> _GDALDataset;

        GDALMajorObject* GDALObject() const {
            return _GDALDataset.get();
        }

    }; //class GeoData

} // namespace gip

#endif
