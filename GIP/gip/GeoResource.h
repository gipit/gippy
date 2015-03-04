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

#ifndef GIP_GEORESOURCE_H
#define GIP_GEORESOURCE_H

#include <string>
#include <vector>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

#include <gdal/gdal_priv.h>
#include <gip/gip_CImg.h>
#include <gip/geometry.h>

namespace gip {
    typedef std::map<std::string,std::string> dictionary;

    //! Base class representing a geospatial resource
    class GeoResource {
    public:
        //! \name Constructors
        //! Default constructor
        GeoResource() : _GDALDataset() {}
        //! Open existing file constructor
        GeoResource(std::string filename, bool update=false);
        //! Create new file - TODO how specify OGRLayer
        GeoResource(int, int, int, GDALDataType, std::string, dictionary = dictionary());

        //! Copy constructor
        GeoResource(const GeoResource& resource);
        //! Assignment copy
        GeoResource& operator=(const GeoResource&);
        //! Destructor
        ~GeoResource();

        //! \name Resource Information
        //! Get the filename of the resource
        std::string Filename() const;
        //! Get path (boost filesystem path)
        boost::filesystem::path Path() const;
        //! Basename, or short name of filename
        std::string Basename() const;
        //! File format of dataset
        std::string Format() const { return _GDALDataset->GetDriver()->GetDescription(); }

        //! \name Geospatial information
        //! X Size of resource, in pixels
        unsigned int XSize() const { return _GDALDataset->GetRasterXSize(); }
        //! Y Size of resource, in pixels
        unsigned int YSize() const { return _GDALDataset->GetRasterYSize(); }
        //! Total size
        unsigned long Size() const { return XSize() * YSize(); }
        //! Geolocated coordinates of a point within the resource
        Point<double> GeoLoc(float xloc, float yloc) const;
        //! Coordinates of top left
        Point<double> TopLeft() const;
        //! Coordinates of lower left
        Point<double> LowerLeft() const;
        //! Coordinates of top right
        Point<double> TopRight() const;
        //! Coordinates of bottom right
        Point<double> LowerRight() const;
        //! Minimum Coordinates of X and Y
        Point<double> MinXY() const;
        //! Maximum Coordinates of X and Y
        Point<double> MaxXY() const;
        //! Extent
        Rect<double> Extent() const { return Rect<double>(LowerLeft(), TopRight()); }
        //! Return projection definition in Well Known Text format
        std::string Projection() const {
            return _GDALDataset->GetProjectionRef();
        }
        //! Set projection definition in Well Known Text format
        GeoResource& SetProjection(std::string proj) {
            _GDALDataset->SetProjection(proj.c_str());
            return *this;
        }
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const;
        //! Get Affine transformation
        CImg<double> Affine() const {
            double affine[6];
            _GDALDataset->GetGeoTransform(affine);
            return CImg<double>(&affine[0], 6);
        }
        //! Set Affine transformation
        GeoResource& SetAffine(CImg<double> affine) {
            _GDALDataset->SetGeoTransform(affine.data());
            return *this;
        }
        GeoResource& SetAffine(double affine[6]) {
            _GDALDataset->SetGeoTransform(affine);
            return *this;
        }
        GeoResource& SetGCPs(CImg<double> gcps, std::string projection) {
            int numgcps(gcps.height());
            GDAL_GCP gdal_gcps[numgcps];
            GDALInitGCPs(numgcps, &gdal_gcps[0]);
            for (int i=0; i<numgcps; i++) {
                //gdal_gcps[i].pszId = &to_string(i)[0];
                gdal_gcps[i].dfGCPPixel = gcps(0, i);
                gdal_gcps[i].dfGCPLine = gcps(1, i);
                gdal_gcps[i].dfGCPX = gcps(2, i);
                gdal_gcps[i].dfGCPY = gcps(3, i);
            }
            _GDALDataset->SetGCPs(numgcps, &gdal_gcps[0], projection.c_str());
            return *this;
        }

        //! Get resolution convenience function
        Point<double> Resolution() const;
        //! Set coordinate system from another GeoResource
        GeoResource& SetCoordinateSystem(const GeoResource& res);


        //! Get chunkset chunking up image
        ChunkSet Chunks(unsigned int padding=0, unsigned int numchunks=0) const;

        //! \name Metadata functions
        //! Get metadata item
        std::string Meta(std::string key) const;
        // Get group of metadata
        std::vector<std::string> MetaGroup(std::string group, std::string filter="") const;
        //! Set metadata item
        GeoResource& SetMeta(std::string key, std::string item);
        //! Set multiple metadata items
        GeoResource& SetMeta(dictionary items);
        //! Copy Meta data from another resource
        GeoResource& CopyMeta(const GeoResource& img);

        //! Retrieve the GDALdataset this is a member of
        GDALDataset* GetGDALDataset() const {
            return _GDALDataset.get();
        }

    protected:
        //! Filename, or some other resource identifier
        boost::filesystem::path _Filename;

        //! Underlying GDALDataset of this file
        boost::shared_ptr<GDALDataset> _GDALDataset;

        //! Retrieve the GDALMajorObject from (GDALDataset, GDALRasterBand, OGRLayer)
        GDALMajorObject* GetGDALObject() const {
            return _GDALDataset.get();
        }

    }; // class GeoResource
} // namespace gip

#endif
