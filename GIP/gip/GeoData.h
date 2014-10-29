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
#include <map>
#include <gdal/gdal_priv.h>
#include <boost/shared_ptr.hpp>

#include <boost/filesystem.hpp>

#include <gip/Utils.h>

#include <gip/geometry.h>

namespace gip {
    typedef std::map<std::string,std::string> dictionary;
    typedef Rect<int> iRect;
    typedef Point<int> iPoint;

    class GeoData {
    public:

        //! \name Constructors/Destructor
        //! Default constructor
        GeoData() : _GDALDataset(), _padding(0) {}
        //! Open existing file
        GeoData(std::string, bool=false);
        //! Create new file on disk
        GeoData(int, int, int, GDALDataType, std::string, dictionary = dictionary());
        //! Copy constructor
        GeoData(const GeoData&);
        //! Assignment copy
        GeoData& operator=(const GeoData&);
        //! Destructor
        ~GeoData();

        //! \name File Information
        //! Boost filesystem path
        boost::filesystem::path Path() const { return _Filename; }
        //! Full filename of dataset
        std::string Filename() const { return _Filename.string(); }
        //! Filename without path
        std::string Basename() const { return _Filename.stem().string(); }
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
        //! Total number of pixels
        unsigned long Size() const { return XSize() * YSize(); }
        //! Geolocated coordinates of a pixel
        Point<double> GeoLoc(float xloc, float yloc) const;
        //! Coordinates of top left
        Point<double> TopLeft() const { return GeoLoc(0,0); }
        //! Coordinates of lower left
        Point<double> LowerLeft() const { return GeoLoc(0,YSize()-1); }
        //! Coordinates of top right
        Point<double> TopRight() const { return GeoLoc(XSize()-1,0); }
        //! Coordinates of bottom right
        Point<double> LowerRight() const { return GeoLoc(XSize()-1,YSize()-1); }
        //! Minimum Coordinates of X and Y
        Point<double> MinXY() const { 
            Point<double> pt1(TopLeft()), pt2(LowerRight());
            double MinX(std::min(pt1.x(), pt2.x()));
            double MinY(std::min(pt1.y(), pt2.y()));
            return Point<double>(MinX, MinY);
        }
        //! Maximum Coordinates of X and Y
        Point<double> MaxXY() const { 
            Point<double> pt1(TopLeft()), pt2(LowerRight());
            double MaxX(std::max(pt1.x(), pt2.x()));
            double MaxY(std::max(pt1.y(), pt2.y()));
            return Point<double>(MaxX, MaxY);
        }
        //! Return projection definition in Well Known Text format
        std::string Projection() const {
            return _GDALDataset->GetProjectionRef();
        }
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const {
            std::string s(Projection());
            return OGRSpatialReference(s.c_str());
        }

        //! \name Metadata functions
        //! Get metadata item
        std::string GetMeta(std::string key) const {
            const char* item = _GDALDataset->GetMetadataItem(key.c_str());
            if (item == NULL) return ""; else return item;
        }
        //! Get group of metadata
        std::vector<std::string> GetMetaGroup(std::string group,std::string filter="") const;
        //! Set metadata item
        GeoData& SetMeta(std::string key, std::string item) {
            _GDALDataset->SetMetadataItem(key.c_str(),item.c_str());
            return *this;
        }
        //! Set multiple metadata items
        GeoData& SetMeta(std::map<std::string, std::string> items) {
            for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
                SetMeta(i->first, i->second);
            }
            return *this;
        }
        //! Copy Meta data from input file.  Currently no error checking
        GeoData& CopyMeta(const GeoData& img);
        //! Copy coordinate system
        GeoData& CopyCoordinateSystem(const GeoData&);


        //! \name Chunking functions
        //! Break up image into chunks
        std::vector< Rect<int> > Chunk(unsigned int = 0) const;

        //! Get the number of chunks used for processing image
        unsigned int NumChunks() const { return _Chunks.size(); }

        //! Retrieve chunks as vector of rects
        std::vector< Rect<int> > Chunks() const { return _Chunks; }

        //! Retreive padded chunks as vector of rects
        std::vector< Rect<int> > PaddedChunks() const { return _PadChunks; }

        //! Get padding for chunks
        unsigned int ChunkPadding() const { return _padding; }

        //! Set padding for chunks
        void SetChunkPadding(unsigned int pad = 0) const { 
            _padding = pad;
            Chunk(NumChunks());
        }

    protected:
        //! Filename to dataset
        boost::filesystem::path _Filename;
        //! Underlying GDALDataset of this file
        boost::shared_ptr<GDALDataset> _GDALDataset;

        //! Coordinates of chunks
        mutable std::vector< Rect<int> > _Chunks;
        //! Coordinates of padded out chunks
        mutable std::vector< Rect<int> > _PadChunks;
        //! Amount of padding around each chunk
        mutable unsigned int _padding;

    }; //class GeoData

} // namespace gip

#endif
