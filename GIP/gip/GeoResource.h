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

#ifndef GIP_GEORESOURCE_H
#define GIP_GEORESOURCE_H

#include <string>
#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <gdal/gdal_priv.h>
#include <gip/gip_CImg.h>
#include <gip/geometry.h>

namespace gip {
    using std::string;
    using std::vector;
    using boost::filesystem::path;
    typedef std::map<std::string,std::string> dictionary;

    //! Base class representing a geospatial resource
    class GeoResource {
    public:
        //! \name Constructors
        //! Default Constructor with filename
        GeoResource(string filename = "");
        //! Copy constructor
        GeoResource(const GeoResource& resource);
        //! Assignment copy
        GeoResource& operator=(const GeoResource&);
        //! Destructor
        virtual ~GeoResource() {}

        //! \name Resource Information
        //! Get the filename of the resource
        string Filename() const;
        //! Get path (boost filesystem path)
        path Path() const;
        //! Basename, or short name of filename
        string Basename() const;
        //! Format of resource
        virtual string Format() const = 0;

        //! \name Geospatial information
        //! Width of resource
        virtual unsigned int XSize() const = 0;
        //! Height of resource
        virtual unsigned int YSize() const = 0;
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
        //! Return projection definition in Well Known Text format
        virtual string Projection() const = 0;
        //! Set projection definition in Well Known Text format
        virtual GeoResource& SetProjection(string) = 0;
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const;
        //! Get Affine transformation
        virtual CImg<double> Affine() const = 0;
        //! Set Affine transformation
        virtual GeoResource& SetAffine(CImg<double>) = 0;
        //! Set coordinate system from another GeoResource
        GeoResource& SetCoordinateSystem(const GeoResource& res);
        //! Copy coordinate system
        GeoResource& CopyCoordinateSystem(const GeoResource& res) {
            std::cerr << "GIPPY Deprecation Warning: Use SetCoordinateSystem instead of CopyCoordinateSystem" << std::endl;
            return SetCoordinateSystem(res);
        }

        //! Get chunkset chunking up image
        ChunkSet Chunks(unsigned int padding=0, unsigned int numchunks=0) const;

        //! \name Metadata functions
        //! Get metadata item
        string Meta(string key) const;
        // Get group of metadata
        vector<string> MetaGroup(string group, string filter="") const;
        //! Set metadata item
        GeoResource& SetMeta(std::string key, std::string item);
        //! Set multiple metadata items
        GeoResource& SetMeta(dictionary items);
        //! Copy Meta data from another resource
        GeoResource& CopyMeta(const GeoResource& img);

    protected:
        //! Filename, or some other resource identifier
        boost::filesystem::path _Filename;

        //! Retrieve the GDALMajorObject from (GDALDataset, GDALRasterBand, OGRLayer)
        virtual GDALMajorObject* GDALObject() const = 0;

    }; // class GeoResource
} // namespace gip

#endif