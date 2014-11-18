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
#include <boost/filesystem.hpp>
#include <gip/geometry.h>

namespace gip {
    using std::string:

    //! Base class representing a geospatial resource
    class GeoResource {
    public:
        //! \name Constructors
        //! Default Constructor with filename
        GeoResource(string filename = "")
            : _Filename(filename) {}

        //! Copy constructor
        GeoResource(const GeoResource&);

        //! Assignment copy
        GeoResource& operator=(const GeoResource&);

        //! Destructor
        ~GeoResource();


        //! \name Resource Information
        //! Filename, or some other resource identifier
        string Filename() const { return _Filename.string(); }
        //! Basename, or short name of filename
        string Basename() const { return _Filename.stem().string(); }
        //! Format of resource
        virtual string Format() const;

        //! \name Geospatial information
        //! Width of resource
        virtual unsigned int XSize() const;
        //! Height of resource
        virtual unsigned int YSize() const;
        //! Geolocated coordinates of a point within the resource
        virtual Point<double> GeoLoc(float xloc, float yloc) const;

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
        virtual string Projection() const;
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const {
            std::string s(Projection());
            return OGRSpatialReference(s.c_str());
        }


        //! \name Metadata functions
        //! Get metadata item
        string GetMeta(string key) const {
            const char* item = GetGDALMajorObject()->GetMetadataItem(key.c_str());
            return (item == NULL) ? "": item;
        }
        // Get group of metadata
        //std::vector<std::string> GetMetaGroup(std::string group,std::string filter="") const;
        //! Set metadata item
        GeoResource& SetMeta(std::string key, std::string item) {
            GetGDALMajorObject()->SetMetadataItem(key.c_str(), item.c_str());
            return *this;
        }
        //! Set multiple metadata items
        GeoResource& SetMeta(std::map<std::string, std::string> items) {
            for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
                SetMeta(i->first, i->second);
            }
            return *this;
        }
        //! Copy Meta data from another resource
        GeoResource& CopyMeta(const GeoResource& img);


    protected:
        virtual GDALMajorObject* GetGDALMajorObject() const;

        boost::filesystem::path _Filename;

    }; // class GeoResource
} // namespace gip

#endif