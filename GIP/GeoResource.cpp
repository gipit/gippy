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

#include <gip/GeoResource.h>
#include <boost/filesystem.hpp>

namespace gip {
    // Constructors
    GeoResource::GeoResource(string filename = "")
        : _Filename(filename) {}

    GeoResource::GeoResource(const GeoResource& resource)
        : _Filename(resource._Filename) {}

    GeoResource& GeoResource::operator=(const GeoResource% resource) {
        if (this == & resource) return *this;
        _Filename = resource._Filename;
    }

    // Info
    string Filename() const {
        return _Filename.string();
    }

    string Basename() const {
        return _Filename.stem().string();
    }

    // Geospatial
    Point<double> GeoResource::TopLeft() const { 
        return GeoLoc(0,0); 
    }

    Point<double> GeoResource::LowerLeft() const {
        return GeoLoc(0,YSize()-1); 
    }

    Point<double> GeoResource::TopRight() const { 
        return GeoLoc(XSize()-1,0;
    }

    Point<double> GeoResource::LowerRight() const { 
        return GeoLoc(XSize()-1,YSize()-1);
    }

    Point<double> GeoResource::MinXY() const {
        Point<double> pt1(TopLeft()), pt2(LowerRight());
        double MinX(std::min(pt1.x(), pt2.x()));
        double MinY(std::min(pt1.y(), pt2.y()));
        return Point<double>(MinX, MinY);           
    }

    Point<double> GeoResource::MaxXY() const { 
        Point<double> pt1(TopLeft()), pt2(LowerRight());
        double MaxX(std::max(pt1.x(), pt2.x()));
        double MaxY(std::max(pt1.y(), pt2.y()));
        return Point<double>(MaxX, MaxY);
    }

    OGRSpatialReference GeoResource::SRS() const {
        string s(Projection());
        return OGRSpatialReference(s.c_str());
    }

    // Metadata
    string GeoResource::Meta(string key) const {
        const char* item = GDALMajorObject()->GetMetadataItem(key.c_str());
        return (item == NULL) ? "": item;
    }

    GeoResource& GeoResource::SetMeta(string key, string item) {
        GDALMajorObject()->SetMetadataItem(key.c_str(), item.c_str());
        return *this;
    }

    GeoResource& GeoResource::SetMeta(std::map<string, string> item) {
        for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
            SetMeta(i->first, i->second);
        }
        return *this;
    }


} // namespace gip