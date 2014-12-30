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
    GeoResource::GeoResource(string filename)
        : _Filename(filename) {}

    GeoResource::GeoResource(const GeoResource& resource)
        : _Filename(resource._Filename) {}

    GeoResource& GeoResource::operator=(const GeoResource& resource) {
        if (this == &resource) return *this;
        _Filename = resource._Filename;
        return *this;
    }

    // Info
    string GeoResource::Filename() const {
        return _Filename.string();
    }

    path GeoResource::Path() const {
        return _Filename;
    }

    string GeoResource::Basename() const {
        return _Filename.stem().string();
    }

    // Geospatial
    Point<double> GeoResource::GeoLoc(float xloc, float yloc) const {
        CImg<double> affine = Affine();
        Point<double> pt(affine[0] + xloc*affine[1] + yloc*affine[2], affine[3] + xloc*affine[4] + yloc*affine[5]);
        return pt;
    }

    Point<double> GeoResource::TopLeft() const { 
        return GeoLoc(0, 0); 
    }

    Point<double> GeoResource::LowerLeft() const {
        return GeoLoc(0, YSize()-1); 
    }

    Point<double> GeoResource::TopRight() const { 
        return GeoLoc(XSize()-1, 0);
    }

    Point<double> GeoResource::LowerRight() const { 
        return GeoLoc(XSize()-1, YSize()-1);
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

    GeoResource& SetCoordinateSystem(const GeoResource& res) {
        SetProjection(res.Projection());
        SetAffine(res.Affine());
        return *this;
    }

    ChunkSet GeoResource::Chunks(unsigned int padding, unsigned int numchunks) const {
        return ChunkSet(XSize(), YSize(), padding, numchunks);
    }

    // Metadata
    string GeoResource::Meta(string key) const {
        const char* item = GDALObject()->GetMetadataItem(key.c_str());
        return (item == NULL) ? "": item;
    }

    // Get metadata group
    vector<string> GeoResource::MetaGroup(string group, string filter) const {
        char** meta= GDALObject()->GetMetadata(group.c_str());
        int num = CSLCount(meta);
        std::vector<string> items;
        for (int i=0;i<num; i++) {
                if (filter != "") {
                        string md = string(meta[i]);
                        string::size_type pos = md.find(filter);
                        if (pos != string::npos) items.push_back(md.substr(pos+filter.length()));
                } else items.push_back( meta[i] );
        }
        return items;
    }

    GeoResource& GeoResource::SetMeta(string key, string item) {
        GDALObject()->SetMetadataItem(key.c_str(), item.c_str());
        return *this;
    }

    GeoResource& GeoResource::SetMeta(std::map<string, string> items) {
        for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
            SetMeta(i->first, i->second);
        }
        return *this;
    }

    GeoResource& GeoResource::CopyMeta(const GeoResource& resource) {
        GDALObject()->SetMetadata(resource.GDALObject()->GetMetadata());
        return *this;
    }


} // namespace gip