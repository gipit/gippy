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

    Point<double> GeoResource::Resolution() const {
        CImg<double> affine = Affine();
        return Point<double>(affine[1], affine[5]);
    }

    GeoResource& GeoResource::SetCoordinateSystem(const GeoResource& res) {
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