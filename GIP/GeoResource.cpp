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

#include <gip/GeoResource.h>
#include <boost/filesystem.hpp>

// logging
//#include <boost/log/core.hpp>
//#include <boost/log/trivial.hpp>
//#include <boost/log/expressions.hpp>

namespace gip {
    using std::string;
    using std::vector;
    using boost::filesystem::path;

    // Options given initial values here
    //boost::filesystem::path Options::_ConfigDir("/usr/share/gip/");
    string Options::_DefaultFormat("GTiff");
    float Options::_ChunkSize(128.0);
    int Options::_Verbose(1);
    int Options::_NumCores(2);
    string Options::_WorkDir("/tmp/");

    // Constructors
    GeoResource::GeoResource(string filename, bool update)
        : _Filename(filename) {

        // read/write permissions
        GDALAccess access = update ? GA_Update : GA_ReadOnly;
        if (access == GA_ReadOnly)
            CPLSetConfigOption("GDAL_PAM_ENABLED","NO");
        else CPLSetConfigOption("GDAL_PAM_ENABLED",NULL);

        // open dataset
        GDALDataset* ds = (GDALDataset*)GDALOpen(_Filename.string().c_str(), access);
        // Check if Update access not supported
        if (ds == NULL) // && CPLGetLastErrorNo() == 6)
            ds = (GDALDataset*)GDALOpen(_Filename.string().c_str(), GA_ReadOnly);
        if (ds == NULL) {
            throw std::runtime_error(to_string(CPLGetLastErrorNo()) + ": " + string(CPLGetLastErrorMsg()));
        }
        _GDALDataset.reset(ds);

        // boost logging test
        //BOOST_LOG_TRIVIAL(trace) << Basename() << ": GeoResource Open (use_count = " << _GDALDataset.use_count() << ")" << std::endl;

        if (Options::Verbose() > 4)
            std::cout << Basename() << ": GeoResource Open (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
    }


    GeoResource::GeoResource(int xsz, int ysz, int bsz, GDALDataType datatype, string filename, dictionary options)
        : _Filename(filename) {

        // format, driver, and file extension
        string format = Options::DefaultFormat();
        //if (format == "GTiff") options["COMPRESS"] = "LZW";
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        // TODO check for null driver and create method
        // Check extension
        string ext = driver->GetMetadataItem(GDAL_DMD_EXTENSION);
        if (ext != "" && _Filename.extension().string() != ('.'+ext)) _Filename = boost::filesystem::path(_Filename.string() + '.' + ext);

        // add options
        char **papszOptions = NULL;
        if (options.size()) {
            for (dictionary::const_iterator imap=options.begin(); imap!=options.end(); imap++)
                papszOptions = CSLSetNameValue(papszOptions,imap->first.c_str(),imap->second.c_str());
        }

        // create file
        //BOOST_LOG_TRIVIAL(info) << Basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        if (Options::Verbose() > 4)
            std::cout << Basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        _GDALDataset.reset( driver->Create(_Filename.string().c_str(), xsz,ysz,bsz,datatype, papszOptions) );
        if (_GDALDataset.get() == NULL) {
            //BOOST_LOG_TRIVIAL(fatal) << "Error creating " << _Filename.string() << CPLGetLastErrorMsg() << std::endl;
            std::cout << "Error creating " << _Filename.string() << CPLGetLastErrorMsg() << std::endl;
        }
    }

    GeoResource::GeoResource(const GeoResource& resource)
        : _Filename(resource._Filename), _GDALDataset(resource._GDALDataset) {}

    GeoResource& GeoResource::operator=(const GeoResource& resource) {
        if (this == &resource) return *this;
        _Filename = resource._Filename;
        _GDALDataset = resource._GDALDataset;
        return *this;
    }

    GeoResource::~GeoResource() {
        // flush GDALDataset if last open pointer
        if (_GDALDataset.unique()) {
            _GDALDataset->FlushCache();
            //BOOST_LOG_TRIVIAL(trace) << Basename() << ": ~GeoResource (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
            if (Options::Verbose() > 4) std::cout << Basename() << ": ~GeoResource (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
        }
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
        const char* item = GetGDALObject()->GetMetadataItem(key.c_str());
        return (item == NULL) ? "": item;
    }

    // Get metadata group
    vector<string> GeoResource::MetaGroup(string group, string filter) const {
        char** meta= GetGDALObject()->GetMetadata(group.c_str());
        int num = CSLCount(meta);
        vector<string> items;
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
        GetGDALObject()->SetMetadataItem(key.c_str(), item.c_str());
        return *this;
    }

    GeoResource& GeoResource::SetMeta(std::map<string, string> items) {
        for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
            SetMeta(i->first, i->second);
        }
        return *this;
    }

    GeoResource& GeoResource::CopyMeta(const GeoResource& resource) {
        GetGDALObject()->SetMetadata(resource.GetGDALObject()->GetMetadata());
        return *this;
    }


} // namespace gip
