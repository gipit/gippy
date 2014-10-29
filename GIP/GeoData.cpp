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

#include <gip/GeoData.h>
#include <gip/gip_gdal.h>
#include <boost/make_shared.hpp>

#include <iostream>

namespace gip {
    using std::string;

    // Options given initial values here
    //boost::filesystem::path Options::_ConfigDir("/usr/share/gip/");
    string Options::_DefaultFormat("GTiff");
    float Options::_ChunkSize(128.0);
    int Options::_Verbose(1);
    string Options::_WorkDir("/tmp/");

    // Open existing file
    GeoData::GeoData(string filename, bool Update) 
        : _Filename(filename), _padding(0) {
        GDALAccess access = Update ? GA_Update : GA_ReadOnly;
        if (access == GA_ReadOnly)
            CPLSetConfigOption("GDAL_PAM_ENABLED","NO");
        else CPLSetConfigOption("GDAL_PAM_ENABLED",NULL);
        GDALDataset* ds = (GDALDataset*)GDALOpenShared(_Filename.string().c_str(), access);
        // Check if Update access not supported
        if (ds == NULL) // && CPLGetLastErrorNo() == 6)
            ds = (GDALDataset*)GDALOpenShared(_Filename.string().c_str(), GA_ReadOnly);
        if (ds == NULL) {
            throw std::runtime_error(to_string(CPLGetLastErrorNo()) + ": " + string(CPLGetLastErrorMsg()));
        }
        _GDALDataset.reset(ds);
        if (Options::Verbose() > 3)
            std::cout << Basename() << ": GeoData Open (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
    }

    // Create new file
    GeoData::GeoData(int xsz, int ysz, int bsz, GDALDataType datatype, string filename, dictionary options)
        :_Filename(filename), _padding(0) {
        string format = Options::DefaultFormat();
        //if (format == "GTiff") options["COMPRESS"] = "LZW";
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        // TODO check for null driver and create method
        // Check extension
        string ext = driver->GetMetadataItem(GDAL_DMD_EXTENSION);
        if (ext != "" && _Filename.extension().string() != ('.'+ext)) _Filename = boost::filesystem::path(_Filename.string() + '.' + ext);
        char **papszOptions = NULL;
        if (options.size()) {
            for (dictionary::const_iterator imap=options.begin(); imap!=options.end(); imap++)
                papszOptions = CSLSetNameValue(papszOptions,imap->first.c_str(),imap->second.c_str());
        }
        _GDALDataset.reset( driver->Create(_Filename.string().c_str(), xsz,ysz,bsz,datatype, papszOptions) );

        if (Options::Verbose() > 3)
            std::cout << Basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        if (_GDALDataset.get() == NULL)
            std::cout << "Error creating " << _Filename.string() << CPLGetLastErrorMsg() << std::endl;
    }

    // Copy constructor
    GeoData::GeoData(const GeoData& geodata)
        : _Filename(geodata._Filename), _GDALDataset(geodata._GDALDataset), 
            _Chunks(geodata._Chunks), _PadChunks(geodata._PadChunks), _padding(geodata._padding) {
    }

    // Assignment copy
    GeoData& GeoData::operator=(const GeoData& geodata) {
    //GeoData& GeoData::operator=(const GeoData& geodata) {
        // Check for self assignment
        if (this == &geodata) return *this;
        _Filename = geodata._Filename;
        _GDALDataset = geodata._GDALDataset;
        _Chunks = geodata._Chunks;
        _PadChunks = geodata._PadChunks;
        _padding = geodata._padding;
        return *this;
    }

    // Destructor
    GeoData::~GeoData() {
        // flush GDALDataset if last open pointer
        if (_GDALDataset.unique()) {
            _GDALDataset->FlushCache();
            if (Options::Verbose() > 3) std::cout << Basename() << ": ~GeoData (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
        }
    }

    // Using GDALDatasets GeoTransform get Geo-located coordinates
    Point<double> GeoData::GeoLoc(float xloc, float yloc) const {
        double Affine[6];
        _GDALDataset->GetGeoTransform(Affine);
        Point<double> Coord(Affine[0] + xloc*Affine[1] + yloc*Affine[2], Affine[3] + xloc*Affine[4] + yloc*Affine[5]);
        return Coord;
    }

    // Get metadata group
    std::vector<string> GeoData::GetMetaGroup(string group,string filter) const {
        char** meta= _GDALDataset->GetMetadata(group.c_str());
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

    // Copy all metadata from input
    GeoData& GeoData::CopyMeta(const GeoData& img) {
        _GDALDataset->SetMetadata(img._GDALDataset->GetMetadata());
        return *this;
    }

    // Copy coordinate system from another image
    GeoData& GeoData::CopyCoordinateSystem(const GeoData& img) {
        GDALDataset* ds = const_cast<GeoData&>(img)._GDALDataset.get();
        _GDALDataset->SetProjection(ds->GetProjectionRef());
        double Affine[6];
        ds->GetGeoTransform(Affine);
        _GDALDataset->SetGeoTransform(Affine);
        return *this;
    }

    //! Break up image into smaller size pieces, each of ChunkSize
    std::vector< Rect<int> > GeoData::Chunk(unsigned int numchunks) const {
        unsigned int rows;

        if (numchunks == 0) {
            rows = floor( ( Options::ChunkSize() *1024*1024) / sizeof(double) / XSize() );
            rows = rows > YSize() ? YSize() : rows;
            numchunks = ceil( YSize()/(float)rows );
        } else {
            rows = int(YSize() / numchunks);
        }

        _Chunks.clear();
        _PadChunks.clear();
        iRect chunk, pchunk;
        /*if (Options::Verbose() > 3) {
            std::cout << Basename() << ": chunking into " << numchunks << " chunks (" 
                << Options::ChunkSize() << " MB max each)" << " padding = " << _padding << std::endl;
        }*/
        for (unsigned int i=0; i<numchunks; i++) {
            chunk = iRect(0, rows*i, XSize(), std::min(rows*(i+1),YSize())-(rows*i) );
            _Chunks.push_back(chunk);
            pchunk = chunk;
            if (_padding > 0)
                pchunk.Pad(_padding).Intersect(iRect(0,0,XSize(),YSize()));
            _PadChunks.push_back(pchunk);
            //if (Options::Verbose() > 3)
            //    std::cout << "  Chunk " << i << ": " << chunk << "\tPadded: " << pchunk << std::endl;
        }
        return _Chunks;
    }

} // namespace gip
