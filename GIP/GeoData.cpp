/*
 * gip_GeoData.cpp
 *
 *  Created on: Aug 26, 2011
 *  Author: mhanson
 */

#include <gip/GeoData.h>
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
    GeoData::GeoData(string filename, bool Update) : _Filename(filename) {
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
        if (Options::Verbose() > 4)
            std::cout << Basename() << ": GeoData Open (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
    }

    // Create new file
    GeoData::GeoData(int xsz, int ysz, int bsz, GDALDataType datatype, string filename, dictionary options)
        :_Filename(filename) {
        if (Options::Verbose() > 3)
            std::cout << Basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        string format = Options::DefaultFormat();
        //if (format == "GTiff" && datatype == GDT_Byte) options["COMPRESS"] = "JPEG";
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
        if (_GDALDataset.get() == NULL)
            std::cout << "Error creating " << _Filename.string() << CPLGetLastErrorMsg() << std::endl;
    }

    // Copy constructor
    GeoData::GeoData(const GeoData& geodata)
        : _Filename(geodata._Filename), _GDALDataset(geodata._GDALDataset), _Chunks(geodata._Chunks), _PadChunks(geodata._PadChunks) {
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
        return *this;
    }

    // Destructor
    GeoData::~GeoData() {
        // flush GDALDataset if last open pointer
        if (_GDALDataset.unique()) {
            _GDALDataset->FlushCache();
            if (Options::Verbose() > 4) std::cout << Basename() << ": ~GeoData (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
        }
    }

    // Data type size
    /*int GeoData::DataTypeSize() const {
        switch(DataType()) {
            case 1: return sizeof(unsigned char);
            case 2: return sizeof(unsigned short);
            case 3: return sizeof(short);
            case 4: return sizeof(unsigned int);
            case 5: return sizeof(int);
            case 6: return sizeof(float);
            case 7: return sizeof(double);
            default: throw(std::exception());
        }
    }*/

    // Using GDALDatasets GeoTransform get Geo-located coordinates
    Point<double> GeoData::GeoLoc(float xloc, float yloc) const {
        double Affine[6];
        _GDALDataset->GetGeoTransform(Affine);
        Point<double> Coord(Affine[0] + xloc*Affine[1] + yloc*Affine[2], Affine[3] + xloc*Affine[4] + yloc*Affine[5]);
        return Coord;
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

    //! Break up image into smaller size pieces, each of ChunkSize
    void GeoData::Chunk(unsigned int pad) const {
        unsigned int rows = floor( ( Options::ChunkSize() *1024*1024) / sizeof(double) / XSize() );
        rows = rows > YSize() ? YSize() : rows;
        int numchunks = ceil( YSize()/(float)rows );
        _Chunks.clear();
        _PadChunks.clear();
        iRect chunk;
        if (Options::Verbose() > 4) {
            std::cout << Basename() << ": chunking into " << numchunks << " chunks (" 
                << Options::ChunkSize() << " MB max each)" << " with pad = " << pad << std::endl;
        }
        for (int i=0; i<numchunks; i++) {
            chunk = iRect(0, rows*i, XSize(), std::min(rows*(i+1),YSize())-(rows*i) );
            _Chunks.push_back(chunk);
            if (Options::Verbose() > 4) std::cout << "  Chunk " << i << ": " << chunk << std::endl;

            if (pad > 0) {
                chunk.Pad(pad).Intersect(iRect(0,0,XSize(),YSize()));
                if (Options::Verbose() > 4) std::cout << "  PadChunk " << i << ": " << chunk << std::endl;
            }
            _PadChunks.push_back(chunk);
        }
    }

} // namespace gip
