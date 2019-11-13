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

#include <cstdio>
#include <gip/GeoResource.h>
#include <gip/utils.h>


namespace gip {
    using std::string;
    using std::vector;

    // Constructors
    GeoResource::GeoResource(string filename, bool update, bool temp)
        : _Filename(filename), _temp(temp) {

        // read/write permissions
        GDALAccess access = update ? GA_Update : GA_ReadOnly;
        if (access == GA_ReadOnly)
            CPLSetConfigOption("GDAL_PAM_ENABLED","NO");
        else CPLSetConfigOption("GDAL_PAM_ENABLED",NULL);

        // open dataset
        GDALDataset* ds = (GDALDataset*)GDALOpen(_Filename.c_str(), access);
        // Check if Update access not supported
        if (ds == NULL) // && CPLGetLastErrorNo() == 6)
            ds = (GDALDataset*)GDALOpen(_Filename.c_str(), GA_ReadOnly);
        if (ds == NULL) {
            throw std::runtime_error(to_string(CPLGetLastErrorNo()) + ": " + string(CPLGetLastErrorMsg()));
        }
        _GDALDataset.reset(ds);

        if (Options::verbose() > 4)
            std::cout << basename() << ": GeoResource Open (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
    }

    //! Create new file
    GeoResource::GeoResource(string filename, int xsz, int ysz, int bsz, 
                             string proj, BoundingBox bbox, 
                             DataType dt, std::string format, bool temp, dictionary options)
        : _Filename(filename), _temp(temp) {

        // format, driver, and file extension
        if (format == "")
            format = Options::defaultformat();
        //if (format == "GTiff") options["COMPRESS"] = "LZW";
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        // TODO check for null driver and create method
        // check if extension (case insensitive) is already in filename
        const char* _ext = driver->GetMetadataItem(GDAL_DMD_EXTENSION);
        string ext = (_ext == NULL) ? "": _ext;
        string curext = extension();
        if ((to_lower(ext) != to_lower(curext)) && ext != "") {
            _Filename = _Filename + '.' + ext;
        }

        // add options
        char **papszOptions = NULL;
        // if tif and 3 or 4 band make RGBA
        if (format == "GTiff") {
            if (bsz == 3 || bsz == 4)
                papszOptions = CSLSetNameValue(papszOptions, "PHOTOMETRIC", "RGB");
            if (bsz == 4)
                papszOptions = CSLSetNameValue(papszOptions, "ALPHA", "YES");
        }
        
        if (options.size()) {
            for (dictionary::const_iterator imap=options.begin(); imap!=options.end(); imap++)
                papszOptions = CSLSetNameValue(papszOptions,imap->first.c_str(),imap->second.c_str());
        }

        // create file
        //BOOST_LOG_TRIVIAL(info) << Basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        if (Options::verbose() > 4)
            std::cout << basename() << ": create new file " << xsz << " x " << ysz << " x " << bsz << std::endl;
        _GDALDataset.reset( driver->Create(_Filename.c_str(), xsz,ysz,bsz, dt.gdal(), papszOptions) );
        if (_GDALDataset.get() == NULL) {
            std::cout << "Error creating " << _Filename << CPLGetLastErrorMsg() << std::endl;
        }
        set_srs(proj);
        CImg<double> affine(6, 1, 1, 1,
            // xmin, xres, xshear
            bbox.x0(), bbox.width() / (float)xsz, 0.0,
            // ymin, yshear, yres
            bbox.y1(), 0.0, -std::abs(bbox.height() / (float)ysz)  
        );
        set_affine(affine);
        CSLDestroy(papszOptions);
    }

    GeoResource::GeoResource(const GeoResource& resource)
        : _Filename(resource._Filename), _GDALDataset(resource._GDALDataset), _temp(resource._temp) {}

    GeoResource& GeoResource::operator=(const GeoResource& resource) {
        if (this == &resource) return *this;
        _Filename = resource._Filename;
        _GDALDataset = resource._GDALDataset;
        _temp = resource._temp;
        return *this;
    }

    GeoResource::~GeoResource() {
        // flush GDALDataset if last open pointer
        if (_GDALDataset.unique()) {
            _GDALDataset->FlushCache();
            //BOOST_LOG_TRIVIAL(trace) << Basename() << ": ~GeoResource (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
            if (Options::verbose() > 4) std::cout << basename() << ": ~GeoResource (use_count = " << _GDALDataset.use_count() << ")" << std::endl;
            if (_temp) {
                std::remove(_Filename.c_str());
            }
        }
    }

    // Info
    //! Get full filename
    string GeoResource::filename() const {
        return _Filename;
    }

    //! Return basename of filename (no path, no extension)
    string GeoResource::basename() const {
        return gip::Basename(_Filename);
    }

    //! Get extension of filename
    string GeoResource::extension() const {
        return gip::Extension(_Filename);
    }

    // Geospatial
    Point<double> GeoResource::geoloc(float xloc, float yloc) const {
        CImg<double> aff = affine();
        Point<double> pt(aff[0] + xloc*aff[1] + yloc*aff[2], aff[3] + xloc*aff[4] + yloc*aff[5]);
        return pt;
    }

    // Geospatial
    Point<double> GeoResource::latlon(float xloc, float yloc) const {
        return geoloc(xloc, yloc).transform(srs(), "EPSG:4326");
    }

    /*Point<double> GeoResource::topleft() const { 
        return geoloc(0, 0); 
    }

    Point<double> GeoResource::lowerleft() const {
        return geoloc(0, YSize()-1); 
    }

    Point<double> GeoResource::topright() const { 
        return geoloc(XSize()-1, 0);
    }

    Point<double> GeoResource::lowerright() const { 
        return geoloc(XSize()-1, YSize()-1);
    }*/

    Point<double> GeoResource::minxy() const {
        Point<double> pt1(geoloc(0,0)), pt2(geoloc(xsize()-1, ysize()-1));
        double MinX(std::min(pt1.x(), pt2.x()));
        double MinY(std::min(pt1.y(), pt2.y()));
        return Point<double>(MinX, MinY);           
    }

    Point<double> GeoResource::maxxy() const { 
        Point<double> pt1(geoloc(0,0)), pt2(geoloc(xsize(), ysize()));
        double MaxX(std::max(pt1.x(), pt2.x()));
        double MaxY(std::max(pt1.y(), pt2.y()));
        return Point<double>(MaxX, MaxY);
    }

    Point<double> GeoResource::resolution() const {
        CImg<double> aff = affine();
        return Point<double>(aff[1], aff[5]);
    }

    std::vector< Chunk > GeoResource::chunks(unsigned int padding, unsigned int numchunks) const {
        std::vector< Chunk > _Chunks;
        unsigned int rows;

        if (numchunks == 0) {
            // calculate based on global variable chunksize
            rows = floor( ( Options::chunksize() *1024*1024) / sizeof(double) / xsize() );
            rows = rows > ysize() ? ysize() : rows;
            numchunks = ceil( ysize()/(float)rows );
        } else {
            rows = ceil(ysize() / (float)numchunks);
        }

        _Chunks.clear();
        Chunk chunk;
        for (unsigned int i=0; i<numchunks; i++) {
            chunk = Chunk(0, rows*i, xsize(), std::min(rows*(i+1),ysize())-(rows*i) );
            chunk.padding(padding);
            _Chunks.push_back(chunk);
            //if (Options::verbose() > 3) std::cout << "  Chunk " << i << ": " << chunk << std::endl;
        }
        return _Chunks;        
    }

    // Metadata
    string GeoResource::meta(string key) const {
        const char* item = _GDALDataset->GetMetadataItem(key.c_str());
        return (item == NULL) ? "": item;
    }

    dictionary GeoResource::meta() const {
        char** meta = _GDALDataset->GetMetadata();
        int num = CSLCount(meta);
        dictionary items;
        for (int i=0;i<num; i++) {
            string md = string(meta[i]);
            string::size_type pos = md.find("=");
            if (pos != string::npos) {
                items[md.substr(0, pos)] = md.substr(pos+1);
            }
        }
        return items;
    }


    GeoResource& GeoResource::add_meta(string key, string item) {
        _GDALDataset->SetMetadataItem(key.c_str(), item.c_str());
        return *this;
    }

    GeoResource& GeoResource::add_meta(std::map<string, string> items) {
        for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
            add_meta(i->first, i->second);
        }
        return *this;
    }

    // Get metadata group - used internally
    vector<string> GeoResource::metagroup(string group, string filter) const {
        char** meta= _GDALDataset->GetMetadata(group.c_str());
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

} // namespace gip
