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

#include <gip/GeoVectorResource.h>

#include <iostream>

#include <gdal/cpl_error.h>


namespace gip {
    using std::string;
    using std::cout;
    using std::endl;
    using boost::filesystem::path;

    // Constructors
    GeoVectorResource::GeoVectorResource(string filename, string layer) 
        : _Filename(filename) {
        _OGRDataSource.reset(OGRSFDriverRegistrar::Open(filename.c_str()), OGRDataSource::DestroyDataSource);
        if (_OGRDataSource == NULL) {
            //BOOST_LOG_TRIVIAL(fatal) << "Error creating " << _Filename.string() << CPLGetLastErrorMsg() << std::endl;
            cout << "Error opening " << _Filename.string() << CPLGetLastErrorMsg() << endl;
        }
        OpenLayer(layer);
    }

    // Copy constructor
    GeoVectorResource::GeoVectorResource(const GeoVectorResource& vector) 
        : _Filename(vector._Filename), _OGRDataSource(vector._OGRDataSource), _Layer(vector._Layer) {
        if (Options::Verbose() > 3) 
            cout << "GeoVectorResource copy: use_count = " << _OGRDataSource.use_count() << endl;
    }

    // Assignment
    GeoVectorResource& GeoVectorResource::operator=(const GeoVectorResource& vector) {
        if (this == &vector) return *this;
        _Filename = vector._Filename;
        _OGRDataSource = vector._OGRDataSource;
        _Layer = vector._Layer;
        if (Options::Verbose() > 3) 
            cout << "GeoVectorResource assignment: use_count = " << _OGRDataSource.use_count() << endl;
        return *this;
    }

    GeoVectorResource::~GeoVectorResource() {
        if (Options::Verbose() > 3) 
            cout << Basename() << ": ~GeoVectorResource (use_count = " << _OGRDataSource.use_count() << ")" << endl;
    }

    // Open layer
    void GeoVectorResource::OpenLayer(string layer) {
        if (Options::Verbose() > 3)
            cout << Basename() << ": opening layer " << layer << endl;
        if (layer == "") {
            _Layer = _OGRDataSource->GetLayer(0);
        } else {
            _Layer = _OGRDataSource->GetLayerByName(layer.c_str());        
        }
    }

    // Info
    string GeoVectorResource::Filename() const {
        return _Filename.string();
    }

    path GeoVectorResource::Path() const {
        return _Filename;
    }

    string GeoVectorResource::Basename() const {
        return _Filename.stem().string();
    }

    // Geospatial
    OGRSpatialReference GeoVectorResource::SRS() const {
       return *_Layer->GetSpatialRef();
    }

    std::string GeoVectorResource::Projection() const {
        char* wkt(NULL);
        _Layer->GetSpatialRef()->exportToWkt(&wkt);
        return std::string(wkt); 
    }

    Rect<double> GeoVectorResource::Extent() const {
        OGREnvelope ext;
        _Layer->GetExtent(&ext, true);
        return Rect<double>(
            Point<double>(ext.MinX, ext.MinY),
            Point<double>(ext.MaxX, ext.MaxY)
        );
    }

} //namespace gip
