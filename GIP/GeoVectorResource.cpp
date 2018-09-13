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

#include <gip/GeoVectorResource.h>

#include <iostream>
#include <functional>

#include <cpl_error.h>


namespace gip {
    using std::string;
    using std::cout;
    using std::endl;

    // Constructors
    GeoVectorResource::GeoVectorResource(string filename, string layer) 
        : _Filename(filename), _PrimaryKey("") {
        // conditional compilation
        #ifdef GDAL2
            _OGRDataSource.reset((OGRDataSource*)GDALOpenEx(filename.c_str(), GDAL_OF_READONLY, NULL, NULL, NULL), OGRDataSource::DestroyDataSource);
        #else
            _OGRDataSource.reset(OGRSFDriverRegistrar::Open(filename.c_str()), OGRDataSource::DestroyDataSource);
        #endif
        if (_OGRDataSource == NULL) {
            throw std::runtime_error("Error opening " + filename + ": " + string(CPLGetLastErrorMsg()));
        }
        OpenLayer(layer);
    }

    // Copy constructor
    GeoVectorResource::GeoVectorResource(const GeoVectorResource& vector) 
        : _Filename(vector._Filename), _OGRDataSource(vector._OGRDataSource), 
            _Layer(vector._Layer), _PrimaryKey(vector._PrimaryKey) {
    }

    // Assignment
    GeoVectorResource& GeoVectorResource::operator=(const GeoVectorResource& vector) {
        if (this == &vector) return *this;
        _Filename = vector._Filename;
        _OGRDataSource = vector._OGRDataSource;
        _Layer = vector._Layer;
        _PrimaryKey = vector._PrimaryKey;
        return *this;
    }

    GeoVectorResource::~GeoVectorResource() {}

    // Open layer
    void GeoVectorResource::OpenLayer(string layer) {
        //if (Options::verbose() > 4)
        //    cout << Basename() << ": opening layer " << layer << endl;
        if (layer == "") {
            _Layer = _OGRDataSource->GetLayer(0);
        } else {
            _Layer = _OGRDataSource->GetLayerByName(layer.c_str());        
        }
    }

    // Info
    string GeoVectorResource::filename() const {
        return _Filename;
    }

    string GeoVectorResource::basename() const {
        return layer_name();
    }

    string GeoVectorResource::layer_name() const {
        return _Layer->GetName();
    }

    // Geospatial
    /*OGRSpatialReference GeoVectorResource::SRS() const {
       return *_Layer->GetSpatialRef();
    }*/

    std::string GeoVectorResource::srs() const {  
        auto deleter = [](char* p) {CPLFree(p); };
        char* wkt(NULL);
        _Layer->GetSpatialRef()->exportToWkt(&wkt);
        std::unique_ptr<char, decltype(deleter)> wktPtr(wkt, deleter); //make sure the char* is freed.
        return std::string(wkt); //char* is copied
    }

    BoundingBox GeoVectorResource::extent() const {
        OGREnvelope ext;
        _Layer->GetExtent(&ext, true);
        return BoundingBox(
            Point<double>(ext.MinX, ext.MinY),
            Point<double>(ext.MaxX, ext.MaxY)
        );
    }

} //namespace gip
