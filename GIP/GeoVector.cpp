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

#include <gip/GeoVector.h>

#include <iostream>

namespace gip {
    using std::string;
    using std::cout;
    using std::endl;
    using boost::filesystem::path;

    // Constructors
    GeoVector::GeoVector(string filename, string layer) {
        _OGRDataSource.reset( OGRSFDriverRegistrar::Open(filename.c_str()) );
        OpenLayer(layer);
    }

    // Copy constructor
    GeoVector::GeoVector(const GeoVector& vector) 
        : _Filename(vector._Filename), _OGRDataSource(vector._OGRDataSource) {
        OpenLayer(vector._OGRLayer->GetName());
    }

    // Assignment
    GeoVector& GeoVector::operator=(const GeoVector& vector) {
        if (this == &vector) return *this;
        _Filename = vector._Filename;
        _OGRDataSource = vector._OGRDataSource;
        OpenLayer(vector._OGRLayer->GetName());
        return *this;
    }

    GeoVector::~GeoVector() {
        //if (_OGRDataSource.unique());
        if (Options::Verbose() > 3) 
            cout << Basename() << ": ~GeoVector (use_count = " << _OGRDataSource.use_count() << ")" << endl;
    }

    // Open layer
    void GeoVector::OpenLayer(string layer) {
        if (layer == "")
            _OGRLayer = _OGRDataSource->GetLayer(0);
        else
            _OGRLayer = _OGRDataSource->GetLayerByName(layer.c_str());        
        // read in features
        _OGRLayer->ResetReading();
        OGRFeature* feature;
        while( (feature = _OGRLayer->GetNextFeature()) != NULL) {
            _Features.push_back(GeoFeature(feature));
        }
    }

    // Info
    string GeoVector::Filename() const {
        return _Filename.string();
    }

    path GeoVector::Path() const {
        return _Filename;
    }

    string GeoVector::Basename() const {
        return _Filename.stem().string();
    }

    // Geospatial
    OGRSpatialReference GeoVector::SRS() const {
       return *_OGRLayer->GetSpatialRef();
    }

    std::string GeoVector::Projection() const {
        char* wkt(NULL);
        _OGRLayer->GetSpatialRef()->exportToWkt(&wkt);
        return std::string(wkt); 
    }

    Rect<double> GeoVector::Extent() const {
        OGREnvelope ext;
        _OGRLayer->GetExtent(&ext, true);
        return Rect<double>(
            Point<double>(ext.MinX, ext.MinY),
            Point<double>(ext.MaxX, ext.MaxY)
        );
    }

} //namespace gip
