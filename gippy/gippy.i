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
%module(docstring="Geospatial Image Processing for Python") gippy
%feature("autodoc", "3");
%{
    #define SWIG_FILE_WITH_INIT
%}


// Exception handling
%include exception.i
%exception {
    try {
        $action
    } catch (const std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError, e.what());
    } catch (const std::invalid_argument& e) {
        SWIG_exception(SWIG_TypeError, e.what());
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}


// STL bindings
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"

namespace std {
    %template(svector) std::vector<std::string>;
    %template(ivector) std::vector<int>;
    %template() std::map<std::string, std::string>;
}


// Ignore these standard functions
%ignore std::cout;
%ignore std::endl;
%ignore operator<<;

%{
    #include <gip/gip.h>
%}


// Wrap CImg
%include "cimg.i"


// Wrap GIPS
%{
    #include <python2.7/Python.h>
    #include <gip/gip.h>
    #include <gip/utils.h>
    #include <gip/geometry.h>
    #include <gip/GeoImage.h>
    #include <gip/GeoVector.h>
    using namespace gip;

    // custom conversions
    // string to DataType
    gip::DataType StringToDataType(PyObject* str) {
        const char* s = PyString_AsString(str);
        return gip::DataType(std::string(s));
    }

%}

// GIP headers and classes to be wrapped - order is important!
//  ignore directives suppress warnings, then operators are redefined through %extend

%include "gip/gip.h"

// Geometry
%ignore gip::Point::operator=;
%ignore gip::Rect::operator=;
%ignore gip::Chunk::operator=;
%include "gip/geometry.h"

%template(iPoint) gip::Point<int>;
%template(dPoint) gip::Point<double>;
%template(Chunk) gip::Rect<int>;
%template(BoundingBox) gip::Rect<double>;
%template(chvector) std::vector< gip::Rect<int> >;
%template(bbvector) std::vector< gip::Rect<double> >;

// DataType
%ignore gip::DataTypes;
%ignore gip::DataType::gdal;
%include "gip/DataType.h"
//%typemap(typecheck) gip::DataType = PyObject*;
//%typemap(in) gip::DataType {
//    $1 = StringToDataType($input);
//}

/*
namesace gip {
    %extend DataType {
        char* __str__() {
            return self->string().c_str();
        }
    }
}
*/


// GeoResource
%ignore gip::GeoResource::operator=;
%include "gip/GeoResource.h"


// GeoRaster
%include "gip/GeoRaster.h"
namespace gip {
    %extend GeoRaster {
        // templated functions that need to be instantiated
        GeoRaster& save(GeoRaster& raster) {
            return self->save<double>(raster);
        }
        PyObject* read_raw(Chunk chunk=Chunk()) {
            switch(self->type().type()) {
                case 1: return CImgToArr(self->read_raw<uint8_t>(chunk));
                case 2: return CImgToArr(self->read_raw<uint16_t>(chunk));
                case 3: return CImgToArr(self->read_raw<int16_t>(chunk));
                case 4: return CImgToArr(self->read_raw<uint32_t>(chunk));
                case 5: return CImgToArr(self->read_raw<int32_t>(chunk));
                case 6: return CImgToArr(self->read_raw<float>(chunk));
                case 7: return CImgToArr(self->read_raw<double>(chunk));
                default: throw(std::runtime_error("error reading raster"));
            }
        }
        %feature("docstring",
                 "PyObject returned is a numpy.array.\n"
                 "Enjoy!\n ");
        PyObject* read(Chunk chunk=Chunk()) {
            if (!self->is_double()) {
                switch(self->type().type()) {
                    case 1: return CImgToArr(self->read<uint8_t>(chunk));
                    case 2: return CImgToArr(self->read<uint16_t>(chunk));
                    case 3: return CImgToArr(self->read<int16_t>(chunk));
                    case 4: return CImgToArr(self->read<uint32_t>(chunk));
                    case 5: return CImgToArr(self->read<int32_t>(chunk));
                    case 6: return CImgToArr(self->read<float>(chunk));
                    case 7: return CImgToArr(self->read<double>(chunk));
                    default: throw(std::runtime_error("error reading raster"));
                }
            }
            return CImgToArr(self->read<double>(chunk));
        }
        %feature("docstring",
                 "PyObject passed in is a numpy.array.\n"
                 "Comply!\n ");
        GeoRaster& write(PyObject* obj, Chunk chunk=Chunk()) {
            PyArrayObject* arr((PyArrayObject*)obj);
            switch(PyArray_TYPE(arr)) {
                case NPY_UINT8: self->write(ArrToCImg<unsigned char>(arr), chunk); break;
                case NPY_UINT16: self->write(ArrToCImg<unsigned short>(arr), chunk); break;
                case NPY_INT16: self->write(ArrToCImg<short>(arr), chunk); break;
                case NPY_UINT32: self->write(ArrToCImg<unsigned int>(arr), chunk); break;
                case NPY_INT32: self->write(ArrToCImg<int>(arr), chunk); break;
                case NPY_UINT64: self->write(ArrToCImg<unsigned int>(arr), chunk); break;
                case NPY_INT64: self->write(ArrToCImg<int>(arr), chunk); break;
                case NPY_FLOAT32: self->write(ArrToCImg<float>(arr), chunk); break;
                case NPY_FLOAT64: self->write(ArrToCImg<double>(arr), chunk); break;
                default:
                    throw(std::invalid_argument("a numpy array is required"));
            }
            return *self;
        }
    }
}


// GeoImage
%ignore gip::GeoImage::operator[];
%include "gip/GeoImage.h"
%template(vector_GeoImage) std::vector< gip::GeoImage >;
namespace gip {
    %extend GeoImage {
        GeoRaster __getitem__(std::string col) {
            return self->GeoImage::operator[](col);
        }
        GeoRaster __getitem__(int band) {
            return self->GeoImage::operator[](band);
        }
        GeoRaster& __setitem__(int band, const GeoRaster& raster) {
            self->operator[](band) = raster;
            return self->GeoImage::operator[](band);
        }
        GeoRaster& __setitem__(std::string col, const GeoRaster& raster) {
            self->operator[](col) = raster;
            return self->operator[](col);
        }
        unsigned long int __len__() {
            return self->nbands();
        }
        GeoImage __deepcopy__(GeoImage image) {
            return GeoImage(image);
        }
        // templated functions that need to be instantiated
        GeoImage save(std::string filename="", std::string dtype="", float nodata=NAN, std::string format="", bool temp=false, bool overviews=false, dictionary options=dictionary()) {
            return self->save<double>(filename, dtype, nodata, format, temp, overviews, options);
        }
        PyObject* read_random_pixels(int num_pixels) {
            return CImgToArr(self->read_random_pixels<double>(num_pixels));
        }
        PyObject* extract_classes(GeoRaster raster) {
            // TODO - look at all bands for gain and offset
            if (!(*self)[0].is_double()) {
                switch(self->type().type()) {
                    case 1: return CImgToArr(self->extract_classes<uint8_t>(raster));
                    case 2: return CImgToArr(self->extract_classes<uint16_t>(raster));
                    case 3: return CImgToArr(self->extract_classes<int16_t>(raster));
                    case 4: return CImgToArr(self->extract_classes<uint32_t>(raster));
                    case 5: return CImgToArr(self->extract_classes<int32_t>(raster));
                    case 6: return CImgToArr(self->extract_classes<float>(raster));
                    case 7: return CImgToArr(self->extract_classes<double>(raster));
                    default: throw(std::runtime_error("error reading raster"));
                }
            }
            return CImgToArr(self->extract_classes<double>(raster));
        } 
        PyObject* read(Chunk chunk=Chunk()) {
            // TODO - look at all bands for gain and offset
            if (!(*self)[0].is_double()) {
                switch(self->type().type()) {
                    case 1: return CImgToArr(self->read<uint8_t>(chunk));
                    case 2: return CImgToArr(self->read<uint16_t>(chunk));
                    case 3: return CImgToArr(self->read<int16_t>(chunk));
                    case 4: return CImgToArr(self->read<uint32_t>(chunk));
                    case 5: return CImgToArr(self->read<int32_t>(chunk));
                    case 6: return CImgToArr(self->read<float>(chunk));
                    case 7: return CImgToArr(self->read<double>(chunk));
                    default: throw(std::runtime_error("error reading raster"));
                }
            }
            return CImgToArr(self->read<double>(chunk));
        }
        GeoImage& write(PyObject* obj, Chunk chunk=Chunk()) {
            PyArrayObject* arr((PyArrayObject*)obj);
            switch( PyArray_TYPE(arr) ) {
                case NPY_UINT8: self->write(ArrToCImg<uint8_t>(arr), chunk); break;
                case NPY_UINT16: self->write(ArrToCImg<uint16_t>(arr), chunk); break;
                case NPY_INT16: self->write(ArrToCImg<int16_t>(arr), chunk); break;
                case NPY_UINT32: self->write(ArrToCImg<uint32_t>(arr), chunk); break;
                case NPY_INT32: self->write(ArrToCImg<int32_t>(arr), chunk); break;
                case NPY_UINT64: self->write(ArrToCImg<uint64_t>(arr), chunk); break;
                case NPY_INT64: self->write(ArrToCImg<int64_t>(arr), chunk); break;
                case NPY_FLOAT32: self->write(ArrToCImg<float>(arr), chunk); break;
                case NPY_FLOAT64: self->write(ArrToCImg<double>(arr), chunk); break;
                default:
                    throw(std::invalid_argument("a numpy array is required"));
            }
            return *self;
        }
    }
}


// GeoVectorResource
%ignore gip::GeoVectorResource::operator=;
%include "gip/GeoVectorResource.h"
namespace gip {
    %extend GeoVectorResource {
        unsigned long int __len__() {
            return self->GeoVectorResource::nfeatures();
        }
    }
}


// GeoFeature
%ignore gip::GeoFeature::operator=;
%ignore gip::GeoFeature::operator[];
%include "gip/GeoFeature.h"
namespace gip {
    %extend GeoFeature {
        GeoFeature __deepcopy__(GeoFeature feature) {
            return GeoFeature(feature);
        }
        std::string __getitem__(std::string att) {
            return self->GeoFeature::operator[](att);
        }
    }
}


// GeoVector
%ignore gip::GeoVector::operator=;
%ignore gip::GeoVector::operator[];
%template(vector_GeoFeature) std::vector< gip::GeoFeature >;
%include "gip/GeoVector.h"
namespace gip {
    %extend GeoVector {
        GeoFeature __getitem__(int index) {
            return self->GeoVector::operator[](index);
        }
        GeoFeature __getitem__(std::string val) {
            return self->GeoVector::operator[](val);
        }
        GeoVector __deepcopy__(GeoVector vector) {
            return GeoVector(vector);
        }
    }    
}
