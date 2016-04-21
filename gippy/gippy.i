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

// STL bindings
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"
namespace std {
    %template(svector) std::vector<std::string>;
    %template(ivector) std::vector<int>;
    %template() std::map<std::string, std::string>;
}

%include "exception.i"
%exception {
    try {
        $action
    } catch (const std::out_of_range& e) {
        //PyErr_SetString(PyExc_StopIteration, e.what());
        PyErr_SetString(PyExc_IndexError, e.what());
        return NULL;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
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
%include "gip/geometry.h"

%template(iPoint) gip::Point<int>;
%template(dPoint) gip::Point<double>;
%template(chvector) std::vector<Chunk>;
%template(bbvector) std::vector<BoundingBox>;

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
        %feature("docstring",
                 "PyObject returned is a numpy.array.\n"
                 "Enjoy!\n ");
        PyObject* read(Chunk chunk=Chunk()) {
            if (self->gain() == 1.0 && self->offset() == 0.0) {
                switch(self->type().type()) {
                    case 1: return CImgToArr(self->read<unsigned char>(chunk));
                    case 2: return CImgToArr(self->read<unsigned short>(chunk));
                    case 3: return CImgToArr(self->read<short>(chunk));
                    case 4: return CImgToArr(self->read<unsigned int>(chunk));
                    case 5: return CImgToArr(self->read<int>(chunk));
                    case 6: return CImgToArr(self->read<float>(chunk));
                    case 7: return CImgToArr(self->read<double>(chunk));
                    default: throw(std::exception());
                }
            }
            return CImgToArr(self->read<float>(chunk));
        }
        %feature("docstring",
                 "PyObject passed in is a numpy.array.\n"
                 "Comply!\n ");
        GeoRaster& write(PyObject* obj, Chunk chunk=Chunk()) {
            switch( PyArray_TYPE((PyArrayObject*)obj)) {
                case NPY_UINT8: self->write(ArrToCImg<unsigned char>(obj), chunk); break;
                case NPY_UINT16: self->write(ArrToCImg<unsigned short>(obj), chunk); break;
                case NPY_INT16: self->write(ArrToCImg<short>(obj), chunk); break;
                case NPY_UINT32: self->write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT32: self->write(ArrToCImg<int>(obj), chunk); break;
                case NPY_UINT64: self->write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT64: self->write(ArrToCImg<int>(obj), chunk); break;
                case NPY_FLOAT32: self->write(ArrToCImg<float>(obj), chunk); break;
                case NPY_FLOAT64: self->write(ArrToCImg<double>(obj), chunk); break;
                default:
                    throw(std::exception());
            }
            return *self;
        }
        GeoRaster& save(GeoRaster& raster) {
            return self->save<double>(raster);
        }
    }
}


// GeoImage
%ignore gip::GeoImage::operator[];
%include "gip/GeoImage.h"
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
        GeoImage save(std::string filename, std::string dtype="unknown") {
            return self->save<double>(filename, dtype);
        }
        /*GeoImage& save() {
            return self->save<double>();
        }*/
        GeoImage __deepcopy__(GeoImage image) {
            return GeoImage(image);
        }
        %feature("docstring",
                 "PyObject returned is a numpy.array.\n"
                 "Enjoy!\n ");
        PyObject* read(Chunk chunk=Chunk()) {
            // Only looks at first band for gain and offset
            if ((*self)[0].gain() == 1.0 && (*self)[0].offset() == 0.0) {
                switch(self->type().type()) {
                    case 1: return CImgToArr(self->read<unsigned char>(chunk));
                    case 2: return CImgToArr(self->read<unsigned short>(chunk));
                    case 3: return CImgToArr(self->read<short>(chunk));
                    case 4: return CImgToArr(self->read<unsigned int>(chunk));
                    case 5: return CImgToArr(self->read<int>(chunk));
                    case 6: return CImgToArr(self->read<float>(chunk));
                    case 7: return CImgToArr(self->read<double>(chunk));
                    default: throw(std::exception());
                }
            }
            return CImgToArr(self->read<float>(chunk));
        }
        GeoImage& write(PyObject* obj, Chunk chunk=Chunk()) {
            switch( PyArray_TYPE((PyArrayObject*)obj)) {
                case NPY_UINT8: self->write(ArrToCImg<unsigned char>(obj), chunk); break;
                case NPY_UINT16: self->write(ArrToCImg<unsigned short>(obj), chunk); break;
                case NPY_INT16: self->write(ArrToCImg<short>(obj), chunk); break;
                case NPY_UINT32: self->write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT32: self->write(ArrToCImg<int>(obj), chunk); break;
                case NPY_UINT64: self->write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT64: self->write(ArrToCImg<int>(obj), chunk); break;
                case NPY_FLOAT32: self->write(ArrToCImg<float>(obj), chunk); break;
                case NPY_FLOAT64: self->write(ArrToCImg<double>(obj), chunk); break;
                default:
                    throw(std::exception());
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
