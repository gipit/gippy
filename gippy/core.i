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
// STL bindings
%include "std_string.i"
%include "std_vector.i"
%include "std_map.i"
namespace std {
    %template(vectors) std::vector<std::string>;
    %template(vectori) std::vector<int>;
    %template(mapss) std::map<std::string, std::string>;
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
        std::cout << "runtime error" << std::endl;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Ignore these standard functions
%ignore std::cout;
%ignore std::endl;
%ignore operator<<;
%ignore boost::filesystem::path;

%{
    #include <gip/gip_CImg.h>
%}
// Wrap CImg
%include "cimg.i"


// Wrap GIPS
%{
    #include <gip/gip.h>
    #include <gip/Utils.h>
    #include <gip/geometry.h>
    #include <gip/GeoImage.h>
    #include <gip/GeoImages.h>
    #include <gip/GeoVector.h>
    using namespace gip;
%}

// GIP headers and classes to be wrapped - order is important!
//  ignore directives suppress warnings, then operators are redefined through %extend
%include "gip/gip.h"

namespace gip {
    // Just wrapping basic options.
    class Options {
    public:
        //static std::string ConfigDir();
        //static void SetConfigDir(std::string dir);
        static std::string DefaultFormat();
        static void SetDefaultFormat(std::string format);
        static float ChunkSize();
        static void SetChunkSize(float sz);
        static int Verbose();
        static void SetVerbose(int v);
        static int NumCores();
        static void SetNumCores(int n);
        static std::string WorkDir();
        static void SetWorkDir(std::string workdir);
    };
}

// TODO - SWIG3 supports C++11 and scoped enums
enum GDALDataType { GDT_Unknown, GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32, GDT_Int32, GDT_Float32, GDT_Float64 };
//GDT_CInt16, GDT_CInt32, GDT_CFloat32, GDT_Float64


// Geometry
%ignore gip::Point::operator=;
%ignore gip::Rect::operator=;
%ignore gip::ChunkSet::operator=;
%ignore gip::ChunkSet::operator[];
%include "gip/geometry.h"

%template(Recti) gip::Rect<int>;
%template(Rectd) gip::Rect<double>;
%template(vectorRecti) std::vector< gip::Rect<int> >;

namespace gip {
    %extend ChunkSet {
        Rect<int> __getitem__(int index) {
            return self->ChunkSet::operator[](index);
        }
        Rect<int>& __setitem__(int index, const Rect<int>& rect) {
            self->operator[](index) = rect;
            return self->ChunkSet::operator[](index);
        }
        unsigned long int __len__() {
            return self->size();
        }        
        ChunkSet __deepcopy__(ChunkSet chunks) {
            return ChunkSet(chunks);
        }
    }
}


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
        PyObject* Read(Rect<int> chunk=Rect<int>()) {
            if (self->Gain() == 1.0 && self->Offset() == 0.0) {
                switch(self->DataType()) {
                    case 1: return CImgToArr(self->Read<unsigned char>(chunk));
                    case 2: return CImgToArr(self->Read<unsigned short>(chunk));
                    case 3: return CImgToArr(self->Read<short>(chunk));
                    case 4: return CImgToArr(self->Read<unsigned int>(chunk));
                    case 5: return CImgToArr(self->Read<int>(chunk));
                    case 6: return CImgToArr(self->Read<float>(chunk));
                    case 7: return CImgToArr(self->Read<double>(chunk));
                    default: throw(std::exception());
                }
            }
            return CImgToArr(self->Read<float>(chunk));
        }
        %feature("docstring",
                 "PyObject passed in is a numpy.array.\n"
                 "Comply!\n ");
        GeoRaster& Write(PyObject* obj, Rect<int> chunk=Rect<int>()) {
            switch( PyArray_TYPE((PyArrayObject*)obj)) {
                case NPY_UINT8: self->Write(ArrToCImg<unsigned char>(obj), chunk); break;
                case NPY_UINT16: self->Write(ArrToCImg<unsigned short>(obj), chunk); break;
                case NPY_INT16: self->Write(ArrToCImg<short>(obj), chunk); break;
                case NPY_UINT32: self->Write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT32: self->Write(ArrToCImg<int>(obj), chunk); break;
                case NPY_UINT64: self->Write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT64: self->Write(ArrToCImg<int>(obj), chunk); break;
                case NPY_FLOAT32: self->Write(ArrToCImg<float>(obj), chunk); break;
                case NPY_FLOAT64: self->Write(ArrToCImg<double>(obj), chunk); break;
                default:
                    throw(std::exception());
            }
            return *self;
        }
        GeoRaster& Process(GeoRaster& raster) {
            return self->Process<double>(raster);
        }
    }
}


// GeoImage
%ignore gip::GeoImage::operator[];
%include "gip/GeoImage.h"
namespace gip {
    %extend GeoImage {
        /*%feature("kwargs") static GeoImage New(std::string filename, const GeoImage& template=GeoImage(),
            int xsz=0, int ysz, int bands=1, GDALDataType dt=GDT_Byte) {
            if (template.Basename() != "") {
            }
        }*/
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
            return self->size();
        }
        GeoImage Process(std::string filename, GDALDataType dtype=GDT_Unknown) {
            return self->Process<double>(filename, dtype);
        }
        GeoImage& Process() {
            return self->Process<double>();
        }
        PyObject* TimeSeries(CImg<double> C, Rect<int> chunk) {
            return CImgToArr(self->TimeSeries<double>(C, chunk));
        }
        PyObject* Extract(const GeoRaster& mask) {
            return CImgToArr(self->Extract<double>(mask));
        }
        PyObject* GetRandomPixels(int NumPixels) {
            return CImgToArr(self->GetRandomPixels<double>(NumPixels));
        }
        PyObject* GetPixelClasses(int NumClasses) {
            return CImgToArr(self->GetPixelClasses<double>(NumClasses));
        }
        GeoImage __deepcopy__(GeoImage image) {
            return GeoImage(image);
        }
        %feature("docstring",
                 "PyObject returned is a numpy.array.\n"
                 "Enjoy!\n ");
        PyObject* Read(Rect<int> chunk=Rect<int>()) {
            // Only looks at first band for gain and offset
            if ((*self)[0].Gain() == 1.0 && (*self)[0].Offset() == 0.0) {
                switch(self->DataType()) {
                    case 1: return CImgToArr(self->Read<unsigned char>(chunk));
                    case 2: return CImgToArr(self->Read<unsigned short>(chunk));
                    case 3: return CImgToArr(self->Read<short>(chunk));
                    case 4: return CImgToArr(self->Read<unsigned int>(chunk));
                    case 5: return CImgToArr(self->Read<int>(chunk));
                    case 6: return CImgToArr(self->Read<float>(chunk));
                    case 7: return CImgToArr(self->Read<double>(chunk));
                    default: throw(std::exception());
                }
            }
            return CImgToArr(self->Read<float>(chunk));
        }
        GeoImage& Write(PyObject* obj, Rect<int> chunk=Rect<int>()) {
            switch( PyArray_TYPE((PyArrayObject*)obj)) {
                case NPY_UINT8: self->Write(ArrToCImg<unsigned char>(obj), chunk); break;
                case NPY_UINT16: self->Write(ArrToCImg<unsigned short>(obj), chunk); break;
                case NPY_INT16: self->Write(ArrToCImg<short>(obj), chunk); break;
                case NPY_UINT32: self->Write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT32: self->Write(ArrToCImg<int>(obj), chunk); break;
                case NPY_UINT64: self->Write(ArrToCImg<unsigned int>(obj), chunk); break;
                case NPY_INT64: self->Write(ArrToCImg<int>(obj), chunk); break;
                case NPY_FLOAT32: self->Write(ArrToCImg<float>(obj), chunk); break;
                case NPY_FLOAT64: self->Write(ArrToCImg<double>(obj), chunk); break;
                default:
                    throw(std::exception());
            }
            return *self;
        }
    }
}


// GeoImages
%ignore gip::GeoImages::operator=;
%ignore gip::GeoImages::operator[];
%include "gip/GeoImages.h"
namespace gip {
    %extend GeoImages {
        GeoImage __getitem__(int index) {
            return self->GeoImages::operator[](index);
        }
        unsigned long int __len__() {
            return self->GeoImages::size();
        }
    }
}


// GeoVectorResource
%ignore gip::GeoVectorResource::operator=;
%include "gip/GeoVectorResource.h"
namespace gip {
    %extend GeoVectorResource {
        unsigned long int __len__() {
            return self->GeoVectorResource::size();
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
        unsigned long int __len__() {
            return self->size();
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
