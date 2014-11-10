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
%module gippy
%feature("autodoc", "1");
%{
    #define SWIG_FILE_WITH_INIT
    #include <gip/GeoImage.h>
    #include <gip/GeoAlgorithms.h>
    #include <python2.7/Python.h>
    #include <numpy/arrayobject.h>
    #include <iostream>
    #include <gip/gip_CImg.h>
    #include <stdint.h>

    using namespace gip;

    // Additional functions used by the SWIG interface but not used directly by users

    void gip_gdalinit() { 
       GDALAllRegister();
       CPLPushErrorHandler(CPLQuietErrorHandler);
    }

    /*template<typename T> int numpytype() {
        int typenum;
        if (typeid(T) == typeid(unsigned char)) typenum = NPY_UINT8;
        else if (typeid(T) == typeid(char)) typenum = NPY_INT8;
        else if (typeid(T) == typeid(unsigned short)) typenum = NPY_UINT16;
        else if (typeid(T) == typeid(short)) typenum = NPY_INT16;
        else if (typeid(T) == typeid(unsigned int)) typenum = NPY_UINT32;
        else if (typeid(T) == typeid(int)) typenum = NPY_INT32;
        else if (typeid(T) == typeid(float)) typenum = NPY_FLOAT32;
        else if (typeid(T) == typeid(double)) typenum = NPY_FLOAT64;
        else throw(std::exception());
        return typenum;
    }*/
    //std::vector<int> test_vectori() { return {1,2,3,4,5}; }

    // Convert CImg into numpy array
    template<typename T> PyObject* CImgToArr(CImg<T> cimg) {
        int typenum;
        if (typeid(T) == typeid(uint8_t)) typenum = NPY_UINT8;
        else if (typeid(T) == typeid(int8_t)) typenum = NPY_INT8;
        else if (typeid(T) == typeid(uint16_t)) typenum = NPY_UINT16;
        else if (typeid(T) == typeid(int16_t)) typenum = NPY_INT16;
        else if (typeid(T) == typeid(uint32_t)) typenum = NPY_UINT32;
        else if (typeid(T) == typeid(int32_t)) typenum = NPY_INT32;
        else if (typeid(T) == typeid(uint64_t)) typenum = NPY_UINT64;
        else if (typeid(T) == typeid(int64_t)) typenum = NPY_INT64;
        else if (typeid(T) == typeid(float)) typenum = NPY_FLOAT32;
        else if (typeid(T) == typeid(double)) typenum = NPY_FLOAT64;
        else throw(std::exception());

        npy_intp dims[] = { cimg.spectrum(), cimg.depth(), cimg.height(), cimg.width() };
        PyObject* arr;
        int numdim = 4;
        if (cimg.spectrum() == 1) {
            numdim = 3;
            if (cimg.depth() == 1) numdim=2;
        }
        arr = PyArray_SimpleNew(numdim, &dims[4-numdim], typenum);
        //if (dims[0] == 1)
        //    arr = PyArray_SimpleNew(numdim-1,&dims[1], typenum);
        //else arr = PyArray_SimpleNew(numdim, dims, typenum);
        void *arr_data = PyArray_DATA((PyArrayObject*)arr);
        memcpy(arr_data, cimg.data(), PyArray_ITEMSIZE((PyArrayObject*) arr) * dims[0] * dims[1] * dims[2] * dims[3]);
        return arr;
    }

    // Convert numpy array into CImg
    template<typename T> CImg<T> ArrToCImg(PyObject* obj) {
        PyArrayObject* arr = (PyArrayObject*)obj;
        if (((PyArrayObject*)arr)->nd == 1) {
            return CImg<T>((T*)arr->data, arr->dimensions[0]);
        } else if (arr->nd == 2) {
            return CImg<T>((T*)arr->data, arr->dimensions[1], arr->dimensions[0]);
        } else {
            throw(std::exception());
        }
    }

    /*
    namespace gip {
        PyObject* test(PyObject* arr) {
            //return CImgToArr(_test(ArrToCImg<float>(arr)));
            switch(((PyArrayObject*)arr)->descr->type_num) {
                case NPY_UINT8: return CImgToArr(_test(ArrToCImg<uint8_t>(arr)));
                case NPY_INT8: return CImgToArr(_test(ArrToCImg<int8_t>(arr)));
                case NPY_UINT16: return CImgToArr(_test(ArrToCImg<uint16_t>(arr)));
                case NPY_INT16: return CImgToArr(_test(ArrToCImg<int16_t>(arr)));
                case NPY_UINT32: return CImgToArr(_test(ArrToCImg<uint32_t>(arr)));
                case NPY_INT32: return CImgToArr(_test(ArrToCImg<int32_t>(arr)));
                case NPY_UINT64: return CImgToArr(_test(ArrToCImg<uint64_t>(arr)));
                case NPY_INT64: return CImgToArr(_test(ArrToCImg<int64_t>(arr)));
                case NPY_FLOAT32: return CImgToArr(_test(ArrToCImg<float>(arr)));
                case NPY_FLOAT64: return CImgToArr(_test(ArrToCImg<double>(arr)));
                default:
                    throw(std::exception());
            }
        }
    }*/

%}

%init %{
    // initialization for using numpy
    import_array();
%}

// Register file formats with GDAL
void gip_gdalinit();

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
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// Typemaps for conversions

// CImg -> numpy
%typemap (out) CImg<unsigned char> { return CImgToArr($1); }
%typemap (out) CImg<uint8_t> { return CImgToArr($1); }
%typemap (out) CImg<int8_t> { return CImgToArr($1); }
%typemap (out) CImg<uint16_t> { return CImgToArr($1); }
%typemap (out) CImg<int16_t> { return CImgToArr($1); }
%typemap (out) CImg<uint32_t> { return CImgToArr($1); }
%typemap (out) CImg<int32_t> { return CImgToArr($1); }
%typemap (out) CImg<uint64_t> { return CImgToArr($1); }
%typemap (out) CImg<int64_t> { return CImgToArr($1); }
%typemap (out) CImg<float> { return CImgToArr($1); }
%typemap (out) CImg<double> { return CImgToArr($1); }

// numpy -> CImg
%typemap (in) CImg<uint8_t> { 
    //std::cout << "uint8" << std::endl;
    $1 = ArrToCImg<uint8_t>($input); 
}
%typemap(typecheck) CImg<uint8_t> = PyObject*;
%typemap (in) CImg<int8_t> { 
    //std::cout << "int8" << std::endl;
    $1 = ArrToCImg<int8_t>($input); 
}
%typemap(typecheck) CImg<int8_t> = PyObject*;
%typemap (in) CImg<uint16_t> { 
    //std::cout << "uint16" << std::endl;
    $1 = ArrToCImg<uint16_t>($input); 
}
%typemap(typecheck) CImg<uint16_t> = PyObject*;
%typemap (in) CImg<int16_t> { 
    //std::cout << "int16" << std::endl;
    $1 = ArrToCImg<int16_t>($input); 
}
%typemap(typecheck) CImg<int16_t> = PyObject*;
%typemap (in) CImg<uint32_t> { 
    //std::cout << "uint32" << std::endl;
    $1 = ArrToCImg<uint32_t>($input); 
}
%typemap(typecheck) CImg<uint32_t> = PyObject*;
%typemap (in) CImg<int32_t> { 
    //std::cout << "int32" << std::endl;
    $1 = ArrToCImg<int32_t>($input); 
}
%typemap(typecheck) CImg<int32_t> = PyObject*;
%typemap (in) CImg<uint64_t> { 
    //std::cout << "uint64" << std::endl;
    $1 = ArrToCImg<uint64_t>($input); 
}
%typemap(typecheck) CImg<uint64_t> = PyObject*;
%typemap (in) CImg<int64_t> { 
    //std::cout << "int64" << std::endl;
    $1 = ArrToCImg<int64_t>($input); 
}
%typemap(typecheck) CImg<int64_t> = PyObject*;
%typemap (in) CImg<float> { 
    //std::cout << "float" << std::endl;
    $1 = ArrToCImg<float>($input); 
}
%typemap(typecheck) CImg<float> = PyObject*;
%typemap (in) CImg<double> { 
    //std::cout << "double" << std::endl;
    $1 = ArrToCImg<double>($input); 
}
%typemap(typecheck) CImg<double> = PyObject*;

// GIP functions to ignore (suppresses warnings) because operators are redefined below
%ignore gip::GeoData::operator=;
%ignore gip::GeoImage::operator[];
%ignore operator<<;

// GIP headers and classes to be wrapped - order is important!
%include "gip/geometry.h"
%include "gip/GeoData.h"
%include "gip/GeoRaster.h"
%include "gip/GeoImage.h"
%include "gip/GeoAlgorithms.h"

// TODO - SWIG3 supports C++11 and scoped enums
enum GDALDataType { GDT_Unknown, GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32, GDT_Int32, GDT_Float32, GDT_Float64 };
//GDT_CInt16, GDT_CInt32, GDT_CFloat32, GDT_Float64

%template(Recti) gip::Rect<int>;
%template(vectorRecti) std::vector< gip::Rect<int> >;

// Additional manual wrapping and redefinition
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
        static std::string WorkDir();
        static void SetWorkDir(std::string workdir);
    };

    %extend GeoRaster {
		%feature("docstring",
				 "PyObject returned is a numpy.array.\n"
				 "Enjoy!\n ");
        PyObject* Read(int chunk=0) {
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
        PyObject* Read(Rect<int> chunk) {
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
        GeoRaster& Write(PyObject* obj, int chunk=0) {
            PyArrayObject* arr = (PyArrayObject*)obj;
            switch(((PyArrayObject*)arr)->descr->type_num) {
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
        GeoImage Process(std::string filename, GDALDataType dtype=GDT_Unknown) {
            return self->Process<double>(filename, dtype);
        }
        GeoImage& Process() {
            return self->Process<double>();
        }
        PyObject* TimeSeries(CImg<double> C, Rect<int> chunk) {
            return CImgToArr(self->TimeSeries<double>(C, chunk));
        }
        PyObject* TimeSeries(CImg<double> C, int chunknum=0) {
            return CImgToArr(self->TimeSeries<double>(C, chunknum));
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
        PyObject* Read(int chunk=0) {
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
    }

}




