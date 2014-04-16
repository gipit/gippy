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
%module gippylib
%feature("autodoc", "1");
%{
    #define SWIG_FILE_WITH_INIT
    //#include "gip/Colors.h"
    //#include "gip/GeoData.h"
    //#include "gip/GeoRaster.h"
    #include "gip/GeoImage.h"
    #include "gip/GeoAlgorithms.h"
    //#include "gdal/gdal_priv.h"
    #include <python2.7/Python.h>
    #include <numpy/arrayobject.h>
    #include <iostream>
    #include "gip/gip_CImg.h"

    using namespace gip;

    namespace gip {
        void reg() { 
            GDALAllRegister(); 
        }
    }

    template<typename T> int numpytype() {
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
    }

    // Convert CImg into numpy array
    template<typename T> PyObject* CImgToArr(cimg_library::CImg<T> cimg) {
        int typenum;
        if (typeid(T) == typeid(unsigned char)) typenum = NPY_UINT8;
        else if (typeid(T) == typeid(char)) typenum = NPY_INT8;
        else if (typeid(T) == typeid(unsigned short)) typenum = NPY_UINT16;
        else if (typeid(T) == typeid(short)) typenum = NPY_INT16;
        else if (typeid(T) == typeid(unsigned int)) typenum = NPY_UINT32;
        else if (typeid(T) == typeid(int)) typenum = NPY_INT32;
        //else if (typeid(T) == typeid(unsigned long)) typenum = NPY_UINT64;
        //else if (typeid(T) == typeid(long)) typenum = NPY_INT64;
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

    // Convert numpy array into CImg...currently 2-D only
    template<typename T> cimg_library::CImg<T> ArrToCImg(PyObject* arr) {
        PyArrayObject* _arr = (PyArrayObject*)arr;
        cimg_library::CImg<T> cimg((T*)_arr->data, _arr->dimensions[1], _arr->dimensions[0]);
        return cimg;
    }

%}

%init %{
    // Not really sure what this does or why it's needed
    import_array();
%}

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

// CImg -> numpy
%typemap (out) cimg_library::CImg<unsigned char> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<char> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<unsigned short> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<short> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<unsigned int> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<int> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<unsigned long> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<long> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<float> { return CImgToArr($1); }
%typemap (out) cimg_library::CImg<double> { return CImgToArr($1); }

// numpy -> CImg
%typemap (in) cimg_library::CImg<unsigned char> { $1 = ArrToCImg<unsigned char>($input); }
%typemap (in) cimg_library::CImg<char> { $1 = ArrToCImg<char>($input); }
%typemap (in) cimg_library::CImg<unsigned short> { 1 = ArrToCImg<unsigned short>($input); }
%typemap (in) cimg_library::CImg<short> { $1 = ArrToCImg<short>($input); }
%typemap (in) cimg_library::CImg<unsigned int> { $1 = ArrToCImg<unsigned int>($input); }
%typemap (in) cimg_library::CImg<int> { $1 = ArrToCImg<int>($input); }
%typemap (in) cimg_library::CImg<float> { $1 = ArrToCImg<float>($input); }
%typemap (in) cimg_library::CImg<double> { $1 = ArrToCImg<double>($input); }

// TODO - Was trying to quiet warnings...didn't work
//%typemap(typecheck) PyArrayObject * = cimg_library::CImg<unsigned char> ;

// GIP functions to ignore (suppresses warnings)
// These operators are redefined below
%ignore gip::GeoData::operator=;
%ignore gip::Colors::operator[];
%ignore gip::GeoImage::operator[];
//%ignore gip::GeoRaster::operator==;

// GIP headers and classes to be wrapped
%include "gip/GeoData.h"
%include "gip/GeoRaster.h"
%include "gip/GeoImage.h"
%include "gip/GeoAlgorithms.h"
%include "gip/geometry.h"
// TODO - Not sure this really needs to be wrapped
%include "gip/Colors.h"

// TODO - improve enums.  C++0x scoped enums ?
enum GDALDataType { GDT_Unknown, GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32, GDT_Int32,
    GDT_Float32, GDT_Float64 };
    #GDT_CInt16, GDT_CInt32, GDT_CFloat32, GDT_Float64
//enum UNITS { RAW, RADIANCE, REFLECTIVITY };

namespace gip {

    // Register file formats with GDAL
    void reg();

    %template(iRect) Rect<int>;

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

    %extend Colors {
        int __getitem__(std::string col) {
            return self->Colors::operator[](col);
        }
        std::string __getitem__(int col) {
            return self->Colors::operator[](col);
        }
    }

    /*%extend Rect {
        std::string __str__() {
            return self->operator<<();
        }
    }*/

    %extend GeoRaster {
        // Processing functions
        //GeoRaster __eq__(double val) {
        //    return self->operator==(val);
        //}
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
        PyObject* Read(iRect chunk) {
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
        GeoRaster& Write(PyObject* arr, int chunk=0) {
            switch(((PyArrayObject*)arr)->descr->type_num) {
                case NPY_UINT8: self->Write(ArrToCImg<unsigned char>(arr), chunk); break;
                case NPY_UINT16: self->Write(ArrToCImg<unsigned short>(arr), chunk); break;
                case NPY_INT16: self->Write(ArrToCImg<short>(arr), chunk); break;
                case NPY_UINT32: self->Write(ArrToCImg<unsigned int>(arr), chunk); break;
                case NPY_INT32: self->Write(ArrToCImg<int>(arr), chunk); break;
                case NPY_UINT64: self->Write(ArrToCImg<unsigned int>(arr), chunk); break;
                case NPY_INT64: self->Write(ArrToCImg<int>(arr), chunk); break;
                case NPY_FLOAT32: self->Write(ArrToCImg<float>(arr), chunk); break;
                case NPY_FLOAT64: self->Write(ArrToCImg<double>(arr), chunk); break;
                default:
                    throw(std::exception());
            }
            return *self;
        }
        GeoRaster& Process(const GeoRaster& raster) {
            return self->Process<double>(raster);
        }
    }

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
        GeoImage Process(std::string filename, GDALDataType dtype=GDT_Unknown) {
            return self->Process<double>(filename, dtype);
        }
        GeoImage& Process() {
            return self->Process<double>();
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




