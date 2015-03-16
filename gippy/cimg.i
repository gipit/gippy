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
%{
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <python2.7/Python.h>
    #include <numpy/arrayobject.h>
    #include <iostream>
    #include <CImg.h>
    #include <stdint.h>

    using cimg_library::CImg;

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
        else throw(std::runtime_error("Error converting CImg to numpy array"));

        npy_intp dims[] = { cimg.spectrum(), cimg.depth(), cimg.height(), cimg.width() };
        PyObject* arr;
        int numdim = 4;
        if (cimg.spectrum() == 1) {
            numdim = 3;
            if (cimg.depth() == 1) {
                numdim=2;
                if (cimg.height() == 1) {
                    numdim=1;
                }
            }
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
        T* data((T*)PyArray_DATA((PyArrayObject*)obj));
        npy_intp* dims(PyArray_DIMS((PyArrayObject*)obj));
        int numdim(PyArray_NDIM((PyArrayObject*)obj));
        if (numdim == 1) {
            return CImg<T>(data, dims[0]);
        } else if (numdim == 2) {
            return CImg<T>(data, dims[1], dims[0]);
        } else if (numdim == 3) {
            return CImg<T>(data, dims[2], dims[1], dims[0]);
        } else if (numdim == 4) {
            return CImg<T>(data, dims[3], dims[2], dims[1], dims[0]);
        } else {
            throw(std::runtime_error("Error converting numpy array to CImg"));
        }
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
        else throw(std::runtime_error());
        return typenum;
    }*/
    //std::vector<int> test_vectori() { return {1,2,3,4,5}; }

    /*
    namespace gip {
        PyObject* test(PyObject* obj) {
            //return CImgToArr(_test(ArrToCImg<float>(obj)));
            switch( PyArray_TYPE((PyArrayObject*)obj)) {
                case NPY_UINT8: return CImgToArr(_test(ArrToCImg<uint8_t>(obj)));
                case NPY_INT8: return CImgToArr(_test(ArrToCImg<int8_t>(obj)));
                case NPY_UINT16: return CImgToArr(_test(ArrToCImg<uint16_t>(obj)));
                case NPY_INT16: return CImgToArr(_test(ArrToCImg<int16_t>(obj)));
                case NPY_UINT32: returnedrn CImgToArr(_test(ArrToCImg<uint32_t>(obj)));
                case NPY_INT32: return CImgToArr(_test(ArrToCImg<int32_t>(obj)));
                case NPY_UINT64: return CImgToArr(_test(ArrToCImg<uint64_t>(obj)));
                case NPY_INT64: return CImgToArr(_test(ArrToCImg<int64_t>(obj)));
                case NPY_FLOAT32: return CImgToArr(_test(ArrToCImg<float>(obj)));
                case NPY_FLOAT64: return CImgToArr(_test(ArrToCImg<double>(obj)));
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
