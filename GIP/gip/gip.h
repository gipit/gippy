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

#ifndef GIP_H
#define GIP_H

#include <gdal_priv.h>
#include <iostream>
#include <string>
#include <map>
#include <queue>

//#define cimg_debug 0
#define cimg_verbosity 1
#define cimg_display 0
#define cimg_plugin "cimg/convolve.h"
#define cimg_plugin1 "cimg/skeleton.h"
#define cimg_plugin2 "cimg/skeletonize.h"

#include <cimg/CImg.h>

/*
    Utility functions that are called only from Python (not used internally)
*/

namespace gip {
    using cimg_library::CImg;
    using cimg_library::CImgList;

    typedef std::map<std::string, std::string> dictionary;

    void init();

    template<typename T> inline void cimg_print(cimg_library::CImg<T> & img, std::string title="") {
        for (int i=0; i<img.height(); i++) {
            std::cout << "\tClass" << " " << i+1 << ": ";
            cimg_forX(img, x) std::cout << img(x,i) << " ";
            std::cout << std::endl;
        }
    }

    class Options {
    public:
        // \name Global Options (static properties)
        //! Get Config directory
        //static std::string ConfigDir() { return _ConfigDir.string(); }
        //! Set Config directory
        //static void SetConfigDir(std::string dir) { _ConfigDir = dir; }
        //! Default format when creating new files
        static std::string defaultformat() { return _DefaultFormat; }
        //! Set default format when creating new files
        static void set_defaultformat(std::string str) { _DefaultFormat = str; }
        //! Default chunk size when chunking an image
        static float chunksize() { return _ChunkSize; }
        //! Set chunk size, used when chunking an image
        static void set_chunksize(float sz) { _ChunkSize = sz; }
        //! Get verbose level
        static int verbose() { return _Verbose; }
        //! Set verbose level
        static void set_verbose(int v) { 
            _Verbose = v;
            if (v > 4) {
                // turn on GDAL output
                CPLPushErrorHandler(CPLDefaultErrorHandler);
            } else {
                CPLPushErrorHandler(CPLQuietErrorHandler);
            }
        }
        //! Get desired number of cores
        static int cores() { return _Cores; }
        //! Set desired number of cores
        static void set_cores(int n) { _Cores = n; }

    private:
        // global options
        //! Default format
        static std::string _DefaultFormat;
        //! Chunk size used when chunking up an image
        static float _ChunkSize;
        //! Verbosity level
        static int _Verbose;
        //! Number of cores to use when multi threading
        static int _Cores;
    };

}

#endif

