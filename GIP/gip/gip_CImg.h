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

#ifndef GIPCIMG_H
#define GIPCIMG_H

//#define cimg_debug 0
#define cimg_verbosity 1
#define cimg_display 0
#define cimg_plugin "CImg_plugins.h"

#include <CImg_152.h>
#include <iostream>

using cimg_library::CImg;
using cimg_library::CImgList;

namespace gip {

    template<typename T> inline void cimg_printclasses(cimg_library::CImg<T> img, std::string prefix="Class") {
        for (int i=0; i<img.height(); i++) {
            std::cout << "\t" << prefix << " " << i+1 << ": ";
            cimg_forX(img,x) std::cout << img(x,i) << "  ";
            std::cout << std::endl;
        }
        return;
    }

    template<typename T> inline void cimg_printstats(cimg_library::CImg<T> img, std::string note="") {
        if (note != "") std::cout << note << "  ";
        CImg<float> stats = img.get_stats();
        std::cout << "Min/Max = " << stats(0) << ", " << stats(1)
            << " Mean/StdDev = " << stats(2) << " +/- " << stats(3)
            << " Sum = " << img.sum()
            << " NumPixels = " << img.size() << std::endl;
    }

    template<typename T> inline void cimg_printinfo(cimg_library::CImg<T> img, std::string note="") {
        std::cout << note << ": " << " (" << img.width() << " x " << img.height() 
            << " x " << img.depth() << " x " << img.spectrum() << "): " << std::endl;
    }    

    template<typename T> inline void cimg_print(cimg_library::CImg<T> img, std::string note="") {
        std::cout << note;
        cimg_for(img,ptr,T) std::cout << *ptr << "  ";
        std::cout << std::endl;
    }

}

#endif
