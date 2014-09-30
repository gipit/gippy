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

#ifndef GIPCIMG_H
#define GIPCIMG_H

//#define cimg_debug 0
#define cimg_verbosity 1
#define cimg_display 0
#define cimg_plugin "CImg_plugins.h"

#include <CImg_152.h>
#include <iostream>

namespace gip {

    using cimg_library::CImg;
    using cimg_library::CImgList;

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
