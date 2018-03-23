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

#ifndef CIMG_PLUGINS_H
#define CIMG_PLUGINS_H

//! Thresholding equality operator
template<typename t>
CImg<T>& operator==(const t value) {
  cimg_for(*this,ptr,T) { if (*ptr == value) *ptr = 1; else *ptr = 0; }
  return *this;
}

/*
const CImg<T>& print_elements(std::string title="") {
    if (title != "") std::cerr << title << std::endl;
    cimg_for(*this, ptr, T) std::cerr << *ptr << " ";
    std::cerr << std::endl;
}
*/


//! Convolve ignoring nodata values
//template<typename t>
CImg<T>& convolve_nodata(CImg<double> kernel, double nodata) {
    int m0((kernel.width())/2);
    int n0((kernel.height())/2);
    int border(std::max(m0,n0));
    int shiftx, shifty;
    double val, total, norm;
    bool valid;
    cimg_for_insideXY(*this,x,y,border) {
        total = 0;
        norm = 0;
        valid = false;
        cimg_forXY(kernel,m,n) {
            shiftx = m - m0;
            shifty = n - n0;
            val = (*this)(x+shiftx,y+shifty);
            if (val != nodata) {
                total = total + (val * kernel(m,n));
                norm = norm + kernel(m,n);
                valid = true;
                //std::cout << "x, y " << x << ", " << y << std::endl;
                //std::cout << "shift " << shiftx << ", " << shifty << std::endl;
                //std::cout << "val " << val << std::endl;
                //std::cout << "total, norm " << total << ", " << norm << std::endl;
            }
        }
        if (valid)
            (*this)(x,y) = total/norm;
        else
            (*this)(x,y) = nodata;
    }
    return *this;
}

#endif // CIMG_PLUGINS_H
