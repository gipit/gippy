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

#ifndef CIMG_PLUGINS_H
#define CIMG_PLUGINS_H

// TEMPORARY INCLUDES (DEBUG)


    //! Thresholding equality operator
    template<typename t>
    CImg<T>& operator==(const t value) {
      cimg_for(*this,ptr,T) { if (*ptr == value) *ptr = 1; else *ptr = 0; }
      return *this;
    }

    //! Convolve ignoring nodata values
    //template<typename t>
    /*CImg<T>& convolve_nodata(double nodata) {
        CImg<double> kernel(3,3,1,1,1);
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
    }*/

#endif // CIMG_PLUGINS_H
