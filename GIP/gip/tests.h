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

#ifndef GIP_TESTS_H
#define GIP_TESTS_H

#include <gip/GeoImage.h>

namespace gip {

	GeoImage test_chunking(int=0, int=100);

    /*template<class T> CImg<T> _test(CImg<T> cimg) {
        //std::cout << "GIPPY CImg input/output test" << std::endl;
        //std::cout << "typeid = " << typeid(T) << std::endl;
        cimg_printinfo(cimg);
        cimg_printstats(cimg);
        cimg_print(cimg);
        return cimg;
    }
    template<class T> CImg<T> _testuc(CImg<unsigned char> cimg) {
        //std::cout << "GIPPY CImg input/output test" << std::endl;
        //std::cout << "typeid = " << typeid(T) << std::endl;
        cimg_printinfo(cimg);
        cimg_printstats(cimg);
        cimg_print(cimg);
        return cimg;
    }*/

} // namespace gip

#endif