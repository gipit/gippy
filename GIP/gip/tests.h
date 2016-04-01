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

#ifndef GIP_TESTS_H
#define GIP_TESTS_H

#include <gip/GeoImage.h>

namespace gip {

    GeoImage test_reading(std::string filename);

    GeoImage create_test_image();

	GeoImage test_chunking(int=0, int=100);

    /*template<class T> CImg<T> _test(CImg<T> cimg) {
        //std::cout << "GIPPY CImg input/output test" << std::endl;
        //std::cout << "typeid = " << typeid(T) << std::endl;
        cimg.print();
        return cimg;
    }
    template<class T> CImg<T> _testuc(CImg<unsigned char> cimg) {
        //std::cout << "GIPPY CImg input/output test" << std::endl;
        //std::cout << "typeid = " << typeid(T) << std::endl;
        cimg.print();
        return cimg;
    }*/

} // namespace gip

#endif
