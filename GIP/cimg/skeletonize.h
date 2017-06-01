/*##############################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    AUTHOR: Matthew Hanson
#    EMAIL:  matt.a.hanson@gmail.com
#
#    Copyright (C) 2015 Matthew Hanson
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

#ifndef CIMG_PLUGIN_SKELETONIZE
#define CIMG_PLUGIN_SKELETONIZE

#include <cimg/skeleton.h>

/**
 * Compute skeleton of binary image 
*/
CImg<T>& skeletonize() {
  CImg<float> distance = get_distance(0);
  CImgList<floatT> grad = get_gradient("xyz");
  CImg<floatT> flux = get_flux(grad, 1, 1);

  // TODO - try Torsello correction of flux
  float thresh = 1;
  bool curve = true;
  skeleton(flux, distance, curve, thresh);
  return *this;
}

#endif // CIMG_PLUGIN_SKELETONIZE