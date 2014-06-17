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

#ifndef GIP_GEOALGORITHMS_H
#define GIP_GEOALGORITHMS_H

#include <gip/GeoImage.h>
#include <gip/GeoRaster.h>
#include <initializer_list>

namespace gip {
    //! Create cloudmask using ACCA
    GeoImage ACCA(const GeoImage&, std::string, float, float, int = 5, int = 10, int = 4000, dictionary=dictionary());

    //! Create single image from multiple input images using vector file footprint
    GeoImage CookieCutter(std::vector<std::string>, std::string, std::string, 
        float=1.0, float=1.0, bool crop=false, dictionary=dictionary());

    //! Create new file with a Fmask cloud mask
    GeoImage Fmask(const GeoImage&, std::string, int=3, int=5, dictionary=dictionary());

    //! Kmeans
    //GeoImage kmeans(const GeoImage&, std::string, int classes=5, int iterations=5, float threshold=1.0);

    //! Create indices in one pass: NDVI, EVI, LSWI, NDSI, BI {product, filename}
    dictionary Indices(const GeoImage&, dictionary, dictionary=dictionary());

    //! Stretch image into byte
    //GeoImage RGB(const GeoImage&, std::string);

    //! Spectral Matched Filter
    //GeoImage SMF(const GeoImage& image, std::string, CImg<double>);

    //! Calculate spectral correlation
    //CImg<double> SpectralCorrelation(const GeoImage&, CImg<double> covariance=CImg<double>() );

    //! Calculate spectral covariance
    //CImg<double> SpectralCovariance(const GeoImage&);

} // namespace gip

#endif // GIP_GEOALGORITHMS_H
