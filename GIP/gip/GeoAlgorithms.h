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

    //! Create a mask of NoData values
    //GeoRaster CreateMask(const GeoImage&, std::string);

    //! Stretch image into byte
    //GeoImage RGB(const GeoImage&, std::string);

    //! Create single image from multiple input images using vector file footprint
    GeoImage CookieCutter(std::vector<std::string>, std::string, std::string, float=1.0, float=1.0);

    //! Create indices in one pass: NDVI, EVI, LSWI, NDSI, BI {product, filename}
    std::map<std::string, std::string> Indices(const GeoImage&, std::map<std::string, std::string>);

    //! Create cloudmask using ACCA
    GeoImage ACCA(const GeoImage&, std::string, float, float, int = 5, int = 10, int = 4000);

    //! Create new file with a Fmask cloud mask
    GeoImage Fmask(const GeoImage&, std::string, int=3, int=5);

    GeoImage RiceDetect(const GeoImage& img, std::string filename, std::vector<int> days,
        float th0, float th1, int dth0=90, int dth1=120);

    // Create new file with AutoCloud algorithm
    //GeoImage AutoCloud(const GeoImage&, std::string, int=4000, float=0.2, float=14, float=0.2, int=20);

    //! Rescale indices (between lo and hi) to between 0 and 1
    //GeoImage Index2Probability(const GeoImage&, std::string, float, float);

    //! Kmeans
    //GeoImage kmeans(const GeoImage&, std::string, int classes=5, int iterations=5, float threshold=1.0);

    //GeoImage BandMath(const GeoImage&, std::string, int, int);
    //! Calculate spectral correlation
    //CImg<double> SpectralCorrelation(const GeoImage&, CImg<double> covariance=CImg<double>() );
    //! Calculate spectral covariance
    //CImg<double> SpectralCovariance(const GeoImage&);
    //! Spectral Matched Filter
    //GeoImage SMF(const GeoImage& image, std::string, CImg<double>);

} // namespace gip

#endif // GIP_GEOALGORITHMS_H
