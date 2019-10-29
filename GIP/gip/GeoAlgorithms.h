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

#ifndef GIP_GEOALGORITHMS_H
#define GIP_GEOALGORITHMS_H

#include <gip/GeoImage.h>
#include <gip/GeoRaster.h>
#include <gip/GeoVector.h>
#include <initializer_list>

namespace gip {
    namespace algorithms {

    /* these are not currently used
    std::map< std::string, std::vector<std::string> > RequiredBands = {
        {"acca", {"RED","GREEN","NIR","SWIR1","LWIR"}},
        {"truecolor", {"RED","GREEN","BLUE"}},
        {"fmask", {"BLUE", "RED", "GREEN", "NIR", "SWIR1", "SWIR2", "LWIR"}},
        // Indices
        {"ndvi", {"NIR","RED"}},
        {"evi", {"NIR","RED","BLUE"}},
        {"lswi", {"NIR","SWIR1"}},
        {"ndsi", {"SWIR1","GREEN"}},
        {"ndwi", {"GREEN","NIR"}},
        {"bi", {"BLUE","NIR"}},
        {"satvi", {"SWIR1","RED", "SWIR2"}},
        {"msavi2", {"NIR","RED"}},
        {"vari", {"RED","GREEN","BLUE"}},
        {"brgt", {"RED","GREEN","BLUE","NIR"}},
    }; */

    //! Create cloudmask using ACCA
    GeoImage acca(const GeoImage&, std::string filename, float, float, int = 5, int = 10, int = 4000);

    //! Create new file with a Fmask cloud mask
    GeoImage fmask(const GeoImage& geoimg, std::string filename, int=3, int=5);

    //! Create single image from multiple input images using vector file footprint
    GeoImage cookie_cutter(const std::vector<GeoImage>& geoimgs, std::string filename="",
        GeoFeature feature=GeoFeature(), bool crop=false, std::string proj="",
        float xres=1.0, float yres=1.0, int interpolation=0, dictionary options=dictionary(),
        bool alltouch=false);

    //! Kmeans
    GeoImage kmeans(const GeoImage&, std::string, unsigned int classes=5, unsigned int iterations=5,
                    float threshold=1.0, unsigned int num_random=500);

    //! Create indices in one pass: NDVI, EVI, LSWI, NDSI, BI {product, filename}
    GeoImage indices(const GeoImage& geoimg, std::vector<std::string> products, std::string filename="");

    //! Create output based on linear combinations of input
    GeoImage linear_transform(const GeoImage& geoimg, CImg<float> coef, std::string filename);

    //! Pansharpen all bands in a GeoImage with a pan band
    GeoImage pansharp_brovey(const GeoImage& geoimg, const GeoImage& panimg,
                             CImg<float> weights=CImg<float>(), std::string filename="");

    //! Runs the RX Detector (RXD) anamoly detection algorithm
    GeoImage rxd(const GeoImage& geoimg, std::string filename="");

    //! Calculate spectral statistics and output to new image
    GeoImage spectral_statistics(const GeoImage&, std::string filename="");

    //! Spectral Matched Filter
    //GeoImage SMF(const GeoImage& image, std::string, CImg<double>);

    //! Get a number of pixel vectors that are spectrally distant from each other
    // TODO - review this function for generality, maybe specific to kmeans?
    template<class T> CImg<T> get_random_classes(const GeoImage img, int num_classes, int num_random=1000) {
        if (Options::verbose()) {
            std::cout << img.basename() << ": get " << num_random << " random pixels" << std::endl;
        }
        CImg<T> stats;
        CImg<T> ClassMeans(img.nbands(), num_classes);
        // Get Random Pixels
        CImg<T> RandomPixels = img.read_random_pixels<T>(num_random);
        // First pixel becomes first class
        cimg_forX(ClassMeans,x) ClassMeans(x,0) = RandomPixels(x,0);
        for (int i=1; i<num_classes; i++) {
            CImg<T> ThisClass = ClassMeans.get_row(i-1);
            long validpixels = 0;
            CImg<T> Dist(RandomPixels.height());
            for (long j=0; j<RandomPixels.height(); j++) {
                // Get current pixel vector
                CImg<T> ThisPixel = RandomPixels.get_row(j);
                // Find distance to last class
                Dist(j) = ThisPixel.sum() ? (ThisPixel-ThisClass).dot( (ThisPixel-ThisClass).transpose() ) : 0;
                if (Dist(j) != 0) validpixels++;
            }
            stats = Dist.get_stats();
            // The pixel farthest away from last class make the new class
            cimg_forX(ClassMeans,x) ClassMeans(x,i) = RandomPixels(x,stats(8));
            // Toss a bunch of pixels away (make zero)
            CImg<T> DistSort = Dist.get_sort();
            T cutoff = DistSort[num_random*i]; //(stats.max-stats.min)/10 + stats.min;
            cimg_forX(Dist,x) if (Dist(x) < cutoff) cimg_forX(RandomPixels,x1) RandomPixels(x1,x) = 0;
        }
        return ClassMeans;
    }


    }
} // namespace gip

#endif // GIP_GEOALGORITHMS_H
