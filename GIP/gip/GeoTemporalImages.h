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

#ifndef GIP_GEOIMAGES_H
#define GIP_GEOIMAGES_H

#include <gip/GeoImage.h>
#include <stdint.h>

namespace gip {
    //! GeoImages class
    /*!
        The GeoImage is a collection of GeoImage objects
    */
    class GeoImages {
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoImages() {};
        //! Create collection with GeoImage objects
        explicit GeoImages(std::vector< GeoImage > imgs) 
            : _GeoImages(imgs) {
            Validate();
        }
        //! Open file from vector of individual files
        explicit GeoImages(std::vector<std::string> filenames) {
            foreach(filenames.begin(); filenames.end(); []() { _GeoImages.push_back()}; )
            Validate();
        }

        //! Copy constructor - copies GeoResource and all bands
        GeoImages(const GeoImages& images);
        //! Assignment Operator
        GeoImages& operator=(const GeoImages& images) ;
        //! Destructor
        ~GeoImages() { _GeoImages.clear(); }

        //! \name File Information
        //! Number of bands
        unsigned int NumImages() const { return _GeoImages.size(); }

        //! \name Bands and colors
        //! Get vector of band names
        std::vector<std::string> Times() const { return _Times; }
        //! Set times for images
        void SetTimes(vector<float> times) {
            _Times = times;
        }

        //! Get band index for provided time
        int ImageIndex(float time) const {
            for (unsigned int i=0; i<_Times.size(); i++) {
                if (time == _Times[i]) return i;
            }
            return -1;
        }

        //! Get image (0-based index)
        GeoImage& operator[](int num) { return _GeoImages[num]; }
        //! Get image, const version
        const GeoImage& operator[](int num) const { return _GeoImages[num]; }

        //! Return bandnum from all images as a GeoImage
        GeoImage& AsGeoImage(int bandnum) const {

        }

        //! Adds a mask band (1 for valid) to every band in image
        GeoImages& AddMask(const GeoRaster& band) {
            for (unsigned int i=0;i<_GeoImages.size();i++) _GeoImages[i].AddMask(band);
            return *this;
        }
        //! Clear all masks
        void ClearMasks() { for (unsigned int i=0;i<_GeoImages.size();i++) _GeoImages[i].ClearMasks(); }
        //! Apply a mask directly to a file
        GeoImage& ApplyMask(CImg<uint8_t> mask, int chunk=0) {
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].ApplyMask(mask, chunk);
            return *this;
        }

        //! \name File I/O
        //! Read raw chunk, across all bands
        template<class T> CImg<T> ReadRaw(int chunk=0) const { //, bool RAW=false) const {
            CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->ReadRaw<T>(chunk) );
            }
            //return images.get_append('c','p');
            return images.get_append('v','p');
        }
        //! Read chunk, across all bands
        template<class T> CImg<T> Read(int chunk=0) const { //, bool RAW=false) const {
            CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->Read<T>(chunk) );
            }
            //return images.get_append('c','p');
            return images.get_append('v','p');
        }

        //! Write cube across all bands
        template<class T> GeoImage& Write(const CImg<T> img, int chunk=0) { //, bool BadValCheck=false) {
            typename std::vector< GeoRaster >::iterator iBand;
            int i(0);
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                CImg<T> tmp = img.get_channel(i++);
                iBand->Write(tmp, chunk); //, BadValCheck);
            }
            return *this;
        }
        // Read Cube as list
        template<class T> CImgList<T> ReadAsList(int chunk=0) const {
            CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->Read<T>(chunk) );
            }
            return images;
        }

        //! Extract, and interpolate, time series (C is time axis)
        // TODO - times can be a fixed datatype CImg
        template<class T, class t> CImg<T> TimeSeries(int chunknum=0) {
            CImg<T> cimg = Read<T>(chunknum);
            T nodata = _RasterBands[0].NoDataValue();
            if (cimg.spectrum() > 2) {
                int lowi, highi;
                float y0, y1, x0, x1;
                for (int c=1; c<cimg.spectrum()-1;c++) {
                    cimg_forXY(cimg,x,y) {
                        if (cimg(x,y,c) == nodata) {
                            // Find next lowest point
                            lowi = highi = 1;
                            while ((cimg(x,y,c-lowi) == nodata) && (lowi<c)) lowi++;
                            while ((cimg(x,y,c+highi) == nodata) && (c+highi < cimg.spectrum()-1) ) highi++;
                            y0 = cimg(x,y,c-lowi);
                            y1 = cimg(x,y,c+highi);
                            x0 = _Times[c-lowi];
                            x1 = _Times[c+highi];
                            if ((y0 != nodata) && (y1 != nodata)) {
                                cimg(x,y,c) = y0 + (y1-y0) * ((_Times[c]-x0)/(x1-x0));
                            }
                        } else if (cimg(x,y,c-1) == nodata) {
                            T val = cimg(x,y,c);
                            for (int i=c-1; i>=0; i--) {
                                if (cimg(x,y,i) == nodata) cimg(x,y,i) = val;
                            }
                        }
                    }
                }
            }
            return cimg;
        }

    protected:
        //! Vector of raster bands
        std::vector< GeoImage > _GeoImages;
        //! Vector of GeoImage times
        std::vector< float > _Times;

        //! Checks that all GeoImage objects in collection are compatible (bands, sizes)
        void Validate();

        // Convert vector of band descriptions to band indices
        std::vector<int> Times2Indices(std::vector<std::string> bands) const {
            std::vector<int> ibands;
            std::vector<int>::const_iterator b;
            if (bands.empty()) {
                // If no bands specified then defaults to all bands
                for (unsigned int c=0; c<NumBands(); c++) ibands.push_back(c);
            } else {
                for (std::vector<std::string>::const_iterator name=bands.begin(); name!=bands.end(); name++) {
                    ibands.push_back( BandIndex(*name) );
                }
            }
            return ibands;
        }

    }; // class GeoImage

} // namespace gip

#endif
