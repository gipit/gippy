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

#include <gip/GeoImage.h>
#include <gip/GeoRaster.h>

//#include <sstream>

namespace gip {
    using std::string;
    using std::vector;
    using std::endl;

    GeoImage::GeoImage(vector<string> filenames)
        : GeoResource(filenames[0]) {
        vector<string>::const_iterator f;
        load_bands();
        unsigned int b;
        for (b=0; b<nbands(); b++) {
            _BandNames[b] = basename() + (nbands() > 1 ? "-" + _BandNames[b] : "");
        }
        for (f=filenames.begin()+1; f!=filenames.end(); f++) {
            GeoImage img(*f);
            for (b=0; b<img.nbands(); b++) {
                add(img[b]);
                _BandNames[nbands()-1] = img.basename() + (img.nbands() > 1 ? "-" + img[b].description() : "");
            }
        }
    }

    // Copy constructor
    GeoImage::GeoImage(const GeoImage& image)
        : GeoResource(image) {
        for (uint i=0;i<image.nbands();i++)
            _RasterBands.push_back( image[i] );
            _BandNames = image.bandnames();
    }

    // Assignment operator
    GeoImage& GeoImage::operator=(const GeoImage& image) {
        // Check for self assignment
        if (this == &image) return *this;
        GeoResource::operator=(image);
        _RasterBands.clear();
        for (uint i=0;i<image.nbands();i++) _RasterBands.push_back( image[i] );
        _BandNames = image.bandnames();
        return *this;
    }

    string GeoImage::info(bool bandinfo, bool stats) const {
        std::stringstream info;
        info << filename() << " - " << _RasterBands.size() << " bands ("
                << xsize() << "x" << ysize() << ") " << endl;
        info << "   References: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << endl;
        info << "   Geo Coordinates (min xy): " << minxy().x() << ", " << minxy().y() << endl;
        info << "   Geo Coordinates (max xy): " << maxxy().x() << ", " << maxxy().y() << endl;
        //info << "   References - GeoImage: " << _Ref << " (&" << this << ")";
        //_GDALDataset->Reference(); int ref = _GDALDataset->Dereference();
        //info << "  GDALDataset: " << ref << " (&" << _GDALDataset << ")" << endl;
        if (bandinfo) {
            for (unsigned int i=0;i<_RasterBands.size();i++) {
                info << "   Band " << i+1 << " (" << _BandNames[i] << "): " << _RasterBands[i].info(stats);
            }
        }
        return info.str();
    }
    // Get band descriptions (not always the same as name)
    /*vector<string> GeoImage::BandDescriptions() const {
        vector<string> names;
        for (vector< GeoRaster >::const_iterator iRaster=_RasterBands.begin();iRaster!=_RasterBands.end();iRaster++) {
            names.push_back(iRaster->Description());
        }
        return names;
    }*/
    const GeoRaster& GeoImage::operator[](unsigned int index) const {
        if (index <_RasterBands.size()) {
            return _RasterBands[index];
        } else {
            throw std::out_of_range ("No band " + to_string(index));
        }        
    }

    // Band indexing
    const GeoRaster& GeoImage::operator[](string name) const {
        int index(band_index(name));
        return this->operator[](index);
    }
    // Add a band (to the end)
    GeoImage& GeoImage::add(GeoRaster band) { //, unsigned int bandnum) {
        string name = (band.description() == "") ? to_string(_RasterBands.size()+1) : band.description();
        if (band_exists(name)) {
            throw std::runtime_error("Band named " + name + " already exists in GeoImage!");
        }
        _RasterBands.push_back(band);
        _BandNames.push_back(name);
        return *this;
    }
    // Keep only these band names
    GeoImage& GeoImage::select(vector<string> names) {
        return select(Descriptions2Indices(names));
    }
    // Keep only these band numbers
    GeoImage& GeoImage::select(vector<int> nums) {
        vector<GeoRaster> _bands;
        vector<string> _names;
        vector<int> _bandnums;
        // TODO - for fun, replace with lambda function and map
        for (vector<int>::const_iterator i=nums.begin(); i!=nums.end(); i++) {
            _bands.push_back(_RasterBands[*i]);
            _names.push_back(_BandNames[*i]);
        }
        _RasterBands = _bands;
        _BandNames = _names;
        return *this;
    }


    /*const GeoImage& GeoImage::ComputeStats() const {
        for (unsigned int b=0;b<nbands();b++) _RasterBands[b].ComputeStats();
        return *this;
    }*/

    //! Load bands from dataset
    void GeoImage::load_bands() {
        vector<unsigned int> bandnums; // = _Options.Bands();
        // Check for subdatasets
        vector<string> names = this->metagroup("SUBDATASETS","_NAME=");
        unsigned int numbands(names.size());
        if (names.empty()) numbands = _GDALDataset->GetRasterCount();
        unsigned int b;
        // If no bands provided, default to all bands in this dataset
        //if (bandnums.empty()) {
        for(b=0;b<numbands;b++) bandnums.push_back(b+1);
        /* else {
            // Check for out of bounds and remove
            for(vector<unsigned int>::iterator bpos=bandnums.begin();bpos!=bandnums.end();) {
                if ((*bpos > numbands) || (*bpos < 1))
                    bpos = bandnums.erase(bpos);
                else bpos++;
            }
        }*/
        if (names.empty()) {
            // Load Bands
            for (b=0;b<bandnums.size(); b++) {
                add(GeoRaster(*this, bandnums[b]));
            }
        } else {
            // Load Subdatasets as bands, assuming 1 band/subdataset
            for(b=0;b<bandnums.size();b++) {
                _RasterBands.push_back( GeoResource(names[bandnums[b]-1],_GDALDataset->GetAccess()) );
                _BandNames.push_back(_RasterBands[b].description());
            }
            // Replace this dataset with first full frame band
            unsigned int index(0);
            for (unsigned int i=0;i<nbands();i++) {
                if (_RasterBands[i].xsize() > _RasterBands[index].xsize()) index = i;
            }
            // Release current dataset, point to new one
            _GDALDataset.reset();
            _GDALDataset = _RasterBands[index]._GDALDataset;
        }
    }

    //! Calculates spectral covariance of image
    CImg<double> GeoImage::spectral_covariance() const {
        unsigned int NumBands(nbands());

        CImg<double> covariance(NumBands, NumBands, 1, 1, 0), bandchunk, matrixchunk;        
        CImg<unsigned char> mask;
        int validsize;

        ChunkSet chset = chunks();
        for (unsigned int iChunk=0; iChunk<chset.size(); iChunk++) {
            // Bands x NumPixels
            matrixchunk = CImg<double>(NumBands, chset[iChunk].area(),1,1,0);
            mask = nodata_mask(chset[iChunk]);
            validsize = mask.size() - mask.sum();

            int p(0);
            for (unsigned int b=0;b<NumBands;b++) {
                bandchunk = (*this)[b].read<double>(chset[iChunk]);
                p = 0;
                cimg_forXY(bandchunk,x,y) {
                    if (mask(x,y)==0) matrixchunk(b,p++) = bandchunk(x,y);
                }
            }
            if (p != (int)size()) matrixchunk.crop(0,0,NumBands-1,p-1);
            covariance += (matrixchunk.get_transpose() * matrixchunk)/(validsize-1);
        }
        // Subtract Mean
        CImg<double> means(NumBands);
        for (unsigned int b=0; b<NumBands; b++) means(b) = (*this)[b].stats()[2]; //cout << "Mean b" << b << " = " << means(b) << endl; }
        covariance -= (means.get_transpose() * means);

        if (Options::verbose() > 2) {
            std::cout << basename() << " Spectral Covariance Matrix:" << endl;
            cimg_forY(covariance,y) {
                std::cout << "\t";
                cimg_forX(covariance,x) {
                    std::cout << std::setw(18) << covariance(x,y);
                }
                std::cout << std::endl;
            }
        }
        return covariance;
    }

    std::vector<int> GeoImage::Descriptions2Indices(std::vector<std::string> bands) const {
        std::vector<int> ibands;
        std::vector<int>::const_iterator b;
        if (bands.empty()) {
            // If no bands specified then defaults to all bands
            for (unsigned int c=0; c<nbands(); c++) ibands.push_back(c);
        } else {
            for (std::vector<std::string>::const_iterator name=bands.begin(); name!=bands.end(); name++) {
                ibands.push_back( band_index(*name) );
            }
        }
        return ibands;
    }


} // namespace gip
