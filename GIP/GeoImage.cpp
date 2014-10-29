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

#include <gip/GeoImage.h>
#include <gip/GeoRaster.h>

//#include <sstream>

namespace gip {
    using std::string;
    using std::vector;

    GeoImage::GeoImage(vector<string> filenames)
        : GeoData(filenames[0]) {
        std::vector<std::string>::const_iterator f;
        LoadBands();
        unsigned int b;
        for (b=0; b<NumBands(); b++) {
            _BandNames[b] = Basename() + (NumBands() > 1 ? "-" + _BandNames[b] : "");
        }
        for (f=filenames.begin()+1; f!=filenames.end(); f++) {
            GeoImage img(*f);
            for (b=0; b<img.NumBands(); b++) {
                AddBand(img[b]);
                _BandNames[NumBands()-1] = img.Basename() + (img.NumBands() > 1 ? "-" + img[b].Description() : "");
            }
        }
    }

    // Copy constructor
    GeoImage::GeoImage(const GeoImage& image)
        : GeoData(image) {
        for (uint i=0;i<image.NumBands();i++)
            _RasterBands.push_back( image[i] );
            _BandNames = image.BandNames();
    }

    // Assignment operator
    GeoImage& GeoImage::operator=(const GeoImage& image) {
        // Check for self assignment
        if (this == &image) return *this;
        GeoData::operator=(image);
        _RasterBands.clear();
        for (uint i=0;i<image.NumBands();i++) _RasterBands.push_back( image[i] );
        _BandNames = image.BandNames();
        return *this;
    }

    string GeoImage::Info(bool bandinfo, bool stats) const {
        std::stringstream info;
        info << Filename() << " - " << _RasterBands.size() << " bands ("
                << XSize() << "x" << YSize() << ") " << std::endl;
        info << "   GeoData References: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << std::endl;
        info << "   Geo Coordinates (top left): " << TopLeft().x() << ", " << TopLeft().y() << std::endl;
        info << "   Geo Coordinates (lower right): " << LowerRight().x() << ", " << LowerRight().y() << std::endl;
        //info << "   References - GeoImage: " << _Ref << " (&" << this << ")";
        //_GDALDataset->Reference(); int ref = _GDALDataset->Dereference();
        //info << "  GDALDataset: " << ref << " (&" << _GDALDataset << ")" << endl;
        if (bandinfo) {
            for (unsigned int i=0;i<_RasterBands.size();i++) {
                info << "   Band " << i+1 << " (" << _BandNames[i] << "): " << _RasterBands[i].Info(stats);
            }
        }
        return info.str();
    }
    // Get band descriptions (not always the same as name)
    /*vector<string> GeoImage::BandDescriptions() const {
        std::vector<string> names;
        for (std::vector< GeoRaster >::const_iterator iRaster=_RasterBands.begin();iRaster!=_RasterBands.end();iRaster++) {
            names.push_back(iRaster->Description());
        }
        return names;
    }*/
    // Band indexing
    const GeoRaster& GeoImage::operator[](std::string name) const {
        int index(BandIndex(name));
        if (index >= 0) {
            return _RasterBands[index];
        } else {
            throw std::out_of_range ("No band named " + name);
        }
    }
    // Add a band (to the end)
    GeoImage& GeoImage::AddBand(GeoRaster band) { //, unsigned int bandnum) {
        std::string name = (band.Description() == "") ? to_string(_RasterBands.size()+1) : band.Description();
        if (BandExists(name)) {
            throw std::runtime_error("Band named " + name + " already exists in GeoImage!");
        }
        _RasterBands.push_back(band);
        _BandNames.push_back(name);
        return *this;
    }
    // Remove a band
    GeoImage& GeoImage::RemoveBand(unsigned int bandnum) {
        if (bandnum <= _RasterBands.size()) {
            _RasterBands.erase(_RasterBands.begin()+bandnum-1);
            _BandNames.erase(_BandNames.begin()+bandnum-1);
        }
        return *this;
    }
    // Remove bands except given band names
    GeoImage& GeoImage::PruneBands(vector<string> names) {
        bool keep;
        vector<int> inds = Descriptions2Indices(names);
        for (int b=NumBands(); b>=0; b--) {
            keep = false;
            for (vector<int>::const_iterator i=inds.begin(); i!=inds.end(); i++) if (*i == b) keep = true;
            if (!keep) RemoveBand(b+1);
        }
        return *this;
    }

    // Replaces all Inf or NaN pixels with NoDataValue
    GeoImage& GeoImage::FixBadPixels() {
        typedef float T;
        for (unsigned int b=0;b<NumBands();b++) {
            for (unsigned int iChunk=1; iChunk<=NumChunks(); iChunk++) {
                CImg<T> img = (*this)[b].ReadRaw<T>(iChunk);
                T nodata = (*this)[b].NoDataValue();
                cimg_for(img,ptr,T) if ( std::isinf(*ptr) || std::isnan(*ptr) ) *ptr = nodata;
                (*this)[b].WriteRaw(img,iChunk);
            }
        }
        return *this;
    }

    /*const GeoImage& GeoImage::ComputeStats() const {
        for (unsigned int b=0;b<NumBands();b++) _RasterBands[b].ComputeStats();
        return *this;
    }*/

    //! Load bands from dataset
    void GeoImage::LoadBands() {
        vector<unsigned int> bandnums; // = _Options.Bands();
        // Check for subdatasets
        vector<string> names = this->GetMetaGroup("SUBDATASETS","_NAME=");
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
                AddBand(GeoRaster(*this, bandnums[b]));
            }
        } else {
            // Load Subdatasets as bands, assuming 1 band/subdataset
            for(b=0;b<bandnums.size();b++) {
                _RasterBands.push_back( GeoData(names[bandnums[b]-1],_GDALDataset->GetAccess()) );
                _BandNames.push_back(_RasterBands[b].Description());
            }
            // Replace this dataset with first full frame band
            unsigned int index(0);
            for (unsigned int i=0;i<NumBands();i++) {
                if (_RasterBands[i].XSize() > _RasterBands[index].XSize()) index = i;
            }
            // Release current dataset, point to new one
            _GDALDataset.reset();
            _GDALDataset = _RasterBands[index]._GDALDataset;
        }
        Chunk();
    }

} // namespace gip
