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

#include <gip/GeoRaster.h>
#include <gip/GeoImage.h>

using namespace std;

namespace gip {

    // Copy constructor
    GeoRaster::GeoRaster(const GeoRaster& image)
        : GeoResource(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), _NoData(image._NoData), 
            _ValidStats(image._ValidStats), _Stats(image._Stats),
            _minDC(image._minDC), _maxDC(image._maxDC), _Functions(image._Functions) {}

    // Copy constructor
    GeoRaster::GeoRaster(const GeoRaster& image, func f)
        : GeoResource(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), _NoData(image._NoData), 
            _ValidStats(image._ValidStats), _Stats(image._Stats),
            _minDC(image._minDC), _maxDC(image._maxDC), _Functions(image._Functions) {
        //if (func.Function() != "") AddFunction(func);
        _Functions.push_back(f);
        //std::cout << Basename() << ": GeoRaster copy (" << this << ")" << std::endl;
    }

    // Assignment
    GeoRaster& GeoRaster::operator=(const GeoRaster& image) {
        // Check for self assignment
        if (this == &image) return *this;
        GeoResource::operator=(image);
        _GDALRasterBand = image._GDALRasterBand;
        _Masks = image._Masks;
        _NoData = image._NoData;
        _ValidStats = image._ValidStats;
        _Stats = image._Stats;
        //_ValidSize = image._ValidSize;
        _minDC = image._minDC;
        _maxDC = image._maxDC;
        _Functions = image._Functions;
        //cout << _GeoImage->Basename() << ": " << ref << " references (GeoRaster Assignment)" << endl;
        return *this;
    }

    string GeoRaster::Info(bool showstats) const {
        std::stringstream info;
        //info << _GeoImage->Basename() << " - b" << _GDALRasterBand->GetBand() << ":" << endl;
        info << XSize() << " x " << YSize() << " " << DataType() << ": " << Description();
        //info << " (GeoData: " << _GDALDataset.use_count() << " " << _GDALDataset << ")";
        //info << " RasterBand &" << _GDALRasterBand << endl;
        info << "   Gain = " << Gain() << ", Offset = " << Offset(); //<< ", Units = " << Units();
        if (_NoData)
            info << ", NoData = " << NoDataValue() << endl;
        else info << endl;
        if (showstats) {
            CImg<float> stats = this->Stats();
            info << "      Min = " << stats(0) << ", Max = " << stats(1) << ", Mean = " << stats(2) << " =/- " << stats(3) << endl;
        }
        /*if (!_Functions.empty()) info << "      Functions:" << endl;
        for (unsigned int i=0;i<_Functions.size();i++) {
          info << "      " << _Functions[i] << endl; //" " << _Functions[i].Operand() << endl;
        }*/
        if (!_Masks.empty()) info << "\tMasks:" << endl;
        for (unsigned int i=0;i<_Masks.size();i++) info << "      " << _Masks[i].Info() << endl;
        //_GeoImage->GetGDALDataset()->Reference(); int ref = _GeoImage->GetGDALDataset()->Dereference();
        //info << "  GDALDataset: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << endl;
        return info.str();
    }

    //! Compute stats
    CImg<float> GeoRaster::Stats() const {
        if (_ValidStats) return _Stats;

        CImg<double> cimg;
        double count(0), total(0), val;
        double min(MaxValue()), max(MinValue());
        ChunkSet chunks(XSize(),YSize());

        for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
            cimg = Read<double>(chunks[iChunk]);
            cimg_for(cimg,ptr,double) {
                if (*ptr != NoDataValue()) {
                    total += *ptr;
                    count++;
                    if (*ptr > max) max = *ptr;
                    if (*ptr < min) min = *ptr;
                }
            }
        }
        float mean = total/count;
        total = 0;
        double total3(0);
        for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
            cimg = Read<double>(chunks[iChunk]);
            cimg_for(cimg,ptr,double) {
                if (*ptr != NoDataValue()) {
                    val = *ptr-mean;
                    total += (val*val);
                    total3 += (val*val*val);
                }
            }
        }
        float var = total/count;
        float stdev = std::sqrt(var);
        float skew = (total3/count)/std::sqrt(var*var*var);
        _Stats = CImg<float>(6,1,1,1,(float)min,(float)max,mean,stdev,skew,count);
        _ValidStats = true;

        return _Stats;
    }

    float GeoRaster::Percentile(float p) const {
        CImg<float> stats = Stats();
        unsigned int bins(100);
        CImg<float> hist = Histogram(bins,true) * 100;
        CImg<float> xaxis(bins);
        float interval( (stats(1)-stats(0))/((float)bins-1) );
        for (unsigned int i=0;i<bins;i++) xaxis[i] = stats(0) + i * interval;
        if (p == 0) return stats(0);
        if (p == 99) return stats(1);
        int ind(1);
        while(hist[ind] < p) ind++;
        float xind( (p-hist[ind-1])/(hist[ind]-hist[ind-1]) );
        return xaxis.linear_atX(ind-1+xind);
    }

    //! Compute histogram
    CImg<float> GeoRaster::Histogram(int bins, bool cumulative) const {
        CImg<double> cimg;
        CImg<float> stats = Stats();
        CImg<float> hist(bins,1,1,1,0);
        long numpixels(0);
        float nodata = NoDataValue();
        ChunkSet chunks(XSize(),YSize());
        for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
            cimg = Read<double>(chunks[iChunk]);
            cimg_for(cimg,ptr,double) {
                if (*ptr != nodata) {
                    hist[(unsigned int)( (*ptr-stats(0))*bins / (stats(1)-stats(0)) )]++;
                    numpixels++;
                }
            }
        }
        hist/=numpixels;
        if (cumulative) for (int i=1;i<bins;i++) hist[i] += hist[i-1];
        //if (Options::Verbose() > 3) hist.display_graph(0,3,1,"Pixel Value",stats(0),stats(1));
        return hist;
    }

    GeoRaster& GeoRaster::ApplyMask(CImg<uint8_t> mask, iRect chunk) {
        CImg<double> cimg = ReadRaw<double>(chunk);
        if (!mask.is_sameXY(cimg))
            throw std::runtime_error("mask wrong size for chunk " + to_string(chunk));
        cimg_forXY(cimg,x,y) if (mask(x,y) == 0) cimg(x,y) = NoDataValue();
        WriteRaw(cimg, chunk);
        return *this;
    }

} // namespace gip
