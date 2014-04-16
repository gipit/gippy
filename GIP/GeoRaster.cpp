#include <gip/GeoRaster.h>
#include <gip/GeoImage.h>

using namespace std;

namespace gip {

    // Copy constructor
    GeoRaster::GeoRaster(const GeoRaster& image)
        : GeoData(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), _NoData(image._NoData), 
            _ValidStats(image._ValidStats), _Stats(image._Stats), //_ValidSize(image._ValidSize),
            _minDC(image._minDC), _maxDC(image._maxDC), _Functions(image._Functions) {}

    // Copy constructor
    GeoRaster::GeoRaster(const GeoRaster& image, func f)
        : GeoData(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), _NoData(image._NoData), 
            _ValidStats(image._ValidStats), _Stats(image._Stats), //_ValidSize(image._ValidSize),
            _minDC(image._minDC), _maxDC(image._maxDC), _Functions(image._Functions) {
        //if (func.Function() != "") AddFunction(func);
        _Functions.push_back(f);
        //std::cout << Basename() << ": GeoRaster copy (" << this << ")" << std::endl;
    }

    // Assignment
    GeoRaster& GeoRaster::operator=(const GeoRaster& image) {
        // Check for self assignment
        if (this == &image) return *this;
        //_GeoData = image._GeoData;
        GeoData::operator=(image);
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
        info << "\t\tGain = " << Gain() << ", Offset = " << Offset(); //<< ", Units = " << Units();
        if (_NoData)
            info << ", NoData = " << NoDataValue() << endl;
        else info << endl;
        if (showstats) {
            cimg_library::CImg<float> stats = this->Stats();
            info << "\t\tMin = " << stats(0) << ", Max = " << stats(1) << ", Mean = " << stats(2) << " =/- " << stats(3) << endl;
        }
        if (!_Functions.empty()) info << "\t\tFunctions:" << endl;
        //for (unsigned int i=0;i<_Functions.size();i++) {
        //  info << "\t\t\t" << _Functions[i].F() << endl; //" " << _Functions[i].Operand() << endl;
        //}
        if (!_Masks.empty()) info << "\tMasks:" << endl;
        for (unsigned int i=0;i<_Masks.size();i++) info << "\t\t\t" << _Masks[i].Info() << endl;
        //_GeoImage->GetGDALDataset()->Reference(); int ref = _GeoImage->GetGDALDataset()->Dereference();
        //info << "  GDALDataset: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << endl;
        return info.str();
    }

    //! Compute stats
    cimg_library::CImg<float> GeoRaster::Stats() const {
        using cimg_library::CImg;

        if (_ValidStats) return _Stats;

        CImg<double> cimg;
        double count(0), total(0), val;
        double min(MaxValue()), max(MinValue());

        for (unsigned int iChunk=1; iChunk<=NumChunks(); iChunk++) {
            cimg = Read<double>(iChunk);
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
        for (unsigned int iChunk=1; iChunk<=NumChunks(); iChunk++) {
            cimg = Read<double>(iChunk);
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
    cimg_library::CImg<float> GeoRaster::Histogram(int bins, bool cumulative) const {
        CImg<double> cimg;
        CImg<float> stats = Stats();
        CImg<float> hist(bins,1,1,1,0);
        long numpixels(0);
        float nodata = NoDataValue();
        for (unsigned int iChunk=1; iChunk<=NumChunks(); iChunk++) {
            cimg = Read<double>(iChunk);
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

} // namespace gip
