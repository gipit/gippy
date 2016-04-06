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

#ifndef GIP_GEORASTER_H
#define GIP_GEORASTER_H

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <chrono>
#include <stdint.h>
#include <functional>

#include <gip/GeoResource.h>


namespace gip {
    typedef Rect<int> iRect;
    typedef Point<int> iPoint;

    //! Extended GDALRasterBand class
    /*!
        The GeoRaster class wraps the GDALRasterBand class
    */
    class GeoRaster : public GeoResource {
        friend class GeoImage;

    public:
        typedef std::function< CImg<double>&(CImg<double>&) > func;
        //! \name Constructors/Destructors
        //! Constructor for new band
        GeoRaster(const GeoResource& georesource, int bandnum=1)
            : GeoResource(georesource), _NoData(false), _ValidStats(false),
            _minDC(1), _maxDC(255) {
            LoadBand(bandnum);
        }
        //! Copy constructor
        GeoRaster(const GeoRaster& image);
        GeoRaster(const GeoRaster& image, func f);
        //! Assignment Operator
        GeoRaster& operator=(const GeoRaster& image);
        //! Destructor
        ~GeoRaster() {}

        //! \name File Information
        std::string Basename() const { return GeoResource::Basename() + "[" + Description() + "]"; }
        //! Band X Size
        unsigned int XSize() const { return _GDALRasterBand->GetXSize(); }
        //! Band Y Size
        unsigned int YSize() const { return _GDALRasterBand->GetYSize(); }
        //! Get GDALDatatype
        DataType Type() const { return DataType(_GDALRasterBand->GetRasterDataType()); }
        //! Output file info
        std::string Info(bool showstats=false) const;

        // Total Valid Pixels
        //unsigned int ValidSize() const { return _ValidSize; }

        //! \name Metadata functions
        //! Get Band Description
        std::string Description() const {
            return _GDALRasterBand->GetDescription();
        }
        //! Get gain
        float Gain() const { return _GDALRasterBand->GetScale(); }
        //! Get offset
        float Offset() const { return _GDALRasterBand->GetOffset(); }
        //! Set gain
        GeoRaster& SetGain(float gain) { _GDALRasterBand->SetScale(gain); return *this; }
        //! Set offset
        GeoRaster& SetOffset(float offset) { _GDALRasterBand->SetOffset(offset); return *this; }

        //! Get NoData value
        double NoData() const {
            return _GDALRasterBand->GetNoDataValue();
        }
        //! deprecated
        double NoDataValue()) const {
            std::cout << "DEPRECATION WARNING: Use NoData() instead of NoDataValue()" << std::endl;
            return NoData();
        }
        //! Set No Data value
        GeoRaster& SetNoData(double val) {
            _GDALRasterBand->SetNoDataValue(val);
            return *this;
        }

        //! Copy all meta data from another raster
        GeoRaster& CopyMeta(const GeoRaster& img) {
            SetGain(img.Gain());
            SetOffset(img.Offset());
            SetNoData(img.NoData());
            //_GDALRasterBand->SetMetadata(img._GDALRasterBand->GetMetadata());
            return *this;
        }

        //! \name Calibration and atmospheric functions
        //! Sets dyanmic range of sensor (min to max digital counts)
        void SetDynamicRange(int min, int max) {
            _minDC = min;
            _maxDC = max;
        }

        //! \name Processing functions

        //! Adds a mask band (1 for valid), applied on read
        const GeoRaster& AddMask(const GeoRaster& band) const {
            _ValidStats = false;
            _Masks.push_back(band);
            return *this;
        }
        //! Remove all masks from band
        const GeoRaster& ClearMasks() const {
            if (!_Masks.empty()) _ValidStats = false;
            _Masks.clear();
            return *this;
        }
        //! Apply a mask directly to a file (inplace)
        GeoRaster& ApplyMask(CImg<uint8_t> mask, iRect chunk=iRect());

        GeoRaster& AddFunction(func f) {
            _ValidStats = false;
            _Functions.push_back(f);
            return *this;
        }
        GeoRaster& ClearFunctions() {
            if (!_Functions.empty()) _ValidStats = false;
            _Functions.clear();
            return *this;
        }

        //! \name Processing functions
        // Logical operators
        GeoRaster operator>(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.threshold(val, false, true); });
        }
        GeoRaster operator>=(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.threshold(val, false, false); });
        }
        GeoRaster operator<(const double &val) const {
            GeoRaster r(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.threshold(val, false, false); });
            return r.BXOR(1);
        }
        GeoRaster operator<=(const double &val) const {
            GeoRaster r(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.threshold(val, false, true); });
            return r.BXOR(1);
        }
        //! Thresholding equality operator
        GeoRaster operator==(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img == val; });
        }
        //! Bitwise XOR
        GeoRaster BXOR(const double val) const {
            return GeoRaster(*this, [=] (CImg<double>& imgin) -> CImg<double>& { return imgin^=(val); });
        }

        //! \name Filter functions
        GeoRaster convolve(const CImg<double> kernel) const {
            return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.convolve_nodata(kernel, this->NoData()); });
        }
        GeoRaster laplacian() const {
            return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.laplacian(); });
        }

        // Arithmetic
        GeoRaster operator+(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img += val; });
        }
        GeoRaster operator-(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img -= val; });
        }
        GeoRaster operator*(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img *= val; });
        }
        GeoRaster operator/(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img /= val; });
        }
        //friend GeoRaster operator/(const double &val, const GeoRaster& raster) {
        //    return raster.pow(-1)*val;
        //}
        //! Pointwise max operator
        GeoRaster max(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.max(val); });
        }
        //! Pointwise min operator
        GeoRaster min(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.min(val); });
        }

        //! Exponent
        GeoRaster pow(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.pow(val); });
        }
        //! Square root
        GeoRaster sqrt() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.sqrt(); });
        }
        //! Natural logarithm
        GeoRaster log() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.log(); });
        }
        //! Log (base 10)
        GeoRaster log10() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.log10(); });

        }
        //! Exponential
        GeoRaster exp() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.exp(); });
        }

        //! Absolute value
        GeoRaster abs() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.abs(); });
        }
        //! Compute sign (-1 if < 0, +1 if > 0, 0 if 0)
        GeoRaster sign() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.sign(); });
        }

        // Cosine
        GeoRaster cos() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.cos(); });
        }
        //! Sine
        GeoRaster sin() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.sin(); });
        }
        //! Tangent
        GeoRaster tan() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.tan(); });
        }
        //! arccosine
        GeoRaster acos() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.acos(); });
        }
        //! arccosine
        GeoRaster asin() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.asin(); });
        }
        //! arctangent
        GeoRaster atan() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.atan(); });
        }
        //! Hyperbolic cosine
        GeoRaster cosh() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.cosh(); });
        }
        //! Hyperbolic sine
        GeoRaster sinh() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.sinh(); });
        }
        //! Hyperbolic tagent
        GeoRaster tanh() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.tanh(); });
        }
        //! Sinc
        GeoRaster sinc() const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.sinc(); });
        }


        // Statistics - should these be stored?
        //double Min() const { return (GetGDALStats())[0]; }
        //double Max() const { return (GetGDALStats())[1]; }
        //double Mean() const { return (GetGDALStats())[2]; }
        //double StdDev() const { return (GetGDALStats())[3]; }
        CImg<float> Stats() const;

        CImg<float> Histogram(int bins=100, bool cumulative=false) const;

        float Percentile(float p) const;

        // TODO - If RAW then can use GDAL Statistics, but compare speeds
        // Compute Statistics
        /*CImg<double> ComputeGDALStats() const {
            double min, max, mean, stddev;
            _GDALRasterBand->GetStatistics(false, true, &min, &max, &mean, &stddev);
            _GDALRasterBand->ComputeStatistics(false, &min, &max, &mean, &stddev, NULL, NULL);
            CImg<double> stats(4);
            stats(0) = min;
            stats(1) = max;
            stats(2) = mean;
            stats(3) = stddev;
            return stats;
        }*/

        //! \name File I/O
        template<class T> CImg<T> ReadRaw(iRect chunk=iRect()) const;
        template<class T> CImg<T> Read(iRect chunk=iRect()) const;
        template<class T> GeoRaster& WriteRaw(CImg<T> img, iRect chunk=iRect());
        template<class T> GeoRaster& Write(CImg<T> img, iRect chunk=iRect());
        template<class T> GeoRaster& Process(GeoRaster& raster);

         //! Get Saturation mask: 1's where it's saturated
        CImg<unsigned char> SaturationMask(iRect chunk=iRect()) const {
            switch (Type().Int()) {
                case 1: return _Mask<unsigned char>(_maxDC, chunk);
                case 2: return _Mask<unsigned short>(_maxDC, chunk);
                case 3: return _Mask<short>(_maxDC, chunk);
                case 4: return _Mask<unsigned int>(_maxDC, chunk);
                case 5: return _Mask<int>(_maxDC, chunk);
                case 6: return _Mask<float>(_maxDC, chunk);
                case 7: return _Mask<double>(_maxDC, chunk);
                default: return _Mask<double>(_maxDC, chunk);
            }
        }

        //! NoData mask: 1's where it's bad data
        CImg<unsigned char> NoDataMask(iRect chunk=iRect()) const {
            if (!chunk.valid()) chunk = iRect(0,0,XSize(),YSize());
            // if NoData not set, return all 1s
            if (!NoData()) return CImg<unsigned char>(chunk.width(),chunk.height(),1,1,0);
            switch (Type().Int()) {
                case 1: return _Mask<unsigned char>(NoData(), chunk);
                case 2: return _Mask<unsigned short>(NoData(), chunk);
                case 3: return _Mask<short>(NoData(), chunk);
                case 4: return _Mask<unsigned int>(NoData(), chunk);
                case 5: return _Mask<int>(NoData(), chunk);
                case 6: return _Mask<float>(NoData(), chunk);
                case 7: return _Mask<double>(NoData(), chunk);
                default: return _Mask<double>(NoData(), chunk);
            }
        }

        CImg<unsigned char> DataMask(iRect chunk=iRect()) const {
            return NoDataMask(chunk)^=1;
        }

        //! Smooth/convolution (3x3) taking into account NoData
        GeoRaster Smooth(GeoRaster raster) {
            CImg<double> kernel(3,3,1,1,1);
            int m0((kernel.width())/2);
            int n0((kernel.height())/2);
            int border(std::max(m0,n0));
            double total, norm;
            CImg<double> cimg0, cimg, subcimg;

            ChunkSet chunks(XSize(),YSize());
            chunks.Padding(border);
            for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
                cimg0 = Read<double>(chunks[iChunk]);
                cimg = cimg0;
                cimg_for_insideXY(cimg,x,y,border) {
                    subcimg = cimg0.get_crop(x-m0,y-n0,x+m0,y+m0);
                    total = 0;
                    norm = 0;
                    cimg_forXY(kernel,m,n) {
                        if (subcimg(m,n) != NoData()) {
                            total = total + (subcimg(m,n) * kernel(m,n));
                            norm = norm + kernel(m,n);
                        }
                    }
                    if (norm == 0)
                        cimg(x,y) = raster.NoData();
                    else
                        cimg(x,y) = total/norm;
                    if (cimg(x,y) == NoData()) cimg(x,y) = raster.NoData();
                }
                // Update nodata values in border region
                cimg_for_borderXY(cimg,x,y,border) {
                    if (cimg(x,y) == NoData()) cimg(x,y) = raster.NoData();
                }
                raster.Write(cimg, chunks[iChunk]);
            }
            return raster;
        }

    protected:
        // TODO - examine why not shared pointer? (I think because it's managed by GDALDataset class)
        //! GDALRasterBand
        GDALRasterBand* _GDALRasterBand;

        //! Vector of masks to apply
        mutable std::vector< GeoRaster > _Masks;

        //! Bool if nodata value is used
        bool _NoData;

        //! Valid Stats Flag
        mutable bool _ValidStats;
        //! Statistics
        mutable CImg<double> _Stats;

        // Constants
        int _minDC;
        int _maxDC;

        //! List of processing functions to apply on reads (in class GeoProcess)
        //std::vector< std::function< CImg<double>& (CImg<double>&) > > _Functions;
        std::vector<func> _Functions;

    private:
        //! Default constructor - private so not callable
        explicit GeoRaster() {}

        //! Load band from GDALDataset
        void LoadBand(int bandnum=1) {
            // TODO - Do i need to reset GDALDataset?   Maybe it needs to be done here...
            // In practice this is called right after GDALDataset is set, so not needed
            _GDALRasterBand = _GDALDataset->GetRasterBand(bandnum);
            int pbSuccess(0);
            _NoData = false;
            _GDALRasterBand->GetNoDataValue(&pbSuccess);
            if (pbSuccess != 0) {
                if (pbSuccess == 1) _NoData = true;
            }
            //Chunk();
        }

        template<class T> inline CImg<unsigned char> _Mask(T val, iRect chunk=iRect()) const {
            CImg<T> img = ReadRaw<T>(chunk);
            CImg<unsigned char> mask(img.width(),img.height(),1,1,0);
            cimg_forXY(img,x,y) if (img(x,y) == val) mask(x,y) = 1;
            return mask;
        }

    }; //class GeoImage

    //! \name File I/O
    //! Read raw chunk given bounding box
    template<class T> CImg<T> GeoRaster::ReadRaw(iRect chunk) const {
        if (!chunk.valid())
            chunk = Rect<int>(0,0,XSize(),YSize());
        else if (chunk.Padding() > 0)
            chunk = chunk.Pad().Intersect(Rect<int>(0,0,XSize(),YSize()));

        // This doesn't check for in bounds, should it?
        int width = chunk.x1()-chunk.x0()+1;
        int height = chunk.y1()-chunk.y0()+1;

        CImg<T> img(width, height);
        DataType dt(typeid(T));
        CPLErr err = _GDALRasterBand->RasterIO(GF_Read, chunk.x0(), chunk.y0(), width, height,
            img.data(), width, height, dt.GDALType(), 0, 0);
        if (err != CE_None) {
            std::stringstream err;
            err << "error reading " << CPLGetLastErrorMsg();
            throw std::runtime_error(err.str());
        }

        // Apply all masks TODO - cmask need to be float ?
        if (_Masks.size() > 0) {
            CImg<float> cmask(_Masks[0].Read<float>(chunk));
            for (unsigned int i=1; i<_Masks.size(); i++) {
                cmask.mul(_Masks[i].Read<float>(chunk));
            }
            cimg_forXY(img,x,y) {
                if (cmask(x,y) != 1) img(x,y) = NoData();
            }
        }

        return img;
    }

    //! Retrieve a piece of the image as a CImg
    template<class T> CImg<T> GeoRaster::Read(iRect chunk) const {
        auto start = std::chrono::system_clock::now();

        CImg<T> img(ReadRaw<T>(chunk));
        CImg<T> imgorig(img);

        bool updatenodata = false;
        // Apply gain and offset
        if (Gain() != 1.0 || Offset() != 0.0) {
            img = Gain() * (img-_minDC) + Offset();
            // Update NoData now so applied functions have proper NoData value set (?)
            if (NoData()) {
                cimg_forXY(img,x,y) {
                    if (imgorig(x,y) == NoData()) img(x,y) = NoData();
                }
            }
        }

        // Apply Processing functions
        if (_Functions.size() > 0) {
            CImg<double> imgd;
            imgd.assign(img);
            for (std::vector<func>::const_iterator iFunc=_Functions.begin();iFunc!=_Functions.end();iFunc++) {
                //if (Options::Verbose() > 2 && (chunk.p0()==iPoint(0,0)))
                //    std::cout << Basename() << ": Applying function " << (*iFunc) << std::endl;
                (*iFunc)(imgd);
            }
            updatenodata = true;
            img.assign(imgd);
        }

        // If processing was applied update NoData values where needed
        if (NoData() && updatenodata) {
            cimg_forXY(img,x,y) {
                if (imgorig(x,y) == NoData()) img(x,y) = NoData();
            }
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float> >(std::chrono::system_clock::now()-start);
        if (Options::Verbose() > 3)
            std::cout << Basename() << ": read " << chunk << " in " << elapsed.count() << " seconds" << std::endl;

        return img;
    }

    //! Write raw CImg to file
    template<class T> GeoRaster& GeoRaster::WriteRaw(CImg<T> img, iRect chunk) {
        if (!chunk.valid()) chunk = Rect<int>(0,0,XSize(),YSize());
        // Depad this if needed
        if (chunk.Padding() > 0) {
            Rect<int> pchunk = chunk.get_Pad().Intersect(Rect<int>(0,0,XSize(),YSize()));
            Point<int> p0(chunk.p0()-pchunk.p0());
            Point<int> p1 = p0 + Point<int>(chunk.width()-1,chunk.height()-1);
            img.crop(p0.x(),p0.y(),p1.x(),p1.y());
        }

        if (Options::Verbose() > 4) {
            std::cout << Basename() << ": writing " << img.width() << " x "
                << img.height() << " image to rect " << chunk << std::endl;
        }
        CPLErr err = _GDALRasterBand->RasterIO(GF_Write, chunk.x0(), chunk.y0(),
            chunk.width(), chunk.height(), img.data(), img.width(), img.height(),
            DataType(typeid(T)).GDALType(), 0, 0);
        if (err != CE_None) {
            std::stringstream err;
            err << "error writing " << CPLGetLastErrorMsg();
            throw std::runtime_error(err.str());
        }
        _ValidStats = false;
        return *this;
    }

    //! Write a Cimg to the file
    template<class T> GeoRaster& GeoRaster::Write(CImg<T> img, iRect chunk) {
        if (Gain() != 1.0 || Offset() != 0.0) {
            cimg_for(img,ptr,T) if (*ptr != NoData()) *ptr = (*ptr-Offset())/Gain();
        }
        if (Options::Verbose() > 3 && (chunk.p0()==iPoint(0,0)))
            std::cout << Basename() << ": Writing (" << Gain() << "x + " << Offset() << ")" << std::endl;
        return WriteRaw(img,chunk);
    }

    //! Process into input band "raster"
    template<class T> GeoRaster& GeoRaster::Process(GeoRaster& raster) {
        GDALRasterBand* band = raster._GDALRasterBand;
        band->SetColorInterpretation(_GDALRasterBand->GetColorInterpretation());
        band->SetMetadata(_GDALRasterBand->GetMetadata());
        raster.SetCoordinateSystem(*this);
        ChunkSet chunks(XSize(), YSize());
        if (Options::Verbose() > 3)
            std::cout << Basename() << ": Processing in " << chunks.Size() << " chunks" << std::endl;
        for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
                CImg<T> cimg = Read<T>(chunks[iChunk]);
                if (NoData() != raster.NoData()) {
                    cimg_for(cimg,ptr,T) { if (*ptr == NoData()) *ptr = raster.NoData(); }
                }
                raster.Write(cimg,chunks[iChunk]);
        }
        return *this;
    }

} // namespace GIP

#endif
