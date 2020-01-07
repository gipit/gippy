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
#include <gip/GeoFeature.h>

namespace gip {
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
        //! Copy constructor
        GeoRaster(const GeoRaster& image);
        //! Assignment Operator
        GeoRaster& operator=(const GeoRaster& image);
        //! Destructor
        ~GeoRaster() {}

        //! \name File Information
        std::string basename() const { return GeoResource::basename() + "[" + to_string(_GDALRasterBand->GetBand() - 1) + "]"; }
        //! Band X Size
        unsigned int xsize() const { return _GDALRasterBand->GetXSize(); }
        //! Band Y Size
        unsigned int ysize() const { return _GDALRasterBand->GetYSize(); }
        //! Get GDALDatatype
        DataType type() const { return DataType(_GDALRasterBand->GetRasterDataType()); }
        //! Output file info
        std::string info(bool showstats=false) const;

        // Total Valid Pixels
        //unsigned int ValidSize() const { return _ValidSize; }

        //! \name Metadata functions
        //! Get Band Description
        std::string description() const {
            return _GDALRasterBand->GetDescription();
        }
        //! Get gain
        double gain() const { return _GDALRasterBand->GetScale(); }
        //! Get offset
        double offset() const { return _GDALRasterBand->GetOffset(); }
        //! Set gain
        GeoRaster& set_gain(double gain) { _GDALRasterBand->SetScale(gain); return *this; }
        //! Set offset
        GeoRaster& set_offset(double offset) { _GDALRasterBand->SetOffset(offset); return *this; }

        //! Indicates if processing will be done on reads
        bool is_processed() const { 
            return (_Functions.size() > 0) ? true : false;
        }
        //! Clears all functions from queue
        GeoRaster& clear_functions() {
            _Functions.clear();
            return *this;
        }

        //! Indicates if data should be read as doubles (functions or gain/offset)
        bool is_double() const {
            return ((is_processed()) || (gain() != 1.0) || (offset() != 0.0)) ? true : false;
        }

        //! Get NoData value
        double nodata() const {
            return _GDALRasterBand->GetNoDataValue();
        }
        //! Set No Data value
        GeoRaster& set_nodata(double val) {
            _GDALRasterBand->SetNoDataValue(val);
            return *this;
        }

        //! \name Metadata functions
        //! Get metadata item
        std::string bandmeta(std::string key) const;
        //! Get all metadata
        dictionary bandmeta() const;

        //! Set metadata item
        GeoRaster& add_bandmeta(std::string key, std::string item);
        //! Set multiple metadata items
        GeoRaster& add_bandmeta(dictionary items);

        //! Set Color Interp
        GeoRaster& set_color(std::string col) {
            _GDALRasterBand->SetDescription(col.c_str());
            to_lower(col);
            GDALColorInterp gdalcol;
            if (col == "red")
                gdalcol = GCI_RedBand;
            else if (col == "green")
                gdalcol = GCI_GreenBand;
            else if (col == "blue")
                gdalcol = GCI_BlueBand;
            else gdalcol = GCI_GrayIndex;
            _GDALRasterBand->SetColorInterpretation(gdalcol);
            return *this;
        }

        //! Add color ramp (byte image only)
        GeoRaster& add_colortable(CImg<uint8_t> color1, CImg<uint8_t> color2,
                unsigned char value1=0, unsigned char value2=255) {
            if (type().type() != 1)
                throw std::runtime_error("can only add color table to byte image");
            if ((color1.size() != 3) || (color2.size() != 3))
                throw std::invalid_argument("colors must be 3 element array");
            GDALColorTable* existing = _GDALRasterBand->GetColorTable();
            GDALColorTable ct;
            if (existing)
                ct = *(existing->Clone());
            GDALColorEntry col1, col2;
            col1.c1 = color1[0]; col1.c2 = color1[1]; col1.c3 = color1[2]; col1.c4 = 255;
            col2.c1 = color2[0]; col2.c2 = color2[1]; col2.c3 = color2[2]; col2.c4 = 255;
            ct.CreateColorRamp(value1, &col1, value2, &col2);
            _GDALRasterBand->SetColorTable(&ct);
            return *this;
        }

        //! Clear color table from image
        GeoRaster& clear_colortable() {
            _GDALRasterBand->SetColorTable(NULL);
            return *this;
        }

        //! \name Masking functions
         //! Get Saturation mask: 1's where it's saturated
        CImg<unsigned char> saturation_mask(float maxDC, Chunk chunk=Chunk()) const {
            switch (type().type()) {
                case 1: return _Mask<unsigned char>(maxDC, chunk);
                case 2: return _Mask<unsigned short>(maxDC, chunk);
                case 3: return _Mask<short>(maxDC, chunk);
                case 4: return _Mask<unsigned int>(maxDC, chunk);
                case 5: return _Mask<int>(maxDC, chunk);
                case 6: return _Mask<float>(maxDC, chunk);
                case 7: return _Mask<double>(maxDC, chunk);
                default: return _Mask<double>(maxDC, chunk);
            }
        }
        //! Adds a mask band (1 for valid), applied on read
        const GeoRaster& add_mask(const GeoRaster& band) const {
            _ValidStats = false;
            _Masks.push_back(band);
            return *this;
        }
        //! Remove all masks from band
        const GeoRaster& clear_masks() const {
            if (!_Masks.empty()) _ValidStats = false;
            _Masks.clear();
            return *this;
        }

        //! \name Morphological operators
        GeoRaster dilate(const double &w, const double &h) const {
            return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.dilate(w, h); });
        }
        //! \name Morphological operators
        GeoRaster erode(const double &w, const double &h) const {
            return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.erode(w, h); });
        }
        GeoRaster skeletonize() const {
            return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.skeletonize(); });
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
            return r.bxor(1);
        }
        GeoRaster operator<=(const double &val) const {
            GeoRaster r(*this, [=] (CImg<double>& img) -> CImg<double>& { return img.threshold(val, false, true); });
            return r.bxor(1);
        }
        //! Thresholding equality operator
        GeoRaster operator==(const double &val) const {
            return GeoRaster(*this, [=] (CImg<double>& img) -> CImg<double>& { return img == val; });
        }
        //! Bitwise XOR
        GeoRaster bxor(const double val) const {
            return GeoRaster(*this, [=] (CImg<double>& imgin) -> CImg<double>& { return imgin^=(val); });
        }

        //! \name Filter functions
        GeoRaster convolve(const CImg<double> kernel, const bool boundary=true, const bool nodata=false) const {
            if (nodata)
                return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.convolve_nodata(kernel, this->nodata()); });
            else
                return GeoRaster(*this, [=] (CImg<double>& img) ->CImg<double>& { return img.convolve(kernel, boundary); });
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


        // Statistics
        CImg<float> stats() const;
        double min() const { return (stats())[0]; }
        double max() const { return (stats())[1]; }
        double mean() const { return (stats())[2]; }
        double stddev() const { return (stats())[3]; }
    
        //! Calculate histogram with provided bins
        CImg<double> histogram(unsigned int bins=100, bool normalize=true, bool cumulative=false) const;

        //! Get value for this percentile in the cumulative distribution histogram
        double percentile(const double& p) const;

        // Scale input image range (minin, maxin) to output range (minout, maxout)
        GeoRaster scale(const double& minin, const double& maxin, const double& minout, const double& maxout) {
            // Calculate gain and offset
            double gain = (maxout-minout)/(maxin-minin);
            double offset = minout - gain*minin;
            return ((*this) * gain + offset).min(maxout).max(minout);
        }

        //! Scale image to given range (minout, maxout)
        GeoRaster autoscale(const double& minout, const double& maxout, const double& percent=0.0) {
            double minin = this->min();
            double maxin = this->max();
            if (percent > 0.0) {
                minin = percentile(percent);
                maxin = percentile(100.0 - percent);
            }
            return scale(minin, maxin, minout, maxout);
        }

        //! \name File I/O
        template<class T> CImg<T> read_raw(Chunk chunk=Chunk()) const;
        template<class T> CImg<T> read(Chunk chunk=Chunk(), bool nogainoff=false) const;
        template<class T> GeoRaster& write_raw(CImg<T> img, Chunk chunk=Chunk());
        template<class T> GeoRaster& write(CImg<T> img, Chunk chunk=Chunk());
        template<class T> GeoRaster& save(GeoRaster& raster) const;

        GeoRaster& warp_into(GeoRaster&, GeoFeature=GeoFeature(), int=0, bool=false, bool=false) const;

        //! NoData mask: 1's where it's bad data
        CImg<unsigned char> nodata_mask(Chunk chunk=Chunk()) const {
            if (!chunk.valid()) chunk = Chunk(0,0,xsize(),ysize());
            switch (type().type()) {
                case 1: return _Mask<unsigned char>(nodata(), chunk);
                case 2: return _Mask<unsigned short>(nodata(), chunk);
                case 3: return _Mask<short>(nodata(), chunk);
                case 4: return _Mask<unsigned int>(nodata(), chunk);
                case 5: return _Mask<int>(nodata(), chunk);
                case 6: return _Mask<float>(nodata(), chunk);
                case 7: return _Mask<double>(nodata(), chunk);
                default: return _Mask<double>(nodata(), chunk);
            }
        }

        CImg<unsigned char> data_mask(Chunk chunk=Chunk()) const {
            return nodata_mask(chunk)^=1;
        }

        //! Smooth/convolution (3x3) taking into account NoData
        //GeoRaster smooth(GeoRaster raster);

    protected:
        // TODO - examine why not shared pointer? (I think because it's managed by GDALDataset class)
        //! GDALRasterBand
        GDALRasterBand* _GDALRasterBand;

        //! Vector of masks to apply
        mutable std::vector< GeoRaster > _Masks;

        //! Valid Stats Flag
        mutable bool _ValidStats;
        //! Statistics
        mutable CImg<double> _Stats;

        //! List of processing functions to apply on reads (in class GeoProcess)
        //std::vector< std::function< CImg<double>& (CImg<double>&) > > _Functions;
        std::vector<func> _Functions;

    private:
        // Private constructors (callable by GeoImage friend)
        //! Default constructor - private so not callable
        GeoRaster() {}

        //! Constructor for new band
        GeoRaster(const GeoResource& georesource, int bandnum=1)
            : GeoResource(georesource), _ValidStats(false) {
            load_band(bandnum);
        }
        //! Copy with a processing function added
        GeoRaster(const GeoRaster& image, func f);

        //! Load band from GDALDataset
        void load_band(int bandnum=1) {
            // TODO - Do i need to reset GDALDataset?   Maybe it needs to be done here...
            // In practice this is called right after GDALDataset is set, so not needed
            _GDALRasterBand = _GDALDataset->GetRasterBand(bandnum);
            int pbSuccess(0);
            _GDALRasterBand->GetNoDataValue(&pbSuccess);
            // if not valid then set a nodata value
            if (pbSuccess == 0) {
                // TODO - also check for out of range value ?
                //set_nodata(type().nodata());
            }
        }

        //! inline function for creating a mask showing this value
        template<class T> inline CImg<unsigned char> _Mask(T val, Chunk chunk=Chunk()) const {
            CImg<T> img = read_raw<T>(chunk);
            CImg<unsigned char> mask(img.width(),img.height(),1,1,0);
            cimg_forXY(img,x,y) if (img(x,y) == val) mask(x,y) = 1;
            return mask;
        }

    }; //class GeoImage

    //! \name File I/O
    //! Read raw chunk given bounding box
    template<class T> CImg<T> GeoRaster::read_raw(Chunk chunk) const {
        if (!chunk.valid())
            // default to entire image
            chunk = Chunk(0,0,xsize(),ysize());
        else if (chunk.padding() > 0)
            // pad out but limit to image
            chunk = chunk.pad().intersect(Chunk(0,0,xsize(),ysize()));

        // This doesn't check for in bounds, should it?
        int w(chunk.width()), h(chunk.height());

        CImg<T> img(w, h);
        DataType dt(typeid(T));
        CPLErr err = _GDALRasterBand->RasterIO(GF_Read, chunk.x0(), chunk.y0(), w, h, 
            img.data(), w, h, dt.gdal(), 0, 0);
        if (err != CE_None) {
            std::stringstream err;
            err << "error reading " << CPLGetLastErrorMsg();
            throw std::runtime_error(err.str());
        }

        // Apply all masks TODO - cmask need to be float ?
        if (_Masks.size() > 0) {
            CImg<float> cmask(_Masks[0].read<float>(chunk));
            for (unsigned int i=1; i<_Masks.size(); i++) {
                cmask.mul(_Masks[i].read<float>(chunk));
            }
            cimg_forXY(img,x,y) {
                if (cmask(x,y) != 1) img(x,y) = nodata();
            }
        }

        return img;
    }

    //! Retrieve a piece of the image as a CImg
    template<class T> CImg<T> GeoRaster::read(Chunk chunk, bool nogainoff) const {
        auto start = std::chrono::system_clock::now();

        CImg<T> img(read_raw<T>(chunk));
        CImg<T> imgorig(img);

        bool updatenodata = false;
        // Apply gain and offset
        if ((gain() != 1.0 || offset() != 0.0) && (!nogainoff)) {
            img *= gain();
            img += offset();
            // Update NoData now so applied functions have proper NoData value set (?)
            updatenodata = true;
        }

        // Apply Processing functions
        if (_Functions.size() > 0) {
            CImg<double> imgd;
            imgd.assign(img);
            for (std::vector<func>::const_iterator iFunc=_Functions.begin();iFunc!=_Functions.end();iFunc++) {
                //if (Options::verbose() > 2 && (chunk.p0()==iPoint(0,0)))
                //    std::cout << basename() << ": Applying function " << (*iFunc) << std::endl;
                (*iFunc)(imgd);
            }
            updatenodata = true;
            img.assign(imgd);
        }

        // If processing was applied update NoData values where needed
        if (updatenodata) {
            T noDataVal = static_cast<T>(nodata());
            cimg_forXY(img,x,y) {
                T sample = imgorig(x, y);
                if (sample == noDataVal || (std::is_floating_point<T>::value && (std::isinf(sample) || std::isnan(sample))))
                    img(x,y) = noDataVal;
            }
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float> >(std::chrono::system_clock::now()-start);
        if (Options::verbose() > 3)
            std::cout << basename() << ": read " << chunk << " in " << elapsed.count() << " seconds" << std::endl;

        return img;
    }

    //! Write raw CImg to file
    template<class T> GeoRaster& GeoRaster::write_raw(CImg<T> img, Chunk chunk) {
        if (!chunk.valid())
            // default to entire image
            chunk = Chunk(0,0,xsize(),ysize());
        // Depad this if needed
        else if (chunk.padding() > 0) {
            Chunk pchunk = chunk.pad().intersect(Chunk(0,0,xsize(),ysize()));
            Point<int> p0(chunk.p0()-pchunk.p0());
            Point<int> p1 = p0 + Point<int>(chunk.width(),chunk.height());
            img.crop(p0.x(),p0.y(),p1.x(),p1.y());
        }

        if (Options::verbose() > 4) {
            std::cout << basename() << ": writing " << img.width() << " x "
                << img.height() << " image to rect " << chunk << std::endl;
        }
        CPLErr err = _GDALRasterBand->RasterIO(GF_Write, chunk.x0(), chunk.y0(),
            chunk.width(), chunk.height(), img.data(), img.width(), img.height(),
            DataType(typeid(T)).gdal(), 0, 0);
        if (err != CE_None) {
            std::stringstream err;
            err << "error writing " << CPLGetLastErrorMsg();
            throw std::runtime_error(err.str());
        }
        _ValidStats = false;
        return *this;
    }

    //! Write a Cimg to the file
    template<class T> GeoRaster& GeoRaster::write(CImg<T> img, Chunk chunk) {
        if (gain() != 1.0 || offset() != 0.0) {
            double noDataVal = nodata(); //virtual call through pointer
            double offsetVal = offset(); //virtual call through pointer
            double invGainVal = 1.0 / gain(); //virtual call through pointer
            cimg_for(img, ptr, T) { 
                double sample = static_cast<double>(*ptr);
                if (sample != noDataVal) { 
                    *ptr = static_cast<T>((sample - offsetVal) * invGainVal); 
                }
            }
        }
        if (Options::verbose() > 3 && (chunk.p0()==iPoint(0,0)))
            std::cout << basename() << ": Writing (" << gain() << "x + " << offset() << ")" << std::endl;
        return write_raw(img,chunk);
    }

    //! Process into input band "raster"
    template<class T> GeoRaster& GeoRaster::save(GeoRaster& raster) const {
        GDALRasterBand* band = raster._GDALRasterBand;
        band->SetColorInterpretation(_GDALRasterBand->GetColorInterpretation());
        band->SetMetadata(_GDALRasterBand->GetMetadata());
        raster.set_srs(this->srs());
        raster.set_affine(this->affine());
        std::vector<Chunk>::const_iterator iCh;
        std::vector<Chunk> _chunks = chunks();
        if (Options::verbose() > 3)
            std::cout << basename() << ": Processing in " << _chunks.size() << " chunks" << std::endl;
        bool nogainoff = false;
        if (this->type().string() == raster.type().string()) nogainoff = true;
        for (iCh=_chunks.begin(); iCh!=_chunks.end(); iCh++) {
                CImg<T> cimg = read<T>(*iCh, nogainoff);
                if (nodata() != raster.nodata()) {
                    cimg_for(cimg,ptr,T) { if (*ptr == nodata()) *ptr = raster.nodata(); }
                }
                raster.write(cimg,*iCh);
        }
        if (nogainoff) {
            raster.set_gain(this->gain());
            raster.set_offset(this->offset());
        }
        return raster;
    }

} // namespace GIP

#endif
