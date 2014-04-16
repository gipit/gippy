#ifndef GIP_GEOIMAGE_H
#define GIP_GEOIMAGE_H

#include <gip/Colors.h>
#include <gip/GeoData.h>
#include <gip/GeoRaster.h>

namespace gip {
    // Forward declaration
    class GeoRaster;

    //! GeoImage class
    /*!
        The GeoImage is a collection of GeoRaster objects
    */
    class GeoImage : public GeoData {
        // friend so GeoRaster can access GeoImage freely
        //friend class GeoRaster;
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoImage() : GeoData() {}
        //! Open file constructor
        explicit GeoImage(std::string filename, bool update=true)
            : GeoData(filename, update) {
            LoadBands();
        }
        //! Open file from vector of individual files
        explicit GeoImage(std::vector<std::string> filenames);
        //! Constructor for creating new file
        explicit GeoImage(std::string filename, int xsz, int ysz, int bsz, GDALDataType datatype=GDT_Byte) :
            GeoData(xsz, ysz, bsz, datatype, filename) {
            LoadBands();
        }
        //! Constructor for creating new file with same properties (xsize, ysize, metadata) as existing file
        explicit GeoImage(std::string filename, const GeoImage& image, GDALDataType datatype, int bsz) :
            GeoData(image.XSize(), image.YSize(), bsz, datatype, filename) {
            //if (datatype == GDT_Unknown) datatype = image->GetDataType();
            //CopyMeta(image);
            CopyCoordinateSystem(image);
            LoadBands();
            //_Colors = image.GetColors();
        }
        //! Constructor for creating new file with same properties (xsize, ysize, bsize) as existing file
        explicit GeoImage(std::string filename, const GeoImage& image, GDALDataType datatype) :
            GeoData(image.XSize(), image.YSize(), image.NumBands(), datatype, filename) {
            //if (datatype == GDT_Unknown) datatype = image->GetDataType();
            //CopyMeta(image);
            CopyCoordinateSystem(image);
            LoadBands();
            _Colors = image.GetColors();
        }
        //! Constructor for creating new file with given properties (xsize, ysize, bsize,datatype) as existing file
        explicit GeoImage(std::string filename, const GeoImage& image) :
            GeoData(image.XSize(), image.YSize(), image.NumBands(), image.DataType(), filename) {
            //if (datatype == GDT_Unknown) datatype = image->GetDataType();
            //CopyMeta(image);
            CopyCoordinateSystem(image);
            LoadBands();
            //_Colors = image.GetColors();
        }

        // Factory functions to support keywords in python bindings
        /*static GeoImage Open(std::string filename, bool update=true) {
            return GeoImage(filename, update);
        }*/

        //static GeoImage New(std::string filename, const GeoImage& template=GeoImage(), int xsz=0, int ysz=0, int bands=1, GDALDataType dt=GDT_Byte)
        //! Constructor to create new file based on input vector extents
        /*explicit GeoImage(std::string filename, std::string vector, float xres, float yres, GDALDataType datatype=GDT_Byte, int bsz=1) {
            OGRDataSource *poDS = OGRSFDriverRegistrar::Open(vector.c_str());
            OGRLayer *poLayer = poDS->GetLayer(0);
            OGREnvelope extent;
            poLayer->GetExtent(&extent, true);
            int xsize = (int)(0.5 + (extent.MaxX - extent.MinX) / xres);
            int ysize = (int)(0.5 + (extent.MaxY - extent.MinY) / yres);
            GeoData::CreateNew(xsize, ysize, bsz, datatype, filename);
            double affine[6];
            affine[0] = extent.MinX;
            affine[1] = xres;
            affine[2] = 0;
            affine[3] = extent.MaxY;
            affine[4] = 0;
            affine[5] = -yres;
            _GDALDataset->SetGeoTransform(affine);
            char* wkt = NULL;
            poLayer->GetSpatialRef()->exportToWkt(&wkt);
            _GDALDataset->SetProjection(wkt);
            OGRDataSource::DestroyDataSource( poDS );
        }*/
        //! Copy constructor - copies GeoData and all bands
        GeoImage(const GeoImage& image);
        //! Assignment Operator
        GeoImage& operator=(const GeoImage& image) ;
        //! Destructor
        ~GeoImage() { _RasterBands.clear(); }

        //! \name File Information
        //! Number of bands
        unsigned int NumBands() const { return _RasterBands.size(); }
        //! Get datatype of image (check all raster bands, return 'largest')
        GDALDataType DataType() const { return _RasterBands[0].DataType(); }
        //! Return information on image as string
        std::string Info(bool=true, bool=false) const;

        //! \name Bands and colors
        //! Get vector of band names
        std::vector<std::string> BandNames() const;
        //! Get raster band (0-based index)
        GeoRaster& operator[](int band) { return _RasterBands[band]; }
        //! Get raster band, const version
        const GeoRaster& operator[](int band) const { return _RasterBands[band]; }
        //! Get raster band by color
        GeoRaster& operator[](std::string col) {
            // Call const version
            return const_cast<GeoRaster&>(static_cast<const GeoImage&>(*this)[col]);
        }
        //! Get raster band by color, const version
        const GeoRaster& operator[](std::string col) const;
        //! Adds a band, at position bandnum (0-based)
        GeoImage& AddBand(const GeoRaster& band); //, unsigned int bandnum=0);
        //! Remove band
        GeoImage& RemoveBand(unsigned int bandnum);
        //! Prune bands to only provided colors
        GeoImage& PruneBands(std::vector<std::string>);
        //! Prune bands to RGB
        GeoImage& PruneToRGB() {
            // TODO - get initializer_list working (c++11 standard)
            std::vector<std::string> cols; cols.push_back("Red"); cols.push_back("Green"); cols.push_back("Blue");
            return PruneBands(cols);
        }
        //! Retrieve colors class
        Colors GetColors() const { return _Colors; }
        //! Set a color
        void SetColor(std::string col,int bandnum) {
            _Colors.SetColor(col,bandnum);
            // Set color on individual band
            _RasterBands[bandnum-1].SetColor(col);
            _RasterBands[bandnum-1].SetDescription(col);
        }
        // TODO - dictionary?
        /*void SetColors(int blue, int green, int red, int nir=0) {
            SetColor("Blue",blue);
            SetColor("Green",green);
            SetColor("Red",red);
            if (nir > 0) SetColor("NIR",nir);
        }*/
        //! Copy color table from another image
        void CopyColorTable(const GeoImage& raster) {
            if (NumBands() == 1) {
                GDALColorTable* table( raster[0].GetGDALRasterBand()->GetColorTable() );
                if (table != NULL) _RasterBands[0].GetGDALRasterBand()->SetColorTable(table);
            }
        }

        //! \name Multiple band convenience functions
        //! Set gain for all bands
        void SetGain(float gain) { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].SetGain(gain); }
        //! Set gain for all bands
        void SetOffset(float offset) { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].SetOffset(offset); }
        //! Set  for all bands
        void SetUnits(std::string units) { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].SetUnits(units); }
        //! Set NoData for all bands
        void SetNoData(double val) { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].SetNoData(val); }
        //! Unset NoData for all bands
        void ClearNoData() { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].ClearNoData(); }

        //! \name Processing functions
        template<class T> GeoImage& Process();
        //! Process band into new file (copy and apply processing functions)
        template<class T> GeoImage Process(std::string, GDALDataType = GDT_Unknown);

        //! Adds a mask band (1 for valid) to every band in image
        GeoImage& AddMask(const GeoRaster& band) {
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].AddMask(band);
            return *this;
        }
        //! Clear all masks
        void ClearMasks() { for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].ClearMasks(); }

        //! Replace all 'Inf' or 'NaN' results with the bands NoData value
        GeoImage& FixBadPixels();

        // hmm, what's this do?
        //const GeoImage& ComputeStats() const;

        //! Add overviews
        GeoData& AddOverviews() {
            int panOverviewList[3] = { 2, 4, 8 };
            _GDALDataset->BuildOverviews( "NEAREST", 3, panOverviewList, 0, NULL, GDALDummyProgress, NULL );
            return *this;
        }

        //! Break up image into chunks
        /*void Chunk(unsigned int pad=0) const {
            GeoData::Chunk(pad);
            for (unsigned int b=0;b<NumBands();b++) _RasterBands.
        }*/

        //! \name File I/O
        //! Read raw chunk, across all bands
        template<class T> cimg_library::CImg<T> ReadRaw(int chunk=0) const { //, bool RAW=false) const {
            cimg_library::CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->ReadRaw<T>(chunk) );
            }
            //return images.get_append('c','p');
            return images.get_append('v','p');
        }
        //! Read chunk, across all bands
        template<class T> cimg_library::CImg<T> Read(int chunk=0) const { //, bool RAW=false) const {
            cimg_library::CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->Read<T>(chunk) );
            }
            //return images.get_append('c','p');
            return images.get_append('v','p');
        }

        //GeoImageIO& Write(const CImg<T> img, int chunk=0, bool BadValCheck=false) {
        //    return Write(img, chunk, BadValCheck);
        //}

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
        template<class T> cimg_library::CImgList<T> ReadAsList(int chunk=0) const {
            cimg_library::CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->Read<T>(chunk) );
            }
            return images;
        }

        //! Mean (per pixel) of all bands, written to raster
        GeoRaster& Mean(GeoRaster& raster) const {
            CImg<unsigned char> mask;
            CImg<int> totalpixels;
            CImg<double> band, total;
            for (unsigned int iChunk=0; iChunk<NumChunks(); iChunk++) {
                for (unsigned int iBand=0;iBand<NumBands();iBand++) {
                    mask = _RasterBands[iBand].DataMask();
                    band = _RasterBands[iBand].Read<double>(iChunk).mul(mask);
                    if (iBand == 0) {
                        totalpixels = mask;
                        total = band;
                    } else {
                        totalpixels += mask;
                        total += band;
                    }
                }
                total = total.div(totalpixels);
                //cimg_for(total,ptr,double) {
                //    if (*ptr != *ptr) *ptr = raster.NoData();
                //}
                raster.Write(total, iChunk);
            }
            return raster;
        }

        //! NoData mask (all bands).  1's where it is nodata
        CImg<unsigned char> NoDataMask(int chunk=0, std::vector<std::string> bands=std::vector<std::string>()) const {
            std::vector<int> ibands = Descriptions2Indices(bands);
            CImg<unsigned char> mask;
            for (std::vector<int>::const_iterator i=ibands.begin(); i!=ibands.end(); i++) {
                if (i==ibands.begin()) 
                    mask = CImg<unsigned char>(_RasterBands[*i].NoDataMask(chunk));
                else
                    mask|=_RasterBands[*i].NoDataMask(chunk);
            }
            return mask;
        }

        //! Saturation mask (all bands).  1's where it is saturated
        CImg<unsigned char> SaturationMask(int chunk=0, std::vector<std::string> bands=std::vector<std::string>()) const {
            std::vector<int> ibands = Descriptions2Indices(bands);
            CImg<unsigned char> mask;
            for (std::vector<int>::const_iterator i=ibands.begin(); i!=ibands.end(); i++) {
                if (i==ibands.begin()) 
                    mask = CImg<unsigned char>(_RasterBands[*i].SaturationMask(chunk));
                else
                    mask|=_RasterBands[*i].SaturationMask(chunk);
            }
            return mask;
        }

        //! Whiteness (created from red, green, blue)
        CImg<float> Whiteness(int chunk=0) const {
            // RAW or RADIANCE ?
            CImg<float> red = operator[]("RED").ReadRaw<float>(chunk);
            CImg<float> green = operator[]("GREEN").ReadRaw<float>(chunk);
            CImg<float> blue = operator[]("BLUE").ReadRaw<float>(chunk);
            CImg<float> white(red.width(),red.height());
            float mu;
            cimg_forXY(white,x,y) {
                mu = (red(x,y) + green(x,y) + blue(x,y))/3;
                white(x,y) = (abs(red(x,y)-mu) + abs(green(x,y)-mu) + abs(blue(x,y)-mu))/mu;
            }
            // Saturation?  If pixel saturated make Whiteness 0 ?
            return white;
        }

        //! Extract, and interpolate, time series (C is time axis)
        template<class T> cimg_library::CImg<T> TimeSeries(cimg_library::CImg<double> C) {
            cimg_library::CImg<T> cimg = Read<T>();
            T nodata = _RasterBands[0].NoDataValue();
            if (cimg.spectrum() > 2) {
                int lowi, highi;
                float y0, y1, x0, x1;
                for (int c=1; c<cimg.spectrum()-1;c++) {
                    if (Options::Verbose() > 3) cimg_print(C, "days vector");
                    cimg_forXY(cimg,x,y) {
                        if (cimg(x,y,c) == nodata) {
                            // Find next lowest point
                            lowi = highi = 1;
                            while ((cimg(x,y,c-lowi) == nodata) && (lowi<c)) lowi++;
                            while ((cimg(x,y,c+highi) == nodata) && (c+highi < cimg.spectrum()-1) ) highi++;
                            y0 = cimg(x,y,c-lowi);
                            y1 = cimg(x,y,c+highi);
                            x0 = C(c-lowi);
                            x1 = C(c+highi);
                            if ((y0 != nodata) && (y1 != nodata)) {
                                cimg(x,y,c) = y0 + (y1-y0) * ((C(c)-x0)/(x1-x0));
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

        //! Extract spectra from select pixels (where mask > 0)
        template<class T> cimg_library::CImg<T> Extract(const GeoRaster& mask) {
            if (Options::Verbose() > 2 ) std::cout << "Pixel spectral extraction" << std::endl;
            cimg_library::CImg<unsigned char> cmask;
            cimg_library::CImg<T> cimg;
            long count = 0;

            for (int iChunk=1; iChunk<=NumChunks(); iChunk++) {
                cmask = mask.Read<unsigned char>(iChunk);
                cimg_for(cmask,ptr,unsigned char) if (*ptr > 0) count++;
            }
            cimg_library::CImg<T> pixels(count,NumBands()+1,1,1,_RasterBands[0].NoDataValue());
            count = 0;
            int ch(0);
            unsigned int c;
            for (int iChunk=1; iChunk<=NumChunks(); iChunk++) {
                cimg = Read<T>(iChunk);
                cmask = mask.Read<unsigned char>(iChunk);
                cimg_forXY(cimg,x,y) {
                    if (cmask(x,y) > 0) {
                        for (c=0;c<NumBands();c++) pixels(count,c+1) = cimg(x,y,c);
                        pixels(count++,0) = cmask(x,y);
                    }
                }
                if (Options::Verbose() > 2) std::cout << "  Chunk " << ch++ << std::endl;
            }
            return pixels;
        }

        //! Get a number of random pixel vectors (spectral vectors)
        template<class T> CImg<T> GetRandomPixels(int NumPixels) const {
            CImg<T> Pixels(NumBands(), NumPixels);
            srand( time(NULL) );
            bool badpix;
            int p = 0;
            while(p < NumPixels) {
                int col = (double)rand()/RAND_MAX * (XSize()-1);
                int row = (double)rand()/RAND_MAX * (YSize()-1);
                T pix[1];
                badpix = false;
                for (unsigned int j=0; j<NumBands(); j++) {
                    _RasterBands[j].GetGDALRasterBand()->RasterIO(GF_Read, col, row, 1, 1, &pix, 1, 1, type2GDALtype(typeid(T)), 0, 0);
                    if (_RasterBands[j].NoData() && pix[0] == _RasterBands[j].NoDataValue()) {
                        badpix = true;
                    } else {
                        Pixels(j,p) = pix[0];
                    }
                }
                if (!badpix) p++;
            }
            return Pixels;
        }

        //! Get a number of pixel vectors that are spectrally distant from each other
        template<class T> CImg<T> GetPixelClasses(int NumClasses) const {
            int RandPixelsPerClass = 500;
            CImg<T> stats;
            CImg<T> ClassMeans(NumBands(), NumClasses);
            // Get Random Pixels
            CImg<T> RandomPixels = GetRandomPixels<T>(NumClasses * RandPixelsPerClass);
            // First pixel becomes first class
            cimg_forX(ClassMeans,x) ClassMeans(x,0) = RandomPixels(x,0);
            for (int i=1; i<NumClasses; i++) {
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
                T cutoff = DistSort[RandPixelsPerClass*i]; //(stats.max-stats.min)/10 + stats.min;
                cimg_forX(Dist,x) if (Dist(x) < cutoff) cimg_forX(RandomPixels,x1) RandomPixels(x1,x) = 0;
            }
            // Output Class Vectors
            //if (Options::Verbose()>1) cimg_printclasses(ClassMeans, "Initial Class");
            return ClassMeans;
        }


    protected:
        //! Vector of raster bands
        std::vector< GeoRaster > _RasterBands;

        //! Map indicating which bands are what colors
        Colors _Colors;

        //! Loads Raster Bands of this GDALDataset into _RasterBands vector
        void LoadBands();

        // Convert vector of band descriptions to band indices
        std::vector<int> Descriptions2Indices(std::vector<std::string> bands) const {
            std::vector<int> ibands;
            std::vector<int>::const_iterator b;
            if (bands.empty()) {
                for (unsigned int c=0; c<NumBands(); c++) ibands.push_back(c);
            } else {
                for (std::vector<std::string>::const_iterator i=bands.begin(); i!=bands.end(); i++) {
                    ibands.push_back(_Colors[*i]-1);
                }
            }
            return ibands;
        }

    }; // class GeoImage

    template<class T> GeoImage& GeoImage::Process() {
        for (unsigned int i=0; i<NumBands(); i++) {
            for (unsigned int iChunk=1; iChunk <= (*this)[i].NumChunks(); iChunk++) {
                (*this)[i].Write((*this)[i].Read<T>(iChunk),iChunk);
            }
        }
        return *this;
    }

    // Copy input file into new output file
    template<class T> GeoImage GeoImage::Process(std::string filename, GDALDataType datatype) {
        // TODO: if not supplied base output datatype on units?
        if (datatype == GDT_Unknown) datatype = this->DataType();
        GeoImage imgout(filename, *this, datatype);
        for (unsigned int i=0; i<imgout.NumBands(); i++) {
            imgout[i].CopyMeta((*this)[i]);
            imgout[i].Process<T>((*this)[i]);
        }
        Colors colors = this->GetColors();
        for (unsigned int i=0;i<this->NumBands();i++) imgout.SetColor(colors[i+1], i+1);
        imgout.CopyColorTable(*this);
        return imgout;
    }

} // namespace gip

#endif
