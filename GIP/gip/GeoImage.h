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

#ifndef GIP_GEOIMAGE_H
#define GIP_GEOIMAGE_H

#include <cstddef>
#include <gip/GeoResource.h>
#include <gip/GeoRaster.h>
#include <gip/GeoFeature.h>
#include <stdint.h>
#include <gip/utils.h>

namespace gip {
    //using std::string;
    using std::vector;

    // Forward declaration
    class GeoRaster;

    //! GeoImage class
    /*!
        The GeoImage is a collection of GeoRaster objects
    */
    class GeoImage : public GeoResource {
    public:
        //! \name Constructors/Destructor
        //! Default constructor
        explicit GeoImage() : GeoResource() {}
        //! Open file constructor
        explicit GeoImage(std::string filename, bool update=false)
            : GeoResource(filename, update) {
            load_bands();
        }
        //! Open file from vector of individual files
        explicit GeoImage(std::vector<std::string> filenames);
        //! Constructor for creating new file
        explicit GeoImage(std::string filename, 
                          int xsz, int ysz, int nb, 
                          std::string proj, BoundingBox bbox, 
                          DataType dt, std::string format="", bool temp=false) :
                GeoResource(filename, xsz, ysz, nb, proj, bbox, dt, format, temp) {
            load_bands();
        }
        //! Copy constructor - copies GeoResource and all bands
        GeoImage(const GeoImage& image);
        //! Assignment operator
        GeoImage& operator=(const GeoImage& image) ;
        //! Destructor
        ~GeoImage() { _RasterBands.clear(); }

        //! \name Factory Functions
        //! Create new image
        static GeoImage create(std::string filename="", 
                unsigned int xsz=1, unsigned int ysz=1, unsigned int nb=1, 
                std::string proj="EPSG:4326",
                CImg<double> bbox=CImg<double>(4, 1, 1, 1, 0.0, 0.0, 1.0, 1.0),
                std::string dtype="uint8", std::string format="", bool temp=false) {
            if (filename == "") {
                filename = std::tmpnam(nullptr);
                temp = true;
            }
            BoundingBox ext(bbox[0], bbox[1], bbox[2], bbox[3]);
            return GeoImage(filename, xsz, ysz, nb, proj, ext, DataType(dtype), format, temp);
        }

        //! Create new image using foorprint of another
        static GeoImage create_from(GeoImage geoimg, std::string filename="", unsigned int nb=0, 
                std::string dtype="unknown", std::string format="", bool temp=false) {
            unsigned int _xs(geoimg.xsize());
            unsigned int _ys(geoimg.ysize());
            unsigned int _bs(geoimg.nbands());
            std::string _srs(geoimg.srs());
            std::string _dtype(geoimg.type().string());
            _bs = nb > 0 ? nb : _bs;
            _dtype = dtype != "unknown" ? dtype : _dtype;
            if (filename == "") {
                filename = std::tmpnam(nullptr);
                temp = true;
            }
            GeoImage img = GeoImage(filename, _xs, _ys, _bs, _srs, geoimg.extent(), _dtype, format, temp);
            // copy metadata
            img.set_meta(geoimg.meta());
            // if same number of bands, set band metadata
            if (geoimg.nbands() == _bs) {
                for (unsigned int b=0;b<_bs;b++) {
                    img[b].set_gain(geoimg[b].gain());
                    img[b].set_offset(geoimg[b].offset());
                    img[b].set_nodata(geoimg[b].nodata());
                }
                img.set_bandnames(geoimg.bandnames());
            }
            return img;
        }

        //! Open new image
        static GeoImage open(std::string filename, bool update=false, float nodata=0,
            std::vector<std::string> bandnames=std::vector<std::string>({}),
            double gain=1.0, double offset=0.0) {
            // open image, then set all these things
            GeoImage geoimg = GeoImage(filename, update);
            geoimg.set_gain(gain);
            geoimg.set_offset(offset);            
            if (bandnames.size() != geoimg.nbands())
                throw std::runtime_error("bandnames must be provided for all bands in file");
            geoimg.set_bandnames(bandnames);

            return geoimg;
        }

        //! \name File Information
        //! Return list of filenames for each band (could be duplicated)
        std::vector<std::string> filenames() const {
            std::vector<std::string> fnames;
            for (unsigned int i=0;i<_RasterBands.size();i++) {
                fnames.push_back(_RasterBands[i].filename());
            }
            return fnames;
        }

        // TODO - support different datatypes across bands easily
        DataType type() const { return _RasterBands[0].type(); }
        //! Return information on image as string
        std::string info(bool=true, bool=false) const;

        //! \name Bands and colors
        //! Number of bands
        unsigned int nbands() const { return _RasterBands.size(); }
        //! Get datatype of image (TODO - check all raster bands, return 'largest')
        //! Get vector of band names
        std::vector<std::string> bandnames() const { return _BandNames; }
        //! Set a band name
        GeoImage& set_bandname(std::string bandname, int bandnum) {
            try {
                // Test if color already exists
                (*this)[bandname];
                throw std::out_of_range ("Band " + bandname + " already exists in GeoImage!");
            } catch(...) {
                _BandNames[bandnum-1] = bandname;
                _RasterBands[bandnum-1].set_color(bandname);
            }
            return *this;
        }
        //! Set all band names with vector size equal to # bands
        GeoImage& set_bandnames(std::vector<std::string> names) {
	       if (names.size() != nbands())
            	throw std::out_of_range("Band list size must be equal to # of bands");
            for (unsigned int i=0; i<names.size(); i++) {
                try {
                    set_bandname(names[i], i+1);
                } catch(...) {
                    // TODO - print to stderr ? or log?
                    std::cout << "Band " + names[i] + " already exists" << std::endl;
                }
            }
            return *this;
        }
        //! Check if this band exists
        bool band_exists(std::string bandname) const {
            try {
                (*this)[bandname];
                return true;
            } catch(...) {
                return false;
            } 
        }   
        //! Check if ALL these bands exist
        bool bands_exist(std::vector<std::string> bnames) const {
            for (std::vector<std::string>::const_iterator i=bnames.begin(); i!=bnames.end(); i++) {
                if (!band_exists(*i)) return false;
            }
            return true;            
        }

        //! Get raster band (0-based index)
        GeoRaster& operator[](unsigned int index) { 
            // Call const version
            return const_cast<GeoRaster&>(static_cast<const GeoImage&>(*this)[index]);
        }
        //! Get raster band, const version
        const GeoRaster& operator[](unsigned int index) const;
        //! Get raster band by description
        GeoRaster& operator[](std::string desc) {
            // Call const version
            return const_cast<GeoRaster&>(static_cast<const GeoImage&>(*this)[desc]);
        }
        //! Get raster band by description, const version
        const GeoRaster& operator[](std::string desc) const;

        //! Adds a band (as last band)
        GeoImage& add(GeoRaster band);
        //! Keep only these band names
        GeoImage select(std::vector<std::string> names);
        //! Keep only these band numbers
        GeoImage select(std::vector<int> nums);

        //! \name Multiple band convenience functions
        //! Set gain for all bands
        GeoImage& set_gain(float gain) { 
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].set_gain(gain);
            return *this;
        }
        //! Set gain for all bands
        GeoImage& set_offset(float offset) { 
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].set_offset(offset);
            return *this;
        }
        //! Set NoData for all bands
        GeoImage& set_nodata(double val) { 
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].set_nodata(val);
            return *this;
        }

        //! \name Processing functions
        //! Auto rescale all bands
        GeoImage autoscale(const double& minout, const double& maxout, const double& percent=0.0) {
            GeoImage geoimg(*this);
            for (unsigned int i=0; i<_RasterBands.size(); i++) {
                geoimg[i] = geoimg[i].autoscale(minout, maxout, percent);
            }
            return geoimg;
        }

        //! Calculate spectral covariance
        CImg<double> spectral_covariance() const;

        //! Calculate spectral correlation
        //CImg<double> SpectralCorrelation(const GeoImage&, CImg<double> covariance=CImg<double>() );

        //! Process band into new file (copy and apply processing functions)
        template<class T> GeoImage save(std::string filename="", std::string dtype="", bool temp=false) const;

        //! Adds a mask band (1 for valid) to every band in image
        GeoImage& add_mask(const GeoRaster& band) {
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].add_mask(band);
            return *this;
        }
        //! Clear all masks
        GeoImage& clear_masks() { 
            for (unsigned int i=0;i<_RasterBands.size();i++) _RasterBands[i].clear_masks();
            return *this;
        }

        //! \name File I/O
        //! Read raw chunk, across all bands
        template<class T> CImg<T> read_raw(Chunk chunk=Chunk()) const {
            CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->read_raw<T>(chunk) );
            }
            return images.get_append('z');
        }

        //! Read chunk, across all bands
        template<class T> CImg<T> read(Chunk chunk=Chunk()) const {
            CImgList<T> images;
            typename std::vector< GeoRaster >::const_iterator iBand;
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                images.insert( iBand->read<T>(chunk) );
            }
            return images.get_append('z');
        }

        //! Write cube across all bands
        template<class T> GeoImage& write(const CImg<T> img, Chunk chunk=Chunk()) {
            typename std::vector< GeoRaster >::iterator iBand;
            int i(0);
            for (iBand=_RasterBands.begin();iBand!=_RasterBands.end();iBand++) {
                iBand->write(img.get_channel(i++), chunk);
            }
            return *this;
        }

        // Generate Masks: NoData, Data, Saturation
        //! NoData mask.  1's where it is nodata
        CImg<uint8_t> nodata_mask(std::vector<std::string> bands, Chunk chunk=Chunk()) const {
            std::vector<int> ibands = Descriptions2Indices(bands);
            CImg<unsigned char> mask;
            for (std::vector<int>::const_iterator i=ibands.begin(); i!=ibands.end(); i++) {
                if (i==ibands.begin()) 
                    mask = CImg<unsigned char>(_RasterBands[*i].nodata_mask(chunk));
                else
                    mask|=_RasterBands[*i].nodata_mask(chunk);
            }
            return mask;
        }

        // NoData mask (all bands)
        CImg<uint8_t> nodata_mask(Chunk chunk=Chunk()) const {
            return nodata_mask({}, chunk);
        }

        //! Data mask. 1's where valid data
        CImg<unsigned char> data_mask(std::vector<std::string> bands, Chunk chunk=Chunk()) const {
            return nodata_mask(bands, chunk)^=1;
        }

        CImg<unsigned char> data_mask(Chunk chunk=Chunk()) const {
            return data_mask({}, chunk);
        }

        //! Saturation mask (all bands).  1's where it is saturated
        CImg<unsigned char> saturation_mask(std::vector<std::string> bands, Chunk chunk=Chunk()) const {
            std::vector<int> ibands = Descriptions2Indices(bands);
            CImg<unsigned char> mask;
            for (std::vector<int>::const_iterator i=ibands.begin(); i!=ibands.end(); i++) {
                if (i==ibands.begin()) 
                    mask = CImg<unsigned char>(_RasterBands[*i].saturation_mask(chunk));
                else
                    mask|=_RasterBands[*i].saturation_mask(chunk);
            }
            return mask;
        }

        CImg<unsigned char> saturation_mask(Chunk chunk=Chunk()) const {
            return saturation_mask({}, chunk);
        }

        //! Whiteness (created from red, green, blue)
        CImg<float> whiteness(Chunk chunk=Chunk()) const {
            if (!bands_exist({"red", "green", "blue"}))
                throw std::out_of_range("Need RGB bands to calculate whiteness");
            CImg<float> red = operator[]("red").read_raw<float>(chunk);
            CImg<float> green = operator[]("green").read_raw<float>(chunk);
            CImg<float> blue = operator[]("blue").read_raw<float>(chunk);
            CImg<float> white(red.width(),red.height());
            float mu;
            cimg_forXY(white,x,y) {
                mu = (red(x,y) + green(x,y) + blue(x,y))/3;
                white(x,y) = (abs(red(x,y)-mu) + abs(green(x,y)-mu) + abs(blue(x,y)-mu))/mu;
            }
            // Saturation?  If pixel saturated make Whiteness 0 ?
            return white;
        }

        GeoImage warp(std::string filename="", GeoFeature feature=GeoFeature(), 
            bool crop=false, std::string proj="EPSG:4326",
            float xres=1.0, float yres=1.0, int interpolation=0) const;

        GeoImage& warp_into(GeoImage&, GeoFeature=GeoFeature(), int=0, bool=false) const;

    protected:
        //! Vector of raster bands
        std::vector< GeoRaster > _RasterBands;
        //! Vector of raster band names
        std::vector< std::string > _BandNames;

        //! Loads Raster Bands of this GDALDataset into _RasterBands vector
        void load_bands();

        // Convert vector of band descriptions to band indices
        std::vector<int> Descriptions2Indices(std::vector<std::string> bands) const;

    private:
        int band_index(std::string name) const {
            name = to_lower(name);
            std::string bname;
            for (unsigned int i=0; i<_BandNames.size(); i++) {
                bname = _BandNames[i];
                if (name == to_lower(bname)) return i;
            }
            throw std::out_of_range("No band " + name);
        }

    }; // class GeoImage

    // GeoImage template function definitions
    //! Process in-place
    //! This is broken
    /*
    template<class T> GeoImage& GeoImage::save() {
        // Create chunks
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = image.chunks();
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
                (*this)[i].write((*this)[i].read<T>(*iCh),*iCh);
            }
            // clear functions after processing
            (*this)[i].ClearFunctions();
        }
        return *this;
    }
    */

    // Save input file with processing applied into new output file
    template<class T> GeoImage GeoImage::save(std::string filename, std::string datatype, bool overviews) const {
        // TODO: if not supplied base output datatype on units?
        if (datatype == "") datatype = this->type().string();

        GeoImage imgout = GeoImage::create_from(*this, filename, nbands(), datatype);
        for (unsigned int i=0; i<imgout.nbands(); i++) {
            (*this)[i].save<T>(imgout[i]);
        }
        if (overviews) {
            int panOverviewList[3] = { 2, 4, 8 };
            imgout._GDALDataset->BuildOverviews( "NEAREST", 3, panOverviewList, 0, NULL, GDALDummyProgress, NULL );
        }
        return imgout;
    }

} // namespace gip

#endif
