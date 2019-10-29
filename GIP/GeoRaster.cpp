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
        : GeoResource(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), 
            _ValidStats(image._ValidStats), _Stats(image._Stats), _Functions(image._Functions) {}

    // Copy constructor with added processing
    GeoRaster::GeoRaster(const GeoRaster& image, func f)
        : GeoResource(image), _GDALRasterBand(image._GDALRasterBand), _Masks(image._Masks), 
            _ValidStats(false), _Stats(image._Stats), _Functions(image._Functions) {
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
        _ValidStats = image._ValidStats;
        _Stats = image._Stats;
        //_ValidSize = image._ValidSize;
        _Functions = image._Functions;
        //cout << _GeoImage->Basename() << ": " << ref << " references (GeoRaster Assignment)" << endl;
        return *this;
    }

    string GeoRaster::info(bool showstats) const {
        std::stringstream info;
        //info << _GeoImage->Basename() << " - b" << _GDALRasterBand->GetBand() << ":" << endl;
        info << xsize() << " x " << ysize() << " " << type().string() << ": " << description();
        //info << " (GeoData: " << _GDALDataset.use_count() << " " << _GDALDataset << ")";
        //info << " RasterBand &" << _GDALRasterBand << endl;
        info << "   Gain = " << gain() << ", Offset = " << offset(); //<< ", Units = " << Units();
        info << ", NoData = " << nodata() << endl;
        if (showstats) {
            CImg<float> st = this->stats();
            info << "      Min = " << st(0) << ", Max = " << st(1) << ", Mean = " << st(2) << " =/- " << st(3) << endl;
        }
        /*if (!_Functions.empty()) info << "      Functions:" << endl;
        for (unsigned int i=0;i<_Functions.size();i++) {
          info << "      " << _Functions[i] << endl; //" " << _Functions[i].Operand() << endl;
        }*/
        if (!_Masks.empty()) info << "\tMasks:" << endl;
        for (unsigned int i=0;i<_Masks.size();i++) info << "      " << _Masks[i].info() << endl;
        //_GeoImage->GetGDALDataset()->Reference(); int ref = _GeoImage->GetGDALDataset()->Dereference();
        //info << "  GDALDataset: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << endl;
        return info.str();
    }

    //! Compute stats
    CImg<float> GeoRaster::stats() const {
        if (_ValidStats) return _Stats;

        CImg<double> cimg;
        double count(0), total(0), val;
        double min(type().maxval()), max(type().minval());
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> _chunks = chunks();

        double noDataVal = nodata();
        for (iCh=_chunks.begin(); iCh!=_chunks.end(); iCh++) {
            cimg = read<double>(*iCh);
            cimg_for(cimg,ptr,double) {
                if (*ptr != noDataVal) {
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
        for (iCh=_chunks.begin(); iCh!=_chunks.end(); iCh++) {
            cimg = read<double>(*iCh);
            cimg_for(cimg,ptr,double) {
                if (*ptr != noDataVal) {
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

    double GeoRaster::percentile(const double& p) const {
        CImg<float> st = stats();
        unsigned int bins(100);
        CImg<float> hist = histogram(bins,true,true) * 100;
        CImg<float> xaxis(bins);
        float interval( (st(1)-st(0))/((float)bins-1) );
        for (unsigned int i=0;i<bins;i++) xaxis[i] = st(0) + i * interval;
        if (p == 0) return st(0);
        if (p == 99) return st(1);
        int ind(1);
        while(hist[ind] < p) ind++;
        float xind( (p-hist[ind-1])/(hist[ind]-hist[ind-1]) );
        return xaxis.linear_atX(ind-1+xind);
    }

    //! Compute histogram
    CImg<double> GeoRaster::histogram(unsigned int bins, bool normalize, bool cumulative) const {
        //CImg<double> cimg;
        CImg<float> st = stats();
        CImg<double> hist(bins,1,1,1,0);
        double numpixels(0);
        float nd = nodata();
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> _chunks = chunks();
        unsigned int index;
        for (iCh=_chunks.begin(); iCh!=_chunks.end(); iCh++) {
            CImg<double> cimg = read<double>(*iCh);
            cimg_for(cimg,ptr,double) {
                if (*ptr != nd) {
                    index = floor((*ptr-st(0))/(st(1)-st(0)) * bins);
                    //std::cout << index << " " << hist[index] << " " << numpixels << std::endl;
                    // this would be due to floating point roundoff error
                    if (index==bins) {
                        index = bins-1;
                    }
                    else if (index > bins) {
                        index = 0;
                    }
                    hist[index] = hist[index]+1;
                    numpixels++;
                    //std::cout << index << " " << hist[index] << " " << numpixels << std::endl;
                }
            }
        }
        // normalize
        if (normalize)
            hist/=numpixels;
        if (cumulative)
            for (unsigned int i=1;i<bins;i++) hist[i] += hist[i-1];
        //if (Options::verbose() > 3) hist.display_graph(0,3,1,"Pixel Value",st(0),stats(1));
        return hist;
    }

    // Metadata
    string GeoRaster::bandmeta(string key) const {
        const char* item = _GDALRasterBand->GetMetadataItem(key.c_str());
        return (item == NULL) ? "": item;
    }

    dictionary GeoRaster::bandmeta() const {
        char** meta = _GDALRasterBand->GetMetadata();
        int num = CSLCount(meta);
        dictionary items;
        for (int i=0;i<num; i++) {
            string md = string(meta[i]);
            string::size_type pos = md.find("=");
            if (pos != string::npos) {
                items[md.substr(0, pos)] = md.substr(pos+1);
            }
        }
        return items;
    }

    GeoRaster& GeoRaster::add_bandmeta(string key, string item) {
        _GDALRasterBand->SetMetadataItem(key.c_str(), item.c_str());
        return *this;
    }

    GeoRaster& GeoRaster::add_bandmeta(std::map<string, string> items) {
        for (dictionary::const_iterator i=items.begin(); i!=items.end(); i++) {
            add_bandmeta(i->first, i->second);
        }
        return *this;
    }

    GeoRaster& GeoRaster::warp_into(GeoRaster& imgout, GeoFeature feature, int interpolation, bool noinit, bool alltouch) const {
        if (Options::verbose() > 2) std::cout << basename() << " warping into " << imgout.basename() << std::endl;

        GeoRaster imgin(*this);

        // warp options
        GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
        GDALDataset* srcDS = imgin._GDALDataset.get();
        GDALDataset* dstDS = imgout._GDALDataset.get();
        psWarpOptions->hSrcDS = srcDS;
        psWarpOptions->hDstDS = dstDS;
        psWarpOptions->nBandCount = 1;
        psWarpOptions->panSrcBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->panDstBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );

        psWarpOptions->panSrcBands[0] = imgin._GDALRasterBand->GetBand();
        psWarpOptions->panDstBands[0] = imgout._GDALRasterBand->GetBand();
        psWarpOptions->padfSrcNoDataReal[0] = imgin.nodata();
        // TODO - note this assumes output nodata is same as input
        psWarpOptions->padfDstNoDataReal[0] = imgout.nodata();
        psWarpOptions->padfSrcNoDataImag[0] = 0.0;
        psWarpOptions->padfDstNoDataImag[0] = 0.0;
        // Copy over any functions to the new image
        imgout._Functions = imgin._Functions;

        psWarpOptions->dfWarpMemoryLimit = Options::chunksize() * 1024.0 * 1024.0;
        switch (interpolation) {
            case 1: psWarpOptions->eResampleAlg = GRA_Bilinear;
                break;
            case 2: psWarpOptions->eResampleAlg = GRA_Cubic;
                break;
            default: psWarpOptions->eResampleAlg = GRA_NearestNeighbour;
        }
        if (Options::verbose() > 2)
            psWarpOptions->pfnProgress = GDALTermProgress;
        else psWarpOptions->pfnProgress = GDALDummyProgress;

        char **papszOptions = NULL;
        //papszOptions = CSLSetNameValue(papszOptions,"SKIP_NOSOURCE","YES");
        if (noinit)
            papszOptions = CSLSetNameValue(papszOptions,"INIT_DEST", NULL);
        else
            papszOptions = CSLSetNameValue(papszOptions,"INIT_DEST","NO_DATA");
        if (alltouch)
            papszOptions = CSLSetNameValue(papszOptions, "CUTLINE_ALL_TOUCHED", "TRUE");
        papszOptions = CSLSetNameValue(papszOptions,"WRITE_FLUSH","YES");
        papszOptions = CSLSetNameValue(papszOptions,"NUM_THREADS",to_string(Options::cores()).c_str());
        psWarpOptions->papszWarpOptions = papszOptions;

        // if valid geometry apply it as a cutline
        OGRGeometry* site_t;
        char **papszOptionsCutline = NULL;
        CutlineTransformer oTransformer;
        if (feature.valid()) {
            OGRGeometry* site = feature.ogr_geometry();
            // if imgout srs different than feature srs
            OGRSpatialReference* srs = new OGRSpatialReference;
            srs->SetFromUserInput(imgout.srs().c_str());
            site->transformTo(srs);
            OGRSpatialReference::DestroySpatialReference(srs);

            // Create cutline transform to pixel coordinates        
            papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "DST_SRS", imgout.srs().c_str() );
            papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "INSERT_CENTER_LONG", "FALSE" );

            oTransformer.hSrcImageTransformer = GDALCreateGenImgProjTransformer2( srcDS, NULL, papszOptionsCutline );
            site_t = site->clone();
            site_t->transform(&oTransformer);

            //psWarpOptions->hCutline = site_t;
            char* wkt;
            site_t->exportToWkt(&wkt);
            psWarpOptions->papszWarpOptions = CSLSetNameValue(psWarpOptions->papszWarpOptions,"CUTLINE", wkt);
            CPLFree(wkt);
        }

        // set options
        //psWarpOptions->papszWarpOptions = CSLDuplicate(papszOptions);
        psWarpOptions->pTransformerArg =
            GDALCreateGenImgProjTransformer( srcDS, srcDS->GetProjectionRef(),
                                             dstDS, dstDS->GetProjectionRef(), TRUE, 0.0, 0 );
        psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

        // Perform transformation
        GDALWarpOperation oOperation;
        oOperation.Initialize( psWarpOptions );
        //if (Options::verbose() > 3) std::cout << "Error: " << CPLGetLastErrorMsg() << endl;
        oOperation.ChunkAndWarpMulti( 0, 0, imgout.xsize(), imgout.ysize() );

        // destroy things
        GDALDestroyGenImgProjTransformer( psWarpOptions->pTransformerArg );
        if (feature.valid()) {
            GDALDestroyGenImgProjTransformer( oTransformer.hSrcImageTransformer );
            CSLDestroy( papszOptionsCutline );
            OGRGeometryFactory::destroyGeometry(site_t);
        }
        GDALDestroyWarpOptions( psWarpOptions );

        return imgout;
    }


    // Smooth/convolution (3x3) taking into account NoData
    /*GeoRaster smooth(GeoRaster raster) {
        CImg<double> kernel(3,3,1,1,1);
        int m0((kernel.width())/2);
        int n0((kernel.height())/2);
        int border(std::max(m0,n0));
        double total, norm;
        CImg<double> cimg0, cimg, subcimg;

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> _chunks = chunks();
        for (iCh=_chunks.begin(); iCh!=_chunks.end(); iCh++) {
            cimg0 = read<double>(*iCh);
            cimg = cimg0;
            cimg_for_insideXY(cimg,x,y,border) {
                subcimg = cimg0.get_crop(x-m0,y-n0,x+m0,y+m0);
                total = 0;
                norm = 0;
                cimg_forXY(kernel,m,n) {
                    if (subcimg(m,n) != nodata()) {
                        total = total + (subcimg(m,n) * kernel(m,n));
                        norm = norm + kernel(m,n);
                    }
                }
                if (norm == 0)
                    cimg(x,y) = raster.nodata();
                else
                    cimg(x,y) = total/norm;
                if (cimg(x,y) == nodata()) cimg(x,y) = raster.nodata();
            }
            // Update nodata values in border region
            cimg_for_borderXY(cimg,x,y,border) {
                if (cimg(x,y) == nodata()) cimg(x,y) = raster.nodata();
            }
            raster.write(cimg, *iCh);
        }
        return raster;
    }*/


} // namespace gip
