// These are old member functions to GeoImage that need review and
// generalizing, and tests, before bringing back, if at all
// Some may just need to be separate functions (such as the random pixel
// vectors and such that are used by kmeans)


//! Mean (per pixel) of all bands, written to raster
// TODO - review this function - it writes to a file, so not a good
// member function
GeoRaster& Mean(GeoRaster& raster) const {
    CImg<unsigned char> mask;
    CImg<int> totalpixels;
    CImg<double> band, total;
    ChunkSet chunks(XSize(),YSize());
    for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
        for (unsigned int iBand=0;iBand<NumBands();iBand++) {
            mask = _RasterBands[iBand].DataMask(chunks[iChunk]);
            band = _RasterBands[iBand].Read<double>(chunks[iChunk]).mul(mask);
            if (iBand == 0) {
                totalpixels = mask;
                total = band;
            } else {
                totalpixels += mask;
                total += band;
            }
        }
        total = total.div(totalpixels);
        cimg_for(total,ptr,double) {
            if (*ptr != *ptr) *ptr = raster.NoData();
        }
        raster.Write(total, chunks[iChunk]);
    }
    return raster;
}

//! Calculate mean, stddev for chunk - must contain data for all bands
// TODO - review this function
CImgList<double> SpectralStatistics(iRect chunk=iRect()) const {
    CImg<unsigned char> mask;
    CImg<double> band, total, mean;
    unsigned int iBand;
    mask = DataMask({}, chunk);
    double nodata = _RasterBands[0].NoData();
    for (iBand=0;iBand<NumBands();iBand++) {
        band = _RasterBands[iBand].Read<double>(chunk).mul(mask);
        if (iBand == 0) {
            total = band;
        } else {
            total += band;
        }
    }
    mean = total / NumBands();
    for (iBand=0;iBand<NumBands();iBand++) {
        band = _RasterBands[iBand].Read<double>(chunk).mul(mask);
        if (iBand == 0) {
            total = (band - mean).pow(2);
        } else {
            total += (band - mean).pow(2);
        }
    }
    CImgList<double> stats(mean, (total / (NumBands()-1)).sqrt());
    cimg_forXY(mask,x,y) { 
        if (mask(x,y) == 0) {
            stats[0](x,y) = nodata;
            stats[1](x,y) = nodata; 
        }
    }
    return stats;          
}

//! Extract, and interpolate, time series (C is time axis)
// TODO - review this function to be more general extraction over bands
template<class T, class t> CImg<T> TimeSeries(CImg<t> times, iRect chunk=iRect()) {
    CImg<T> cimg = Read<T>(chunk);
    T nodata = _RasterBands[0].NoData();
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
                    x0 = times(c-lowi);
                    x1 = times(c+highi);
                    if ((y0 != nodata) && (y1 != nodata)) {
                        cimg(x,y,c) = y0 + (y1-y0) * ((times(c)-x0)/(x1-x0));
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
// TODO - review this function to be more general, combined with TimeSeries
template<class T> CImg<T> Extract(const GeoRaster& mask) {
    if (Options::verbose() > 2 ) std::cout << "Pixel spectral extraction" << std::endl;
    CImg<unsigned char> cmask;
    CImg<T> cimg;
    long count = 0;
    ChunkSet chunks(XSize(),YSize());
    for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
        cmask = mask.Read<unsigned char>(chunks[iChunk]);
        cimg_for(cmask,ptr,unsigned char) if (*ptr > 0) count++;
    }
    CImg<T> pixels(count,NumBands()+1,1,1,_RasterBands[0].NoData());
    count = 0;
    unsigned int c;
    for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
        if (Options::verbose() > 3) std::cout << "Extracting from chunk " << iChunk << std::endl;
        cimg = Read<T>(chunks[iChunk]);
        cmask = mask.Read<unsigned char>(chunks[iChunk]);
        cimg_forXY(cimg,x,y) {
            if (cmask(x,y) > 0) {
                for (c=0;c<NumBands();c++) pixels(count,c+1) = cimg(x,y,c);
                pixels(count++,0) = cmask(x,y);
            }
        }
    }
    return pixels;
}

//! Get a number of random pixel vectors (spectral vectors)
// TODO - review this function, which is used by k-means, likely too specific
// generalize to get spectra of passed in indices maybe?
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
            DataType dt(typeid(T));
            _RasterBands[j]._GDALRasterBand->RasterIO(GF_Read, col, row, 1, 1, &pix, 1, 1, dt.GDALType(), 0, 0);
            if (_RasterBands[j].NoData() && pix[0] == _RasterBands[j].NoData()) {
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
// TODO - review this function for generality, maybe specific to kmeans?
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
    return ClassMeans;
}
