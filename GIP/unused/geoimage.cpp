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


