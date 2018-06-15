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


