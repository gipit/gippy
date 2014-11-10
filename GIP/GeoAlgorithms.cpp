/*##############################################################################
#    GIPPY: Geospatial Image Processing library for Python
#
#    Copyright (C) 2014 Matthew A Hanson
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <http://www.gnu.org/licenses/>
##############################################################################*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <set>

#include <gip/GeoAlgorithms.h>
#include <gip/gip_CImg.h>
#include <gip/gip_gdal.h>

//#include <gdal/ogrsf_frmts.h>
//#include <gdal/gdalwarper.h>

namespace gip {
    using std::string;
    using std::vector;
    using std::cout;
    using std::cerr;
    using std::endl;
    namespace fs = boost::filesystem;

    /** ACCA (Automatic Cloud Cover Assessment). Takes in TOA Reflectance,
     * temperature, sun elevation, solar azimuth, and number of pixels to
     * dilate.
     */
    GeoImage ACCA(const GeoImage& image, std::string filename, float se_degrees,
                  float sa_degrees, int erode, int dilate, int cloudheight, dictionary metadata ) {
        if (Options::Verbose() > 1) cout << "GIPPY: ACCA - " << image.Basename() << endl;

        float th_red(0.08);
        float th_ndsi(0.7);
        float th_temp(27);
        float th_comp(225);
        float th_nirred(2.0);
        float th_nirgreen(2.0);
        float th_nirswir1(1.0);
        //float th_warm(210);

        GeoImage imgout(filename, image, GDT_Byte, 4);
        imgout.SetNoData(0);
        imgout.SetUnits("other");
        // Band indices
        int b_pass1(3);
        int b_ambclouds(2);
        int b_cloudmask(1);
        int b_finalmask(0);
        imgout[b_finalmask].SetDescription("finalmask");
        imgout[b_cloudmask].SetDescription("cloudmask");
        imgout[b_ambclouds].SetDescription("ambclouds");
        imgout[b_pass1].SetDescription("pass1");
        imgout.SetMeta(metadata);

        vector<string> bands_used({"RED","GREEN","NIR","SWIR1","LWIR"});

        CImg<float> red, green, nir, swir1, temp, ndsi, b56comp;
        CImg<unsigned char> nonclouds, ambclouds, clouds, mask, temp2;
        float cloudsum(0), scenesize(0);

        //if (Options::Verbose()) cout << image.Basename() << " - ACCA (dev-version)" << endl;
        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            red = image["RED"].Read<float>(iChunk);
            green = image["GREEN"].Read<float>(iChunk);
            nir = image["NIR"].Read<float>(iChunk);
            swir1 = image["SWIR1"].Read<float>(iChunk);
            temp = image["LWIR"].Read<float>(iChunk);

            mask = image.NoDataMask(iChunk, bands_used)^=1;

            ndsi = (green - swir1).div(green + swir1);
            b56comp = (1.0 - swir1).mul(temp + 273.15);

            // Pass one
            nonclouds = // 1's where they are non-clouds
                // Filter1
                (red.get_threshold(th_red)^=1) |=
                // Filter2
                ndsi.get_threshold(th_ndsi) |=
                // Filter3
                temp.get_threshold(th_temp);

            ambclouds =
                (nonclouds^1).mul(
                // Filter4
                b56comp.get_threshold(th_comp) |=
                // Filter5
                nir.get_div(red).threshold(th_nirred) |=
                // Filter6
                nir.get_div(green).threshold(th_nirgreen) |=
                // Filter7
                (nir.get_div(swir1).threshold(th_nirswir1)^=1) );

            clouds =
                (nonclouds + ambclouds)^=1;

                // Filter8 - warm/cold
                //b56comp.threshold(th_warm) + 1);

            //nonclouds.mul(mask);
            clouds.mul(mask);
            ambclouds.mul(mask);

            cloudsum += clouds.sum();
            scenesize += mask.sum();

            imgout[b_pass1].Write<unsigned char>(clouds,iChunk);
            imgout[b_ambclouds].Write<unsigned char>(ambclouds,iChunk);
            //imgout[0].Write(nonclouds,iChunk);
            if (Options::Verbose() > 3) cout << "Processed chunk " << iChunk << " of " << image[0].NumChunks() << endl;
        }
        // Cloud statistics
        float cloudcover = cloudsum / scenesize;
        CImg<float> tstats = image["LWIR"].AddMask(imgout[b_pass1]).Stats();
        if (Options::Verbose() > 1) {
            cout.precision(4);
            cout << "   Cloud Cover = " << cloudcover*100 << "%" << endl;
            cimg_print(tstats, "Cloud stats(min,max,mean,sd,skew,count)");
        }

        // Pass 2 (thermal processing)
        bool addclouds(false);
        if ((cloudcover > 0.004) && (tstats(2) < 22.0)) {
            float th0 = image["LWIR"].Percentile(83.5);
            float th1 = image["LWIR"].Percentile(97.5);
            if (tstats[4] > 0) {
                float th2 = image["LWIR"].Percentile(98.75);
                float shift(0);
                shift = tstats[3] * ((tstats[4] > 1.0) ? 1.0 : tstats[4]);
                //cout << "Percentiles = " << th0 << ", " << th1 << ", " << th2 << ", " << shift << endl;
                if (th2-th1 < shift) shift = th2-th1;
                th0 += shift;
                th1 += shift;
            }
            image["LWIR"].ClearMasks();
            CImg<float> warm_stats = image["LWIR"].AddMask(imgout[b_ambclouds]).AddMask(image["LWIR"] < th1).AddMask(image["LWIR"] > th0).Stats();
            if (Options::Verbose() > 1) cimg_print(warm_stats, "Warm Cloud stats(min,max,mean,sd,skew,count)");
            image["LWIR"].ClearMasks();
            if (((warm_stats(5)/scenesize) < 0.4) && (warm_stats(2) < 22)) {
                if (Options::Verbose() > 2) cout << "Accepting warm clouds" << endl;
                imgout[b_ambclouds].AddMask(image["LWIR"] < th1).AddMask(image["LWIR"] > th0);
                addclouds = true;
            } else {
                // Cold clouds
                CImg<float> cold_stats = image["LWIR"].AddMask(imgout[b_ambclouds]).AddMask(image["LWIR"] < th0).Stats();
                if (Options::Verbose() > 1) cimg_print(cold_stats, "Cold Cloud stats(min,max,mean,sd,skew,count)");
                image["LWIR"].ClearMasks();
                if (((cold_stats(5)/scenesize) < 0.4) && (cold_stats(2) < 22)) {
                    if (Options::Verbose() > 2) cout << "Accepting cold clouds" << endl;
                    imgout[b_ambclouds].AddMask(image["LWIR"] < th0);
                    addclouds = true;
                } else
                    if (Options::Verbose() > 2) cout << "Rejecting all ambiguous clouds" << endl;
            }
        } else image["LWIR"].ClearMasks();

        //! Coarse shadow covering smear of image
        float xres(30.0);
        float yres(30.0);
        float sunelevation(se_degrees*M_PI/180.0);
        float solarazimuth(sa_degrees*M_PI/180.0);
        float distance = cloudheight/tan(sunelevation);
        int dx = -1.0 * sin(solarazimuth) * distance / xres;
        int dy = cos(solarazimuth) * distance / yres;
        int padding(double(dilate)/2+std::max(abs(dx),abs(dy))+1);
        int smearlen = sqrt(dx*dx+dy*dy);
        if (Options::Verbose() > 2)
            cerr << "distance = " << distance << endl
                 << "dx       = " << dx << endl
                 << "dy       = " << dy << endl
                 << "smearlen = " << smearlen << endl ;

        // shift-style smear
        int signX(dx/abs(dx));
        int signY(dy/abs(dy));
        int xstep = std::max(signX*dx/dilate/4, 1);
        int ystep = std::max(signY*dy/dilate/4, 1);
        if (Options::Verbose() > 2)
            cerr << "dilate = " << dilate << endl
                 << "xstep  = " << signX*xstep << endl
                 << "ystep  = " << signY*ystep << endl ;

        for (unsigned int b=0;b<imgout.NumBands();b++) { imgout[b].SetChunkPadding(padding); }
        for (unsigned int b=0;b<image.NumBands();b++) { image[b].SetChunkPadding(padding); }

        for (unsigned int iChunk=1; iChunk<=imgout[0].NumChunks(); iChunk++) {
            if (Options::Verbose() > 3) cout << "Chunk " << iChunk << " of " << imgout[0].NumChunks() << endl;
            clouds = imgout[b_pass1].Read<unsigned char>(iChunk);
            // should this be a |= ?
            if (addclouds) clouds += imgout[b_ambclouds].Read<unsigned char>(iChunk);
            clouds|=(image.SaturationMask(iChunk, bands_used));
            // Majority filter
            //clouds|=clouds.get_convolve(filter).threshold(majority));
            if (erode > 0)
                clouds.erode(erode, erode);
            if (dilate > 0)
                clouds.dilate(dilate,dilate);
            if (smearlen > 0) {
                temp2 = clouds;
                // walking back to 0,0 from dx,dy
                for(int xN=abs(dx),yN=abs(dy); xN>0 && yN>0; xN-=xstep,yN-=ystep)
                    clouds|=temp2.get_shift(signX*xN,signY*yN);
            }
            imgout[b_cloudmask].Write<unsigned char>(clouds,iChunk);
            // Inverse and multiply by nodata mask to get good data mask
            imgout[b_finalmask].Write<unsigned char>((clouds^=1).mul(image.NoDataMask(iChunk, bands_used)^=1), iChunk);
            // TODO - add in snow mask
        }
        return imgout;
    }

    //! Generate byte-scaled image (grayscale or 3-band RGB if available) for easy viewing
    std::string BrowseImage(const GeoImage& image, int quality) {
        //if (Options::Verbose() > 1) cout << "GIPPY: BrowseImage - " << image.Basename() << endl;

        GeoImage img(image);
        if (img.BandsExist({"RED","GREEN","BLUE"})) {
            img.PruneToRGB();
        } else {
            img.PruneBands({img[0].Description()});
        }
        boost::filesystem::path dir(img.Path().parent_path() / "browse");
        if (!fs::is_directory(dir)) {
            if(!boost::filesystem::create_directory(dir)) 
                throw std::runtime_error("Could not create browse directory " + dir.string());
        }
        std::string filename = (dir / img.Path().stem()).string() + ".jpg";

        CImg<double> stats;
        float lo, hi;
        for (unsigned int b=0; b<img.NumBands(); b++) {
            stats = img[b].Stats();
            lo = std::max(stats(2) - 3*stats(3), stats(0));
            hi = std::min(stats(2) + 3*stats(3), stats(1));
            if ((lo == hi) && (lo == 1)) lo = 0;
            img[b] = ((img[b] - lo) * (255.0/(hi-lo))).max(0.0).min(255.0);
        }
        CImg<double> cimg(img.Read<double>());
        // TODO - alpha channel?
        cimg_for(cimg, ptr, double) { if (*ptr == img[0].NoDataValue()) *ptr = 0; }

        cimg.round().save_jpeg(filename.c_str(), quality);

        if (Options::Verbose() > 1) cout << image.Basename() << ": BrowseImage written to " << filename << endl;
        return filename;
    }

    //! Merge images into one file and crop to vector
    GeoImage CookieCutter(vector<std::string> imgnames, string filename, string vectorname, 
            float xres, float yres, bool crop, dictionary metadata) {
        // TODO - pass in vector of GeoRaster's instead
        if (Options::Verbose() > 1)
            cout << "GIPPY: CookieCutter (" << imgnames.size() << " files) - " << filename << endl;

        // Open input images
        vector<GeoImage> imgs;
        vector<std::string>::const_iterator iimgs;
        for (iimgs=imgnames.begin();iimgs!=imgnames.end();iimgs++) imgs.push_back(GeoImage(*iimgs));
        unsigned int bsz = imgs[0].NumBands();
        GDALDataType dtype = imgs[0].DataType();

        // Create output file based on input vector
        // TODO - GeoImage constructor that takes in vector
        OGRDataSource *poDS = OGRSFDriverRegistrar::Open(vectorname.c_str());
        OGRLayer *poLayer = poDS->GetLayer(0);
        OGRSpatialReference* vSRS = poLayer->GetSpatialRef();
        OGREnvelope _extent;
        poLayer->GetExtent(&_extent, true);
        Rect<double> extent(Point<double>(_extent.MinX,_extent.MinY),Point<double>(_extent.MaxX,_extent.MaxY));
        // Crop down to minimum bounding box if crop set
        if (crop) {
            // Find transformed union of all raster bounding boxes
            vector< Rect<double> > r_extents;
            for (vector<GeoImage>::const_iterator i=imgs.begin(); i!=imgs.end(); i++) {
                Rect<double> ext(Point<double>(i->LowerLeft()),Point<double>(i->TopRight()));
                ext.Transform(i->SRS(), *vSRS);
                r_extents.push_back(ext);
            }
            Rect<double> r_extent(r_extents[0]);
            for (vector< Rect<double> >::const_iterator r=r_extents.begin(); r!=r_extents.end(); r++) {
                r_extent.Union(*r);
            }
            // Limit to vector extent
            r_extent.Intersect(extent);

            // anchor to top left of vector (MinX, MaxY) and make multiple of resolution
            extent = Rect<double>(
                Point<double>(extent.x0() + std::floor((r_extent.x0()-extent.x0()) / xres) * xres, r_extent.y0()),
                Point<double>(r_extent.x1(), extent.y1() - std::floor((extent.y1()-r_extent.y1()) / yres) * yres)
            );
        }
        // Need to convert extent to resolution units
        int xsize = std::ceil(extent.width() / xres);
        int ysize = std::ceil(extent.height() / yres);
        GeoImage imgout(filename, xsize, ysize, bsz, dtype);
        imgout.CopyMeta(imgs[0]);
        imgout.CopyColorTable(imgs[0]);
        for (unsigned int b=0;b<bsz;b++) imgout[b].CopyMeta(imgs[0][b]);
        imgout.SetMeta(metadata);

        double affine[6];
        affine[0] = extent.x0();
        affine[1] = xres;
        affine[2] = 0;
        affine[3] = extent.y1();
        affine[4] = 0;
        affine[5] = -std::abs(yres);
        char* orig_wkt = NULL;
        poLayer->GetSpatialRef()->exportToWkt(&orig_wkt);
        imgout.GetGDALDataset()->SetProjection(orig_wkt);
        imgout.GetGDALDataset()->SetGeoTransform(affine);

        // Combine shape geometries into single geometry cutline
        OGRGeometry* site = OGRGeometryFactory::createGeometry( wkbMultiPolygon );
        OGRGeometry* poGeometry;
        OGRFeature *poFeature;
        poLayer->ResetReading();
        poFeature = poLayer->GetNextFeature();
        site = poFeature->GetGeometryRef();
        while( (poFeature = poLayer->GetNextFeature()) != NULL ) {
            poGeometry = poFeature->GetGeometryRef();
            if( poGeometry == NULL ) fprintf( stderr, "ERROR: Cutline feature without a geometry.\n" );
            //OGRwkbGeometryType eType = wkbFlatten(poGeometry->getGeometryType());
            site = site->Union(poGeometry);
            /*if( eType == wkbPolygon )
                site->addGeometry(poGeometry);
            else if( eType == wkbMultiPolygon ) {
                for(int iGeom = 0; iGeom < OGR_G_GetGeometryCount( poGeometry ); iGeom++ ) {
                    site->addGeometry( poGeometry->getGeometryRef(iGeom)  );
                }
            }
            else fprintf( stderr, "ERROR: Cutline not of polygon type.\n" );*/

            OGRFeature::DestroyFeature( poFeature );
        }
        OGRDataSource::DestroyDataSource( poDS );

        // Warp options
        GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
        psWarpOptions->hDstDS = imgout.GetGDALDataset();
        psWarpOptions->nBandCount = bsz;
        psWarpOptions->panSrcBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->panDstBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        for (unsigned int b=0;b<bsz;b++) {
            psWarpOptions->panSrcBands[b] = b+1;
            psWarpOptions->panDstBands[b] = b+1;
            psWarpOptions->padfSrcNoDataReal[b] = imgs[0][b].NoDataValue();
            psWarpOptions->padfDstNoDataReal[b] = imgout[b].NoDataValue();
            psWarpOptions->padfSrcNoDataImag[b] = 0.0;
            psWarpOptions->padfDstNoDataImag[b] = 0.0;
        }
        psWarpOptions->dfWarpMemoryLimit = Options::ChunkSize() * 1024.0 * 1024.0;
        if (Options::Verbose() > 2)
            psWarpOptions->pfnProgress = GDALTermProgress;
        else psWarpOptions->pfnProgress = GDALDummyProgress;

        char **papszOptions = NULL;
        //papszOptions = CSLSetNameValue(papszOptions,"SKIP_NOSOURCE","YES");
        papszOptions = CSLSetNameValue(papszOptions,"INIT_DEST","NO_DATA");
        papszOptions = CSLSetNameValue(papszOptions,"WRITE_FLUSH","YES");
        //papszOptions = CSLSetNameValue(papszOptions,"NUM_THREADS","ALL_CPUS");
        psWarpOptions->papszWarpOptions = papszOptions;

        for (vector<GeoImage>::iterator iimg=imgs.begin();iimg!=imgs.end();iimg++) {
            WarpToImage(*iimg, imgout, psWarpOptions, site);

            psWarpOptions->papszWarpOptions = CSLSetNameValue(psWarpOptions->papszWarpOptions,"INIT_DEST",NULL);
        }

        GDALDestroyWarpOptions( psWarpOptions );
        return imgout;
    }

    //! Fmask cloud mask
    GeoImage Fmask(const GeoImage& image, string filename, int tolerance, int dilate, dictionary metadata) {
        if (Options::Verbose() > 1)
            cout << "GIPPY: Fmask (tol=" << tolerance << ", d=" << dilate << ") - " << filename << endl;

        GeoImage imgout(filename, image, GDT_Byte, 5);
        int b_final(0); imgout[b_final].SetDescription("finalmask");
        int b_clouds(1);  imgout[b_clouds].SetDescription("cloudmask");
        int b_pcp(2);   imgout[b_pcp].SetDescription("PCP");
        int b_water(3); imgout[b_water].SetDescription("clearskywater");
        int b_land(4);  imgout[b_land].SetDescription("clearskyland");
        imgout.SetNoData(0);
        imgout.SetMeta(metadata);
        float nodataval(-32768);
        // Output probabilties (for debugging/analysis)
        GeoImage probout(filename + "-prob", image, GDT_Float32, 2);
        probout[0].SetDescription("wcloud");
        probout[1].SetDescription("lcloud");
        probout.SetNoData(nodataval);

        CImg<unsigned char> clouds, pcp, wmask, lmask, mask, redsatmask, greensatmask;
        CImg<float> red, nir, green, blue, swir1, swir2, BT, ndvi, ndsi, white, vprob;
        float _ndvi, _ndsi;
        long datapixels(0);
        long cloudpixels(0);
        long landpixels(0);
        //CImg<double> wstats(image.Size()), lstats(image.Size());
        //int wloc(0), lloc(0);

        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            blue = image["BLUE"].Read<double>(iChunk);
            red = image["RED"].Read<double>(iChunk);
            green = image["GREEN"].Read<double>(iChunk);
            nir = image["NIR"].Read<double>(iChunk);
            swir1 = image["SWIR1"].Read<double>(iChunk);
            swir2 = image["SWIR2"].Read<double>(iChunk);
            BT = image["LWIR"].Read<double>(iChunk);
            mask = image.NoDataMask(iChunk)^=1;
            ndvi = (nir-red).div(nir+red);
            ndsi = (green-swir1).div(green+swir1);
            white = image.Whiteness(iChunk);

            // Potential cloud pixels
            pcp =
                swir2.get_threshold(0.03)
                & BT.get_threshold(27,false,true)^=1
                // NDVI
                & ndvi.get_threshold(0.8,false,true)^=1
                // NDSI
                & ndsi.get_threshold(0.8,false,true)^=1
                // HazeMask
                & (blue - 0.5*red).threshold(0.08)
                & white.get_threshold(0.7,false,true)^=1
                & nir.get_div(swir1).threshold(0.75);

            redsatmask = image["RED"].SaturationMask(iChunk);
            greensatmask = image["GREEN"].SaturationMask(iChunk);
            vprob = red;
            // Calculate "variability probability"
            cimg_forXY(vprob,x,y) {
                _ndvi = (redsatmask(x,y) && nir(x,y) > red(x,y)) ? 0 : abs(ndvi(x,y));
                _ndsi = (greensatmask(x,y) && swir1(x,y) > green(x,y)) ? 0 : abs(ndsi(x,y));
                vprob(x,y) = 1 - std::max(white(x,y), std::max(_ndsi, _ndvi));
            }
            probout[1].Write(vprob, iChunk);

            datapixels += mask.sum();
            cloudpixels += pcp.sum();
            wmask = ((ndvi.get_threshold(0.01,false,true)^=1) &= (nir.get_threshold(0.01,false,true)^=1))|=
                    ((ndvi.get_threshold(0.1,false,true)^=1) &= (nir.get_threshold(0.05,false,true)^=1));

            imgout[b_pcp].Write(pcp.mul(mask), iChunk);        // Potential cloud pixels
            imgout[b_water].Write(wmask.get_mul(mask), iChunk);   // Clear-sky water
            CImg<unsigned char> landimg((wmask^1).mul(pcp^1).mul(mask));
            landpixels += landimg.sum();
            imgout[b_land].Write(landimg, iChunk);    // Clear-sky land
        }
        // floodfill....seems bad way
        //shadowmask = nir.draw_fill(nir.width()/2,nir.height()/2,)

        // If not enough non-cloud pixels then return existing mask
        if (cloudpixels >= (0.999*imgout[0].Size())) return imgout;
        // If not enough clear-sky land pixels then use all
        //GeoRaster msk;
        //if (landpixels < (0.001*imgout[0].Size())) msk = imgout[1];

        // Clear-sky water
        double Twater(image["LWIR"].AddMask(image["SWIR2"] < 0.03).AddMask(imgout[b_water]).AddMask(imgout[b_pcp]).Percentile(82.5));
        image["LWIR"].ClearMasks();
        GeoRaster landBT(image["LWIR"].AddMask(imgout[b_land]));
        image["LWIR"].ClearMasks();
        double Tlo(landBT.Percentile(17.5));
        double Thi(landBT.Percentile(82.5));

        if (Options::Verbose() > 2) {
            cout << "PCP = " << 100*cloudpixels/(double)datapixels << "%" << endl;
            cout << "Water (82.5%) = " << Twater << endl;
            cout << "Land (17.5%) = " << Tlo << ", (82.5%) = " << Thi << endl;
        }

        // Calculate cloud probabilities for over water and land
        CImg<float> wprob, lprob;
        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            mask = image.NoDataMask(iChunk)^=1;
            BT = image["LWIR"].Read<double>(iChunk);
            swir1 = image["SWIR1"].Read<double>(iChunk);

            // Water Clouds = temp probability x brightness probability
            wprob = ((Twater - BT)/=4.0).mul( swir1.min(0.11)/=0.11 ).mul(mask);
            probout[0].Write(wprob, iChunk);

            // Land Clouds = temp probability x variability probability
            vprob = probout[0].Read<double>(iChunk);
            lprob = ((Thi + 4-BT)/=(Thi+4-(Tlo-4))).mul( vprob ).mul(mask);
            //1 - image.NDVI(*iChunk).abs().max(image.NDSI(*iChunk).abs()).max(image.Whiteness(*iChunk).abs()) );
            probout[1].Write( lprob, iChunk);
        }

        // Thresholds
        float tol((tolerance-3)*0.1);
        float wthresh = 0.5 + tol;
        float lthresh(probout[1].AddMask(imgout[b_land]).Percentile(82.5)+0.2+tol);
        probout[1].ClearMasks();
        if (Options::Verbose() > 2)
            cout << "Thresholds: water = " << wthresh << ", land = " << lthresh << endl;

        // 3x3 filter of 1's for majority filter
        //CImg<int> filter(3,3,1,1, 1);
        int erode = 5;
        int padding(double(std::max(dilate,erode)+1)/2);
        for (unsigned int b=0;b<image.NumBands();b++) image[b].SetChunkPadding(padding);
        for (unsigned int b=0;b<imgout.NumBands();b++) imgout[b].SetChunkPadding(padding);

        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            mask = image.NoDataMask(iChunk)^=1;
            pcp = imgout[b_pcp].Read<double>(iChunk);
            wmask = imgout[b_water].Read<double>(iChunk);
            BT = image["LWIR"].Read<double>(iChunk);

            lprob = probout[1].Read<double>(iChunk);
            
            clouds = 
                (pcp & wmask & wprob.threshold(0.5))|=
                (pcp & (wmask^1) & lprob.threshold(lthresh))|=
                (lprob.get_threshold(0.99) & (wmask^1))|=
                (BT.threshold(Tlo-35,false,true)^=1);

            // Majority filter
            //mask.convolve(filter).threshold(5);
            if (erode > 0)
                clouds.erode(erode, erode);
            if (dilate > 0)
                clouds.dilate(dilate, dilate);

            //cimg_forXY(nodatamask,x,y) if (!nodatamask(x,y)) mask(x,y) = 0;
            clouds.mul(mask);
            imgout[b_clouds].Write(clouds, iChunk);
            imgout[b_final].Write((clouds^=1).mul(mask), iChunk);
        }

        return imgout;
    }

    //! k-means unsupervised classifier
    /*GeoImage kmeans( const GeoImage& image, string filename, int classes, int iterations, float threshold ) {
        //if (Image.NumBands() < 2) throw GIP::Gexceptions::errInvalidParams("At least two bands must be supplied");
        if (Options::Verbose()) {
            cout << image.Basename() << " - k-means unsupervised classifier:" << endl
                << "  Classes = " << classes << endl
                << "  Iterations = " << iterations << endl
                << "  Pixel Change Threshold = " << threshold << "%" << endl;
        }
        // Calculate threshold in # of pixels
        threshold = threshold/100.0 * image.Size();

        GeoImageIO<float> img(image);
        // Create new output image
        GeoImageIO<unsigned char> imgout(GeoImage(filename, image, GDT_Byte, 1));

        // Get initial class estimates (uses random pixels)
        CImg<float> ClassMeans = img.GetPixelClasses(classes);

        int i;
        CImg<double> Pixel, C_img, DistanceToClass(classes), NumSamples(classes), ThisClass;
        CImg<unsigned char> C_imgout, C_mask;
        CImg<double> RunningTotal(classes,image.NumBands(),1,1,0);

        int NumPixelChange, iteration=0;
        do {
            NumPixelChange = 0;
            for (i=0; i<classes; i++) NumSamples(i) = 0;
            if (Options::Verbose()) cout << "  Iteration " << iteration+1 << std::flush;

            for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
                C_img = img.Read(iChunk);
                C_mask = img.NoDataMask(iChunk);
                C_imgout = imgout[0].Read(iChunk);

                CImg<double> stats;
                cimg_forXY(C_img,x,y) { // Loop through image
                    // Calculate distance between this pixel and all classes
                    if (C_mask(x,y)) {
                        Pixel = C_img.get_crop(x,y,0,0,x,y,0,C_img.spectrum()-1).unroll('x');
                        cimg_forY(ClassMeans,cls) {
                            ThisClass = ClassMeans.get_row(cls);
                            DistanceToClass(cls) = (Pixel - ThisClass).dot(Pixel - ThisClass);
                        }
                        // Get closest distance and see if it's changed since last time
                        stats = DistanceToClass.get_stats();
                        if (C_imgout(x,y) != (stats(4)+1)) {
                            NumPixelChange++;
                            C_imgout(x,y) = stats(4)+1;
                        }
                        NumSamples(stats(4))++;
                        cimg_forY(RunningTotal,iband) RunningTotal(stats(4),iband) += Pixel(iband);
                    } else C_imgout(x,y) = 0;
                }
                imgout[0].Write(C_imgout,iChunk);
                if (Options::Verbose()) cout << "." << std::flush;
            }

            // Calculate new Mean class vectors
            for (i=0; i<classes; i++) {
                if (NumSamples(i) > 0) {
                    cimg_forX(ClassMeans,x) {
                        ClassMeans(x,i) = RunningTotal(i,x)/NumSamples(i);
                        RunningTotal(i,x) = 0;
                    }
                    NumSamples(i) = 0;
                }
            }
            if (Options::Verbose()) cout << 100.0*((double)NumPixelChange/image.Size()) << "% pixels changed class" << endl;
            if (Options::Verbose()>1) cimg_printclasses(ClassMeans);
        } while ( (++iteration < iterations) && (NumPixelChange > threshold) );

        imgout[0].SetDescription("k-means");
        //imgout.GetGDALDataset()->FlushCache();
        return imgout;
    }*/

    //void Indices(const GeoImage& ImageIn, string basename, std::vector<std::string> products) {
    dictionary Indices(const GeoImage& image, dictionary products, dictionary metadata) {
        if (Options::Verbose() > 1) std::cout << "GIPPY: Indices" << std::endl;

        float nodataout = -32768;

        std::map< string, GeoImage > imagesout;
        std::map<string, string>::const_iterator iprod;
        std::map<string, string> filenames;
        string prodname;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            //imagesout[*iprod] = GeoImageIO<float>(GeoImage(basename + '_' + *iprod, image, GDT_Int16));
            if (Options::Verbose() > 2) cout << iprod->first << " -> " << iprod->second << endl;
            prodname = iprod->first;
            imagesout[prodname] = GeoImage(iprod->second, image, GDT_Int16, 1);
            imagesout[prodname].SetNoData(nodataout);
            imagesout[prodname].SetGain(0.0001);
            imagesout[prodname].SetUnits("other");
            imagesout[prodname].SetMeta(metadata);
            imagesout[prodname][0].SetDescription(prodname);
            filenames[prodname] = imagesout[prodname].Filename();
        }
        if (imagesout.size() == 0) throw std::runtime_error("No indices selected for calculation!");

        std::map< string, std::vector<string> > colors;
        colors["ndvi"] = {"NIR","RED"};
        colors["evi"] = {"NIR","RED","BLUE"};
        colors["lswi"] = {"NIR","SWIR1"};
        colors["ndsi"] = {"SWIR1","GREEN"};
        colors["ndwi"] = {"GREEN","NIR"};
        colors["bi"] = {"BLUE","NIR"};
        colors["satvi"] = {"SWIR1","RED", "SWIR2"};
        colors["msavi2"] = {"NIR","RED"};
        colors["vari"] = {"RED","GREEN","BLUE"};
        colors["brgt"] = {"RED","GREEN","BLUE","NIR"};
        // Tillage indices
        colors["ndti"] = {"SWIR2","SWIR1"};
        colors["crc"] = {"SWIR1","SWIR2","BLUE"};
        colors["crcm"] = {"SWIR1","SWIR2","GREEN"};
        colors["isti"] = {"SWIR1","SWIR2"};
        colors["sti"] = {"SWIR1","SWIR2"};

        // Figure out what colors are needed
        std::set< string > used_colors;
        std::set< string >::const_iterator isstr;
        std::vector< string >::const_iterator ivstr;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            for (ivstr=colors[iprod->first].begin();ivstr!=colors[iprod->first].end();ivstr++) {
                used_colors.insert(*ivstr);
            }
        }
        if (Options::Verbose() > 2) {
            cout << "Colors used: ";
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) cout << " " << *isstr;
            cout << endl;
        }

        CImg<float> red, green, blue, nir, swir1, swir2, cimgout, cimgmask, tmpimg;

        // need to add overlap
        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            if (Options::Verbose() > 3) cout << "Chunk " << iChunk << " of " << image[0].NumChunks() << endl;
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) {
                if (*isstr == "RED") red = image["RED"].Read<float>(iChunk);
                else if (*isstr == "GREEN") green = image["GREEN"].Read<float>(iChunk);
                else if (*isstr == "BLUE") blue = image["BLUE"].Read<float>(iChunk);
                else if (*isstr == "NIR") nir = image["NIR"].Read<float>(iChunk);
                else if (*isstr == "SWIR1") swir1 = image["SWIR1"].Read<float>(iChunk);
                else if (*isstr == "SWIR2") swir2 = image["SWIR2"].Read<float>(iChunk);
            }

            for (iprod=products.begin(); iprod!=products.end(); iprod++) {
                prodname = iprod->first;
                //string pname = iprod->toupper();
                if (prodname == "ndvi") {
                    cimgout = (nir-red).div(nir+red);
                } else if (prodname == "evi") {
                    cimgout = 2.5*(nir-red).div(nir + 6*red - 7.5*blue + 1);
                } else if (prodname == "lswi") {
                    cimgout = (nir-swir1).div(nir+swir1);
                } else if (prodname == "ndsi") {
                    cimgout = (green-swir1).div(green+swir1);
                } else if (prodname == "ndwi") {
                    cimgout = (green-nir).div(green+nir);
                } else if (prodname == "bi") {
                    cimgout = 0.5*(blue+nir);
                } else if (prodname == "satvi") {
                    float L(0.5);
                    cimgout = (((1.0+L)*(swir1 - red)).div(swir1+red+L)) - (0.5*swir2);
                } else if (prodname == "msavi2") {
                    tmpimg = (nir*2)+1;
                    cimgout = (tmpimg - (tmpimg.pow(2) - ((nir-red)*8).sqrt())) * 0.5;
                } else if (prodname == "vari") {
                    cimgout = (green-red).div(green+red-blue);
                } else if (prodname == "brgt") {
                    cimgout = (0.3*blue + 0.3*red + 0.1*nir + 0.3*green);
                // Tillage indices
                } else if (prodname == "ndti") {
                    cimgout = (swir1-swir2).div(swir1+swir2);
                } else if (prodname == "crc") {
                    cimgout = (swir1-blue).div(swir2+blue);
                } else if (prodname == "crcm") {
                    cimgout = (swir1-green).div(swir2+green);
                } else if (prodname == "isti") {
                    cimgout = swir2.div(swir1);
                } else if (prodname == "sti") {
                    cimgout = swir1.div(swir2);
                }
                // TODO don't read mask again...create here
                cimgmask = image.NoDataMask(iChunk, colors[prodname]);
                cimg_forXY(cimgout,x,y) if (cimgmask(x,y)) cimgout(x,y) = nodataout;
                imagesout[prodname].Write(cimgout,iChunk);
            }
        }
        return filenames;
    }

    //! Perform linear transform with given coefficients (e.g., PC transform)
    GeoImage LinearTransform(const GeoImage& img, string filename, CImg<float> coef) {
        // Verify size of array
        unsigned int numbands = img.NumBands();
        if ((coef.height() != (int)numbands) || (coef.width() != (int)numbands))
            throw std::runtime_error("Coefficient array needs to be of size NumBands x NumBands!");
        float nodataout = -32768;
        GeoImage imgout(filename, img, GDT_Float32);
        imgout.SetNoData(nodataout);
        //imgout.SetGain(0.0001);
        imgout.CopyMeta(img);
        CImg<float> cimg;
        CImg<unsigned char> mask;
        for (unsigned int bout=0; bout<numbands; bout++) {
            //if (Options::Verbose() > 4) cout << "Band " << bout << endl;
            for (unsigned int iChunk=1; iChunk<=imgout[bout].NumChunks(); iChunk++) {
                cimg = img[0].Read<float>(iChunk) * coef(0, bout);;
                for (unsigned int bin=1; bin<numbands; bin++) {
                    cimg = cimg + (img[bin].Read<float>(iChunk) * coef(bin, bout));
                }
                mask = img.NoDataMask(iChunk);
                cimg_forXY(cimg,x,y) if (mask(x,y)) cimg(x,y) = nodataout;
                imgout[bout].Write(cimg, iChunk);
            }
        }
        return imgout;
    }

    //! Calculate spectral statistics and output to new image
    GeoImage SpectralStatistics(const GeoImage& img, string filename) {
        if (img.NumBands() < 2) {
            throw std::runtime_error("Must have at least 2 bands!");
        }
        GeoImage imgout(filename, img, GDT_Float32, 2);
        imgout.SetNoData(img[0].NoDataValue());
        imgout.CopyMeta(img);
        imgout.SetBandName("Mean", 1);
        imgout.SetBandName("StdDev", 2);

        CImgList<double> stats;
        for (unsigned int iChunk=1; iChunk<=img.NumChunks(); iChunk++) {
            if (Options::Verbose() > 2) 
                std::cout << "Processing chunk " << iChunk << " of " << img.NumChunks() << std::endl;
            stats = img.SpectralStatistics(iChunk);
            imgout[0].Write(stats[0], iChunk);
            imgout[1].Write(stats[1], iChunk);
        }
        if (Options::Verbose())
            std::cout << "Spectral statistics written to " << imgout.Filename() << std::endl;
        return imgout;
    }

    //! Spectral Matched Filter, with missing data
    /*GeoImage SMF(const GeoImage& image, string filename, CImg<double> Signature) {
        GeoImage output(filename, image, GDT_Float32, 1);

        // Band Means
        CImg<double> means(image.NumBands());
        for (unsigned int b=0;b<image.NumBands();b++) means(b) = image[b].Mean();

        //vector< box<point> > Chunks = ImageIn.Chunk();
        return output;
    }*/

    /*
    CImg<double> SpectralCorrelation(const GeoImage& image, CImg<double> covariance) {
        // Correlation matrix
        if (covariance.size() == 0) covariance = SpectralCovariance(image);

        unsigned int NumBands = image.NumBands();
        unsigned int b;

        // Subtract Mean
        //CImg<double> means(NumBands);
        //for (b=0; b<NumBands; b++) means(b) = image[b].Mean();
        //covariance -= (means.get_transpose() * means);

        CImg<double> stddev(NumBands);
        for (b=0; b<NumBands; b++) stddev(b) = image[b].StdDev();
        CImg<double> Correlation = covariance.div(stddev.get_transpose() * stddev);

        if (Options::Verbose() > 0) {
            cout << image.Basename() << " Spectral Correlation Matrix:" << endl;
            cimg_forY(Correlation,y) {
                cout << "\t";
                cimg_forX(Correlation,x) {
                    cout << std::setw(18) << Correlation(x,y);
                }
                cout << endl;
            }
        }

        return Correlation;
    }*/

    /*CImg<double> SpectralCovariance(const GeoImage& image) {
        typedef double T;

        GeoImageIO<T> img(image);

        unsigned int NumBands(image.NumBands());
        CImg<double> Covariance(NumBands, NumBands);

        // Calculate Covariance
        vector<bbox> Chunks = image.Chunk();
        vector<bbox>::const_iterator iChunk;
        CImg<T> bandchunk;
        CImg<unsigned char> mask;
        for (iChunk=Chunks.begin(); iChunk!=Chunks.end(); iChunk++) {
            int chunksize = boost::geometry::area(*iChunk);
            CImg<T> matrixchunk(NumBands, chunksize);
            mask = img.NoDataMask(*iChunk);
            int validsize = mask.size() - mask.sum();

            int p(0);
            for (unsigned int b=0;b<NumBands;b++) {
                cout << "band" << b << endl;
                CImg<T> bandchunk( img[b].Read(*iChunk) );
                p = 0;
                cimg_forXY(bandchunk,x,y) {
                    if (mask(x,y)==0) matrixchunk(b,p++) = bandchunk(x,y);
                }
                //cout << "p = " << matrixchunk[p-1] << endl;
            }
            if (p != (int)image.Size()) matrixchunk.crop(0,0,NumBands-1,p-1);
            Covariance += (matrixchunk.get_transpose() * matrixchunk)/(validsize-1);
        }
        cout << "done cov" << endl;
        // Subtract Mean
        CImg<double> means(NumBands);
        for (unsigned int b=0; b<NumBands; b++) means(b) = image[b].Mean(); //cout << "Mean b" << b << " = " << means(b) << endl; }
        Covariance -= (means.get_transpose() * means);

        if (Options::Verbose() > 0) {
            cout << image.Basename() << " Spectral Covariance Matrix:" << endl;
            cimg_forY(Covariance,y) {
                cout << "\t";
                cimg_forX(Covariance,x) {
                    cout << std::setw(18) << Covariance(x,y);
                }
                cout << endl;
            }
        }
        return Covariance;
    }*/


} // namespace gip
