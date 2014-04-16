/*
 * gip_GeoAlgorithms.cpp
 *
 *  Created on: Aug 26, 2011
 *      Author: mhanson
 */
#define _USE_MATH_DEFINES
#include <cmath>
#include <set>

#include <gip/GeoAlgorithms.h>
#include <gip/gip_CImg.h>

#include <gdal/ogrsf_frmts.h>
#include <gdal/gdalwarper.h>

class CutlineTransformer : public OGRCoordinateTransformation
{
public:

    void         *hSrcImageTransformer;

    virtual OGRSpatialReference *GetSourceCS() { return NULL; }
    virtual OGRSpatialReference *GetTargetCS() { return NULL; }

    virtual int Transform( int nCount, double *x, double *y, double *z = NULL ) {
        int nResult;

        int *pabSuccess = (int *) CPLCalloc(sizeof(int),nCount);
        nResult = TransformEx( nCount, x, y, z, pabSuccess );
        CPLFree( pabSuccess );

        return nResult;
    }

    virtual int TransformEx( int nCount, double *x, double *y, double *z = NULL, int *pabSuccess = NULL ) {
        return GDALGenImgProjTransform( hSrcImageTransformer, TRUE, nCount, x, y, z, pabSuccess );
    }
};

namespace gip {
    using std::string;
    using std::vector;
    using std::cout;
    using std::cerr;
    using std::endl;

    //! Create mask based on NoData values for all bands
    /*GeoRaster CreateMask(const GeoImage& image, string filename) {
        typedef float T;

        //CImg<T> imgchunk;

        // if (filename == "") filename = image.Basename() + "_mask";
        GeoImage mask(filename, image, GDT_Byte, 1);
        CImg<unsigned char> imgout;
        mask.SetNoData(0);
        //unsigned int validpixels(0);
        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            //imgchunk = imageIO[0].Read(*iChunk);
            //imgout = Pbands[0].NoDataMask(imgchunk);
            //for (unsigned int b=1;b<image.NumBands();b++) {
            //        imgout &= Pbands[b].NoDataMask( Pbands[b].Read(*iChunk) );
            //}
            //validpixels += imgout.sum();
            imgout = image.NoDataMask(iChunk);
            mask[0].Write(imgout,iChunk);
        }
        //mask[0].SetValidSize(validpixels);
        return mask[0];
    }*/

    //! Generate 3-band RGB image scaled to 1 byte for easy viewing
    /*GeoImage RGB(const GeoImage& image, string filename) {
        GeoImageIO<float> img(image);
        img.SetUnitsOut("reflectance");
        img.PruneToRGB();
        GeoImageIO<unsigned char> imgout(GeoImage(filename, img, GDT_Byte));
        imgout.SetNoData(0);
        imgout.SetUnits("other");
        CImg<float> stats, cimg;
        CImg<unsigned char> mask;
        for (unsigned int b=0;b<img.NumBands();b++) {
            stats = img[b].Stats();
            float lo = std::max(stats(2) - 3*stats(3), stats(0)-1);
            float hi = std::min(stats(2) + 3*stats(3), stats(1));
            for (unsigned int iChunk=1; iChunk<=img[b].NumChunks(); iChunk++) {
                cimg = img[b].Read(iChunk);
                mask = img[b].NoDataMask(iChunk);
                ((cimg-=lo)*=(255.0/(hi-lo))).max(0.0).min(255.0);
                //cimg_printstats(cimg,"after stretch");
                cimg_forXY(cimg,x,y) { if (!mask(x,y)) cimg(x,y) = imgout[b].NoDataValue(); }
                imgout[b].Write(CImg<unsigned char>().assign(cimg.round()),iChunk);
            }
        }
        return imgout;
    }*/

    //! Merge images into one file and crop to vector
    GeoImage CookieCutter(vector<std::string> imgnames, string filename, string vectorname, float xres, float yres) {
        // TODO - pass in vector of GeoRaster's instead
        if (Options::Verbose() > 2) {
            cout << filename << ": CookieCutter" << endl;
        }

        // Open input images
        vector<GeoImage> imgs;
        vector<std::string>::const_iterator iimgs;
        for (iimgs=imgnames.begin();iimgs!=imgnames.end();iimgs++) imgs.push_back(GeoImage(*iimgs));
        unsigned int bsz = imgs[0].NumBands();
        GDALDataType dtype = imgs[0].DataType();

        // Create output file based on input vector
        OGRDataSource *poDS = OGRSFDriverRegistrar::Open(vectorname.c_str());
        OGRLayer *poLayer = poDS->GetLayer(0);
        OGREnvelope extent;
        poLayer->GetExtent(&extent, true);
        // Need to convert extent to resolution units
        int xsize = (int)(0.5 + (extent.MaxX - extent.MinX) / xres);
        int ysize = (int)(0.5 + (extent.MaxY - extent.MinY) / yres);
        GeoImage imgout(filename, xsize, ysize, bsz, dtype);
        imgout.CopyMeta(imgs[0]);
        imgout.CopyColorTable(imgs[0]);
        for (unsigned int b=0;b<bsz;b++) imgout[b].CopyMeta(imgs[0][b]);

        double affine[6];
        affine[0] = extent.MinX;
        affine[1] = xres;
        affine[2] = 0;
        affine[3] = extent.MaxY;
        affine[4] = 0;
        affine[5] = -yres;
        char* wkt = NULL;
        poLayer->GetSpatialRef()->exportToWkt(&wkt);
        imgout.GetGDALDataset()->SetProjection(wkt);
        imgout.GetGDALDataset()->SetGeoTransform(affine);
        // Compute union
        /*OGRPolygon* site; // = new OGRPolygon();
        OGRFeature *poFeature;
        poLayer->ResetReading();
        while( (poFeature = poLayer->GetNextFeature()) != NULL )
        {
            OGRGeometry *poGeometry;
            poGeometry = poFeature->GetGeometryRef();

            site = (OGRPolygon*)site->Union(poGeometry);
            OGRFeature::DestroyFeature( poFeature );
        }*/

        // Combine shape geoemtries into single geometry cutline
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

        // Cutline transform to pixel coordinates
        char **papszOptionsCutline = NULL;
        //papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "DST_SRS", wkt );
        //papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "SRC_SRS", wkt );
        //papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "INSERT_CENTER_LONG", "FALSE" );
        CutlineTransformer oTransformer;

        /* The cutline transformer will *invert* the hSrcImageTransformer */
        /* so it will convert from the cutline SRS to the source pixel/line */
        /* coordinates */
        oTransformer.hSrcImageTransformer = GDALCreateGenImgProjTransformer2( imgout.GetGDALDataset(), NULL, papszOptionsCutline );
        site->transform(&oTransformer);

        GDALDestroyGenImgProjTransformer( oTransformer.hSrcImageTransformer );
        CSLDestroy( papszOptionsCutline );

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
        if (Options::Verbose() > 2)
            psWarpOptions->pfnProgress = GDALTermProgress;
        else psWarpOptions->pfnProgress = GDALDummyProgress;
        char **papszOptions = NULL;
        //psWarpOptions->hCutline = site;
        //papszOptions = CSLSetNameValue(papszOptions,"SKIP_NOSOURCE","YES");
        papszOptions = CSLSetNameValue(papszOptions,"INIT_DEST","NO_DATA");
        papszOptions = CSLSetNameValue(papszOptions,"WRITE_FLUSH","YES");
        papszOptions = CSLSetNameValue(papszOptions,"NUM_THREADS","ALL_CPUS");
        //site->exportToWkt(&wkt);
        //papszOptions = CSLSetNameValue(papszOptions,"CUTLINE",wkt);
        psWarpOptions->papszWarpOptions = CSLDuplicate(papszOptions);

        GDALWarpOperation oOperation;
        // Perform warp for each input file
        vector<GeoImage>::iterator iimg;
        for (iimg=imgs.begin();iimg!=imgs.end();iimg++) {
            if (Options::Verbose() > 2) cout << iimg->Basename() << " warping " << std::flush;
            psWarpOptions->hSrcDS = iimg->GetGDALDataset();
            psWarpOptions->pTransformerArg =
                GDALCreateGenImgProjTransformer( iimg->GetGDALDataset(), iimg->GetGDALDataset()->GetProjectionRef(),
                                                imgout.GetGDALDataset(), imgout.GetGDALDataset()->GetProjectionRef(), TRUE, 0.0, 0 );
            psWarpOptions->pfnTransformer = GDALGenImgProjTransform;
            oOperation.Initialize( psWarpOptions );
            //if (Options::Verbose() > 3) cout << "Error: " << CPLGetLastErrorMsg() << endl;
            oOperation.ChunkAndWarpMulti( 0, 0, imgout.XSize(), imgout.YSize() );

            GDALDestroyGenImgProjTransformer( psWarpOptions->pTransformerArg );
            psWarpOptions->papszWarpOptions = CSLSetNameValue(psWarpOptions->papszWarpOptions,"INIT_DEST",NULL);
        }
        GDALDestroyWarpOptions( psWarpOptions );

        return imgout;
    }

    //void Indices(const GeoImage& ImageIn, string basename, std::vector<std::string> products) {
    std::map<std::string, std::string> Indices(const GeoImage& image, std::map<std::string, std::string> products) {
        float nodataout = -32768;

        std::map< string, GeoImage > imagesout;
        std::map<string, string>::const_iterator iprod;
        std::map<string, string> filenames;
        string prodname;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            //imagesout[*iprod] = GeoImageIO<float>(GeoImage(basename + '_' + *iprod, image, GDT_Int16));
            if (Options::Verbose() > 2) cout << iprod->first << ", " << iprod->second << endl;
            prodname = iprod->first;
            imagesout[prodname] = GeoImage(iprod->second, image, GDT_Int16, 1);
            imagesout[prodname].SetNoData(nodataout);
            imagesout[prodname].SetGain(0.0001);
            imagesout[prodname].SetUnits("other");
            imagesout[prodname][0].SetDescription(prodname);
            filenames[prodname] = imagesout[prodname].Filename();
        }
        if (imagesout.size() == 0) throw std::runtime_error("No indices selected for calculation!");

        std::map< string, std::vector<string> > colors;
        colors["ndvi"] = {"NIR","RED"};
        colors["evi"] = {"NIR","RED","BLUE"};
        colors["lswi"] = {"NIR","SWIR1"};
        colors["ndsi"] = {"SWIR1","GREEN"};
        colors["bi"] = {"BLUE","NIR"};
        colors["satvi"] = {"SWIR1","RED"};
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

        CImg<float> red, green, blue, nir, swir1, swir2, cimgout, cimgmask;

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
                if (Options::Verbose() > 4) cout << "Product " << prodname << endl;
                //cout << "Products: " << prodname << std::flush;
                //string pname = iprod->toupper();
                if (prodname == "ndvi") {
                    cimgout = (nir-red).div(nir+red);
                } else if (prodname == "evi") {
                    cimgout = 2.5*(nir-red).div(nir + 6*red - 7.5*blue + 1);
                } else if (prodname == "lswi") {
                    cimgout = (nir-swir1).div(nir+swir1);
                } else if (prodname == "ndsi") {
                    cimgout = (green-swir1).div(green+swir1);
                } else if (prodname == "bi") {
                    cimgout = 0.5*(blue+nir);
                } else if (prodname == "satvi") {
                    float L(0.5);
                    cimgout = (((1.0+L)*(swir1 - red)).div(swir1+red+L)) - (0.5*swir2);
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
                //if (Options::Verbose() > 2) cout << "Getting mask" << endl;
                // TODO don't read mask again...create here
                cimgmask = image.NoDataMask(iChunk, colors[prodname]);
                cimg_forXY(cimgout,x,y) if (cimgmask(x,y)) cimgout(x,y) = nodataout;
                imagesout[prodname].Write(cimgout,iChunk);
            }
        }
        return filenames;
    }

    //! Auto cloud mask - toaref input
    /*GeoImage AutoCloud(const GeoImage& image, string filename, int cheight, float minred, float maxtemp, float maxndvi, int morph) {
        typedef float outtype;
        GeoImageIO<float> imgin(image);
        GeoImageIO<outtype> imgout(GeoImage(filename, image, GDT_Byte, 1));
        imgout.SetNoData(0);

        CImg<float> red, nir, temp, ndvi;
        CImg<outtype> mask;

        // need to add overlap
        for (int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {

            red = imgin["RED"].Ref(iChunk);
            temp = imgin["LWIR"].Ref(iChunk);
            nir = imgin["NIR"].Ref(iChunk);
            ndvi = (nir-red).div(nir+red);

            mask =
                temp.threshold(maxtemp,false,true)^=1 &
                ndvi.threshold(maxndvi,false,true)^=1 &
                red.get_threshold(minred);

            if (morph != 0) mask.dilate(morph);

            imgout[0].Write(mask,iChunk);
            //CImg<double> stats = img.get_stats();
            //cout << "stats " << endl;
            //for (int i=0;i<12;i++) cout << stats(i) << " ";
            //cout << endl;
        }
        return imgout;
    }*/

    /** ACCA (Automatic Cloud Cover Assessment). Takes in TOA Reflectance,
     * temperature, sun elevation, solar azimuth, and number of pixels to
     * dilate.
     */
    GeoImage ACCA(const GeoImage& image, std::string filename, float se_degrees,
                  float sa_degrees, int erode, int dilate, int cloudheight ) {
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

        vector<string> bands_used({"RED","GREEN","NIR","SWIR1","LWIR"});

        CImg<float> red, green, nir, swir1, temp, ndsi, b56comp;
        CImg<unsigned char> nonclouds, ambclouds, clouds, mask, temp2;
        float cloudsum(0), scenesize(0);

        if (Options::Verbose()) cout << image.Basename() << " - ACCA" << endl;
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

        for (unsigned int b=0;b<imgout.NumBands();b++) imgout[b].Chunk(padding);
        for (unsigned int b=0;b<image.NumBands();b++) image[b].Chunk(padding);

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

    //! Fmask cloud mask
    GeoImage Fmask(const GeoImage& image, string filename, int tolerance, int dilate) {
        if (Options::Verbose() > 1)
            std::cout << image.Basename() << ": Fmask - dilate(" << dilate << ")" << std::endl;

        GeoImage imgout(filename, image, GDT_Byte, 5);
        int b_final(0); imgout[b_final].SetDescription("finalmask");
        int b_clouds(1);  imgout[b_clouds].SetDescription("cloudmask");
        int b_pcp(2);   imgout[b_pcp].SetDescription("PCP");
        int b_water(3); imgout[b_water].SetDescription("clearskywater");
        int b_land(4);  imgout[b_land].SetDescription("clearskyland");
        imgout.SetNoData(0);
        float nodataval(-32768);
        // Output probabilties (for debugging/analysis)
        GeoImage probout(filename + "_prob", image, GDT_Float32, 2);
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
        for (unsigned int b=0;b<image.NumBands();b++) image[b].Chunk(padding);
        for (unsigned int b=0;b<imgout.NumBands();b++) imgout[b].Chunk(padding);

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

    //! Convert lo-high of index into probability
    /*GeoImage Index2Probability(const GeoImage& image, string filename, float min, float max) {
        // Need method of generating new GeoImage with GeoRaster template in
        int bandnum = 1;
        GeoImageIO<float> imagein(image);
        GeoImageIO<float> imageout(GeoImage(filename, image, GDT_Float32, 2));
        float nodatain = imagein[0].NoDataValue();
        float nodataout = -32768;
        imageout.SetNoData(nodataout);

        CImg<float> cimgin, cimgout;
        for (unsigned int iChunk=1; iChunk<=image[bandnum-1].NumChunks(); iChunk++) {
            cimgin = imagein[bandnum-1].Read(iChunk);
            cimgout = (cimgin - min)/(max-min);
            cimgout.min(1.0).max(0.0);
            cimg_forXY(cimgin,x,y) if (cimgin(x,y) == nodatain) cimgout(x,y) = nodataout;
            imageout[0].Write(cimgout, iChunk);
            cimg_for(cimgout,ptr,float) if (*ptr != nodataout) *ptr = 1.0 - *ptr;
            imageout[1].Write(cimgout, iChunk);
        }
        return imageout;
    }*/

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

    //! Rice detection algorithm
    /*GeoImage RiceDetect(const GeoImage& image, string filename, vector<int> days, float th0, float th1, int dth0, int dth1) {
        if (Options::Verbose() > 1) cout << "RiceDetect(" << image.Basename() << ") -> " << filename << endl;

        GeoImageIO<float> img(image);
        GeoImageIO<unsigned char> imgout(GeoImage(filename, image, GDT_Byte, img.NumBands()));
        imgout.SetNoData(0);
        imgout[0].SetDescription("rice");
        for (unsigned int b=1;b<img.NumBands();b++) {
            imgout[b].SetDescription("day"+to_string(days[b]));
        }

        CImg<float> cimg;
        CImg<unsigned char> cimg_nodata, cimg_dmask;
        CImg<int> cimg_th0, cimg_flood;

        for (unsigned int iChunk=1; iChunk<=img[0].NumChunks(); iChunk++) {
            if (Options::Verbose() > 3) cout << "Chunk " << iChunk << " of " << img[0].NumChunks() << endl;
            cimg = img[0].Read(iChunk);
            cimg_nodata = img[0].NoDataMask(iChunk);
            int delta_day(0);
            CImg<int> DOY(cimg.width(), cimg.height(), 1, 1, 0);
            CImg<int> cimg_rice(cimg.width(), cimg.height(), 1, 1, 0);
            cimg_flood = (cimg.get_threshold(th0)^=1).mul(cimg_nodata);

            for (unsigned int b=1;b<image.NumBands();b++) {
                if (Options::Verbose() > 3) cout << "Day " << days[b] << endl;
                delta_day = days[b]-days[b-1];
                cimg = img[b].Read(iChunk);
                cimg_nodata = img[b].NoDataMask(iChunk);
                cimg_th0 = cimg.get_threshold(th0)|=(cimg_nodata^1);    // >= th0 and assume nodata >= th0

                DOY += delta_day;                                       // running total of days
                DOY.mul(cimg_flood);                                    // reset if it hasn't been flooded yet
                DOY.mul(cimg_th0);                                      // reset if in hydroperiod

                cimg_dmask = DOY.get_threshold(dth1,false,true)^=1;      // mask of where past high date
                DOY.mul(cimg_dmask);

                // locate (and count) where rice criteria met
                CImg<unsigned char> newrice = cimg.threshold(th1,false,true) & DOY.get_threshold(dth0,false,true);
                cimg_rice = cimg_rice + newrice;

                // update flood map
                cimg_flood |= (cimg_th0^=1);
                // remove new found rice pixels, and past high date
                cimg_flood.mul(newrice^=1).mul(cimg_dmask);

                imgout[b].Write(DOY, iChunk);
            }
            imgout[0].Write(cimg_rice,iChunk);              // rice map count
        }
        return imgout;
    }*/

    // Perform band math (hard coded subtraction)
    /*GeoImage BandMath(const GeoImage& image, string filename, int band1, int band2) {
        GeoImageIO<float> imagein(image);
        GeoImageIO<float> imageout(GeoImage(filename, image, GDT_Float32, 1));
        float nodataout = -32768;
        imageout.SetNoData(nodataout);
        std::vector<bbox>::const_iterator iChunk;
        std::vector<bbox> Chunks = image.Chunk();
        CImg<float> cimgout;
        CImg<unsigned char> mask;
        for (iChunk=Chunks.begin(); iChunk!=Chunks.end(); iChunk++) {
            mask = imagein[band1-1].NoDataMask(*iChunk)|=(imagein[band2-1].NoDataMask(*iChunk));
            cimgout = imagein[band1-1].Read(*iChunk) - imagein[band2-1].Read(*iChunk);
            cimg_forXY(mask,x,y) if (mask(x,y)) cimgout(x,y) = nodataout;
            imageout[0].WriteChunk(cimgout,*iChunk);
        }
        return imageout;
    }*/

    //! Spectral Matched Filter, with missing data
    /*GeoImage SMF(const GeoImage& image, string filename, CImg<double> Signature) {
        GeoImage output(filename, image, GDT_Float32, 1);

        // Band Means
        CImg<double> means(image.NumBands());
        for (unsigned int b=0;b<image.NumBands();b++) means(b) = image[b].Mean();

        //vector< box<point> > Chunks = ImageIn.Chunk();
        return output;
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

    //! Rewrite file (applying processing, masks, etc)
    /* GeoImage Process(const GeoImage& image) {
        for (unsigned int i=0l i<Output.NumBands(); i++) {

        }
    }
    // Apply a mask to existing file (where mask>0 change to NoDataValue)
    GeoImage Process(const GeoImage& image, GeoRaster& mask) {
        image.AddMask(mask);
        for (unsigned int i=0; i<image.NumBands(); i++) {
            switch (image.DataType()) {
                case GDT_Byte: GeoRasterIO<unsigned char>(image[i]).ApplyMask(mask);
                    break;
                case GDT_UInt16: GeoRasterIO<unsigned short>(image[i]).ApplyMask(mask);
                    break;
                case GDT_Int16: GeoRasterIO<short>(image[i]).ApplyMask(mask);
                    break;
                case GDT_UInt32: GeoRasterIO<unsigned int>(image[i]).ApplyMask(mask);
                    break;
                case GDT_Int32: GeoRasterIO<int>(image[i]).ApplyMask(mask);
                    break;
                case GDT_Float32: GeoRasterIO<float>(image[i]).ApplyMask(mask);
                    break;
                case GDT_Float64: GeoRasterIO<double>(image[i]).ApplyMask(mask);
                    break;
                default: GeoRasterIO<unsigned char>(image[i]).ApplyMask(mask);
            }
        }
        return image;
    }   */

} // namespace gip
