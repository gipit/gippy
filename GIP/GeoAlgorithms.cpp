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

#define _USE_MATH_DEFINES
#include <cmath>
#include <set>

#include <gip/GeoAlgorithms.h>
#include <gip/gip_gdal.h>


namespace gip {
    namespace algorithms {
    using std::string;
    using std::vector;
    using std::cout;
    using std::cerr;
    using std::endl;

    /** ACCA (Automatic Cloud Cover Assessment). Takes in TOA Reflectance,
     * temperature, sun elevation, solar azimuth, and number of pixels to
     * dilate.
     */
    GeoImage acca(const GeoImage& image, std::string filename, float se_degrees,
                  float sa_degrees, int erode, int dilate, int cloudheight) {
        if (Options::verbose() > 1) cout << "GIPPY: ACCA - " << image.basename() << endl;

        float th_red(0.08);
        float th_ndsi(0.7);
        float th_temp(27);
        float th_comp(225);
        float th_nirred(2.0);
        float th_nirgreen(2.0);
        float th_nirswir1(1.0);
        //float th_warm(210);

        GeoImage imgout = GeoImage::create_from(filename, image, 4, "uint8");
        imgout.set_nodata(0);
        imgout.set_bandnames({"finalmask", "cloudmask", "ambclouds", "pass1"});
        vector<string> bands_used({"RED","GREEN","NIR","SWIR1","LWIR"});

        CImg<float> red, green, nir, swir1, temp, ndsi, b56comp;
        CImg<unsigned char> nonclouds, ambclouds, clouds, mask, temp2;
        float cloudsum(0), scenesize(0);

        ChunkSet chunks(image.xsize(),image.ysize());
        Rect<int> chunk;

        //if (Options::verbose()) cout << image.basename() << " - ACCA (dev-version)" << endl;
        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            chunk = chunks[iChunk];
            red = image["RED"].read<float>(chunk);
            green = image["GREEN"].read<float>(chunk);
            nir = image["NIR"].read<float>(chunk);
            swir1 = image["SWIR1"].read<float>(chunk);
            temp = image["LWIR"].read<float>(chunk);

            mask = image.nodata_mask(bands_used, chunk)^=1;

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

            imgout["pass1"].write<unsigned char>(clouds,chunk);
            imgout["ambclouds"].write<unsigned char>(ambclouds,chunk);
            //imgout[0].write(nonclouds,iChunk);
            if (Options::verbose() > 3) cout << "Processed chunk " << chunk << " of " << chunks.size() << endl;
        }
        // Cloud statistics
        float cloudcover = cloudsum / scenesize;
        CImg<float> tstats = image["LWIR"].add_mask(imgout["pass1"]).stats();
        if (Options::verbose() > 1) {
            cout.precision(4);
            cout << "   Cloud Cover = " << cloudcover*100 << "%" << endl;
            //cimg_print(tstats, "Cloud stats(min,max,mean,sd,skew,count)");
        }

        // Pass 2 (thermal processing)
        bool addclouds(false);
        if ((cloudcover > 0.004) && (tstats(2) < 22.0)) {
            float th0 = image["LWIR"].percentile(83.5);
            float th1 = image["LWIR"].percentile(97.5);
            if (tstats[4] > 0) {
                float th2 = image["LWIR"].percentile(98.75);
                float shift(0);
                shift = tstats[3] * ((tstats[4] > 1.0) ? 1.0 : tstats[4]);
                //cout << "Percentiles = " << th0 << ", " << th1 << ", " << th2 << ", " << shift << endl;
                if (th2-th1 < shift) shift = th2-th1;
                th0 += shift;
                th1 += shift;
            }
            image["LWIR"].clear_masks();
            CImg<float> warm_stats = image["LWIR"].add_mask(imgout["ambclouds"]).add_mask(image["LWIR"] < th1).add_mask(image["LWIR"] > th0).stats();
            if (Options::verbose() > 1) 
                warm_stats.print("Warm Cloud stats(min,max,mean,sd,skew,count)");
            image["LWIR"].clear_masks();
            if (((warm_stats(5)/scenesize) < 0.4) && (warm_stats(2) < 22)) {
                if (Options::verbose() > 2) cout << "Accepting warm clouds" << endl;
                imgout["ambclouds"].add_mask(image["LWIR"] < th1).add_mask(image["LWIR"] > th0);
                addclouds = true;
            } else {
                // Cold clouds
                CImg<float> cold_stats = image["LWIR"].add_mask(imgout["ambclouds"]).add_mask(image["LWIR"] < th0).stats();
                if (Options::verbose() > 1) 
                    cold_stats.print("Cold Cloud stats(min,max,mean,sd,skew,count)");
                image["LWIR"].clear_masks();
                if (((cold_stats(5)/scenesize) < 0.4) && (cold_stats(2) < 22)) {
                    if (Options::verbose() > 2) cout << "Accepting cold clouds" << endl;
                    imgout["ambclouds"].add_mask(image["LWIR"] < th0);
                    addclouds = true;
                } else
                    if (Options::verbose() > 2) cout << "Rejecting all ambiguous clouds" << endl;
            }
        } else image["LWIR"].clear_masks();

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
        if (Options::verbose() > 2)
            cerr << "distance = " << distance << endl
                 << "dx       = " << dx << endl
                 << "dy       = " << dy << endl
                 << "smearlen = " << smearlen << endl ;

        // shift-style smear
        int signX(dx/abs(dx));
        int signY(dy/abs(dy));
        int xstep = std::max(signX*dx/dilate/4, 1);
        int ystep = std::max(signY*dy/dilate/4, 1);
        if (Options::verbose() > 2)
            cerr << "dilate = " << dilate << endl
                 << "xstep  = " << signX*xstep << endl
                 << "ystep  = " << signY*ystep << endl ;

        chunks.padding(padding);

        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            chunk = chunks[iChunk];
            if (Options::verbose() > 3) cout << "Chunk " << chunk << " of " << chunks.size() << endl;
            clouds = imgout["pass1"].read<unsigned char>(chunk);
            // should this be a |= ?
            if (addclouds) clouds += imgout["ambclouds"].read<unsigned char>(chunk);
            clouds|=(image.saturation_mask(bands_used, chunk));
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
            imgout["cloudmask"].write<unsigned char>(clouds,chunk);
            // Inverse and multiply by nodata mask to get good data mask
            imgout["finalmask"].write<unsigned char>((clouds^=1).mul(image.nodata_mask(bands_used, chunk)^=1), chunk);
            // TODO - add in snow mask
        }
        return imgout;
    }


    //! Merge images into one file and crop to vector
    GeoImage cookie_cutter(GeoImages images, GeoFeature feature, std::string filename, 
        float xres, float yres, bool crop, unsigned char interpolation) {
        if (Options::verbose() > 1)
            cout << "GIPPY: cookie_cutter (" << images.nimages() << " files) - " << filename << endl;
        Rect<double> extent = feature.extent();

        if (crop) {
            Rect<double> _extent = images.extent(feature.srs());
            // limit to feature extent
            _extent.intersect(extent);
            // anchor to top left of feature (MinX, MaxY) and make multiple of resolution
            extent = Rect<double>(
                Point<double>(extent.x0() + std::floor((_extent.x0()-extent.x0()) / xres) * xres, _extent.y0()),
                Point<double>(_extent.x1(), extent.y1() - std::floor((extent.y1()-_extent.y1()) / yres) * yres)
            );
        }

        // create output
        // convert extent to resolution units
        int xsize = std::ceil(extent.width() / xres);
        int ysize = std::ceil(extent.height() / yres);
        GeoImage imgout(filename, xsize, ysize, images.nbands(), images.type());
        imgout.set_meta(images[0].meta());
        for (unsigned int b=0;b<imgout.nbands();b++) {
            imgout[b].set_gain(images[0][b].gain());
            imgout[b].set_offset(images[0][b].offset());
            imgout[b].set_nodata(images[0][b].nodata());
        }

        // add additional metadata to output
        dictionary metadata;
        metadata["SourceFiles"] = to_string(images.basenames());
        if (interpolation > 1) metadata["Interpolation"] = to_string(interpolation);
        imgout.set_meta(metadata);

        // set projection and affine transformation
        imgout.set_srs(feature.srs());
        // TODO - set affine based on extent and resolution (?)
        CImg<double> affine(6);
        affine[0] = extent.x0();
        affine[1] = xres;
        affine[2] = 0;
        affine[3] = extent.y1();
        affine[4] = 0;
        affine[5] = -std::abs(yres);
        imgout.set_affine(affine);

        // warp options
        GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
        psWarpOptions->hDstDS = imgout.GetGDALDataset();
        psWarpOptions->nBandCount = imgout.nbands();
        psWarpOptions->panSrcBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->panDstBands = (int *) CPLMalloc(sizeof(int) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfSrcNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataReal = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        psWarpOptions->padfDstNoDataImag = (double *) CPLMalloc(sizeof(double) * psWarpOptions->nBandCount );
        for (unsigned int b=0;b<imgout.nbands();b++) {
            psWarpOptions->panSrcBands[b] = b+1;
            psWarpOptions->panDstBands[b] = b+1;
            psWarpOptions->padfSrcNoDataReal[b] = images[0][b].nodata();
            psWarpOptions->padfDstNoDataReal[b] = imgout[b].nodata();
            psWarpOptions->padfSrcNoDataImag[b] = 0.0;
            psWarpOptions->padfDstNoDataImag[b] = 0.0;
        }
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
        papszOptions = CSLSetNameValue(papszOptions,"INIT_DEST","NO_DATA");
        papszOptions = CSLSetNameValue(papszOptions,"WRITE_FLUSH","YES");
        papszOptions = CSLSetNameValue(papszOptions,"NUM_THREADS",to_string(Options::cores()).c_str());
        psWarpOptions->papszWarpOptions = papszOptions;

        OGRGeometry* geom = feature.geometry();

        for (unsigned int i=0; i<images.nimages(); i++) {
            WarpToImage(images[i], imgout, psWarpOptions, geom);
            psWarpOptions->papszWarpOptions = CSLSetNameValue(psWarpOptions->papszWarpOptions,"INIT_DEST",NULL);
        }
        GDALDestroyWarpOptions( psWarpOptions );

        return imgout;
    }


    //! Fmask cloud mask
    GeoImage fmask(const GeoImage& image, string filename, int tolerance, int dilate) {
        if (Options::verbose() > 1)
            cout << "GIPPY: Fmask (tol=" << tolerance << ", d=" << dilate << ") - " << filename << endl;

        GeoImage imgout = GeoImage::create_from(filename, image, 5, "uint8");
        imgout.set_bandnames({"finalmask", "cloudmask", "PCP", "clearskywater", "clearskyland"});
        imgout.set_nodata(0);
        float nodataval(-32768);
        // Output probabilties (for debugging/analysis)
        GeoImage probout = GeoImage::create_from(filename + "-prob", image, 2, "float32");
        probout.set_bandnames({"wcloud", "lcloud"});
        probout.set_nodata(nodataval);

        CImg<unsigned char> clouds, pcp, wmask, lmask, mask, redsatmask, greensatmask;
        CImg<float> red, nir, green, blue, swir1, swir2, BT, ndvi, ndsi, white, vprob;
        float _ndvi, _ndsi;
        long datapixels(0);
        long cloudpixels(0);
        long landpixels(0);
        //CImg<double> wstats(image.Size()), lstats(image.Size());
        //int wloc(0), lloc(0);

        ChunkSet chunks(image.xsize(),image.ysize());

        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            blue = image["blue"].read<double>(chunks[iChunk]);
            red = image["red"].read<double>(chunks[iChunk]);
            green = image["green"].read<double>(chunks[iChunk]);
            nir = image["nir"].read<double>(chunks[iChunk]);
            swir1 = image["swir1"].read<double>(chunks[iChunk]);
            swir2 = image["swir2"].read<double>(chunks[iChunk]);
            BT = image["lwir"].read<double>(chunks[iChunk]);
            mask = image.nodata_mask(chunks[iChunk])^=1;
            ndvi = (nir-red).div(nir+red);
            ndsi = (green-swir1).div(green+swir1);
            white = image.whiteness(chunks[iChunk]);

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

            redsatmask = image["red"].saturation_mask(chunks[iChunk]);
            greensatmask = image["green"].saturation_mask(chunks[iChunk]);
            vprob = red;
            // Calculate "variability probability"
            cimg_forXY(vprob,x,y) {
                _ndvi = (redsatmask(x,y) && nir(x,y) > red(x,y)) ? 0 : abs(ndvi(x,y));
                _ndsi = (greensatmask(x,y) && swir1(x,y) > green(x,y)) ? 0 : abs(ndsi(x,y));
                vprob(x,y) = 1 - std::max(white(x,y), std::max(_ndsi, _ndvi));
            }
            probout["lcloud"].write(vprob, chunks[iChunk]);

            datapixels += mask.sum();
            cloudpixels += pcp.sum();
            wmask = ((ndvi.get_threshold(0.01,false,true)^=1) &= (nir.get_threshold(0.01,false,true)^=1))|=
                    ((ndvi.get_threshold(0.1,false,true)^=1) &= (nir.get_threshold(0.05,false,true)^=1));

            imgout["pcp"].write(pcp.mul(mask), chunks[iChunk]);        // Potential cloud pixels
            imgout["water"].write(wmask.get_mul(mask), chunks[iChunk]);   // Clear-sky water
            CImg<unsigned char> landimg((wmask^1).mul(pcp^1).mul(mask));
            landpixels += landimg.sum();
            imgout["land"].write(landimg, chunks[iChunk]);    // Clear-sky land
        }
        // floodfill....seems bad way
        //shadowmask = nir.draw_fill(nir.width()/2,nir.height()/2,)

        // If not enough non-cloud pixels then return existing mask
        if (cloudpixels >= (0.999*imgout[0].size())) return imgout;
        // If not enough clear-sky land pixels then use all
        //GeoRaster msk;
        //if (landpixels < (0.001*imgout[0].Size())) msk = imgout[1];

        // Clear-sky water
        double Twater(image["lwir"].add_mask(image["swir2"] < 0.03).add_mask(imgout["water"]).add_mask(imgout["pcp"]).percentile(82.5));
        image["lwir"].clear_masks();
        GeoRaster landBT(image["lwir"].add_mask(imgout["land"]));
        image["lwir"].clear_masks();
        double Tlo(landBT.percentile(17.5));
        double Thi(landBT.percentile(82.5));

        if (Options::verbose() > 2) {
            cout << "PCP = " << 100*cloudpixels/(double)datapixels << "%" << endl;
            cout << "Water (82.5%) = " << Twater << endl;
            cout << "Land (17.5%) = " << Tlo << ", (82.5%) = " << Thi << endl;
        }

        // Calculate cloud probabilities for over water and land
        CImg<float> wprob, lprob;
        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            mask = image.nodata_mask(chunks[iChunk])^=1;
            BT = image["lwir"].read<double>(chunks[iChunk]);
            swir1 = image["swir1"].read<double>(chunks[iChunk]);

            // Water Clouds = temp probability x brightness probability
            wprob = ((Twater - BT)/=4.0).mul( swir1.min(0.11)/=0.11 ).mul(mask);
            probout["wcloud"].write(wprob, chunks[iChunk]);

            // Land Clouds = temp probability x variability probability
            vprob = probout["wcloud"].read<double>(chunks[iChunk]);
            lprob = ((Thi + 4-BT)/=(Thi+4-(Tlo-4))).mul( vprob ).mul(mask);
            //1 - image.NDVI(*chunks[iChunk]).abs().max(image.NDSI(*chunks[iChunk]).abs()).max(image.Whiteness(*chunks[iChunk]).abs()) );
            probout["lcloud"].write( lprob, chunks[iChunk]);
        }

        // Thresholds
        float tol((tolerance-3)*0.1);
        float wthresh = 0.5 + tol;
        float lthresh(probout["lcloud"].add_mask(imgout["land"]).percentile(82.5)+0.2+tol);
        probout["lcloud"].clear_masks();
        if (Options::verbose() > 2)
            cout << "Thresholds: water = " << wthresh << ", land = " << lthresh << endl;

        // 3x3 filter of 1's for majority filter
        //CImg<int> filter(3,3,1,1, 1);
        int erode = 5;
        int padding(double(std::max(dilate,erode)+1)/2);
        chunks.padding(padding);
        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            mask = image.nodata_mask(chunks[iChunk])^=1;
            pcp = imgout["pcp"].read<double>(chunks[iChunk]);
            wmask = imgout["water"].read<double>(chunks[iChunk]);
            BT = image["lwir"].read<double>(chunks[iChunk]);

            lprob = probout["lcloud"].read<double>(chunks[iChunk]);
            
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
            imgout["clouds"].write(clouds, chunks[iChunk]);
            imgout["final"].write((clouds^=1).mul(mask), chunks[iChunk]);
        }

        return imgout;
    }

    GeoImage indices(const GeoImage& image, dictionary products) {
        if (Options::verbose() > 1) std::cout << "GIPPY: Indices" << std::endl;

        float nodataout = -32768;

        std::map< string, GeoImage > imagesout;
        std::map<string, string>::const_iterator iprod;
        std::vector<string> filenames;
        string prodname;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            //imagesout[*iprod] = GeoImageIO<float>(GeoImage(basename + '_' + *iprod, image, GDT_Int16));
            if (Options::verbose() > 2) cout << iprod->first << " -> " << iprod->second << endl;
            prodname = iprod->first;
            imagesout[prodname] = GeoImage::create_from(iprod->second, image, 1, "int16");
            imagesout[prodname].set_nodata(nodataout);
            imagesout[prodname].set_gain(0.0001);
            imagesout[prodname].set_bandname(prodname, 1);
            filenames.push_back(imagesout[prodname].filename());
        }
        if (imagesout.size() == 0) throw std::runtime_error("No indices selected for calculation!");

        std::map< string, std::vector<string> > colors;
        colors["ndvi"] = {"nir","red"};
        colors["evi"] = {"nir","red","blue"};
        colors["lswi"] = {"nir","swir1"};
        colors["ndsi"] = {"swir1","green"};
        colors["ndwi"] = {"green","nir"};
        colors["bi"] = {"blue","nir"};
        colors["satvi"] = {"swir1","red", "swir2"};
        colors["msavi2"] = {"nir","red"};
        colors["vari"] = {"red","green","blue"};
        colors["brgt"] = {"red","green","blue","nir"};
        // Tillage indices
        colors["ndti"] = {"swir2","swir1"};
        colors["crc"] = {"swir1","swir2","blue"};
        colors["crcm"] = {"swir1","swir2","green"};
        colors["isti"] = {"swir1","swir2"};
        colors["sti"] = {"swir1","swir2"};

        // Figure out what colors are needed
        std::set< string > used_colors;
        std::set< string >::const_iterator isstr;
        std::vector< string >::const_iterator ivstr;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            for (ivstr=colors[iprod->first].begin();ivstr!=colors[iprod->first].end();ivstr++) {
                used_colors.insert(*ivstr);
            }
        }
        if (Options::verbose() > 2) {
            cout << "Colors used: ";
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) cout << " " << *isstr;
            cout << endl;
        }

        CImg<float> red, green, blue, nir, swir1, swir2, cimgout, cimgmask, tmpimg;

        ChunkSet chunks(image.xsize(),image.ysize());

        // need to add overlap
        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            if (Options::verbose() > 3) cout << "Chunk " << chunks[iChunk] << " of " << image[0].size() << endl;
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) {
                if (*isstr == "red") red = image["red"].read<float>(chunks[iChunk]);
                else if (*isstr == "green") green = image["green"].read<float>(chunks[iChunk]);
                else if (*isstr == "blue") blue = image["blue"].read<float>(chunks[iChunk]);
                else if (*isstr == "nir") nir = image["nir"].read<float>(chunks[iChunk]);
                else if (*isstr == "swir1") swir1 = image["swir1"].read<float>(chunks[iChunk]);
                else if (*isstr == "swir2") swir2 = image["swir2"].read<float>(chunks[iChunk]);
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
                cimgmask = image.nodata_mask(colors[prodname], chunks[iChunk]);
                cimg_forXY(cimgout,x,y) if (cimgmask(x,y)) cimgout(x,y) = nodataout;
                imagesout[prodname].write(cimgout,chunks[iChunk]);
            }
        }
        return GeoImage(filenames);
    }

    //! Perform linear transform with given coefficients (e.g., PC transform)
    GeoImage linear_transform(const GeoImage& img, CImg<float> coef, string filename) {
        // Verify size of array
        unsigned int numbands = img.nbands();
        if ((coef.height() != (int)numbands) || (coef.width() != (int)numbands))
            throw std::runtime_error("Coefficient array needs to be of size NumBands x NumBands!");
        float nodataout = -32768;
        GeoImage imgout = GeoImage::create_from(filename, img, img.nbands(), "float32");
        imgout.set_nodata(nodataout);
        imgout.set_meta(img.meta());
        CImg<float> cimg;
        CImg<unsigned char> mask;

        ChunkSet chunks(img.xsize(),img.ysize());

        for (unsigned int bout=0; bout<numbands; bout++) {
            //if (Options::verbose() > 4) cout << "Band " << bout << endl;
            for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
                cimg = img[0].read<float>(chunks[iChunk]) * coef(0, bout);;
                for (unsigned int bin=1; bin<numbands; bin++) {
                    cimg = cimg + (img[bin].read<float>(chunks[iChunk]) * coef(bin, bout));
                }
                mask = img.nodata_mask(chunks[iChunk]);
                cimg_forXY(cimg,x,y) if (mask(x,y)) cimg(x,y) = nodataout;
                imgout[bout].write(cimg, chunks[iChunk]);
            }
        }
        return imgout;
    }

    //! Runs the RX Detector (RXD) anamoly detection algorithm
    GeoImage rxd(const GeoImage& img, string filename) {
        if (img.nbands() < 2) throw std::runtime_error("RXD: At least two bands must be supplied");

        GeoImage imgout = GeoImage::create_from(filename, img, 1, "uint8");
        imgout.set_bandname("RXD", 1);

        CImg<double> covariance = img.spectral_covariance();
        CImg<double> K = covariance.invert();
        CImg<double> chip, chipout, pixel;

        // Calculate band means
        CImg<double> bandmeans(img.nbands());
        cimg_forX(bandmeans, x) {
            bandmeans(x) = img[x].stats()[2];
        }

        ChunkSet chunks(img.xsize(),img.ysize());
        for (unsigned int iChunk=0; iChunk<chunks.size(); iChunk++) {
            chip = img.read<double>(chunks[iChunk]);
            chipout = CImg<double>(chip, "xyzc");
            cimg_forXY(chip,x,y) {
                pixel = chip.get_crop(x,y,0,0,x,y,0,chip.spectrum()-1).unroll('x') - bandmeans;
                chipout(x,y) = (pixel * K.get_transpose() * pixel.get_transpose())[0];
            }
            imgout[0].write(chipout, chunks[iChunk]);
        }
        return imgout;
    }




    }
} // namespace gip
