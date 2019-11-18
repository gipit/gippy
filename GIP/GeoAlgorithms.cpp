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

        GeoImage imgout = GeoImage::create_from(image, filename, 4, "uint8");
        imgout.set_nodata(0);
        imgout.set_bandnames({"finalmask", "cloudmask", "ambclouds", "pass1"});
        vector<string> bands_used({"RED","GREEN","NIR","SWIR1","LWIR"});

        CImg<float> red, green, nir, swir1, temp, ndsi, b56comp;
        CImg<unsigned char> nonclouds, ambclouds, clouds, mask, temp2;
        float cloudsum(0), scenesize(0);

        //if (Options::verbose()) cout << image.basename() << " - ACCA (dev-version)" << endl;
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = image.chunks();
        int i(0);
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            red = image["RED"].read<float>(*iCh);
            green = image["GREEN"].read<float>(*iCh);
            nir = image["NIR"].read<float>(*iCh);
            swir1 = image["SWIR1"].read<float>(*iCh);
            temp = image["LWIR"].read<float>(*iCh);

            mask = image.nodata_mask(bands_used, *iCh)^=1;

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

            imgout["pass1"].write<unsigned char>(clouds,*iCh);
            imgout["ambclouds"].write<unsigned char>(ambclouds,*iCh);
            //imgout[0].write(nonclouds,iCh);
            if (Options::verbose() > 3) cout << "Processed chunk " << i++ << " of " << chunks.size() << endl;
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

        chunks = image.chunks(padding);
        i = 0;
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            if (Options::verbose() > 3) cout << "Chunk " << i++ << " of " << chunks.size() << endl;
            clouds = imgout["pass1"].read<unsigned char>(*iCh);
            // should this be a |= ?
            if (addclouds) clouds += imgout["ambclouds"].read<unsigned char>(*iCh);
            clouds|=(image.saturation_mask(bands_used, 255, *iCh));
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
            imgout["cloudmask"].write<unsigned char>(clouds,*iCh);
            // Inverse and multiply by nodata mask to get good data mask
            imgout["finalmask"].write<unsigned char>((clouds^=1).mul(image.nodata_mask(bands_used, *iCh)^=1), *iCh);
            // TODO - add in snow mask
        }
        return imgout;
    }


    //! Merge images into one file and crop to vector
    /*!
        Assumptions
        - all input bands have same number of bands, and output band will have the same
        - if GeoFeature is provided it will it's SRS. If not, proj parameter will be used (EPSG:4326 default)
    */
    GeoImage cookie_cutter(const std::vector<GeoImage>& geoimgs, string filename,
            GeoFeature feature, bool crop, string proj, float xres, float yres, int interpolation, dictionary options, bool alltouch) {
        if (Options::verbose() > 1)
            cout << "GIPPY: cookie_cutter (" << geoimgs.size() << " files) - " << filename << endl;

        // calculate union of all image extents
        vector<BoundingBox> extents;
        for (vector<GeoImage>::const_iterator i=geoimgs.begin(); i!=geoimgs.end(); i++) {
            extents.push_back(i->extent());
        }
        BoundingBox ext = union_all(extents);

        // if valid feature provided use that extent
        if (feature.valid()) {
            if (proj == "")
                proj = feature.srs();
            // transform extent to desired srs
            ext.transform(geoimgs[0].srs(), proj);
            if (crop) {
                BoundingBox fext = feature.extent();
                // limit to feature extent
                ext = ext.intersect(feature.extent());
                // anchor to top left of feature (MinX, MaxY) and make multiple of resolution
                ext = BoundingBox(
                    Point<double>(fext.x0() + std::floor((ext.x0()-fext.x0()) / xres) * xres, ext.y0()),
                    Point<double>(ext.x1(), fext.y1() - std::floor((fext.y1()-ext.y1()) / yres) * yres)
                );
            } else
                // make the extent just the feature
                ext = feature.extent().transform(feature.srs(), proj);
        }

        // create output
        // convert extent to resolution units
	// one pixel is added for transition from vector to raster space
        int xsz = std::ceil(ext.width() / std::abs(xres)) + 1;
        int ysz = std::ceil(ext.height() / std::abs(yres)) + 1;

        double xshift = -0.5 * std::abs(xres);
        double yshift = -0.5 * std::abs(yres);

        /* Multiply the x and y size by the desired resolution to force the output
           image to have a size evenly divisible by the res. xsz and ysz above have
           been increased by one pixel to avoid the infamous "lost pixel"
           when cutting a raster with a vector. Finally, to minimize pixel drift
           amongst all of this, the whole image is shifted NW a half pixel. */
        CImg<double> bbox(4,1,1,1, ext.x0() + xshift, ext.y0() + yshift, xsz * std::abs(xres), ysz * std::abs(yres));
        GeoImage imgout = GeoImage::create(filename, xsz, ysz, geoimgs[0].nbands(),
                            proj, bbox, geoimgs[0].type().string(), "", false, options);

        imgout.add_meta(geoimgs[0].meta());
        for (unsigned int b=0;b<imgout.nbands();b++) {
            imgout[b].set_gain(geoimgs[0][b].gain());
            imgout[b].set_offset(geoimgs[0][b].offset());
            imgout[b].set_nodata(geoimgs[0][b].nodata());
        }

        // add additional metadata to output
        dictionary metadata;
        //metadata["SourceFiles"] = to_string(geoimgs.basenames());
        if (interpolation > 1) metadata["Interpolation"] = to_string(interpolation);
        imgout.add_meta(metadata);
      
        bool noinit(false);
        for (unsigned int i=0; i<geoimgs.size(); i++) {
            geoimgs[i].warp_into(imgout, feature, interpolation, noinit, alltouch);
            noinit = true;
        }

        return imgout;
    }


    //! Fmask cloud mask
    GeoImage fmask(const GeoImage& image, string filename, int tolerance, int dilate) {
        if (Options::verbose() > 1)
            cout << "GIPPY: Fmask (tol=" << tolerance << ", d=" << dilate << ") - " << filename << endl;

        GeoImage imgout = GeoImage::create_from(image, filename, 5, "uint8");
        imgout.set_bandnames({"finalmask", "cloudmask", "PCP", "clearskywater", "clearskyland"});
        imgout.set_nodata(0);
        float nodataval(-32768);
        // Output probabilties (for debugging/analysis)
        GeoImage probout = GeoImage::create_from(image, filename + "-prob", 2, "float32");
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

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = image.chunks();

        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            blue = image["blue"].read<double>(*iCh);
            red = image["red"].read<double>(*iCh);
            green = image["green"].read<double>(*iCh);
            nir = image["nir"].read<double>(*iCh);
            swir1 = image["swir1"].read<double>(*iCh);
            swir2 = image["swir2"].read<double>(*iCh);
            BT = image["lwir"].read<double>(*iCh);
            mask = image.nodata_mask(*iCh)^=1;
            ndvi = (nir-red).div(nir+red);
            ndsi = (green-swir1).div(green+swir1);
            white = image.whiteness(*iCh);

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

            redsatmask = image["red"].saturation_mask(255, *iCh);
            greensatmask = image["green"].saturation_mask(255, *iCh);
            vprob = red;
            // Calculate "variability probability"
            cimg_forXY(vprob,x,y) {
                _ndvi = (redsatmask(x,y) && nir(x,y) > red(x,y)) ? 0 : abs(ndvi(x,y));
                _ndsi = (greensatmask(x,y) && swir1(x,y) > green(x,y)) ? 0 : abs(ndsi(x,y));
                vprob(x,y) = 1 - std::max(white(x,y), std::max(_ndsi, _ndvi));
            }
            probout["lcloud"].write(vprob, *iCh);

            datapixels += mask.sum();
            cloudpixels += pcp.sum();
            wmask = ((ndvi.get_threshold(0.01,false,true)^=1) &= (nir.get_threshold(0.01,false,true)^=1))|=
                    ((ndvi.get_threshold(0.1,false,true)^=1) &= (nir.get_threshold(0.05,false,true)^=1));

            imgout["pcp"].write(pcp.mul(mask), *iCh);        // Potential cloud pixels
            imgout["water"].write(wmask.get_mul(mask), *iCh);   // Clear-sky water
            CImg<unsigned char> landimg((wmask^1).mul(pcp^1).mul(mask));
            landpixels += landimg.sum();
            imgout["land"].write(landimg, *iCh);    // Clear-sky land
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
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            mask = image.nodata_mask(*iCh)^=1;
            BT = image["lwir"].read<double>(*iCh);
            swir1 = image["swir1"].read<double>(*iCh);

            // Water Clouds = temp probability x brightness probability
            wprob = ((Twater - BT)/=4.0).mul( swir1.min(0.11)/=0.11 ).mul(mask);
            probout["wcloud"].write(wprob, *iCh);

            // Land Clouds = temp probability x variability probability
            vprob = probout["wcloud"].read<double>(*iCh);
            lprob = ((Thi + 4-BT)/=(Thi+4-(Tlo-4))).mul( vprob ).mul(mask);
            //1 - image.NDVI(**iCh).abs().max(image.NDSI(**iCh).abs()).max(image.Whiteness(**iCh).abs()) );
            probout["lcloud"].write( lprob, *iCh);
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
        chunks = image.chunks(padding);
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            mask = image.nodata_mask(*iCh)^=1;
            pcp = imgout["pcp"].read<double>(*iCh);
            wmask = imgout["water"].read<double>(*iCh);
            BT = image["lwir"].read<double>(*iCh);

            lprob = probout["lcloud"].read<double>(*iCh);
            
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
            imgout["clouds"].write(clouds, *iCh);
            imgout["final"].write((clouds^=1).mul(mask), *iCh);
        }

        return imgout;
    }

    GeoImage indices(const GeoImage& image, vector<string> products, string filename) {
        if (Options::verbose() > 1) std::cout << "GIPPY: Indices" << std::endl;

        float nodataout = -32768;

        GeoImage imgout = GeoImage::create_from(image, filename, products.size(), "int16");
        imgout.set_bandnames(products);
        imgout.set_nodata(nodataout);
        imgout.set_gain(0.0001);

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
        std::vector< string >::const_iterator iprod, ivstr;
        for (iprod=products.begin(); iprod!=products.end(); iprod++) {
            for (ivstr=colors[*iprod].begin();ivstr!=colors[*iprod].end();ivstr++) {
                used_colors.insert(*ivstr);
            }
        }
        if (Options::verbose() > 2) {
            cout << "Colors used: ";
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) cout << " " << *isstr;
            cout << endl;
        }

        CImg<float> red, green, blue, nir, swir1, swir2, cimgout, cimgmask, tmpimg;

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = image.chunks();
        std::string prodname;

        // need to add overlap
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            if (Options::verbose() > 3) cout << "Chunk " << *iCh << " of " << image[0].size() << endl;
            for (isstr=used_colors.begin();isstr!=used_colors.end();isstr++) {
                if (*isstr == "red") red = image["red"].read<float>(*iCh);
                else if (*isstr == "green") green = image["green"].read<float>(*iCh);
                else if (*isstr == "blue") blue = image["blue"].read<float>(*iCh);
                else if (*isstr == "nir") nir = image["nir"].read<float>(*iCh);
                else if (*isstr == "swir1") swir1 = image["swir1"].read<float>(*iCh);
                else if (*isstr == "swir2") swir2 = image["swir2"].read<float>(*iCh);
            }

            for (iprod=products.begin(); iprod!=products.end(); iprod++) {
                prodname = *iprod;
                prodname = to_lower(prodname);
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
                cimgmask = image.nodata_mask(colors[prodname], *iCh);
                cimg_forXY(cimgout,x,y) if (cimgmask(x,y)) cimgout(x,y) = nodataout;
                imgout[prodname].write(cimgout, *iCh);
            }
        }
        return imgout;
    }


    //! k-means unsupervised classifier
    GeoImage kmeans( const GeoImage& image, string filename,
                     unsigned int classes, unsigned int iterations, 
                     float threshold, unsigned int num_random) {
        //if (Image.NumBands() < 2) throw GIP::Gexceptions::errInvalidParams("At least two bands must be supplied");
        if (Options::verbose()) {
            cout << image.basename() << " - k-means unsupervised classifier:" << endl
                << "  Classes = " << classes << endl
                << "  Iterations = " << iterations << endl
                << "  Pixel Change Threshold = " << threshold << "%" << endl;
        }
        // Calculate threshold in # of pixels
        threshold = threshold/100.0 * image.size();

        GeoImage img(image);
        // Create new output image
        GeoImage imgout = GeoImage::create_from(image, filename, 1, "uint8");

        // Get initial class estimates (uses random pixels)
        CImg<float> ClassMeans = get_random_classes<float>(img, classes, num_random);
        if (Options::verbose() > 1) cimg_print(ClassMeans);

        CImg<double> Pixel, C_img, DistanceToClass(classes), NumSamples(classes), ThisClass;
        CImg<unsigned char> C_imgout, C_mask;
        CImg<double> RunningTotal(classes,image.nbands(),1,1,0);

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = image.chunks();

        unsigned int NumPixelChange, iteration=0;
        do {
            NumPixelChange = 0;
            for (unsigned int i=0; i<classes; i++) NumSamples(i) = 0;
            if (Options::verbose()) cout << "  Iteration " << iteration+1 << std::flush;

            // reset running total to zero
            cimg_forXY(RunningTotal,x,y) RunningTotal(x,y) = 0.0;

            for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
                C_img = img.read<float>(*iCh);
                C_mask = img.nodata_mask(*iCh);
                C_imgout = imgout[0].read<float>(*iCh);

                CImg<double> stats;
                cimg_forXY(C_img,x,y) { // Loop through image
                    // Calculate distance between this pixel and all classes
                    if (!C_mask(x,y)) {
                        Pixel = C_img.get_crop(x,y,0,0,x,y,C_img.depth()-1,0).unroll('x');
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
                imgout[0].write<unsigned int>(C_imgout,*iCh);
                if (Options::verbose()) cout << "." << std::flush;
            }

            // Calculate new Mean class vectors
            for (unsigned int c=0; c<classes; c++) {
                if (NumSamples(c) > 0) {
                    cimg_forX(ClassMeans,x) {
                        ClassMeans(x,c) = RunningTotal(c,x)/NumSamples(c);
                        RunningTotal(c,x) = 0;
                    }
                    NumSamples(c) = 0;
                }
            }
            if (Options::verbose()) cout << 100.0*((double)NumPixelChange/image.size()) << "% pixels changed class" << endl;
            if (Options::verbose() > 2) cimg_print(ClassMeans);
        } while ( (++iteration < iterations) && (NumPixelChange > threshold) );

        imgout.set_bandname("k-means", 1);
        //imgout.GetGDALDataset()->FlushCache();
        return imgout;
    }


    //! Perform linear transform with given coefficients (e.g., PC transform)
    GeoImage linear_transform(const GeoImage& img, CImg<float> coef, string filename) {
        // Verify size of array
        unsigned int numbands = img.nbands();
        if ((coef.height() != (int)numbands) || (coef.width() != (int)numbands))
            throw std::runtime_error("Coefficient array needs to be of size NumBands x NumBands!");
        float nodataout = -32768;
        GeoImage imgout = GeoImage::create_from(img, filename, img.nbands(), "float32");
        imgout.set_nodata(nodataout);
        imgout.add_meta(img.meta());
        CImg<float> cimg;
        CImg<unsigned char> mask;

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = img.chunks();

        for (unsigned int bout=0; bout<numbands; bout++) {
            //if (Options::verbose() > 4) cout << "Band " << bout << endl;
            for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
                cimg = img[0].read<float>(*iCh) * coef(0, bout);;
                for (unsigned int bin=1; bin<numbands; bin++) {
                    cimg = cimg + (img[bin].read<float>(*iCh) * coef(bin, bout));
                }
                mask = img.nodata_mask(*iCh);
                cimg_forXY(cimg,x,y) if (mask(x,y)) cimg(x,y) = nodataout;
                imgout[bout].write(cimg, *iCh);
            }
        }
        return imgout;
    }


    //! Perform Brovey pansharpening
    /*!
        geoimg: red, greem, and blue bands, optionally nir band, in any order
        weights: weights of Red, Green, Blue, and NIR (if provided), in that order
    */
    GeoImage pansharp_brovey(const GeoImage& geoimg, const GeoImage& panimg, CImg<float> weights, std::string filename) {
        // TODO - check inputs to contain RGB, and maybe NIR
        if (weights.size()==0)
            weights = geoimg.nbands() == 4 ? CImg<float>(4,1,1,1, 0.25, 0.25, 0.25, 0.25) : CImg<float>(3,1,1,1, 0.34, 0.33, 0.33);

        // create output image
        BoundingBox ext = geoimg.extent().intersect(panimg.extent());
        Point<double> res = panimg.resolution();
        int xsz(ext.width()/std::abs(res.x()));
        int ysz(ext.height()/std::abs(res.y()));
        CImg<double> bbox(4,1,1,1, ext.x0(), ext.y0(), ext.width(), ext.height());

        // create upscaled output file using nearest neighbor
        GeoImage imgout = GeoImage::create(filename, xsz, ysz, geoimg.nbands(), 
                                           panimg.srs(), bbox, geoimg.type().string());
        imgout.set_bandnames(geoimg.bandnames());
        // warp to common footprint and resolution
        geoimg.warp_into(imgout, GeoFeature(), 2);

        // create warped pan-band - TODO make faster by adjust extents directly
        GeoImage panout = GeoImage::create("", xsz, ysz, 1, panimg.srs(), bbox, panimg.type().string());
        panimg.warp_into(panout);

        // Chunk image
        CImg<float> r, g, b, n;
        CImg<float> pancimg;
        CImg<float> dnf;
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = imgout.chunks();
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            // this is a multidimensional array (X x Y x 1 x B)
            r = imgout["red"].read<float>(*iCh);
            g = imgout["green"].read<float>(*iCh);
            b = imgout["blue"].read<float>(*iCh);
            pancimg = panout.read<float>(*iCh);
            if (geoimg.band_exists("nir")) {
                n = imgout["nir"].read<float>(*iCh);
                pancimg = pancimg - weights[3] * n;
            }
            dnf = pancimg.get_div(weights[0] * r + weights[1] * g + weights[2] * b);
            imgout["red"].write<float>(r.mul(dnf), *iCh);
            imgout["green"].write<float>(g.mul(dnf), *iCh);
            imgout["blue"].write<float>(b.mul(dnf), *iCh);
            if (geoimg.band_exists("nir"))
                imgout["nir"].write<float>(n.mul(dnf), *iCh);
        }

        return imgout;
    }


    //! Runs the RX Detector (RXD) anamoly detection algorithm
    GeoImage rxd(const GeoImage& img, string filename) {
        if (img.nbands() < 2) throw std::runtime_error("RXD: At least two bands must be supplied");

        GeoImage imgout = GeoImage::create_from(img, filename, 1, "uint8");
        imgout.set_bandname("RXD", 1);

        CImg<double> covariance = img.spectral_covariance();
        CImg<double> K = covariance.invert();
        CImg<double> chip, chipout, pixel;

        // Calculate band means
        CImg<double> bandmeans(img.nbands());
        cimg_forX(bandmeans, x) {
            bandmeans(x) = img[x].stats()[2];
        }

        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = img.chunks();
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            chip = img.read<double>(*iCh);
            chipout = CImg<double>(chip);
            cimg_forXY(chip,x,y) {
                pixel = chip.get_crop(x,y,0,0,x,y,chip.depth()-1,0).unroll('x') - bandmeans;
                chipout(x,y) = (pixel * K.get_transpose() * pixel.get_transpose())[0];
            }
            imgout[0].write(chipout, *iCh);
        }
        return imgout;
    }

    //! Calculate spectral statistics and output to new image
    GeoImage spectral_statistics(const GeoImage& img, string filename) {
        if (img.nbands() < 2) {
            throw std::runtime_error("Must have at least 2 bands!");
        }

        GeoImage imgout = GeoImage::create_from(img, filename, 3, "float64");
        imgout.set_nodata(-32768);
        imgout.set_bandname("mean", 1);
        imgout.set_bandname("stddev", 2);
        imgout.set_bandname("numpixels", 3);

        CImg<double> stats;
        vector<Chunk>::const_iterator iCh;
        vector<Chunk> chunks = img.chunks();
        for (iCh=chunks.begin(); iCh!=chunks.end(); iCh++) {
            stats = img.spectral_statistics(*iCh);
            imgout[0].write(stats.get_slice(0), *iCh);
            imgout[1].write(stats.get_slice(1), *iCh);
            imgout[2].write(stats.get_slice(2), *iCh);
        }
        if (Options::verbose())
            std::cout << "Spectral statistics written to " << imgout.filename() << std::endl;
        return imgout;
    }



    }
} // namespace gip
