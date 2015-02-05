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

#include <gip/gip_gdal.h>

namespace gip {
	using std::cout;

    std::string FileExtension() {
        std::string format = Options::DefaultFormat();
        GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format.c_str());
        return driver->GetMetadataItem(GDAL_DMD_EXTENSION);
    }

    //! Warp a single image into output image with cutline
    GeoImage& WarpToImage(const GeoImage& imgin, GeoImage& imgout, GDALWarpOptions *psWarpOptions, OGRGeometry* site) {
        if (Options::Verbose() > 2) cout << imgin.Basename() << " warping into " << imgout.Basename() << " " << std::flush;

        // Create cutline transform to pixel coordinates
        char **papszOptionsCutline = NULL;
        papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "DST_SRS", imgout.Projection().c_str() );
        papszOptionsCutline = CSLSetNameValue( papszOptionsCutline, "INSERT_CENTER_LONG", "FALSE" );
        CutlineTransformer oTransformer;

        oTransformer.hSrcImageTransformer = GDALCreateGenImgProjTransformer2( imgin.GetGDALDataset(), NULL, papszOptionsCutline );
        OGRGeometry* site_t = site->clone();
        site_t->transform(&oTransformer);

        //psWarpOptions->hCutline = site_t;
        char* wkt;
        site_t->exportToWkt(&wkt);
        psWarpOptions->papszWarpOptions = CSLSetNameValue(psWarpOptions->papszWarpOptions,"CUTLINE", wkt);

        // set options
        //psWarpOptions->papszWarpOptions = CSLDuplicate(papszOptions);
        psWarpOptions->hSrcDS = imgin.GetGDALDataset();
        psWarpOptions->pTransformerArg =
            GDALCreateGenImgProjTransformer( imgin.GetGDALDataset(), imgin.GetGDALDataset()->GetProjectionRef(),
                                            imgout.GetGDALDataset(), imgout.GetGDALDataset()->GetProjectionRef(), TRUE, 0.0, 0 );
        psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

        // Perform transformation
        GDALWarpOperation oOperation;
        oOperation.Initialize( psWarpOptions );
        //if (Options::Verbose() > 3) cout << "Error: " << CPLGetLastErrorMsg() << endl;
        oOperation.ChunkAndWarpMulti( 0, 0, imgout.XSize(), imgout.YSize() );

        // destroy things
        GDALDestroyGenImgProjTransformer( psWarpOptions->pTransformerArg );
        GDALDestroyGenImgProjTransformer( oTransformer.hSrcImageTransformer );
        CSLDestroy( papszOptionsCutline );
        OGRGeometryFactory::destroyGeometry(site_t);
        return imgout;
    }

} // namespace gip