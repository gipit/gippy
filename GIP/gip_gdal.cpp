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