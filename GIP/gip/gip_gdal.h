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

#ifndef GIP_GDAL_H
#define GIP_GDAL_H

#include <string>
#include <gip/GeoImage.h>
#include <gdal/ogrsf_frmts.h>
#include <gdal/gdalwarper.h>

namespace gip {

    class CutlineTransformer : public OGRCoordinateTransformation {
    public:
        void *hSrcImageTransformer;

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

    //! Get file extension for currently set file format
    std::string FileExtension();

    GeoImage& WarpToImage(const GeoImage&, GeoImage&, GDALWarpOptions*, OGRGeometry*);

}


#endif