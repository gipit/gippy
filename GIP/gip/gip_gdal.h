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