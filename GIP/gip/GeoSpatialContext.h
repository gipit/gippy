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

#ifndef GIP_GEOSPATIALCONTEXT_H
#define GIP_GEOSPATIALCONTEXT_H

#include <gip/geometry.h>


namespace gip {

    class GeoSpatialContext {
    public:
    	/*
        //! \name Constructors
        GeoSpatialContext(GeoResource* resource) {
        	_GeoResource.reset(resource);
        }
        ~GeoSpatialContext() {}

        //! Geolocated coordinates of a point within the resource
        Point<double> GeoLoc(float xloc, float yloc) const;
        //! Coordinates of top left
        Point<double> TopLeft() const;
        //! Coordinates of lower left
        Point<double> LowerLeft() const;
        //! Coordinates of top right
        Point<double> TopRight() const;
        //! Coordinates of bottom right
        Point<double> LowerRight() const;
        //! Minimum Coordinates of X and Y
        Point<double> MinXY() const;
        //! Maximum Coordinates of X and Y
        Point<double> MaxXY() const;
        //! Return projection definition in Well Known Text format
        string Projection() const {
            return _GDALDataset->GetProjectionRef();
        }
        //! Set projection definition in Well Known Text format
        GeoResource& SetProjection(string proj) {
            _GDALDataset->SetProjection(proj.c_str());
            return *this;
        }
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const;
        //! Get Affine transformation
        CImg<double> Affine() const {
            double affine[6];
            _GDALDataset->GetGeoTransform(affine);
            return CImg<double>(&affine[0], 6);
        }
        //! Set Affine transformation
        GeoResource& SetAffine(CImg<double> affine) {
            _GDALDataset->SetGeoTransform(affine.data());
            return *this;
        }
        GeoResource& SetAffine(double affine[6]) {
            _GDALDataset->SetGeoTransform(affine);
            return *this;
        }
        //! Get resolution convenience function
        Point<double> Resolution() const;
        //! Set coordinate system from another GeoResource
        GeoResource& SetCoordinateSystem(const GeoResource& res);

    private:
    	std::shared_ptr<GeoResource> _GeoResource;
    	*/

    }; // class GeoSpatialContext


} // namespace gip

#endif
