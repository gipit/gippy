/*##############################################################################
#    GIPPY: Geospatial Image Processing library for Python
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

#ifndef GIP_GEOEXTENT_H
#define GIP_GEOEXTENT_H


namespace gip {

    class GeoExtent {
    public:
        //! \name Constructors
        //! Constructor requires a GeoResource
        GeoExtent(GeoResource* resource) 
            : _GeoResource(resource) {}
        //! Destructor
        ~GeoExtent() {}

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
        virtual string Projection() const = 0;
        //! Return projection as OGRSpatialReference
        OGRSpatialReference SRS() const;
        //! Get Affine transformation
        virtual CImg<double> Affine() const = 0;

    protected:

    private:
        GeoResource* _GeoResource;

    } // class GeoExtent


} // namespace gip

#endif