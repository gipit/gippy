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

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <ogr_spatialref.h>
#include <gip/utils.h>

namespace gip {

    //! (2D) Point Class
    template<typename T=int> class Point {
    public:
        //! Default Constructor
        Point() : _x(0), _y(0) {}
        //! Constructor takes in point coordinate
        Point(T x, T y) : _x(x), _y(y) {}
        //! Constructor takes in comma delimited string
        //! Destructor
        ~Point() {}

        Point<T> operator-(const Point<T>& p) { return Point<T>(_x-p._x, _y-p._y); }
        Point<T> operator+(const Point<T>& p) { return Point<T>(_x+p._x, _y+p._y); }

        bool operator==(const Point<T>& p) const {
            if (_x == p._x && _y == p._y)
                return true;
            else return false;
        }
        bool operator!=(const Point<T>& p) const {
            return operator==(p) ? false : true;
        }

        //! Get x coordinate
        T x() const { return _x; }
        //! Get y coordinate
        T y() const { return _y; }

        //! Transform between coordinate systems
        Point transform(std::string src, std::string dst) {
            if (src == dst) return *this;
            OGRSpatialReference _src;
            _src.SetFromUserInput(src.c_str());
            OGRSpatialReference _dst;
            _dst.SetFromUserInput(dst.c_str());
            OGRCoordinateTransformation* trans = OGRCreateCoordinateTransformation(&_src, &_dst);
            double newx(x()), newy(y());
            trans->Transform(1, &newx, &newy);
            OCTDestroyCoordinateTransformation(trans);
            return Point<T>(newx, newy);
        }

        // output operator
        friend std::ostream& operator<<(std::ostream& stream,const Point& p) {
            return stream << "(" << p._x << "," << p._y << ")";
        }
    protected:
        T _x;
        T _y;
    };

    //! (2D) Rect class
    template<typename T> class Rect {
    public:
        //! Default Constructor
        Rect() : _p0(0,0), _p1(0,0), _padding(0)  {}
        //! Constructor takes in top left coordinate and width/height
        Rect(T x, T y, T width, T height) 
            : _p0(x,y), _p1(x+width,y+height), _padding(0) {
            // Validate, x0 and y0 should always be the  min values
            /*if (_width < 0) {
                _width = abs(m_width);
                _x = _x - _width + 1;
            }
            if (_height < 0) {
                _height = abs(_height);
                _y = _y - _height + 1;
            }*/
        }
        Rect(Point<T> p0, Point<T> p1)
            : _p0(p0), _p1(p1), _padding(0) {
        }
        //! Copy constructor
        Rect(const Rect<T>& rect)
            : _p0(rect._p0), _p1(rect._p1), _padding(rect._padding) {}
        //! Assignment operator
        Rect& operator=(const Rect& rect) {
            if (this == &rect) return *this;
            _p0 = rect._p0;
            _p1 = rect._p1;
            _padding = rect._padding;
            return *this;
        }

        //! Destructor
        ~Rect() {};

        Point<T> p0() const { return _p0; }
        Point<T> p1() const { return _p1; }

        //Point<T> min_corner() const { return Point<T>(_x,_y); }
        //Point<T> max_corner() const { return Point<T>(_x+_width,_y+height); }

        //! Validity of rect
        T valid() const { return (area() == 0) ? false : true; }
        //! Area of the Rect
        T area() const { return abs(width()*height()); }
        //! Width of Rect
        T width() const { return _p1.x()-_p0.x(); }
        //! Height of Rect
        T height() const { return _p1.y()-_p0.y(); }
        //! Left x coordinate
        T x0() const { return _p0.x(); }
        //! Top y coordinate
        T y0() const { return _p0.y(); }
        //! Right x coordinate
        T x1() const { return _p1.x(); }
        //! Bottom y coordinate
        T y1() const { return _p1.y(); }

        bool operator==(const Rect<T>& rect) const {
            if (_p0 == rect._p0 && _p1 == rect._p1)
                return true;
            return false;
        }
        bool operator!=(const Rect<T>& rect) const {
            return !operator==(rect);
        }

        //! Transform between coordinate systems
        Rect<T> transform(std::string src, std::string dst) {
            if (src == dst) return *this;
            return Rect<T>(_p0.transform(src, dst), _p1.transform(src, dst));
        }

        //! Intersects Rect with argument Rect
        Rect intersect(const Rect& rect) {
            // transform rect
            Rect<T> r = Rect<T>(
                Point<T>( std::max(_p0.x(), rect.x0()), std::max(_p0.y(), rect.y0()) ),
                Point<T>( std::min(_p1.x(), rect.x1()), std::min(_p1.y(), rect.y1()) )
            );
            r.padding(_padding);
            return r;
        }

        //! Get padding
        T padding() const { return _padding; }
        //! Set padding
        Rect<T>& padding(T padding) {
            _padding = padding;
            return *this;
        }

        Rect<T>& pad() {
            return pad(_padding);
        }

        Rect<T>& pad(T pad) {
            _p0 = _p0 - Point<T>(pad,pad);
            _p1 = _p1 + Point<T>(pad,pad);
            return *this;
        }

        // Calculates union (outer bounding box) of Rect with argument Rect
        Rect union_with(const Rect& rect) {
            return Rect<T>(
                Point<T>( std::min(_p0.x(), rect.x0()), std::min(_p0.y(), rect.y0()) ),
                Point<T>( std::max(_p1.x(), rect.x1()), std::max(_p1.y(), rect.y1()) )
            );
        }

        friend std::ostream& operator<<(std::ostream& stream,const Rect& r) {
            return stream << r._p0 << "-" << r._p1;
        }

    protected:
        // top-left
        Point<T> _p0;
        // bottom-right
        Point<T> _p1;
        // Amount of padding around the rect (Rect is always stored WITHOUT padding)
        T _padding;
    };

    //! calculate union of all rects 
    template<typename T> Rect<T> union_all(std::vector< Rect<T> > rects) {
        Rect<T> unioned(rects[0]);
        for (unsigned int i=1; i<rects.size(); i++) {
            unioned = unioned.union_with(rects[i]);
        }
        return unioned;
    }

    typedef Rect<int> Chunk;
    typedef Rect<double> BoundingBox;

} // namespace GIP


#endif
