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
        Rect() : _p0(0,0), _p1(0,0) {}
        //! Constructor takes in top left coordinate and width/height
        Rect(T x, T y, T width, T height) 
            : _p0(x,y), _p1(x+width,y+height) {
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
            : _p0(p0), _p1(p1) {
        }
        //! Copy constructor
        Rect(const Rect<T>& rect)
            : _p0(rect._p0), _p1(rect._p1) {}
        //! Assignment operator
        Rect& operator=(const Rect& rect) {
            if (this == &rect) return *this;
            _p0 = rect._p0;
            _p1 = rect._p1;
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
        Rect transform(std::string src, std::string dst) {
            if (src == dst) return *this;
            OGRSpatialReference _src;
            _src.SetFromUserInput(src.c_str());
            OGRSpatialReference _dst;
            _dst.SetFromUserInput(dst.c_str());
            OGRCoordinateTransformation* trans = OGRCreateCoordinateTransformation(&_src, &_dst);
            double x, y;
            x = _p0.x();
            y = _p0.y();
            trans->Transform(1, &x, &y);
            Point<T> pt0 (x, y);
            x = _p1.x();
            y = _p1.y();
            trans->Transform(1, &x, &y);
            Point<T> pt1(x, y);
            delete trans;
            return Rect<T>(pt0, pt1);
        }

        //! Intersects Rect with argument Rect
        Rect intersect(const Rect& rect) {
            // transform rect
            return Rect<T>(
                Point<T>( std::max(_p0.x(), rect.x0()), std::max(_p0.y(), rect.y0()) ),
                Point<T>( std::min(_p1.x(), rect.x1()), std::min(_p1.y(), rect.y1()) )
            );
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
    };

    //! calculate union of all rects 
    /*
    template<typename T> Rect<T> union_all(std::vector< Rect<T> > rects) {
        Rect<T> unioned(rects[0]);
        for (unsigned int i=1; i<rects.size(); i++) {
            unioned.union_with(rects[i]);
        }
        return unioned;
    }
    */

    //! Rect representing region of interest on a raster (ie pixel coordinates)
    class Chunk : public Rect<int> {
    public:
        //! Default Constructor
        Chunk() : Rect<int>(), _padding(0) {}
        //! Constructor takes in top left coordinate and width/height
        Chunk(int x, int y, int width, int height) 
            : Rect<int>(x, y, width, height), _padding(0) {}
        Chunk(Point<int> p0, Point<int> p1)
            : Rect<int>(p0, p1), _padding(0) {
        }
        //! Copy constructor
        Chunk(const Chunk& ch)
            : Rect<int>(ch), _padding(ch._padding) {}
        //! Assignment operator
        Chunk& operator=(const Chunk& ch) {
            if (this == &ch) return *this;
            Rect<int>::operator=(ch);
            _padding = ch._padding;
            return *this;
        }

        //! Intersects Rect with argument Rect
        Chunk intersect(const Chunk& rect) {
            // transform rect
            Chunk ch = Chunk(
                Point<int>( std::max(_p0.x(), rect.x0()), std::max(_p0.y(), rect.y0()) ),
                Point<int>( std::min(_p1.x(), rect.x1()), std::min(_p1.y(), rect.y1()) )
            );
            ch.padding(_padding);
            return ch;
        }

        //! Get padding
        unsigned int padding() const { return _padding; }
        //! Set padding
        Chunk& padding(unsigned int padding) {
            _padding = padding;
            return *this;
        }

        //! Determines if ROI is valid (not valid if height or width is 0 or less)
        //bool valid() const { if (width() <= 0 || height() <=0) return false; else return true; }

        //! Shift Rect by (x,y,z)
        /*Rect& Shift(T x, T y) {
            _x += x;
            _y += y;
            return *this;
        }
        //! Shift Rect by (x,y) and return new Rect
        Rect get_Shift(T x, T y) const {
            return Rect(*this).Shift(x,y);
        }*/

        Chunk& pad() {
            return pad(_padding);
        }

        Chunk& pad(int pad) {
            _p0 = _p0 - Point<int>(pad,pad);
            _p1 = _p1 + Point<int>(pad,pad);
            return *this;
        }

    private:
        // Amount of padding around the rect (Rect is always stored WITHOUT padding)
        unsigned int _padding;
    };

    typedef Rect<double> BoundingBox;

} // namespace GIP


#endif
