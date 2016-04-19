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
//#include <ogr_srs_api.h>
#include <ogr_spatialref.h>
#include <gip/gip.h>
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
    template<typename T=int> class Rect {
    public:
        //! Default Constructor
        Rect() : _p0(0,0), _p1(-1,-1), _padding(0) {}
        //! Constructor takes in top left coordinate and width/height
        Rect(T x, T y, T width, T height) 
            : _p0(x,y), _p1(x+width-1,y+height-1), _padding(0) {
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
        T width() const { return _p1.x()-_p0.x()+1; }
        //! Height of Rect
        T height() const { return _p1.y()-_p0.y()+1; }
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

        //! Get padding
        unsigned int padding() const { return _padding; }
        //! Set padding
        Rect& padding(unsigned int padding) {
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

        //! Transform between coordinate systems
        Rect& transform(std::string src, std::string dst) {
            if (src == dst) return *this;
            OGRSpatialReference _src = OGRSpatialReference(src.c_str());
            OGRSpatialReference _dst = OGRSpatialReference(src.c_str());
            OGRCoordinateTransformation* trans = OGRCreateCoordinateTransformation(&_src, &_dst);
            double x, y;
            x = _p0.x();
            y = _p0.y();
            trans->Transform(1, &x, &y);
            _p0 = Point<T>(x, y);
            x = _p1.x();
            y = _p1.y();
            trans->Transform(1, &x, &y);
            _p1 = Point<T>(x, y);
            delete trans;
            return *this;
        }

        Rect& pad() {
            return pad(_padding);
        }

        Rect& pad(int pad) {
            _p0 = _p0 - Point<T>(pad,pad);
            _p1 = _p1 + Point<T>(pad,pad);
            return *this;
        }

        Rect get_pad() const {
            return get_pad(_padding);
        }

        Rect get_pad(int pad) const {
            return Rect<T>(*this).pad(pad);
        }

        //! Intersects Rect with argument Rect
        Rect& intersect(const Rect& rect) {
            _p0 = Point<T>( std::max(_p0.x(), rect.x0()), std::max(_p0.y(), rect.y0()) );
            _p1 = Point<T>( std::min(_p1.x(), rect.x1()), std::min(_p1.y(), rect.y1()) );
            return *this;
        }
        //! Returns intersection of two Rects
        Rect get_intersect(const Rect& rect) const {
            return Rect<T>(*this).intersect(rect);
        }

        // Calculates union of Rect with argument Rect
        Rect& union_with(const Rect& rect) {
            _p0 = Point<T>( std::min(_p0.x(), rect.x0()), std::min(_p0.y(), rect.y0()) );
            _p1 = Point<T>( std::max(_p1.x(), rect.x1()), std::max(_p1.y(), rect.y1()) );
            return *this;            
        }
        //! Returns outer bounding box of two rects
        /*Rect get_union(const Rect& rect) const {
            return Rect<T>(*this).union(rect);
        }*/
        friend std::ostream& operator<<(std::ostream& stream,const Rect& r) {
            return stream << r._p0 << "-" << r._p1;
        }
    private:
        // top-left
        Point<T> _p0;
        // bottom-right
        Point<T> _p1;
        // Amount of padding around the rect (Rect is always stored WITHOUT padding)
        unsigned int _padding;
    };

    //! calculate union of all rects 
    template<typename T> Rect<T> union_all(std::vector< Rect<T> > rects) {
        Rect<T> unioned(rects[0]);
        for (unsigned int i=1; i<rects.size(); i++) {
            unioned.union_with(rects[i]);
        }
        return unioned;
    }


    //! Collection of Rects representing a chunked up region (i.e. image)
    class ChunkSet {
    public:
        //! Default constructor
        ChunkSet()
            : _xsize(0), _ysize(0), _padding(0) {}

        //! Constructor taking in image size
        ChunkSet(unsigned int xsize, unsigned int ysize, unsigned int padding=0, unsigned int numchunks=0)
            : _xsize(xsize), _ysize(ysize), _padding(padding) {
            create_chunks(numchunks);
        }

        //! Copy constructor
        ChunkSet(const ChunkSet& chunks)
            : _xsize(chunks._xsize), _ysize(chunks._ysize), _padding(chunks._padding) {
            create_chunks(chunks.size());
        }

        //! Assignment copy
        ChunkSet& operator=(const ChunkSet& chunks) {
            if (this == & chunks) return *this;
            _xsize = chunks._xsize;
            _ysize = chunks._ysize;
            _padding = chunks._padding;
            create_chunks(chunks.size());
            return *this;
        }
        ~ChunkSet() {}

        //! Get width of region
        unsigned int xsize() const { return _xsize; }

        //! Get height of region
        unsigned int ysize() const { return _ysize; }

        //! Determine if region is valid
        bool valid() const {
            return (((_xsize * _ysize) == 0) || (size() == 0)) ? false : true;
        }

        //! Get number of chunks
        unsigned int size() const { return _Chunks.size(); }

        //! Get amount of padding in pixels
        unsigned int padding() const { return _padding; }

        //! Set padding
        ChunkSet& padding(unsigned int _pad) {
            _padding = _pad;
            create_chunks(_Chunks.size());
            return *this;
        }

        //! Get a chunk
        Rect<int>& operator[](unsigned int index) { 
            // Call const version
            return const_cast< Rect<int>& >(static_cast<const ChunkSet&>(*this)[index]);
        }
        //! Get a chunk, const version
        const Rect<int>& operator[](unsigned int index) const { 
            if (index < _Chunks.size())
                return _Chunks[index];
            throw std::out_of_range("No chunk " + to_string(index));
        }

    private:
        //! Function to chunk up region
        std::vector< Rect<int> > create_chunks(unsigned int numchunks=0) {
            unsigned int rows;

            if (numchunks == 0) {
                rows = floor( ( Options::chunksize() *1024*1024) / sizeof(double) / xsize() );
                rows = rows > ysize() ? ysize() : rows;
                numchunks = ceil( ysize()/(float)rows );
            } else {
                rows = int(ysize() / numchunks);
            }

            _Chunks.clear();
            Rect<int> chunk;
            /*if (Options::verbose() > 3) {
                std::cout << Basename() << ": chunking into " << numchunks << " chunks (" 
                    << Options::chunksize() << " MB max each)" << std::endl;
            }*/
            for (unsigned int i=0; i<numchunks; i++) {
                chunk = Rect<int>(0, rows*i, xsize(), std::min(rows*(i+1),ysize())-(rows*i) );
                chunk.padding(_padding);
                _Chunks.push_back(chunk);
                //if (Options::verbose() > 3) std::cout << "  Chunk " << i << ": " << chunk << std::endl;
            }
            return _Chunks;            
        }

        //! Width (columns) of region
        unsigned int _xsize;
        //! Height (rows) of region
        unsigned int _ysize;
        //! Padding to apply to rects (dimensions are always the rect without padding)
        unsigned int _padding;

        //! Coordinates of the chunks
        std::vector< Rect<int> > _Chunks;
    };

} // namespace GIP


#endif
