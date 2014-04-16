#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>

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
			return stream << "(" << p._x << ", " << p._y << ")";
		}
	protected:
		T _x;
		T _y;
	};

	//! (2D) Rect class
	template<typename T=int> class Rect {
	public:
		//! Default Constructor
		Rect() : _p0(0,0), _p1(0,0) {}
		//! Constructor takes in top left coordinate and width/height
		Rect(T x, T y, T width, T height) 
			: _p0(x,y), _p1(x+width-1,y+height-1) {
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

		//! Destructor
		~Rect() {};

		Point<T> p0() const { return _p0; }
		Point<T> p1() const { return _p1; }

		//Point<T> min_corner() const { return Point<T>(_x,_y); }
		//Point<T> max_corner() const { return Point<T>(_x+_width,_y+height); }

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
		Rect& Pad(int pad) {
			_p0 = _p0 - Point<T>(pad,pad);
			_p1 = _p1 + Point<T>(pad,pad);
			return *this;
		}
		Rect get_Pad(int pad) const {
			return Rect<T>(*this).Pad(pad);
		}
		//! Intersects Rect with argument Rect
		Rect& Intersect(const Rect& rect) {
			_p0 = Point<T>( std::max(_p0.x(), rect.x0()), std::max(_p0.y(), rect.y0()) );
			_p1 = Point<T>( std::min(_p1.x(), rect.x1()), std::min(_p1.y(), rect.y1()) );
			return *this;
		}
		//! Returns intersection of two Rects
		Rect get_Intersect(const Rect& rect) const {
			return Rect<T>(*this).Intersect(rect);
		}
		friend std::ostream& operator<<(std::ostream& stream,const Rect& r) {
			return stream << r._p0 << ", " << r._p1;
		}
	private:
		// top-left
		Point<T> _p0;
		// bottom-right
		Point<T> _p1;
	};


} // namespace GIP


#endif