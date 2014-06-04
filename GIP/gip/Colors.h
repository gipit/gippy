#!/usr/bin/env python
################################################################################
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
################################################################################

#ifndef COLORS_H_INCLUDED
#define COLORS_H_INCLUDED

#include <map>
#include <string>

namespace gip {
    //!  Colors class to specify series of band colors
    class Colors {
    public:
        //! Default Colors (none set)
        Colors() {}
        //! Set standard band numbers colors (RGB+NIR)
        Colors(int blue, int green, int red, int nir) {
            //, int swir1, int swir2, int lwir) {
            SetColor("Blue", blue);
            SetColor("Green", green);
            SetColor("Red", red);
            SetColor("NIR", nir);
            //SetColor("SWIR1", swir1);
            //SetColor("SWIR2", swir2);
            //SetColor("LWIR", lwir);
        }
        //! Default destructor
        ~Colors() {}
        //! Set color by color name and band number
        void SetColor(std::string col, int bandnum) {
            _ColorsToBandNums[col] = bandnum;
            _BandNumsToColors[bandnum] = col;
        }
        // Set color by color number (1+) and band number
        /*void SetColor(int col, int bandnum) {
            std::string color;
            switch (col) {
                case 1: color = "Blue";
                case 2: color = "Green";
                case 3: color = "Red";
                case 4: color = "NIR";
                case 5: color = "SWIR1";
                case 6: color = "SWIR2";
                case 7: color = "LWIR";
                default: return;
            }
            SetColor(color, bandnum);
        }*/
        //! Get Bandnumber for given color (0 if no band)
        int operator[](std::string col) const {
            if (_ColorsToBandNums.find(col) != _ColorsToBandNums.end())
                return _ColorsToBandNums[col];
            else return 0;
        }
        //! Get Color for given band number
        std::string operator[](int col) const {
            if (_BandNumsToColors.find(col) != _BandNumsToColors.end()) {
                return _BandNumsToColors[col];
            }
            else return "None";
        }
        void Remove(unsigned int bandnum) {
            if (bandnum <= _ColorsToBandNums.size()) {
                _ColorsToBandNums.erase(_BandNumsToColors[bandnum]);
                _BandNumsToColors.erase(bandnum);
            }
        }

    private:
        //! Mapping of colors to band numbers
        mutable std::map<std::string,int> _ColorsToBandNums;
        //! Mapping of band numbers to colors
        mutable std::map<int,std::string> _BandNumsToColors;
    };
}

#endif // COLORS_H_INCLUDED
