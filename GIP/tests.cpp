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

#include <gip/tests.h>

namespace gip {

	GeoImage test_chunking() {
		// Create new image
		GeoImage img("test_chunking.tif", 100, 100, 1, GDT_Byte);
		ChunkSet chunks = img.Chunks(0, 100);
		CImg<unsigned char> cimg_in, cimg_out;
		for (unsigned int i=0; i<chunks.Size(); i++) {
			cimg_in = img.Read<unsigned char>(chunks[i]);
			img.Write(cimg_in + i, chunks[i]);
		}
		return img;
	}


} // namespace gip