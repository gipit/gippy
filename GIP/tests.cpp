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

#include <iostream>
#include <gip/tests.h>
#include <gip/Utils.h>

namespace gip {
    using std::cout;
    using std::endl;

    GeoImage test_chunking(int pad, int chunk) {
        cout << "Chunking test with padding=" + to_string(pad) + " and " + to_string(chunk) + " chunks" << endl;
        // Create new image
        GeoImage img("test_chunking.tif", 100, 100, 1, GDT_Byte);
        ChunkSet chunks = img.Chunks(pad, chunk);
        CImg<unsigned char> cimg_in, cimg_out;
        for (unsigned int i=0; i<chunks.Size(); i++) {
            cimg_in = img.Read<unsigned char>(chunks[i]);
            img.Write(cimg_in + i, chunks[i]);
        }
        // Verify image
        chunks.Padding(0);
        CImg<double> stats;
        bool success = true;
        for (unsigned int i=0; i<chunks.Size(); i++) {
            stats = img.Read<unsigned char>(chunks[i]).get_stats();
            if ((stats[0] != i) || (stats[1] != i) || (stats[2] != i)) {
                cimg_printstats(stats, "Block " + to_string(i) + " stats");
                success = false;
            }
        }
        if (success)
            cout << "Test succeeded" << endl;
        else cout << "Test failed" << endl;
        return img;
    }


} // namespace gip