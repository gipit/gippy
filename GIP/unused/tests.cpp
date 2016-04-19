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

#include <iostream>
#include <gip/tests.h>
#include <gip/utils.h>

namespace gip {
    using std::string;
    using std::cout;
    using std::endl;

    GeoImage test_reading(string filename) {
        cout << "Reading test: " << filename << endl;
        GeoImage img(filename);
        cout << img.Info() << endl;
        return img;
    }

    GeoImage create_test_image() {
        return GeoImage("test_image.tif", 100, 100, 1, DataType("UInt8"));
    }

    GeoImage test_chunking(int pad, int chunk) {
        cout << "Chunking test with padding=" + to_string(pad) + " and " + to_string(chunk) + " chunks" << endl;
        // Create new image
        GeoImage img("test_chunking.tif", 100, 100, 1, DataType("UInt8"));
        ChunkSet chunks = img.chunks(pad, chunk);
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
                //stats.print("Block " + to_string(i) + " stats");
                success = false;
            }
        }
        if (success)
            cout << "Test succeeded" << endl;
        else cout << "Test failed" << endl;
        return img;
    }


} // namespace gip
