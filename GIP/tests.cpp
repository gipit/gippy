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
#include <gip/Utils.h>

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
                //stats.print("Block " + to_string(i) + " stats");
                success = false;
            }
        }
        if (success)
            cout << "Test succeeded" << endl;
        else cout << "Test failed" << endl;
        return img;
    }

    GeoImage test_padded_chunk_registration(int pad, int chunk) {
        cout << "Chunking test with padding=" + to_string(pad) + " and " + to_string(chunk) + " chunks" << endl;
        // Create new image
        GeoImage img0("test_chunk_reg_input.tif", 10, 10, 1, GDT_Byte);
        CImg<unsigned char> arr(10, 10);
        arr *= 0;
        for (int i(4); i<7; ++i)
            for (int j(4); j<7; ++j)
                arr(i,j) = 1;
        img0.Write(arr);
        GeoImage img("test_chunk_reg_output.tif", 10, 10, 1, GDT_Byte);
        ChunkSet chunks = img.Chunks(pad, chunk);
        CImg<unsigned char> cimg_in, cimg_out;
        for (unsigned int i=0; i<chunks.Size(); i++) {
            cimg_in = img0.Read<unsigned char>(chunks[i]);
            img.Write(cimg_in, chunks[i]);
        }
        // Verify image
        chunks.Padding(0);
        bool success = true;
        CImg <unsigned char> ck0, ck1;
        for (unsigned int i=0; i<chunks.Size(); i++) {
            ck0 = img0.Read<unsigned char>(chunks[i]);
            ck1 = img.Read<unsigned char>(chunks[i]);
            if (ck0 != ck1)
                success = false;
        }
        if (success)
            cout << "Test succeeded" << endl;
        else cout << "Test failed" << endl;
        return img;
    }


} // namespace gip
