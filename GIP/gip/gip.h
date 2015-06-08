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

#ifndef GIP_H
#define GIP_H

// logging
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <gdal/gdal_priv.h>
#include <gdal/ogrsf_frmts.h>

namespace gip {

    //enum DataType { Unknown, Byte, UInt16, Int16, UInt32, Int32, Float32, Float64 };

    void LogLevel(int level) {
        boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= (5-level)
        );
    }

    // Register file formats with GDAL and OGR
    void gdalinit() {
        GDALAllRegister();
        OGRRegisterAll();
        CPLPushErrorHandler(CPLQuietErrorHandler);
    }

}

#endif

