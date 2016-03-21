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

#ifndef GIP_DATATYPE_H
#define GIP_DATATYPE_H

#include <string>
#include <vector>
#include <gdal/gdal_priv.h>

namespace gip {

	enum DataTypes { Unknown=0, UInt8, UInt16, Int16, UInt32, Int32, Float32, Float64 };

	class DataType {
	public:
	    DataType() : _Type(0) {}

	    DataType(int dtype) : _Type(dtype) {}

	    DataType(GDALDataType dtype) : _Type(dtype) {}

	    DataType(std::string dtype) {
	        if (dtype == "UInt8") _Type = 1;
	        else if (dtype == "UInt16") _Type = 2;
	        else if (dtype == "Int16") _Type = 3;
	        else if (dtype == "UInt32") _Type = 4;
	        else if (dtype == "Int32") _Type = 5;
	        else if (dtype == "Float32") _Type = 6;
	        else if (dtype == "Float64") _Type = 7;
	        else _Type = 0;
	    }

	    DataType(const std::type_info& info) {
	        if (info == typeid(unsigned char)) _Type = 1;
	        else if (info == typeid(unsigned short)) _Type = 2;
	        else if (info == typeid(short)) _Type = 3;
	        else if (info == typeid(unsigned int)) _Type = 4;
	        else if (info == typeid(int)) _Type = 5;
	        else if (info == typeid(float)) _Type = 6;
	        else if (info == typeid(double)) _Type = 7;
	        else _Type = 0;
	    }

	    ~DataType() {}

	    DataTypes Type() {
	        return DataTypes(_Type);
	    }

	    int Int() {
	    	return _Type;
	    }

	    std::string String() {
	        std::vector<std::string> dts = {"Unknown", "UInt8", "UInt16", "Int16", "UInt32", "Int32", "Float32", "Float64"};
	        return dts[_Type];
	    }

	    GDALDataType GDALType() {
	        switch (_Type) {
	            case 1: return GDALDataType::GDT_Byte;
	            case 2: return GDALDataType::GDT_UInt16;
	            case 3: return GDALDataType::GDT_Int16;
	            case 4: return GDALDataType::GDT_UInt32;
	            case 5: return GDALDataType::GDT_Int32;
	            case 6: return GDALDataType::GDT_Float32;
	            case 7: return GDALDataType::GDT_Float64;
	            default: return GDALDataType::GDT_Unknown;
	        }
	    }

	    //! Return maximum value based on datatype
	    double MaxValue() const {
	        // TODO - base this on platform, not hard-coded
	        switch (_Type) {
	            case 1: return 255;
	            case 2: return 65535;
	            case 3: return 32767;
	            case 4: return 4294967295;
	            case 5: return 2147183647;
	            case 6: return 3.4E38;
	            case 7: return 3.4E38;
	            default: return 1.79E308;
	        }
	    }

	    //! Return minimum value based on datatype (TODO - get from limits?)
	    double MinValue() const {
	        switch (_Type) {
	            case 1: return 0;
	            case 2: return 0;
	            case 3: return -32768;
	            case 4: return 0;
	            case 5: return -2147183648;
	            case 6: return -3.4E38;
	            case 7: return -3.4E38;
	            default: return -1.79E308;
	        }
	    }

	    //! Return a suitable nodata value for this datatype
	    double NoData() const {
	        switch (_Type) {
	            case 1: return 0;
	            case 2: return 0;
	            case 3: return -32768;
	            case 4: return 0;
	            case 5: return -32768;
	            case 6: return -32768;
	            case 7: return -32768;
	            default: return 0;
	        }            
	    }

	    /*friend std::ostream& operator<< (std::ostream& o, DataType const& dt) {
	        o << dt.String();
	        return o;
	    }*/

	private:
	    int _Type;
	};

} // namespace gip

#endif