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

#ifndef GIP_UTILS_H
#define GIP_UTILS_H

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <gdal/gdal_priv.h>
#include <vector>
#include <boost/filesystem.hpp>

namespace gip {

    // inline utility functions

    //! Conversion function, any type to string
    template<typename T> inline std::string to_string(const T& t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
    }

    inline std::string to_string(std::vector<std::string> vector) {
        std::string str("");
        for (unsigned int i=0; i<vector.size(); i++) 
            str = str + " " + vector[i];
        return str;
    }

    //! Returns GDAL Type corresponding to type passed in
    inline GDALDataType type2GDALtype(const std::type_info& info) {
        if (info == typeid(unsigned char)) return GDT_Byte;
        else if (info == typeid(unsigned short)) return GDT_UInt16;
        else if (info == typeid(short)) return GDT_Int16;
        else if (info == typeid(unsigned int)) return GDT_UInt32;
        else if (info == typeid(int)) return GDT_Int32;
        else if (info == typeid(float)) return GDT_Float32;
        else if (info == typeid(double)) return GDT_Float64;
        else {
            std::cout << "Data Type " << info.name() << " not supported" << std::endl;
            throw(std::exception());
        }
    }

    class Options {
    public:
        // \name Global Options (static properties)
        //! Get Config directory
        //static std::string ConfigDir() { return _ConfigDir.string(); }
        //! Set Config directory
        //static void SetConfigDir(std::string dir) { _ConfigDir = dir; }
        //! Default format when creating new files
        static std::string DefaultFormat() { return _DefaultFormat; }
        //! Set default format when creating new files
        static void SetDefaultFormat(std::string str) { _DefaultFormat = str; }
        //! Default chunk size when chunking an image
        static float ChunkSize() { return _ChunkSize; }
        //! Set chunk size, used when chunking an image
        static void SetChunkSize(float sz) { _ChunkSize = sz; }
        //! Get verbose level
        static int Verbose() { return _Verbose; }
        //! Set verbose level
        static void SetVerbose(int v) { 
            _Verbose = v;
            if (v > 4) {
                CPLPushErrorHandler(CPLDefaultErrorHandler);
            } else {
                CPLPushErrorHandler(CPLQuietErrorHandler);
            }
        }
        //! Get desired number of cores
        static int NumCores() { return _NumCores; }
        //! Set desired number of cores
        static void SetNumCores(int n) {
            _NumCores = n;
        }
        //! Get workdir
        static std::string WorkDir() { return _WorkDir; }
        //! Set workdir
        static void SetWorkDir(std::string workdir) { _WorkDir = workdir; }

    private:
            // Static options
        //! Configuration file directory
        //static boost::filesystem::path _ConfigDir;
        //! Default format
        static std::string _DefaultFormat;
        //! Chunk size used when chunking up an image
        static float _ChunkSize;
        //! Verbosity level
        static int _Verbose;
        //! Number of cores to use when multi threading
        static int _NumCores;
        //! Work dir
        static std::string _WorkDir;

    };

    //! Splits the string s on the given delimiter(s) and returns a list of tokens without the delimiter(s)
    /// <param name=s>The string being split</param>
    /// <param name=match>The delimiter(s) for splitting</param>
    /// <param name=removeEmpty>Removes empty tokens from the list</param>
    /// <param name=fullMatch>
    /// True if the whole match string is a match, false
    /// if any character in the match string is a match
    /// </param>
    /// <returns>A list of tokens</returns>
    inline std::vector<std::string> Split(const std::string& s, const std::string& match, bool removeEmpty=false, bool fullMatch=false) {
        using std::string;
        typedef string::size_type (string::*find_t)(const string& delim, string::size_type offset) const;
        std::vector<string> result;                 // return container for tokens
        string::size_type start = 0,           // starting position for searches
                          skip = 1;            // positions to skip after a match
        find_t pfind = &std::string::find_first_of; // search algorithm for matches

        if (fullMatch)
        {
            // use the whole match string as a key instead of individual characters skip might be 0. see search loop comments
            skip = match.length();
            pfind = &string::find;
        }

        while (start != std::string::npos)
        {
            // get a complete range [start..end)
            string::size_type end = (s.*pfind)(match, start);

            // null strings always match in string::find, but
            // a skip of 0 causes infinite loops. pretend that
            // no tokens were found and extract the whole string
            if (skip == 0) end = string::npos;

            string token = s.substr(start, end - start);
            if (!(removeEmpty && token.empty()))
            {
                // extract the token and add it to the result list
                result.push_back(token);
            }
            // start the next range
            if ((start = end) != string::npos) start += skip;
        }
        return result;
    }

    //! Parse string to array of ints
    inline std::vector<unsigned int> ParseToInts(const std::string& s) {
        std::vector<std::string> str = Split(s, " ,");
        std::vector<std::string>::const_iterator iv;
        std::vector<unsigned int> intarray;
        size_t loc;
        for (iv=str.begin();iv!=str.end();iv++) {
            loc = iv->find("-");
            if (loc==std::string::npos)
                intarray.push_back( atoi(iv->c_str()) );
            else {
                int b1 = atoi(iv->substr(0,loc).c_str());
                int b2 = atoi(iv->substr(loc+1).c_str());
                for (int i=b1;i<=b2;i++) intarray.push_back(i);
            }
        }
        return intarray;
    }

}

#endif
