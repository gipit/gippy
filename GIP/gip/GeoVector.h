#ifndef GIP_GEOVECTOR_H
#define GIP_GEOVECTOR_H

#include <string>

#include <gdal/ogrsf_frmts.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

namespace gip {

    class GeoVector {
    public:

        //! \name Constructors/Destructors
        //! Default constructor
        GeoVector() : _OGRDataSource() {}
        //! Open existing source
        GeoVector(std::string);
        //! Create new file on disk
        GeoVector(std::string filename, OGRwkbGeometryType dtype);

        //! Copy constructor
        GeoVector(const GeoVector& vector);
        //! Assignment operator
        GeoVector& operator=(const GeoVector& vector);
        //! Destructor
        ~GeoVector() {}

        //! \name Data Information
        //! Get number of layers
        //int NumLayers() const { return _OGRDataSource.GetLayerCount(); }

    protected:

        //! Filename to dataset
        boost::filesystem::path _Filename;

        //! Underlying OGRDataSource
        boost::shared_ptr<OGRDataSource> _OGRDataSource;

    }; // class GeoVector

} // namespace gip

#endif
