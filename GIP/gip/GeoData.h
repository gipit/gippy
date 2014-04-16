#ifndef GIP_GEODATA_H
#define GIP_GEODATA_H

#include <vector>
#include <string>
#include <map>
#include <gdal/gdal_priv.h>
#include <boost/shared_ptr.hpp>

#include <boost/filesystem.hpp>

#include <gip/Utils.h>

#include <gip/geometry.h>

namespace gip {
	//class GeoData : boost::enable_shared_from_this<GeoData> {
	typedef std::map<std::string,std::string> dictionary;
	typedef Rect<int> iRect;
	typedef Point<int> iPoint;

	class GeoData {
	public:

		//! \name Constructors/Destructor
		//! Default constructor
		GeoData() : _GDALDataset() {}
		//! Open existing file
		GeoData(std::string, bool=true);
		//! Create new file on disk
		GeoData(int, int, int, GDALDataType, std::string, dictionary = dictionary());
		//! Copy constructor
		GeoData(const GeoData&);
		//! Assignment copy
		GeoData& operator=(const GeoData&);
		//! Destructor
		~GeoData();

		//! \name File Information
		//! Full filename of dataset
		//boost::filesystem::path Filename() const { return _Filename; }
		std::string Filename() const { return _Filename.string(); }
		//! Filename without path
		std::string Basename() const { return _Filename.stem().string(); }
		//! File format of dataset
		std::string Format() const { return _GDALDataset->GetDriver()->GetDescription(); }
		//! Return data type
		virtual GDALDataType DataType() const { return GDT_Unknown; }
		//! Return size of data type (in bytes)
		//int DataTypeSize() const;

		//! Get GDALDataset object - use cautiously
		GDALDataset* GetGDALDataset() const { return _GDALDataset.get(); }

		//! \name Spatial Information
		//! X Size of image/raster, in pixels
		unsigned int XSize() const { return _GDALDataset->GetRasterXSize(); }
		//! Y Size of image/raster, in pixels
		unsigned int YSize() const { return _GDALDataset->GetRasterYSize(); }
		//! Total number of pixels
		unsigned long Size() const { return XSize() * YSize(); }
		//! Geolocated coordinates of a pixel
		Point<double> GeoLoc(float xloc, float yloc) const;
		//! Coordinates of top left
		Point<double> TopLeft() const { return GeoLoc(0,0); }
		//! Coordinates of bottom right
		Point<double> LowerRight() const { return GeoLoc(XSize()-1,YSize()-1); }

		//! \name Metadata functions
		//! Get metadata item
		std::string GetMeta(std::string key) const {
			const char* item = _GDALDataset->GetMetadataItem(key.c_str());
			if (item == NULL) return ""; else return item;
		}
		//! Set metadata item
		GeoData& SetMeta(std::string key, std::string item) {
			_GDALDataset->SetMetadataItem(key.c_str(),item.c_str());
			return *this;
		}
		//! Copy Meta data from input file.  Currently no error checking
		GeoData& CopyMeta(const GeoData& img);
		//! Copy coordinate system
		GeoData& CopyCoordinateSystem(const GeoData&);
		//! Get group of metadata
		std::vector<std::string> GetMetaGroup(std::string group,std::string filter="") const;

		//! \name Processing functions
        //! Get the number of chunks used for processing image
		unsigned int NumChunks() const {
			//if (_Chunks.size() == 0) Chunk();
            return _Chunks.size();
		}

		//! Break up image into chunks
		void Chunk(unsigned int = 0) const;

	protected:
		//! Filename to dataset
		boost::filesystem::path _Filename;
		//! Underlying GDALDataset of this file
		boost::shared_ptr<GDALDataset> _GDALDataset;

		//! Vector of chunk coordinates
		mutable std::vector< Rect<int> > _Chunks;
		mutable std::vector< Rect<int> > _PadChunks;

	}; //class GeoData

} // namespace gip

#endif
