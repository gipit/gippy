#include <gip/GeoImage.h>
#include <gip/GeoRaster.h>

//#include <sstream>

namespace gip {
    using std::string;
    using std::vector;

    GeoImage::GeoImage(vector<string> filenames)
    	: GeoData(filenames[0]) {
    	vector<string>::const_iterator f;
    	LoadBands();
    	for (f=filenames.begin()+1; f!=filenames.end(); f++) {
    		GeoImage img(*f);
    		for (unsigned int b=0;b<img.NumBands(); b++) {
    			AddBand(img[b]);
    		}
    	}
    }

	// Copy constructor
	GeoImage::GeoImage(const GeoImage& image)
		: GeoData(image), _Colors(image.GetColors()) {
		for (uint i=0;i<image.NumBands();i++)
			_RasterBands.push_back( image[i] );
	}

	// Assignment operator
	GeoImage& GeoImage::operator=(const GeoImage& image) {
		// Check for self assignment
		if (this == &image) return *this;
		GeoData::operator=(image);
		_RasterBands.clear();
		for (uint i=0;i<image.NumBands();i++) _RasterBands.push_back( image[i] );
		_Colors = image.GetColors();
		//cout << Basename() << ": GeoImage Assignment - " << _GDALDataset.use_count() << " GDALDataset references" << endl;
		return *this;
	}

	string GeoImage::Info(bool bandinfo, bool stats) const {
		std::stringstream info;
		info << Filename() << " - " << _RasterBands.size() << " bands ("
				<< XSize() << "x" << YSize() << ") " << std::endl;
		info << "\tGeoData References: " << _GDALDataset.use_count() << " (&" << _GDALDataset << ")" << std::endl;
		info << "\tGeo Coordinates (top left): " << TopLeft().x() << ", " << TopLeft().y() << std::endl;
		info << "\tGeo Coordinates (lower right): " << LowerRight().x() << ", " << LowerRight().y() << std::endl;
		//info << "   References - GeoImage: " << _Ref << " (&" << this << ")";
		//_GDALDataset->Reference(); int ref = _GDALDataset->Dereference();
		//info << "  GDALDataset: " << ref << " (&" << _GDALDataset << ")" << endl;
		if (bandinfo) {
			for (unsigned int i=0;i<_RasterBands.size();i++) {
				string color = _Colors[i+1];
				if (!color.empty()) color = " (" + color + ")";
				info << "\tBand " << i+1 << color << ": " << _RasterBands[i].Info(stats);
			}
			//std::map<Colors::Color,int>::const_iterator pos;
			//for (pos=_Colors.begin();pos!=_Colors.end();pos++)
			//	info << Color::Name(pos->first) << " = " << pos->second << std::endl;
		}
		return info.str();
	}
	// Get band names
	vector<string> GeoImage::BandNames() const {
		std::vector<string> names;
		for (std::vector< GeoRaster >::const_iterator iRaster=_RasterBands.begin();iRaster!=_RasterBands.end();iRaster++) {
			names.push_back(iRaster->Description());
		}
		return names;
	}
	// Band indexing
	const GeoRaster& GeoImage::operator[](std::string col) const {
		//std::map<Color::Enum,int>::const_iterator pos;
		//pos = _Colors.find(col);
		int index(_Colors[col]);
		if (index > 0)
			return _RasterBands[index-1];
		else {
			// TODO - fix this - can't return NULL reference...?
			//std::cout << "No band of that color, returning band 0" << std::endl;
			throw std::out_of_range ("No band of color "+col);
		}
	}
    // Add a band
	GeoImage& GeoImage::AddBand(const GeoRaster& band) { //, unsigned int bandnum) {
		//if ((bandnum == 0) || (bandnum > _RasterBands.size())) bandnum = _RasterBands.size();
		//_RasterBands.insert(_RasterBands.begin()+bandnum-1, band);
		_RasterBands.push_back(band);
        SetColor(band.Description(),_RasterBands.size());
		return *this;
	}
    // Remove a band
	GeoImage& GeoImage::RemoveBand(unsigned int bandnum) {
		if (bandnum <= _RasterBands.size()) {
		    _RasterBands.erase(_RasterBands.begin()+bandnum-1);
            _Colors.Remove(bandnum);
		}
		return *this;
	}
    // Remove bands except given colors
	GeoImage& GeoImage::PruneBands(vector<string> colors) {
        bool keep = false;
        for (int i=NumBands(); i>0; i--) {
            keep = false;
            for (vector<string>::const_iterator icol=colors.begin(); icol!=colors.end(); icol++) if (*icol == _Colors[i]) keep = true;
            if (!keep) RemoveBand(i);
        }
        return *this;
    }

	// Replaces all Inf or NaN pixels with NoDataValue
	GeoImage& GeoImage::FixBadPixels() {
		typedef float T;
		for (unsigned int b=0;b<NumBands();b++) {
			for (unsigned int iChunk=1; iChunk<=NumChunks(); iChunk++) {
				CImg<T> img = (*this)[b].ReadRaw<T>(iChunk);
				T nodata = (*this)[b].NoDataValue();
				cimg_for(img,ptr,T)	if ( std::isinf(*ptr) || std::isnan(*ptr) ) *ptr = nodata;
				(*this)[b].WriteRaw(img,iChunk);
			}
		}
		return *this;
	}

	/*const GeoImage& GeoImage::ComputeStats() const {
		for (unsigned int b=0;b<NumBands();b++) _RasterBands[b].ComputeStats();
		return *this;
	}*/

	//! Load bands from dataset
	void GeoImage::LoadBands() {
		vector<unsigned int> bandnums; // = _Options.Bands();
		// Check for subdatasets
		vector<string> names = this->GetMetaGroup("SUBDATASETS","_NAME=");
		unsigned int numbands(names.size());
		if (names.empty()) numbands = _GDALDataset->GetRasterCount();
		unsigned int b;
		// If no bands provided, default to all bands in this dataset
		//if (bandnums.empty()) {
        for(b=0;b<numbands;b++) bandnums.push_back(b+1);
		/* else {
			// Check for out of bounds and remove
			for(vector<unsigned int>::iterator bpos=bandnums.begin();bpos!=bandnums.end();) {
				if ((*bpos > numbands) || (*bpos < 1))
					bpos = bandnums.erase(bpos);
				else bpos++;
			}
		}*/
		if (names.empty()) {
			// Load Bands
			for (b=0;b<bandnums.size(); b++) {
				_RasterBands.push_back( GeoRaster(*this,bandnums[b]) );
				SetColor(_RasterBands[b].Description(), bandnums[b]);
			}
		} else {
			// Load Subdatasets as bands, assuming 1 band/subdataset
			for(b=0;b<bandnums.size();b++) {
				_RasterBands.push_back( GeoData(names[bandnums[b]-1],_GDALDataset->GetAccess()) );
			}
			// Replace this dataset with first full frame band
			unsigned int index(0);
			for (unsigned int i=0;i<NumBands();i++) {
				if (_RasterBands[i].XSize() > _RasterBands[index].XSize()) index = i;
			}
			// Release current dataset, point to new one
			_GDALDataset.reset();
			_GDALDataset = _RasterBands[index]._GDALDataset;
		}
		Chunk();
	}

} // namespace gip
