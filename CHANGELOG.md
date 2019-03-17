
# Changelog
All notable changes to this project, (gippy)[https://github.com/gipit/gippy], will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [v1.0.1] - 2019-03-17
- fixes setting of nodata in GeoImage.open function
- allow dictionary file creation options to be passed to GeoImage.create() and cookie_cutter
- added kmeans algorithm
- fix histogram function, now returns doubles
- added spectral_statistics algorithm and GeoImage::spectral_statistics function
- Detailed CHANGELOG added

## [v1.0.0]
- Major restructuring and refactor
- Function naming changed, breaking backwards compatability
- Compiles and installs on MacOS
- GDAL data types now replaced with gip DataType class
- Tests added

## v0.3.5
- fixed install issues through PyPi
- removed uncessary swig generated functions

## v0.3.1
- fixed indexing of vectors to get features

## [v0.3.0]
- License changed from GPL v2 to Apache 2.0
- Added test framework
- Separated into python modules: gippy (core), gippy.tests, gippy.algorithms
- Cleaned up setup to use single 'gippy' distro directory
- Refactored: new base class GeoResource (linked to GDALMajorObject)
- Chunking code moved from GeoData to new ChunkSet class
- Removed 'chunking by chunk index' feature (poor implementation made it error prone)
- Added interpolation option to CookieCutter
- Added Resolution convenience function to GeoResource (gets resolution from Affine)
- Added SpectralCovariance and RXD (RX Anomoly detector) to algorithms
- Added GeoVectorResource, GeoVector, GeoFeature classes
- Added GeoImages class as collection of GeoImage objects
- Refactored CookieCutter
- Properly clear functions after image has been Processed to itself (written to own file)
- Stop opening GDALDataset as shared, which causes problems when user expects a different object
- Iterable containers: GeoImage (bands), GeoImages (images), GeoVector (features), GeoFeatures (attributes)
- Added SetGCPs function to GeoResource
- Added additional functions for retriving and searching values of attributes in vectors

## v0.2.4
- Added Affine to get affine transformation for SRS
- Python bindings now correctly handling Point data type (returned as tuple)

## v0.2.3
- Changed name of secondary probability image in fmask algorithm to -prob (from _prob)
- Suppressed stderr messages from GDAL (these should all be handled internally by GIPPY)

## v0.2.2
- Fixed bug in Indices algorithm

## v0.2.1
- Added SpectralStatistics algorithm and functions to GeoImage and GeoAlgorithms
- Added additional gdal helper functions
- Additional chunking functions and options. Chunking added to TimeSeries
- Cleanup

## [v0.2.0]
- Added BrowseImage for creating an RGB or single band JPG image from a GeoImage for easy viewing
- Refactored color management (removed class, uses single vector of band names)
- misc fixes
- Added ApplyMask
- Updated swig bindings to properly support numpy arrays as input
- CookieCutter now properly uses cutline

## v0.1.9
- Added TimeSeries, Extract, GetRandomPixels, GetPixelClasses functions to GeoImage python bindings
- Standardized datatypes using stdint.h for better interoperability
- 1-d Numpy to CImg conversion fixed

## v0.1.8
- Added additional operators on GeoRaster: min, max, exp, sign, acos, etc.
- Added LinearTransform algorithm

## v0.1.7
- Added union and transform functions to Rect
- Added additional coordinate convenience functions to GeoData
- Added crop option to CookieCutter

## v0.1.6
- Added NDWI and MSAVI2 to Indices function
- Overloaded SetMeta function to take in map<string, string> for multiple items
- Updated algorithms to all take optional metadata dictionary for output
- cleanup of GeoAlgorithms.h and GeoAlgorithms.cpp

## v0.1.5
- When opening an image default is now read-only
- Fixed bug with Process to new image
- Added license info to all files

## v0.1.0

Initial Release


[Unreleased]: https://github.com/gipit/gippy/compare/master...develop
[v1.0.1]: https://github.com/gipit/gippy/compare/1.0.0...1.0.1
[v1.0.0]: https://github.com/gipit/gippy/compare/0.3.0...1.0.0
[v0.3.0]: https://github.com/gipit/gippy/compare/0.2.0...0.3.0
[v0.2.0]: https://github.com/gipit/gippy/tree/0.2.0
