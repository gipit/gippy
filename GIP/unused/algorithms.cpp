




//! Spectral Matched Filter, with missing data
GeoImage SMF(const GeoImage& image, string filename, CImg<double> Signature) {
    GeoImage output(filename, image, GDT_Float32, 1);

    // Band Means
    CImg<double> means(image.NumBands());
    for (unsigned int b=0;b<image.NumBands();b++) means(b) = image[b].Mean();

    //vector< box<point> > Chunks = ImageIn.Chunk();
    return output;
}



CImg<double> SpectralCorrelation(const GeoImage& image, CImg<double> covariance) {
    // Correlation matrix
    if (covariance.size() == 0) covariance = SpectralCovariance(image);

    unsigned int NumBands = image.NumBands();
    unsigned int b;

    // Subtract Mean
    //CImg<double> means(NumBands);
    //for (b=0; b<NumBands; b++) means(b) = image[b].Mean();
    //covariance -= (means.get_transpose() * means);

    CImg<double> stddev(NumBands);
    for (b=0; b<NumBands; b++) stddev(b) = image[b].StdDev();
    CImg<double> Correlation = covariance.div(stddev.get_transpose() * stddev);

    if (Options::verbose() > 0) {
        cout << image.Basename() << " Spectral Correlation Matrix:" << endl;
        cimg_forY(Correlation,y) {
            cout << "\t";
            cimg_forX(Correlation,x) {
                cout << std::setw(18) << Correlation(x,y);
            }
            cout << endl;
        }
    }

    return Correlation;
}