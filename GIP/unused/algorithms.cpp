


//! k-means unsupervised classifier
GeoImage kmeans( const GeoImage& image, string filename, int classes, int iterations, float threshold ) {
    //if (Image.NumBands() < 2) throw GIP::Gexceptions::errInvalidParams("At least two bands must be supplied");
    if (Options::Verbose()) {
        cout << image.Basename() << " - k-means unsupervised classifier:" << endl
            << "  Classes = " << classes << endl
            << "  Iterations = " << iterations << endl
            << "  Pixel Change Threshold = " << threshold << "%" << endl;
    }
    // Calculate threshold in # of pixels
    threshold = threshold/100.0 * image.Size();

    GeoImageIO<float> img(image);
    // Create new output image
    GeoImageIO<unsigned char> imgout(GeoImage(filename, image, GDT_Byte, 1));

    // Get initial class estimates (uses random pixels)
    CImg<float> ClassMeans = img.GetPixelClasses(classes);

    int i;
    CImg<double> Pixel, C_img, DistanceToClass(classes), NumSamples(classes), ThisClass;
    CImg<unsigned char> C_imgout, C_mask;
    CImg<double> RunningTotal(classes,image.NumBands(),1,1,0);

    int NumPixelChange, iteration=0;
    do {
        NumPixelChange = 0;
        for (i=0; i<classes; i++) NumSamples(i) = 0;
        if (Options::Verbose()) cout << "  Iteration " << iteration+1 << std::flush;

        for (unsigned int iChunk=1; iChunk<=image[0].NumChunks(); iChunk++) {
            C_img = img.Read(iChunk);
            C_mask = img.NoDataMask(iChunk);
            C_imgout = imgout[0].Read(iChunk);

            CImg<double> stats;
            cimg_forXY(C_img,x,y) { // Loop through image
                // Calculate distance between this pixel and all classes
                if (C_mask(x,y)) {
                    Pixel = C_img.get_crop(x,y,0,0,x,y,0,C_img.spectrum()-1).unroll('x');
                    cimg_forY(ClassMeans,cls) {
                        ThisClass = ClassMeans.get_row(cls);
                        DistanceToClass(cls) = (Pixel - ThisClass).dot(Pixel - ThisClass);
                    }
                    // Get closest distance and see if it's changed since last time
                    stats = DistanceToClass.get_stats();
                    if (C_imgout(x,y) != (stats(4)+1)) {
                        NumPixelChange++;
                        C_imgout(x,y) = stats(4)+1;
                    }
                    NumSamples(stats(4))++;
                    cimg_forY(RunningTotal,iband) RunningTotal(stats(4),iband) += Pixel(iband);
                } else C_imgout(x,y) = 0;
            }
            imgout[0].Write(C_imgout,iChunk);
            if (Options::Verbose()) cout << "." << std::flush;
        }

        // Calculate new Mean class vectors
        for (i=0; i<classes; i++) {
            if (NumSamples(i) > 0) {
                cimg_forX(ClassMeans,x) {
                    ClassMeans(x,i) = RunningTotal(i,x)/NumSamples(i);
                    RunningTotal(i,x) = 0;
                }
                NumSamples(i) = 0;
            }
        }
        if (Options::Verbose()) cout << 100.0*((double)NumPixelChange/image.Size()) << "% pixels changed class" << endl;
        if (Options::Verbose()>1) cimg_printclasses(ClassMeans);
    } while ( (++iteration < iterations) && (NumPixelChange > threshold) );

    imgout.SetBandName("k-means", 1);
    //imgout.GetGDALDataset()->FlushCache();
    return imgout;
}

//! Spectral Matched Filter, with missing data
GeoImage SMF(const GeoImage& image, string filename, CImg<double> Signature) {
    GeoImage output(filename, image, GDT_Float32, 1);

    // Band Means
    CImg<double> means(image.NumBands());
    for (unsigned int b=0;b<image.NumBands();b++) means(b) = image[b].Mean();

    //vector< box<point> > Chunks = ImageIn.Chunk();
    return output;
}

//! Calculate spectral statistics and output to new image
GeoImage SpectralStatistics(const GeoImage& img, string filename) {
    if (img.NumBands() < 2) {
        throw std::runtime_error("Must have at least 2 bands!");
    }
    GeoImage imgout(filename, img, DataType("Float32"), 2);
    imgout.SetNoData(img[0].NoData());
    imgout.CopyMeta(img);
    imgout.SetBandName("Mean", 1);
    imgout.SetBandName("StdDev", 2);

    CImgList<double> stats;
    ChunkSet chunks(img.XSize(),img.YSize());
    for (unsigned int iChunk=0; iChunk<chunks.Size(); iChunk++) {
        if (Options::Verbose() > 2) 
            std::cout << "Processing chunk " << chunks[iChunk] << " of " << img.Size() << std::endl;
        stats = img.SpectralStatistics(chunks[iChunk]);
        imgout[0].Write(stats[0], chunks[iChunk]);
        imgout[1].Write(stats[1], chunks[iChunk]);
    }
    if (Options::Verbose())
        std::cout << "Spectral statistics written to " << imgout.Filename() << std::endl;
    return imgout;
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

    if (Options::Verbose() > 0) {
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