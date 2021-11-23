#ifndef GAMMAFUNCTION
#define GAMMAFUNCTION

#ifndef _VECTOR_
    #include <vector>
#endif

    #define ITERATIONSMAX 100
    #define EPSTHRESHOLD 3.0e-7
    #define FPMINIMUM 1.0e-30


    double factorial(int n);
    double standardGaussianInvCDF(double prob);
    float gammaCDF(float x, double beta, double gamma,  double pZero) ;
    void probabilityWeightedMoments(std::vector<float> series, int n, std::vector<float> &probWeightedMoments, float a, float b, bool isBeta);
    void logLogisticFitting(std::vector<float> probWeightedMoments, double *alpha, double *beta, double *gamma);
    float logLogisticCDF(float myValue, double alpha, double beta, double gamma);

    double gammaFunction(double value);
    double gammaNaturalLogarithm(double value);
    void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn);
    void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn);
    double incompleteGamma(double alpha, double x, double *lnGammaValue); // incomplete + complete
    double incompleteGamma(double alpha, double x); // only incomplete
    bool gammaFitting(std::vector<float> &series, int n, double* beta, double* gamma,  double* pZero);



#endif // GAMMAFUNCTION

