#ifndef GAMMAFUNCTION
#define GAMMAFUNCTION

#ifndef _VECTOR_
    #include <vector>
#endif

    #define ITERATIONSMAX 100
    #define EPSTHRESHOLD 3.0e-7
    #define FPMINIMUM 1.0e-30

// functions
    double factorial(int n);
    double standardGaussianInvCDF(double prob);
    float generalizedGammaCDF(float x, double beta, double gamma,  double pZero) ;
    double generalizedGammaCDF(double x, double beta, double gamma,  double pZero);
    float inverseGeneralizedGammaCDF(double valueProbability, double alpha, double beta, double accuracy,double pZero,double outlierStep);
    float probabilityGamma(float x, double alfa, double gamma, float gammaFunc);
    float probabilityGamma(float x, double alpha, double beta);
    void probabilityWeightedMoments(std::vector<float> series, int n, std::vector<float> &probWeightedMoments, float a, float b, bool isBeta);
    void logLogisticFitting(std::vector<float> probWeightedMoments, double *alpha, double *beta, double *gamma);
    double logLogisticCDF(float myValue, double alpha, double beta, double gamma);
    double logLogisticCDFRobust(float myValue, double alpha, double beta, double gamma);
    bool getGammaParameters(double mean, double variance, double* alpha, double* beta);
    double gammaFunction(double value);
    double gammaNaturalLogarithm(double value);
    void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn);
    void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn);
    double incompleteGamma(double alpha, double x, double *lnGammaValue); // incomplete + complete
    double incompleteGamma(double alpha, double x); // only incomplete
    double inverseGammaCumulativeDistributionFunction(double valueProbability, double alpha, double beta, double accuracy);
    double inverseGeneralizedGammaCDFDoublePrecision(double valueProbability, double alpha, double beta, double accuracy,double pZero,double outlierStep);
    float inverseGeneralizedGammaCDF(float valueProbability, double alpha, double beta, double accuracy,double pZero,double outlierStep);
    bool generalizedGammaFitting(std::vector<float> &series, int n, double* beta, double* alpha,  double* pZero);

    double weibullCDF(double x, double lambda, double kappa);
    double inverseWeibullCDF(double x, double lambda, double kappa);
    double weibullPDF(double x, double lambda, double kappa);
    double meanValueWeibull(double lambda, double kappa);
    double varianceValueWeibull(double lambda, double kappa);
    double functionValueVarianceWeibullDependingOnKappa(double mean, double variance, double kappa);
    void parametersWeibullFromObservations(double mean, double variance, double* lambda, double* kappa, double leftBound, double rightBound);


#endif // GAMMAFUNCTION

