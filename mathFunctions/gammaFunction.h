#ifndef GAMMAFUNCTION
#define GAMMAFUNCTION

#ifndef _VECTOR_
    #include <vector>
#endif

    /*!
     * code from http://www.mymathlib.com/
     * Copyright © 2004 RLH. All rights reserved.
     */

    #define ITERATIONSMAX 100
    #define EPSTHRESHOLD 3.0e-7
    #define FPMINIMUM 1.0e-30

    double Entire_Incomplete_Gamma_Function(double x, double nu);
    long double xEntire_Incomplete_Gamma_Function(long double x, long double nu);

    double Factorial(int n);
    long double xFactorial(int n);
    int Factorial_Max_Arg( void );

    double Gamma_Function(double x);
    long double xGamma_Function(long double x);
    double Gamma_Function_Max_Arg( void );
    long double xGamma_Function_Max_Arg( void );

    double Incomplete_Gamma_Function(double x, double nu);
    long double xIncomplete_Gamma_Function(long double x, long double nu);

    double Ln_Gamma_Function(double x);
    long double xLn_Gamma_Function(long double x);

    float standardGaussianInvCDF(float prob);
    float gammaCDF(float x, double beta, double gamma,  double pZero) ;
    void probabilityWeightedMoments(std::vector<float> series, int n, std::vector<float> &probWeightedMoments, float a, float b, bool isBeta);
    void logLogisticFitting(std::vector<float> probWeightedMoments, double *alpha, double *beta, double *gamma);
    float logLogisticCDF(float myValue, double alpha, double beta, double gamma);


    double gammaNaturalLogarithm(double value);
    void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn);
    void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn);
    double incompleteGamma(double alpha, double x, double *lnGammaValue); // incomplete + complete
    double incompleteGamma(double alpha, double x); // only incomplete
    bool gammaFitting(std::vector<float> &series, int n, double* beta, double* gamma,  double* pZero);



#endif // GAMMAFUNCTION

