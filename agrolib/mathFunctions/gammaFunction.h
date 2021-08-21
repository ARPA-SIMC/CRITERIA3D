#ifndef GAMMAFUNCTION
#define GAMMAFUNCTION

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

    namespace gammaDistributions
    {
        double gammaNaturalLogarithm(double value);
        void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn);
        void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn);
        double incompleteGamma(double alpha, double x, double *lnGammaValue); // incomplete + complete
        double incompleteGamma(double alpha, double x); // only incomplete
    }


#endif // GAMMAFUNCTION

