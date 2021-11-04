#ifndef MATHEMATICALFUNCTIONS_H
#define MATHEMATICALFUNCTIONS_H

    #ifndef VECTOR_H
        #include <vector>
    #endif
    #ifndef _STRING_
        #include <string>
    #endif

    enum estimatedFunction {FUNCTION_CODE_SPHERICAL, FUNCTION_CODE_LINEAR, FUNCTION_CODE_PARABOLIC,
                       FUNCTION_CODE_EXPONENTIAL, FUNCTION_CODE_LOGARITMIC,
                       FUNCTION_CODE_TWOPARAMETERSPOLYNOMIAL, FUNCTION_CODE_FOURIER_2_HARMONICS,
                       FUNCTION_CODE_FOURIER_GENERAL_HARMONICS,
                       FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN, FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN_RESTRICTED};


    struct TfunctionInput{
        float x ;
        int nrPar ;
        float *par ;
    };

    struct TfunctionInputMonteCarlo2D{
        float x ;
        float y ;
        int nrPar ;
        float *par ;
    };

    struct TfunctionInputMonteCarlo3D{
        float x ;
        float y ;
        float z ;
        int nrPar ;
        float *par ;
    };

    double twoParametersAndExponentialPolynomialFunctions(double x, double* par);
    double twoHarmonicsFourier(double x, double* par);
    double harmonicsFourierGeneral(double x, double* par,int nrPar);
    float errorFunctionPrimitive(float x);
    double errorFunctionPrimitive(double x);
    double parabolicFunction(double x, double* par);
    float gaussianFunction(TfunctionInput fInput);




    namespace integration
    {
        float trapzdParametric(float (*func)(TfunctionInput), int nrPar, float *par , float a , float b , int n);
        float qsimpParametric(float (*func)(TfunctionInput), int nrPar, float *par,float a , float b , float EPS);
        float trapzd(float (*func)(float) , float a , float b , int n);
        float trapezoidalRule(float (*func)(float) , float a , float b , int n);
        double trapezoidalRule(double (*func)(double) , double a , double b , int n);
        float simpsonRule(float (*func)(float),float a , float b , float EPS);
        double simpsonRule(double (*func)(double),double a , double b , double EPS);
        float monteCarlo3D(bool (*func)(TfunctionInputMonteCarlo3D),float den,float xLower, float xUpper, float yLower , float yUpper , float zLower , float zUpper,int nrPar, float *par, float requiredPercentageError ,  float *reachedPercentageErrorW);
        float monteCarlo2D(bool (*func)(TfunctionInputMonteCarlo2D),float den,float xLower, float xUpper, float yLower , float yUpper,int nrPar, float *par, float requiredPercentageError ,  float *reachedPercentageErrorW);
    }

    namespace interpolation
    {
        double linearInterpolation (double x, double *xColumn , double *yColumn, int dimTable);
        float linearInterpolation (float x, float *xColumn , float *yColumn, int dimTable );
        float linearExtrapolation(double x3,double x1,double y1,double x2 , double y2);

        void leastSquares(int idFunction, double* parameters, int nrParameters,
                          double* x, double* y, int nrData, double* lambda,
                          double* parametersDelta, double* parametersChange);

        bool fittingMarquardt(double *myParMin, double *myParMax,
                    double *myPar, int nrMyPar, double *parametersDelta, int myMaxIterations,
                    double myEpsilon, int idFunction, double *x, double *y, int nrData);

        double estimateFunction(int idFunction, double *parameters, int nrParameters, double x);
        double normGeneric(int idFunction, double *parameters, int nrParameters, double *x, double *y,  int nrData);

        double modifiedVanGenuchten(double psi, double *parameters, bool isRestricted);
        double cubicSpline(double x , double *firstColumn , double *secondColumn, int dim); // not working to be checked
        void punctualSecondDerivative(int dim, double *firstColumn , double *secondColumn, double* secondDerivative); // not working to be checked
        void tridiagonalThomasAlgorithm (int n, double *subDiagonal, double *mainDiagonal, double *superDiagonal, double *constantTerm, double* output); // not working to be checked

    }

    namespace matricial
    {
        int matrixSum(double**a , double**b, int rowA , int rowB, int colA, int colB,double **c);
        int matrixDifference(double**a , double**b, int rowA , int rowB, int colA, int colB, double **c);
        int matrixProduct(double **first,double**second,int colFirst, int rowFirst,int colSecond,int rowSecond,double ** multiply);
        int matrixProductNoCheck(double **first,double**second,int rowFirst,int colFirst,int colSecond,double ** multiply);
        void matrixProductSquareMatricesNoCheck(double **first,double**second,int dimension,double ** multiply);
        void  multiplyStrassen(double **c, double **d, int size, double **newMatrix);
        void add(double **a, double **b, int size,double **c);
        void sub(double **a,double **b,int size,double **c);
        void choleskyDecompositionSinglePointer(double *a, int n, double *p);
        void choleskyDecompositionTriangularMatrix(double **a, int n, bool isLowerMatrix);
        void transposedSquareMatrix(double **a, int n);
        void transposedMatrix(double** inputMatrix, int nrRows, int nrColumns,double** outputMatrix);
        void inverse(double** a,double** d,int n);
        void cofactor(double** a,double** d,int n,double determinantOfMatrix);
        double determinant(double** a,int n);
        void minorMatrix(double** b,double** a,int i,int n);
        int eigenSystemMatrix2x2(double** a, double* eigenvalueA, double** eigenvectorA, int n);
        bool inverseGaussJordan(double** a,double** d,int n);
    }

    namespace distribution
    {
        float normalGauss(TfunctionInput fInput);
    }

    namespace myrandom
    {
        //float ran1(long *idum);
        //float gasdev(long *idum);
        double cauchyRandom(double gamma);
        float normalRandom(int *gasDevIset,float *gasDevGset);
        double normalRandom(int *gasDevIset,double *gasDevGset);
        double normalRandomLongSeries(int *gasDevIset,double *gasDevGset,int* randomNumberInitial);
        int getUniformallyDistributedRandomNumber(int* randomNumberInitial);
    }

    namespace statistics
    {
        double ERF(double x, double accuracy);
        double ERFC(double x, double accuracy);
        double inverseERFC(double value, double accuracy);
        double inverseERF(double value, double accuracy);
        double tabulatedERF(double x);
        double tabulatedERFC(double x);
        double inverseTabulatedERF(double value);
        double inverseTabulatedERFC(double value);
        double functionCDFCauchy(double gamma,double x0, double x);
        double functionPDFCauchy(double gamma,double x0, double x);
        double functionQuantileCauchy(double gamma,double x0, double F);
    }


#endif
