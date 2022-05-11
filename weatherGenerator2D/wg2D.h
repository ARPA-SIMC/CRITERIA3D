#ifndef WEATHERGENERATOR2D_H
#define WEATHERGENERATOR2D_H

/*
 * Code translated from the MulGets model:
 * https://it.mathworks.com/matlabcentral/fileexchange/47537-multi-site-stochstic-weather-generator--mulgets-
*/

    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif

    #define TOLERANCE_MULGETS 0.001
    #define MAX_ITERATION_MULGETS 180
    #define ONELESSEPSILON 0.999999
    #define TEMPERATURE_THRESHOLD 60
    #define RAINFALL_THRESHOLD 1000

    enum Tseason {DJF,MAM,JJA,SON};
    enum TaverageTempMethod{ROLLING_AVERAGE,FOURIER_HARMONICS_AVERAGE};

    struct TconsecutiveDays{
        //double** dry;
        //double** wet;
        double dry[12][91];
        double wet[12][91];
    };

    struct TseasonPrec{
        double* DJF;
        double* MAM;
        double* JJA;
        double* SON;
    };

    struct ToccurrenceIndexSeasonal{
        double** meanP;
        double** stdDevP;
        double** meanFit;
        double** stdDevFit;
        double*** parMultiexp;
        double** binCenter;
        double** bin;
    };


    struct TprecOccurrence{
        double p00;
        double p10;

        double p000;
        double p100;
        double p010;
        double p110;

        double p0000;
        double p00000;

        double pDry[60];
        double pWet[60];

        double pDryWeight[60];

        int month;
    };

    struct TcorrelationVar{
        double meanValue1;
        double meanValue2;
        double covariance;
        double variance1, variance2;
    };

    struct TObsPrecDataD{
        Crit3DDate date;
        double prec;
        double amounts;
        double amountsLessThreshold;
        double occurrences;
    };

    struct TcorrelationMatrix{
        double** amount;
        double** occurrence;
        int month;
    };

    struct TcorrelationMatrixTemperature{
        double** maxT;
        double** minT;
        double** meanT;
        double** deltaT;
    };

    struct TseasonalCorrelationMatrix{
        double** amount;
        double** occurrence;
        int beginDoySeason;
        int lengthSeason;
        Tseason season;
    };

    struct TrandomMatrix{
        double** matrixM;
        double** matrixK;
        double** matrixOccurrences;
        int month;
    };

    struct TsimulatedPrecipitationAmounts{
        double** matrixM;
        double** matrixK;
        double** matrixAmounts;
        Tseason season;
    };

    struct TparametersModel{
        int yearOfSimulation;
        int distributionPrecipitation; //Select a distribution to generate daily precipitation amount,1: Multi-exponential or 2: Multi-gamma
        double precipitationThreshold;
    };

    /*struct TfourierParameters{
        double a0;
        double aSin1;
        double aCos1;
        double aSin2;
        double aCos2;
    };*/

    struct Tvariable{
        //TfourierParameters averageFourierParameters;
        //TfourierParameters standardDeviationFourierParameters;
        double* averageEstimation;
        double* stdDevEstimation;
    };

    struct TtemperatureCoefficients{
        double A[2][2];
        double B[2][2];
        double A_mean_delta[2][2];
        double B_mean_delta[2][2];

        Tvariable minTWet;
        Tvariable minTDry;
        Tvariable maxTWet;
        Tvariable maxTDry;

        Tvariable meanTWet;
        Tvariable meanTDry;
        Tvariable deltaTWet;
        Tvariable deltaTDry;

    };

    struct TdailyResidual{
        double maxTDry;
        double minTDry;
        double maxTWet;
        double minTWet;
        double maxT;
        double minT;

        double meanTDry;
        double deltaTDry;
        double meanTWet;
        double deltaTWet;
        double meanT;
        double deltaT;
    };

    struct TmultiOccurrenceTemperature
    {
        int year_simulated;
        int month_simulated;
        int day_simulated;
        double* occurrence_simulated;
    };

    struct ToutputWeatherData
    {
        int* yearSimulated;
        int* monthSimulated;
        int* daySimulated;
        int* doySimulated;
        double* maxT;
        double* minT;
        double* precipitation;
        double* maxTClimate;
        double* minTClimate;

    };


    //void randomSet(double *arrayNormal,int dimArray);
    class weatherGenerator2D
    {
    private:

        bool isPrecWG2D,isTempWG2D;
        int nrData;
        int nrDataWithout29February;
        int nrStations;
        TparametersModel parametersModel;
        int *month,*lengthMonth;
        int lengthSeason[4];
        int numberObservedDJF,numberObservedMAM,numberObservedJJA,numberObservedSON;
        int numberObservedMax;
        bool computeStatistics;
        int consecutiveDayTransition;
        TconsecutiveDays* observedConsecutiveDays;
        TconsecutiveDays* simulatedConsecutiveDays;
        TaverageTempMethod averageTempMethod;
        TObsDataD** obsDataD;
        TObsPrecDataD** obsPrecDataD;
        TprecOccurrence** precOccurence;
        TcorrelationMatrix *correlationMatrix;
        TprecOccurrence precOccurrenceGlobal[12];

        TrandomMatrix *randomMatrix;
        ToccurrenceIndexSeasonal* occurrenceIndexSeasonal;
        TsimulatedPrecipitationAmounts *simulatedPrecipitationAmounts;
        //double** tagDoyPrec;

        // create the seasonal correlation matrices
        double** occurrenceMatrixSeasonDJF;
        double** occurrenceMatrixSeasonMAM;
        double** occurrenceMatrixSeasonJJA;
        double** occurrenceMatrixSeasonSON;

        double** wDJF;
        double** wMAM;
        double** wJJA;
        double** wSON;
        double** wSeason;
        // new distribution for precipitation
        Tvariable* precipitationAmount;
        TseasonPrec* seasonPrec;
        double** precGenerated;

        void initializeTemperatureVariables();
        bool isTemperatureRecordOK(double value);
        bool isPrecipitationRecordOK(double value);
        void initializePrecipitationAmountParameters();
        void computeprecipitationAmountParameters();
        void getSeasonalMeanPrecipitation(int iStation, int iSeason, double* meanPrec);
        void spatialIterationAmountsMonthly(int iMonth, double** correlationMatrixSimulatedData,double ** amountsCorrelationMatrix , double** randomMatrix, int lengthSeries, double** occurrences, double** simulatedPrecipitationAmountsSeasonal);
        void precipitationCorrelationMatricesSimulation();
        void precipitationMonthlyAverage(float** averageSimulation, float** averageClimate);

        // variables only for temperatures
        TtemperatureCoefficients* temperatureCoefficients;
        TtemperatureCoefficients* temperatureCoefficientsFourier;
        TdailyResidual* dailyResidual;
        TcorrelationMatrixTemperature correlationMatrixTemperature;
        double** normRandomMaxT;
        double** normRandomMinT;
        double** normRandomMeanT;
        double** normRandomDeltaT;

        TmultiOccurrenceTemperature* multiOccurrenceTemperature;
        double** maxTGenerated;
        double** minTGenerated;
        double** occurrencePrecGenerated;
        double** amountsPrecGenerated;

        // new variables

        double** meanTGenerated;
        double** deltaTGenerated;


        double* normalRandomNumbers;


        float** monthlyAverageTmax;
        float** monthlyAverageTmin;
        float** monthlyAverageTmean;
        float** monthlyAveragePrec;
        float** monthlyStdDevTmax;
        float** monthlyStdDevTmin;
        float** monthlyStdDevTmean;
        float** monthlyStdDevPrec;
        float** interpolatedDailyValuePrecAverage;
        float** interpolatedDailyValuePrecVariance;
        double** weibullDailyParameterLambda;
        double** weibullDailyParameterKappa;

        //float** monthlyRandomDeviationTmean;
        //functions
        void commonModuleCompute();
        void precipitationCompute();
        void precipitation29February(int idStation);
        void precipitationAmountsOccurences(int idStation, double* precipitationAmountsD,bool* precipitationOccurencesD);
        void precipitationP00P10();
        void precipitationPDryUntilNSteps();
        int recursiveAccountDryDays(int idStation, int i, int iMonth,int step, std::vector<std::vector<int> > &consecutiveDays,int nrFollowingSteps);
        int recursiveAccountWetDays(int idStation, int i, int iMonth,int step, std::vector<std::vector<int> > &consecutiveDays,int nrFollowingSteps);
        void precipitationCorrelationMatrices();
        void precipitationMultisiteOccurrenceGeneration();
        void spatialIterationOccurrence(double ** M, double **K, double **occurrences, double** matrixOccurrence, double** normalizedMatrixRandom, double **transitionNormal, double ***transitionNormalAugmentedMemory, int lengthSeries);
        void precipitationMultiDistributionParameterization();
        void precipitationMultisiteAmountsGeneration();
        void initializeBaseWeatherVariables();
        void initializeOccurrenceIndex();
        void initializePrecipitationOutputs(int lengthSeason[]);
        void initializePrecipitationInternalArrays();
        void spatialIterationAmounts(double** correlationMatrixSimulatedData,double ** amountsCorrelationMatrix , double** randomMatrix, int length, double** occurrences, double** phatAlpha, double** phatBeta,double** simulatedPrecipitationAmounts);
        void temperatureCompute();
        void computeMonthlyVariables();
        void computeTemperatureParameters();
        void initializeTemperatureParameters();
        int  doyFromDate(int day,int month,int year);
        int  dateFromDoy(int doy, int year, int *day, int *month);
        void harmonicsFourier(double* variable, double *par, int nrPar, double* estimatedVariable, int nrEstimatedVariable);
        void computeResiduals(double* averageTMaxDry,double* averageTMaxWet,double* stdDevTMaxDry,double* stdDevTMaxWet,double* averageTMinDry,double* averageTMinWet,double* stdDevTMinDry,double* stdDevTMinWet,int idStation);
        void computeResidualsMeanDelta(double* averageTMeanDry, double* averageTMeanWet, double* stdDevTMeanDry, double* stdDevTMeanWet, double* averageTDeltaDry, double* averageTDeltaWet, double* stdDevTDeltaDry, double* stdDevTDeltaWet, int idStation);
        void temperaturesCorrelationMatrices();
        void covarianceOfResiduals(double** covarianceMatrix, int lag);
        void covarianceOfResidualsMeanDelta(double** covarianceMatrix, int lag);
        void initializeTemperaturecorrelationMatrices();
        void multisiteRandomNumbersTemperature();
        void initializeNormalRandomMatricesTemperatures();
        void multisiteTemperatureGeneration();
        void multisiteTemperatureGenerationMeanDelta();
        void initializeMultiOccurrenceTemperature(int length);
        void initializeTemperaturesOutput(int length);
        void createAmountOutputSerie();
        void prepareWeatherGeneratorOutput();

        //void initializeTemperatureParametersMeanDelta();

        void initializeOutputData(int* nrDays);
        void randomSet(double *arrayNormal,int dimArray);

    public:
        // variables
        ToutputWeatherData *outputWeatherData;
        //functions
        weatherGenerator2D() {}
        bool initializeData(int lengthDataSeries, int nrStations);
        void initializeParameters(float thresholdPrecipitation, int simulatedYears, int distributionType, bool computePrecWG2D, bool computeTempWG2D, bool computeStatistics, TaverageTempMethod tempMethod);
        void setObservedData(TObsDataD** observations);
        void computeWeatherGenerator2D();
        void pressEnterToContinue();
        void initializeRandomNumbers(double* vector);
        ToutputWeatherData* getWeatherGeneratorOutput(int startingYear);
    };

#endif // WEATHERGENERATOR2D_H
