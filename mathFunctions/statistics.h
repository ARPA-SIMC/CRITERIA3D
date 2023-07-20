#ifndef STATISTICS_H
#define STATISTICS_H

    #ifndef _VECTOR_
        #include <vector>
    #endif

    enum meteoComputation { average, stdDev, sum, maxInList, minInList, timeIntegration,
                            differenceWithThreshold, lastDayBelowThreshold,
                            sumAbove, avgAbove, stdDevAbove,
                            percentile, median, freqPositive,
                            daysAbove, daysBelow, consecutiveDaysAbove, consecutiveDaysBelow,
                            prevailingWindDir,
                            trend, mannKendall,
                            phenology,
                            winkler, huglin, fregoni,
                            correctedDegreeDaysSum, erosivityFactorElab, rainIntensityElab, yearMax, yearMin, noMeteoComp};

    enum aggregationMethod {noAggrMethod, aggrAverage, aggrMedian, aggrStdDeviation, aggrMin, aggrMax, aggrSum, aggrPrevailing, aggrIntegral, aggrCenter, aggr95Perc};

    float statisticalElab(meteoComputation elab, float param, std::vector<float> values, int nValues, float rainfallThreshold);

    namespace statistics
    {
        double rootMeanSquareError(double *measured , double *simulated , int nrData);
        float rootMeanSquareError(float *measured , float *simulated , int nrData);
        double rootMeanSquareError(std::vector <float> measured, std::vector <float> simulated);
        float meanError(std::vector<float> measured , std::vector<float> simulated );
        float meanAbsoluteError(std::vector <float> measured, std::vector <float> simulated);
        double compoundRelativeError(std::vector <float> measured, std::vector <float> simulated);
        float coefficientOfVariation(float *measured , float *simulated , int nrData);
        float weighedMean(float *data , float *weights, int nrData);
        float linearInterpolation(float x1, float y1, float x2, float y2, float xx);
        void multiRegression1D(float** x,  float* y, long nrItems,float* q,float* m, int nrPredictors);
        void linearRegression(float* x, float* y, long nrItems, bool zeroIntercept, float* y_intercept, float* mySlope, float* r2);
        void linearRegression( std::vector<float> x,  std::vector<float> y, long nrItems, bool zeroIntercept, float* y_intercept, float* mySlope, float* r2);
        float standardDeviation(float *myList, int nrList);
        float standardDeviation(std::vector<float> myList, int nrList);
        double standardDeviation(std::vector<double> myList, int nrList);
        double standardDeviation(double *myList, int nrList);
        float variance(float *myList, int nrList);
        float variance(std::vector<float> myList, int nrList);
        double variance(std::vector<double> myList, int nrList);
        double variance(double *myList, int nrList);
        float mean(float *myList, int nrList);
        float mean(std::vector<float> myList, int nrList);
        double mean(std::vector<double> myList, int nrList);
        double mean(double *myList, int nrList);
        float covariance(float *myList1, int nrList1,float *myList2, int nrList2);
        double covariance(double *myList1, int nrList1,double *myList2, int nrList2);
        float coefficientPearson(float *myList1, int nrList1,float *myList2, int nrList2);
        float** covariancesMatrix(int nrRowCol, float**myLists,int nrLists);
        void correlationsMatrix(int nrRowCol, double**myLists,int nrLists, double** c);
        float maxList(std::vector<float> values, int nValue);
        float minList(std::vector<float> values, int nValue);
        float sumList(std::vector<float> values, int nValue);
        float sumListThreshold(std::vector<float> values, int nValue, float threshold);
        float diffListThreshold(std::vector<float> values, int nValue, float threshold);
        float countAbove(std::vector<float> values, int nValue, float threshold);
        float countBelow(std::vector<float> values, int nValue, float threshold);
        float countConsecutive(std::vector<float> values, int nValue, float threshold, bool isPositive);
        float frequencyPositive(std::vector<float> values, int nValue);
        float trend(std::vector<float> values, int nValues, float myFirstYear);
        float mannKendall(std::vector<float> values, int nValues);
        bool rollingAverage(double* arrayInput, int sizeArray, int lag,double* arrayOutput);

        double meanNoCheck(double *myList, int nrList);
        double varianceNoCheck(double *myList, int nrList);
        double covarianceNoCheck(double *myList1, int nrList1,double *myList2, int nrList2);
        void correlationsMatrixNoCheck(int nrRowCol, double**myLists,int nrLists, double** c);
    }

#endif // STATISTICS_H
