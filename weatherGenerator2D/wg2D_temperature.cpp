#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "wg2D.h"
#include "commonConstants.h"
#include "furtherMathFunctions.h"
#include "eispack.h"
#include "gammaFunction.h"
#include "crit3dDate.h"

void weatherGenerator2D::initializeTemperatureParameters()
{
    // initialize temp parameters
    temperatureCoefficients = (TtemperatureCoefficients *)calloc(nrStations, sizeof(TtemperatureCoefficients));
    temperatureCoefficientsFourier =  (TtemperatureCoefficients *)calloc(nrStations, sizeof(TtemperatureCoefficients));
    for (int i = 0; i < nrStations; i++)
    {
        temperatureCoefficients[i].maxTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        temperatureCoefficientsFourier[i].maxTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].maxTDry.averageEstimation[j] = NODATA;
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].maxTDry.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].maxTDry.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].maxTDry.stdDevEstimation[j] = NODATA;
        /*
        temperatureCoefficients[i].maxTDry.averageFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].maxTDry.averageFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].maxTDry.averageFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].maxTDry.averageFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].maxTDry.averageFourierParameters.aSin2 = NODATA;
        temperatureCoefficients[i].maxTDry.standardDeviationFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].maxTDry.standardDeviationFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].maxTDry.standardDeviationFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].maxTDry.standardDeviationFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].maxTDry.standardDeviationFourierParameters.aSin2 = NODATA;
        */
        temperatureCoefficients[i].minTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        temperatureCoefficientsFourier[i].minTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].minTDry.averageEstimation[j] = NODATA;
        for (int j=0; j<365; j++) temperatureCoefficients[i].minTDry.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].minTDry.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].minTDry.stdDevEstimation[j] = NODATA;
        /*
        temperatureCoefficients[i].minTDry.averageFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].minTDry.averageFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].minTDry.averageFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].minTDry.averageFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].minTDry.averageFourierParameters.aSin2 = NODATA;
        temperatureCoefficients[i].minTDry.standardDeviationFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].minTDry.standardDeviationFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].minTDry.standardDeviationFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].minTDry.standardDeviationFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].minTDry.standardDeviationFourierParameters.aSin2 = NODATA;
        */
        temperatureCoefficients[i].maxTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].maxTWet.averageEstimation[j] = NODATA;
        temperatureCoefficientsFourier[i].maxTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].maxTWet.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].maxTWet.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].maxTWet.stdDevEstimation[j] = NODATA;
        /*
        temperatureCoefficients[i].maxTWet.averageFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].maxTWet.averageFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].maxTWet.averageFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].maxTWet.averageFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].maxTWet.averageFourierParameters.aSin2 = NODATA;
        temperatureCoefficients[i].maxTWet.standardDeviationFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].maxTWet.standardDeviationFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].maxTWet.standardDeviationFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].maxTWet.standardDeviationFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].maxTWet.standardDeviationFourierParameters.aSin2 = NODATA;
        */
        temperatureCoefficients[i].minTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].minTWet.averageEstimation[j] = NODATA;
        temperatureCoefficientsFourier[i].minTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].minTWet.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].minTWet.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].minTWet.stdDevEstimation[j] = NODATA;
        /*
        temperatureCoefficients[i].minTWet.averageFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].minTWet.averageFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].minTWet.averageFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].minTWet.averageFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].minTWet.averageFourierParameters.aSin2 = NODATA;
        temperatureCoefficients[i].minTWet.standardDeviationFourierParameters.a0 = NODATA;
        temperatureCoefficients[i].minTWet.standardDeviationFourierParameters.aCos1 = NODATA;
        temperatureCoefficients[i].minTWet.standardDeviationFourierParameters.aSin1 = NODATA;
        temperatureCoefficients[i].minTWet.standardDeviationFourierParameters.aCos2 = NODATA;
        temperatureCoefficients[i].minTWet.standardDeviationFourierParameters.aSin2 = NODATA;
        */
        for (int k=0; k<2; k++)
        {
            for (int j=0; j<2; j++)
            {
                temperatureCoefficients[i].A[k][j]= NODATA;
                temperatureCoefficients[i].B[k][j]= NODATA;
            }
        }
     }
    dailyResidual = (TdailyResidual*) calloc(nrData,sizeof(TdailyResidual));
    for (int i=0; i<nrData; i++)
    {
        dailyResidual[i].maxTDry = 0;
        dailyResidual[i].minTDry = 0;
        dailyResidual[i].maxTWet = 0;
        dailyResidual[i].minTWet = 0;
    }

    for (int i = 0; i < nrStations; i++)
    {
        temperatureCoefficients[i].meanTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        temperatureCoefficientsFourier[i].meanTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].meanTDry.averageEstimation[j] = NODATA;
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].meanTDry.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].meanTDry.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].meanTDry.stdDevEstimation[j] = NODATA;

        temperatureCoefficients[i].deltaTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        temperatureCoefficientsFourier[i].deltaTDry.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].deltaTDry.averageEstimation[j] = NODATA;
        for (int j=0; j<365; j++) temperatureCoefficients[i].deltaTDry.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].deltaTDry.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].deltaTDry.stdDevEstimation[j] = NODATA;

        temperatureCoefficients[i].meanTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].meanTWet.averageEstimation[j] = NODATA;
        temperatureCoefficientsFourier[i].meanTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].meanTWet.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].meanTWet.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].meanTWet.stdDevEstimation[j] = NODATA;

        temperatureCoefficients[i].deltaTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].deltaTWet.averageEstimation[j] = NODATA;
        temperatureCoefficientsFourier[i].deltaTWet.averageEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficientsFourier[i].deltaTWet.averageEstimation[j] = NODATA;
        temperatureCoefficients[i].deltaTWet.stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++) temperatureCoefficients[i].deltaTWet.stdDevEstimation[j] = NODATA;

        for (int k=0; k<2; k++)
        {
            for (int j=0; j<2; j++)
            {
                temperatureCoefficients[i].A_mean_delta[k][j]= NODATA;
                temperatureCoefficients[i].B_mean_delta[k][j]= NODATA;
            }
        }
     }

}

void weatherGenerator2D::initializeTemperaturecorrelationMatrices()
{
    correlationMatrixTemperature.maxT = (double **)calloc(nrStations, sizeof(double *));
    correlationMatrixTemperature.minT = (double **)calloc(nrStations, sizeof(double *));
    for (int i=0;i<nrStations;i++)
    {
        correlationMatrixTemperature.maxT[i] = (double *)calloc(nrStations, sizeof(double));
        correlationMatrixTemperature.minT[i] = (double *)calloc(nrStations, sizeof(double));
        for (int j=0;j<nrStations;j++)
        {
            correlationMatrixTemperature.maxT[i][j] = NODATA;
            correlationMatrixTemperature.minT[i][j] = NODATA;
        }
    }

    correlationMatrixTemperature.meanT = (double **)calloc(nrStations, sizeof(double *));
    correlationMatrixTemperature.deltaT = (double **)calloc(nrStations, sizeof(double *));
    for (int i=0;i<nrStations;i++)
    {
        correlationMatrixTemperature.meanT[i] = (double *)calloc(nrStations, sizeof(double));
        correlationMatrixTemperature.deltaT[i] = (double *)calloc(nrStations, sizeof(double));
        for (int j=0;j<nrStations;j++)
        {
            correlationMatrixTemperature.meanT[i][j] = NODATA;
            correlationMatrixTemperature.deltaT[i][j] = NODATA;
        }
    }


}

void weatherGenerator2D::initializeTemperatureVariables()
{
    weatherGenerator2D::initializeMultiOccurrenceTemperature(365*parametersModel.yearOfSimulation);
    weatherGenerator2D::initializeTemperaturesOutput(365*parametersModel.yearOfSimulation);
    weatherGenerator2D::initializeTemperatureParameters();

}




void weatherGenerator2D::computeTemperatureParameters()
{
    //weatherGenerator2D::computeMonthlyVariables();
    for (int iStation=0; iStation<nrStations; iStation++)
    {
        double averageTMaxDry[365]={0};
        double averageTMaxWet[365]={0};
        double stdDevTMaxDry[365]={0};
        double stdDevTMaxWet[365]={0};
        double averageTMinDry[365]={0};
        double averageTMinWet[365]={0};
        double stdDevTMinDry[365]={0};
        double stdDevTMinWet[365]={0};
        int countTMaxDry[365]={0};
        int countTMaxWet[365]={0};
        int countTMinDry[365]={0};
        int countTMinWet[365]={0};


        double averageTMeanDry[365]={0};
        double averageTMeanWet[365]={0};
        double stdDevTMeanDry[365]={0};
        double stdDevTMeanWet[365]={0};
        double averageTDeltaDry[365]={0};
        double averageTDeltaWet[365]={0};
        double stdDevTDeltaDry[365]={0};
        double stdDevTDeltaWet[365]={0};
        int countTMeanDry[365]={0};
        int countTMeanWet[365]={0};
        int countTDeltaDry[365]={0};
        int countTDeltaWet[365]={0};


        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            if(fabs(obsDataD[iStation][iDatum].tMax) > 60) obsDataD[iStation][iDatum].tMax = NODATA;
            if(fabs(obsDataD[iStation][iDatum].tMin) > 60) obsDataD[iStation][iDatum].tMin = NODATA;
        }
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
           if ((fabs((obsDataD[iStation][iDatum].tMax)))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMax += 3.*EPSILON;

            }
            if ((fabs(obsDataD[iStation][iDatum].tMin))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMin += 3.*EPSILON;
            }
        }
        // compute average temperatures of the stations
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            int dayOfYear;
            dayOfYear = weatherGenerator2D::doyFromDate(obsDataD[iStation][iDatum].date.day,obsDataD[iStation][iDatum].date.month,obsDataD[iStation][iDatum].date.year);
            if ((isLeapYear(obsDataD[iStation][iDatum].date.year)) && (obsDataD[iStation][iDatum].date.month > 2)) dayOfYear--;
            dayOfYear--;
            if (obsDataD[iStation][iDatum].date.month == 2 && obsDataD[iStation][iDatum].date.day == 29)
            {
                dayOfYear-- ;
            }


            if (fabs(obsDataD[iStation][iDatum].prec - NODATA) > EPSILON)
            {
                if (obsDataD[iStation][iDatum].prec > parametersModel.precipitationThreshold)
                {
                    if (isTemperatureRecordOK(obsDataD[iStation][iDatum].tMax))
                    {
                        ++countTMaxWet[dayOfYear];
                        averageTMaxWet[dayOfYear] += obsDataD[iStation][iDatum].tMax;
                    }
                    if (isTemperatureRecordOK(obsDataD[iStation][iDatum].tMin))
                    {
                        ++countTMinWet[dayOfYear];
                        averageTMinWet[dayOfYear] += obsDataD[iStation][iDatum].tMin;
                    }
                    if (isTemperatureRecordOK(obsDataD[iStation][iDatum].tMax) && isTemperatureRecordOK(obsDataD[iStation][iDatum].tMin))
                    {
                        ++countTMeanWet[dayOfYear];
                        averageTMeanWet[dayOfYear] += 0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin);
                        ++countTDeltaWet[dayOfYear];
                        averageTDeltaWet[dayOfYear] += (obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin);
                    }
                }
                else if (obsDataD[iStation][iDatum].prec <= parametersModel.precipitationThreshold)
                {
                    if ((fabs((obsDataD[iStation][iDatum].tMax)- NODATA))> EPSILON)
                    {
                        ++countTMaxDry[dayOfYear];
                        averageTMaxDry[dayOfYear] += obsDataD[iStation][iDatum].tMax;
                    }
                    if ((fabs((obsDataD[iStation][iDatum].tMin)- NODATA))> EPSILON)
                    {
                        ++countTMinDry[dayOfYear];
                        averageTMinDry[dayOfYear] += obsDataD[iStation][iDatum].tMin;
                    }
                    if (isTemperatureRecordOK(obsDataD[iStation][iDatum].tMax) && isTemperatureRecordOK(obsDataD[iStation][iDatum].tMin))
                    {
                        ++countTMeanDry[dayOfYear];
                        averageTMeanDry[dayOfYear] += 0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin);
                        ++countTDeltaDry[dayOfYear];
                        averageTDeltaDry[dayOfYear] += (obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin);
                    }

                }
            }

        }
        for (int iDay=0; iDay<365; iDay++)
        {
            if (countTMaxDry[iDay] != 0) averageTMaxDry[iDay] /= countTMaxDry[iDay];
            else averageTMaxDry[iDay] = NODATA;
            if (countTMaxWet[iDay] != 0) averageTMaxWet[iDay] /= countTMaxWet[iDay];
            else averageTMaxWet[iDay] = NODATA;
            if (countTMinDry[iDay] != 0) averageTMinDry[iDay] /= countTMinDry[iDay];
            else averageTMinDry[iDay] = NODATA;
            if (countTMinWet[iDay] != 0) averageTMinWet[iDay] /= countTMinWet[iDay];
            else averageTMinWet[iDay] = NODATA;

            if (countTMeanDry[iDay] != 0) averageTMeanDry[iDay] /= countTMeanDry[iDay];
            else averageTMeanDry[iDay] = NODATA;
            if (countTMeanWet[iDay] != 0) averageTMeanWet[iDay] /= countTMeanWet[iDay];
            else averageTMeanWet[iDay] = NODATA;
            if (countTDeltaDry[iDay] != 0) averageTDeltaDry[iDay] /= countTDeltaDry[iDay];
            else averageTMeanDry[iDay] = NODATA;
            if (countTDeltaWet[iDay] != 0) averageTDeltaWet[iDay] /= countTDeltaWet[iDay];
            else averageTDeltaWet[iDay] = NODATA;
        }


        double* rollingAverageTMinDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMinWet = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMaxDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMaxWet = (double*)calloc(385,sizeof(double));

        double* rollingAverageTMeanDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMeanWet = (double*)calloc(385,sizeof(double));
        double* rollingAverageTDeltaDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTDeltaWet = (double*)calloc(385,sizeof(double));


        for (int i=0;i<385;i++)
        {
            rollingAverageTMaxDry[i] = NODATA;
            rollingAverageTMinDry[i] = NODATA;
            rollingAverageTMaxWet[i] = NODATA;
            rollingAverageTMinWet[i] = NODATA;
            rollingAverageTMeanDry[i] = NODATA;
            rollingAverageTDeltaDry[i] = NODATA;
            rollingAverageTMeanWet[i] = NODATA;
            rollingAverageTDeltaWet[i] = NODATA;

        }
        double inputT[385];
        int lag = 10;
        // t min dry
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMinDry[355+i];
            inputT[384-i] = averageTMinDry[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMinDry[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMinDry);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].minTDry.averageEstimation[i] = rollingAverageTMinDry[i+10];
        }
        // t max dry
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMaxDry[355+i];
            inputT[384-i] = averageTMaxDry[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMaxDry[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMaxDry);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].maxTDry.averageEstimation[i] = rollingAverageTMaxDry[i+10];
        }
        // t min wet
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMinWet[355+i];
            inputT[384-i] = averageTMinWet[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMinWet[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMinWet);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].minTWet.averageEstimation[i] = rollingAverageTMinWet[i+10];
        }
        // t max wet
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMaxWet[355+i];
            inputT[384-i] = averageTMaxWet[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMaxWet[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMaxWet);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].maxTWet.averageEstimation[i] = rollingAverageTMaxWet[i+10];
        }


        // t delta dry
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTDeltaDry[355+i];
            inputT[384-i] = averageTDeltaDry[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTDeltaDry[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTDeltaDry);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].deltaTDry.averageEstimation[i] = rollingAverageTDeltaDry[i+10];
        }
        // t mean dry
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMeanDry[355+i];
            inputT[384-i] = averageTMeanDry[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMeanDry[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMeanDry);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].meanTDry.averageEstimation[i] = rollingAverageTMeanDry[i+10];
        }
        // t delta wet
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTDeltaWet[355+i];
            inputT[384-i] = averageTDeltaWet[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTDeltaWet[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTDeltaWet);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].deltaTWet.averageEstimation[i] = rollingAverageTDeltaWet[i+10];
        }
        // t mean wet
        for (int i=0;i<10;i++)
        {
            inputT[i] = averageTMeanWet[355+i];
            inputT[384-i] = averageTMeanWet[9-i];
        }
        for (int i=0;i<365;i++)
        {
            inputT[i+10]= averageTMeanWet[i];
        }
        statistics::rollingAverage(inputT,385,lag,rollingAverageTMeanWet);
        for (int i=0;i<365;i++)
        {
            temperatureCoefficients[iStation].meanTWet.averageEstimation[i] = rollingAverageTMeanWet[i+10];
        }

        free(rollingAverageTMinDry);
        free(rollingAverageTMinWet);
        free(rollingAverageTMaxDry);
        free(rollingAverageTMaxWet);

        free(rollingAverageTMeanDry);
        free(rollingAverageTMeanWet);
        free(rollingAverageTDeltaDry);
        free(rollingAverageTDeltaWet);

        // compute standard deviation temperatures of the stations
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            int dayOfYear;
            dayOfYear = weatherGenerator2D::doyFromDate(obsDataD[iStation][iDatum].date.day,obsDataD[iStation][iDatum].date.month,obsDataD[iStation][iDatum].date.year);
            if ((isLeapYear(obsDataD[iStation][iDatum].date.year)) && (obsDataD[iStation][iDatum].date.month > 2))
            {
                dayOfYear--; // to avoid problems in leap years
            }
            dayOfYear--; // to change from 1-365 to 0-364
            if (obsDataD[iStation][iDatum].date.month == 2 && obsDataD[iStation][iDatum].date.day == 29)
            {
                dayOfYear-- ; // to avoid problems in leap years
            }

            if (fabs(obsDataD[iStation][iDatum].prec - NODATA) > EPSILON)
            {
                if (obsDataD[iStation][iDatum].prec > parametersModel.precipitationThreshold)
                {
                    if ((fabs((obsDataD[iStation][iDatum].tMax) - NODATA))> EPSILON)
                    {
                        stdDevTMaxWet[dayOfYear] += (obsDataD[iStation][iDatum].tMax - averageTMaxWet[dayOfYear])*(obsDataD[iStation][iDatum].tMax - averageTMaxWet[dayOfYear]);
                    }
                    if ((fabs(obsDataD[iStation][iDatum].tMin - NODATA))> EPSILON)
                    {
                        stdDevTMinWet[dayOfYear] += (obsDataD[iStation][iDatum].tMin - averageTMinWet[dayOfYear])*(obsDataD[iStation][iDatum].tMin - averageTMinWet[dayOfYear]);
                    }
                    if ((fabs(obsDataD[iStation][iDatum].tMin - NODATA))> EPSILON && (fabs((obsDataD[iStation][iDatum].tMax) - NODATA))> EPSILON)
                    {
                        stdDevTMeanWet[dayOfYear] += (0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin) - averageTMeanWet[dayOfYear])*(0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin) - averageTMeanWet[dayOfYear]);
                        stdDevTDeltaWet[dayOfYear] += ((obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin) - averageTDeltaWet[dayOfYear])*((obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin) - averageTDeltaWet[dayOfYear]);
                    }


                }
                else if (obsDataD[iStation][iDatum].prec <= parametersModel.precipitationThreshold && obsDataD[iStation][iDatum].prec >= 0)
                {
                    if ((fabs((obsDataD[iStation][iDatum].tMax) - NODATA))> EPSILON)
                    {
                        stdDevTMaxDry[dayOfYear] += (obsDataD[iStation][iDatum].tMax - averageTMaxDry[dayOfYear])*(obsDataD[iStation][iDatum].tMax - averageTMaxDry[dayOfYear]);
                    }
                    if ((fabs((obsDataD[iStation][iDatum].tMin) - NODATA))> EPSILON)
                    {
                        stdDevTMinDry[dayOfYear] += (obsDataD[iStation][iDatum].tMin - averageTMinDry[dayOfYear])*(obsDataD[iStation][iDatum].tMin - averageTMinDry[dayOfYear]);
                    }
                    if ((fabs(obsDataD[iStation][iDatum].tMin - NODATA))> EPSILON && (fabs((obsDataD[iStation][iDatum].tMax) - NODATA))> EPSILON)
                    {
                        stdDevTMeanDry[dayOfYear] += (0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin) - averageTMeanDry[dayOfYear])*(0.5*(obsDataD[iStation][iDatum].tMax + obsDataD[iStation][iDatum].tMin) - averageTMeanDry[dayOfYear]);
                        stdDevTDeltaDry[dayOfYear] += ((obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin) - averageTDeltaDry[dayOfYear])*((obsDataD[iStation][iDatum].tMax - obsDataD[iStation][iDatum].tMin) - averageTDeltaDry[dayOfYear]);
                    }

                }
            }

        }
        for (int iDay=0; iDay<365; iDay++)
        {
            if (countTMaxDry[iDay] != 0) stdDevTMaxDry[iDay] /= countTMaxDry[iDay];
            else stdDevTMaxDry[iDay] = NODATA;
            if (countTMaxWet[iDay] != 0) stdDevTMaxWet[iDay] /= countTMaxWet[iDay];
            else stdDevTMaxWet[iDay] = NODATA;
            if (countTMinDry[iDay] != 0) stdDevTMinDry[iDay] /= countTMinDry[iDay];
            else stdDevTMinDry[iDay] = NODATA;
            if (countTMinWet[iDay] != 0) stdDevTMinWet[iDay] /= countTMinWet[iDay];
            else stdDevTMinWet[iDay] = NODATA;

            if (countTMeanDry[iDay] != 0) stdDevTMeanDry[iDay] /= countTMeanDry[iDay];
            else stdDevTMeanDry[iDay] = NODATA;
            if (countTMeanWet[iDay] != 0) stdDevTMeanWet[iDay] /= countTMeanWet[iDay];
            else stdDevTMeanWet[iDay] = NODATA;
            if (countTDeltaDry[iDay] != 0) stdDevTDeltaDry[iDay] /= countTDeltaDry[iDay];
            else stdDevTDeltaDry[iDay] = NODATA;
            if (countTDeltaWet[iDay] != 0) stdDevTDeltaWet[iDay] /= countTDeltaWet[iDay];
            else stdDevTDeltaWet[iDay] = NODATA;

            if (countTMaxDry[iDay] != 0) stdDevTMaxDry[iDay] = sqrt(stdDevTMaxDry[iDay]);
            if (countTMaxWet[iDay] != 0) stdDevTMaxWet[iDay] = sqrt(stdDevTMaxWet[iDay]);
            if (countTMinDry[iDay] != 0) stdDevTMinDry[iDay] = sqrt(stdDevTMinDry[iDay]);
            if (countTMinWet[iDay] != 0) stdDevTMinWet[iDay] = sqrt(stdDevTMinWet[iDay]);

            if (countTMeanDry[iDay] != 0) stdDevTMeanDry[iDay] = sqrt(stdDevTMeanDry[iDay]);
            if (countTMeanWet[iDay] != 0) stdDevTMeanWet[iDay] = sqrt(stdDevTMeanWet[iDay]);
            if (countTDeltaDry[iDay] != 0) stdDevTDeltaDry[iDay] = sqrt(stdDevTDeltaDry[iDay]);
            if (countTDeltaWet[iDay] != 0) stdDevTDeltaWet[iDay] = sqrt(stdDevTDeltaWet[iDay]);

        }

        // compute the Fourier coefficients

        double *par;
        int nrPar = 11;
        par = (double *) calloc(nrPar, sizeof(double));
        if (averageTempMethod == FOURIER_HARMONICS_AVERAGE)
        {
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMaxDry,par,nrPar,temperatureCoefficients[iStation].maxTDry.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMinDry,par,nrPar,temperatureCoefficients[iStation].minTDry.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMaxWet,par,nrPar,temperatureCoefficients[iStation].maxTWet.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMinWet,par,nrPar,temperatureCoefficients[iStation].minTWet.averageEstimation,365);


            // /////////////////////////////////////////////////////////////////
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMeanDry,par,nrPar,temperatureCoefficients[iStation].meanTDry.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTDeltaDry,par,nrPar,temperatureCoefficients[iStation].deltaTDry.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTMeanWet,par,nrPar,temperatureCoefficients[iStation].meanTWet.averageEstimation,365);
            for (int i=0;i<nrPar;i++)
            {
                par[i] = NODATA;
            }
            weatherGenerator2D::harmonicsFourier(averageTDeltaWet,par,nrPar,temperatureCoefficients[iStation].deltaTWet.averageEstimation,365);

        }

        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMaxDry,par,nrPar,temperatureCoefficients[iStation].maxTDry.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMinDry,par,nrPar,temperatureCoefficients[iStation].minTDry.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMaxWet,par,nrPar,temperatureCoefficients[iStation].maxTWet.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMinWet,par,nrPar,temperatureCoefficients[iStation].minTWet.stdDevEstimation,365);

        // //////////////////////////////////////////////////////////////////////////////

        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMeanDry,par,nrPar,temperatureCoefficients[iStation].meanTDry.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTDeltaDry,par,nrPar,temperatureCoefficients[iStation].deltaTDry.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTMeanWet,par,nrPar,temperatureCoefficients[iStation].meanTWet.stdDevEstimation,365);
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(stdDevTDeltaWet,par,nrPar,temperatureCoefficients[iStation].deltaTWet.stdDevEstimation,365);

        // free memory of parameters, variable par[]
        free(par);
        /*
        for (int iDoy=0;iDoy<365;iDoy++)
        {
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].maxTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].meanTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].deltaTWet.averageEstimation[iDoy]);
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].maxTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].meanTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].deltaTWet.stdDevEstimation[iDoy]);
            //getchar();
        }

        for (int iDoy=0;iDoy<365;iDoy++)
        {
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].maxTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].meanTWet.averageEstimation[iDoy],temperatureCoefficients[iStation].deltaTWet.averageEstimation[iDoy]);
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].maxTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].meanTWet.stdDevEstimation[iDoy],temperatureCoefficients[iStation].deltaTWet.stdDevEstimation[iDoy]);
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTDry.averageEstimation[iDoy],temperatureCoefficients[iStation].maxTDry.averageEstimation[iDoy],temperatureCoefficients[iStation].meanTDry.averageEstimation[iDoy],temperatureCoefficients[iStation].deltaTDry.averageEstimation[iDoy]);
            printf("%.2f,%.2f,%.2f,%.2f\n",temperatureCoefficients[iStation].minTDry.stdDevEstimation[iDoy],temperatureCoefficients[iStation].maxTDry.stdDevEstimation[iDoy],temperatureCoefficients[iStation].meanTDry.stdDevEstimation[iDoy],temperatureCoefficients[iStation].deltaTDry.stdDevEstimation[iDoy]);
            getchar();
        }
        */
        weatherGenerator2D::computeResiduals(temperatureCoefficients[iStation].maxTDry.averageEstimation,
                                             temperatureCoefficients[iStation].maxTWet.averageEstimation,
                                             temperatureCoefficients[iStation].maxTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].maxTWet.stdDevEstimation,
                                             temperatureCoefficients[iStation].minTDry.averageEstimation,
                                             temperatureCoefficients[iStation].minTWet.averageEstimation,
                                             temperatureCoefficients[iStation].minTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].minTWet.stdDevEstimation,iStation);

        weatherGenerator2D::computeResidualsMeanDelta(temperatureCoefficients[iStation].meanTDry.averageEstimation,
                                             temperatureCoefficients[iStation].meanTWet.averageEstimation,
                                             temperatureCoefficients[iStation].meanTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].meanTWet.stdDevEstimation,
                                             temperatureCoefficients[iStation].deltaTDry.averageEstimation,
                                             temperatureCoefficients[iStation].deltaTWet.averageEstimation,
                                             temperatureCoefficients[iStation].deltaTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].deltaTWet.stdDevEstimation,iStation);

        int matrixRang = 2;
        double** matrixCovarianceLag0 = (double **) calloc(matrixRang, sizeof(double*));
        double** matrixCovarianceLag1 = (double **) calloc(matrixRang, sizeof(double*));
        double** matrixA = (double **) calloc(matrixRang, sizeof(double*));
        double** matrixC = (double **) calloc(matrixRang, sizeof(double*));
        double** matrixB = (double **) calloc(matrixRang, sizeof(double*));
        double** matrixDummy = (double **) calloc(matrixRang, sizeof(double*));
        double** eigenvectors = (double **) calloc(matrixRang, sizeof(double*));
        double* eigenvalues = (double *) calloc(matrixRang, sizeof(double));

        for (int i=0;i<matrixRang;i++)
        {
            matrixCovarianceLag0[i] = (double *) calloc(matrixRang, sizeof(double));
            matrixCovarianceLag1[i] = (double *) calloc(matrixRang, sizeof(double));
            matrixA[i] = (double *) calloc(matrixRang, sizeof(double));
            matrixC[i] = (double *) calloc(matrixRang, sizeof(double));
            matrixB[i] = (double *) calloc(matrixRang, sizeof(double));
            matrixDummy[i] = (double *) calloc(matrixRang, sizeof(double));
            eigenvectors[i]=  (double *) calloc(matrixRang, sizeof(double));
            for (int j=0;j<matrixRang;j++)
            {
                matrixCovarianceLag0[i][j] = NODATA;
                matrixCovarianceLag1[i][j] = NODATA;
                matrixA[i][j] = NODATA;
                matrixC[i][j] = NODATA;
                matrixB[i][j] = NODATA;
                matrixDummy[i][j] = NODATA;
                eigenvectors[i][j] = NODATA;
            }
            eigenvalues[i] = NODATA;
        }
        weatherGenerator2D::covarianceOfResiduals(matrixCovarianceLag0,0);
        weatherGenerator2D::covarianceOfResiduals(matrixCovarianceLag1,1);
        double ratioLag1 = 0;
        //double ratioLag0 = 0;
        double thresholdLag1 = 0.9;
        //thresholdLag1 = 0.999;
        if (matrixCovarianceLag1[1][1] > thresholdLag1)  // the numeric value is thought in order to avoid too extreme values
        {
            ratioLag1 = thresholdLag1/matrixCovarianceLag1[1][1];
            for (int j=0;j<matrixRang;j++)
            {
                for (int k=0;k<matrixRang;k++)
                {
                    matrixCovarianceLag1[j][k] *= ratioLag1;
                }
            }
        }
        double thresholdLag0 = 0.8;
        //thresholdLag0 = 0.999;
        if (matrixCovarianceLag0[0][1] > thresholdLag0) // the numeric value is thought in order to avoid too extreme values
        {
            matrixCovarianceLag0[0][1] = matrixCovarianceLag0[1][0] = thresholdLag0;
        }
        /*
        for (int j=0;j<matrixRang;j++)
        {
            for (int k=0;k<matrixRang;k++)
            {
                printf("%f  ",matrixCovarianceLag0[j][k]);
            }
            printf("\n");
        }
        printf("\n");
        for (int j=0;j<matrixRang;j++)
        {
            for (int k=0;k<matrixRang;k++)
            {
                printf("%f  ",matrixCovarianceLag1[j][k]);
            }
            printf("\n");
        }
        getchar();
        */
        matricial::inverse(matrixCovarianceLag0,matrixC,matrixRang); // matrixC becomes temporarely the inverse of lag0
        matricial::matrixProduct(matrixCovarianceLag1,matrixC,matrixRang,matrixRang,matrixRang,matrixRang,matrixA);
        matricial::transposedSquareMatrix(matrixCovarianceLag1,matrixRang);
        matricial::matrixProduct(matrixA,matrixCovarianceLag1,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
        matricial::matrixDifference(matrixCovarianceLag0,matrixC,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
        matricial::eigenSystemMatrix2x2(matrixC,eigenvalues,eigenvectors,matrixRang);
        int negativeEigenvalues = 0;
        if (eigenvalues[0] < 0)
        {
            eigenvalues[0] = EPSILON;
            negativeEigenvalues++;
        }
        if (eigenvalues[1] < 0)
        {
            eigenvalues[1] = EPSILON;
            negativeEigenvalues++;
        }
        if (negativeEigenvalues > 0)
        {
            matricial::inverse(eigenvectors,matrixDummy,matrixRang); // matrix C temporarely becomes the inverse matrix of the right eigenvectors
            matrixDummy[0][0] *= eigenvalues[0];
            matrixDummy[0][1] *= eigenvalues[0];
            matrixDummy[1][0] *= eigenvalues[1];
            matrixDummy[1][1] *= eigenvalues[1];
            matricial::matrixProduct(eigenvectors,matrixDummy,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
            matricial::eigenSystemMatrix2x2(matrixC,eigenvalues,eigenvectors,matrixRang);
        }
        matricial::inverse(eigenvectors,matrixDummy,2); // compulsory because our algorithm does not produce orthogonal vectors;
        matrixDummy[0][0] *= sqrt(eigenvalues[0]);
        matrixDummy[0][1] *= sqrt(eigenvalues[0]);
        matrixDummy[1][0] *= sqrt(eigenvalues[1]);
        matrixDummy[1][1] *= sqrt(eigenvalues[1]);
        matricial::matrixProduct(eigenvectors,matrixDummy,matrixRang,matrixRang,matrixRang,matrixRang,matrixB);

        for (int i=0;i<matrixRang;i++)
        {
            for (int j=0; j<matrixRang; j++)
            {
                temperatureCoefficients[iStation].A[i][j] = matrixA[i][j];
                temperatureCoefficients[iStation].B[i][j] = matrixB[i][j];
                //printf("%f  ",temperatureCoefficients[iStation].B[i][j]);
            }
            //printf("\n");
        }
        //getchar();

        for (int i=0;i<matrixRang;i++)
        {
            //matrixCovarianceLag0[i] = (double *) calloc(matrixRang, sizeof(double));
            //matrixCovarianceLag1[i] = (double *) calloc(matrixRang, sizeof(double));
            //matrixA[i] = (double *) calloc(matrixRang, sizeof(double));
            //matrixC[i] = (double *) calloc(matrixRang, sizeof(double));
            //matrixB[i] = (double *) calloc(matrixRang, sizeof(double));
            //matrixDummy[i] = (double *) calloc(matrixRang, sizeof(double));
            //eigenvectors[i]=  (double *) calloc(matrixRang, sizeof(double));
            for (int j=0;j<matrixRang;j++)
            {
                matrixCovarianceLag0[i][j] = NODATA;
                matrixCovarianceLag1[i][j] = NODATA;
                matrixA[i][j] = NODATA;
                matrixC[i][j] = NODATA;
                matrixB[i][j] = NODATA;
                matrixDummy[i][j] = NODATA;
                eigenvectors[i][j] = NODATA;
            }
            eigenvalues[i] = NODATA;
        }

        weatherGenerator2D::covarianceOfResidualsMeanDelta(matrixCovarianceLag0,0);
        weatherGenerator2D::covarianceOfResidualsMeanDelta(matrixCovarianceLag1,1);
        ratioLag1 = 0;
        //double ratioLag0 = 0;
        //thresholdLag1 = 0.9; // to check !!!
        if (matrixCovarianceLag1[1][1] > thresholdLag1)  // the numeric value is thought in order to avoid too extreme values
        {
            ratioLag1 = thresholdLag1/matrixCovarianceLag1[1][1];
            for (int j=0;j<matrixRang;j++)
            {
                for (int k=0;k<matrixRang;k++)
                {
                    matrixCovarianceLag1[j][k] *= ratioLag1;
                }
            }
        }
        //thresholdLag0 = 0.8; // to check !!!
        if (matrixCovarianceLag0[0][1] > thresholdLag0) // the numeric value is thought in order to avoid too extreme values
        {
            matrixCovarianceLag0[0][1] = matrixCovarianceLag0[1][0] = thresholdLag0;
        }

        matricial::inverse(matrixCovarianceLag0,matrixC,matrixRang); // matrixC becomes temporarely the inverse of lag0
        matricial::matrixProduct(matrixCovarianceLag1,matrixC,matrixRang,matrixRang,matrixRang,matrixRang,matrixA);
        matricial::transposedSquareMatrix(matrixCovarianceLag1,matrixRang);
        matricial::matrixProduct(matrixA,matrixCovarianceLag1,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
        matricial::matrixDifference(matrixCovarianceLag0,matrixC,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
        matricial::eigenSystemMatrix2x2(matrixC,eigenvalues,eigenvectors,matrixRang);
        negativeEigenvalues = 0;
        if (eigenvalues[0] < 0)
        {
            eigenvalues[0] = EPSILON;
            negativeEigenvalues++;
        }
        if (eigenvalues[1] < 0)
        {
            eigenvalues[1] = EPSILON;
            negativeEigenvalues++;
        }
        if (negativeEigenvalues > 0)
        {
            matricial::inverse(eigenvectors,matrixDummy,matrixRang); // matrix C temporarely becomes the inverse matrix of the right eigenvectors
            matrixDummy[0][0] *= eigenvalues[0];
            matrixDummy[0][1] *= eigenvalues[0];
            matrixDummy[1][0] *= eigenvalues[1];
            matrixDummy[1][1] *= eigenvalues[1];
            matricial::matrixProduct(eigenvectors,matrixDummy,matrixRang,matrixRang,matrixRang,matrixRang,matrixC);
            matricial::eigenSystemMatrix2x2(matrixC,eigenvalues,eigenvectors,matrixRang);
        }
        matricial::inverse(eigenvectors,matrixDummy,2); // compulsory because our algorithm does not produce orthogonal vectors;
        matrixDummy[0][0] *= sqrt(eigenvalues[0]);
        matrixDummy[0][1] *= sqrt(eigenvalues[0]);
        matrixDummy[1][0] *= sqrt(eigenvalues[1]);
        matrixDummy[1][1] *= sqrt(eigenvalues[1]);
        matricial::matrixProduct(eigenvectors,matrixDummy,matrixRang,matrixRang,matrixRang,matrixRang,matrixB);

        for (int i=0;i<matrixRang;i++)
        {
            for (int j=0; j<matrixRang; j++)
            {
                temperatureCoefficients[iStation].A_mean_delta[i][j] = matrixA[i][j];
                temperatureCoefficients[iStation].B_mean_delta[i][j] = matrixB[i][j];
                //printf("%f  ",temperatureCoefficients[iStation].A_mean_delta[i][j]);
            }
            //printf("\n");
        }

        //getchar();


        for (int i=0;i<matrixRang;i++)
        {
            free(matrixCovarianceLag0[i]);
            free(matrixCovarianceLag1[i]);
            free(matrixA[i]);
            free(matrixC[i]);
            free(matrixB[i]);
            free(matrixDummy[i]);
            free(eigenvectors[i]);
        }
        free(matrixCovarianceLag0);
        free(matrixCovarianceLag1);
        free(matrixA);
        free(matrixC);
        free(matrixB);
        free(matrixDummy);
        free(eigenvalues);
        free(eigenvectors);


    } // end of iStation "for" cycle
    free(dailyResidual);

}

void weatherGenerator2D::harmonicsFourier(double* variable, double *par,int nrPar, double* estimatedVariable, int nrEstimatedVariable)
{
    //int maxIterations = 100000;
    //int functionCode;

    // find the upper bound
    double valueMax,valueMin;
    bool validDays[365];
    int nrValidDays;

    //double valueCurrent;
    valueMax = 0;
    nrValidDays = 0;
    for (int i=0;i<365;i++)
    {
        //valueCurrent = variable[i];
        //valueCurrent = variable[i];
        if (fabs(variable[i] - NODATA)< EPSILON  || fabs(variable[i]) < EPSILON)
        {
            validDays[i] = false;
        }
        else
        {
            if (fabs(variable[i]) > valueMax) valueMax = fabs(variable[i]);
            validDays[i] = true;
            nrValidDays++;
        }
    }
    valueMin = -valueMax;

    double* x = (double *) calloc(nrValidDays, sizeof(double));
    double* y = (double *) calloc(nrValidDays, sizeof(double));
    int indexVariable = 0;
    for (int i=0;i<365;i++)
    {
        if(validDays[i])
        {
            x[indexVariable] = i + 1.0;
            y[indexVariable] = variable[i];
            indexVariable++;
        }
    }

    double *parMin = (double *) calloc(nrPar+1, sizeof(double));
    double* parMax = (double *) calloc(nrPar+1, sizeof(double));
    double* parDelta = (double *) calloc(nrPar+1, sizeof(double));
    double* parMarquardt = (double *) calloc(nrPar+1, sizeof(double));
    for (int i=0;i<(nrPar);i++)
    {
        parMin[i]= valueMin;
        parMax[i]= valueMax;
        parDelta[i] = 0.0001;
    }
    parMin[11]= 365;
    parMax[11]= 365; // da chiedere a Tomei come modificare questo !!
    parDelta[11] = 0;
    double meanVariable = 0;
    for (int i=0;i<nrValidDays;i++) meanVariable += y[i];
    meanVariable /= nrValidDays;
    parMarquardt[0] = par[0] = meanVariable;
    parMarquardt[1] = par[1] = 0;
    parMarquardt[2] = par[2] = 0;
    parMarquardt[3] = par[3] = 0;
    parMarquardt[4] = par[4] = 0;
    parMarquardt[5] = par[5] = 0;
    parMarquardt[6] = par[6] = 0;
    parMarquardt[7] = par[7] = 0;
    parMarquardt[8] = par[8] = 0;
    parMarquardt[9] = par[9] = 0;
    parMarquardt[10] = par[10] = 0;
    parMarquardt[11] = 365;

    interpolation::fittingMarquardt(parMin,parMax,parMarquardt,nrPar+1,parDelta,20000,0.0001,FUNCTION_CODE_FOURIER_GENERAL_HARMONICS,x,y,nrValidDays);

    for (int i=0;i<nrPar;i++)
    {
        par[i] = parMarquardt[i];
    }
    for (int i=0;i<365;i++)
    {
        estimatedVariable[i] = par[0]
                + par[1]*cos(2*PI/nrEstimatedVariable*i) + par[2]*sin(2*PI/nrEstimatedVariable*i)
                + par[3]*cos(4*PI/nrEstimatedVariable*i) + par[4]*sin(4*PI/nrEstimatedVariable*i)
                + par[5]*cos(6*PI/nrEstimatedVariable*i) + par[6]*sin(6*PI/nrEstimatedVariable*i)
                + par[7]*cos(8*PI/nrEstimatedVariable*i) + par[8]*sin(8*PI/nrEstimatedVariable*i)
                + par[9]*cos(10*PI/nrEstimatedVariable*i) + par[10]*sin(10*PI/nrEstimatedVariable*i);
    }

    // free memory
    free(x);
    free(y);
    free(parMin);
    free(parMax);
    free(parDelta);
    free(parMarquardt);

}

void weatherGenerator2D::computeResiduals(double* averageTMaxDry,double* averageTMaxWet,
                                          double* stdDevTMaxDry,double* stdDevTMaxWet,
                                          double* averageTMinDry,double* averageTMinWet,
                                          double* stdDevTMinDry,double* stdDevTMinWet,int idStation)
{
    for (int i=0; i<nrData; i++)
    {
        dailyResidual[i].maxTDry = 0.;
        dailyResidual[i].minTDry = 0.;
        dailyResidual[i].maxTWet = 0.;
        dailyResidual[i].minTWet = 0.;
        dailyResidual[i].maxT = 0.;
        dailyResidual[i].minT = 0.;
    }
    double maxResidual = NODATA;
    for (int i = 0; i< nrData ; i++)
    {

        int currentDayOfYear;
        currentDayOfYear = weatherGenerator2D::doyFromDate(obsDataD[idStation][i].date.day,obsDataD[idStation][i].date.month,obsDataD[idStation][i].date.year);

        if ((isLeapYear(obsDataD[idStation][i].date.year)) && (obsDataD[idStation][i].date.month > 2)) currentDayOfYear--;
        currentDayOfYear--;
        if (obsDataD[idStation][i].date.month == 2 && obsDataD[idStation][i].date.day == 29)
        {
            currentDayOfYear-- ;
        }
        if ((isTemperatureRecordOK(obsDataD[idStation][i].tMax)) && (fabs(obsDataD[idStation][i].tMax)> EPSILON) && (obsDataD[idStation][i].prec >= 0))
        {
            if(obsDataD[idStation][i].prec > parametersModel.precipitationThreshold)
            {
                dailyResidual[i].maxTWet = (obsDataD[idStation][i].tMax - averageTMaxWet[currentDayOfYear])/stdDevTMaxWet[currentDayOfYear];
                //dailyResidual[i].maxTWet = MINVALUE(dailyResidual[i].maxTWet,3);
            }
            else
            {
                dailyResidual[i].maxTDry = (obsDataD[idStation][i].tMax - averageTMaxDry[currentDayOfYear])/stdDevTMaxDry[currentDayOfYear];
                //dailyResidual[i].maxTDry = MINVALUE(dailyResidual[i].maxTDry,3);
            }
        }
        if ((isTemperatureRecordOK(obsDataD[idStation][i].tMin)) && (fabs(obsDataD[idStation][i].tMin)> EPSILON) && (obsDataD[idStation][i].prec >= 0))
        {
            if(obsDataD[idStation][i].prec > parametersModel.precipitationThreshold)
            {
                dailyResidual[i].minTWet = (obsDataD[idStation][i].tMin - averageTMinWet[currentDayOfYear])/stdDevTMinWet[currentDayOfYear];
                //dailyResidual[i].minTWet = MINVALUE(dailyResidual[i].minTWet,3);
            }
            else
            {
                dailyResidual[i].minTDry = (obsDataD[idStation][i].tMin - averageTMinDry[currentDayOfYear])/stdDevTMinDry[currentDayOfYear];
                //dailyResidual[i].minTDry = MINVALUE(dailyResidual[i].minTDry,3);
            }
        }
        dailyResidual[i].maxT = dailyResidual[i].maxTWet + dailyResidual[i].maxTDry;
        dailyResidual[i].minT = dailyResidual[i].minTWet + dailyResidual[i].minTDry;
        maxResidual = MAXVALUE(maxResidual,dailyResidual[i].minT * dailyResidual[i].maxT);
    }
}

void weatherGenerator2D::computeResidualsMeanDelta(double* averageTMeanDry,double* averageTMeanWet,
                                          double* stdDevTMeanDry,double* stdDevTMeanWet,
                                          double* averageTDeltaDry,double* averageTDeltaWet,
                                          double* stdDevTDeltaDry,double* stdDevTDeltaWet,int idStation)
{
    for (int i=0; i<nrData; i++)
    {
        dailyResidual[i].meanTDry = 0.;
        dailyResidual[i].deltaTDry = 0.;
        dailyResidual[i].meanTWet = 0.;
        dailyResidual[i].deltaTWet = 0.;
        dailyResidual[i].meanT = 0.;
        dailyResidual[i].deltaT = 0.;
    }
    //double maxResidual = NODATA;
    for (int i = 0; i< nrData ; i++)
    {

        int currentDayOfYear;
        currentDayOfYear = weatherGenerator2D::doyFromDate(obsDataD[idStation][i].date.day,obsDataD[idStation][i].date.month,obsDataD[idStation][i].date.year);

        if ((isLeapYear(obsDataD[idStation][i].date.year)) && (obsDataD[idStation][i].date.month > 2)) currentDayOfYear--;
        currentDayOfYear--;
        if (obsDataD[idStation][i].date.month == 2 && obsDataD[idStation][i].date.day == 29)
        {
            currentDayOfYear-- ;
        }
        if ((isTemperatureRecordOK(obsDataD[idStation][i].tMax)) && (fabs(obsDataD[idStation][i].tMax)> EPSILON) && (obsDataD[idStation][i].prec >= 0) && (isTemperatureRecordOK(obsDataD[idStation][i].tMin)) && (fabs(obsDataD[idStation][i].tMin)> EPSILON))
        {
            if(obsDataD[idStation][i].prec > parametersModel.precipitationThreshold)
            {
                dailyResidual[i].meanTWet = (0.5*(obsDataD[idStation][i].tMax+obsDataD[idStation][i].tMin) - averageTMeanWet[currentDayOfYear])/stdDevTMeanWet[currentDayOfYear];
                //dailyResidual[i].maxTWet = MINVALUE(dailyResidual[i].maxTWet,3);
            }
            else
            {
                dailyResidual[i].meanTDry = (0.5*(obsDataD[idStation][i].tMax+obsDataD[idStation][i].tMin) - averageTMeanDry[currentDayOfYear])/stdDevTMeanDry[currentDayOfYear];
                //dailyResidual[i].maxTDry = MINVALUE(dailyResidual[i].maxTDry,3);
            }
        }
        if ((isTemperatureRecordOK(obsDataD[idStation][i].tMax)) && (fabs(obsDataD[idStation][i].tMax)> EPSILON) && (obsDataD[idStation][i].prec >= 0) && (isTemperatureRecordOK(obsDataD[idStation][i].tMin)) && (fabs(obsDataD[idStation][i].tMin)> EPSILON))
        {
            if(obsDataD[idStation][i].prec > parametersModel.precipitationThreshold)
            {
                dailyResidual[i].deltaTWet = ((obsDataD[idStation][i].tMax-obsDataD[idStation][i].tMin) - averageTDeltaWet[currentDayOfYear])/stdDevTDeltaWet[currentDayOfYear];
                //dailyResidual[i].minTWet = MINVALUE(dailyResidual[i].minTWet,3);
            }
            else
            {
                dailyResidual[i].deltaTDry = ((obsDataD[idStation][i].tMax-obsDataD[idStation][i].tMin) - averageTDeltaDry[currentDayOfYear])/stdDevTDeltaDry[currentDayOfYear];
                //dailyResidual[i].minTDry = MINVALUE(dailyResidual[i].minTDry,3);
            }
        }
        dailyResidual[i].meanT = dailyResidual[i].meanTWet + dailyResidual[i].meanTDry;
        dailyResidual[i].deltaT = dailyResidual[i].deltaTWet + dailyResidual[i].deltaTDry;
    }
}

void weatherGenerator2D::covarianceOfResiduals(double** covarianceMatrix, int lag)
{
    if (lag == 0)
    {
        covarianceMatrix[0][0] = 1.0;
        covarianceMatrix[1][1] = 1.0;
        covarianceMatrix[1][0] = 0.0;
        covarianceMatrix[0][1] = 0.0;
        int denominator = -1;

        for (int i=0; i<nrData; i++)
        {
            if ((fabs(dailyResidual[i].maxT) > EPSILON) && (fabs(dailyResidual[i].minT) > EPSILON))
            {
                denominator++;
                covarianceMatrix[0][1] += dailyResidual[i].maxT * dailyResidual[i].minT;
            }
        }
        if (denominator > 0)
        {
            covarianceMatrix[0][1] /= denominator;
        }

        covarianceMatrix[1][0] =  covarianceMatrix[0][1];

    }
    else
    {
        covarianceMatrix[0][0] = 0.0;
        covarianceMatrix[1][1] = 0.0;
        covarianceMatrix[1][0] = 0.0;
        covarianceMatrix[0][1] = 0.0;
        int denominator11 = -2;
        int denominator12 = -2;
        int denominator21 = -2;
        int denominator22 = -2;
        for (int i=0; i<nrData-1; i++)
        {
            if ((fabs(dailyResidual[i].maxT) > EPSILON) && (fabs(dailyResidual[i+1].maxT) > EPSILON))
            {
                denominator11++;
                covarianceMatrix[0][0] += dailyResidual[i].maxT*dailyResidual[i+1].maxT;
            }
            if ((fabs(dailyResidual[i].maxT)> EPSILON) && (fabs(dailyResidual[i+1].minT)> EPSILON))
            {
                denominator12++;
                covarianceMatrix[0][1] += dailyResidual[i].maxT*dailyResidual[i+1].minT;
            }
            if ((fabs(dailyResidual[i].minT)> EPSILON) && (fabs(dailyResidual[i+1].maxT)> EPSILON))
            {
                denominator21++;
                covarianceMatrix[1][0] += dailyResidual[i].minT*dailyResidual[i+1].maxT;
            }
            if ((fabs(dailyResidual[i].minT)> EPSILON) && (fabs(dailyResidual[i+1].minT)> EPSILON))
            {
                denominator22++;
                covarianceMatrix[1][1] += dailyResidual[i].minT*dailyResidual[i+1].minT;
            }
        }
        //printf("%d %d %d %d\n",covarianceMatrix[0][0],covarianceMatrix[0][1],covarianceMatrix[1][0],covarianceMatrix[1][1]);
        //printf("%d %d %d %d\n",denominator11,denominator12,denominator21,denominator22);

        if (denominator11 != 0)
        {
            covarianceMatrix[0][0] /= denominator11;
        }
        if (denominator12 != 0)
        {
            covarianceMatrix[0][1] /= denominator12;
        }
        if (denominator21 != 0)
        {
            covarianceMatrix[1][0] /= denominator21;
        }
        if (denominator22 != 0)
        {
            covarianceMatrix[1][1] /= denominator22;
        }

    }

}

void weatherGenerator2D::covarianceOfResidualsMeanDelta(double** covarianceMatrix, int lag)
{
    if (lag == 0)
    {
        covarianceMatrix[0][0] = 1.0;
        covarianceMatrix[1][1] = 1.0;
        covarianceMatrix[1][0] = 0.0;
        covarianceMatrix[0][1] = 0.0;
        int denominator = -1;

        for (int i=0; i<nrData; i++)
        {
            if ((fabs(dailyResidual[i].meanT) > EPSILON) && (fabs(dailyResidual[i].deltaT) > EPSILON))
            {
                denominator++;
                covarianceMatrix[0][1] += dailyResidual[i].meanT * dailyResidual[i].deltaT;
            }
        }
        if (denominator > 0)
        {
            covarianceMatrix[0][1] /= denominator;
        }

        covarianceMatrix[1][0] =  covarianceMatrix[0][1];

    }
    else
    {
        covarianceMatrix[0][0] = 0.0;
        covarianceMatrix[1][1] = 0.0;
        covarianceMatrix[1][0] = 0.0;
        covarianceMatrix[0][1] = 0.0;
        int denominator11 = -2;
        int denominator12 = -2;
        int denominator21 = -2;
        int denominator22 = -2;
        for (int i=0; i<nrData-1; i++)
        {
            if ((fabs(dailyResidual[i].meanT) > EPSILON) && (fabs(dailyResidual[i+1].meanT) > EPSILON))
            {
                denominator11++;
                covarianceMatrix[0][0] += dailyResidual[i].maxT*dailyResidual[i+1].meanT;
            }
            if ((fabs(dailyResidual[i].meanT)> EPSILON) && (fabs(dailyResidual[i+1].deltaT)> EPSILON))
            {
                denominator12++;
                covarianceMatrix[0][1] += dailyResidual[i].meanT*dailyResidual[i+1].deltaT;
            }
            if ((fabs(dailyResidual[i].deltaT)> EPSILON) && (fabs(dailyResidual[i+1].meanT)> EPSILON))
            {
                denominator21++;
                covarianceMatrix[1][0] += dailyResidual[i].deltaT*dailyResidual[i+1].meanT;
            }
            if ((fabs(dailyResidual[i].deltaT)> EPSILON) && (fabs(dailyResidual[i+1].deltaT)> EPSILON))
            {
                denominator22++;
                covarianceMatrix[1][1] += dailyResidual[i].deltaT*dailyResidual[i+1].deltaT;
            }
        }
        //printf("%d %d %d %d\n",covarianceMatrix[0][0],covarianceMatrix[0][1],covarianceMatrix[1][0],covarianceMatrix[1][1]);
        //printf("%d %d %d %d\n",denominator11,denominator12,denominator21,denominator22);

        if (denominator11 != 0)
        {
            covarianceMatrix[0][0] /= denominator11;
        }
        if (denominator12 != 0)
        {
            covarianceMatrix[0][1] /= denominator12;
        }
        if (denominator21 != 0)
        {
            covarianceMatrix[1][0] /= denominator21;
        }
        if (denominator22 != 0)
        {
            covarianceMatrix[1][1] /= denominator22;
        }

    }

}

void weatherGenerator2D::temperaturesCorrelationMatrices()
{
    weatherGenerator2D::initializeTemperaturecorrelationMatrices();
    TcorrelationVar maxT;
    TcorrelationVar minT;
    int counterMaxT = 0;
    int counterMinT = 0;
    for (int j=0; j<nrStations;j++)
    {
        correlationMatrixTemperature.maxT[j][j] = 1;
        correlationMatrixTemperature.minT[j][j] = 1;
    }

    // compute correlation for tmax
    for (int j=0; j<nrStations-1;j++)
    {
        for (int i=j+1; i<nrStations;i++)
        {
            counterMaxT = 0;
            maxT.meanValue1=0.;
            maxT.meanValue2=0.;
            maxT.covariance = maxT.variance1 = maxT.variance2 = 0.;

            for (int k=0; k<nrData;k++) // compute the monthly means
            {
                if ((fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON))
                {
                    counterMaxT++;
                    maxT.meanValue1 += obsDataD[j][k].tMax ;
                    maxT.meanValue2 += obsDataD[i][k].tMax;

                }
            }
            if (counterMaxT != 0)
            {
                maxT.meanValue1 /= counterMaxT;
                maxT.meanValue2 /= counterMaxT;
            }

            // compute the monthly rho off-diagonal elements
            for (int k=0; k<nrData;k++)
            {
                if ((fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON))
                {
                    double value1,value2;
                    value1 = obsDataD[j][k].tMax;
                    value2 = obsDataD[i][k].tMax;

                    maxT.covariance += (value1 - maxT.meanValue1)*(value2 - maxT.meanValue2);
                    maxT.variance1 += (value1 - maxT.meanValue1)*(value1 - maxT.meanValue1);
                    maxT.variance2 += (value2 - maxT.meanValue2)*(value2 - maxT.meanValue2);
                }
            }
            correlationMatrixTemperature.maxT[j][i]= maxT.covariance / sqrt(maxT.variance1*maxT.variance2);
            correlationMatrixTemperature.maxT[i][j] = correlationMatrixTemperature.maxT[j][i];
        }
    }

    // compute correlation for tmin
    for (int j=0; j<nrStations-1;j++)
    {
        for (int i=j+1; i<nrStations;i++)
        {
            counterMinT = 0;
            minT.meanValue1=0.;
            minT.meanValue2=0.;
            minT.covariance = minT.variance1 = minT.variance2 = 0.;

            for (int k=0; k<nrData;k++) // compute the monthly means
            {
                if ((fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON))
                {
                    counterMinT++;
                    minT.meanValue1 += obsDataD[j][k].tMin ;
                    minT.meanValue2 += obsDataD[i][k].tMin;

                }
            }
            if (counterMinT != 0)
            {
                minT.meanValue1 /= counterMinT;
                minT.meanValue2 /= counterMinT;
            }
            // compute the monthly rho off-diagonal elements
            for (int k=0; k<nrData;k++)
            {
                if ((fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON))
                {
                    double value1,value2;
                    value1 = obsDataD[j][k].tMin;
                    value2 = obsDataD[i][k].tMin;

                    minT.covariance += (value1 - minT.meanValue1)*(value2 - minT.meanValue2);
                    minT.variance1 += (value1 - minT.meanValue1)*(value1 - minT.meanValue1);
                    minT.variance2 += (value2 - minT.meanValue2)*(value2 - minT.meanValue2);
                }
            }
            correlationMatrixTemperature.minT[j][i]= minT.covariance / sqrt(minT.variance1*minT.variance2);
            correlationMatrixTemperature.minT[i][j] = correlationMatrixTemperature.minT[j][i];
        }
    }


    TcorrelationVar meanT;
    TcorrelationVar deltaT;
    int counterMeanT = 0;
    int counterDeltaT = 0;
    for (int j=0; j<nrStations;j++)
    {
        correlationMatrixTemperature.meanT[j][j] = 1;
        correlationMatrixTemperature.deltaT[j][j] = 1;
    }

    // compute correlation for tmean
    for (int j=0; j<nrStations-1;j++)
    {
        for (int i=j+1; i<nrStations;i++)
        {
            counterMeanT = 0;
            meanT.meanValue1=0.;
            meanT.meanValue2=0.;
            meanT.covariance = meanT.variance1 = meanT.variance2 = 0.;

            for (int k=0; k<nrData;k++) // compute the monthly means
            {
                if ((fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON))
                {
                    counterMeanT++;
                    meanT.meanValue1 += 0.5*(obsDataD[j][k].tMax + obsDataD[j][k].tMin) ;
                    meanT.meanValue2 += 0.5*(obsDataD[i][k].tMax + obsDataD[i][k].tMin);

                }
            }
            if (counterMeanT != 0)
            {
                meanT.meanValue1 /= counterMeanT;
                meanT.meanValue2 /= counterMeanT;
            }

            // compute the monthly rho off-diagonal elements
            for (int k=0; k<nrData;k++)
            {
                if ((fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON))
                {
                    double value1,value2;
                    value1 = 0.5*(obsDataD[j][k].tMax + obsDataD[j][k].tMin);
                    value2 = 0.5*(obsDataD[i][k].tMax + obsDataD[i][k].tMin);

                    meanT.covariance += (value1 - meanT.meanValue1)*(value2 - meanT.meanValue2);
                    meanT.variance1 += (value1 - meanT.meanValue1)*(value1 - meanT.meanValue1);
                    meanT.variance2 += (value2 - meanT.meanValue2)*(value2 - meanT.meanValue2);
                }
            }
            correlationMatrixTemperature.meanT[j][i]= meanT.covariance / sqrt(meanT.variance1*meanT.variance2);
            correlationMatrixTemperature.meanT[i][j] = correlationMatrixTemperature.meanT[j][i];
        }
    }

    // compute correlation for tdelta
    for (int j=0; j<nrStations-1;j++)
    {
        for (int i=j+1; i<nrStations;i++)
        {
            counterDeltaT = 0;
            deltaT.meanValue1=0.;
            deltaT.meanValue2=0.;
            deltaT.covariance = deltaT.variance1 = deltaT.variance2 = 0.;

            for (int k=0; k<nrData;k++) // compute the monthly means
            {
                if ((fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON))
                {
                    counterDeltaT++;
                    deltaT.meanValue1 += obsDataD[j][k].tMax - obsDataD[j][k].tMin ;
                    deltaT.meanValue2 += obsDataD[i][k].tMax - obsDataD[i][k].tMin;

                }
            }
            if (counterMinT != 0)
            {
                deltaT.meanValue1 /= counterDeltaT;
                deltaT.meanValue2 /= counterDeltaT;
            }
            // compute the monthly rho off-diagonal elements
            for (int k=0; k<nrData;k++)
            {
                if ((fabs(obsDataD[j][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[j][k].tMax - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMin - NODATA) > EPSILON) && (fabs(obsDataD[i][k].tMax - NODATA) > EPSILON))
                {
                    double value1,value2;
                    value1 = obsDataD[j][k].tMax - obsDataD[j][k].tMin;
                    value2 = obsDataD[i][k].tMax - obsDataD[i][k].tMin;

                    deltaT.covariance += (value1 - deltaT.meanValue1)*(value2 - deltaT.meanValue2);
                    deltaT.variance1 += (value1 - deltaT.meanValue1)*(value1 - deltaT.meanValue1);
                    deltaT.variance2 += (value2 - deltaT.meanValue2)*(value2 - deltaT.meanValue2);
                }
            }
            correlationMatrixTemperature.deltaT[j][i]= deltaT.covariance / sqrt(deltaT.variance1*deltaT.variance2);
            correlationMatrixTemperature.deltaT[i][j] = correlationMatrixTemperature.deltaT[j][i];
        }
    }
}

void weatherGenerator2D::initializeNormalRandomMatricesTemperatures()
{
    int lengthOfRandomSeries;
    lengthOfRandomSeries = parametersModel.yearOfSimulation*365;
    normRandomMaxT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));
    normRandomMinT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));
    normRandomMeanT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));
    normRandomDeltaT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));

    for (int i=0;i<lengthOfRandomSeries;i++)
    {
        normRandomMaxT[i] = (double*)calloc(nrStations, sizeof(double));
        normRandomMinT[i] = (double*)calloc(nrStations, sizeof(double));
        normRandomMeanT[i] = (double*)calloc(nrStations, sizeof(double));
        normRandomDeltaT[i] = (double*)calloc(nrStations, sizeof(double));
    }
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandomMaxT[j][i] = NODATA;
            normRandomMinT[j][i] = NODATA;
            normRandomMeanT[j][i] = NODATA;
            normRandomDeltaT[j][i] = NODATA;
        }
    }
}

void weatherGenerator2D::multisiteRandomNumbersTemperature()
{
    weatherGenerator2D::initializeNormalRandomMatricesTemperatures();
    int gasDevIset = 0;
    double gasDevGset = 0;
    srand (time(nullptr));
    //int firstRandomNumber;
    //firstRandomNumber = rand();
    int lengthOfRandomSeries;
    lengthOfRandomSeries = parametersModel.yearOfSimulation*365;
    int nrSquareOfStations;
    nrSquareOfStations = nrStations*nrStations;

    double* correlationArray = (double *) calloc(nrSquareOfStations, sizeof(double));
    double* eigenvectors = (double *) calloc(nrSquareOfStations, sizeof(double));
    double* eigenvalues = (double *) calloc(nrStations, sizeof(double));
    double** dummyMatrix = (double**)calloc(nrStations, sizeof(double*));
    double** dummyMatrix2 = (double**)calloc(nrStations, sizeof(double*));
    double** dummyMatrix3 = (double**)calloc(nrStations, sizeof(double*));
    double** normRandom = (double**)calloc(nrStations, sizeof(double*));
    double** normRandom2 = (double**)calloc(nrStations, sizeof(double*));
    // initialization internal arrays
    for (int i=0;i<nrStations;i++)
    {
        dummyMatrix[i]= (double*)calloc(nrStations, sizeof(double));
        dummyMatrix2[i]= (double*)calloc(nrStations, sizeof(double));
        dummyMatrix3[i] = (double*)calloc(nrStations, sizeof(double));
        normRandom[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
        normRandom2[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
    }
    int counter;
    bool isLowerDiagonal;

    // for Tmax
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<nrStations;j++)
        {
            correlationArray[counter] = correlationMatrixTemperature.maxT[i][j];
            eigenvectors[counter] = NODATA;
            counter++;
        }
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = NODATA;
            normRandom2[i][j] = NODATA;
        }
        eigenvalues[i]= NODATA;
    }

    eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors);
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        if (eigenvalues[i] < 0)
        {
            counter++;
            eigenvalues[i] = 0.000001;
        }
    }

    if (counter > 0)
    {
        counter=0;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                dummyMatrix[j][i]= eigenvectors[counter];
                dummyMatrix2[i][j]= eigenvectors[counter]*eigenvalues[i];
                counter++;
            }
        }
        matricial::matrixProduct(dummyMatrix,dummyMatrix2,nrStations,nrStations,nrStations,nrStations,correlationMatrixTemperature.maxT);
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrStations;j++)
        {
            dummyMatrix[i][j] = correlationMatrixTemperature.maxT[i][j];
        }

    }

    isLowerDiagonal = false;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,isLowerDiagonal);
    matricial::transposedMatrix(dummyMatrix,nrStations,nrStations,dummyMatrix2);
    matricial::matrixProduct(dummyMatrix2,dummyMatrix,nrStations,nrStations,nrStations,nrStations,dummyMatrix3);
    isLowerDiagonal = true;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix3,nrStations,isLowerDiagonal);


    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
        }
    }

    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomMaxT);

    // for Tmin
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<nrStations;j++)
        {
            correlationArray[counter] = correlationMatrixTemperature.minT[i][j];
            eigenvectors[counter] = NODATA;
            counter++;
        }
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = NODATA;
            normRandom2[i][j] = NODATA;
        }
        eigenvalues[i]= NODATA;
    }

    eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors);
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        if (eigenvalues[i] < 0)
        {
            counter++;
            eigenvalues[i] = 0.000001;
        }
    }

    if (counter > 0)
    {
        counter=0;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                dummyMatrix[j][i]= eigenvectors[counter];
                dummyMatrix2[i][j]= eigenvectors[counter]*eigenvalues[i];
                counter++;
            }
        }
        matricial::matrixProduct(dummyMatrix,dummyMatrix2,nrStations,nrStations,nrStations,nrStations,correlationMatrixTemperature.minT);
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrStations;j++)
        {
            dummyMatrix[i][j] = correlationMatrixTemperature.minT[i][j];
        }
    }

    isLowerDiagonal = false;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,isLowerDiagonal);
    matricial::transposedMatrix(dummyMatrix,nrStations,nrStations,dummyMatrix2);
    matricial::matrixProduct(dummyMatrix2,dummyMatrix,nrStations,nrStations,nrStations,nrStations,dummyMatrix3);
    isLowerDiagonal = true;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix3,nrStations,isLowerDiagonal);

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
        }
    }


    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomMinT);

    // for tmean
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<nrStations;j++)
        {
            correlationArray[counter] = correlationMatrixTemperature.meanT[i][j];
            eigenvectors[counter] = NODATA;
            counter++;
        }
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = NODATA;
            normRandom2[i][j] = NODATA;
        }
        eigenvalues[i]= NODATA;
    }

    eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors);
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        if (eigenvalues[i] < 0)
        {
            counter++;
            eigenvalues[i] = 0.000001;
        }
    }

    if (counter > 0)
    {
        counter=0;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                dummyMatrix[j][i]= eigenvectors[counter];
                dummyMatrix2[i][j]= eigenvectors[counter]*eigenvalues[i];
                counter++;
            }
        }
        matricial::matrixProduct(dummyMatrix,dummyMatrix2,nrStations,nrStations,nrStations,nrStations,correlationMatrixTemperature.meanT);
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrStations;j++)
        {
            dummyMatrix[i][j] = correlationMatrixTemperature.meanT[i][j];
        }

    }

    isLowerDiagonal = false;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,isLowerDiagonal);
    matricial::transposedMatrix(dummyMatrix,nrStations,nrStations,dummyMatrix2);
    matricial::matrixProduct(dummyMatrix2,dummyMatrix,nrStations,nrStations,nrStations,nrStations,dummyMatrix3);
    isLowerDiagonal = true;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix3,nrStations,isLowerDiagonal);


    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
        }
    }

    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomMeanT);

    // for Tdelta
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<nrStations;j++)
        {
            correlationArray[counter] = correlationMatrixTemperature.deltaT[i][j];
            eigenvectors[counter] = NODATA;
            counter++;
        }
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = NODATA;
            normRandom2[i][j] = NODATA;
        }
        eigenvalues[i]= NODATA;
    }

    eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors);
    counter = 0;
    for (int i=0; i<nrStations;i++)
    {
        if (eigenvalues[i] < 0)
        {
            counter++;
            eigenvalues[i] = 0.000001;
        }
    }

    if (counter > 0)
    {
        counter=0;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                dummyMatrix[j][i]= eigenvectors[counter];
                dummyMatrix2[i][j]= eigenvectors[counter]*eigenvalues[i];
                counter++;
            }
        }
        matricial::matrixProduct(dummyMatrix,dummyMatrix2,nrStations,nrStations,nrStations,nrStations,correlationMatrixTemperature.deltaT);
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrStations;j++)
        {
            dummyMatrix[i][j] = correlationMatrixTemperature.deltaT[i][j];
        }
    }

    isLowerDiagonal = false;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,isLowerDiagonal);
    matricial::transposedMatrix(dummyMatrix,nrStations,nrStations,dummyMatrix2);
    matricial::matrixProduct(dummyMatrix2,dummyMatrix,nrStations,nrStations,nrStations,nrStations,dummyMatrix3);
    isLowerDiagonal = true;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix3,nrStations,isLowerDiagonal);

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
        }
    }


    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomDeltaT);

    for (int i=0;i<nrStations;i++)
    {
        free(dummyMatrix[i]);
        free(dummyMatrix2[i]);
        free(dummyMatrix3[i]);
        free(normRandom[i]);
        free(normRandom2[i]);
    }

    free(dummyMatrix);
    free(dummyMatrix2);
    free(dummyMatrix3);
    free(correlationArray);
    free(eigenvalues);
    free(eigenvectors);
    free(normRandom);
    free(normRandom2);

    // free memory of class
    for (int i=0;i<nrStations;i++)
    {
        free(correlationMatrixTemperature.maxT[i]);
        free(correlationMatrixTemperature.minT[i]);
    }
    free(correlationMatrixTemperature.maxT);
    free(correlationMatrixTemperature.minT);

    for (int i=0;i<nrStations;i++)
    {
        free(correlationMatrixTemperature.meanT[i]);
        free(correlationMatrixTemperature.deltaT[i]);
    }
    free(correlationMatrixTemperature.meanT);
    free(correlationMatrixTemperature.deltaT);
    printf("abcd\n");
}

void weatherGenerator2D::initializeMultiOccurrenceTemperature(int length)
{
    multiOccurrenceTemperature = (TmultiOccurrenceTemperature *) calloc(length, sizeof(TmultiOccurrenceTemperature));
    for (int i=0;i<length;i++)
    {
        multiOccurrenceTemperature[i].occurrence_simulated = (double *) calloc(nrStations, sizeof(double));
        for(int j=0;j<nrStations;j++)
        {
            multiOccurrenceTemperature[i].occurrence_simulated[j] = NODATA;
        }
        multiOccurrenceTemperature[i].day_simulated = NODATA;
        multiOccurrenceTemperature[i].month_simulated = NODATA;
        multiOccurrenceTemperature[i].year_simulated = NODATA;
    }
}

void weatherGenerator2D::initializeTemperaturesOutput(int length)
{
    maxTGenerated = (double **) calloc(length, sizeof(double *));
    minTGenerated = (double **) calloc(length, sizeof(double *));
    meanTGenerated = (double **) calloc(length, sizeof(double *));
    deltaTGenerated = (double **) calloc(length, sizeof(double *));
    occurrencePrecGenerated = (double **) calloc(length, sizeof(double *));
    amountsPrecGenerated = (double **) calloc(length, sizeof(double *));
    for (int i=0;i<length;i++)
    {
        maxTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        minTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        meanTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        deltaTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        occurrencePrecGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        amountsPrecGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        for(int j=0;j<nrStations;j++)
        {
            maxTGenerated[i][j] = NODATA;
            minTGenerated[i][j] = NODATA;
            meanTGenerated[i][j] = NODATA;
            deltaTGenerated[i][j] = NODATA;
            occurrencePrecGenerated[i][j] = NODATA;
            amountsPrecGenerated[i][j] = NODATA;
        }

    }
}



void weatherGenerator2D::multisiteTemperatureGeneration()
{
    int lengthOfRandomSeries;
    lengthOfRandomSeries = parametersModel.yearOfSimulation*365;
    //weatherGenerator2D::initializeMultiOccurrenceTemperature(lengthOfRandomSeries);
    // fill in the data of simulations
    int daySimulated,monthSimulated;
    int counter = 0;
    for (int j=1;j<=parametersModel.yearOfSimulation;j++)
    {
        for (int i=0; i<365;i++)
        {
            weatherGenerator2D::dateFromDoy(i+1,1,&daySimulated,&monthSimulated); // 1 to avoid leap years
            multiOccurrenceTemperature[counter].year_simulated = j;
            multiOccurrenceTemperature[counter].month_simulated = monthSimulated;
            multiOccurrenceTemperature[counter].day_simulated = daySimulated;
            counter++;
        }
    }

    for (int j=0;j<12;j++)
    {
        int counter2 = 0;
        counter = 0;
        for (int k=0; k<lengthOfRandomSeries; k++)
        {
            if (multiOccurrenceTemperature[counter].month_simulated == (j+1))
            {
                for (int i=0;i<nrStations;i++)
                {
                    multiOccurrenceTemperature[counter].occurrence_simulated[i] = randomMatrix[j].matrixOccurrences[i][counter2];
                }
                counter2++;
            }
            counter++;
        }
    }

    //weatherGenerator2D::initializeTemperaturesOutput(lengthOfRandomSeries);
    double* X = (double*)calloc(lengthOfRandomSeries, sizeof(double));
    double** averageT = (double**)calloc(4, sizeof(double*));
    double** stdDevT = (double**)calloc(4, sizeof(double*));
    for (int i=0;i<4;i++)
    {
        averageT[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
        stdDevT[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            averageT[i][j] = NODATA;
            stdDevT[i][j] = NODATA;
        }
    }

    for (int i=0; i<lengthOfRandomSeries ; i++)
    {
        X[i] = NODATA;
    }
    for (int i=0;i<nrStations;i++)
    {
        int jModulo;
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            X[j] = multiOccurrenceTemperature[j].occurrence_simulated[i];
            jModulo = j%365;

            averageT[0][j] = temperatureCoefficients[i].maxTDry.averageEstimation[jModulo];
            averageT[1][j] = temperatureCoefficients[i].minTDry.averageEstimation[jModulo];
            averageT[2][j] = temperatureCoefficients[i].maxTWet.averageEstimation[jModulo];
            averageT[3][j] = temperatureCoefficients[i].minTWet.averageEstimation[jModulo];

            stdDevT[0][j] = temperatureCoefficients[i].maxTDry.stdDevEstimation[jModulo];
            stdDevT[1][j] = temperatureCoefficients[i].minTDry.stdDevEstimation[jModulo];
            stdDevT[2][j] = temperatureCoefficients[i].maxTWet.stdDevEstimation[jModulo];
            stdDevT[3][j] = temperatureCoefficients[i].minTWet.stdDevEstimation[jModulo];

        }
        double* residuals = (double*)calloc(2, sizeof(double));
        residuals[0] = residuals[1] = 0;
        double** ksi = (double**)calloc(2, sizeof(double*));
        double** eps = (double**)calloc(2, sizeof(double*));
        for (int j=0;j<2;j++)
        {
            ksi[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            eps[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            for (int k=0;k<lengthOfRandomSeries;k++)
            {
               ksi[j][k] = 0;  // initialization
               eps[j][k] = 0;  // initialization
            }
        }

        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            eps[0][j] = normRandomMaxT[j][i];
            eps[1][j] = normRandomMinT[j][i];
            //printf("%.1f %.1f\n",eps[0][j],eps[1][j]);
        }
        //getchar();
        double res0,res1;
        res1 = res0 = 0;
        for (int j=0;j<lengthOfRandomSeries;j++)
        {


            //printf("%.1f %.1f\n",residuals[0],residuals[1]);

            res0 = temperatureCoefficients[i].A[0][0]*residuals[0] + temperatureCoefficients[i].A[0][1]*residuals[1];
            res0 += temperatureCoefficients[i].B[0][0]*eps[0][j] + temperatureCoefficients[i].B[0][1]*eps[1][j];
            res1 = temperatureCoefficients[i].A[1][0]*residuals[0] + temperatureCoefficients[i].A[1][1]*residuals[1];
            res1 += temperatureCoefficients[i].B[1][0]*eps[0][j] + temperatureCoefficients[i].B[1][1]*eps[1][j];
            ksi[0][j] = residuals[0] = res0;
            ksi[1][j] = residuals[1] = res1;
            //printf("%.1f %.1f\n",eps[0][j],eps[1][j]);
            //printf("%d %.1f %.1f %.1f %.1f\n",i, temperatureCoefficients[i].B[0][0],temperatureCoefficients[i].B[0][1],temperatureCoefficients[i].B[1][0],temperatureCoefficients[i].B[1][1]);
            //printf("%d %.1f %.1f %.1f %.1f\n",i, temperatureCoefficients[i].A[0][0],temperatureCoefficients[i].A[0][1],temperatureCoefficients[i].A[1][0],temperatureCoefficients[i].A[1][1]);
            //residuals[0] = residuals[1]=0;
            //printf("%.1f %.1f\n",residuals[0],residuals[1]);
            //getchar();
        }
        //getchar();
        double** cAverage = (double**)calloc(2, sizeof(double*));
        double** cStdDev = (double**)calloc(2, sizeof(double*));
        double** Xp = (double**)calloc(2, sizeof(double*));

        for (int j=0;j<2;j++)
        {
            cAverage[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            cStdDev[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            Xp[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));

            for (int k=0;k<lengthOfRandomSeries;k++)
            {
                cAverage[j][k] = NODATA;
                cStdDev[j][k] = NODATA;
                Xp[j][k] = NODATA;
            }
        }
        // !!!!!! this part has been changed from the original one. To be verified with authors and colleagues!
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            //X[j] = 1 - X[j]; // to come back to the original code
            cAverage[0][j] = X[j]*averageT[2][j] + (1- X[j])*averageT[0][j]; // for Tmax
            cAverage[1][j] = X[j]*averageT[3][j] + (1- X[j])*averageT[1][j]; // for Tmin
            cStdDev[0][j] = X[j]*stdDevT[2][j] + (1-X[j])*stdDevT[0][j]; // for Tmax
            cStdDev[1][j] = X[j]*stdDevT[3][j] + (1-X[j])*stdDevT[1][j]; // for Tmin
        }
        // !!!!!!
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            if(cStdDev[0][j] >= cStdDev[1][j])
            {
                Xp[1][j] = ksi[1][j]*cStdDev[1][j] + cAverage[1][j];
                Xp[0][j] = ksi[0][j]*sqrt(cStdDev[0][j]*cStdDev[0][j] - cStdDev[1][j]*cStdDev[1][j]) + (cAverage[0][j] - cAverage[1][j]) + Xp[1][j];
            }
            else
            {
                Xp[0][j] = ksi[0][j]*cStdDev[0][j] + cAverage[0][j];
                Xp[1][j] = ksi[1][j]*sqrt(cStdDev[1][j]*cStdDev[1][j] - cStdDev[0][j]*cStdDev[0][j]) - (cAverage[0][j] - cAverage[1][j]) + Xp[0][j];
            }
             //printf("%.1f %.1f\n",ksi[0][j],ksi[1][j]);
        }
        //getchar();

        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            if (Xp[0][j] <= Xp[1][j])
            {
                Xp[1][j] = Xp[0][j] - 0.2*fabs(Xp[0][j]);
            }
        }
        double averageTmax[365]={0};
        double averageTmin[365]={0};
        srand(time(nullptr));
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            int getDecadal = (multiOccurrenceTemperature[j].month_simulated-1)*3 + floor(MINVALUE(multiOccurrenceTemperature[j].day_simulated,29)/10.);
            int getYear = floor(1.*j/365.);
            double random1,random2;
            //random1 = 0.5*((double) rand() / (RAND_MAX) -0.5) + monthlyRandomDeviationTmean[getYear][getDecadal];
            //random2 = 0.5*((double) rand() / (RAND_MAX) -0.5)+ monthlyRandomDeviationTmean[getYear][getDecadal];
            //random1 = monthlyRandomDeviationTmean[getYear][getDecadal];
            //random2 = monthlyRandomDeviationTmean[getYear][getDecadal];
            //double thresholdVariation = 7;
            //random1 = MINVALUE(MAXVALUE(random1,-thresholdVariation),thresholdVariation);
            //random2 = MINVALUE(MAXVALUE(random2,-thresholdVariation),thresholdVariation);
            random1 = random2 = 0;
            maxTGenerated[j][i] = Xp[0][j] + random1;
            minTGenerated[j][i] = Xp[1][j] + random2;
            double tempTemp;
            if (maxTGenerated[j][i] < minTGenerated[j][i])
            {
                tempTemp = maxTGenerated[j][i];
                maxTGenerated[j][i] = minTGenerated[j][i];
                minTGenerated[j][i] = tempTemp;
            }
            //random1 = 0.5*((double) rand() / (RAND_MAX) -0.5);
            //random2 = 0.5*((double) rand() / (RAND_MAX) -0.5);
            //maxTGenerated[j][i] = Xp[0][j] + monthlyRandomDeviationTmean[getYear][getDecadal]*1.0; //multiOccurrenceTemperature[j].;
            //minTGenerated[j][i] = Xp[1][j] + monthlyRandomDeviationTmean[getYear][getDecadal]*1.0;

            occurrencePrecGenerated[j][i] = X[j];
            //maxTGenerated[j][i] = Xp[0][j];
            //minTGenerated[j][i] = Xp[1][j];
            //occurrencePrecGenerated[j][i] = X[j];
            averageTmax[j%365] += maxTGenerated[j][i]/parametersModel.yearOfSimulation;
            averageTmin[j%365] += minTGenerated[j][i]/parametersModel.yearOfSimulation;
            //printf("random1 %f random2 %f\n",random1,random2);
        }
        //pressEnterToContinue();
        // free memory
        free(ksi[0]);
        free(ksi[1]);
        free(eps[0]);
        free(eps[1]);
        free(cAverage[0]);
        free(cAverage[1]);
        free(cStdDev[0]);
        free(cStdDev[1]);
        free(Xp[0]);
        free(Xp[1]);
        free(ksi);
        free(eps);
        free(cAverage);
        free(cStdDev);
        free(residuals);
        free(Xp);
    }

    for (int i=0;i<4;i++)
    {
        free(averageT[i]);
        free(stdDevT[i]);
    }

    free(averageT);
    free(stdDevT);
    free(X);

    // free class memory
    for (int j=0;j<lengthOfRandomSeries;j++)
    {
        free(multiOccurrenceTemperature[j].occurrence_simulated);
    }
    free(multiOccurrenceTemperature);

}

void weatherGenerator2D::multisiteTemperatureGenerationMeanDelta()
{
    int lengthOfRandomSeries;
    lengthOfRandomSeries = parametersModel.yearOfSimulation*365;
    //weatherGenerator2D::initializeMultiOccurrenceTemperature(lengthOfRandomSeries);
    // fill in the data of simulations
    int daySimulated,monthSimulated;
    int counter = 0;
    for (int j=1;j<=parametersModel.yearOfSimulation;j++)
    {
        for (int i=0; i<365;i++)
        {
            weatherGenerator2D::dateFromDoy(i+1,1,&daySimulated,&monthSimulated); // 1 to avoid leap years
            multiOccurrenceTemperature[counter].year_simulated = j;
            multiOccurrenceTemperature[counter].month_simulated = monthSimulated;
            multiOccurrenceTemperature[counter].day_simulated = daySimulated;
            counter++;
        }
    }

    for (int j=0;j<12;j++)
    {
        int counter2 = 0;
        counter = 0;
        for (int k=0; k<lengthOfRandomSeries; k++)
        {
            if (multiOccurrenceTemperature[counter].month_simulated == (j+1))
            {
                for (int i=0;i<nrStations;i++)
                {
                    multiOccurrenceTemperature[counter].occurrence_simulated[i] = randomMatrix[j].matrixOccurrences[i][counter2];
                }
                counter2++;
            }
            counter++;
        }
    }

    //weatherGenerator2D::initializeTemperaturesOutput(lengthOfRandomSeries);
    double* X = (double*)calloc(lengthOfRandomSeries, sizeof(double));
    double** averageT = (double**)calloc(4, sizeof(double*));
    double** stdDevT = (double**)calloc(4, sizeof(double*));
    for (int i=0;i<4;i++)
    {
        averageT[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
        stdDevT[i] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            averageT[i][j] = NODATA;
            stdDevT[i][j] = NODATA;
        }
    }

    for (int i=0; i<lengthOfRandomSeries ; i++)
    {
        X[i] = NODATA;
    }
    for (int i=0;i<nrStations;i++)
    {
        int jModulo;
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            X[j] = multiOccurrenceTemperature[j].occurrence_simulated[i];
            jModulo = j%365;

            averageT[0][j] = temperatureCoefficients[i].meanTDry.averageEstimation[jModulo];
            averageT[1][j] = temperatureCoefficients[i].deltaTDry.averageEstimation[jModulo];
            averageT[2][j] = temperatureCoefficients[i].meanTWet.averageEstimation[jModulo];
            averageT[3][j] = temperatureCoefficients[i].deltaTWet.averageEstimation[jModulo];

            stdDevT[0][j] = temperatureCoefficients[i].meanTDry.stdDevEstimation[jModulo];
            stdDevT[1][j] = temperatureCoefficients[i].deltaTDry.stdDevEstimation[jModulo];
            stdDevT[2][j] = temperatureCoefficients[i].meanTWet.stdDevEstimation[jModulo];
            stdDevT[3][j] = temperatureCoefficients[i].deltaTWet.stdDevEstimation[jModulo];

        }
        double* residuals = (double*)calloc(2, sizeof(double));
        residuals[0] = residuals[1] = 0;
        double** ksi = (double**)calloc(2, sizeof(double*));
        double** eps = (double**)calloc(2, sizeof(double*));
        for (int j=0;j<2;j++)
        {
            ksi[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            eps[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            for (int k=0;k<lengthOfRandomSeries;k++)
            {
               ksi[j][k] = 0;  // initialization
               eps[j][k] = 0;  // initialization
            }
        }

        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            eps[0][j] = normRandomMeanT[j][i];
            eps[1][j] = normRandomDeltaT[j][i];
            //printf("%.1f %.1f\n",eps[0][j],eps[1][j]);
        }
        //getchar();
        double res0,res1;
        res1 = res0 = 0;
        for (int j=0;j<lengthOfRandomSeries;j++)
        {


            //printf("%.1f %.1f\n",residuals[0],residuals[1]);

            res0 = temperatureCoefficients[i].A_mean_delta[0][0]*residuals[0] + temperatureCoefficients[i].A_mean_delta[0][1]*residuals[1];
            res0 += temperatureCoefficients[i].B_mean_delta[0][0]*eps[0][j] + temperatureCoefficients[i].B_mean_delta[0][1]*eps[1][j];
            res1 = temperatureCoefficients[i].A_mean_delta[1][0]*residuals[0] + temperatureCoefficients[i].A_mean_delta[1][1]*residuals[1];
            res1 += temperatureCoefficients[i].B_mean_delta[1][0]*eps[0][j] + temperatureCoefficients[i].B_mean_delta[1][1]*eps[1][j];
            ksi[0][j] = residuals[0] = res0;
            ksi[1][j] = residuals[1] = res1;
            //printf("%.1f %.1f\n",eps[0][j],eps[1][j]);
            //printf("%d %.1f %.1f %.1f %.1f\n",i, temperatureCoefficients[i].B[0][0],temperatureCoefficients[i].B[0][1],temperatureCoefficients[i].B[1][0],temperatureCoefficients[i].B[1][1]);
            //printf("%d %.1f %.1f %.1f %.1f\n",i, temperatureCoefficients[i].A[0][0],temperatureCoefficients[i].A[0][1],temperatureCoefficients[i].A[1][0],temperatureCoefficients[i].A[1][1]);
            //residuals[0] = residuals[1]=0;
            //printf("%.1f %.1f\n",residuals[0],residuals[1]);
            //getchar();
        }
        //getchar();
        double** cAverage = (double**)calloc(2, sizeof(double*));
        double** cStdDev = (double**)calloc(2, sizeof(double*));
        double** Xp = (double**)calloc(2, sizeof(double*));

        for (int j=0;j<2;j++)
        {
            cAverage[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            cStdDev[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));
            Xp[j] = (double*)calloc(lengthOfRandomSeries, sizeof(double));

            for (int k=0;k<lengthOfRandomSeries;k++)
            {
                cAverage[j][k] = NODATA;
                cStdDev[j][k] = NODATA;
                Xp[j][k] = NODATA;
            }
        }
        // !!!!!! this part has been changed from the original one. To be verified with authors and colleagues!
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            //X[j] = 1 - X[j]; // to come back to the original code
            cAverage[0][j] = X[j]*averageT[2][j] + (1- X[j])*averageT[0][j]; // for Tmax
            cAverage[1][j] = X[j]*averageT[3][j] + (1- X[j])*averageT[1][j]; // for Tmin
            cStdDev[0][j] = X[j]*stdDevT[2][j] + (1-X[j])*stdDevT[0][j]; // for Tmax
            cStdDev[1][j] = X[j]*stdDevT[3][j] + (1-X[j])*stdDevT[1][j]; // for Tmin
        }
        // !!!!!!
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            if(cStdDev[0][j] >= cStdDev[1][j])
            {
                Xp[1][j] = ksi[1][j]*cStdDev[1][j] + cAverage[1][j];
                Xp[0][j] = ksi[0][j]*sqrt(cStdDev[0][j]*cStdDev[0][j] - cStdDev[1][j]*cStdDev[1][j]) + (cAverage[0][j] - cAverage[1][j]) + Xp[1][j];
            }
            else
            {
                Xp[0][j] = ksi[0][j]*cStdDev[0][j] + cAverage[0][j];
                Xp[1][j] = ksi[1][j]*sqrt(cStdDev[1][j]*cStdDev[1][j] - cStdDev[0][j]*cStdDev[0][j]) - (cAverage[0][j] - cAverage[1][j]) + Xp[0][j];
            }
             //printf("%.1f %.1f\n",ksi[0][j],ksi[1][j]);
        }
        //getchar();

        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            if (Xp[0][j] <= Xp[1][j])
            {
                Xp[1][j] = Xp[0][j] - 0.2*fabs(Xp[0][j]);
            }
        }
        double averageTmean[365]={0};
        double averageTdelta[365]={0};
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            meanTGenerated[j][i] = Xp[0][j];
            deltaTGenerated[j][i] = Xp[1][j];
            occurrencePrecGenerated[j][i] = X[j];
            minTGenerated[j][i] = meanTGenerated[j][i] - 0.5*deltaTGenerated[j][i];
            maxTGenerated[j][i] = meanTGenerated[j][i] + 0.5*deltaTGenerated[j][i];
            averageTmean[j%365] += meanTGenerated[j][i]/parametersModel.yearOfSimulation;
            averageTdelta[j%365] += deltaTGenerated[j][i]/parametersModel.yearOfSimulation;
            //printf("%.1f %d\n",maxTGenerated[j][i],parametersModel.yearOfSimulation);
        }
        //getchar();
        // free memory
        free(ksi[0]);
        free(ksi[1]);
        free(eps[0]);
        free(eps[1]);
        free(cAverage[0]);
        free(cAverage[1]);
        free(cStdDev[0]);
        free(cStdDev[1]);
        free(Xp[0]);
        free(Xp[1]);
        free(ksi);
        free(eps);
        free(cAverage);
        free(cStdDev);
        free(residuals);
        free(Xp);
    }

    for (int i=0;i<4;i++)
    {
        free(averageT[i]);
        free(stdDevT[i]);
    }

    free(averageT);
    free(stdDevT);
    free(X);

    // free class memory
    for (int j=0;j<lengthOfRandomSeries;j++)
    {
        free(multiOccurrenceTemperature[j].occurrence_simulated);
    }
    free(multiOccurrenceTemperature);

}
