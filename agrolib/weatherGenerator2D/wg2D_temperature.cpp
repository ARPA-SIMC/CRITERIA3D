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

}

void weatherGenerator2D::initializeTemperatureVariables()
{
    weatherGenerator2D::initializeMultiOccurrenceTemperature(365*parametersModel.yearOfSimulation);
    weatherGenerator2D::initializeTemperaturesOutput(365*parametersModel.yearOfSimulation);
    weatherGenerator2D::initializeTemperatureParameters();

}

void weatherGenerator2D::computeTemperatureParameters()
{
    for (int iStation=0; iStation<nrStations; iStation++)
    {
        double averageTMaxDry[365];
        double averageTMaxWet[365];
        double stdDevTMaxDry[365];
        double stdDevTMaxWet[365];
        double averageTMinDry[365];
        double averageTMinWet[365];
        double stdDevTMinDry[365];
        double stdDevTMinWet[365];
        int countTMaxDry[365];
        int countTMaxWet[365];
        int countTMinDry[365];
        int countTMinWet[365];

        int finalDay = 365;
        for (int iDay=0;iDay<finalDay;iDay++)
        {
            averageTMaxDry[iDay]=0;
            averageTMaxWet[iDay]=0;
            stdDevTMaxDry[iDay]=0;
            stdDevTMaxWet[iDay]=0;
            averageTMinDry[iDay]=0;
            averageTMinWet[iDay]=0;
            stdDevTMinDry[iDay]=0;
            stdDevTMinWet[iDay]=0;
            countTMaxDry[iDay] = 0;
            countTMaxWet[iDay] = 0;
            countTMinDry[iDay] = 0;
            countTMinWet[iDay] = 0;

        }
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            if(fabs(obsDataD[iStation][iDatum].tMax) > 60) obsDataD[iStation][iDatum].tMax = NODATA;
            if(fabs(obsDataD[iStation][iDatum].tMin) > 60) obsDataD[iStation][iDatum].tMin = NODATA;
        }
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
           if ((fabs((obsDataD[iStation][iDatum].tMax)))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMax += EPSILON;

            }
            if ((fabs(obsDataD[iStation][iDatum].tMin))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMin += EPSILON;
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
        }
        double* rollingAverageTMinDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMinWet = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMaxDry = (double*)calloc(385,sizeof(double));
        double* rollingAverageTMaxWet = (double*)calloc(385,sizeof(double));
        for (int i=0;i<385;i++)
        {
            rollingAverageTMaxDry[i] = NODATA;
            rollingAverageTMinDry[i] = NODATA;
            rollingAverageTMaxWet[i] = NODATA;
            rollingAverageTMinWet[i] = NODATA;
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

        free(rollingAverageTMinDry);
        free(rollingAverageTMinWet);
        free(rollingAverageTMaxDry);
        free(rollingAverageTMaxWet);

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
                }
                else if (obsDataD[iStation][iDatum].prec <= parametersModel.precipitationThreshold)
                {
                    if ((fabs((obsDataD[iStation][iDatum].tMax) - NODATA))> EPSILON)
                    {
                        stdDevTMaxDry[dayOfYear] += (obsDataD[iStation][iDatum].tMax - averageTMaxDry[dayOfYear])*(obsDataD[iStation][iDatum].tMax - averageTMaxDry[dayOfYear]);
                    }
                    if ((fabs((obsDataD[iStation][iDatum].tMin) - NODATA))> EPSILON)
                    {
                        stdDevTMinDry[dayOfYear] += (obsDataD[iStation][iDatum].tMin - averageTMinDry[dayOfYear])*(obsDataD[iStation][iDatum].tMin - averageTMinDry[dayOfYear]);
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

            if (countTMaxDry[iDay] != 0) stdDevTMaxDry[iDay] = sqrt(stdDevTMaxDry[iDay]);
            if (countTMaxWet[iDay] != 0) stdDevTMaxWet[iDay] = sqrt(stdDevTMaxWet[iDay]);
            if (countTMinDry[iDay] != 0) stdDevTMinDry[iDay] = sqrt(stdDevTMinDry[iDay]);
            if (countTMinWet[iDay] != 0) stdDevTMinWet[iDay] = sqrt(stdDevTMinWet[iDay]);
        }

        // compute the Fourier coefficients

        double *par;
        int nrPar = 11;
        par = (double *) calloc(nrPar, sizeof(double));
        /*for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(averageTMaxDry,par,nrPar,temperatureCoefficients[iStation].maxTDry.averageEstimation,365);
        */
        /*
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(averageTMinDry,par,nrPar,temperatureCoefficients[iStation].minTDry.averageEstimation,365);
        */
        /*
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(averageTMaxWet,par,nrPar,temperatureCoefficients[iStation].maxTWet.averageEstimation,365);
        */
        /*
        for (int i=0;i<nrPar;i++)
        {
            par[i] = NODATA;
        }
        weatherGenerator2D::harmonicsFourier(averageTMinWet,par,nrPar,temperatureCoefficients[iStation].minTWet.averageEstimation,365);
        */
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
        // free memory of parameters, variable par[]
        free(par);

        //for (int i=0;i<365;i++)
        //{
            //printf("std %d %.1f %.1f %.1f %.1f\n",iStation, temperatureCoefficients[iStation].minTDry.stdDevEstimation[i],temperatureCoefficients[iStation].minTWet.stdDevEstimation[i],temperatureCoefficients[iStation].maxTDry.stdDevEstimation[i],temperatureCoefficients[iStation].maxTWet.stdDevEstimation[i]);
            //printf("ave %d %.1f %.1f %.1f %.1f\n",iStation, temperatureCoefficients[iStation].minTDry.averageEstimation[i],temperatureCoefficients[iStation].minTWet.averageEstimation[i],temperatureCoefficients[iStation].maxTDry.averageEstimation[i],temperatureCoefficients[iStation].maxTWet.averageEstimation[i]);
        //}
        //getchar();

        weatherGenerator2D::computeResiduals(temperatureCoefficients[iStation].maxTDry.averageEstimation,
                                             temperatureCoefficients[iStation].maxTWet.averageEstimation,
                                             temperatureCoefficients[iStation].maxTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].maxTWet.stdDevEstimation,
                                             temperatureCoefficients[iStation].minTDry.averageEstimation,
                                             temperatureCoefficients[iStation].minTWet.averageEstimation,
                                             temperatureCoefficients[iStation].minTDry.stdDevEstimation,
                                             temperatureCoefficients[iStation].minTWet.stdDevEstimation,iStation);

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
        double ratioLag0 = 0;
        if (matrixCovarianceLag1[1][1] > 0.9)
        {
            ratioLag1 = 0.9/matrixCovarianceLag1[1][1];
            for (int j=0;j<matrixRang;j++)
            {
                for (int k=0;k<matrixRang;k++)
                {
                    matrixCovarianceLag1[j][k] *= ratioLag1;
                }
            }
        }
        if (matrixCovarianceLag0[0][1] > 0.8)
        {
            matrixCovarianceLag0[0][1] = matrixCovarianceLag0[1][0] = 0.8;
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
            }
        }

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
        /*if (fabs(dailyResidual[i].maxT) > EPSILON)
        {
            printf("%d  %f\n",i,dailyResidual[i].minT * dailyResidual[i].maxT);
            getchar();
        }*/

    }
    //printf("%f\n",maxResidual);
    //getchar();
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
}

void weatherGenerator2D::initializeNormalRandomMatricesTemperatures()
{
    int lengthOfRandomSeries;
    lengthOfRandomSeries = parametersModel.yearOfSimulation*365;
    normRandomMaxT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));
    normRandomMinT = (double**)calloc(lengthOfRandomSeries, sizeof(double*));
    for (int i=0;i<lengthOfRandomSeries;i++)
    {
        normRandomMaxT[i] = (double*)calloc(nrStations, sizeof(double));
        normRandomMinT[i] = (double*)calloc(nrStations, sizeof(double));
    }
    for (int i=0; i<nrStations;i++)
    {
        for (int j=0; j<lengthOfRandomSeries;j++)
        {
            normRandomMaxT[j][i] = NODATA;
            normRandomMinT[j][i] = NODATA;
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
    /*for (int iProva=0; iProva<nrStations; iProva++)
    {
        for (int jProva=0; jProva<nrStations; jProva++)
        {
            //printf("%d %d %f\n",iProva, jProva,dummyMatrix[iProva][jProva]);
        }
        //getchar();
    }*/
    isLowerDiagonal = false;
    matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,isLowerDiagonal);
    /*for (int iProva=0; iProva<nrStations; iProva++)
    {
        for (int jProva=0; jProva<nrStations; jProva++)
        {
            //printf("%d %d %f\n",iProva, jProva,dummyMatrix[iProva][jProva]);
        }
        //getchar();
    }*/
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
    // !! this part was added to use "fixed" random numbers
    /*double* arrayRandomNormalNumbers = (double *)calloc(nrStations*lengthOfRandomSeries, sizeof(double));
    randomSet(arrayRandomNormalNumbers,lengthOfRandomSeries);
    int countRandom = 0;
    for (int i=0;i<nrStations;i++)
    {
        //randomSet(arrayRandomNormalNumbers,lengthOfRandomSeries);
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = arrayRandomNormalNumbers[countRandom];
            countRandom++;
            //printf("%f  ",normalizedRandomMatrix[i][j]);
        }
        //printf("\n");
    }*/
    //free(arrayRandomNormalNumbers);
    // fine parte da togliere
    /*for (int iProva=0; iProva<nrStations; iProva++)
    {
        for (int jProva=0; jProva<nrStations; jProva++)
        {
            printf("%d %d %f\n",iProva, jProva,dummyMatrix3[iProva][jProva]);
        }
        getchar();
    }*/
    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomMaxT);
    for (int iProva=0; iProva<nrStations; iProva++)
    {
        for (int jProva=0; jProva<lengthOfRandomSeries; jProva++)
        {
            //printf("%d %d %f\n",iProva, jProva,normRandom2[iProva][jProva]);
        }
        //getchar();
    }


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
            //printf("%f ",correlationMatrixTemperature.minT[i][j]);
        }
        //printf("\n");
    }
    //getchar();
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
    // !! this part was added to use "fixed" random numbers
    //double* arrayRandomNormalNumbers = (double *)calloc(lengthOfRandomSeries, sizeof(double));
    //randomSet(arrayRandomNormalNumbers,lengthOfRandomSeries);
    /*countRandom = 0;
    for (int i=0;i<nrStations;i++)
    {
        //randomSet(arrayRandomNormalNumbers,lengthOfRandomSeries);
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            normRandom[i][j] = arrayRandomNormalNumbers[countRandom];
            countRandom++;
            //printf("%f  ",normalizedRandomMatrix[i][j]);
        }
        //printf("\n");
    }
    free(arrayRandomNormalNumbers); */
    // fine parte da togliere

    matricial::matrixProduct(dummyMatrix3,normRandom,nrStations,nrStations,lengthOfRandomSeries,nrStations,normRandom2);
    matricial::transposedMatrix(normRandom2,nrStations,lengthOfRandomSeries,normRandomMinT);

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
    occurrencePrecGenerated = (double **) calloc(length, sizeof(double *));
    amountsPrecGenerated = (double **) calloc(length, sizeof(double *));
    for (int i=0;i<length;i++)
    {
        maxTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        minTGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        occurrencePrecGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        amountsPrecGenerated[i] = (double *) calloc(nrStations, sizeof(double));
        for(int j=0;j<nrStations;j++)
        {
            maxTGenerated[i][j] = NODATA;
            minTGenerated[i][j] = NODATA;
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
    int day,month;
    int counter = 0;
    for (int j=1;j<=parametersModel.yearOfSimulation;j++)
    {
        for (int i=0; i<365;i++)
        {
            weatherGenerator2D::dateFromDoy(i+1,1,&day,&month); // 1 to avoid leap years
            multiOccurrenceTemperature[counter].year_simulated = j;
            multiOccurrenceTemperature[counter].month_simulated = month;
            multiOccurrenceTemperature[counter].day_simulated = day;
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
        for (int j=0;j<lengthOfRandomSeries;j++)
        {
            maxTGenerated[j][i] = Xp[0][j];
            minTGenerated[j][i] = Xp[1][j];
            occurrencePrecGenerated[j][i] = X[j];
            averageTmax[j%365] += maxTGenerated[j][i]/parametersModel.yearOfSimulation;
            averageTmin[j%365] += minTGenerated[j][i]/parametersModel.yearOfSimulation;
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
