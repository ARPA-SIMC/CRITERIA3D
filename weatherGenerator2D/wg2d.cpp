/*!
    \file wg2D.cpp

    \abstract 2D weather generator

    \note
    The spatial weather generator model is translated from the MulGets model available online on:
    https://it.mathworks.com/matlabcentral/fileexchange/47537-multi-site-stochstic-weather-generator--mulgets-

    \copyright
    This file is part of CRITERIA-3D distribution.
    CRITERIA-3D has been developed by A.R.P.A.E. Emilia-Romagna.

    CRITERIA-3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    CRITERIA-3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA-3D.  If not, see <http://www.gnu.org/licenses/>.

    \authors
    Antonio Volta avolta@arpae.it
    Fausto Tomei ftomei@arpae.it
    Gabriele Antolini gantolini@arpae.it
*/


/*
 *************************************************************************
 Multisite weather generator of the Ecole de technologie superieure
                               (MulGETS)
 *************************************************************************

 MulGETS is Matlab-based multisite weather generator for generating
 spatially correlated daily precipitation, maximum and minimum
 temperatures (Tmax and Tmin) series. The algorithm originates from the
 Wilks approach proposed in 1998. In 2007, Francois Brissette et al.
 presented an algorithm for efficient generation of multisite
 precipitation data following the Wilks approach. Afterward, A minor tune
 was conducted in 2013 after a further evaluation over various climates. A
 component for generating multisite temperature was also added.

 The Matlab code was programmed by Dr. Jie Chen, followed a work version
 provided by Prof. Francois Brissette, at the Ecole de technologie
 superieure, University of Quebec
 Contact email: jie.chen@etsmtl.ca (Jie Chen)

 ****************************
 Input data
 ****************************
 1. The input data consists of daily precipitation, Tmax and Tmin for
 multisite, meteorological data shall be separated by stations with a matlab
 structure named "observation", the data of each station shall be provided
 with the order of year, month, day, Tmax, Tmin and precipitation
 Missing data should be assigned as NaN.

 2. nome_exp: nome esperimento

 varargin	: optional parameters
        'threshold'	    -	soglia per definire un giorno con precipitazione (1 by default)
        'years_sim'  	-	anni simulati (30 by default)
        'dist'			-   tipo di distribuzione 1: Multi-exponential or 2: Multi-gamma (2 by default)
        'DIR'	        -	directory dove lavoro ('mac' by default)
       'soft'          -   software che uso %1=OCTAVE; 2=MATLAB; (2 by default)

 ****************************
 Output data
 ****************************
 The output consists of daily precipitation, Tmax and Tmin, the generated
 meteorological time series is separated by stations with a matlab structure
 named "generation", the order of each station is year, month, day, Tmax,
 Tmin and precipitation

 ****************************
 Example
 ****************************
generation=MulGETS(observation,varargin)

 ****************************
 Generation procedure
 ****************************
 the generation of precipitation includes five steps
 step 1: determination of weather generator parameters: p00, p10 and
         precip distribution function on a monthly scale
 step 2: computation of correlation matrices of precipatation occurrence
         and amounts
 step 3: generate spatially correlated precipitation occurrence
 step 4: establish link between occurrence index and average
         precip amounts for each station and construct the multi-exponential
         or multi-gamma distribution for each station
 step 5: generate precipitation amounts based on the occurrence index of
         generated occurrence

 the generation of temperature includes four steps
 step 1: calculate weather generator parameters for Tmax and Tmin
 step 2: calculate the spatial correlation for observed Tmax and Tmin
 step 3: generate spatial correlated random number
 step 4: generate spatial correlated Tmax and Tmin

 References:
 (1) Wilks, D. S., 1998. Multisite generalization of a daily stochastic
 precipitation generation model. J. Hydrol. 210, 178-191.
 (2) Brissette, F.P., Khalili, M., Leconte, R., 2007. Efficient stochastic
 generation of multi-site sythetic precipitation data. J. Hydrol. 345,
 121-133.
 (3) Chen, J., Brissette, F. P.,Leconte, R., Caron, A., 2012. A versatile
 weather generator for daily precipitation and temperature. Transactions
 of the ASABE. 55(3): 895-906.
 (4) Chen, J., Brissette, F. P.,Zhang, X.-C., 2014. A multi-site stochastic
 weather generator for daily precipitation and temperature. Transaction of
 the ASABE (Accepted)

MULGETS C/C++ version provided by Arpae-SIMC
contributors:
1) Dr. Antonio Volta avolta@arpae.it

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "wg2D.h"
#include "commonConstants.h"
#include "furtherMathFunctions.h"
#include "statistics.h"
#include "eispack.h"
#include "gammaFunction.h"

#include "weatherGenerator.h"
#include "wgClimate.h"


void weatherGenerator2D::initializeBaseWeatherVariables()
{
    month = (int *)calloc(12, sizeof(int));
    for (int i=0; i<12;i++) month[i] = NODATA;
    lengthMonth = (int *)calloc(12, sizeof(int));
    for (int i=0; i<12;i++) lengthMonth[i] = NODATA;
    int monthNumber = 0 ;
    lengthMonth[monthNumber] = 31;
    month[monthNumber] = monthNumber + 1;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 28;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 30;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 30;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 30;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 30;
    monthNumber++;
    month[monthNumber] = monthNumber + 1;
    lengthMonth[monthNumber] = 31;

    lengthSeason[0] = lengthMonth[11]+lengthMonth[0]+lengthMonth[1]; // DJF
    lengthSeason[1] = lengthMonth[2]+lengthMonth[3]+lengthMonth[4];  // MAM
    lengthSeason[2] = lengthMonth[5]+lengthMonth[6]+lengthMonth[7];  // JJA
    lengthSeason[3] = lengthMonth[8]+lengthMonth[9]+lengthMonth[10]; // SON
}


bool weatherGenerator2D::initializeData(int lengthDataSeries, int stations)

{
    nrData = lengthDataSeries;
    nrStations = stations;

    // use of PRAGA formats from meteoPoint.h
    obsDataD = (TObsDataD **)calloc(nrStations, sizeof(TObsDataD*));

    for (int i=0;i<nrStations;i++)
    {
        obsDataD[i] = (TObsDataD *)calloc(nrData, sizeof(TObsDataD));
    }
    // occurrence structure
    precOccurence = (TprecOccurrence **) calloc(nrStations, sizeof(TprecOccurrence*));

    for (int i=0;i<nrStations;i++)
    {
        precOccurence[i] = (TprecOccurrence *)calloc(12, sizeof(TprecOccurrence));
    }
    // correlation matrix structure
    correlationMatrix = (TcorrelationMatrix*)calloc(12, sizeof(TcorrelationMatrix));
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        correlationMatrix[iMonth].amount = (double**)calloc(nrStations, sizeof(double*));
        correlationMatrix[iMonth].occurrence = (double**)calloc(nrStations, sizeof(double*));
        for (int i=0;i<nrStations;i++)
        {
            correlationMatrix[iMonth].amount[i]= (double*)calloc(nrStations, sizeof(double));
            correlationMatrix[iMonth].occurrence[i]= (double*)calloc(nrStations, sizeof(double));
        }
    }

    obsPrecDataD = (TObsPrecDataD **)calloc(nrStations, sizeof(TObsPrecDataD*));
    for (int i=0;i<nrStations;i++)
    {
        obsPrecDataD[i] = (TObsPrecDataD *)calloc(nrData, sizeof(TObsPrecDataD));
    }
    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrData;j++)
        {
            obsPrecDataD[i][j].amounts = NODATA;
            obsPrecDataD[i][j].amountsLessThreshold = NODATA;
            obsPrecDataD[i][j].date.day = NODATA;
            obsPrecDataD[i][j].date.month = NODATA;
            obsPrecDataD[i][j].date.year = NODATA;
            obsPrecDataD[i][j].occurrences = NODATA;
            obsPrecDataD[i][j].prec = NODATA;
        }

    }

    // step 0 of precipitation WG2D initialization of variables
        weatherGenerator2D::initializeBaseWeatherVariables();

    return 0;
}

void weatherGenerator2D::initializeParameters(float thresholdPrecipitation, int simulatedYears, int distributionType, bool computePrecWG2D, bool computeTempWG2D,bool computeStats)
{
    computeStatistics = computeStats;
    isPrecWG2D = computePrecWG2D;
    isTempWG2D = computeTempWG2D;
    // default parameters
    if (fabs(double(thresholdPrecipitation) - NODATA) < EPSILON) parametersModel.precipitationThreshold = 1.; //1 mm is the default
    else parametersModel.precipitationThreshold = double(thresholdPrecipitation);
    if (fabs(simulatedYears - NODATA) < EPSILON) parametersModel.yearOfSimulation = 30;
    else parametersModel.yearOfSimulation = simulatedYears;
    if (fabs(distributionType - NODATA) < EPSILON) parametersModel.distributionPrecipitation = 2; //Select a distribution to generate daily precipitation amount,1: Multi-exponential or 2: Multi-gamma
    else parametersModel.distributionPrecipitation = distributionType;
}

void weatherGenerator2D::setObservedData(TObsDataD** observations)
{
    for(int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrData;j++)
        {
            obsDataD[i][j].date.day = observations[i][j].date.day ;
            obsDataD[i][j].date.month = observations[i][j].date.month;
            obsDataD[i][j].date.year = observations[i][j].date.year;
            obsDataD[i][j].tMin = observations[i][j].tMin;
            obsDataD[i][j].tMax = observations[i][j].tMax;
            obsDataD[i][j].prec = observations[i][j].prec;
        }
    }
}


void weatherGenerator2D::computeWeatherGenerator2D()
{
        weatherGenerator2D::commonModuleCompute();
        if (isTempWG2D)
            weatherGenerator2D::temperatureCompute();
        if (isPrecWG2D)
            weatherGenerator2D::precipitationCompute();
        weatherGenerator2D::prepareWeatherGeneratorOutput();
        //return outputWeatherData;
}

void weatherGenerator2D::commonModuleCompute()
{
    // step 1 of precipitation WG2D
    printf("step 1/9 \n");
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    weatherGenerator2D::precipitationP00P10(); // it computes the monthly probabilities p00 and p10
    printf("step 2/9 \n");
    //time_t rawtime;
    //struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 2 of precipitation WG2D
    weatherGenerator2D::precipitationCorrelationMatrices(); // computation of monthly correlation amongst stations
    printf("step 3/9 \n");
    //time_t rawtime;
    //struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 3 of precipitation WG2D
    weatherGenerator2D::precipitationMultisiteOccurrenceGeneration(); // generation of a sequence of dry/wet days after statistics and random numbers

}

void weatherGenerator2D::temperatureCompute()
{
    // step 1 of temperature WG2D
    printf("step 4/9\n");
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    weatherGenerator2D::computeTemperatureParameters();
    printf("step 5/9\n");
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 2 of temperature WG2D
    weatherGenerator2D::temperaturesCorrelationMatrices();
    printf("step 6/9\n");
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 3 of temperature WG2D
    weatherGenerator2D::multisiteRandomNumbersTemperature();
    printf("step 7/9\n");
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 4 of temperature WG2D
    weatherGenerator2D::multisiteTemperatureGeneration();
    printf("end temperature module\n");
    }

void weatherGenerator2D::precipitationCompute()
{

    weatherGenerator2D::initializePrecipitationInternalArrays();
    weatherGenerator2D::initializePrecipitationOutputs(lengthSeason);
    // step 4 of precipitation WG2D
    printf("step 8/9 \n");
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    weatherGenerator2D::precipitationMultiDistributionParameterization(); // seasonal amounts distribution
    printf("step 9/9\n");
    //time_t rawtime;
    //struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    // step 5 of precipitation WG2D
    if (parametersModel.distributionPrecipitation == 3)
    {
        weatherGenerator2D::getPrecipitationAmount();
    }
    else
    {
        weatherGenerator2D::precipitationMultisiteAmountsGeneration(); // generation of synthetic series
    }
    if (!isPrecWG2D) printf("step 8/9 & 9/9 not computed\n");
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );
    printf("end precipitation module\n");
}

void weatherGenerator2D::initializeRandomNumbers(double *vector)
{
    normalRandomNumbers = (double*)calloc(10000, sizeof(TObsPrecDataD));
    for (int i=0;i<10000;i++)
    {
        normalRandomNumbers[i] = vector[i];
    }
}
void weatherGenerator2D::precipitationP00P10()
{
    // initialization
    for (int idStation=0;idStation<nrStations;idStation++)
    {
        int daysWithoutRain[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        int daysWithRain[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        int occurrence00[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        int occurrence10[12]={0,0,0,0,0,0,0,0,0,0,0,0};

        for(int i=0;i<nrData-1;i++)
        {
            if ((obsDataD[idStation][i].prec != NODATA && obsDataD[idStation][i+1].prec != NODATA))
            {
                for (int month=1;month<13;month++)
                {
                    if(obsDataD[idStation][i].date.month == month)
                    {
                        if (obsDataD[idStation][i].prec > parametersModel.precipitationThreshold)
                        {
                            daysWithRain[month-1]++;
                            if (obsDataD[idStation][i+1].prec <= parametersModel.precipitationThreshold)
                                occurrence10[month-1]++;
                        }
                        else
                        {
                            daysWithoutRain[month-1]++;
                            if (obsDataD[idStation][i+1].prec <= parametersModel.precipitationThreshold)
                                occurrence00[month-1]++;
                        }
                    }
                }
            }
        }

        for (int month=0;month<12;month++)
        {
            if (daysWithoutRain[month] != 0)
                precOccurence[idStation][month].p00 = MINVALUE(ONELESSEPSILON,(double)((1.0*occurrence00[month])/daysWithoutRain[month]));
            else
                precOccurence[idStation][month].p00 = 0.0;
            if (daysWithRain[month] != 0)
                precOccurence[idStation][month].p10 = MINVALUE(ONELESSEPSILON,(double)((1.0*occurrence10[month])/daysWithRain[month]));
            else
                precOccurence[idStation][month].p10 = 0.0;

            precOccurence[idStation][month].month = month +1;
        }
    }

}



void weatherGenerator2D::precipitationCorrelationMatrices()
{
    int counter =0;
    TcorrelationVar amount,occurrence;
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        correlationMatrix[iMonth].month = iMonth + 1 ; // define the month of the correlation matrix;
        for (int k=0; k<nrStations;k++) // correlation matrix diagonal elements;
        {
            correlationMatrix[iMonth].amount[k][k] = 1.;
            correlationMatrix[iMonth].occurrence[k][k]= 1.;
        }

        for (int j=0; j<nrStations-1;j++)
        {
            for (int i=j+1; i<nrStations;i++)
            {
                counter = 0;
                amount.meanValue1=0.;
                amount.meanValue2=0.;
                amount.covariance = amount.variance1 = amount.variance2 = 0.;
                occurrence.meanValue1=0.;
                occurrence.meanValue2=0.;
                occurrence.covariance = occurrence.variance1 = occurrence.variance2 = 0.;

                for (int k=0; k<nrData;k++) // compute the monthly means
                {
                    if (obsDataD[j][k].date.month == (iMonth+1) && obsDataD[i][k].date.month == (iMonth+1))
                    {
                        if (((obsDataD[j][k].prec - NODATA) > EPSILON) && ((obsDataD[i][k].prec - NODATA) > EPSILON))
                        {
                            counter++;
                            if (obsDataD[j][k].prec > parametersModel.precipitationThreshold)
                            {
                                amount.meanValue1 += obsDataD[j][k].prec ;
                                occurrence.meanValue1++ ;
                            }
                            if (obsDataD[i][k].prec > parametersModel.precipitationThreshold)
                            {
                                amount.meanValue2 += obsDataD[i][k].prec;
                                occurrence.meanValue2++ ;
                            }
                        }
                    }
                }
                if (counter != 0)
                {
                    amount.meanValue1 /= counter;
                    occurrence.meanValue1 /= counter;
                }

                if (counter != 0)
                {
                    amount.meanValue2 /= counter;
                    occurrence.meanValue2 /= counter;
                }
                // compute the monthly rho off-diagonal elements
                for (int k=0; k<nrData;k++)
                {
                    if (obsDataD[j][k].date.month == (iMonth+1) && obsDataD[i][k].date.month == (iMonth+1))
                    {
                        if ((obsDataD[j][k].prec != NODATA) && (obsDataD[i][k].prec != NODATA))
                        {
                            double value1,value2;
                            if (obsDataD[j][k].prec <= parametersModel.precipitationThreshold) value1 = 0.;
                            else value1 = obsDataD[j][k].prec;
                            if (obsDataD[i][k].prec <= parametersModel.precipitationThreshold) value2 = 0.;
                            else value2 = obsDataD[i][k].prec;

                            amount.covariance += (value1 - amount.meanValue1)*(value2 - amount.meanValue2);
                            amount.variance1 += (value1 - amount.meanValue1)*(value1 - amount.meanValue1);
                            amount.variance2 += (value2 - amount.meanValue2)*(value2 - amount.meanValue2);

                            if (obsDataD[j][k].prec <= parametersModel.precipitationThreshold) value1 = 0.;
                            else value1 = 1.;
                            if (obsDataD[i][k].prec <= parametersModel.precipitationThreshold) value2 = 0.;
                            else value2 = 1.;

                            occurrence.covariance += (value1 - occurrence.meanValue1)*(value2 - occurrence.meanValue2);
                            occurrence.variance1 += (value1 - occurrence.meanValue1)*(value1 - occurrence.meanValue1);
                            occurrence.variance2 += (value2 - occurrence.meanValue2)*(value2 - occurrence.meanValue2);
                        }
                    }
                }
                correlationMatrix[iMonth].amount[j][i]= amount.covariance / sqrt(amount.variance1*amount.variance2);
                correlationMatrix[iMonth].amount[i][j] = correlationMatrix[iMonth].amount[j][i];
                correlationMatrix[iMonth].occurrence[j][i]= occurrence.covariance / sqrt(occurrence.variance1*occurrence.variance2);
                correlationMatrix[iMonth].occurrence[i][j] = correlationMatrix[iMonth].occurrence[j][i];
            }
        }

    }

}



void weatherGenerator2D::precipitationMultisiteOccurrenceGeneration()
{
    int nrDaysIterativeProcessMonthly[12];
    int gasDevIset = 0;
    double gasDevGset = 0;
    srand(time(NULL));
    rand();

    for (int i=0;i<12;i++)
    {
        nrDaysIterativeProcessMonthly[i] = lengthMonth[i]*parametersModel.yearOfSimulation;
    }
    double** matrixOccurrence;
    matrixOccurrence = (double **)calloc(nrStations, sizeof(double*));
    double** normalizedTransitionProbability;
    normalizedTransitionProbability = (double **)calloc(nrStations, sizeof(double*));

    for (int i=0;i<nrStations;i++)
    {
        matrixOccurrence[i] = (double *)calloc(nrStations, sizeof(double));
        normalizedTransitionProbability[i]= (double *)calloc(2, sizeof(double));
        for (int j=0;j<nrStations;j++)
        {
           matrixOccurrence[i][j]= NODATA;
        }
        normalizedTransitionProbability[i][0]= NODATA;
        normalizedTransitionProbability[i][1]= NODATA;        
    }
    // random Occurrence structure. Used from step 3 on

        randomMatrix = (TrandomMatrix*)calloc(12,sizeof(TrandomMatrix));
        for (int iMonth=0;iMonth<12;iMonth++)
        {
            randomMatrix[iMonth].matrixK = (double**)calloc(nrStations, sizeof(double*));
            randomMatrix[iMonth].matrixM = (double**)calloc(nrStations, sizeof(double*));
            randomMatrix[iMonth].matrixOccurrences = (double**)calloc(nrStations, sizeof(double*));

            for (int i=0;i<nrStations;i++)
            {
                randomMatrix[iMonth].matrixK[i]= (double*)calloc(nrStations, sizeof(double));
                randomMatrix[iMonth].matrixM[i]= (double*)calloc(nrStations, sizeof(double));
                randomMatrix[iMonth].matrixOccurrences[i]= (double*)calloc(nrDaysIterativeProcessMonthly[iMonth], sizeof(double));
            }
        }

    // arrays initialization
    for (int iMonth=0; iMonth<12; iMonth++)
    {
        // initialization and definition of the random matrix
        double** normalizedRandomMatrix;
        normalizedRandomMatrix = (double **)calloc(nrStations, sizeof(double*));
        for (int i=0;i<nrStations;i++)
        {
            normalizedRandomMatrix[i] = (double *)calloc(nrDaysIterativeProcessMonthly[iMonth], sizeof(double));
            for (int j=0;j<nrDaysIterativeProcessMonthly[iMonth];j++)
            {
               normalizedRandomMatrix[i][j]= NODATA;
            }
        }

        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                matrixOccurrence[i][j]= correlationMatrix[iMonth].occurrence[i][j]; //checked
            }

            /* since random numbers generated have a normal distribution, each p00 and
               p10 have to be recalculated according to a normal number*/
            normalizedTransitionProbability[i][0]= - (SQRT_2*(statistics::inverseTabulatedERFC(2*precOccurence[i][iMonth].p00)));
            normalizedTransitionProbability[i][1]= - (SQRT_2*(statistics::inverseTabulatedERFC(2*precOccurence[i][iMonth].p10)));
            for (int jCount=0;jCount<nrDaysIterativeProcessMonthly[iMonth];jCount++)
            {
               normalizedRandomMatrix[i][jCount]= myrandom::normalRandom(&gasDevIset,&gasDevGset);
            }
        }
        // !! questa parte è stata aggiunta per fare uno studio comparativo tra weather generaotr in Matlab e in C usando gli stessi numeri random
        //double* arrayRandomNormalNumbers = (double *)calloc(nrStations*nrDaysIterativeProcessMonthly[iMonth], sizeof(double));
        //randomSet(arrayRandomNormalNumbers,nrStations*nrDaysIterativeProcessMonthly[iMonth]);
        //int countRandom = 0;
        //for (int i=0;i<nrStations;i++)
        //{
            //for (int j=0;j<nrDaysIterativeProcessMonthly[iMonth];j++)
            //{
                //normalizedRandomMatrix[i][j] = arrayRandomNormalNumbers[countRandom];
                //countRandom++;
                //printf("%f  ",normalizedRandomMatrix[i][j]);
            //}
            //printf("\n");
        //}
        //free(arrayRandomNormalNumbers);
        // fine parte da togliere

        // initialization outputs of weatherGenerator2D::spatialIterationOccurrence
        //double** M;
        //double** K;
        //double** occurrences;
        //M = (double **)calloc(nrStations, sizeof(double*));
        //K = (double **)calloc(nrStations, sizeof(double*));
        //occurrences = (double **)calloc(nrStations, sizeof(double*));

        //for (int i=0;i<nrStations;i++)
        //{
            //M[i] = (double *)calloc(nrStations, sizeof(double));
            //K[i] = (double *)calloc(nrStations, sizeof(double));
            //for (int j=0;j<nrStations;j++)
            //{
                //M[i][j]= NODATA;
                //K[i][j]= NODATA;
            //}
        //}
        //for (int i=0;i<nrStations;i++)
        //{
          //occurrences[i] = (double *)calloc(nrDaysIterativeProcessMonthly[iMonth], sizeof(double));
          //for (int j=0;j<nrDaysIterativeProcessMonthly[iMonth];j++)
          //{
              //occurrences[i][j]= NODATA;
          //}
        //}

        weatherGenerator2D::spatialIterationOccurrence(randomMatrix[iMonth].matrixM,randomMatrix[iMonth].matrixK,randomMatrix[iMonth].matrixOccurrences,matrixOccurrence,normalizedRandomMatrix,normalizedTransitionProbability,nrDaysIterativeProcessMonthly[iMonth]);
        randomMatrix[iMonth].month = iMonth + 1;
        // free memory
        for (int i=0;i<nrStations;i++)
        {
            free(normalizedRandomMatrix[i]);            
        }
        free(normalizedRandomMatrix);
        printf("step 3/9 step %d/12\n",iMonth+1);
    }


    // free memory
    for (int i=0;i<nrStations;i++)
    {
        free(matrixOccurrence[i]);
        free(normalizedTransitionProbability[i]);
    }
    free(matrixOccurrence);
    free(normalizedTransitionProbability);

}



void weatherGenerator2D::spatialIterationOccurrence(double ** M, double** K,double** occurrences, double** matrixOccurrence, double** normalizedMatrixRandom,double ** transitionNormal,int lengthSeries)
{

    // M and K matrices are also used as ancillary dummy matrices
    double val = 5;
    int ii = 0;
    double kiter = 0.1;   // iteration parameter in calculation of new estimate of matrix 'mat'
    double* eigenvalues = (double*)calloc(nrStations, sizeof(double));
    double* eigenvectors = (double*)calloc(nrStations*nrStations, sizeof(double));
    double* correlationArray = (double*)calloc(nrStations*nrStations, sizeof(double));
    double** dummyMatrix = (double**)calloc(nrStations, sizeof(double*));
    double** dummyMatrix2 = (double**)calloc(nrStations, sizeof(double*));
    double** dummyMatrix3 = (double**)calloc(nrStations, sizeof(double*));
    double** normRandom = (double**)calloc(nrStations, sizeof(double*));

    // initialization internal arrays
    for (int i=0;i<nrStations;i++)
    {
        dummyMatrix[i]= (double*)calloc(nrStations, sizeof(double));
        dummyMatrix2[i]= (double*)calloc(nrStations, sizeof(double));

    }
    for (int i=0;i<nrStations;i++)
    {
        dummyMatrix3[i]= (double*)calloc(lengthSeries, sizeof(double));
        normRandom[i]= (double*)calloc(lengthSeries, sizeof(double));
    }

    // initialization output M
    for (int i=0;i<nrStations;i++)
    {
       for (int j=0;j<nrStations;j++)
       {
            M[i][j] = MINVALUE(matrixOccurrence[i][j],1);  //MINVALUE(M[i][j],1) M is the matrix named mat in the original code
       }
    }

    double minimalValueToExitFromCycle = NODATA;
    int counterConvergence=0;
    //bool exitWhileCycle = false;
    int nrEigenvaluesLessThan0;
    int counter;
    double meanValue,stdDevValue;
    double myDiff;
    while ((val>TOLERANCE_MULGETS) && (ii<MAX_ITERATION_MULGETS))
    { // !! chiedere a Fausto se vale la pena nelle matrici interne passare da calloc a malloc

        ii++;
        nrEigenvaluesLessThan0 = 0;
        counter = 0;
        /*for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++) // avoid solutions with correlation coefficient greater than 1
            {
                printf("%f  ",M[i][j]);
            }
            printf("\n");
        }
        pressEnterToContinue();*/
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++) // avoid solutions with correlation coefficient greater than 1
            {
                correlationArray[counter] = M[i][j];
                counter++;
            }
        }

        eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors); // !! potrei riformulare rs come rsWeatherGenerator2D senza controlli

        for (int i=0;i<nrStations;i++)
        {
            if (eigenvalues[i] <= 0)
            {
                nrEigenvaluesLessThan0++;
                eigenvalues[i] = 0.000001;
            }
        }
        if (nrEigenvaluesLessThan0 > 0)
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
            int cMatrix, dMatrix, kMatrix;
            double sumMatrix = 0;
            for ( cMatrix = 0 ; cMatrix < nrStations ; cMatrix++ )
            {
                for ( dMatrix = cMatrix ; dMatrix < nrStations ; dMatrix++ )
                {
                    for ( kMatrix = 0 ; kMatrix < nrStations ; kMatrix++ )
                    {
                        sumMatrix += dummyMatrix[cMatrix][kMatrix] * dummyMatrix2[kMatrix][dMatrix];
                    }
                    M[dMatrix][cMatrix] = M[cMatrix][dMatrix] = sumMatrix;
                    sumMatrix = 0.;
                }
            }
            for (int i=0;i<nrStations;i++)
            {
                dummyMatrix[i][i] = 1.;
                for (int j=i+1;j<nrStations;j++)
                {
                         dummyMatrix[i][j] = MINVALUE(2*M[i][j]/(M[i][i] + M[j][j]),ONELESSEPSILON);
                         dummyMatrix[j][i] = dummyMatrix[i][j];
                }
             }

        }
        // the matrix called M is the final matrix exiting from the calculation
        else
        {
            for (int i=0;i<nrStations;i++)
                for (int j=i;j<nrStations;j++)
                {
                    dummyMatrix[j][i] = dummyMatrix[i][j] = M[i][j]; // !! verificare che M sia simmetrica nel caso si può accorciare il for interno
                }
        }

        //bool isLowerDiagonal = true;

        matricial::matrixProductNoCheck(dummyMatrix, normalizedMatrixRandom, nrStations, nrStations, lengthSeries, dummyMatrix3);


        for (int i=0;i<nrStations;i++)
        {
            meanValue = stdDevValue = 0;
            for (int j=0;j<lengthSeries;j++)
                meanValue += dummyMatrix3[i][j];
            meanValue /= lengthSeries;
            for (int j=0;j<lengthSeries;j++)
            {
                myDiff = (dummyMatrix3[i][j]- meanValue);
                stdDevValue += (myDiff)*(myDiff);
            }
            stdDevValue /= (lengthSeries-1);
            stdDevValue = sqrt(stdDevValue);

            for (int j=0;j<lengthSeries;j++)
            {
                normRandom[i][j]= (dummyMatrix3[i][j]-meanValue)/stdDevValue;
            }
        }

        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<lengthSeries;j++)
                occurrences[i][j]= 0.;
        }

        for (int i=0;i<nrStations;i++)
        {
            for (int j=1;j<lengthSeries;j++)
            {
                if(fabs(occurrences[i][j-1]) < EPSILON)
                {
                    if(normRandom[i][j]> transitionNormal[i][0]) occurrences[i][j] = 1.;
                }
                else
                {
                    if(normRandom[i][j]> transitionNormal[i][1]) occurrences[i][j] = 1.;
                }
            }
        }

        statistics::correlationsMatrixNoCheck(nrStations,occurrences,lengthSeries,K);

        val = 0;
        for (int i=0; i<nrStations;i++)
        {
            for (int j=i;j<nrStations;j++)
            {
                val = MAXVALUE(val, fabs(K[i][j] - matrixOccurrence[i][j]));
            }
        }
        if (val < fabs(minimalValueToExitFromCycle))
        {
            minimalValueToExitFromCycle = val;
            counterConvergence = 0;
        }
        else
        {
            counterConvergence++;
        }
        if (counterConvergence > 20)
        {
            if (val <= fabs(minimalValueToExitFromCycle) + TOLERANCE_MULGETS) return;
        }
        for (int i=0; i<nrStations;i++)
        {
            M[i][i]= 1.;
        }
        if ((ii != MAX_ITERATION_MULGETS) && (val > TOLERANCE_MULGETS))
        {
            for (int i=0; i<nrStations;i++)
            {
                for (int j=i+1;j<nrStations;j++)
                {
                    M[i][j] += kiter*(matrixOccurrence[i][j]-K[i][j]);
                    M[j][i] = MINVALUE(M[i][j],ONELESSEPSILON);
                }
            }
        }

    }  // end of the while cycle


    // free memory

    for (int i=0;i<nrStations;i++)
    {
        free(dummyMatrix[i]);
        free(dummyMatrix2[i]);
        free(dummyMatrix3[i]);
        free(normRandom[i]);
    }
    free(normRandom);
    free(dummyMatrix);
    free(dummyMatrix2);
    free(dummyMatrix3);
    free(eigenvalues);
    free(eigenvectors);
    free(correlationArray);

}

int weatherGenerator2D::dateFromDoy(int doy,int year, int* day, int* month)
{
    int daysOfMonth[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (isLeapYear(year)) (daysOfMonth[1])++;
    int counter = 0;
    while(doy > daysOfMonth[counter])
    {
        doy -= daysOfMonth[counter];
        counter++;
        if (counter >= 12) return PARAMETER_ERROR;
    }
    *day = doy;
    *month = ++counter;
    return 0;
}

int weatherGenerator2D::doyFromDate(int day,int month,int year)
{
    int daysOfMonth[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (isLeapYear(year)) (daysOfMonth[1])++;
    if (month == 1)
    {
        return day;
    }
    int doy = 0;
    int counter =0;
    while (counter < (month-1))
    {
        doy += daysOfMonth[counter];
        counter++;
    }
    doy += day;
    return doy;
}



