#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>

#include "wg2D.h"
#include "commonConstants.h"
#include "weatherGenerator.h"
#include "wgClimate.h"

void weatherGenerator2D::initializeOutputData(int* nrDays)
{
    int length = 365*parametersModel.yearOfSimulation;
    for (int i=1; i<= parametersModel.yearOfSimulation;i++)
    {
        if (isLeapYear(i)) length++;
    }
    *nrDays = length;
    outputWeatherData = (ToutputWeatherData*)calloc(nrStations, sizeof(ToutputWeatherData));
    for (int iStation=0;iStation<nrStations;iStation++)
    {
        outputWeatherData[iStation].yearSimulated = (int*)calloc(length, sizeof(int));
        outputWeatherData[iStation].monthSimulated = (int*)calloc(length, sizeof(int));
        outputWeatherData[iStation].daySimulated = (int*)calloc(length, sizeof(int));
        outputWeatherData[iStation].doySimulated = (int*)calloc(length, sizeof(int));
        outputWeatherData[iStation].maxT = (double*)calloc(length, sizeof(double));
        outputWeatherData[iStation].minT = (double*)calloc(length, sizeof(double));
        outputWeatherData[iStation].precipitation = (double*)calloc(length, sizeof(double));
        outputWeatherData[iStation].maxTClimate = (double*)calloc(length, sizeof(double));
        outputWeatherData[iStation].minTClimate = (double*)calloc(length, sizeof(double));
    }
}

void weatherGenerator2D::prepareWeatherGeneratorOutput()
{
    int counter;
    int counterSeason[4];
    int dayCurrent,monthCurrent;
    monthCurrent = dayCurrent = NODATA;
    int nrDays = NODATA;
    weatherGenerator2D::initializeOutputData(&nrDays);
    Crit3DDate inputFirstDate;
    TweatherGenClimate weatherGenClimate;
    QString outputFileName;
   //QString outputFileName2;


    float *inputTMin = nullptr;
    float *inputTMax = nullptr;
    float *inputPrec = nullptr;
    float precThreshold = parametersModel.precipitationThreshold;
    float minPrecData = NODATA;
    bool writeOutput = true;
    inputTMin = (float*)calloc(nrDays, sizeof(float));
    inputTMax = (float*)calloc(nrDays, sizeof(float));
    inputPrec = (float*)calloc(nrDays, sizeof(float));
    inputFirstDate.day = 1;
    inputFirstDate.month = 1;
    inputFirstDate.year = 1;
    float ** monthlySimulatedAveragePrecipitation = nullptr;
    float ** monthlyClimateAveragePrecipitation = nullptr;
    float ** monthlySimulatedAveragePrecipitationInternalFunction = nullptr;
    float ** monthlyClimateAveragePrecipitationInternalFunction = nullptr;
    float** meanAmountsPrecGenerated = nullptr;
    float** cumulatedOccurrencePrecGenerated = nullptr;
    monthlySimulatedAveragePrecipitation = (float**)calloc(nrStations, sizeof(float*));
    monthlyClimateAveragePrecipitation = (float**)calloc(nrStations, sizeof(float*));
    monthlySimulatedAveragePrecipitationInternalFunction = (float**)calloc(nrStations, sizeof(float*));
    monthlyClimateAveragePrecipitationInternalFunction = (float**)calloc(nrStations, sizeof(float*));
    meanAmountsPrecGenerated = (float**)calloc(nrStations, sizeof(float*));
    cumulatedOccurrencePrecGenerated = (float**)calloc(nrStations, sizeof(float*));

    observedConsecutiveDays = (TconsecutiveDays*)calloc(nrStations,sizeof(TconsecutiveDays));
    simulatedConsecutiveDays = (TconsecutiveDays*)calloc(nrStations,sizeof(TconsecutiveDays));

    for (int iStation=0;iStation<nrStations;iStation++)
    {
        meanAmountsPrecGenerated[iStation] = (float*)calloc(12, sizeof(float));
        cumulatedOccurrencePrecGenerated[iStation] = (float*)calloc(12, sizeof(float));
        monthlySimulatedAveragePrecipitation[iStation] = (float*)calloc(12, sizeof(float));
        monthlyClimateAveragePrecipitation[iStation] = (float*)calloc(12, sizeof(float));
        monthlySimulatedAveragePrecipitationInternalFunction[iStation] = (float*)calloc(12, sizeof(float));
        monthlyClimateAveragePrecipitationInternalFunction[iStation] = (float*)calloc(12, sizeof(float));
        for (int j=0;j<12;j++)
        {
            meanAmountsPrecGenerated[iStation][j] = 0;
            cumulatedOccurrencePrecGenerated[iStation][j] = 0;
            monthlySimulatedAveragePrecipitation[iStation][j] = NODATA;
            monthlyClimateAveragePrecipitation[iStation][j] = NODATA;
            monthlySimulatedAveragePrecipitationInternalFunction[iStation][j] = 0;
            monthlyClimateAveragePrecipitationInternalFunction[iStation][j] = 0;
        }
    }
    for (int iStation=0;iStation<nrStations;iStation++)
    {
        for (int iDay=0;iDay<nrData;iDay++)
        {
            if (obsDataD[iStation][iDay].prec > parametersModel.precipitationThreshold)
            {
                meanAmountsPrecGenerated[iStation][obsPrecDataD[iStation][iDay].date.month-1] += obsDataD[iStation][iDay].prec;
                ++cumulatedOccurrencePrecGenerated[iStation][obsPrecDataD[iStation][iDay].date.month-1];
            }
        }
        for (int jMonth=0;jMonth<12;jMonth++)
        {
            meanAmountsPrecGenerated[iStation][jMonth] /= cumulatedOccurrencePrecGenerated[iStation][jMonth];
        }
    }

    for (int iStation=0;iStation<nrStations;iStation++)
    {
        outputFileName = "outputData/wgSimulation_station_" + QString::number(iStation) + ".txt";
        counter = 0;
        counterSeason[3] = counterSeason[2] = counterSeason[1] = counterSeason[0] = 0;
        for (int iYear=1;iYear<=parametersModel.yearOfSimulation;iYear++)
        {
            for (int iDoy=0; iDoy<365; iDoy++)
            {
                outputWeatherData[iStation].yearSimulated[counter] = iYear;
                outputWeatherData[iStation].doySimulated[counter] = iDoy+1;
                weatherGenerator2D::dateFromDoy(outputWeatherData[iStation].doySimulated[counter],1,&dayCurrent,&monthCurrent);
                outputWeatherData[iStation].daySimulated[counter] = dayCurrent;
                outputWeatherData[iStation].monthSimulated[counter] = monthCurrent;
                if (isTempWG2D)
                {
                    outputWeatherData[iStation].maxT[counter] = maxTGenerated[counter][iStation];
                    outputWeatherData[iStation].minT[counter] = minTGenerated[counter][iStation];
                    outputWeatherData[iStation].maxTClimate[counter] = temperatureCoefficients[iStation].maxTDry.averageEstimation[iDoy];
                    outputWeatherData[iStation].minTClimate[counter] = temperatureCoefficients[iStation].minTDry.averageEstimation[iDoy];
                }
                else
                {
                    outputWeatherData[iStation].maxT[counter] = NODATA;
                    outputWeatherData[iStation].minT[counter] = NODATA;

                }

                if (isPrecWG2D)
                {
                    outputWeatherData[iStation].precipitation[counter] = amountsPrecGenerated[counter][iStation];
                }
                else
                {
                    outputWeatherData[iStation].precipitation[counter] = occurrencePrecGenerated[counter][iStation] + 0.1;
                }
                counter++;
            }
        }
        counter = 0;
        for (int i=0;i<nrDays;i++)
        {
            inputTMin[i]= (float)(outputWeatherData[iStation].minT[counter]);
            inputTMax[i]= (float)(outputWeatherData[iStation].maxT[counter]);
            inputPrec[i]= (float)(outputWeatherData[iStation].precipitation[counter]);
            if (isLeapYear(outputWeatherData[iStation].yearSimulated[counter]) && outputWeatherData[iStation].monthSimulated[counter] == 2 && outputWeatherData[iStation].daySimulated[counter] == 28)
            {
                ++i;
                inputTMin[i]= (float)(outputWeatherData[iStation].minT[counter]);
                inputTMax[i]= (float)(outputWeatherData[iStation].maxT[counter]);
                inputPrec[i]= (float)(outputWeatherData[iStation].precipitation[counter]);

            }
            counter++;
        }
        if(computeStatistics)
        {
            computeWG2DClimate(nrDays,inputFirstDate,inputTMin,inputTMax,inputPrec,precThreshold,minPrecData,&weatherGenClimate,false,false,outputFileName,monthlySimulatedAveragePrecipitation[iStation]);
            //for (int iMonth=0;iMonth<12;iMonth++)
                //printf("%d  %d  %f\n",iStation,iMonth,monthlySimulatedAveragePrecipitation[iStation]);
            //getchar();
        }

    }

    free(inputTMin);
    free(inputTMax);
    free(inputPrec);

    precThreshold = float(parametersModel.precipitationThreshold);
    minPrecData = NODATA;
    nrDays = nrData;
    if(computeStatistics)
    {
        int nrConsecutiveDryDays = 0;
        int nrConsecutiveWetDays = 0;
        inputTMin = (float*)calloc(nrDays, sizeof(float));
        inputTMax = (float*)calloc(nrDays, sizeof(float));
        inputPrec = (float*)calloc(nrDays, sizeof(float));
        // compute climate statistics from observed data
        for (int iStation=0;iStation<nrStations;iStation++)
        {
            outputFileName = "outputData/wgClimate_station_" + QString::number(iStation) + ".txt";
            //outputFileName2 = "outputData/statistics_wgClimate_station_" + QString::number(iStation) + ".txt";
            inputFirstDate.day = obsDataD[iStation][0].date.day;
            inputFirstDate.month = obsDataD[iStation][0].date.month;
            inputFirstDate.year = obsDataD[iStation][0].date.year;
            nrDays = nrData;
            nrConsecutiveDryDays = 0;
            nrConsecutiveWetDays = 0;
            for (int i=0;i<nrDays;i++)
            {
                if (obsDataD[iStation][i].prec < precThreshold)
                {
                    nrConsecutiveDryDays++;
                    ++(observedConsecutiveDays[iStation].dry[obsDataD[iStation][i].date.month-1][MINVALUE(nrConsecutiveDryDays,90)]);
                    nrConsecutiveWetDays = 0;
                }
                if (obsDataD[iStation][i].prec >= precThreshold)
                {
                    nrConsecutiveWetDays++;
                    ++(observedConsecutiveDays[iStation].wet[obsDataD[iStation][i].date.month-1][MINVALUE(nrConsecutiveWetDays,90)]);
                    nrConsecutiveDryDays = 0;
                }
            }
            float sumOfEventsDry[12]= {0};
            float sumOfEventsWet[12]= {0};
            for (int iInit=0;iInit<12;iInit++)
            {
                sumOfEventsDry[iInit] = 0;
                sumOfEventsWet[iInit] = 0;
            }

            for (int jMonth=0;jMonth<12;jMonth++)
            {
                //printf("month %d\n",jMonth+1);
                for (int iMonth=0;iMonth<91;iMonth++)
                {
                    sumOfEventsDry[jMonth] += observedConsecutiveDays[iStation].dry[jMonth][iMonth];
                    sumOfEventsWet[jMonth] += observedConsecutiveDays[iStation].wet[jMonth][iMonth];
                    //printf("%.0f,",observedConsecutiveDays[iStation].dry[jMonth][iMonth]);
                }
                //printf("total events %f %f \n",sumOfEventsDry[jMonth],sumOfEventsWet[jMonth]);
                for (int iMonth=0;iMonth<91;iMonth++)
                {
                    observedConsecutiveDays[iStation].dry[jMonth][iMonth] /= sumOfEventsDry[jMonth];
                    observedConsecutiveDays[iStation].wet[jMonth][iMonth] /= sumOfEventsWet[jMonth];
                    //printf("%f,",observedConsecutiveDays[iStation].dry[jMonth][iMonth]);
                }
            }
            //pressEnterToContinue();
            float **consecutiveDry,**consecutiveWet;
            consecutiveDry = (float**)calloc(12, sizeof(float*));
            consecutiveWet = (float**)calloc(12, sizeof(float*));
            for (int jMonth=0;jMonth<12;jMonth++)
            {
                consecutiveDry[jMonth] = (float*)calloc(91, sizeof(float));
                consecutiveWet[jMonth] = (float*)calloc(91, sizeof(float));
            }
            for (int j=0;j<12;j++)
            {
                //printf("month %d\n",j+1);
                for (int i=0;i<91;i++)
                {
                    consecutiveDry[j][i] = observedConsecutiveDays[iStation].dry[j][i];
                    consecutiveWet[j][i] = observedConsecutiveDays[iStation].wet[j][i];
                }
            }


            for (int i=0;i<nrDays;i++)
            {
                inputTMin[i] = obsDataD[iStation][i].tMin;
                inputTMax[i] = obsDataD[iStation][i].tMax;
                inputPrec[i] = obsDataD[iStation][i].prec;
            }
            //computeWG2DClimate(nrDays,inputFirstDate,inputTMin,inputTMax,inputPrec,precThreshold,minPrecData,&weatherGenClimate,writeOutput,true,outputFileName,monthlyClimateAveragePrecipitation[iStation],consecutiveDry,consecutiveWet,12,91);
            computeWG2DClimate(nrDays,inputFirstDate,inputTMin,inputTMax,inputPrec,precThreshold,minPrecData,&weatherGenClimate,writeOutput,false,outputFileName,monthlyClimateAveragePrecipitation[iStation]);
            for (int jMonth=0;jMonth<12;jMonth++)
            {
                free(consecutiveDry[jMonth]);
                free(consecutiveWet[jMonth]);
            }
            free(consecutiveDry);
            free(consecutiveWet);
        }
        free(inputTMin);
        free(inputTMax);
        free(inputPrec);
    }
    weatherGenerator2D::precipitationMonthlyAverage(monthlySimulatedAveragePrecipitationInternalFunction,monthlyClimateAveragePrecipitationInternalFunction);

    for (int iStation=0;iStation<nrStations;iStation++)
    {
        double cumulatedResidual[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        int nrDaysOfPrec[12]={0,0,0,0,0,0,0,0,0,0,0,0};
        for (int iDate=0;iDate<parametersModel.yearOfSimulation*365;iDate++)
        {
            int doy;//,day,month;
            dayCurrent = monthCurrent = 0;
            doy = iDate%365 + 1;
            weatherGenerator2D::dateFromDoy(doy,2001,&dayCurrent,&monthCurrent);
            if (outputWeatherData[iStation].precipitation[iDate] > parametersModel.precipitationThreshold + EPSILON)
            {
                //outputWeatherData[iStation].precipitation[iDate] = MAXVALUE(parametersModel.precipitationThreshold + EPSILON,outputWeatherData[iStation].precipitation[iDate]* monthlyClimateAveragePrecipitationInternalFunction[iStation][monthCurrent-1] / monthlySimulatedAveragePrecipitationInternalFunction[iStation][monthCurrent-1]);
                outputWeatherData[iStation].precipitation[iDate] = outputWeatherData[iStation].precipitation[iDate]* monthlyClimateAveragePrecipitationInternalFunction[iStation][monthCurrent-1] / monthlySimulatedAveragePrecipitationInternalFunction[iStation][monthCurrent-1];
                nrDaysOfPrec[monthCurrent-1]++;
                if (outputWeatherData[iStation].precipitation[iDate] < parametersModel.precipitationThreshold && outputWeatherData[iStation].precipitation[iDate]> EPSILON)
                {
                    cumulatedResidual[monthCurrent-1] += parametersModel.precipitationThreshold + EPSILON - outputWeatherData[iStation].precipitation[iDate];
                    outputWeatherData[iStation].precipitation[iDate] = parametersModel.precipitationThreshold + EPSILON;
                    nrDaysOfPrec[monthCurrent-1]--;
                }
            }

        }
        for (int iDate=0;iDate<parametersModel.yearOfSimulation*365;iDate++)
        {
            int doy;//,day,month;
            dayCurrent = monthCurrent = 0;
            doy = iDate%365 + 1;
            weatherGenerator2D::dateFromDoy(doy,2001,&dayCurrent,&monthCurrent);
            if (outputWeatherData[iStation].precipitation[iDate] > parametersModel.precipitationThreshold + EPSILON && nrDaysOfPrec[monthCurrent-1]>0)
            {
                outputWeatherData[iStation].precipitation[iDate] = MAXVALUE(parametersModel.precipitationThreshold + EPSILON,outputWeatherData[iStation].precipitation[iDate] - cumulatedResidual[monthCurrent-1]/nrDaysOfPrec[monthCurrent-1]);
            }
        }



        free(monthlyClimateAveragePrecipitation[iStation]);
        free(monthlyClimateAveragePrecipitationInternalFunction[iStation]);
    }
    free(monthlyClimateAveragePrecipitation);
    free(monthlyClimateAveragePrecipitationInternalFunction);

    free(observedConsecutiveDays);
    //free(simulatedConsecutiveDays);

    if(computeStatistics)
    {
        nrDays = 365*parametersModel.yearOfSimulation;
        for (int i=1;i<=parametersModel.yearOfSimulation;i++)
        {
            if (isLeapYear(i)) nrDays++;
        }

        inputTMin = (float*)calloc(nrDays, sizeof(float));
        inputTMax = (float*)calloc(nrDays, sizeof(float));
        inputPrec = (float*)calloc(nrDays, sizeof(float));
        inputFirstDate.day = 1;
        inputFirstDate.month = 1;
        inputFirstDate.year = 1;


        for (int iStation=0;iStation<nrStations;iStation++)
        {
            outputFileName = "outputData/wgSimulation_station_" + QString::number(iStation) + ".txt";
            //outputFileName2 = "outputData/statistics_wgSimulation_station_" + QString::number(iStation) + ".txt";
            counter = 0;
            for (int i=0;i<nrDays;i++)
            {



                inputTMin[i]= (float)(outputWeatherData[iStation].minT[counter]);
                inputTMax[i]= (float)(outputWeatherData[iStation].maxT[counter]);
                inputPrec[i]= (float)(outputWeatherData[iStation].precipitation[counter]);
                //printf("%f\n",outputWeatherData[iStation].minT[counter]);
                if (isLeapYear(outputWeatherData[iStation].yearSimulated[counter]) && outputWeatherData[iStation].monthSimulated[counter] == 2 && outputWeatherData[iStation].daySimulated[counter] == 28)
                {
                    ++i;
                    inputTMin[i]= (float)(outputWeatherData[iStation].minT[counter]);
                    inputTMax[i]= (float)(outputWeatherData[iStation].maxT[counter]);
                    inputPrec[i]= (float)(outputWeatherData[iStation].precipitation[counter]);

                }
                counter++;
            }
            /*
            int nrConsecutiveDryDays = 0;
            int nrConsecutiveWetDays = 0;

            for (int iDays=0;iDays<nrDays;iDays++)
            {

                if (outputWeatherData[iStation].precipitation[iDays] < precThreshold)
                {
                    nrConsecutiveDryDays++;
                    if (nrConsecutiveDryDays < 90)
                    {
                        ++(simulatedConsecutiveDays[iStation].dry[outputWeatherData[iStation].monthSimulated[iDays]-1][MINVALUE(nrConsecutiveDryDays,90)]);
                    }
                    nrConsecutiveWetDays = 0;
                }
                if (outputWeatherData[iStation].precipitation[iDays] >= precThreshold)
                {
                    nrConsecutiveWetDays++;
                    if (nrConsecutiveWetDays < 90)
                    {
                        ++(simulatedConsecutiveDays[iStation].wet[outputWeatherData[iStation].monthSimulated[iDays]-1][MINVALUE(nrConsecutiveWetDays,90)]);
                    }
                    nrConsecutiveDryDays = 0;
                }

            }

            double sumOfEventsDry[12]= {0};
            double sumOfEventsWet[12]= {0};

            for (int jMonth=0;jMonth<12;jMonth++)
            {
                //printf("month %d\n",jMonth+1);
                for (int iMonth=0;iMonth<91;iMonth++)
                {
                    sumOfEventsDry[jMonth] += simulatedConsecutiveDays[iStation].dry[jMonth][iMonth];
                    sumOfEventsWet[jMonth] += simulatedConsecutiveDays[iStation].wet[jMonth][iMonth];
                    // printf("%.0f,",simulatedConsecutiveDays[iStation].dry[jMonth][iMonth]);
                }
                //printf("%f,%f\n",sumOfEventsWet[jMonth],sumOfEventsDry[jMonth]);

                for (int iMonth=0;iMonth<91;iMonth++)
                {
                    if (sumOfEventsDry[jMonth] != 0)
                        simulatedConsecutiveDays[iStation].dry[jMonth][iMonth] /= sumOfEventsDry[jMonth];
                    else
                        simulatedConsecutiveDays[iStation].dry[jMonth][iMonth] = 0;
                    if (sumOfEventsWet[jMonth] != 0)
                        simulatedConsecutiveDays[iStation].wet[jMonth][iMonth] /= sumOfEventsWet[jMonth];
                    else
                        simulatedConsecutiveDays[iStation].wet[jMonth][iMonth] = 0;
                }
            }
            //pressEnterToContinue();

            float **consecutiveDry,**consecutiveWet;
            consecutiveDry = (float**)calloc(12, sizeof(float*));
            consecutiveWet = (float**)calloc(12, sizeof(float*));
            for (int jMonth=0;jMonth<12;jMonth++)
            {
                consecutiveDry[jMonth] = (float*)calloc(91, sizeof(float));
                consecutiveWet[jMonth] = (float*)calloc(91, sizeof(float));
            }

            for (int j=0;j<12;j++)
            {
                //printf("month %d\n",j+1);
                for (int i=0;i<91;i++)
                {
                    consecutiveDry[j][i] = simulatedConsecutiveDays[iStation].dry[j][i];
                    consecutiveWet[j][i] = simulatedConsecutiveDays[iStation].wet[j][i];
                }
            }
            */
            //computeWG2DClimate(nrDays,inputFirstDate,inputTMin,inputTMax,inputPrec,precThreshold,minPrecData,&weatherGenClimate,writeOutput,true,outputFileName,monthlySimulatedAveragePrecipitation[iStation],consecutiveDry,consecutiveWet,12,91);
            computeWG2DClimate(nrDays,inputFirstDate,inputTMin,inputTMax,inputPrec,precThreshold,minPrecData,&weatherGenClimate,writeOutput,false,outputFileName,monthlySimulatedAveragePrecipitation[iStation]);
            /*
            for (int jMonth=0;jMonth<12;jMonth++)
            {
                free(consecutiveDry[jMonth]);
                free(consecutiveWet[jMonth]);
            }
            free(consecutiveDry);
            free(consecutiveWet);
            */
            free(monthlySimulatedAveragePrecipitation[iStation]);
            free(monthlySimulatedAveragePrecipitationInternalFunction[iStation]);
            free(meanAmountsPrecGenerated[iStation]);
            free(cumulatedOccurrencePrecGenerated[iStation]);

        }


        free(monthlySimulatedAveragePrecipitation);
        //free(monthlyClimateAveragePrecipitation);
        free(monthlySimulatedAveragePrecipitationInternalFunction);
        //free(monthlyClimateAveragePrecipitationInternalFunction);
        free(meanAmountsPrecGenerated);
        free(cumulatedOccurrencePrecGenerated);
        free(inputTMin);
        free(inputTMax);
        free(inputPrec);


    }
    weatherGenerator2D::precipitationCorrelationMatricesSimulation();
    free(simulatedConsecutiveDays);


    //printf("%f\n",outputWeatherData[3].minT[5]);
}

void weatherGenerator2D::precipitationCorrelationMatricesSimulation()
{
    int counter =0;
    TcorrelationVar amount,occurrence;
    TcorrelationMatrix* correlationMatrixSimulation = nullptr;
    correlationMatrixSimulation = (TcorrelationMatrix*)calloc(12, sizeof(TcorrelationMatrix));
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        correlationMatrixSimulation[iMonth].amount = (double**)calloc(nrStations, sizeof(double*));
        correlationMatrixSimulation[iMonth].occurrence = (double**)calloc(nrStations, sizeof(double*));
        for (int i=0;i<nrStations;i++)
        {
            correlationMatrixSimulation[iMonth].amount[i]= (double*)calloc(nrStations, sizeof(double));
            correlationMatrixSimulation[iMonth].occurrence[i]= (double*)calloc(nrStations, sizeof(double));
            for (int ii=0;ii<nrStations;ii++)
            {
                correlationMatrixSimulation[iMonth].amount[i][ii]= NODATA;
                correlationMatrixSimulation[iMonth].occurrence[i][ii]= NODATA;
            }
        }
    }
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        correlationMatrixSimulation[iMonth].month = iMonth + 1 ; // define the month of the correlation matrix;
        for (int k=0; k<nrStations;k++) // correlation matrix diagonal elements;
        {
            correlationMatrixSimulation[iMonth].amount[k][k] = 1.;
            correlationMatrixSimulation[iMonth].occurrence[k][k]= 1.;
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

                for (int k=0; k<365*parametersModel.yearOfSimulation;k++) // compute the monthly means
                {
                    int doy,dayCurrent,monthCurrent;
                    dayCurrent = monthCurrent = 0;
                    doy = k%365 + 1;
                    weatherGenerator2D::dateFromDoy(doy,2001,&dayCurrent,&monthCurrent);
                    if (monthCurrent == (iMonth+1))
                    {
                        if (((outputWeatherData[j].precipitation[k] - NODATA) > EPSILON) && ((outputWeatherData[i].precipitation[k] - NODATA) > EPSILON))
                        {
                            counter++;
                            if (outputWeatherData[j].precipitation[k] > parametersModel.precipitationThreshold)
                            {
                                amount.meanValue1 += outputWeatherData[j].precipitation[k] ;
                                occurrence.meanValue1++ ;
                            }
                            if (outputWeatherData[i].precipitation[k] > parametersModel.precipitationThreshold)
                            {
                                amount.meanValue2 += outputWeatherData[i].precipitation[k];
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
                for (int k=0; k<365*parametersModel.yearOfSimulation;k++)
                {
                    int doy,dayCurrent,monthCurrent;
                    dayCurrent = monthCurrent = 0;
                    doy = k%365+1;
                    weatherGenerator2D::dateFromDoy(doy,2001,&dayCurrent,&monthCurrent);
                    if (monthCurrent == (iMonth+1))
                    {
                        if ((outputWeatherData[j].precipitation[k] != NODATA) && (outputWeatherData[i].precipitation[k] != NODATA))
                        {
                            double value1,value2;
                            if (outputWeatherData[j].precipitation[k] <= parametersModel.precipitationThreshold) value1 = 0.;
                            else value1 = outputWeatherData[j].precipitation[k];
                            if (outputWeatherData[i].precipitation[k] <= parametersModel.precipitationThreshold) value2 = 0.;
                            else value2 = outputWeatherData[i].precipitation[k];

                            amount.covariance += (value1 - amount.meanValue1)*(value2 - amount.meanValue2);
                            amount.variance1 += (value1 - amount.meanValue1)*(value1 - amount.meanValue1);
                            amount.variance2 += (value2 - amount.meanValue2)*(value2 - amount.meanValue2);

                            if (outputWeatherData[j].precipitation[k] <= parametersModel.precipitationThreshold) value1 = 0.;
                            else value1 = 1.;
                            if (outputWeatherData[i].precipitation[k] <= parametersModel.precipitationThreshold) value2 = 0.;
                            else value2 = 1.;

                            occurrence.covariance += (value1 - occurrence.meanValue1)*(value2 - occurrence.meanValue2);
                            occurrence.variance1 += (value1 - occurrence.meanValue1)*(value1 - occurrence.meanValue1);
                            occurrence.variance2 += (value2 - occurrence.meanValue2)*(value2 - occurrence.meanValue2);
                        }
                    }
                }
                correlationMatrixSimulation[iMonth].amount[j][i]= amount.covariance / sqrt(amount.variance1*amount.variance2);
                correlationMatrixSimulation[iMonth].amount[i][j] = correlationMatrixSimulation[iMonth].amount[j][i];
                correlationMatrixSimulation[iMonth].occurrence[j][i]= occurrence.covariance / sqrt(occurrence.variance1*occurrence.variance2);
                correlationMatrixSimulation[iMonth].occurrence[i][j] = correlationMatrixSimulation[iMonth].occurrence[j][i];
            }
        }

    }
    FILE* fp;
    fp = fopen("outputData/correlationMatricesAmountAnomaly.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nsimulated - observed\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrixSimulation[iMonth].amount[j][i]-correlationMatrix[iMonth].amount[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    fp = fopen("outputData/correlationMatricesAmountSimulation.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nsimulated\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrixSimulation[iMonth].amount[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    fp = fopen("outputData/correlationMatricesAmountObserved.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nobserved\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrix[iMonth].amount[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    fp = fopen("outputData/correlationMatricesAmountMatrixDistribution.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d\n",iMonth+1);
        double bins[20];
        int counterBin[20];
        for (int i=0;i<20;i++)
        {
            counterBin[i] = 0;
        }
        for (int i=0;i<nrStations-1;i++)
        {
            for (int j=i+1;j<nrStations;j++)
            {
                int counter=0;
                double value = 0;
                bool isTheRightBin = false;
                do{
                    value += 0.05;
                    if (counter > 19) isTheRightBin = true;
                    if(!isTheRightBin && correlationMatrix[iMonth].amount[j][i] < value)
                    {
                        counterBin[counter] = counterBin[counter] + 1;
                        isTheRightBin = true;
                    }
                    counter++;

                } while (!isTheRightBin);

            }
        }
        for (int i=0;i<20;i++)
        {
            fprintf(fp,"%f,%d \n", 0.025 + i*0.05, counterBin[i]);
        }

    }

    fclose(fp);

    fp = fopen("outputData/correlationMatricesAmountStats.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        double maxCorrelationAnomaly = 0;
        double minCorrelationAnomaly = 0;
        int occurrenceAnomaly[61];
        for (int i=0;i<61;i++)
        {
            occurrenceAnomaly[i] = 0;
        }
        fprintf(fp,"month %d \n",iMonth+1);
        double value = -0.295;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                maxCorrelationAnomaly = MAXVALUE(maxCorrelationAnomaly,correlationMatrixSimulation[iMonth].amount[j][i]-correlationMatrix[iMonth].amount[j][i]);
                minCorrelationAnomaly = MINVALUE(minCorrelationAnomaly,correlationMatrixSimulation[iMonth].amount[j][i]-correlationMatrix[iMonth].amount[j][i]);
                int counter = 0;
                value = -0.295;
                while (correlationMatrixSimulation[iMonth].amount[j][i]-correlationMatrix[iMonth].amount[j][i]> value && counter<60)
                {
                    value += 0.01;
                    counter++;
                }
                ++occurrenceAnomaly[counter];
            }
        }
        fprintf(fp,"maxValue %f minValue %f\n", maxCorrelationAnomaly, minCorrelationAnomaly);


        for (int i=0;i<61;i++)
        {
            fprintf(fp,"%f,%d \n",-0.3 + i*0.01, occurrenceAnomaly[i]);
        }
    }

    fclose(fp);
       // occurrences
    fp = fopen("outputData/correlationMatricesOccurrenceAnomaly.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nsimulated - observed\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrixSimulation[iMonth].occurrence[j][i]-correlationMatrix[iMonth].occurrence[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    fp = fopen("outputData/correlationMatricesOccurrenceSimulation.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nsimulated\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrixSimulation[iMonth].occurrence[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    fp = fopen("outputData/correlationMatricesOccurrenceObserved.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        fprintf(fp,"month %d \nobserved\n",iMonth+1);
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                fprintf(fp,"%.2f ", correlationMatrix[iMonth].occurrence[j][i]);
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);

    fp = fopen("outputData/correlationMatricesOccurrenceStats.txt","w");
    for (int iMonth=0;iMonth<12;iMonth++)
    {
        double maxCorrelationAnomaly = 0;
        double minCorrelationAnomaly = 0;
        int occurrenceAnomaly[61];
        for (int i=0;i<61;i++)
        {
            occurrenceAnomaly[i] = 0;
        }
        fprintf(fp,"month %d \n",iMonth+1);
        double value = -0.295;
        for (int i=0;i<nrStations;i++)
        {
            for (int j=0;j<nrStations;j++)
            {
                maxCorrelationAnomaly = MAXVALUE(maxCorrelationAnomaly,correlationMatrixSimulation[iMonth].occurrence[j][i]-correlationMatrix[iMonth].occurrence[j][i]);
                minCorrelationAnomaly = MINVALUE(minCorrelationAnomaly,correlationMatrixSimulation[iMonth].occurrence[j][i]-correlationMatrix[iMonth].occurrence[j][i]);
                int counter = 0;
                value = -0.295;
                while (correlationMatrixSimulation[iMonth].occurrence[j][i]-correlationMatrix[iMonth].occurrence[j][i]> value && counter<60)
                {
                    value += 0.01;
                    counter++;
                }
                ++occurrenceAnomaly[counter];
            }
        }
        fprintf(fp,"maxValue %f minValue %f\n", maxCorrelationAnomaly,minCorrelationAnomaly);
        for (int i=0;i<61;i++)
        {
            fprintf(fp,"%f,%d \n",-0.3 + i*0.01, occurrenceAnomaly[i]);
        }
    }

    fclose(fp);


    for (int iMonth=0;iMonth<12;iMonth++)
    {
        for (int i=0;i<nrStations;i++)
        {
            free(correlationMatrixSimulation[iMonth].amount[i]);
            free(correlationMatrixSimulation[iMonth].occurrence[i]);
        }
    }
    free(correlationMatrixSimulation);
}

void weatherGenerator2D::precipitationMonthlyAverage(float** averageSimulation, float** averageClimate)
{
    for (int iStation=0 ; iStation < nrStations ; iStation++)
    {
        int counterMonth[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        int doy,dayCurrent,monthCurrent;
        doy = dayCurrent = monthCurrent = 0;

        for (int i=0; i<365*parametersModel.yearOfSimulation; i++)
        {
            doy = i%365+1;
            weatherGenerator2D::dateFromDoy(doy,2001,&dayCurrent,&monthCurrent);
            if (outputWeatherData[iStation].precipitation[i] > parametersModel.precipitationThreshold)
            {
                averageSimulation[iStation][monthCurrent-1] += outputWeatherData[iStation].precipitation[i];
                ++counterMonth[monthCurrent-1];
            }
        }
        for (int iMonth=0; iMonth<12; iMonth++)
        {
            averageSimulation[iStation][iMonth] /= parametersModel.yearOfSimulation;
        }
    }

    for (int iStation=0 ; iStation < nrStations ; iStation++)
    {
        int counterMonth[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        int monthCurrent = 0;
        int counterNODATA[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        int counterDATA[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        double daysOfMonth[12] = {31,28.25,31,30,31,30,31,31,30,31,30,31};
        for (int i=0; i<nrData; i++)
        {
            monthCurrent = obsDataD[iStation][i].date.month;
            if (!isPrecipitationRecordOK(obsDataD[iStation][i].prec)) ++counterNODATA[monthCurrent-1];
            ++counterDATA[monthCurrent-1];
            if (obsDataD[iStation][i].prec > parametersModel.precipitationThreshold) // including that prec != -9999
            {
                averageClimate[iStation][monthCurrent-1] += obsDataD[iStation][i].prec;
                ++counterMonth[monthCurrent-1];
            }
        }

        for (int iMonth=0; iMonth<12; iMonth++)
        {
            averageClimate[iStation][iMonth] /= (1+obsDataD[iStation][nrData-1].date.year - obsDataD[iStation][0].date.year);
            //averageClimate[iStation][iMonth] /= counterMonth[iMonth];
            //averageClimate[iStation][iMonth] *= daysOfMonth[iMonth];
            if (counterNODATA[iMonth] > 0)
            {
                averageClimate[iStation][iMonth] = averageClimate[iStation][iMonth]*counterDATA[iMonth]/(counterDATA[iMonth] - counterNODATA[iMonth]);
                //averageClimate[iStation][iMonth] *= (1 + counterMonth[iMonth]/(counterDATA[iMonth] - counterNODATA[iMonth]));
            }
        }
    }
}

ToutputWeatherData* weatherGenerator2D::getWeatherGeneratorOutput(int startingYear)
{
    for (int iStation = 0; iStation <nrStations; iStation++)
    {
        int counter = 0;
        for (int iYear=1;iYear<=parametersModel.yearOfSimulation;iYear++)
        {
            for (int iDoy=0; iDoy<365; iDoy++)
            {
                outputWeatherData[iStation].yearSimulated[counter] += startingYear-1;
                counter++;
            }
        }
    }
    return outputWeatherData;
}
