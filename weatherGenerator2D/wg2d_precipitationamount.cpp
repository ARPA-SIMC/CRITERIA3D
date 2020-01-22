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
//#include "gammaFunction.h"
#include "crit3dDate.h"
#include "weatherGenerator.h"
//#include "eispack.h"

void weatherGenerator2D::initializePrecipitationAmountParameters()
{
    precipitationAmount = (Tvariable *)calloc(nrStations, sizeof(Tvariable));
    for (int i = 0; i < nrStations; i++)
    {
        precipitationAmount[i].averageEstimation = (double *)calloc(365, sizeof(double));
        precipitationAmount[i].stdDevEstimation = (double *)calloc(365, sizeof(double));
        for (int j=0; j<365; j++)
        {
            precipitationAmount[i].averageEstimation[j] = NODATA;
            precipitationAmount[i].stdDevEstimation[j] = NODATA;
        }
        /*
        precipitationAmount[i].averageFourierParameters.a0 = NODATA;
        precipitationAmount[i].averageFourierParameters.aCos1 = NODATA;
        precipitationAmount[i].averageFourierParameters.aSin1 = NODATA;
        precipitationAmount[i].averageFourierParameters.aCos2 = NODATA;
        precipitationAmount[i].averageFourierParameters.aSin2 = NODATA;
        precipitationAmount[i].standardDeviationFourierParameters.a0 = NODATA;
        precipitationAmount[i].standardDeviationFourierParameters.aCos1 = NODATA;
        precipitationAmount[i].standardDeviationFourierParameters.aSin1 = NODATA;
        precipitationAmount[i].standardDeviationFourierParameters.aCos2 = NODATA;
        precipitationAmount[i].standardDeviationFourierParameters.aSin2 = NODATA;
        */
    }
}

void weatherGenerator2D::computeprecipitationAmountParameters()
{
    weatherGenerator2D::initializePrecipitationAmountParameters();
    for (int iStation=0; iStation<nrStations; iStation++)
    {
        //double averageAmountPrec[365];
        double stdDevAmountPrec[365];
        int countAmountPrec[365];

        float* averageMonthlyAmountPrec = nullptr;
        averageMonthlyAmountPrec = (float *)calloc(12, sizeof(float));
        double* averageMonthlyAmountPrecLarger = nullptr;
        averageMonthlyAmountPrecLarger = (double *)calloc(16, sizeof(double));
        double* month = nullptr;
        month = (double *)calloc(16, sizeof(double));
        float* averageAmountPrec = nullptr;
        averageAmountPrec = (float *)calloc(365, sizeof(float));
        //double stdDevMonthlyAmountPrec[12];
        int countMonthlyAmountPrec[12];

        int finalDay = 365;
        for (int iMonth=0;iMonth<12;iMonth++)
        {
            averageMonthlyAmountPrec[iMonth] = 0;
            countMonthlyAmountPrec[iMonth] = 0;
        }

        month[0] = -61;
        month[1] = -31;
        month[2] = 0;
        month[3] = 31;
        month[4] = 59;
        month[5] = 90;
        month[6] = 120;
        month[7] = 151;
        month[8] = 181;
        month[9] = 212;
        month[10] = 243;
        month[11] = 273;
        month[12] = 304;
        month[13] = 334;
        month[14] = 365;
        month[15] = 396;

        for (int iMonth=0;iMonth<16;iMonth++)
        {
            month[iMonth] += 15;
        }

        for (int iDay=0;iDay<finalDay;iDay++)
        {

            averageAmountPrec[iDay] = 0;
            stdDevAmountPrec[iDay] = 0;
            countAmountPrec[iDay] = 0;
        }
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            if(fabs(obsDataD[iStation][iDatum].prec) > 600) obsDataD[iStation][iDatum].prec = NODATA;
            if((obsDataD[iStation][iDatum].prec) < 0) obsDataD[iStation][iDatum].prec = NODATA;
        }
        /*for (int iDatum=0; iDatum<nrData; iDatum++)
        {
           if ((fabs((obsDataD[iStation][iDatum].tMax)))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMax += EPSILON;

            }
            if ((fabs(obsDataD[iStation][iDatum].tMin))< EPSILON)
            {
                obsDataD[iStation][iDatum].tMin += EPSILON;
            }
        }*/
        // compute average precipitation of the stations
        for (int iDatum=0; iDatum<nrData; iDatum++)
        {
            //int dayOfYear;
            //dayOfYear = weatherGenerator2D::doyFromDate(obsDataD[iStation][iDatum].date.day,obsDataD[iStation][iDatum].date.month,obsDataD[iStation][iDatum].date.year);
            if (fabs(obsDataD[iStation][iDatum].prec - NODATA) > EPSILON)
            {
                if (obsDataD[iStation][iDatum].prec > parametersModel.precipitationThreshold)
                {
                    if ((fabs((obsDataD[iStation][iDatum].prec)- NODATA))> EPSILON)
                    {
                        ++countMonthlyAmountPrec[obsDataD[iStation][iDatum].date.month-1];
                        averageMonthlyAmountPrec[obsDataD[iStation][iDatum].date.month-1] += obsDataD[iStation][iDatum].prec - parametersModel.precipitationThreshold;
                        //printf("%d %f\n",countMonthlyAmountPrec[obsDataD[iStation][iDatum].date.month-1],averageMonthlyAmountPrec[obsDataD[iStation][iDatum].date.month-1]);
                    }

                }

            }
            //pressEnterToContinue();
        }

        for (int iMonth=0; iMonth<12; iMonth++)
        {
            if (countMonthlyAmountPrec[iMonth] != 0) averageMonthlyAmountPrec[iMonth] /= countMonthlyAmountPrec[iMonth];
            else averageMonthlyAmountPrec[iMonth] = NODATA;
        }

        for (int iMonth=0; iMonth<12; iMonth++)
        {
            averageMonthlyAmountPrecLarger[iMonth+2] = double(averageMonthlyAmountPrec[iMonth]);
        }
        averageMonthlyAmountPrecLarger[0]  = double(averageMonthlyAmountPrec[10]);
        averageMonthlyAmountPrecLarger[1]  = double(averageMonthlyAmountPrec[11]);
        averageMonthlyAmountPrecLarger[14] = double(averageMonthlyAmountPrec[0]);
        averageMonthlyAmountPrecLarger[15]  = double(averageMonthlyAmountPrec[1]);

        //double risultato;
        for (int jjj=0; jjj<365; jjj++)
        {
            int iDay,iMonth;
            iDay = iMonth = 0;
            averageAmountPrec[jjj] = interpolation::cubicSpline(jjj*1.0,month,averageMonthlyAmountPrecLarger,16);

        }

        for (int i=0; i<365 ; i++)
        {
            precipitationAmount[iStation].averageEstimation[i] = double(averageAmountPrec[i]);
        }

        free (averageAmountPrec);
        free (averageMonthlyAmountPrec);
        free (averageMonthlyAmountPrecLarger);
        free (month);
    }
}

void weatherGenerator2D::getSeasonalMeanPrecipitation(int iStation, int iSeason, int length, double* meanPrec)
{
    int index = 0;
    if (iSeason == 0)
    {
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+31];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<28 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+62];
                index++;
            }
        }
    }
    else if (iSeason == 1)
    {
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+90];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<30 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+121];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+151];
                index++;
            }
        }
    }
    else if (iSeason == 2)
    {
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<30 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+182];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+212];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+243];
                index++;
            }
        }
    }
    else if (iSeason == 3)
    {
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<30 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+274];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<31 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+304];
                index++;
            }
        }
        for (int j=0; j<parametersModel.yearOfSimulation ; j++)
        {
            for (int i=0; i<30 ; i++)
            {
                meanPrec[index] = precipitationAmount[iStation].averageEstimation[i+335];
                index++;
            }
        }
    }


}


void weatherGenerator2D::getPrecipitationAmount()
{

    precGenerated = (double **)calloc(nrStations, sizeof(double*));
    for (int i=0;i<nrStations;i++)
    {
        precGenerated[i] = (double *)calloc(365*parametersModel.yearOfSimulation, sizeof(double));
        for (int j=0;j<365*parametersModel.yearOfSimulation;j++)
        {
            precGenerated[i][j] = NODATA;
        }
    }
    /*double*** wMonth;

    wMonth = (double ***)calloc(12, sizeof(double**));
    for (int k=0;k<12;k++)
    {
        wMonth[k] = (double **)calloc(nrStations, sizeof(double*));
        for (int i=0;i<nrStations;i++)
        {
            wMonth[k][i]= (double *)calloc(nrStations, sizeof(double));
            for (int j=0;j<nrStations;j++)
            {
               wMonth[k][i][j] = NODATA;
            }
        }
    }
    for (int k=0;k<12;k++)
    {
        statistics::correlationsMatrix(nrStations,randomMatrix[k].matrixOccurrences,lengthMonth[k]*parametersModel.yearOfSimulation,wMonth[k]);
    }
    */
    weatherGenerator2D::computeprecipitationAmountParameters();



    for (int iMonth=0;iMonth<12;iMonth++)
    {

        int gasDevIset = 0;
        double gasDevGset = 0;
        srand (time(nullptr));
        int firstRandomNumber = rand();
        double** randomMatrixNormalDistributionMonthly = (double **)calloc(nrStations, sizeof(double*));
        double** simulatedPrecipitationAmountsMonthly = (double **)calloc(nrStations, sizeof(double*));
        double** amountCorrelationMatrixMonthSimulated = (double **)calloc(nrStations, sizeof(double*));
        for (int i=0;i<nrStations;i++)
        {
             randomMatrixNormalDistributionMonthly[i] = (double *)calloc(lengthMonth[iMonth]*parametersModel.yearOfSimulation, sizeof(double));
             simulatedPrecipitationAmountsMonthly[i] = (double *)calloc(lengthMonth[iMonth]*parametersModel.yearOfSimulation, sizeof(double));
             amountCorrelationMatrixMonthSimulated[i] = (double *)calloc(nrStations, sizeof(double));
        }

        for (int j=0;j<lengthMonth[iMonth]*parametersModel.yearOfSimulation;j++)
        {
            for (int i=0;i<nrStations;i++)
            {
                 //randomMatrixNormalDistribution[i][j] = myrandom::normalRandomLongSeries(&gasDevIset,&gasDevGset,&firstRandomNumber);
                 randomMatrixNormalDistributionMonthly[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
            }
        }

        weatherGenerator2D::spatialIterationAmountsMonthly(iMonth, amountCorrelationMatrixMonthSimulated , correlationMatrix[iMonth].amount,randomMatrixNormalDistributionMonthly,lengthMonth[iMonth]*parametersModel.yearOfSimulation,randomMatrix[iMonth].matrixOccurrences,simulatedPrecipitationAmountsMonthly);
        for (int j=0;j<lengthMonth[iMonth]*parametersModel.yearOfSimulation;j++)
        {
            for (int i=0;i<nrStations;i++)
            {

                int dayUntilThefirstDayOfTheMonth,counter;
                dayUntilThefirstDayOfTheMonth = counter = 0;
                while (counter < iMonth)
                {
                    dayUntilThefirstDayOfTheMonth += lengthMonth[counter]*parametersModel.yearOfSimulation;
                    counter++;
                }
                precGenerated[i][j+dayUntilThefirstDayOfTheMonth] = simulatedPrecipitationAmountsMonthly[i][j];
            }
        }



        for (int i=0;i<nrStations;i++)
        {
            free(randomMatrixNormalDistributionMonthly[i]);
            free(simulatedPrecipitationAmountsMonthly[i]);
            free(amountCorrelationMatrixMonthSimulated[i]);
        }
        free(randomMatrixNormalDistributionMonthly);
        free(simulatedPrecipitationAmountsMonthly);
        free(amountCorrelationMatrixMonthSimulated);
    }
}


void weatherGenerator2D::spatialIterationAmountsMonthly(int iMonth, double** correlationMatrixSimulatedData,double ** amountsCorrelationMatrix , double** randomMatrix, int lengthSeries, double** occurrences, double** simulatedPrecipitationAmountsMonthly)
{
   double val=5;
   int ii=0;
   double kiter=0.1;
   double** dummyMatrix = (double**)calloc(nrStations, sizeof(double*));
   double** dummyMatrix2 = (double**)calloc(nrStations, sizeof(double*));
   double* correlationArray =(double*)calloc(nrStations*nrStations, sizeof(double));
   double* eigenvalues =(double*)calloc(nrStations, sizeof(double));
   double* eigenvectors =(double*)calloc(nrStations*nrStations, sizeof(double));
   double** dummyMatrix3 = (double**)calloc(nrStations, sizeof(double*));


   //double** normRandom = (double**)calloc(nrStations, sizeof(double*));
   //double** uniformRandom = (double**)calloc(nrStations, sizeof(double*));
   double normRandomVar;
   double uniformRandomVar;
   //double** correlationMatrixSimulatedData = (double**)calloc(nrStations, sizeof(double*));
   double** initialAmountsCorrelationMatrix = (double**)calloc(nrStations, sizeof(double*));


   // initialization internal arrays
   for (int i=0;i<nrStations;i++)
   {
       dummyMatrix[i]= (double*)calloc(nrStations, sizeof(double));
       dummyMatrix2[i]= (double*)calloc(nrStations, sizeof(double));
       //correlationMatrixSimulatedData[i]= (double*)calloc(nrStations, sizeof(double));
       initialAmountsCorrelationMatrix[i]= (double*)calloc(nrStations, sizeof(double));
       for (int j=0;j<nrStations;j++)
       {
           dummyMatrix[i][j]= NODATA;
           dummyMatrix2[i][j]= NODATA;
           //correlationMatrixSimulatedData[i][j]= NODATA;
           initialAmountsCorrelationMatrix[i][j]= NODATA;
       }
   }

   for (int i=0;i<nrStations;i++)
   {
       eigenvalues[i]=NODATA;
       for (int j=0;j<nrStations;j++) eigenvectors[i*nrStations+j] = NODATA;
   }
   for (int i=0;i<nrStations;i++)
   {
       dummyMatrix3[i]= (double*)calloc(lengthSeries, sizeof(double));
       //normRandom[i]= (double*)calloc(lengthSeries, sizeof(double));
       //uniformRandom[i]= (double*)calloc(lengthSeries, sizeof(double));
       for (int j=0;j<lengthSeries;j++)
       {
           dummyMatrix3[i][j]= NODATA;
           //normRandom[i][j]= NODATA;
           //uniformRandom[i][j]= NODATA;
       }

   }

   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<nrStations;j++)
       {
           initialAmountsCorrelationMatrix[i][j] = amountsCorrelationMatrix[i][j];
       }

   }

   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<nrStations;j++)
       {
            //printf("%.4f ",amountsCorrelationMatrix[i][j]);
       }
       //printf("mat \n");
   }
   //pressEnterToContinue();

   double minimalValueToExitFromCycle = NODATA;
   int counterConvergence=0;
   bool exitWhileCycle = false;
   int nrEigenvaluesLessThan0;
   int counter;
   while ((val>TOLERANCE_MULGETS) && (ii<MAX_ITERATION_MULGETS) && (!exitWhileCycle))
   {
       ++ii;
       nrEigenvaluesLessThan0 = 0;
       counter = 0;
       for (int i=0;i<nrStations;i++)
       {
           for (int j=0;j<nrStations;j++) // avoid solutions with correlation coefficient greater than 1
           {
               correlationArray[counter] = amountsCorrelationMatrix[i][j];
               counter++;
           }

       }

       eigenproblem::rs(nrStations,correlationArray,eigenvalues,true,eigenvectors);


       for (int i=0;i<nrStations;i++)
       {
           if (eigenvalues[i] <= 0)
           {
               ++nrEigenvaluesLessThan0;
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
                   ++counter;
               }
           }
           matricial::matrixProductSquareMatricesNoCheck(dummyMatrix,dummyMatrix2,nrStations,amountsCorrelationMatrix);
           for (int i=0;i<nrStations-1;i++)
           {
               dummyMatrix[i][i] = 1.;
               for (int j=i+1;j<nrStations;j++)
               {
                    /*if (i == j)
                    {
                        dummyMatrix[i][j] = 1.;
                    }
                    else
                    {*/
                        dummyMatrix[i][j] = MINVALUE(2*amountsCorrelationMatrix[i][j]/(amountsCorrelationMatrix[i][i]+ amountsCorrelationMatrix[j][j]),ONELESSEPSILON);
                        dummyMatrix[j][i] = dummyMatrix[i][j];
                    //}
               }
            }
            dummyMatrix[nrStations-1][nrStations-1]=1.;
       }
       else
       {
            for (int i=0;i<nrStations;i++)
                for (int j=0;j<nrStations;j++)
                    dummyMatrix[i][j] = amountsCorrelationMatrix[i][j];
       }
       matricial::choleskyDecompositionTriangularMatrix(dummyMatrix,nrStations,true);
       /*for (int i=0;i<nrStations;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
                //printf("%.4f ",dummyMatrix[i][j]);
           }
           //printf(" cholesky \n");
       }*/
       //pressEnterToContinue();
       matricial::matrixProductNoCheck(dummyMatrix, randomMatrix, nrStations, nrStations, lengthSeries, dummyMatrix3);
       /*for (int i=0;i<lengthSeries;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
                //printf("%.4f ",dummyMatrix3[j][i]);
           }
           //printf(" corr_random \n");
       }*/
       //pressEnterToContinue();
       double meanValue,stdDevValue;
       for (int i=0;i<nrStations;i++)
       {
           // compute mean and standard deviation without NODATA check
           meanValue = stdDevValue = 0;
           for (int j=0;j<lengthSeries;j++)
               meanValue += dummyMatrix3[i][j];
           meanValue /= lengthSeries;
           for (int j=0;j<lengthSeries;j++)
               stdDevValue += (dummyMatrix3[i][j]- meanValue)*(dummyMatrix3[i][j]- meanValue);
           stdDevValue /= (lengthSeries-1);
           stdDevValue = sqrt(stdDevValue);

           for (int j=0;j<lengthSeries;j++)
           {
               //normRandomVar= (dummyMatrix3[i][j]-meanValue)/stdDevValue;
               uniformRandomVar =0.5*statistics::tabulatedERFC(-(dummyMatrix3[i][j]-meanValue)/stdDevValue/SQRT_2);
               simulatedPrecipitationAmountsMonthly[i][j]=0.;
               if (occurrences[i][j] > EPSILON)
               {

                        int dayOfYear,counter;
                        //double weibullOutput;
                        dayOfYear = counter = 0;
                        while (counter < iMonth)
                        {
                            dayOfYear += lengthMonth[counter];
                            counter++;
                        }
                        dayOfYear += j%lengthMonth[iMonth];
                        simulatedPrecipitationAmountsMonthly[i][j] = MAXVALUE(parametersModel.precipitationThreshold+EPSILON,0.84 * precipitationAmount[i].averageEstimation[dayOfYear] * pow(-log(uniformRandomVar), 1.3333)) ;
               }
           }
           //printf("\n");
       }
       for (int i=0;i<lengthSeries;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
              //printf("%.4f ",simulatedPrecipitationAmountsMonthly[j][i]);
           }
           //printf("\n");
       }
       //printf("%d\n", ii);
       //pressEnterToContinue();
       for (int i=0;i<nrStations;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
               statistics::correlationsMatrix(nrStations,simulatedPrecipitationAmountsMonthly,lengthSeries,correlationMatrixSimulatedData);
               // da verificare dovrebbe esserci correlazione solo per i dati diversi da zero
               //printf("%.4f ",correlationMatrixSimulatedData[i][j]);
           }
           //printf("\n");
       }
       //pressEnterToContinue();
       val = 0;
       for (int i=0;i<nrStations;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
               val = MAXVALUE(val,fabs(correlationMatrixSimulatedData[i][j] - initialAmountsCorrelationMatrix[i][j]));
           }
       }
       if (val < fabs(minimalValueToExitFromCycle))
       {
           minimalValueToExitFromCycle = val;
           counterConvergence = 0;
       }
       else
       {
           ++counterConvergence;
       }

       if (counterConvergence > 20)
       {
           if (val <= fabs(minimalValueToExitFromCycle) + TOLERANCE_MULGETS) exitWhileCycle = true;
       }

       if (ii != MAX_ITERATION_MULGETS && val> TOLERANCE_MULGETS && (!exitWhileCycle))
       {
           for (int i=0;i<nrStations;i++)
           {
               for (int j=0;j<nrStations;j++)
               {
                   if (i == j)
                   {
                       amountsCorrelationMatrix[i][j]=1.;
                   }
                   else
                   {
                       amountsCorrelationMatrix[i][j] += kiter*(initialAmountsCorrelationMatrix[i][j]-correlationMatrixSimulatedData[i][j]);
                       amountsCorrelationMatrix[i][j] = MINVALUE(amountsCorrelationMatrix[i][j],ONELESSEPSILON);
                   }
               }
           }
       }


   }
   //pressEnterToContinue();
   // free memory
   for (int i=0;i<nrStations;i++)
   {
       free(dummyMatrix[i]);
       free(dummyMatrix2[i]);
       free(dummyMatrix3[i]);
       //free(normRandom[i]);
       //free(uniformRandom[i]);
       free(initialAmountsCorrelationMatrix[i]);
   }


       free(dummyMatrix);
       free(dummyMatrix2);
       free(dummyMatrix3);
       //free(normRandom);
       //free(uniformRandom);
       free(correlationArray);
       free(eigenvalues);
       free(eigenvectors);
       free(initialAmountsCorrelationMatrix);

}
