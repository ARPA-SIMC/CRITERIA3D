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


void weatherGenerator2D::initializePrecipitationInternalArrays()
{

    // create the seasonal correlation matrices
    occurrenceMatrixSeasonDJF = (double **)calloc(nrStations, sizeof(double*));
    occurrenceMatrixSeasonMAM = (double **)calloc(nrStations, sizeof(double*));
    occurrenceMatrixSeasonJJA = (double **)calloc(nrStations, sizeof(double*));
    occurrenceMatrixSeasonSON = (double **)calloc(nrStations, sizeof(double*));
    for (int i=0;i<nrStations;i++)
    {
        occurrenceMatrixSeasonDJF[i] = (double *)calloc(lengthSeason[0]*parametersModel.yearOfSimulation, sizeof(double));
        occurrenceMatrixSeasonMAM[i] = (double *)calloc(lengthSeason[1]*parametersModel.yearOfSimulation, sizeof(double));
        occurrenceMatrixSeasonJJA[i] = (double *)calloc(lengthSeason[2]*parametersModel.yearOfSimulation, sizeof(double));
        occurrenceMatrixSeasonSON[i] = (double *)calloc(lengthSeason[3]*parametersModel.yearOfSimulation, sizeof(double));
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<lengthSeason[0]*parametersModel.yearOfSimulation;j++)
        {
            occurrenceMatrixSeasonDJF[i][j] = NODATA;
        }
        for (int j=0;j<lengthSeason[1]*parametersModel.yearOfSimulation;j++)
        {
            occurrenceMatrixSeasonMAM[i][j] = NODATA;
        }
        for (int j=0;j<lengthSeason[2]*parametersModel.yearOfSimulation;j++)
        {
            occurrenceMatrixSeasonJJA[i][j] = NODATA;
        }
        for (int j=0;j<lengthSeason[3]*parametersModel.yearOfSimulation;j++)
        {
            occurrenceMatrixSeasonSON[i][j] = NODATA;
        }

    }

    wDJF = (double **)calloc(nrStations, sizeof(double*));
    wMAM = (double **)calloc(nrStations, sizeof(double*));
    wJJA = (double **)calloc(nrStations, sizeof(double*));
    wSON = (double **)calloc(nrStations, sizeof(double*));
    wSeason = (double **)calloc(nrStations, sizeof(double*));

    for (int i=0;i<nrStations;i++)
    {
        wDJF[i]= (double *)calloc(nrStations, sizeof(double));
        wMAM[i]= (double *)calloc(nrStations, sizeof(double));
        wJJA[i]= (double *)calloc(nrStations, sizeof(double));
        wSON[i]= (double *)calloc(nrStations, sizeof(double));
        wSeason[i]= (double *)calloc(nrStations, sizeof(double));
    }

    for (int i=0;i<nrStations;i++)
    {
        for (int j=0;j<nrStations;j++)
        {
           wDJF[i][j] = wMAM[i][j] = wJJA[i][j] = wSON[i][j] = wSeason[i][j] = NODATA;
        }
    }

}

void weatherGenerator2D::precipitationMultiDistributionParameterization()
{

   int beginYear,endYear;
   beginYear = endYear = obsDataD[0][0].date.year;
   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<nrData;j++)
       {
           beginYear = MINVALUE(beginYear,obsDataD[i][j].date.year);
           endYear =  MAXVALUE(endYear,obsDataD[i][j].date.year);
       }
   }

   for (int i=0;i<nrStations;i++)
   {
       int counterMonth = 11;
       int nrDaysOfMonth = 0;
       for (int j=0;j<lengthSeason[0]*parametersModel.yearOfSimulation;j++)
       {
               occurrenceMatrixSeasonDJF[i][j] = randomMatrix[counterMonth].matrixOccurrences[i][nrDaysOfMonth];
               nrDaysOfMonth++;
               if (nrDaysOfMonth >= lengthMonth[counterMonth]*parametersModel.yearOfSimulation)
               {
                   counterMonth++;
                   counterMonth = counterMonth%12;
                   nrDaysOfMonth = 0;
               }
       }
   }

   for (int i=0;i<nrStations;i++)
   {
       int counterMonth = 2;
       int nrDaysOfMonth = 0;
       for (int j=0;j<lengthSeason[1]*parametersModel.yearOfSimulation;j++)
       {
               occurrenceMatrixSeasonMAM[i][j] = randomMatrix[counterMonth].matrixOccurrences[i][nrDaysOfMonth];
               nrDaysOfMonth++;
               if (nrDaysOfMonth >= lengthMonth[counterMonth]*parametersModel.yearOfSimulation)
               {
                   counterMonth++;
                   counterMonth = counterMonth%12;
                   nrDaysOfMonth = 0;
               }
       }
   }

   for (int i=0;i<nrStations;i++)
   {
       int counterMonth = 5;
       int nrDaysOfMonth = 0;
       for (int j=0;j<lengthSeason[2]*parametersModel.yearOfSimulation;j++)
       {
               occurrenceMatrixSeasonJJA[i][j] = randomMatrix[counterMonth].matrixOccurrences[i][nrDaysOfMonth];
               nrDaysOfMonth++;
               if (nrDaysOfMonth >= lengthMonth[counterMonth]*parametersModel.yearOfSimulation)
               {
                   counterMonth++;
                   counterMonth = counterMonth%12;
                   nrDaysOfMonth = 0;
               }
       }
   }
   for (int i=0;i<nrStations;i++)
   {
       int counterMonth = 8;
       int nrDaysOfMonth = 0;
       for (int j=0;j<lengthSeason[3]*parametersModel.yearOfSimulation;j++)
       {
               occurrenceMatrixSeasonSON[i][j] = randomMatrix[counterMonth].matrixOccurrences[i][nrDaysOfMonth];
               nrDaysOfMonth++;
               if (nrDaysOfMonth >= lengthMonth[counterMonth]*parametersModel.yearOfSimulation)
               {
                   counterMonth++;
                   counterMonth = counterMonth%12;
                   nrDaysOfMonth = 0;
               }
       }
   }
   statistics::correlationsMatrix(nrStations,occurrenceMatrixSeasonDJF,lengthSeason[0]*parametersModel.yearOfSimulation,wDJF);
   statistics::correlationsMatrix(nrStations,occurrenceMatrixSeasonMAM,lengthSeason[1]*parametersModel.yearOfSimulation,wMAM);
   statistics::correlationsMatrix(nrStations,occurrenceMatrixSeasonJJA,lengthSeason[2]*parametersModel.yearOfSimulation,wJJA);
   statistics::correlationsMatrix(nrStations,occurrenceMatrixSeasonSON,lengthSeason[3]*parametersModel.yearOfSimulation,wSON);


   // initialize amounts and occurrences structures for precipitation
   // !!!!!! if precipitation is equal to threshold it could generate some computational problems
   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<nrData;j++)
       {
          obsPrecDataD[i][j].date.day = obsDataD[i][j].date.day;
          obsPrecDataD[i][j].date.month = obsDataD[i][j].date.month;
          obsPrecDataD[i][j].date.year = obsDataD[i][j].date.year;
          obsPrecDataD[i][j].prec = obsDataD[i][j].prec;
          if (!isPrecipitationRecordOK(obsPrecDataD[i][j].prec))
          {
              obsPrecDataD[i][j].amounts = NODATA;
              obsPrecDataD[i][j].occurrences = NODATA;
              obsPrecDataD[i][j].amountsLessThreshold = NODATA;
          }
          else
          {
              if  (obsPrecDataD[i][j].prec <= parametersModel.precipitationThreshold)
              {
                  obsPrecDataD[i][j].amounts = 0.;
                  obsPrecDataD[i][j].occurrences = 0.;
                  obsPrecDataD[i][j].amountsLessThreshold = 0.;
              }
              else
              {
                  obsPrecDataD[i][j].amounts = obsDataD[i][j].prec;
                  obsPrecDataD[i][j].occurrences = 1.;
                  obsPrecDataD[i][j].amountsLessThreshold = obsPrecDataD[i][j].amounts - parametersModel.precipitationThreshold;
              }
          }
       }
   }
   weatherGenerator2D::initializeOccurrenceIndex();

   double*** moran; // coefficient for each station, each season, each nr of days of simulation in the season.
   moran = (double***)calloc(nrStations, sizeof(double**));
   double*** rainfallLessThreshold; // coefficient for each station, each season, each nr of days of simulation in the season.
   rainfallLessThreshold = (double***)calloc(nrStations, sizeof(double**));


   for (int ijk=0;ijk<nrStations;ijk++)
   {
       numberObservedDJF = numberObservedMAM = numberObservedJJA = numberObservedSON = 0;
       for (int j=0;j<nrData;j++)
       {
           if (obsPrecDataD[ijk][j].date.month == 12 || obsPrecDataD[ijk][j].date.month == 1 || obsPrecDataD[ijk][j].date.month == 2)
               numberObservedDJF++;
           if (obsPrecDataD[ijk][j].date.month == 3 || obsPrecDataD[ijk][j].date.month == 4 || obsPrecDataD[ijk][j].date.month == 5)
               numberObservedMAM++;
           if (obsPrecDataD[ijk][j].date.month == 6 || obsPrecDataD[ijk][j].date.month == 7 || obsPrecDataD[ijk][j].date.month == 8)
               numberObservedJJA++;
           if (obsPrecDataD[ijk][j].date.month == 9 || obsPrecDataD[ijk][j].date.month == 10 || obsPrecDataD[ijk][j].date.month == 11)
               numberObservedSON++;
       }
       numberObservedMax = MAXVALUE(numberObservedDJF,MAXVALUE(numberObservedMAM,MAXVALUE(numberObservedJJA,numberObservedSON)));

       moran[ijk] = (double**)calloc(4, sizeof(double*));
       rainfallLessThreshold[ijk] = (double**)calloc(4, sizeof(double*));
       for (int qq=0;qq<4;qq++)
       {
           moran[ijk][qq] = (double*)calloc(numberObservedMax, sizeof(double));
           rainfallLessThreshold[ijk][qq] = (double*)calloc(numberObservedMax, sizeof(double));
           for (int i=0;i<numberObservedMax;i++)
           {
               // initialize the two vectors
               moran[ijk][qq][i] = NODATA;
               rainfallLessThreshold[ijk][qq][i] = NODATA;
           }
       }
       for (int qq=0;qq<4;qq++)
       {
           //double* occCoeff;
           int monthList[3];
           if (qq == 0)
           {
               monthList[0]= 12;
               monthList[1]= 1;
               monthList[2]= 2;
               for (int j=0;j<nrStations;j++)
               {
                   for (int i=0;i<nrStations;i++)
                       wSeason[i][j]=wDJF[i][j];
               }

           }
           else if (qq == 1)
           {
               monthList[0]= 3;
               monthList[1]= 4;
               monthList[2]= 5;
               for (int j=0;j<nrStations;j++)
               {
                   for (int i=0;i<nrStations;i++)
                       wSeason[i][j]=wMAM[i][j];
               }
           }
           else if (qq == 2)
           {
               monthList[0]= 6;
               monthList[1]= 7;
               monthList[2]= 8;
               for (int j=0;j<nrStations;j++)
               {
                   for (int i=0;i<nrStations;i++)
                       wSeason[i][j]=wJJA[i][j];
               }
           }
           else if (qq == 3)
           {
               monthList[0]= 9;
               monthList[1]= 10;
               monthList[2]= 11;
               for (int j=0;j<nrStations;j++)
               {
                   for (int i=0;i<nrStations;i++)
                       wSeason[i][j]=wSON[i][j];
               }
           }
           int counterData = 0;
           for (int i=0;i<nrData;i++)
           {
               if ((obsPrecDataD[ijk][i].date.month == monthList[0]) || (obsPrecDataD[ijk][i].date.month == monthList[1]) || (obsPrecDataD[ijk][i].date.month == monthList[2]))
               {
                   if (obsPrecDataD[ijk][i].occurrences>0)
                   {
                       double denominatorMoran = 0.;
                       double numeratorMoran = 0.;
                       for (int iStations=0;iStations<nrStations;iStations++)
                       {
                                   if (fabs(obsPrecDataD[iStations][i].occurrences - NODATA) > EPSILON)
                                   {
                                       numeratorMoran += obsPrecDataD[iStations][i].occurrences*wSeason[ijk][iStations];
                                       denominatorMoran += wSeason[ijk][iStations];
                                   }
                       }
                       if (denominatorMoran > EPSILON)
                       {
                           moran[ijk][qq][counterData] = numeratorMoran / denominatorMoran;
                           rainfallLessThreshold[ijk][qq][counterData] = obsPrecDataD[ijk][i].amountsLessThreshold ;
                       }

                       else
                       {
                          moran[ijk][qq][counterData]= 1;
                          rainfallLessThreshold[ijk][qq][counterData] = obsPrecDataD[ijk][i].amountsLessThreshold;
                       }
                   }
                   counterData++;
               }
           }
           int lengthBins = 11;
           double bins[11], bins2[11],bincenter[10];
           int nrBins[10];
           double Pmean[10];
           double PstdDev[10];
           bins[10]= 1.0001;
           for (int i=0;i<10;i++)
           {
               bins[i] = i*0.1;
               bincenter[i] = bins[i] + 0.05;
               nrBins[i] = 0;
               Pmean[i] = 0;
               PstdDev[i] = 0;
           }
           for (int i=0;i<10;i++)
           {
               for (int j=0;j<numberObservedMax;j++)
               {
                   if (moran[ijk][qq][j] >= bins[i] && moran[ijk][qq][j] < bins[i+1])
                   {
                       nrBins[i]++;
                   }
               }
           }
           int counter = 1;

           for (int i=1;i<11;i++)
               bins2[i] = NODATA;
           bins2[0]= 0;
           int nrMinimalPointsForBins = 50;
           for (int i=0;i<9;i++)
           {
               if(nrBins[i] < nrMinimalPointsForBins)
               {
                   nrBins[i+1] += nrBins[i];
               }
               else
               {
                   bins2[counter] = bins[i+1];
                   counter++;
               }
           }

           if (nrBins[9] < nrMinimalPointsForBins)
           {
               --counter;
           }
           bins2[counter] = bins[10];
           int newCounter = 1;
           for (int i=1;i<11;i++)
           {
               bins[i]=bins2[i];
               nrBins[i-1] = 0;
               if (fabs(bins2[i] - NODATA) > EPSILON)
               {
                   bincenter[i-1]= (bins2[i-1] + bins[i])*0.5; //?????
                   newCounter++;
               }
               else
                   bincenter[i-1]= NODATA;

           }

           for (int i=0;i<(newCounter-1);i++)
           {
               for (int j=0;j<numberObservedMax;j++)
               {
                   if (moran[ijk][qq][j] >= bins[i] && moran[ijk][qq][j] < bins[i+1])
                   {
                       nrBins[i]++;
                       Pmean[i] += rainfallLessThreshold[ijk][qq][j];
                   }
               }
               Pmean[i] /= nrBins[i];  // mean for each bin
               if (parametersModel.distributionPrecipitation > 1)
               {
                   for (int j=0;j<numberObservedMax;j++)
                   {
                       if (moran[ijk][qq][j] >= bins[i] && moran[ijk][qq][j] < bins[i+1])
                       {
                           double product;
                           product = rainfallLessThreshold[ijk][qq][j] - Pmean[i];
                           PstdDev[i] += product*product;
                       }
                   }
                   PstdDev[i] = sqrt(PstdDev[i]/(nrBins[i]-1));
               }

           }

           double *parMin;
           double *parMax;
           double *par;
           double *parDelta;
           int maxIterations;
           double epsilon;
           int functionCode;
           int nrPar = 3;
           parMin = (double *) calloc(nrPar, sizeof(double));
           parMax = (double *) calloc(nrPar, sizeof(double));
           par = (double *) calloc(nrPar, sizeof(double));
           parDelta = (double *) calloc(nrPar, sizeof(double));
           parMin[0]= -20.;
           parMax[0]= 20;
           parDelta[0] = 0.01;
           par[0] = (parMin[0]+parMax[0])*0.5;
           parMin[1]= 0;
           parMax[1]= 50.;
           parDelta[1] = 0.0001;
           par[1] = (parMin[1]+parMax[1])*0.5;
           parMin[2]= 1.5;
           parMax[2]= 20.;
           parDelta[2] = 0.01;
           par[2] = (parMin[2]+parMax[2])*0.5;

           maxIterations = 1000000;
           epsilon = 0.0001;
           functionCode = FUNCTION_CODE_TWOPARAMETERSPOLYNOMIAL;
           int nrBincenter=0;
           for (int i=0;i<(lengthBins-1);i++)
           {
               if(fabs(bincenter[i] - NODATA) > EPSILON)
                   nrBincenter++;
           }
           double* meanPFit = (double *) calloc(nrBincenter, sizeof(double));
           double* stdDevFit = (double *) calloc(nrBincenter, sizeof(double));
           double* binCenter = (double *) calloc(nrBincenter, sizeof(double));
           for (int i=0;i<nrBincenter;i++)
           {
              meanPFit[i]=Pmean[i];
              stdDevFit[i]=PstdDev[i];
              binCenter[i]= bincenter[i];
           }

           interpolation::fittingMarquardt(parMin,parMax,par,nrPar,parDelta,maxIterations,epsilon,functionCode,binCenter,meanPFit,nrBincenter);

           // Unless the original Mulgets, in this version we make use of Marquardt algorithm for best fitting
           // of all parameters. For this reason the "for" cycle from
           // from 2 to 20 to get the best exponent has not been translated
           for (int i=0;i<nrBincenter;i++)
           {
               meanPFit[i] = par[0]+par[1]* pow(binCenter[i],par[2]);
           }


           if (parametersModel.distributionPrecipitation >1 )
           {
               interpolation::fittingMarquardt(parMin,parMax,par,nrPar,parDelta,maxIterations,epsilon,functionCode,binCenter,stdDevFit,nrBincenter);

               for (int i=0;i<nrBincenter;i++)
               {
                   stdDevFit[i] = par[0]+par[1]* pow(binCenter[i],par[2]);
               }
           }

           double** occurrenceMatrixSeason = (double **)calloc(nrStations, sizeof(double*));
           double* moranArray = (double *)calloc(lengthSeason[qq]*parametersModel.yearOfSimulation, sizeof(double));
           int counterMoranPrec = 0;
           for(int i=0; i< nrStations;i++)
           {
               occurrenceMatrixSeason[i] = (double *)calloc(lengthSeason[qq]*parametersModel.yearOfSimulation, sizeof(double));
           }

           for(int i=0; i< nrStations;i++)
           {
               for(int j=0; j< parametersModel.yearOfSimulation*lengthSeason[qq];j++)
               {
                   if (qq == 0)
                   {
                       occurrenceMatrixSeason[i][j]= occurrenceMatrixSeasonDJF[i][j];
                   }
                   else if (qq == 1)
                   {
                       occurrenceMatrixSeason[i][j]= occurrenceMatrixSeasonMAM[i][j];
                   }
                   else if (qq == 2)
                   {
                       occurrenceMatrixSeason[i][j]= occurrenceMatrixSeasonJJA[i][j];
                   }
                   else if (qq == 3)
                   {
                       occurrenceMatrixSeason[i][j]= occurrenceMatrixSeasonSON[i][j];
                   }
               }
           }

           for(int i=0; i< parametersModel.yearOfSimulation*lengthSeason[qq];i++)
           {
               double weightSum = 0;
               moranArray[i]= 0;

               for(int j=0; j< nrStations;j++)
               {
                   weightSum += wSeason[ijk][j];
               }

               for(int j=0;j<nrStations;j++)
               {

                   moranArray[i] += occurrenceMatrixSeason[j][i]* wSeason[ijk][j];
               }
               if (fabs(weightSum) < EPSILON) moranArray[i] = 0;
               else moranArray[i] /= weightSum;
               if (occurrenceMatrixSeason[ijk][i] > (ONELESSEPSILON)) counterMoranPrec++ ;

           }
           int indexMoranArrayPrec = 0;
           double* moranArrayPrec = (double *)calloc(counterMoranPrec, sizeof(double));
           double*  frequencyBins = (double *)calloc(counterMoranPrec, sizeof(double));
           int* nrBinsFrequency = (int *)calloc(counterMoranPrec,sizeof(int));
           for(int i=0; i< parametersModel.yearOfSimulation*lengthSeason[qq];i++)
           {
               if (occurrenceMatrixSeason[ijk][i] > (ONELESSEPSILON))
               {
                   moranArrayPrec[indexMoranArrayPrec] = moranArray[i];
                   indexMoranArrayPrec++;
               }
           }

           for (int i=0;i<counterMoranPrec;i++)
           {
                frequencyBins[i] = 0.;
                nrBinsFrequency[i] = 0;
           }

           int counterBins = 0;
           for(int i=0; i<11;i++)
           {
               if (bins[i] > NODATA + EPSILON) counterBins++;
           }
           for(int i=0; i<counterBins-1;i++)
           {
               for (int j=0;j<indexMoranArrayPrec;j++)
               {
                   if (moranArrayPrec[j] >= bins[i] && moranArrayPrec[j] < bins[i+1]) nrBinsFrequency[i]++;
               }
           }
           int nrTotal = 0;
           for(int i=0; i<counterMoranPrec;i++)
           {
               nrTotal += nrBinsFrequency[i];
               frequencyBins[i]=0;
           }

           for(int i=0; i<counterBins-1;i++)
           {
               frequencyBins[i]= (double)(nrBinsFrequency[i])/(double)(nrTotal);

           }

           int numberOfDaysAbovePrecThreshold=0;
           double precipitationMean=0.0;
           for(int i=0; i<numberObservedMax ;i++)
           {
               if(rainfallLessThreshold[ijk][qq][i] > 0)
               {
                   precipitationMean += rainfallLessThreshold[ijk][qq][i];
                   numberOfDaysAbovePrecThreshold++;
               }
           }
           if (numberOfDaysAbovePrecThreshold > 0) precipitationMean /= numberOfDaysAbovePrecThreshold;

           double correctionFactor = 0.;
           double cumulatedWeight=0;
           for(int i=0; i<nrBincenter ;i++)
           {
               cumulatedWeight += frequencyBins[i]*meanPFit[i];
           }
           correctionFactor = precipitationMean / cumulatedWeight;

           double* meanFit = (double *) calloc(nrBincenter, sizeof(double));
           double* lambda = (double *) calloc(nrBincenter, sizeof(double));
           //double* lambda2 = (double *) calloc(nrBincenter, sizeof(double));
           for(int i=0; i<nrBincenter ;i++)
           {
               meanFit[i]= meanPFit[i];
               meanPFit[i] *= correctionFactor;
               if (parametersModel.distributionPrecipitation == 1) lambda[i] = 1.0/meanPFit[i];
           }
           // start to fill the module output
           if (parametersModel.distributionPrecipitation == 1)
           {
               for (int i=0;i<nrBincenter;i++)
                   occurrenceIndexSeasonal[ijk].parMultiexp[qq][i][0]= lambda[i];
           }
           else if (parametersModel.distributionPrecipitation == 2)
           {
               for (int i=0;i<nrBincenter;i++)
               {
                   occurrenceIndexSeasonal[ijk].parMultiexp[qq][i][0]= meanPFit[i]*meanPFit[i]/(PstdDev[i]*PstdDev[i]);
                   occurrenceIndexSeasonal[ijk].parMultiexp[qq][i][1]=(PstdDev[i]*PstdDev[i])/meanPFit[i];
               }
           }
           else if (parametersModel.distributionPrecipitation == 3)
           {
               for (int i=0;i<nrBincenter;i++)
               {

                   double mean,variance;
                   double lambdaWeibull,kappaWeibull;
                   double rightBound,leftBound;
                   rightBound = 10;
                   leftBound = 0.2;
                   mean = meanPFit[i];
                   variance = PstdDev[i]*PstdDev[i];
                   parametersWeibullFromObservations(mean,variance, &lambdaWeibull,&kappaWeibull,leftBound,rightBound);
                   occurrenceIndexSeasonal[ijk].parMultiexp[qq][i][0] = mean;
                   //kappaWeibull = 1.333333333;
                   double testVarianceWeibull,testVarianceWeibull2;
                   testVarianceWeibull = varianceValueWeibull(mean,kappaWeibull);
                   occurrenceIndexSeasonal[ijk].parMultiexp[qq][i][1] = kappaWeibull+0.4;
                   testVarianceWeibull2 = varianceValueWeibull(mean,kappaWeibull+0.4);
                   //printf("%f  %f\n",testVarianceWeibull,testVarianceWeibull2);
                   //pressEnterToContinue();
               }
           }



           for (int i=0;i<nrBincenter;i++)
           {
               occurrenceIndexSeasonal[ijk].meanP[qq][i] = Pmean[i];
               occurrenceIndexSeasonal[ijk].meanFit[qq][i] = meanFit[i];
               occurrenceIndexSeasonal[ijk].binCenter[qq][i] = binCenter[i];
               if (parametersModel.distributionPrecipitation > 1)
               {
                   occurrenceIndexSeasonal[ijk].stdDevP[qq][i] = PstdDev[i];
                   occurrenceIndexSeasonal[ijk].stdDevFit[qq][i] = stdDevFit[i];
               }
           }
           for (int i=0;i<= nrBincenter;i++)
           {
               occurrenceIndexSeasonal[ijk].bin[qq][i] = bins2[i];
           }



           // free memory moran and occCoeff

           for(int i=0; i< nrStations;i++)
           {
               free(occurrenceMatrixSeason[i]);
           }
           free (occurrenceMatrixSeason);
           free (moranArray);
           free (par);
           free(parDelta);
           free(parMax);
           free(parMin);
           free(meanPFit);
           free(stdDevFit);
           free(binCenter);
           free(meanFit);
           free(lambda);
           free(frequencyBins);
           free(nrBinsFrequency);
       }

   }

   // free the memory step 4

   for (int i=0;i<nrStations;i++)
   {
       for (int qq=0;qq<4;qq++)
       {
           free(moran[i][qq]);
           free(rainfallLessThreshold[i][qq]);
       }
       free(moran[i]);
       free(rainfallLessThreshold[i]);
   }
   free(moran);
   free(rainfallLessThreshold);


}

void weatherGenerator2D::precipitationMultisiteAmountsGeneration()
{
    // begin of step 5 of the original weather generator
   double** amountMatrixSeasonDJF = (double **)calloc(nrStations, sizeof(double*));
   double** amountMatrixSeasonMAM = (double **)calloc(nrStations, sizeof(double*));
   double** amountMatrixSeasonJJA = (double **)calloc(nrStations, sizeof(double*));
   double** amountMatrixSeasonSON = (double **)calloc(nrStations, sizeof(double*));

   for (int i=0;i<nrStations;i++)
   {
       amountMatrixSeasonDJF[i] = (double *)calloc(numberObservedMax, sizeof(double));
       amountMatrixSeasonMAM[i] = (double *)calloc(numberObservedMax, sizeof(double));
       amountMatrixSeasonJJA[i] = (double *)calloc(numberObservedMax, sizeof(double));
       amountMatrixSeasonSON[i] = (double *)calloc(numberObservedMax, sizeof(double));
   }

   double** amountCorrelationMatrixDJF = (double **)calloc(nrStations, sizeof(double*));
   double** amountCorrelationMatrixMAM = (double **)calloc(nrStations, sizeof(double*));
   double** amountCorrelationMatrixJJA = (double **)calloc(nrStations, sizeof(double*));
   double** amountCorrelationMatrixSON = (double **)calloc(nrStations, sizeof(double*));

   for (int i=0;i<nrStations;i++)
   {
       amountCorrelationMatrixDJF[i] = (double *)calloc(nrStations, sizeof(double));
       amountCorrelationMatrixMAM[i] = (double *)calloc(nrStations, sizeof(double));
       amountCorrelationMatrixJJA[i] = (double *)calloc(nrStations, sizeof(double));
       amountCorrelationMatrixSON[i] = (double *)calloc(nrStations, sizeof(double));
   }

   int counterDJF, counterMAM, counterJJA, counterSON;
    counterDJF = counterJJA = counterMAM = counterSON = 0;
   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<numberObservedMax;j++)
       {
           amountMatrixSeasonDJF[i][j] = NODATA;
           amountMatrixSeasonMAM[i][j] = NODATA;
           amountMatrixSeasonJJA[i][j] = NODATA;
           amountMatrixSeasonSON[i][j] = NODATA;
       }

       counterDJF = counterJJA = counterMAM = counterSON = 0;
       for (int j=0;j<nrData;j++)
       {
           if (obsPrecDataD[i][j].date.month == 12 || obsPrecDataD[i][j].date.month == 1 || obsPrecDataD[i][j].date.month == 2)
           {
               if (obsPrecDataD[i][j].amounts > parametersModel.precipitationThreshold) amountMatrixSeasonDJF[i][counterDJF] = obsPrecDataD[i][j].amounts;
               else amountMatrixSeasonDJF[i][counterDJF] = 0.;
               counterDJF++;
           }
           if (obsPrecDataD[i][j].date.month == 3 || obsPrecDataD[i][j].date.month == 4 || obsPrecDataD[i][j].date.month == 5)
           {
               if (obsPrecDataD[i][j].amounts > parametersModel.precipitationThreshold) amountMatrixSeasonMAM[i][counterMAM] = obsPrecDataD[i][j].amounts;
               else amountMatrixSeasonMAM[i][counterMAM] = 0.;
               counterMAM++;
           }
           if (obsPrecDataD[i][j].date.month == 6 || obsPrecDataD[i][j].date.month == 7 || obsPrecDataD[i][j].date.month == 8)
           {
               if (obsPrecDataD[i][j].amounts > parametersModel.precipitationThreshold)amountMatrixSeasonJJA[i][counterJJA] = obsPrecDataD[i][j].amounts;
               else amountMatrixSeasonJJA[i][counterJJA] = 0.;
               counterJJA++;
           }
           if (obsPrecDataD[i][j].date.month == 9 || obsPrecDataD[i][j].date.month == 10 || obsPrecDataD[i][j].date.month == 11)
           {
               if (obsPrecDataD[i][j].amounts > parametersModel.precipitationThreshold) amountMatrixSeasonSON[i][counterSON] = obsPrecDataD[i][j].amounts;
               else amountMatrixSeasonSON[i][counterSON] = 0.;
               counterSON++;
           }

       }

   }
   statistics::correlationsMatrix(nrStations,amountMatrixSeasonDJF,numberObservedMax,amountCorrelationMatrixDJF);
   statistics::correlationsMatrix(nrStations,amountMatrixSeasonMAM,numberObservedMax,amountCorrelationMatrixMAM);
   statistics::correlationsMatrix(nrStations,amountMatrixSeasonJJA,numberObservedMax,amountCorrelationMatrixJJA);
   statistics::correlationsMatrix(nrStations,amountMatrixSeasonSON,numberObservedMax,amountCorrelationMatrixSON);

   int gasDevIset = 0;
   double gasDevGset = 0;
   srand(unsigned(time(nullptr)));

   for (int iSeason=0;iSeason<4;iSeason++)
   {
      double** occurrenceSeason = (double **)calloc(nrStations, sizeof(double*));
      double** moranRandom = (double **)calloc(nrStations, sizeof(double*));
      for (int i=0;i<nrStations;i++)
      {
          occurrenceSeason[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
          moranRandom[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
      }
      for (int i=0;i<nrStations;i++)
      {
          for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
          {
              occurrenceSeason[i][j]= NODATA;
              moranRandom[i][j] = NODATA;
          }
      }

      if (iSeason == 0)
      {
          for (int i=0;i<nrStations;i++)
          {
              for (int j=0;j<nrStations;j++)
              {
                  wSeason[i][j] = wDJF[i][j];
                  simulatedPrecipitationAmounts[iSeason].matrixM[i][j]= amountCorrelationMatrixDJF[i][j];
              }
              for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
              {
                   occurrenceSeason[i][j] = occurrenceMatrixSeasonDJF[i][j];
              }
          }

      }
      if (iSeason == 1)
      {
          for (int i=0;i<nrStations;i++)
          {
              for (int j=0;j<nrStations;j++)
              {
                  wSeason[i][j] = wMAM[i][j];
                  simulatedPrecipitationAmounts[iSeason].matrixM[i][j]= amountCorrelationMatrixMAM[i][j];
              }
              for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
              {
                   occurrenceSeason[i][j] = occurrenceMatrixSeasonMAM[i][j];

              }
          }
      }
      if (iSeason == 2)
      {
          for (int i=0;i<nrStations;i++)
          {
              for (int j=0;j<nrStations;j++)
              {
                  wSeason[i][j] = wJJA[i][j];
                  simulatedPrecipitationAmounts[iSeason].matrixM[i][j]= amountCorrelationMatrixJJA[i][j];
              }
              for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
              {
                   occurrenceSeason[i][j] = occurrenceMatrixSeasonJJA[i][j];
              }
          }
      }
      if (iSeason == 3)
      {
          for (int i=0;i<nrStations;i++)
          {
              for (int j=0;j<nrStations;j++)
              {
                  wSeason[i][j] = wSON[i][j];
                  simulatedPrecipitationAmounts[iSeason].matrixM[i][j]= amountCorrelationMatrixSON[i][j];
              }
              for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
              {
                   occurrenceSeason[i][j] = occurrenceMatrixSeasonSON[i][j];
              }
          }
      }
      for (int iStations=0;iStations<nrStations;iStations++)
      {
                  for (int i=0;i<lengthSeason[iSeason]*parametersModel.yearOfSimulation;i++)
                  {

                      if (occurrenceSeason[iStations][i] > EPSILON)
                      {
                          double denominatorMoran = 0.;
                          double numeratorMoran = 0.;
                          for (int jStations=0;jStations<nrStations;jStations++)
                          {
                                numeratorMoran += occurrenceSeason[jStations][i]*wSeason[nrStations-1][jStations];
                                denominatorMoran += wSeason[nrStations-1][jStations];
                          }
                          if (fabs(denominatorMoran) > EPSILON)
                          {
                              moranRandom[iStations][i] = numeratorMoran / denominatorMoran;
                          }
                          else
                          {
                             moranRandom[iStations][i]= 1;
                          }

                      }
                  }

      }


      double** phatAlpha = (double **)calloc(nrStations, sizeof(double*));
      double** phatBeta = (double **)calloc(nrStations, sizeof(double*));
      double** randomMatrixNormalDistribution = (double **)calloc(nrStations, sizeof(double*));
      double** simulatedPrecipitationAmountsSeasonal = (double **)calloc(nrStations, sizeof(double*));

      for (int i=0;i<nrStations;i++)
      {
           phatAlpha[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
           phatBeta[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
           randomMatrixNormalDistribution[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
           simulatedPrecipitationAmountsSeasonal[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
      }

      for (int i=0;i<nrStations;i++)
      {
          for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
          {
             simulatedPrecipitationAmountsSeasonal[i][j]= NODATA;
          }
      }

      for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
      {
          for (int i=0;i<nrStations;i++)
          {
               for (int k=0;k<10;k++) //to improve
               {
                   if ((moranRandom[i][j] > occurrenceIndexSeasonal[i].bin[iSeason][k]) && (moranRandom[i][j] <= occurrenceIndexSeasonal[i].bin[iSeason][k+1]))
                   {
                       phatAlpha[i][j] = occurrenceIndexSeasonal[i].parMultiexp[iSeason][k][0];
                       if (parametersModel.distributionPrecipitation > 1)
                       {
                           phatBeta[i][j] = occurrenceIndexSeasonal[i].parMultiexp[iSeason][k][1];
                       }
                   }
               }
          }
     }
    /*
     if (parametersModel.distributionPrecipitation == 3)
     {
         for (int i=0;i<nrStations;i++)
         {
            for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
            {
                 if (iSeason == 0)
                 {
                    int indexDay;
                    indexDay = lengthSeason[iSeason]*parametersModel.yearOfSimulation;
                    indexDay = (j%lengthSeason[iSeason])-31;
                    if (indexDay < 0) indexDay += 365;
                    phatAlpha[i][j] = weibullDailyParameterLambda[i][indexDay];
                    phatBeta[i][j] = weibullDailyParameterKappa[i][indexDay];
                 }
                 if (iSeason == 1)
                 {
                    int indexDay;
                    indexDay = (j%lengthSeason[iSeason])+59;

                    phatAlpha[i][j] = weibullDailyParameterLambda[i][indexDay];
                    phatBeta[i][j] = weibullDailyParameterKappa[i][indexDay];
                 }
                 if (iSeason == 2)
                 {
                    int indexDay;
                    indexDay = (j%lengthSeason[iSeason])+151;

                    phatAlpha[i][j] = weibullDailyParameterLambda[i][indexDay];
                    phatBeta[i][j] = weibullDailyParameterKappa[i][indexDay];
                 }
                 if (iSeason == 3)
                 {
                    int indexDay;
                    indexDay = (j%lengthSeason[iSeason])+243;

                    phatAlpha[i][j] = weibullDailyParameterLambda[i][indexDay];
                    phatBeta[i][j] = weibullDailyParameterKappa[i][indexDay];
                 }

             }
          }
      }
    */
      for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
      {
          for (int i=0;i<nrStations;i++)
          {
               randomMatrixNormalDistribution[i][j] = myrandom::normalRandom(&gasDevIset,&gasDevGset);
          }
      }

      weatherGenerator2D::spatialIterationAmounts(simulatedPrecipitationAmounts[iSeason].matrixK , simulatedPrecipitationAmounts[iSeason].matrixM,randomMatrixNormalDistribution,lengthSeason[iSeason]*parametersModel.yearOfSimulation,occurrenceSeason,phatAlpha,phatBeta,simulatedPrecipitationAmountsSeasonal);
      time_t rawtime;
      struct tm * timeinfo;
      time ( &rawtime );
      timeinfo = localtime ( &rawtime );
      printf ( "Current local time and date: %s", asctime (timeinfo) );
      printf("step 9/9 substep %d/4\n",iSeason+1);
      for (int i=0;i<nrStations;i++)
      {
           for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
           {
               simulatedPrecipitationAmounts[iSeason].matrixAmounts[i][j]= simulatedPrecipitationAmountsSeasonal[i][j];

           }
      }

      // free memory
      for (int i=0;i<nrStations;i++)
      {
          free(phatAlpha[i]);
          free(phatBeta[i]);
          free(randomMatrixNormalDistribution[i]);
          free(occurrenceSeason[i]);
          free(moranRandom[i]);
          free(simulatedPrecipitationAmountsSeasonal[i]);

      }
      free(phatAlpha);
      free(phatBeta);
      free(randomMatrixNormalDistribution);
      free(occurrenceSeason);
      free(moranRandom);
      free(simulatedPrecipitationAmountsSeasonal);

   }

    weatherGenerator2D::createAmountOutputSerie();

   // free the memory step 5
   for (int i=0;i<nrStations;i++)
   {

       free(amountMatrixSeasonJJA[i]);
       free(amountMatrixSeasonDJF[i]);
       free(amountMatrixSeasonMAM[i]);
       free(amountMatrixSeasonSON[i]);
   }

   free(amountMatrixSeasonJJA);
   free(amountMatrixSeasonDJF);
   free(amountMatrixSeasonMAM);
   free(amountMatrixSeasonSON);

   for (int i=0;i<nrStations;i++)
   {

       free(amountCorrelationMatrixDJF[i]);
       free(amountCorrelationMatrixJJA[i]);
       free(amountCorrelationMatrixMAM[i]);
       free(amountCorrelationMatrixSON[i]);
   }

   free(amountCorrelationMatrixDJF);
   free(amountCorrelationMatrixJJA);
   free(amountCorrelationMatrixMAM);
   free(amountCorrelationMatrixSON);

   // free the memory of arrays declared in step 4 but used in 5
   for (int i=0;i<nrStations;i++)
   {
       free(wDJF[i]);
       free(wJJA[i]);
       free(wMAM[i]);
       free(wSON[i]);
       free(wSeason[i]);
       free(occurrenceMatrixSeasonJJA[i]);
       free(occurrenceMatrixSeasonDJF[i]);
       free(occurrenceMatrixSeasonMAM[i]);
       free(occurrenceMatrixSeasonSON[i]);
   }
   free(wDJF);
   free(wJJA);
   free(wMAM);
   free(wSON);
   free(wSeason);
   free(occurrenceMatrixSeasonDJF);
   free(occurrenceMatrixSeasonJJA);
   free(occurrenceMatrixSeasonMAM);
   free(occurrenceMatrixSeasonSON);
}

void weatherGenerator2D::initializeOccurrenceIndex()
{
   occurrenceIndexSeasonal = (ToccurrenceIndexSeasonal *)calloc(nrStations, sizeof(ToccurrenceIndexSeasonal));
   for (int i=0;i<nrStations;i++)
   {
       occurrenceIndexSeasonal[i].binCenter = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].bin = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].meanFit = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].meanP = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].stdDevFit = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].stdDevP = (double **)calloc(4, sizeof(double*));
       occurrenceIndexSeasonal[i].parMultiexp = (double ***)calloc(4, sizeof(double**));
   }
   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<4;j++)
       {
           occurrenceIndexSeasonal[i].bin[j] = (double *)calloc(11, sizeof(double));
           occurrenceIndexSeasonal[i].binCenter[j] = (double *)calloc(10, sizeof(double));
           occurrenceIndexSeasonal[i].meanFit[j] = (double *)calloc(10, sizeof(double));
           occurrenceIndexSeasonal[i].meanP[j] = (double *)calloc(10, sizeof(double));
           occurrenceIndexSeasonal[i].stdDevFit[j] = (double *)calloc(10, sizeof(double));
           occurrenceIndexSeasonal[i].stdDevP[j] = (double *)calloc(10, sizeof(double));

           occurrenceIndexSeasonal[i].parMultiexp[j] = (double **)calloc(10, sizeof(double*));
       }
       for (int j=0;j<4;j++)
       {
           for (int k=0;k<10;k++)
           {
               occurrenceIndexSeasonal[i].parMultiexp[j][k] = (double *)calloc(2, sizeof(double));
           }
       }
   }

   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<4;j++)
       {
           for (int k=0;k<10;k++)
           {
               occurrenceIndexSeasonal[i].bin[j][k] = NODATA;
               occurrenceIndexSeasonal[i].binCenter[j][k] = NODATA;
               occurrenceIndexSeasonal[i].meanFit[j][k] = NODATA;
               occurrenceIndexSeasonal[i].meanP[j][k] = NODATA;
               occurrenceIndexSeasonal[i].stdDevFit[j][k] = NODATA;
               occurrenceIndexSeasonal[i].stdDevP[j][k] = NODATA;
               occurrenceIndexSeasonal[i].parMultiexp[j][k][0] = NODATA;
               occurrenceIndexSeasonal[i].parMultiexp[j][k][1] = NODATA;
           }
           occurrenceIndexSeasonal[i].bin[j][10] = NODATA;
       }
   }
}

void weatherGenerator2D::initializePrecipitationOutputs(int lengthSeason[])
{
   simulatedPrecipitationAmounts = (TsimulatedPrecipitationAmounts *)calloc(4, sizeof(TsimulatedPrecipitationAmounts));
   for (int iSeason=0;iSeason<4;iSeason++)
   {
        if (iSeason == 0)
           simulatedPrecipitationAmounts[iSeason].season = DJF;
       else if (iSeason == 1)
           simulatedPrecipitationAmounts[iSeason].season = MAM;
       else if (iSeason == 2)
           simulatedPrecipitationAmounts[iSeason].season = JJA;
       else if (iSeason == 3)
           simulatedPrecipitationAmounts[iSeason].season = SON;

       simulatedPrecipitationAmounts[iSeason].matrixAmounts = (double **)calloc(nrStations, sizeof(double*));
       simulatedPrecipitationAmounts[iSeason].matrixK = (double **)calloc(nrStations, sizeof(double*));
       simulatedPrecipitationAmounts[iSeason].matrixM = (double **)calloc(nrStations, sizeof(double*));

       for (int i=0;i<nrStations;i++)
       {
           simulatedPrecipitationAmounts[iSeason].matrixAmounts[i] = (double *)calloc(lengthSeason[iSeason]*parametersModel.yearOfSimulation, sizeof(double));
           simulatedPrecipitationAmounts[iSeason].matrixK[i] = (double *)calloc(nrStations, sizeof(double));
           simulatedPrecipitationAmounts[iSeason].matrixM[i] = (double *)calloc(nrStations, sizeof(double));
       }

       for (int i=0;i<nrStations;i++)
       {
           for (int j=0;j<nrStations;j++)
           {
               simulatedPrecipitationAmounts[iSeason].matrixK[i][j] = NODATA;
               simulatedPrecipitationAmounts[iSeason].matrixM[i][j] = NODATA;
           }
           for (int j=0;j<lengthSeason[iSeason]*parametersModel.yearOfSimulation;j++)
           {
               simulatedPrecipitationAmounts[iSeason].matrixAmounts[i][j] = NODATA;
           }
       }

   }
}


void weatherGenerator2D::precipitation29February(int idStation)
{
   nrDataWithout29February = nrData;

   for (int i=0; i<nrData;i++)
   {
       if (isLeapYear(obsDataD[idStation][i].date.year))
       {
           if ((obsDataD[idStation][i].date.day == 29) && (obsDataD[idStation][i].date.month == 2))
           {
               if (i!= 0)obsDataD[idStation][i-1].prec += obsDataD[idStation][i-1].prec /2;
               if (i != (nrData-1)) obsDataD[idStation][i+1].prec += obsDataD[idStation][i+1].prec /2;
               nrDataWithout29February--;
           }
       }

   }

}

void weatherGenerator2D::precipitationAmountsOccurences(int idStation, double* precipitationAmountsD,bool* precipitationOccurrencesD)
{
   int counter = 0;
   for (int i=0; i<nrData;i++)
   {
           if (!((obsDataD[idStation][i].date.day == 29) && (obsDataD[idStation][i].date.month == 2)))
           {

               if (precipitationAmountsD[counter] < parametersModel.precipitationThreshold)
               {
                   precipitationOccurrencesD[counter]= false;
                   precipitationAmountsD[counter]=0;
               }
               else
               {
                   precipitationOccurrencesD[counter] = true;
                   precipitationAmountsD[counter]=obsDataD[idStation][i].prec;
               }
               counter++;
           }
   }
}

void weatherGenerator2D::spatialIterationAmounts(double** correlationMatrixSimulatedData,double ** amountsCorrelationMatrix , double** randomMatrix, int lengthSeries, double** occurrences, double** phatAlpha, double** phatBeta, double** simulatedPrecipitationAmountsSeasonal)
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



   double uniformRandomVar;
   double** initialAmountsCorrelationMatrix = (double**)calloc(nrStations, sizeof(double*));


   // initialization internal arrays
   for (int i=0;i<nrStations;i++)
   {
       dummyMatrix[i]= (double*)calloc(nrStations, sizeof(double));
       dummyMatrix2[i]= (double*)calloc(nrStations, sizeof(double));
       initialAmountsCorrelationMatrix[i]= (double*)calloc(nrStations, sizeof(double));
       for (int j=0;j<nrStations;j++)
       {
           dummyMatrix[i][j]= NODATA;
           dummyMatrix2[i][j]= NODATA;
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
       for (int j=0;j<lengthSeries;j++)
       {
           dummyMatrix3[i][j]= NODATA;
       }

   }

   for (int i=0;i<nrStations;i++)
   {
       for (int j=0;j<nrStations;j++)
       {
           initialAmountsCorrelationMatrix[i][j] = amountsCorrelationMatrix[i][j];
       }

   }

   double minimalValueToExitFromCycle = NODATA;
   int counterConvergence=0;
   int nrEigenvaluesLessThan0;
   int counter;
   while ((val>TOLERANCE_MULGETS) && (ii<MAX_ITERATION_MULGETS))
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
                   amountsCorrelationMatrix[dMatrix][cMatrix] = amountsCorrelationMatrix[cMatrix][dMatrix] = sumMatrix;
                   sumMatrix = 0.;
               }
           }
           //matricial::matrixProductSquareMatricesNoCheck(dummyMatrix,dummyMatrix2,nrStations,amountsCorrelationMatrix);
           /*for (int i=0;i<nrStations;i++)
           {
               for (int j=0;j<nrStations;j++) // avoid solutions with correlation coefficient greater than 1
               {
                   printf("%f  ",amountsCorrelationMatrix[i][j]);
               }
               printf("\n");
           }
           pressEnterToContinue();
            */
           for (int i=0;i<nrStations-1;i++)
           {
               dummyMatrix[i][i] = 1.;
               for (int j=i+1;j<nrStations;j++)
               {
                        dummyMatrix[i][j] = MINVALUE(2*amountsCorrelationMatrix[i][j]/(amountsCorrelationMatrix[i][i]+ amountsCorrelationMatrix[j][j]),ONELESSEPSILON);
                        dummyMatrix[j][i] = dummyMatrix[i][j];
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
       matricial::matrixProductNoCheck(dummyMatrix,randomMatrix,nrStations,nrStations,lengthSeries,dummyMatrix3);
       double meanValue,stdDevValueOverSQRT_2;
       for (int i=0;i<nrStations;i++)
       {
           // compute mean and standard deviation without NODATA check
           meanValue = stdDevValueOverSQRT_2 = 0;
           for (int j=0;j<lengthSeries;j++)
               meanValue += dummyMatrix3[i][j];
           meanValue /= lengthSeries;
           for (int j=0;j<lengthSeries;j++)
               stdDevValueOverSQRT_2 += (dummyMatrix3[i][j]- meanValue)*(dummyMatrix3[i][j]- meanValue);
           stdDevValueOverSQRT_2 /= (lengthSeries-1);
           stdDevValueOverSQRT_2 = sqrt(stdDevValueOverSQRT_2)/SQRT_2;

           for (int j=0;j<lengthSeries;j++)
           {
               uniformRandomVar =0.5*statistics::tabulatedERFC(-(dummyMatrix3[i][j]-meanValue)/stdDevValueOverSQRT_2);
               uniformRandomVar = MINVALUE(uniformRandomVar,ONELESSEPSILON);
               simulatedPrecipitationAmountsSeasonal[i][j]=0.;
               if (occurrences[i][j] > EPSILON)
               {
                   if (parametersModel.distributionPrecipitation == 1)
                   {
                       simulatedPrecipitationAmountsSeasonal[i][j] =-log(1-uniformRandomVar)/phatAlpha[i][j]+ parametersModel.precipitationThreshold;
                   }
                   else if (parametersModel.distributionPrecipitation == 2)
                   {
                       simulatedPrecipitationAmountsSeasonal[i][j] = inverseGammaCumulativeDistributionFunction(uniformRandomVar,phatAlpha[i][j],phatBeta[i][j],0.0001) + parametersModel.precipitationThreshold;
                   }
                   else if (parametersModel.distributionPrecipitation == 3)
                   {
                        //simulatedPrecipitationAmountsSeasonal[i][j] = phatAlpha[i][j]*pow(-log(1-uniformRandomVar),1./phatBeta[i][j])+ parametersModel.precipitationThreshold;
                        simulatedPrecipitationAmountsSeasonal[i][j] = phatAlpha[i][j]*pow(-log(1-uniformRandomVar),1./(phatBeta[i][j]))+ parametersModel.precipitationThreshold;
                        //simulatedPrecipitationAmountsSeasonal[i][j] = 0.84* phatAlpha[i][j]*pow(-log(1-uniformRandomVar),1.0)+ parametersModel.precipitationThreshold;
                        //simulatedPrecipitationAmountsSeasonal[i][j] =-log(1-uniformRandomVar)*phatAlpha[i][j]+ parametersModel.precipitationThreshold;
                   }
               }
           }
       }
       statistics::correlationsMatrixNoCheck(nrStations,simulatedPrecipitationAmountsSeasonal,lengthSeries,correlationMatrixSimulatedData);
       val = 0;
       for (int i=0;i<nrStations;i++)
       {
           for (int j=i;j<nrStations;j++)
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
       if (counterConvergence > 30)
       {
           if (val <= fabs(minimalValueToExitFromCycle) + TOLERANCE_MULGETS)
           {
               for (int i=0;i<nrStations;i++)
               {
                   free(dummyMatrix[i]);
                   free(dummyMatrix2[i]);
                   free(dummyMatrix3[i]);
                   free(initialAmountsCorrelationMatrix[i]);
               }


                free(dummyMatrix);
                free(dummyMatrix2);
                free(dummyMatrix3);
                free(correlationArray);
                free(eigenvalues);
                free(eigenvectors);
                free(initialAmountsCorrelationMatrix);

               return;
           }
       }

       if (ii != MAX_ITERATION_MULGETS && val> TOLERANCE_MULGETS )
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
                       amountsCorrelationMatrix[i][j] = MINVALUE(amountsCorrelationMatrix[i][j],1);
                   }
               }
           }
       }
       //printf("iter %d value %f \n",ii,val);
   }
   // free memory
   for (int i=0;i<nrStations;i++)
   {
       free(dummyMatrix[i]);
       free(dummyMatrix2[i]);
       free(dummyMatrix3[i]);
       free(initialAmountsCorrelationMatrix[i]);
   }


       free(dummyMatrix);
       free(dummyMatrix2);
       free(dummyMatrix3);
       free(correlationArray);
       free(eigenvalues);
       free(eigenvectors);
       free(initialAmountsCorrelationMatrix);

}

/*
double weatherGenerator2D::inverseGammaFunction(double valueProbability, double alpha, double beta, double accuracy)
{
   double x;
   double y;
   double rightBound = 25.0;
   double leftBound = 0.0;
   int counter = 0;
   do {
       y = incompleteGamma(alpha,rightBound/beta);
       if (valueProbability>y)
       {
           rightBound *= 2;
           counter++;
           if (counter == 7) return rightBound;
       }
   } while ((valueProbability>y));

   x = (rightBound + leftBound)*0.5;
   y = incompleteGamma(alpha,x/beta);
   while ((fabs(valueProbability - y) > accuracy) && (counter < 200))
   {
       if (y > valueProbability)
       {
           rightBound = x;
       }
       else
       {
           leftBound = x;
       }
       x = (rightBound + leftBound)*0.5;
       y = incompleteGamma(alpha,x/beta);
       ++counter;
   }
   x = (rightBound + leftBound)*0.5;
   return x;
}
*/

void weatherGenerator2D::pressEnterToContinue()
{
   printf("press return to continue\n");
   getchar();
}


void weatherGenerator2D::createAmountOutputSerie()
{
    //int dayOfYear;
    //int iSeason;
    int day,monthInternal;
    double* january =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* february =(double*)calloc(28*parametersModel.yearOfSimulation, sizeof(double));
    double* march =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* april =(double*)calloc(30*parametersModel.yearOfSimulation, sizeof(double));
    double* may =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* june =(double*)calloc(30*parametersModel.yearOfSimulation, sizeof(double));
    double* july =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* august =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* september =(double*)calloc(30*parametersModel.yearOfSimulation, sizeof(double));
    double* october =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));
    double* november =(double*)calloc(30*parametersModel.yearOfSimulation, sizeof(double));
    double* december =(double*)calloc(31*parametersModel.yearOfSimulation, sizeof(double));

    for(int j=0;j<nrStations;j++)
    {
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                december[k] = simulatedPrecipitationAmounts[0].matrixAmounts[j][k];
            }
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                january[k] = simulatedPrecipitationAmounts[0].matrixAmounts[j][31*parametersModel.yearOfSimulation+k];
            }
            for (int k=0;k<28*parametersModel.yearOfSimulation;k++)
            {
                february[k] = simulatedPrecipitationAmounts[0].matrixAmounts[j][2*31*parametersModel.yearOfSimulation+k];
            }

            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                march[k] = simulatedPrecipitationAmounts[1].matrixAmounts[j][k];
            }
            for (int k=0;k<30*parametersModel.yearOfSimulation;k++)
            {
                april[k] = simulatedPrecipitationAmounts[1].matrixAmounts[j][31*parametersModel.yearOfSimulation+k];
            }
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                may[k] = simulatedPrecipitationAmounts[1].matrixAmounts[j][(30+31)*parametersModel.yearOfSimulation+k];
            }

            for (int k=0;k<30*parametersModel.yearOfSimulation;k++)
            {
                june[k] = simulatedPrecipitationAmounts[2].matrixAmounts[j][k];
            }
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                july[k] = simulatedPrecipitationAmounts[2].matrixAmounts[j][30*parametersModel.yearOfSimulation+k];
            }
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                august[k] = simulatedPrecipitationAmounts[2].matrixAmounts[j][(30+31)*parametersModel.yearOfSimulation+k];
            }

            for (int k=0;k<30*parametersModel.yearOfSimulation;k++)
            {
                september[k] = simulatedPrecipitationAmounts[3].matrixAmounts[j][k];
            }
            for (int k=0;k<31*parametersModel.yearOfSimulation;k++)
            {
                october[k] = simulatedPrecipitationAmounts[3].matrixAmounts[j][30*parametersModel.yearOfSimulation+k];
            }
            for (int k=0;k<30*parametersModel.yearOfSimulation;k++)
            {
                november[k] = simulatedPrecipitationAmounts[3].matrixAmounts[j][(30+31)*parametersModel.yearOfSimulation+k];
            }


            int count[12];
            for (int counterMonth=0;counterMonth<12;counterMonth++)
            {
                count[counterMonth]=0;
            }

            for(int i=0;i<parametersModel.yearOfSimulation*365;i++)
            {
                dateFromDoy(i%365+1,1,&day,&monthInternal);
                if ( monthInternal == 1)
                {
                    amountsPrecGenerated[i][j] = january[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 2)
                {
                    amountsPrecGenerated[i][j] = february[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 3)
                {
                    amountsPrecGenerated[i][j] = march[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 4)
                {
                    amountsPrecGenerated[i][j] = april[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 5)
                {
                    amountsPrecGenerated[i][j] = may[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 6)
                {
                    amountsPrecGenerated[i][j] = june[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 7)
                {
                    amountsPrecGenerated[i][j] = july[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 8)
                {
                    amountsPrecGenerated[i][j] = august[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 9)
                {
                    amountsPrecGenerated[i][j] = september[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 10)
                {
                    amountsPrecGenerated[i][j] = october[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 11)
                {
                    amountsPrecGenerated[i][j] = november[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
                if ( monthInternal == 12)
                {
                    amountsPrecGenerated[i][j] = december[count[monthInternal-1]];
                    ++count[monthInternal-1];
                }
            }
    }
    free(january);
    free(february);
    free(march);
    free(april);
    free(may);
    free(june);
    free(july);
    free(august);
    free(september);
    free(october);
    free(november);
    free(december);
}

