/*!
    \copyright 2023
    Fausto Tomei, Gabriele Antolini, Antonio Volta

    This file is part of AGROLIB distribution.
    AGROLIB has been developed under contract issued by A.R.P.A. Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    Contacts:
    ftomei@arpae.it
    gantolini@arpae.it
    avolta@arpae.it
*/

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>    // std::find
#include <vector>

#include "commonConstants.h"
#include "basicMath.h"
#include "statistics.h"
#include "physics.h"
#include "furtherMathFunctions.h"


float statisticalElab(meteoComputation elab, float param, std::vector<float> values, int nValues, float rainfallThreshold)
{
    switch(elab)
    {
        case average:
            return statistics::mean(values);
        case maxInList:
            return statistics::maxList(values, nValues);
        case minInList:
            return statistics::minList(values, nValues);
        case sum:
            return statistics::sumList(values, nValues);
        case sumAbove:
            return statistics::sumListThreshold(values, nValues, param);
        case differenceWithThreshold:
            return statistics::diffListThreshold(values, nValues, param);
        case daysAbove:
            return statistics::countAbove(values, nValues, param);
        case daysBelow:
            return statistics::countBelow(values, nValues, param);
        case consecutiveDaysAbove:
            return statistics::countConsecutive(values, nValues, param, true);
        case consecutiveDaysBelow:
            return statistics::countConsecutive(values, nValues, param, false);
        case percentile:
            return sorting::percentile(values, nValues, param, true);
        case freqPositive:
            return statistics::frequencyPositive(values, nValues);
        case prevailingWindDir:
        {
            std::vector<float> intensity = values;
            return float(windPrevailingDir(intensity, values, nValues, false));
        }
        case trend:
            return statistics::trend(values, nValues, param);
        case mannKendall:
            return statistics::mannKendall(values, nValues);
        case erosivityFactorElab:
            return erosivityFactor(values, nValues);
        case rainIntensityElab:
            return rainIntensity(values, nValues, rainfallThreshold);
        case stdDev:
            return statistics::standardDeviation(values, nValues);
        case timeIntegration:
            return timeIntegrationFunction(values, param);
        case yearMax:
        {
            float maxValue = statistics::maxList(values, nValues);
            std::vector<float>::iterator it = std::find(values.begin(), values.end(), maxValue);
            if (it != values.end())
            {
                int index = int(std::distance(values.begin(), it));
                return float(index);
            }
            else
            {
                return NODATA;
            }
        }
        case yearMin:
        {
            float minValue = statistics::minList(values, nValues);
            std::vector<float>::iterator it = std::find(values.begin(), values.end(), minValue);
            if (it != values.end())
            {
                int index = int(std::distance(values.begin(), it));
                return float(index);
            }
            else
            {
                return NODATA;
            }
        }

        default:
        {
            if (elab == noMeteoComp && nValues == 1)
            {
                return values[0];
            }
            else
            {
                return NODATA;
            }
        }
    }
}


namespace statistics
{

    double rootMeanSquareError(double *measured , double *simulated , int nrData)
    {
        double sigma=0.;
        double diff;
        long nrValidValues = 0;

        for (int i=0; i< nrData; i++)
        {
            if ((measured[i] != NODATA) && (simulated[i] != NODATA))
            {
                diff = measured[i]-simulated[i];
                sigma += (diff * diff);
                nrValidValues++;
            }
        }

        if (nrValidValues == 0) return NODATA;

        sigma /= nrValidValues;
        sigma = sqrt(sigma);
        return sigma;
    }

    float rootMeanSquareError(float *measured , float *simulated , int nrData)
    {
        double sigma=0.;
        double diff;
        long nrValidValues = 0;

        for (int i=0; i< nrData; i++)
        {
            if ((measured[i] != NODATA) && (simulated[i] != NODATA))
            {
                diff = measured[i]-simulated[i];
                sigma += (diff * diff);
                nrValidValues++;
            }
        }

        if (nrValidValues == 0) return NODATA;

        sigma /= nrValidValues;
        sigma = sqrt(sigma);
        return float(sigma);
    }

    float meanError(std::vector <float> measured, std::vector <float> simulated)
    {
        float sum=0.;
        long nrValidValues = 0;

        if (measured.size() != simulated.size()) return NODATA;

        for (unsigned int i=0; i < measured.size(); i++)
        {
            if ((measured[i] != NODATA) && (simulated[i] != NODATA))
            {
                sum += measured[i]-simulated[i];
                nrValidValues++;
            }
        }

        if (nrValidValues == 0) return NODATA;

        return sum / nrValidValues;
    }

    float meanAbsoluteError(std::vector <float> measured, std::vector <float> simulated)
    {
        double sum=0.;
        long nrValidValues = 0;

        if (measured.size() != simulated.size()) return NODATA;

        for (unsigned int i=0; i < measured.size(); i++)
        {
            if ((measured[i] != NODATA) && (simulated[i] != NODATA))
            {
                sum += fabs(measured[i]-simulated[i]);
                nrValidValues++;
            }
        }

        if (nrValidValues == 0) return NODATA;

        return float(sum / nrValidValues);
    }

    double rootMeanSquareError(std::vector <float> measured, std::vector <float> simulated)
    {
        double sigma=0.;
        double diff;
        long nrValidValues = 0;

        if (measured.size() != simulated.size()) return NODATA;

        for (unsigned int i=0; i < measured.size(); i++)
        {
            if ((measured[i] != NODATA) && (simulated[i] != NODATA))
            {
                diff = measured[i]-simulated[i];
                sigma += (diff * diff);
                nrValidValues++;
            }
        }

        if (nrValidValues == 0) return NODATA;

        sigma /= nrValidValues;
        sigma = sqrt(sigma);
        return sigma;
    }

    double compoundRelativeError(std::vector <float> measured, std::vector <float> simulated)
    {
        if (measured.size() != simulated.size()) return NODATA;

        float obsAvg = mean(measured);

        if (isEqual(obsAvg, NODATA)) return NODATA;

        float sumDev = 0;

        for (unsigned int i=0; i < measured.size(); i++)
            if (!isEqual(measured[i], NODATA))
                sumDev += (measured[i] - obsAvg) * (measured[i] - obsAvg);

        if (sumDev == 0) return NODATA;

        float sumError = 0;

        for (unsigned int i = 0; i < measured.size(); i++)
            if (!isEqual(measured[i], NODATA) && !isEqual(simulated[i], NODATA))
                sumError += (simulated[i] - measured[i]) * (simulated[i] - measured[i]);

        return sumError / sumDev;
    }

    float coefficientOfVariation(float *measured , float *simulated , int nrData)
    {
        double sigma=0;
        double measuredMean = 0;
        double coefVar;
        double diff;
        for (int i=0; i< nrData; i++)
        {
            diff = measured[i]-simulated[i];
            sigma += diff * diff;
            measuredMean += measured[i];
        }
        sigma /= nrData;
        measuredMean /= nrData;
        sigma = sqrt(sigma);
        coefVar = sigma / measuredMean ;
        return float(coefVar);
    }

    float weighedMean(float *data , float *weights, int nrData)
    {
        float mean=0 ;
        float weightsSum=0 ;

        for (int i = 0 ; i<nrData;i++)
            weightsSum += weights[i];

        if ((weightsSum< 0.99) || (weightsSum > 1.01))
            return NODATA;

        for (int i = 0 ; i<nrData;i++)
            mean += weights[i]*data[i];

        return mean ;
    }

    float linearInterpolation(float x1, float y1, float x2, float y2, float xx)
    {
        float rate;

        if (isEqual(x1, x2))
            return y1;

        rate = (y2 - y1) / (x2 - x1);

        return (y1 + rate * (xx - x1));
    }


    void weightedMultiRegressionLinear(float** x,  float* y, float* weight, long nrItems,float* q,float* m, int nrPredictors)
    {
        //int matrixSize = nrPredictors+1;
        double** XT = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X = (double**)calloc(nrItems, sizeof(double*));
        double** X2 = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X2Inverse = (double**)calloc(nrPredictors+1, sizeof(double*));
        for (int i=0;i<nrPredictors+1;i++)
        {
            XT[i]= (double*)calloc(nrItems, sizeof(double));
            X2[i]= (double*)calloc(nrItems, sizeof(double));
            X2Inverse[i]= (double*)calloc(nrItems, sizeof(double));
        }
        for (int i=0;i<nrItems;i++)
        {
            X[i]= (double*)calloc(nrPredictors+1, sizeof(double));
        }
        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i=0;i<nrItems;i++)
            {
               if (j == 0)
               {
                    XT[j][i] = 1.;
               }
               else
               {
                    XT[j][i] = x[i][j-1];
               }
            }

        }
        matricial::transposedMatrix(XT,nrPredictors+1,nrItems,X);
        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i =0; i<nrItems; i++)
            {
                X[i][j]= 1./weight[i]*X[i][j];
            }
        }
        matricial::matrixProduct(XT,X,nrItems,nrPredictors+1,nrPredictors+1,nrItems,X2);
        matricial::inverse(X2,X2Inverse,nrPredictors+1);
        //matricial::matrixProduct(X2Inverse,XT,nrPredictors+1,nrPredictors+1,nrItems,nrPredictors+1,X);
        for (int i=0;i<nrItems;i++)
        {
            y[i] /= weight[i];
        }

        double* roots = (double*)calloc(nrPredictors+1, sizeof(double));
        for (int j=0;j<nrPredictors+1;j++)
        {
            roots[j]=0;
            for (int i=0;i<nrItems;i++)
            {
                roots[j] += (XT[j][i]*y[i]);
            }
        }

        *q = 0;
        for (int j=0; j<nrPredictors; j++)
        {
            m[j]=0;
        }

        for (int i=0; i<nrPredictors+1; i++)
        {
            *q += float(X2Inverse[0][i]*roots[i]);
        }

        for (int j=1; j<nrPredictors+1; j++)
        {
            for (int i=0;i<nrPredictors+1;i++)
            {
                m[j-1] += float(X2Inverse[j][i]*roots[i]);
            }
        }

        for (int j=0;j<nrPredictors+1;j++)
        {
            free(XT[j]);
            free(X2[j]);
            free(X2Inverse[j]);
        }
        for (int i=0;i<nrItems;i++)
        {
            free(X[i]);
        }
        free(X);
        free(XT);
        free(X2);
        free(X2Inverse);
        free(roots);
    }

    void weightedMultiRegressionLinear(const std::vector <std::vector <float>> &x, std::vector <float> &y, const std::vector <float> &weight, long nrItems,float* q,std::vector <float> &m, int nrPredictors)
    {
        double** XT = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X = (double**)calloc(nrItems, sizeof(double*));
        double** X2 = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X2Inverse = (double**)calloc(nrPredictors+1, sizeof(double*));
        for (int i=0;i<nrPredictors+1;i++)
        {
            XT[i]= (double*)calloc(nrItems, sizeof(double));
            X2[i]= (double*)calloc(nrItems, sizeof(double));
            X2Inverse[i]= (double*)calloc(nrItems, sizeof(double));
        }
        for (int i=0;i<nrItems;i++)
        {
            X[i]= (double*)calloc(nrPredictors+1, sizeof(double));
        }
        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i=0;i<nrItems;i++)
            {
                if (j == 0)
                {
                    XT[j][i] = 1.;
                }
                else
                {
                    XT[j][i] = x[i][j-1];
                }
            }

        }
        matricial::transposedMatrix(XT,nrPredictors+1,nrItems,X);
        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i =0; i<nrItems; i++)
            {
                X[i][j]= weight[i]*X[i][j];
            }
        }
        matricial::matrixProduct(XT,X,nrItems,nrPredictors+1,nrPredictors+1,nrItems,X2);
        matricial::inverse(X2,X2Inverse,nrPredictors+1);
        //matricial::matrixProduct(X2Inverse,XT,nrPredictors+1,nrPredictors+1,nrItems,nrPredictors+1,X);
        for (int i=0;i<nrItems;i++)
        {
            y[i] *= weight[i];
        }
        double* roots = (double*)calloc(nrPredictors+1, sizeof(double));
        for (int j=0;j<nrPredictors+1;j++)
        {
            roots[j]=0;
            for (int i=0;i<nrItems;i++)
            {
                roots[j] += (XT[j][i]*y[i]);
            }
        }
        *q=0;
        for (int j=0;j<nrPredictors;j++)
        {
            m.push_back(0);
            //m[j]=0;
        }
        for (int i=0;i<nrPredictors+1;i++)
        {
            *q += float(X2Inverse[0][i]*roots[i]);
        }

        for (int j=1;j<nrPredictors+1;j++)
        {
            for (int i=0;i<nrPredictors+1;i++)
            {
                m[j-1] += float(X2Inverse[j][i]*roots[i]);
            }
        }
        for (int j=0;j<nrPredictors+1;j++)
        {
            free(XT[j]);
            free(X2[j]);
            free(X2Inverse[j]);
        }
        for (int i=0;i<nrItems;i++)
        {
            free(X[i]);
        }
        free(X);
        free(XT);
        free(X2);
        free(X2Inverse);
        free(roots);
    }

    void weightedMultiRegressionLinearWithStats(const std::vector <std::vector <float>> &x, std::vector <float> &y, const std::vector <float> &weight,float* q,std::vector <float> &m,bool calculateR2, bool calculateStdError,float* R2, float* stdError, float *qSE,std::vector <float> &mSE)
    {
        int nrPredictors = int(x[0].size());
        int nrItems = int(x.size());
        double** XT = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X = (double**)calloc(nrItems, sizeof(double*));
        double** X2 = (double**)calloc(nrPredictors+1, sizeof(double*));
        double** X2Inverse = (double**)calloc(nrPredictors+1, sizeof(double*));
        for (int i=0;i<nrPredictors+1;i++)
        {
            XT[i]= (double*)calloc(nrItems, sizeof(double));
            X2[i]= (double*)calloc(nrItems, sizeof(double));
            X2Inverse[i]= (double*)calloc(nrItems, sizeof(double));
        }
        for (int i=0;i<nrItems;i++)
        {
            X[i]= (double*)calloc(nrPredictors+1, sizeof(double));
        }

        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i=0;i<nrItems;i++)
            {
                if (j == 0)
                {
                    XT[j][i] = 1.;
                }
                else
                {
                    XT[j][i] = x[i][j-1];
                }
            }
        }
        matricial::transposedMatrix(XT,nrPredictors+1,nrItems,X);
        for (int j=0;j<nrPredictors+1;j++)
        {
            for (int i =0; i<nrItems; i++)
            {
                X[i][j]= weight[i]*X[i][j];
            }
        }
        matricial::matrixProduct(XT,X,nrItems,nrPredictors+1,nrPredictors+1,nrItems,X2);
        matricial::inverse(X2,X2Inverse,nrPredictors+1);
        //matricial::matrixProduct(X2Inverse,XT,nrPredictors+1,nrPredictors+1,nrItems,nrPredictors+1,X);
        for (int i=0;i<nrItems;i++)
        {
            y[i] *= weight[i];
        }
        double* roots = (double*)calloc(nrPredictors+1, sizeof(double));
        for (int j=0;j<nrPredictors+1;j++)
        {
            roots[j]=0;
            for (int i=0;i<nrItems;i++)
            {
                roots[j] += (XT[j][i]*y[i]);
            }
        }
        *q=0;
        for (int j=0;j<nrPredictors;j++)
        {
            m.push_back(0);
            mSE.push_back(0);
            //m[j] = 0;
            //mSE[j] = 0;
        }
        for (int i=0;i<nrPredictors+1;i++)
        {
            *q += float(X2Inverse[0][i]*roots[i]);
        }

        for (int j=1;j<nrPredictors+1;j++)
        {
            for (int i=0;i<nrPredictors+1;i++)
            {
                m[j-1] += float(X2Inverse[j][i]*roots[i]);
            }
        }

        if (calculateR2)
        {
            std::vector <double> ySim(nrItems);
            std::vector <double> yObs(nrItems);
            std::vector <double> weightDouble(nrItems);
            for (int i=0;i<nrItems;i++)
            {
                ySim[i]=(*q);
                for (int j=0;j<nrPredictors;j++)
                {
                    ySim[i]+=x[i][j]*m[j];
                }
                yObs[i]=y[i];
                weightDouble[i]=weight[i];
            }
            *R2 = float(interpolation::computeWeighted_R2(yObs,ySim,weightDouble));
        }
        if (calculateStdError)
        {
            std::vector <double> ySim(nrItems);
            std::vector <double> yObs(nrItems);
            std::vector <double> weightDouble(nrItems);
            for (int i=0;i<nrItems;i++)
            {
                ySim[i]=(*q);
                for (int j=0;j<nrPredictors;j++)
                {
                    ySim[i]+=x[i][j]*m[j];
                }
                yObs[i]=y[i];
                weightDouble[i]=weight[i];
            }
            *stdError = float(interpolation::computeWeighted_StandardError(yObs,ySim,weightDouble,nrPredictors+1));
            matricial::transposedMatrix(XT,nrPredictors+1,nrItems,X);
            for (int j=0;j<nrPredictors+1;j++)
            {
                for (int i =0; i<nrItems; i++)
                {
                    X[i][j]= weight[i]*X[i][j];
                }
            }
            matricial::matrixProduct(XT,X,nrItems,nrPredictors+1,nrPredictors+1,nrItems,X2);
            matricial::inverse(X2,X2Inverse,nrPredictors+1);
            *qSE = (*stdError) * float(sqrt(X2Inverse[0][0]));
            for (int j=1;j<nrPredictors+1;j++)
            {
                mSE[j-1]= (*stdError) * float(sqrt(X2Inverse[j][j]));
            }
        }

        for (int j=0;j<nrPredictors+1;j++)
        {
            free(XT[j]);
            free(X2[j]);
            free(X2Inverse[j]);
        }
        for (int i=0;i<nrItems;i++)
        {
            free(X[i]);
        }
        free(X);
        free(XT);
        free(X2);
        free(X2Inverse);
        free(roots);

    }

    void weightedMultiRegressionLinearNoOffset(const std::vector <std::vector <float>> &x, std::vector <float> &y, const std::vector <float> &weight, long nrItems,std::vector <float> &m, int nrPredictors)
    {
        double** XT = (double**)calloc(nrPredictors, sizeof(double*));
        double** X = (double**)calloc(nrItems, sizeof(double*));
        double** X2 = (double**)calloc(nrPredictors, sizeof(double*));
        double** X2Inverse = (double**)calloc(nrPredictors, sizeof(double*));
        for (int i=0;i<nrPredictors;i++)
        {
            XT[i]= (double*)calloc(nrItems, sizeof(double));
            X2[i]= (double*)calloc(nrItems, sizeof(double));
            X2Inverse[i]= (double*)calloc(nrItems, sizeof(double));
        }
        for (int i=0;i<nrItems;i++)
        {
            X[i]= (double*)calloc(nrPredictors, sizeof(double));
        }
        for (int j=0;j<nrPredictors;j++)
        {
            for (int i=0;i<nrItems;i++)
            {
                XT[j][i] = x[i][j];
            }

        }
        matricial::transposedMatrix(XT,nrPredictors,nrItems,X);
        for (int j=0;j<nrPredictors;j++)
        {
            for (int i =0; i<nrItems; i++)
            {
                X[i][j]= weight[i]*X[i][j];
            }
        }
        matricial::matrixProduct(XT,X,nrItems,nrPredictors,nrPredictors,nrItems,X2);
        matricial::inverse(X2,X2Inverse,nrPredictors);
        //matricial::matrixProduct(X2Inverse,XT,nrPredictors+1,nrPredictors+1,nrItems,nrPredictors+1,X);
        for (int i=0;i<nrItems;i++)
        {
            y[i] *= weight[i];
        }
        double* roots = (double*)calloc(nrPredictors, sizeof(double));
        for (int j=0;j<nrPredictors;j++)
        {
            roots[j]=0;
            for (int i=0;i<nrItems;i++)
            {
                roots[j] += (XT[j][i]*y[i]);
            }
        }
        for (int j=0;j<nrPredictors;j++)
        {
            m.push_back(0);
            //m[j]=0;
        }

        for (int j=0;j<nrPredictors;j++)
        {
            for (int i=0;i<nrPredictors;i++)
            {
                m[j] += float(X2Inverse[j][i]*roots[i]);
            }
        }
        for (int j=0;j<nrPredictors;j++)
        {
            free(XT[j]);
            free(X2[j]);
            free(X2Inverse[j]);
        }
        for (int i=0;i<nrItems;i++)
        {
            free(X[i]);
        }
        free(X);
        free(XT);
        free(X2);
        free(X2Inverse);
        free(roots);
    }
    void weightedMultiRegressionLinearWithStatsNoOffset(const std::vector <std::vector <float>> &x, std::vector <float> &y, const std::vector <float> &weight,std::vector <float> &m,bool calculateR2, bool calculateStdError,float* R2, float* stdError, float *qSE,std::vector <float> &mSE)
    {
        int nrPredictors = int(x[0].size());
        int nrItems = int(x.size());
        double** XT = (double**)calloc(nrPredictors, sizeof(double*));
        double** X = (double**)calloc(nrItems, sizeof(double*));
        double** X2 = (double**)calloc(nrPredictors, sizeof(double*));
        double** X2Inverse = (double**)calloc(nrPredictors, sizeof(double*));
        for (int i=0;i<nrPredictors;i++)
        {
            XT[i]= (double*)calloc(nrItems, sizeof(double));
            X2[i]= (double*)calloc(nrItems, sizeof(double));
            X2Inverse[i]= (double*)calloc(nrItems, sizeof(double));
        }
        for (int i=0;i<nrItems;i++)
        {
            X[i]= (double*)calloc(nrPredictors, sizeof(double));
        }

        for (int j=0;j<nrPredictors;j++)
        {
            for (int i=0;i<nrItems;i++)
            {
                XT[j][i] = x[i][j];
            }
        }
        matricial::transposedMatrix(XT,nrPredictors,nrItems,X);
        for (int j=0;j<nrPredictors;j++)
        {
            for (int i =0; i<nrItems; i++)
            {
                X[i][j]= weight[i]*X[i][j];
            }
        }
        matricial::matrixProduct(XT,X,nrItems,nrPredictors,nrPredictors,nrItems,X2);
        matricial::inverse(X2,X2Inverse,nrPredictors);
        //matricial::matrixProduct(X2Inverse,XT,nrPredictors+1,nrPredictors+1,nrItems,nrPredictors+1,X);
        for (int i=0;i<nrItems;i++)
        {
            y[i] *= weight[i];
        }
        double* roots = (double*)calloc(nrPredictors, sizeof(double));
        for (int j=0;j<nrPredictors;j++)
        {
            roots[j]=0;
            for (int i=0;i<nrItems;i++)
            {
                roots[j] += (XT[j][i]*y[i]);
            }
        }

        for (int j=0;j<nrPredictors;j++)
        {
            m.push_back(0);
            mSE.push_back(0);
            //m[j] = 0;
            //mSE[j] = 0;
        }

        for (int j=0;j<nrPredictors;j++)
        {
            for (int i=0;i<nrPredictors;i++)
            {
                m[j] += float(X2Inverse[j][i]*roots[i]);
            }
        }

        if (calculateR2)
        {
            std::vector <double> ySim(nrItems);
            std::vector <double> yObs(nrItems);
            std::vector <double> weightDouble(nrItems);
            for (int i=0;i<nrItems;i++)
            {
                ySim[i]=0;
                for (int j=0;j<nrPredictors;j++)
                {
                    ySim[i]+=x[i][j]*m[j];
                }
                yObs[i]=y[i];
                weightDouble[i]=weight[i];
            }
            *R2 = float(interpolation::computeWeighted_R2(yObs,ySim,weightDouble));
        }
        if (calculateStdError)
        {
            std::vector <double> ySim(nrItems);
            std::vector <double> yObs(nrItems);
            std::vector <double> weightDouble(nrItems);
            for (int i=0;i<nrItems;i++)
            {
                ySim[i]=0;
                for (int j=0;j<nrPredictors;j++)
                {
                    ySim[i]+=x[i][j]*m[j];
                }
                yObs[i]=y[i];
                weightDouble[i]=weight[i];
            }
            *stdError = float(interpolation::computeWeighted_StandardError(yObs,ySim,weightDouble,nrPredictors));
            matricial::transposedMatrix(XT,nrPredictors,nrItems,X);
            for (int j=0;j<nrPredictors;j++)
            {
                for (int i =0; i<nrItems; i++)
                {
                    X[i][j]= weight[i]*X[i][j];
                }
            }
            matricial::matrixProduct(XT,X,nrItems,nrPredictors,nrPredictors,nrItems,X2);
            matricial::inverse(X2,X2Inverse,nrPredictors);
            *qSE = (*stdError) * float(sqrt(X2Inverse[0][0]));
            for (int j=1;j<nrPredictors;j++)
            {
                mSE[j-1]= (*stdError) * float(sqrt(X2Inverse[j][j]));
            }
        }

        for (int j=0;j<nrPredictors;j++)
        {
            free(XT[j]);
            free(X2[j]);
            free(X2Inverse[j]);
        }
        for (int i=0;i<nrItems;i++)
        {
            free(X[i]);
        }
        free(X);
        free(XT);
        free(X2);
        free(X2Inverse);
        free(roots);

    }


    void multiRegressionLinear(float** x,  float* y, long nrItems,float* q,float* m, int nrPredictors)
    {
        double* SUMx;
        double* SUMxx;
        double* SUMxy;
        double SUMy = 0;
        SUMx = (double*)calloc(nrPredictors, sizeof(double));
        SUMxx = (double*)calloc(nrPredictors, sizeof(double));
        SUMxy = (double*)calloc(nrPredictors, sizeof(double));
        for(int i = 0;i<nrPredictors;i++)
        {
            SUMx[i] = 0.;
            SUMxx[i] = 0.;
            SUMxy[i] = 0.;
            for (int j = 0; j<nrItems;j++)
            {
                SUMx[i] += x[i][j];
                SUMxx[i] += x[i][j]*x[i][j];
                SUMxy[i] += x[i][j]*y[j];
            }
        }
        for (int j = 0; j<nrItems;j++)
        {
            SUMy += y[j];
        }
        int matrixSize = nrPredictors+1;
        double* roots = (double*)calloc(matrixSize, sizeof(double));
        double* constantTerm=(double*)calloc(matrixSize, sizeof(double));
        double** coefficientMatrix = (double**)calloc(matrixSize, sizeof(double*));
        for(int i = 0;i<matrixSize;i++)
        {
            coefficientMatrix[i] = (double*)calloc(matrixSize, sizeof(double));
        }

        // set the constant term
        for (int j = 0; j<nrPredictors;j++)
        {
            constantTerm[j] = SUMxy[j];
        }
        constantTerm[nrPredictors] = SUMy;
        // set the coefficient matrix
        for (int k = 0; k<nrPredictors;k++)
        {
            for (int j = 0; j<matrixSize;j++)
            {
                if (k != j)
                {
                    coefficientMatrix[k][j] = SUMx[k];
                }
                else
                {
                    coefficientMatrix[k][j] = SUMxx[k];
                }
            }
        }
        for (int j = 0; j<nrPredictors;j++)
        {
            coefficientMatrix[nrPredictors][j] = SUMx[nrPredictors];
        }
        coefficientMatrix[nrPredictors][nrPredictors] = 1.0*nrItems;

        matricial::linearSystemResolutionByCramerMethod(constantTerm,coefficientMatrix,matrixSize,roots);
        for (int j = 0; j<nrPredictors;j++)
        {
            m[j] = (float)(roots[j]);
        }
        *q = (float)(roots[nrPredictors]);

        for (int j = 0; j<matrixSize;j++)
        {
            free(coefficientMatrix[j]);
        }
        free(coefficientMatrix);
        free(roots);
        free(constantTerm);
        free(SUMx);
        free(SUMxx);
    }


    void linearRegression( float* x,  float* y, long nrItems, bool zeroIntercept, float* y_intercept, float* mySlope, float* r2)
    {
       double SUMx = 0;         /*!< sum of x values */
       double SUMy = 0;         /*!< sum of y values */
       double SUMxy = 0;        /*!< sum of x * y */
       double SUMxx = 0;        /*!< sum of x^2 */
       double AVGy = 0;         /*!< mean of y */
       double AVGx = 0;         /*!< mean of x */
       double dy = 0;           /*!< squared of the discrepancies */
       double SUM_dy = 0;       /*!< sum of squared of the discrepancies */
       double SUMres = 0;       /*!< sum of squared residue */
       double res = 0;          /*!< residue squared */

       long nrValidItems = 0;

       *mySlope = 0;             /*!< slope of regression line */
       *y_intercept = 0;         /*!< y intercept of regression line */
       *r2 = 0;                  /*!< coefficient of determination */

       /*! calculate various sums */
       for (int i = 0; i < nrItems; i++)
           if ((x[i] != NODATA) && (y[i] != NODATA))
           {
                nrValidItems++;
                SUMx += x[i];
                SUMy += y[i];
                SUMxy += x[i] * y[i];
                SUMxx += x[i] * x[i];
           }

        /*! means of x and y */
        AVGy = SUMy / nrValidItems;
        AVGx = SUMx / nrValidItems;

        if (!zeroIntercept)
        {
            *mySlope = float((SUMxy - SUMx * SUMy / nrValidItems) / (SUMxx - SUMx * SUMx / nrValidItems));
            *y_intercept = float(AVGy - *mySlope * AVGx);
        }
        else
        {
            *mySlope = float(SUMxy / SUMxx);
            *y_intercept = 0.;
        }

        /*! calculate squared residues and their sum */
        for (int i = 0; i < nrItems; i++)
           if ((x[i] != NODATA) && (y[i] != NODATA))
           {
              /*! sum of squared of the discrepancies */
              dy = y[i] - (*y_intercept + *mySlope * x[i]);
              dy *= dy;
              SUM_dy += dy;

              /*! sum of squared residues */
              res = y[i] - AVGy;
              res *= res;
              SUMres += res;
           }

        /*! calculate r^2 (coefficient of determination) */
        *r2 = float((SUMres - SUM_dy) / SUMres);
    }

    void linearRegression( std::vector<float> x,  std::vector<float> y, long nrItems, bool zeroIntercept, float* y_intercept, float* mySlope, float* r2)
    {
       double SUMx = 0;         /*!< sum of x values */
       double SUMy = 0;         /*!< sum of y values */
       double SUMxy = 0;        /*!< sum of x * y */
       double SUMxx = 0;        /*!< sum of x^2 */
       double AVGy = 0;         /*!< mean of y */
       double AVGx = 0;         /*!< mean of x */
       double dy = 0;           /*!< squared of the discrepancies */
       double SUM_dy = 0;       /*!< sum of squared of the discrepancies */
       double SUMres = 0;       /*!< sum of squared residue */
       double res = 0;          /*!< residue squared */

       long nrValidItems = 0;

       *mySlope = 0;             /*!< slope of regression line */
       *y_intercept = 0;         /*!< y intercept of regression line */
       *r2 = 0;                  /*!< coefficient of determination */

       /*! calculate various sums */
       for (int i = 0; i < nrItems; i++)
           if ((x[i] != NODATA) && (y[i] != NODATA))
           {
                nrValidItems++;
                SUMx += x[i];
                SUMy += y[i];
                SUMxy += x[i] * y[i];
                SUMxx += x[i] * x[i];
           }

        /*! means of x and y */
        AVGy = SUMy / nrValidItems;
        AVGx = SUMx / nrValidItems;

        if (!zeroIntercept)
        {
            *mySlope = float((SUMxy - SUMx * SUMy / nrValidItems) / (SUMxx - SUMx * SUMx / nrValidItems));
            *y_intercept = float(AVGy - *mySlope * AVGx);
        }
        else
        {
            *mySlope = float(SUMxy / SUMxx);
            *y_intercept = 0.;
        }

        /*! calculate squared residues and their sum */
        for (int i = 0; i < nrItems; i++)
           if ((x[i] != NODATA) && (y[i] != NODATA))
           {
              /*! sum of squared of the discrepancies */
              dy = y[i] - (*y_intercept + *mySlope * x[i]);
              dy *= dy;
              SUM_dy += dy;

              /*! sum of squared residues */
              res = y[i] - AVGy;
              res *= res;
              SUMres += res;
           }

        /*! calculate r^2 (coefficient of determination) */
        *r2 = float((SUMres - SUM_dy) / SUMres);
    }

    /*! Variance */
    float variance(float *myList, int nrList)
    {
        float myMean, myDiff, squareDiff;
        int i, nrValidValues;

        if (nrList <= 1) return NODATA;

        myMean = mean(myList,nrList);

        squareDiff = 0;
        nrValidValues = 0;
        for (i = 0; i<nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                myDiff = (myList[i] - myMean);
                squareDiff += (myDiff * myDiff);
                nrValidValues++;
            }
        }

        if (nrValidValues > 1)
            return (squareDiff / (nrValidValues - 1));
        else
            return NODATA;
    }

    float variance(std::vector<float> myList, int nrList)
    {
        float myMean, myDiff, squareDiff;
        int i, nrValidValues;

        if (nrList <= 1) return NODATA;

        myMean = mean(myList);

        squareDiff = 0;
        nrValidValues = 0;
        for (i = 0; i<nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                myDiff = (myList[i] - myMean);
                squareDiff += (myDiff * myDiff);
                nrValidValues++;
            }
        }

        if (nrValidValues > 1)
            return (squareDiff / (nrValidValues - 1));
        else
            return NODATA;
    }

    double variance(std::vector<double> myList, int nrList)
    {
        double myMean, myDiff, squareDiff;
        int i, nrValidValues;

        if (nrList <= 1) return NODATA;

        myMean = mean(myList);

        squareDiff = 0;
        nrValidValues = 0;
        for (i = 0; i<nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                myDiff = (myList[i] - myMean);
                squareDiff += (myDiff * myDiff);
                nrValidValues++;
            }
        }

        if (nrValidValues > 1)
            return (squareDiff / (nrValidValues - 1));
        else
            return NODATA;
    }

    double variance(double *myList, int nrList)
    {
        double myMean, myDiff, squareDiff;
        int i, nrValidValues;

        if (nrList <= 1) return NODATA;

        myMean = mean(myList,nrList);

        squareDiff = 0;
        nrValidValues = 0;
        for (i = 0; i<nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                myDiff = (myList[i] - myMean);
                squareDiff += (myDiff * myDiff);
                nrValidValues++;
            }
        }

        if (nrValidValues > 1)
            return (squareDiff / (nrValidValues - 1));
        else
            return NODATA;
    }

    float mean(float *myList, int nrList)
    {
        float sum=0.;
        int i, nrValidValues;

        if (nrList < 1) return NODATA;
        nrValidValues = 0;

        for (i = 0; i < nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                sum += myList[i];
                nrValidValues++;
            }
        }

        if (nrValidValues > 0)
            return (sum/(float)(nrValidValues));
        else
            return NODATA;
    }


    float mean(std::vector<float> list)
    {
        if (list.size() < 1)
            return NODATA;

        int nrValidValues = 0;
        double sum = 0.;

        for (int i = 0; i < list.size(); i++)
        {
            if (! isEqual(list[i], NODATA))
            {
                sum += double(list[i]);
                nrValidValues++;
            }
        }

        if (nrValidValues == 0)
            return NODATA;

        return float(sum / double(nrValidValues));
    }


    double mean(std::vector<double> list)
    {
        if (list.size() < 1) return NODATA;

        int nrValidValues = 0;
        double sum=0;

        for (int i = 0; i < int(list.size()); i++)
        {
            if (list[i] != NODATA)
            {
                sum += list[i];
                nrValidValues++;
            }
        }

        if (nrValidValues > 0)
            return (sum / (double)(nrValidValues));
        else
            return NODATA;
    }


    double mean(double *myList, int nrList)
    {
        double sum=0.;
        int i, nrValidValues;

        if (nrList < 1) return NODATA;
        nrValidValues = 0;

        for (i = 0; i < nrList; i++)
        {
            if (myList[i]!= NODATA)
            {
                sum += myList[i];
                nrValidValues++;
            }
        }

        if (nrValidValues > 0)
            return (sum/(nrValidValues));
        else
            return NODATA;
    }

    float standardDeviation(float *myList, int nrList)
    {
        return sqrtf(variance(myList,nrList));
    }

    float standardDeviation(std::vector<float> myList, int nrList)
    {
        return sqrtf(variance(myList,nrList));
    }

    double standardDeviation(std::vector<double> myList, int nrList)
    {
        return sqrt(variance(myList,nrList));
    }

    double standardDeviation(double *myList, int nrList)
    {
        return sqrt(variance(myList,nrList));
    }

    /*! covariance */
    float covariance(float *myList1, int nrList1,float *myList2, int nrList2)
    {
        float myMean1,myMean2,myDiff1,myDiff2,prodDiff;
        int i, nrValidValues;

        if (nrList1 <= 1) return NODATA;
        if (nrList2 <= 1) return NODATA;
        if (nrList2 != nrList1) return NODATA;

        myMean1 = mean(myList1,nrList1);
        myMean2 = mean(myList2,nrList2);
        prodDiff = 0;
        nrValidValues = 0;
        for (i = 0;i<nrList1;i++)
        {
            if ((myList1[i]!= NODATA)&&myList2[i]!=NODATA)
            {
                myDiff1 = (myList1[i] - myMean1);
                myDiff2 = (myList2[i] - myMean2);
                prodDiff += myDiff1*myDiff2;
                nrValidValues++;
            }
        }
        return (prodDiff / (nrValidValues - 1));
    }
    double covariance(double *myList1, int nrList1,double *myList2, int nrList2)
    {
        double myMean1,myMean2,myDiff1,myDiff2,prodDiff;
        int i, nrValidValues;

        if (nrList1 <= 1) return NODATA;
        if (nrList2 <= 1) return NODATA;
        if (nrList2 != nrList1) return NODATA;

        myMean1 = mean(myList1,nrList1);
        myMean2 = mean(myList2,nrList2);
        prodDiff = 0;
        nrValidValues = 0;
        for (i = 0;i<nrList1;i++)
        {
            if ((myList1[i]!= NODATA)&&myList2[i]!=NODATA)
            {
                myDiff1 = (myList1[i] - myMean1);
                myDiff2 = (myList2[i] - myMean2);
                prodDiff += myDiff1*myDiff2;
                nrValidValues++;
            }
        }
        return (prodDiff / (nrValidValues - 1));
    }

    double meanNoCheck(double *myList, int nrList)
    {
        double sum=0.;
        //int i;// nrValidValues;

        //if (nrList < 1) return NODATA;
        //nrValidValues = 0;

        for (int i = 0; i < nrList; i++)
        {
            //if (myList[i]!= NODATA)
            //{
                sum += myList[i];
                //nrValidValues++;
            //}
        }

        //if (nrValidValues > 0)
            return (sum/(nrList));
        //else
            //return NODATA;
    }
    double covarianceNoCheck(double *myList1, int nrList1,double *myList2, int nrList2)
    {
        double myMean1,myMean2,myDiff1,myDiff2,prodDiff;
        int i;//, nrValidValues;

        //if (nrList1 <= 1) return NODATA;
        //if (nrList2 <= 1) return NODATA;
        //if (nrList2 != nrList1) return NODATA;

        myMean1 = meanNoCheck(myList1,nrList1);
        myMean2 = meanNoCheck(myList2,nrList2);
        prodDiff = 0;
        //nrValidValues = 0;
        for (i = 0;i<nrList1;i++)
        {
            //if ((myList1[i]!= NODATA)&&myList2[i]!=NODATA)
            //{
                myDiff1 = (myList1[i] - myMean1);
                myDiff2 = (myList2[i] - myMean2);
                prodDiff += myDiff1*myDiff2;
                //nrValidValues++;
            //}
        }
        return (prodDiff / (--nrList1));
    }

    double varianceNoCheck(double *myList, int nrList)
    {
        double myMean, myDiff, squareDiff;
        int i;//, nrValidValues;

        //if (nrList <= 1) return NODATA;

        myMean = meanNoCheck(myList,nrList);

        squareDiff = 0;
        //nrValidValues = 0;
        for (i = 0; i<nrList; i++)
        {
            //if (myList[i]!= NODATA)
            //{
                myDiff = (myList[i] - myMean);
                squareDiff += (myDiff * myDiff);
                //nrValidValues++;
            //}
        }

        //if (nrValidValues > 1)
            return (squareDiff / (--nrList));
        //else
            //return NODATA;
    }

    float coefficientPearson(float *myList1, int nrList1,float *myList2, int nrList2)
    {
        return (covariance(myList1,nrList1,myList2,nrList2)/(standardDeviation(myList1,nrList1)*standardDeviation(myList2,nrList2)));
    }

    float** covariancesMatrix(int nrRowCol, float**myLists,int nrLists)
    {
        float** c = (float**)calloc(nrRowCol, sizeof(float*));
        for(int i = 0;i<nrRowCol;i++)
        {
            c[i] = (float*)calloc(nrRowCol, sizeof(float));
        }
        for(int i = 0;i<nrRowCol;i++)
        {
            c[i][i]= variance(myLists[i],nrLists);
            for(int j = i+1;j<nrRowCol;j++)
            {
                c[i][j]= covariance(myLists[i],nrLists,myLists[j],nrLists);
                c[j][i]=c[i][j];
            }

        }
        return c;
    }

    void correlationsMatrix(int nrRowCol, double**myLists,int nrLists, double** c)
    {
        // input: myLists matrix
        // output: c matrix

        for(int i = 0;i<nrRowCol;i++)
        {
            c[i][i]=1.;
            for(int j = i+1;j<nrRowCol;j++)
            {
                c[i][j]= covariance(myLists[i],nrLists,myLists[j],nrLists);
                if (c[i][j] != 0) c[i][j] /= sqrt(variance(myLists[i],nrLists)*variance(myLists[j],nrLists));
                c[j][i] = c[i][j];
            }

        }
    }

    void correlationsMatrixNoCheck(int nrRowCol, double**myLists,int nrLists, double** c)
    {
        // input: myLists matrix
        // output: c matrix

        for(int i = 0;i<nrRowCol;i++)
        {
            c[i][i]=1.;
            for(int j = i+1;j<nrRowCol;j++)
            {
                c[i][j]= covarianceNoCheck(myLists[i],nrLists,myLists[j],nrLists);
                if (c[i][j] != 0)
                    c[i][j] /= sqrt(varianceNoCheck(myLists[i],nrLists)*varianceNoCheck(myLists[j],nrLists));
                c[j][i] = c[i][j];
            }

        }
    }



    float maxList(std::vector<float> values, int nValue)
    {

        float max = -FLT_MAX;

        if (nValue == 0)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] > max) && (values[i] != NODATA))
            {
                max = values[i] ;
            }
        }

        return max;
    }


    float minList(std::vector<float> values, int nValue)
    {

        float min = FLT_MAX;

        if (nValue == 0)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] < min) && (values[i] != NODATA))
            {
                min = values[i] ;
            }
        }

        return min;
    }

    float sumList(std::vector<float> values, int nValue)
    {

        float sum = 0;

        if (nValue == 0)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if (values[i] != NODATA)
                sum += values[i] ;
        }

        return sum;
    }

    float sumListThreshold(std::vector<float> values, int nValue, float threshold)
    {

        float sum = 0;

        if (nValue == 0 || threshold == NODATA)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] > threshold) && (values[i] != NODATA))
            {
                sum += ( values[i] - threshold);
            }
        }

        return sum;
    }

    float diffListThreshold(std::vector<float> values, int nValue, float threshold)
    {

        float diff = 0;

        if (nValue == 0 || threshold == NODATA)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] < threshold)&& (values[i] != NODATA))
            {
                diff += (threshold - values[i]);
            }
        }

        return diff;
    }


    float countAbove(std::vector<float> values, int nValue, float threshold)
    {

        float countAbove = 0;

        if (nValue == 0 || threshold == NODATA)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] > threshold)&& (values[i] != NODATA))
            {
                countAbove++;
            }
        }

        return countAbove;
    }

    float countBelow(std::vector<float> values, int nValue, float threshold)
    {

        float countBelow = 0;

        if (nValue == 0 || threshold == NODATA)
            return NODATA;

        for (int i = 0; i < nValue; i++)
        {
            if ((values[i] < threshold) && (values[i] != NODATA))
            {
                countBelow++;
            }
        }

        return countBelow;
    }

    float countConsecutive(std::vector<float> values, int nValue, float threshold, bool isPositive)
    {

        float countConsecutive = 0;

        if (nValue == 0 || threshold == NODATA)
            return NODATA;

        bool inPeriod = false;
        int nPeriod = 0;

        std::vector<float> myListNumDays;
        myListNumDays.push_back(0);

        for (int i = 0; i < nValue; i++)
        {
            if (values[i] != NODATA)
            {
                if (inPeriod == false)
                {
                    if (compareValue( values[i], threshold, isPositive))
                    {
                        inPeriod = true;
                        myListNumDays[nPeriod]++;
                    }
                }
                else
                {
                    if (compareValue( values[i], threshold, isPositive))
                    {
                        myListNumDays[nPeriod]++;
                        if (i == (nValue - 1))
                        {
                            nPeriod++;
                            myListNumDays.push_back(0);
                        }
                    }
                    else
                    {
                        nPeriod++;
                        myListNumDays.push_back(0);
                        inPeriod = false;
                    }
                }
            }
            else if (inPeriod == true)
            {
                nPeriod++;
                myListNumDays.push_back(0);
                inPeriod = false;
            }
        }
        if (nPeriod == 0)
        {
            countConsecutive = 0;
        }
        else
        {
            countConsecutive = statistics::maxList(myListNumDays, nPeriod);
        }

        return countConsecutive;

    }

    float frequencyPositive(std::vector<float> values, int nValue)
    {

        if (nValue <= 0)
            return NODATA;

        float frequencyPositive = 0;

        for (int i = 0; i < nValue; i++)
        {
            if ( (values[i] > 0) && (values[i] != NODATA))
                frequencyPositive++;

        }
        frequencyPositive /= nValue;

        return frequencyPositive;

    }

    float trend(std::vector<float> values, int nValues, float myFirstYear)
    {

        float trend;
        float r2, y_intercept;


        if (nValues < 2 || myFirstYear == NODATA)
            return NODATA;

        std::vector<float> myYears(nValues);

        for (int i = 0; i < nValues; i++)
        {
            myYears[i] = (myFirstYear + i) ;
        }

        statistics::linearRegression(myYears, values, nValues, false, &y_intercept, &trend, &r2);
        return trend;

    }

    float mannKendall(std::vector<float> values, int nValues)
    {

        double variable;
        float zMK;

        // minimum 3 values
        if (nValues < 3)
            return NODATA;

        int myS = 0;
        int myValidNR = 0;

        for (int i = 0; i < nValues - 1; i++)
        {
             if ( values[i] != NODATA )
             {
                for (int j = i + 1; j < nValues ; j++)
                {
                    if (values[j] != NODATA)
                    {
                        myS += sgn(values[j] - values[i]);
                    }
                }
                myValidNR++;
            }
        }
        variable = myValidNR * (myValidNR - 1) * (2.0 * myValidNR + 5.0) / 18.0;

        if (myS > 0)
        {
            zMK = static_cast<float>((myS - 1) / sqrt(variable));
        }
        else if (myS < 0)
        {
            zMK = -static_cast<float>((myS + 1) / sqrt(variable));
        }
        else
        {
            return 0;
        }

        int stdDev = 1;
        int mean = 0;

        double sumGauss = 0.0;
        double deltaXGauss = 0.001;
        double myX = 0.0;
        std::vector<float> GaussIntegralTwoTailsFactor1000(10000);

        for (unsigned int i = 0; i < 10000; i++)
        {
            myX += deltaXGauss;
            double ratio = (myX - mean) / stdDev;
            double gauss = (1 / (sqrt(2 * PI) * stdDev)) * exp(-0.5 * (ratio * ratio));
            sumGauss += gauss * deltaXGauss;
            GaussIntegralTwoTailsFactor1000[i] = float(sumGauss * 2.f);
        }
        float result = GaussIntegralTwoTailsFactor1000[int(zMK * 1000)];
        return result;

    }

    bool rollingAverage(double* arrayInput, int sizeArray, int lag,double* arrayOutput)
    {
        if (lag < 0 )
        {
            return false;
        }
        for (int i=0;i<sizeArray;i++)
        {
            arrayOutput[i] = 0;
        }
        int counter = 0;
        for (int i=0;i<sizeArray;i++)
        {
            counter = 0;
            if (fabs(arrayInput[i] - NODATA)>EPSILON)
            {
                arrayOutput[i]+= arrayInput[i];
                counter++;
            }
            if (lag > 0)
            {
                for (int j=1; j<=lag;j++)
                {
                    if (((i-j)>=0) && (fabs(arrayInput[i-j] - NODATA) > EPSILON))
                    {
                        arrayOutput[i] += arrayInput[i-j];
                        counter++;
                    }
                    if (((i+j) < sizeArray) && (fabs(arrayInput[i+j] - NODATA)> EPSILON))
                    {
                        arrayOutput[i] += arrayInput[i+j];
                        counter++;
                    }
                }

            }
            if (counter > 0)
                arrayOutput[i] /= counter;
            else
                arrayOutput[i] = NODATA;
        }
        return true;
    }

}

namespace stat_openai
{
    // Funzione per calcolare la trasposta di una matrice
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) {
        int rows = int(matrix.size());
        int cols = int(matrix[0].size());

        std::vector<std::vector<float>> result(cols, std::vector<float>(rows));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    // Funzione per calcolare la regressione lineare multipla
    std::vector<double> multipleLinearRegression(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
    {
        int numSamples = int(X.size());
        int numFeatures = int(X[0].size());

        // Calcola la matrice X^T * X
        std::vector<std::vector<double>> XTX(numFeatures, std::vector<double>(numFeatures));
        for (int i = 0; i < numFeatures; ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                for (int k = 0; k < numSamples; ++k) {
                    XTX[i][j] += X[k][i] * X[k][j];
                }
            }
        }

        // Calcola il vettore X^T * y
        std::vector<double> XTy(numFeatures);
        for (int i = 0; i < numFeatures; ++i) {
            for (int j = 0; j < numSamples; ++j) {
                XTy[i] += X[j][i] * y[j];
            }
        }

        // Risoluzione del sistema di equazioni lineari per ottenere i coefficienti
        std::vector<double> coefficients(numFeatures);
        for (int i = 0; i < numFeatures; ++i) {
            coefficients[i] = XTy[i] / XTX[i][i];
        }

        return coefficients;
    }

}
