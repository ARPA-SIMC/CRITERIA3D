/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    Contacts:
    Antonio Volta  avolta@arpae.it
*/

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "commonConstants.h"
#include "furtherMathFunctions.h"

double functionTemperatureVsHeight(double* x, double* par,int xDim,int nrParameters)
{
    double y;
    y = par[0] + par[1]*x[0] + par[2]*(1/(1+exp(-par[3]*(x[0] - par[4]))));
    return y;
}
double bilinear(double* x, double* par,int xDim,int nrParameters)
{
    double y = 0;
    for (int i=0;i<2;i++)
        y += par[i] * x[i];
    y += par[2];
    return y;
}

double multilinear(double* x, double* par, int xDim, int nrParameters)
{
    double y = 0;
    for (int i=0;i<xDim;i++)
        y += par[i] * x[i];
    y += par[xDim];
    return y;
}

double parabolicFunction(double* x, double* par)
{
    return par[1]*(x[0] - par[0])*(x[0] - par[0]) + par[2] ;
}
float lapseRateSigmoidalFunction(float x, float par1, float par2, float par3, float par4, float par5)
{
    // par1 correponds to T0
    // par2 descent slope (negative value)
    // par3 magnitude of the sigmoid par2 >= 0, if par2==0 no sigmoid
    // par4 steepness of the sigmoid function
    // par5 center of the sigmoid
    return par1 + par2*x + par3*(1/(1+exp(-par4*(x-par5))));
}
float gaussianFunction(TfunctionInput fInput)
{
    double y = (1 / (fInput.par[1] * sqrt(2*PI))
            * exp(-0.5*(fInput.x-fInput.par[0])*(fInput.x-fInput.par[0])/(fInput.par[1]*fInput.par[1])));
    return float(y);
}

float gaussianFunction(float x, float mean, float devStd)
{
    double devStd_d = MAXVALUE(double(devStd), 0.00001);
    double ratio = double(x - mean) / devStd_d;
    double y = 1 / (devStd_d * sqrt(2*PI)) * exp(-0.5*(ratio * ratio));
    return float(y);
}

float errorFunctionPrimitive(float x)
{
    return expf(-x*x);
}

double errorFunctionPrimitive(double x)
{
    return exp(-x*x);
}

float blackBodyShape(TfunctionInput fInput)
{
    float b, y;
    b = fInput.par[0];
    y = b * float(pow(fInput.x, 3)*(1. / (exp(b*fInput.x)-0.99)));
    return (y);
}
double parabolicFunction(double x, double* par)
{
    return par[1]*(x - par[0])*(x - par[0]) + par[2] ;
}

double twoParametersAndExponentialPolynomialFunctions(double x, double* par)
{
    return double(par[0]+par[1]*pow(x,par[2]));
}

double twoHarmonicsFourier(double x, double* par)
{
    return par[0] + par[1]*cos(2*PI/par[5]*x) + par[2]*sin(2*PI/par[5]*x) + par[3]*cos(4*PI/par[5]*x) + par[4]*sin(4*PI/par[5]*x);
}

double harmonicsFourierGeneral(double x, double* par,int nrPar)
{
    // the last parameter is the period
    if (nrPar == 2) return par[0];
    else
    {
        int counter = 0;
        double y = par[0];
        int requiredHarmonics;
        requiredHarmonics = (nrPar - 2)/2;
        double angularVelocity;
        angularVelocity = 2*PI/par[nrPar-1];
        while (counter < requiredHarmonics)
        {
            y += par[1+counter*2]*cos(angularVelocity*(counter+1)*x) + par[2+counter*2]*sin(angularVelocity*(counter+1)*x);
            counter++;
        }
        return y;
    }
}

/*float straightLine(TfunctionInput fInput)
{
    float m,q,y;
    m = fInput.par[0];
    q = fInput.par[1];
    y = m*fInput.x + q;
    return(y);
}
*/

namespace integration
{
    /*! this is a set for function integrations by means of the Simpson */

    float qsimpParametric(float (*func)(TfunctionInput), int nrPar, float *par,float a , float b , float EPS)
    {
        /*! this function calculates definte integrals using the Simpson rule */
        if (a > b)
        {
            return (-qsimpParametric(func,nrPar, par , b, a , EPS)); //recursive formula
        }
        float old_s [10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        //void nerror(char error_text[]);
        int j;
        float s = NODATA , st = NODATA , ost = 0.0 , os = 0.0 ;
        float s1 = 0.;
        for ( j=1 ; j <= 20 ; j++)
        {
            st = trapzdParametric(func,nrPar,par,a,b,j) ;
            s = float((4*st-ost)/3) ;
            for ( short k=1 ; k < 10 ; k++)
            {
                old_s[k-1]=old_s[k];
            }
            old_s[9] = s ;
            if (j == 5) s1 = s ;
            if (j > 5 )
            {
                if (fabs(s-os) < EPS*fabs(os) || (s == 0.0 && os == 0.0) ) return s ;
            }
            os = s ;
            ost = st ;
        }
        float average_s=0.0 , average_s2 = 0.0 , variance ;
        for ( short k=0 ; k < 10 ; k++)
        {
            average_s  += old_s[k];
            average_s2 += powf(old_s[k],2) ;
        }
        average_s  /= 10.f ;
        average_s2 /= 10.f ;
        variance = average_s2 - powf(average_s,2) ;

        if (variance < 0.01*fabs(s1)) return s ; // s is converging slowly
        else return average_s ; // s ondulates
    }


    float trapzdParametric(float (*func)(TfunctionInput) , int nrPar, float *par , float a , float b , int n)
    {
        float x, tnm, sum, del;
        static float s;
        TfunctionInput functionInput;

        if (nrPar > 0)
        {
            functionInput.par = (float *) calloc(nrPar, sizeof(float));
            functionInput.nrPar = nrPar ;
            for (int i = 0 ; i<nrPar ; i++)
            {
                functionInput.par[i]=par[i];
            }
        }
        else
        {
            functionInput.par = (float *) calloc(1, sizeof(float));
            functionInput.nrPar = 1 ;
            for (int i = 0 ; i<1 ; i++)
            {
                functionInput.par[i]=1;
            }
        }
        int it , j ;

        if (n == 1)
        {
            functionInput.x = a ;
            s = float(0.5*(b-a)*((*func)(functionInput))) ;
            functionInput.x = b ;
            s += float(0.5*(b-a)*((*func)(functionInput))) ;
            //return (s) ;
        }
        else
        {
            for (it = 1 , j = 1 ; j < n-1 ; j++ ) it <<= 1 ;
            tnm = (float)(it) ;
            del = (b-a) / tnm ;
            x = (float)(a + 0.5 * del) ;
            for(sum = 0.0 , j=1 ; j <= it ; j++ , x += del)
            {
                functionInput.x = x ;
                sum += (*func)(functionInput) ;
            }
            //s = (float)(0.5 * (s + (b-a)*sum/tnm)) ;
            s= (float)(0.5 * (s + (b-a)*sum/tnm)) ;
        }
        free(functionInput.par);
        return s;
    }

    float trapezoidalRule(float (*func)(float) , float a , float b , int n)
    {
        float x , tnm , sum , del ;
        static float sumInfinitesimal ;
        int it , j ;

        if (n == 1)
        {
            return (sumInfinitesimal=(float)(0.5*(b-a)* ((*func) (a) +(*func)(b)))) ;
        }
        else
        {
            for (it = 1 , j = 1 ; j < n-1 ; j++ ) it <<= 1 ;
            tnm = (float)(it) ;
            del = (b-a) / tnm ;
            x = (float)(a + 0.5 * del) ;
            for(sum = 0.0 , j=1 ; j <= it ; j++ , x += del) sum += (*func)(x) ;
            //s = (float)(0.5 * (s + (b-a)*sum/tnm)) ;
            return (sumInfinitesimal= (float)(0.5 * (sumInfinitesimal + (b-a)*sum/tnm))) ;
        }
    }


    float simpsonRule(float (*func)(float), float a, float b, float EPS)
    {
        /*! this function calculates definte integrals using the Simpson rule */
        if (a > b)
        {
            return (-simpsonRule(func,b, a , EPS)); //recursive formula
        }

        float old_s [10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        //float trapezoidalRule(float (*func)(float) , float a , float b , int n) ;
        int j;
        float sumInfenitesimal , sumTrapezoidal , old_sumTrapezoidal=0.0 , old_sumInfinitesimal = 0.0 ;
        float s1 = 0.;

        for ( j=1 ; j <= 20 ; j++)
        {
            sumTrapezoidal = trapezoidalRule(func,a,b,j) ;
            sumInfenitesimal = float((4*sumTrapezoidal-old_sumTrapezoidal)/3) ;
            for ( short k=1 ; k < 10 ; k++)
            {
                old_s[k-1]=old_s[k];
            }
            old_s[9] = sumInfenitesimal ;
            if (j == 5) s1 = sumInfenitesimal ;
            if (j > 5 )
            {
                if (fabs(sumInfenitesimal-old_sumInfinitesimal) < EPS*fabs(old_sumInfinitesimal)
                    || (sumInfenitesimal == 0 && old_sumInfinitesimal == 0) ) return sumInfenitesimal;
            }
            old_sumInfinitesimal = sumInfenitesimal ;
           old_sumTrapezoidal = sumTrapezoidal ;
        }

        float average_s=0.0, average_s2 = 0.0, variance;

        for (short k=0; k < 10; k++)
        {
            average_s  += old_s[k];
            average_s2 += old_s[k]*old_s[k] ;
        }
        average_s  /= 10.0 ;
        average_s2 /= 10.0 ;
        variance = average_s2 - average_s * average_s ;
        if (variance < 0.01*fabs(s1)) return sumInfenitesimal ; // s is converging slowly
        else return average_s ; // s ondulates

    }

    double trapezoidalRule(double (*func)(double) , double a , double b , int n)
    {
        double x , tnm , sum , del ;
        static double sumInfinitesimal ;
        int it , j ;

        if (n == 1)
        {
            return (sumInfinitesimal=0.5*(b-a)* ((*func) (a) +(*func)(b)));
        }
        else
        {
            for (it = 1 , j = 1 ; j < n-1 ; j++ ) it <<= 1 ;
            tnm = it;
            del = (b-a) / tnm ;
            x = a + 0.5 * del;
            for(sum = 0.0 , j=1 ; j <= it ; j++ , x += del) sum += (*func)(x) ;
            //s = (float)(0.5 * (s + (b-a)*sum/tnm)) ;
            return (sumInfinitesimal= 0.5 * (sumInfinitesimal + (b-a)*sum/tnm));
        }
    }


    double simpsonRule(double (*func)(double),double a , double b , double EPS)
    {
        /*! this function calculates definte integrals using the Simpson rule */
        if (a > b)
        {
            return (-simpsonRule(func,b, a , EPS)); //recursive formula
        }
        double old_s [10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        //double trapezoidalRule(double (*func)(double) , double a , double b , int n) ;
        int j;
        double sumInfenitesimal = 0. , sumTrapezoidal = 0. , old_sumTrapezoidal=0.0 , old_sumInfinitesimal = 0.0 ;
        double s1 = 0.;
        for ( j=1 ; j <= 20 ; j++)
        {
            sumTrapezoidal = trapezoidalRule(func,a,b,j) ;
            sumInfenitesimal = (4.0*sumTrapezoidal-old_sumTrapezoidal)/3.0 ;
            for ( short k=1 ; k < 10 ; k++)
            {
                old_s[k-1]=old_s[k];
            }
            old_s[9] = sumInfenitesimal ;
            if (j == 5) s1 = sumInfenitesimal ;
            if (j > 5)
            {
                if (fabs(sumInfenitesimal-old_sumInfinitesimal) < EPS*fabs(old_sumInfinitesimal)
                    || (sumInfenitesimal == 0.0 && old_sumInfinitesimal == 0.0) ) return sumInfenitesimal ;
            }
            old_sumInfinitesimal = sumInfenitesimal;
            old_sumTrapezoidal = sumTrapezoidal;
        }
        double average_s=0.0 , average_s2 = 0.0 , variance ;
        for ( short k=0 ; k < 10 ; k++)
        {
            average_s  += old_s[k];
            average_s2 += old_s[k]*old_s[k] ;
        }
        average_s  /= 10.0 ;
        average_s2 /= 10.0 ;
        variance = average_s2 - average_s * average_s ;
        if (variance < 0.01*fabs(s1)) return sumInfenitesimal ; // s is converging slowly
        else return average_s ; // s ondulates

    }
}


namespace interpolation
{
    double secant_method(float (*func) (float),  double x0, double x1)
    {
        double x2, fx0, fx1, error;

        for (int i = 0; i < MAX_NUMBER_ITERATIONS; i++)
        {
            fx0 = (*func)(x0);
            fx1 = (*func)(x1);

            // secant method formula
            x2 = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0);
            error = fabs(x2 - x1);
            //printf("Iteration %d: x = %.8f, f(x) = %.8f, error = %.8f\n", i+1, x2, fx2, error);
            if (error < EPSILON)
                return x2; // the roots was successfully approximated

            // Update the values for the next iteration
            x0 = x1;
            x1 = x2;
        }

        //printf("No converge after %d iterations.\n", MAX_ITER);
        return 0.0;
    }


    float linearInterpolation (float x, float *xColumn , float *yColumn, int dimTable )
    {
        //float *firstColumn = (float *) calloc(dimTable, sizeof(float));
        //float *secondColumn = (float *) calloc(dimTable, sizeof(float));
        //firstColumn = xColumn ;
        //secondColumn = yColumn ;
        float slope , offset ;
        short stage=1;
        if (x < xColumn[0]) return yColumn[0] ;
        if (x > xColumn[dimTable-1]) return yColumn[dimTable-1];
        while (x > xColumn[stage]) stage++ ;
        slope = (yColumn[stage]- yColumn[stage-1])/(xColumn[stage] - xColumn[stage-1]);
        offset = -xColumn[stage-1]*slope + yColumn[stage-1];
        //free(firstColumn);
        //free(secondColumn);
        return (slope * x + offset) ;
    }

    double linearInterpolation (double x, double *xColumn , double *yColumn, int dimTable )
    {
        //double *firstColumn = (double *) calloc(dimTable, sizeof(double));
        //double *secondColumn = (double *) calloc(dimTable, sizeof(double));
        //firstColumn = xColumn ;
        //secondColumn = yColumn ;
        double slope , offset ;
        short stage=1;
        if (x < xColumn[0]) return yColumn[0] ;
        if (x > xColumn[dimTable-1]) return yColumn[dimTable-1];
        while (x > xColumn[stage] && stage <= (dimTable-1)) stage++ ;
        slope = (yColumn[stage]- yColumn[stage-1])/(xColumn[stage] - xColumn[stage-1]);
        offset = -xColumn[stage-1]*slope + yColumn[stage-1];
        //free(firstColumn);
        //free(secondColumn);
        return (slope * x + offset) ;
    }

    float linearExtrapolation(double x3,double x1,double y1,double x2 , double y2)
    {
        double m,y3 ;
        m = (y1 - y2)/(x1-x2);
        y3 = m*(x3-x2) + y2 ;
        return float(y3) ;
    }


    /* -------------------------------------------------------
     parametersMin              parameters minimum values
     parametersMax              parameters maximum values
     parameters                 parameters first guess values
    ----------------------------------------------------------- */
    bool fittingMarquardt(double* parametersMin, double* parametersMax, double* parameters, int nrParameters,
                          double* parametersDelta, int maxIterationsNr, double myEpsilon, int idFunction,
                          double* x, double* y, int nrData)
    {
        // Sum of Squared Erros
        double mySSE, diffSSE, newSSE;
        static double VFACTOR = 10;

        double* paramChange = (double *) calloc(nrParameters, sizeof(double));
        double* newParameters = (double *) calloc(nrParameters, sizeof(double));
        double* lambda = (double *) calloc(nrParameters, sizeof(double));

        for(int i = 0; i < nrParameters; i++)
        {
            lambda[i] = 0.01;       // damping parameter
            paramChange[i] = 0;
        }

        mySSE = normGeneric(idFunction, parameters, nrParameters, x, y, nrData);

        int iterationNr = 0;
        do
        {
            leastSquares(idFunction, parameters, nrParameters, x, y, nrData, lambda, parametersDelta, paramChange);

            // change parameters
            for (int i = 0; i < nrParameters; i++)
            {
                newParameters[i] = parameters[i] + paramChange[i];
                if ((newParameters[i] > parametersMax[i]) && (lambda[i] < 1000))
                {
                    newParameters[i] = parametersMax[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
                if (newParameters[i] < parametersMin[i])
                {
                    newParameters[i] = parametersMin[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
            }

            newSSE = normGeneric(idFunction, newParameters, nrParameters, x, y, nrData);

            if (newSSE == NODATA)
            {
                // free memory
                free(lambda);
                free(paramChange);
                free(newParameters);

                return false;
            }

            diffSSE = mySSE - newSSE ;

            if (diffSSE > 0)
            {
                mySSE = newSSE;
                for (int i = 0; i < nrParameters ; i++)
                {
                    parameters[i] = newParameters[i];
                    lambda[i] /= VFACTOR;
                }
            }
            else
            {
                for(int i = 0; i < nrParameters; i++)
                {
                    lambda[i] *= VFACTOR;
                }
            }

            iterationNr++;
        }
        while (iterationNr <= maxIterationsNr && fabs(diffSSE) > myEpsilon);

        // free memory
        free(lambda);
        free(paramChange);
        free(newParameters);

        return (fabs(diffSSE) <= myEpsilon);
    }


    void leastSquares(int idFunction, double* parameters, int nrParameters,
                      double* x, double* y, int nrData, double* lambda,
                      double* parametersDelta, double* parametersChange)
    {
        int i, j, k;
        double pivot, mult, top;

        double* g = (double *) calloc(nrParameters+1, sizeof(double));
        double* z = (double *) calloc(nrParameters+1, sizeof(double));
        double* firstEst = (double *) calloc(nrData+1, sizeof(double));

        double** a = (double **) calloc(nrParameters+1, sizeof(double*));
        double** P = (double **) calloc(nrParameters+1, sizeof(double*));

        for (i = 0; i < nrParameters+1; i++)
        {
                a[i] = (double *) calloc(nrParameters+1, sizeof(double));
                P[i] = (double *) calloc(nrData+1, sizeof(double));
        }

        // first set of estimates
        for (i = 0; i < nrData; i++)
        {
            firstEst[i] = estimateFunction(idFunction, parameters, nrParameters, x[i]);
        }

        // change parameters and compute derivatives
        for (i = 0; i < nrParameters; i++)
        {
            parameters[i] += parametersDelta[i];
            for (j = 0; j < nrData; j++)
            {
                double newEst = estimateFunction(idFunction, parameters, nrParameters, x[j]);
                P[i][j] = (newEst - firstEst[j]) / MAXVALUE(parametersDelta[i], EPSILON) ;
            }
            parameters[i] -= parametersDelta[i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] = 0;
                for (k = 0; k < nrData; k++)
                {
                    a[i][j] += P[i][k] * P[j][k];
                }
            }
            z[i] = sqrt(a[i][i]) + EPSILON; //?
        }

        for (i = 0; i < nrParameters; i++)
        {
            g[i] = 0.;
            for (k = 0 ; k<nrData ; k++)
            {
                g[i] += P[i][k] * (y[k] - firstEst[k]);
            }
            g[i] /= z[i];
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] /= (z[i] * z[j]);
            }
        }

        for (i = 0; i < (nrParameters+1); i++)
        {
            a[i][i] += lambda[i];
            for (j = i+1; j < nrParameters; j++)
            {
                a[j][i] = a[i][j];
            }
        }

        for (j = 0; j < (nrParameters - 1); j++)
        {
            pivot = a[j][j];
            for (i = j + 1 ; i < nrParameters; i++)
            {
                mult = a[i][j] / pivot;
                for (k = j + 1; k < nrParameters; k++)
                {
                    a[i][k] -= mult * a[j][k];
                }
                g[i] -= mult * g[j];
            }
        }

        parametersChange[nrParameters - 1] = g[nrParameters - 1] / a[nrParameters - 1][nrParameters - 1];

        for (i = nrParameters - 2; i >= 0; i--)
        {
            top = g[i];
            for (k = i + 1; k < nrParameters; k++)
            {
                top -= a[i][k] * parametersChange[k];
            }
            parametersChange[i] = top / a[i][i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            parametersChange[i] /= z[i];
        }

        // free memory
        for (i = 0; i < nrParameters+1; i++)
        {
            free(a[i]);
            free(P[i]);
        }
        free(a);
        free(P);
        free(g);
        free(z);
        free(firstEst);
    }


    double estimateFunction(int idFunction, double *parameters, int nrParameters, double x)
    {
        switch (idFunction)
        {
            case FUNCTION_CODE_SPHERICAL :
                /*
                    parameters(0): range
                    parameters(1): nugget
                    parameters(2): sill
                */
                if (parameters[0] == 0)
                    return NODATA;
                if (x < parameters[0])
                {
                    double tmp = x / parameters[0];
                    return (parameters[1] + (parameters[2] - parameters[1]) * (1.5 * tmp - 0.5 * tmp * tmp * tmp));
                }
                else
                    return parameters[2];

            case FUNCTION_CODE_TWOPARAMETERSPOLYNOMIAL :
                return twoParametersAndExponentialPolynomialFunctions(x, parameters);

            case FUNCTION_CODE_FOURIER_2_HARMONICS :
                return twoHarmonicsFourier(x, parameters);

            case FUNCTION_CODE_FOURIER_GENERAL_HARMONICS :
                return harmonicsFourierGeneral(x, parameters, nrParameters);

            case FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN :
                return modifiedVanGenuchten(x, parameters, false);

            case FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN_RESTRICTED :
                return modifiedVanGenuchten(x, parameters, true);

            case FUNCTION_CODE_PARABOLIC :
            return parabolicFunction(x, parameters);

            default:
                return NODATA ;
        }
    }


    double normGeneric(int idFunction, double *parameters,int nrParameters, double *x, double *y, int nrData)
    {
        double estimate, error;
        double norm = 0;

        for (int i = 0; i < nrData; i++)
        {
            estimate = estimateFunction(idFunction, parameters, nrParameters, x[i]);
            if (estimate == NODATA)
            {
                return NODATA;
            }
            error = y[i] - estimate;
            norm += error * error;
        }

        return norm;
    }


    /*!
     * \brief Compute soil water content from water potential
     * \param water potential (psi) [kPa]
     * \return volumetric water content [m^3 m-3]
     */
    double modifiedVanGenuchten(double psi, double *parameters, bool isRestricted)
    {
        psi = fabs(psi);
        double thetaS, thetaR, he;
        double alpha, n, m;

        thetaS = parameters[0];         // water content at saturation [m^3 m^-3]
        thetaR = parameters[1];         // water content residual [m^3 m^-3]
        he = parameters[2];             // air entry [kPa]

        if (psi <= he) return thetaS;

        alpha = parameters[3];          // Van Genuchten curve parameter [kPa^-1]
        n = parameters[4];              // Van Genuchten curve parameter [-]
        if (isRestricted) {
            m = 1 - 1/n;                // Van Genuchten curve parameter (restricted: 1-1/n) [-]
        }
        else {
            m = parameters[5];
        }

        // reduction factor for modified VG (Ippisch, 2006) [-]
        double sc = pow(1 + pow(alpha * he, n), -m);

        // degree of saturation [-]
        double Se = pow(1 + pow(alpha * psi, n), -m) / sc;

        // volumetric water content [m^3 m^-3]
        return Se * (thetaS - thetaR) + thetaR;
    }


    double cubicSpline(double x, double *firstColumn, double *secondColumn, int dim)
    {
        double a,b,c,d,y;
        int i = 0;
        double *secondDerivative = (double *) calloc(dim, sizeof(double));

        for (int i=0 ; i < dim; i++)
        {
            secondDerivative[i] = NODATA;
        }

        punctualSecondDerivative(dim, firstColumn, secondColumn, secondDerivative);

        while (x > firstColumn[i])
            i++;

        double step = (firstColumn[i]- firstColumn[i-1]);
        a = (firstColumn[i] - x)/ step;
        b = 1 - a;
        d = c = step*step/6;
        c *= (a*a*a - a);
        d *= (b*b*b - b);
        y = a*secondColumn[i-1]+b*secondColumn[i]+c*secondDerivative[i-1]+d*secondDerivative[i];

        free(secondDerivative);
        return y ;
    }

    void punctualSecondDerivative(int dim, double *firstColumn , double *secondColumn, double* secondDerivative)
    {
        int matrixDimension;
        matrixDimension = dim-2;
        double *y2 = (double *) calloc(matrixDimension, sizeof(double));
        double *constantTerm = (double *) calloc(matrixDimension, sizeof(double));
        double *diagonal =  (double *) calloc(matrixDimension, sizeof(double));
        double *subDiagonal =  (double *) calloc(matrixDimension, sizeof(double));
        double *superDiagonal =  (double *) calloc(matrixDimension, sizeof(double));

        for (int i=0 ; i < matrixDimension; i++)
        {
            y2[i] = 0;
            diagonal[i] = (firstColumn[i+2]-firstColumn[i])/3 ;
            subDiagonal[i] = (firstColumn[i+1]-firstColumn[i])/6;
            superDiagonal[i] = (firstColumn[i+2]-firstColumn[i+1])/6;
            constantTerm[i] = (secondColumn[i+2]-secondColumn[i+1])/(firstColumn[i+2]-firstColumn[i+1])
                    -(secondColumn[i+1]-secondColumn[i])/(firstColumn[i+1]-firstColumn[i]);
        }

        tridiagonalThomasAlgorithm(matrixDimension,subDiagonal,diagonal,superDiagonal,constantTerm,y2);

        for (int i = 0 ; i < dim ; i++) secondDerivative[i]= 0;
        for (int i = 1 ; i < dim-1 ; i++) secondDerivative[i] = y2[i-1];

        free(y2);
        free(constantTerm);
        free(diagonal);
        free(subDiagonal);
        free(superDiagonal);

    }

    void tridiagonalThomasAlgorithm (int n, double *subDiagonal, double *mainDiagonal, double *superDiagonal, double *constantTerm, double* output)
    {
        // * n - number of equations
        // * subDiagonal - sub-diagonal (means it is the diagonal below the main diagonal) -- indexed from 1..n-1
        // * b - the main diagonal
        // * c - sup-diagonal (means it is the diagonal above the main diagonal) -- indexed from 0..n-2
        // * v - right part
        // * output - the answer

        double *newDiagonal, *newConstantTerm;
        newDiagonal = (double *) calloc(n, sizeof(double));
        newConstantTerm =   (double *) calloc(n, sizeof(double));

        newDiagonal[0] = mainDiagonal[0];
        newConstantTerm[0]= constantTerm[0];
        for (int i = 1; i < n; i++)
        {
                double m = subDiagonal[i]/mainDiagonal[i-1];
                newDiagonal[i] = mainDiagonal[i] - m*superDiagonal[i-1];
                newConstantTerm[i] = constantTerm[i] - m*constantTerm[i-1];
        }

        output[n-1] = newConstantTerm[n-1]/newDiagonal[n-1];
        for (int i = n - 2; i >= 0; i--)
                output[i]=(newConstantTerm[i]-superDiagonal[i]*output[i+1])/newDiagonal[i];

        free(newDiagonal);
        free(newConstantTerm);
    }

    double computeR2(double *obs, double* sim, int nrPoints)
    {
        double R2=0;
        double meanObs=0;
        double RSS=0;
        double TSS=0;
        for (int i=0;i<nrPoints;i++)
        {
            meanObs += obs[i];
        }
        meanObs /= nrPoints;
        //compute RSS and TSS
        for (int i=0;i<nrPoints;i++)
        {
            RSS += (obs[i]-sim[i])*(obs[i]-sim[i]);
            TSS += (obs[i]-meanObs)*(obs[i]-meanObs);
        }
        R2 = 1 - RSS/TSS;
        return R2;
    }

    int bestFittingMarquardt_nDimension(double (*func)(double*, double*,int,int), int nrTrials, int nrMinima,
                                                        double* parametersMin, double* parametersMax, double* parameters, int nrParameters,
                                                        double* parametersDelta, int maxIterationsNr, double myEpsilon,
                                                        double** x, double* y, int nrData, int xDim, bool isWeighted, double* weights)
    {
        double bestR2 = -9999;
        double R2;
        double* R2Previous = (double *) calloc(nrMinima, sizeof(double));
        double* ySim = (double *) calloc(nrData, sizeof(double));
        double* bestParameters = (double *) calloc(nrParameters, sizeof(double));
        double* xPoint = (double *) calloc(nrParameters, sizeof(double));
        int i;
        int iRandom = 0;
        int counter = 0;
        for (i=0; i<nrMinima; i++)
        {
            R2Previous[i] = NODATA;
        }
        srand (time(nullptr));
        do
        {
            for (i=0;i<nrParameters;i++)
            {
                parameters[i] = parametersMin[i] + ((double) rand() / (RAND_MAX))*(parametersMax[i]-parametersMin[i]);
            }
            fittingMarquardt_nDimension(func,parametersMin,parametersMax,parameters,nrParameters,parametersDelta,maxIterationsNr,myEpsilon,x,y,nrData,xDim,isWeighted,weights);
            for (i=0;i<nrData;i++)
            {
                //double xSim;
                //xSim = x[i][0];
                for (int k=0; k<xDim; k++)
                {
                    xPoint[k] = x[i][k];
                }
                ySim[i]= func(xPoint,parameters,xDim,nrParameters);//
            }
            R2 = computeR2(y,ySim,nrData);
            //printf("%d R2 = %f\n",iRandom,R2);
            if (R2 > bestR2-EPSILON)
            {
                for (int j=0;j<nrMinima-1;j++)
                {
                    R2Previous[j] = R2Previous[j+1];
                }
                R2Previous[nrMinima-1] = R2;
                bestR2 = R2;
                for (i=0;i<nrParameters;i++)
                {
                    bestParameters[i] = parameters[i];
                }
            }
            iRandom++;
            counter++;
        } while(iRandom<nrTrials && R2<(1 - EPSILON) && fabs(R2Previous[0]-R2Previous[nrMinima-1])>0.0001);

        for (i=0;i<nrParameters;i++)
        {
            parameters[i] = bestParameters[i];
        }
        free(xPoint);
        free(bestParameters);
        free(ySim);
        free(R2Previous);
        return counter;
    }

    bool fittingMarquardt_nDimension(double (*func)(double*, double*,int,int),double* parametersMin, double* parametersMax, double* parameters, int nrParameters,
                          double* parametersDelta, int maxIterationsNr, double myEpsilon,
                          double** x, double* y, int nrData, int xDim,bool isWeighted, double* weights)
    {
        // Sum of Squared Erros
        double mySSE, diffSSE, newSSE;
        static double VFACTOR = 10;

        double* paramChange = (double *) calloc(nrParameters, sizeof(double));
        double* newParameters = (double *) calloc(nrParameters, sizeof(double));
        double* lambda = (double *) calloc(nrParameters, sizeof(double));

        for(int i = 0; i < nrParameters; i++)
        {
            lambda[i] = 0.01;       // damping parameter
            paramChange[i] = 0;
        }

        mySSE = normGeneric_nDimension(func, parameters, nrParameters, x, y, nrData,xDim);

        int iterationNr = 0;
        do
        {
            leastSquares_nDimension(func, parameters, nrParameters, x, y, nrData,xDim, lambda, parametersDelta, paramChange,isWeighted,weights);
                // change parameters
            for (int i = 0; i < nrParameters; i++)
            {
                newParameters[i] = parameters[i] + paramChange[i];
                if ((newParameters[i] > parametersMax[i]) && (lambda[i] < 1000))
                {
                    newParameters[i] = parametersMax[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
                if (newParameters[i] < parametersMin[i])
                {
                    newParameters[i] = parametersMin[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
            }

            newSSE = normGeneric_nDimension(func, newParameters, nrParameters, x, y, nrData,xDim);

            if (newSSE == NODATA)
            {
                // free memory
                free(lambda);
                free(paramChange);
                free(newParameters);

                return false;
            }

            diffSSE = mySSE - newSSE ;

            if (diffSSE > 0)
            {
                mySSE = newSSE;
                for (int i = 0; i < nrParameters ; i++)
                {
                    parameters[i] = newParameters[i];
                    lambda[i] /= VFACTOR;
                }
            }
            else
            {
                for(int i = 0; i < nrParameters; i++)
                {
                    lambda[i] *= VFACTOR;
                }
            }

            iterationNr++;
        }
        while (iterationNr <= maxIterationsNr && fabs(diffSSE) > myEpsilon);

        // free memory
        free(lambda);
        free(paramChange);
        free(newParameters);


        return (fabs(diffSSE) <= myEpsilon);
    }

    void leastSquares_nDimension(double (*func)(double*, double*,int,int), double* parameters, int nrParameters,
                      double** x, double* y, int nrData,int xDim, double* lambda,
                      double* parametersDelta, double* parametersChange, bool isWeighted, double* weights)
    {
        int i, j, k;
        double pivot, mult, top;

        double* g = (double *) calloc(nrParameters+1, sizeof(double));
        double* z = (double *) calloc(nrParameters+1, sizeof(double));
        double* firstEst = (double *) calloc(nrData+1, sizeof(double));

        double** a = (double **) calloc(nrParameters+1, sizeof(double*));
        double** P = (double **) calloc(nrParameters+1, sizeof(double*));
        double* xPoint = (double *) calloc(xDim, sizeof(double));
        for (i = 0; i < nrParameters+1; i++)
        {
                a[i] = (double *) calloc(nrParameters+1, sizeof(double));
                P[i] = (double *) calloc(nrData+1, sizeof(double));
        }

        // first set of estimates
        for (i = 0; i < nrData; i++)
        {
            for (k=0; k<xDim; k++)
            {
                xPoint[k] = x[i][k];
            }
            firstEst[i] = func(xPoint,parameters,xDim,nrParameters);
            //firstEst[i] = estimateFunction_nDimensionExternalFunction(idFunction, parameters, nrParameters, xPoint,xDim);
        }

        // change parameters and compute derivatives
        for (i = 0; i < nrParameters; i++)
        {
            parameters[i] += parametersDelta[i];
            for (j = 0; j < nrData; j++)
            {
                for (k=0; k<xDim; k++)
                {
                    xPoint[k] = x[j][k];
                }
                double newEst = func(xPoint,parameters,xDim,nrParameters);
                //double newEst = estimateFunction_nDimensionExternalFunction(idFunction, parameters, nrParameters, xPoint,xDim);
                P[i][j] = (newEst - firstEst[j]) / MAXVALUE(parametersDelta[i], EPSILON) ;
            }
            parameters[i] -= parametersDelta[i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] = 0;
                for (k = 0; k < nrData; k++)
                {
                    if (isWeighted)
                    {
                        a[i][j] += (weights[k]*(P[i][k] * P[j][k]));
                    }
                    else
                    {
                        a[i][j] += (P[i][k] * P[j][k]);
                    }
                }
            }
            z[i] = sqrt(a[i][i]) + EPSILON; //?
        }

        for (i = 0; i < nrParameters; i++)
        {
            g[i] = 0.;
            for (k = 0 ; k<nrData ; k++)
            {
                g[i] += P[i][k] * (y[k] - firstEst[k]);
            }
            g[i] /= z[i];
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] /= (z[i] * z[j]);
            }
        }

        for (i = 0; i < (nrParameters+1); i++)
        {
            a[i][i] += lambda[i];
            for (j = i+1; j < nrParameters; j++)
            {
                a[j][i] = a[i][j];
            }
        }

        for (j = 0; j < (nrParameters - 1); j++)
        {
            pivot = a[j][j];
            for (i = j + 1 ; i < nrParameters; i++)
            {
                mult = a[i][j] / pivot;
                for (k = j + 1; k < nrParameters; k++)
                {
                    a[i][k] -= mult * a[j][k];
                }
                g[i] -= mult * g[j];
            }
        }

        parametersChange[nrParameters - 1] = g[nrParameters - 1] / a[nrParameters - 1][nrParameters - 1];

        for (i = nrParameters - 2; i >= 0; i--)
        {
            top = g[i];
            for (k = i + 1; k < nrParameters; k++)
            {
                top -= a[i][k] * parametersChange[k];
            }
            parametersChange[i] = top / a[i][i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            parametersChange[i] /= z[i];
        }

        // free memory
        for (i = 0; i < nrParameters+1; i++)
        {
            free(a[i]);
            free(P[i]);
        }
        free(a);
        free(P);
        free(g);
        free(z);
        free(firstEst);
        free(xPoint);
    }

    double normGeneric_nDimension(double (*func)(double*, double*,int,int), double *parameters,int nrParameters, double** x, double *y, int nrData, int xDim)
    {
        double estimate, error;
        double norm = 0;
        double* xPoint = (double *) calloc(xDim, sizeof(double));
        for (int i = 0; i < nrData; i++)
        {
            for (int j=0; j<xDim; j++)
            {
                xPoint[j] = x[i][j];
            }
            estimate = func(xPoint,parameters,xDim,nrParameters);
            if (estimate == NODATA)
            {
                free(xPoint);
                return NODATA;
            }
            error = y[i] - estimate;
            norm += error * error;
        }
        free(xPoint);
        return norm;
    }


    /* ////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////// */
    /* ////////////////////////////////////////////////////////////////// */

    int bestFittingMarquardt_nDimension(int nrTrials,int nrMinima, double* parametersMin, double* parametersMax, double* parameters, int nrParameters,
                                          double* parametersDelta, int maxIterationsNr, double myEpsilon, int idFunction,
                                          double** x, double* y, int nrData, int xDim,bool isWeighted, double* weights)
    {
        double bestR2 = -9999;
        double R2;
        double* R2Previous = (double *) calloc(nrMinima, sizeof(double));
        double* ySim = (double *) calloc(nrData, sizeof(double));
        double* bestParameters = (double *) calloc(nrParameters, sizeof(double));
        double* xPoint = (double *) calloc(xDim, sizeof(double));
        int i;
        int iRandom = 0;
        int counter = 0;
        for (i=0; i<nrMinima; i++)
        {
            R2Previous[i] = NODATA;
        }
        srand (time(nullptr));
        do
        {
            for (i=0;i<nrParameters;i++)
            {
                parameters[i] = parametersMin[i] + ((double) rand() / (RAND_MAX))*(parametersMax[i]-parametersMin[i]);
            }
            fittingMarquardt_nDimension(parametersMin,parametersMax,parameters,nrParameters,parametersDelta,maxIterationsNr,myEpsilon,idFunction,x,y,nrData,xDim,isWeighted,weights);
            for (i=0;i<nrData;i++)
            {
                for (int k=0; k<xDim; k++)
                {
                    xPoint[k] = x[i][k];
                }
                ySim[i]= estimateFunction_nDimension(idFunction,parameters,nrParameters,xPoint,xDim);
            }
            R2 = computeR2(y,ySim,nrData);
            if (R2 > bestR2-EPSILON)
            {
                for (int j=0;j<nrMinima-1;j++)
                {
                    R2Previous[j] = R2Previous[j+1];
                }
                R2Previous[nrMinima-1] = R2;
                bestR2 = R2;
                for (i=0;i<nrParameters;i++)
                {
                    bestParameters[i] = parameters[i];
                }
            }
            iRandom++;
            counter++;
        } while(iRandom<nrTrials && R2<(1 - EPSILON) && fabs(R2Previous[0]-R2Previous[nrMinima-1])>0.0001);

        for (i=0;i<nrParameters;i++)
        {
            parameters[i] = bestParameters[i];
        }
        free(bestParameters);
        free(ySim);
        free(R2Previous);
        free(xPoint);
        return counter;
    }

    bool fittingMarquardt_nDimension(double* parametersMin, double* parametersMax, double* parameters, int nrParameters,
                          double* parametersDelta, int maxIterationsNr, double myEpsilon, int idFunction,
                          double** x, double* y, int nrData, int xDim,bool isWeighted, double* weights)
    {
        // Sum of Squared Erros
        double mySSE, diffSSE, newSSE;
        static double VFACTOR = 10;

        double* paramChange = (double *) calloc(nrParameters, sizeof(double));
        double* newParameters = (double *) calloc(nrParameters, sizeof(double));
        double* lambda = (double *) calloc(nrParameters, sizeof(double));

        for(int i = 0; i < nrParameters; i++)
        {
            lambda[i] = 0.01;       // damping parameter
            paramChange[i] = 0;
        }

        mySSE = normGeneric_nDimension(idFunction, parameters, nrParameters, x, y, nrData,xDim);

        int iterationNr = 0;
        do
        {
            leastSquares_nDimension(idFunction, parameters, nrParameters, x, y, nrData,xDim, lambda, parametersDelta, paramChange,isWeighted,weights);
                // change parameters
            for (int i = 0; i < nrParameters; i++)
            {
                newParameters[i] = parameters[i] + paramChange[i];
                if ((newParameters[i] > parametersMax[i]) && (lambda[i] < 1000))
                {
                    newParameters[i] = parametersMax[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
                if (newParameters[i] < parametersMin[i])
                {
                    newParameters[i] = parametersMin[i];
                    if (lambda[i] < 1000)
                        lambda[i] *= VFACTOR;
                }
            }

            newSSE = normGeneric_nDimension(idFunction, newParameters, nrParameters, x, y, nrData,xDim);

            if (newSSE == NODATA)
            {
                // free memory
                free(lambda);
                free(paramChange);
                free(newParameters);

                return false;
            }

            diffSSE = mySSE - newSSE ;

            if (diffSSE > 0)
            {
                mySSE = newSSE;
                for (int i = 0; i < nrParameters ; i++)
                {
                    parameters[i] = newParameters[i];
                    lambda[i] /= VFACTOR;
                }
            }
            else
            {
                for(int i = 0; i < nrParameters; i++)
                {
                    lambda[i] *= VFACTOR;
                }
            }

            iterationNr++;
        }
        while (iterationNr <= maxIterationsNr && fabs(diffSSE) > myEpsilon);

        // free memory
        free(lambda);
        free(paramChange);
        free(newParameters);

        return (fabs(diffSSE) <= myEpsilon);
    }

    void leastSquares_nDimension(int idFunction, double* parameters, int nrParameters,
                      double** x, double* y, int nrData,int xDim, double* lambda,
                      double* parametersDelta, double* parametersChange, bool isWeighted, double* weights)
    {
        int i, j, k;
        double pivot, mult, top;

        double* g = (double *) calloc(nrParameters+1, sizeof(double));
        double* z = (double *) calloc(nrParameters+1, sizeof(double));
        double* firstEst = (double *) calloc(nrData+1, sizeof(double));

        double** a = (double **) calloc(nrParameters+1, sizeof(double*));
        double** P = (double **) calloc(nrParameters+1, sizeof(double*));
        double* xPoint = (double *) calloc(xDim, sizeof(double));
        for (i = 0; i < nrParameters+1; i++)
        {
                a[i] = (double *) calloc(nrParameters+1, sizeof(double));
                P[i] = (double *) calloc(nrData+1, sizeof(double));
        }

        // first set of estimates
        for (i = 0; i < nrData; i++)
        {
            for (k=0; k<xDim; k++)
            {
                xPoint[k] = x[i][k];
            }
            firstEst[i] = estimateFunction_nDimension(idFunction, parameters, nrParameters, xPoint,xDim);
        }

        // change parameters and compute derivatives
        for (i = 0; i < nrParameters; i++)
        {
            parameters[i] += parametersDelta[i];
            for (j = 0; j < nrData; j++)
            {
                for (k=0; k<xDim; k++)
                {
                    xPoint[k] = x[j][k];
                }
                double newEst = estimateFunction_nDimension(idFunction, parameters, nrParameters, xPoint,xDim);
                P[i][j] = (newEst - firstEst[j]) / MAXVALUE(parametersDelta[i], EPSILON) ;
            }
            parameters[i] -= parametersDelta[i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] = 0;
                for (k = 0; k < nrData; k++)
                {
                    if (isWeighted)
                    {
                        a[i][j] += (weights[k]*(P[i][k] * P[j][k]));
                    }
                    else
                    {
                        a[i][j] += (P[i][k] * P[j][k]);
                    }
                }
            }
            z[i] = sqrt(a[i][i]) + EPSILON; //?
        }

        for (i = 0; i < nrParameters; i++)
        {
            g[i] = 0.;
            for (k = 0 ; k<nrData ; k++)
            {
                g[i] += P[i][k] * (y[k] - firstEst[k]);
            }
            g[i] /= z[i];
            for (j = i; j < nrParameters; j++)
            {
                a[i][j] /= (z[i] * z[j]);
            }
        }

        for (i = 0; i < (nrParameters+1); i++)
        {
            a[i][i] += lambda[i];
            for (j = i+1; j < nrParameters; j++)
            {
                a[j][i] = a[i][j];
            }
        }

        for (j = 0; j < (nrParameters - 1); j++)
        {
            pivot = a[j][j];
            for (i = j + 1 ; i < nrParameters; i++)
            {
                mult = a[i][j] / pivot;
                for (k = j + 1; k < nrParameters; k++)
                {
                    a[i][k] -= mult * a[j][k];
                }
                g[i] -= mult * g[j];
            }
        }

        parametersChange[nrParameters - 1] = g[nrParameters - 1] / a[nrParameters - 1][nrParameters - 1];

        for (i = nrParameters - 2; i >= 0; i--)
        {
            top = g[i];
            for (k = i + 1; k < nrParameters; k++)
            {
                top -= a[i][k] * parametersChange[k];
            }
            parametersChange[i] = top / a[i][i];
        }

        for (i = 0; i < nrParameters; i++)
        {
            parametersChange[i] /= z[i];
        }

        // free memory
        for (i = 0; i < nrParameters+1; i++)
        {
            free(a[i]);
            free(P[i]);
        }
        free(a);
        free(P);
        free(g);
        free(z);
        free(firstEst);
        free(xPoint);
    }


    double estimateFunction_nDimension(int idFunction, double *parameters, int nrParameters, double* xPoint, int xDim)
    {
        switch (idFunction)
        {
            case FUNCTION_CODE_PARABOLIC :
                return parabolicFunction(xPoint, parameters);

            case FUNCTION_CODE_MULTILINEAR :
                return multilinear(xPoint,parameters,xDim,nrParameters);

            case FUNCTION_CODE_TEMPVSHEIGHT :
                return functionTemperatureVsHeight(xPoint,parameters,xDim,nrParameters);

            default:
                return NODATA ;
        }
    }


    double normGeneric_nDimension(int idFunction, double *parameters,int nrParameters, double** x, double *y, int nrData, int xDim)
    {
        double estimate, error;
        double norm = 0;
        double* xPoint = (double *) calloc(xDim, sizeof(double));
        for (int i = 0; i < nrData; i++)
        {
            for (int j=0; j<xDim; j++)
            {
                xPoint[j] = x[i][j];
            }
            estimate = estimateFunction_nDimension(idFunction, parameters, nrParameters, xPoint,xDim);
            if (estimate == NODATA)
            {
                free(xPoint);
                return NODATA;
            }
            error = y[i] - estimate;
            norm += error * error;
        }
        free(xPoint);
        return norm;
    }



}


namespace matricial
{
    int matrixSum(double**a , double**b, int rowA , int rowB, int colA, int colB, double **c)
    {
        if ((rowA != rowB) || (colA!=colB)) return NODATA;
        for (int i = 0 ; i< rowA; i++)
        {
            for (int j=0 ; j< colA ; j++)
            {
                c[i][j]= a[i][j] + b[i][j];
            }
        }
        return CRIT3D_OK ;
    }

    int matrixDifference(double**a , double**b, int rowA , int rowB, int colA, int colB, double **c)
    {
        if ((rowA != rowB) || (colA!=colB)) return NODATA;
        for (int i = 0 ; i< rowA; i++)
        {
            for (int j=0 ; j< colA ; j++)
            {
                c[i][j]= a[i][j] - b[i][j];
            }
        }
        return CRIT3D_OK ;
    }

    int matrixProduct(double **first,double**second,int colFirst,int rowFirst,int colSecond,int rowSecond,double ** multiply)
    {
        int c, d, k;
        double sum = 0;
        if ((colFirst != rowSecond)) return NODATA;
        for ( c = 0 ; c < rowFirst ; c++ )
        {
            for ( d = 0 ; d < colSecond ; d++ )
            {
                for ( k = 0 ; k < colFirst ; k++ )
                {
                    sum += first[c][k] * second[k][d];
                }
                multiply[c][d] = sum;
                sum = 0.;
            }
        }
        return CRIT3D_OK;
    }

    // it assume that rowSecond == colFirst
    int matrixProductNoCheck(double **first, double**second,int rowFirst, int colFirst, int colSecond, double ** multiply)
    {
        int c, d, k;
        double sum = 0;

        for ( c = 0 ; c < rowFirst ; c++ )
        {
            for ( d = 0 ; d < colSecond ; d++ )
            {
                for ( k = 0 ; k < colFirst ; k++ )
                {
                    sum += first[c][k] * second[k][d];
                }
                multiply[c][d] = sum;
                sum = 0.;
            }
        }
        return CRIT3D_OK;
    }

    void  multiplyStrassen(double **c,double **d,int size,double **newMatrix)
    {
        if(size == 1){
            newMatrix[0][0] = c[0][0] *d[0][0];
        }
        else {
            int i,j;
            int nsize =size/2;
            double **c11 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                c11[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **c12 =(double**)  malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                c12[i]= (double *)malloc(nsize * sizeof(double));
            }
            double **c21 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                c21[i]= (double *)malloc(nsize * sizeof(double));
            }
            double **c22 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                c22[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **d11 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                d11[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **d12 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                d12[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **d21 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                d21[i]= (double*) malloc(nsize*sizeof(double));
            }
            double **d22 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                d22[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m1 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m1[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m2 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m2[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m3 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m3[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m4 =(double**)  malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m4[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m5 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m5[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m6 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m6[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **m7 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                m7[i]= (double *)malloc(nsize * sizeof(double));
            }
            for(i=0;i<nsize;i++){
                for(j=0;j<nsize;j++){
                    c11[i][j]=c[i][j];
                    c12[i][j]=c[i][j+nsize];
                    c21[i][j]=c[i+nsize][j];
                    c22[i][j]=c[i+nsize][j+nsize];
                    d11[i][j]=d[i][j];
                    d12[i][j]=d[i][j+nsize];
                    d21[i][j]=d[i+nsize][j];
                    d22[i][j]=d[i+nsize][j+nsize];
                }
            }
            double **temp1 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp1[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **temp2 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp2[i]= (double *)malloc(nsize*sizeof(double));
            }

            add(c11,c22,nsize,temp1);
            add(d11,d22,nsize,temp2);
            multiplyStrassen(temp1,temp2,nsize,m1);
            free(temp1);
            free(temp2);

            double **temp3 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp3[i]= (double *)malloc(nsize * sizeof(double));
            }
            add(c21,c22,nsize,temp3);
            multiplyStrassen(temp3,d11,nsize,m2);
            free(temp3);


            double **temp4 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp4[i]= (double *)malloc(nsize*sizeof(double));
            }
            sub(d12,d22,nsize,temp4);
            multiplyStrassen(c11,temp4,nsize,m3);
            free(temp4);


            double **temp5 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp5[i]= (double *)malloc(nsize*sizeof(double));
            }
            sub(d21,d11,nsize,temp5);
            multiplyStrassen(c22,temp5,nsize,m4);
            free(temp5);


            double **temp6 =(double**)  malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp6[i]= (double *)malloc(nsize*sizeof(double));
            }
            add(c11,c12,nsize,temp6);
            multiplyStrassen(temp6,d22,nsize,m5);
            free(temp6);

            double **temp7 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp7[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **temp8 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp8[i]= (double *)malloc(nsize*sizeof(double));
            }
            sub(c21,c11,nsize,temp7);
            add(d11,d12,nsize,temp8);
            multiplyStrassen(temp7,temp8,nsize,m6);
            free(temp7);
            free(temp8);

            double **temp9 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp9[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **temp10 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                temp10[i]= (double *)malloc(nsize*sizeof(double));
            }
            sub(c12,c22,nsize,temp9);
            add(d21,d22,nsize,temp10);
            multiplyStrassen(temp9,temp10,nsize,m7);
            free(temp9);
            free(temp10);


            double **te1 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te1[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te2 =(double**)  malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te2[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te3 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te3[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te4 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te4[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te5 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te5[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te6 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te6[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te7 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te7[i]= (double *)malloc(nsize*sizeof(double));
            }
            double **te8 = (double**) malloc(nsize * sizeof(double *));
            for(i=0;i<nsize;i++){
                te8[i]= (double *)malloc(nsize*sizeof(double));
            }

            add(m1,m7,nsize,te1);
            sub(m4,m5,nsize,te2);
            add(te1,te2,nsize,te3);    //c11

            add(m3,m5,nsize,te4);//c12
            add(m2,m4,nsize,te5);//c21

            add(m3,m6,nsize,te6);
            sub(m1,m2,nsize,te7);

            add(te6,te7,nsize,te8);//c22

            int a=0;
            // TODO check: b is unused
            //int b=0;
            int c=0;
            int d=0;
            int e=0;
            int nsize2= 2*nsize;
            for(i=0;i<nsize2;i++){
                for(j=0;j<nsize2;j++){
                    if(j>=0 && j<nsize && i>=0 && i<nsize){
                        newMatrix[i][j] = te3[i][j];
                    }
                    if(j>=nsize && j<nsize2 && i>=0 && i<nsize){
                        a=j-nsize;
                        newMatrix[i][j] = te4[i][a];
                    }
                    if(j>=0 && j<nsize && i>= nsize && i < nsize2){
                        c=i-nsize;
                        newMatrix[i][j] = te5[c][j];
                    }
                    if(j>=nsize && j< nsize2 && i>= nsize && i< nsize2 ){
                        d=i-nsize;
                        e=j-nsize;
                        newMatrix[i][j] =te8[d][e];
                    }
                }
            }
        free(m1);
        free(m2);
        free(m3);
        free(m4);
        free(m5);
        free(m6);
        free(m7);
        free(te1);
        free(te2);
        free(te3);
        free(te4);
        free(te5);
        free(te6);
        free(te7);
        free(te8);
        free(c11);
        free(c12);
        free(c21);
        free(c22);
        free(d11);
        free(d12);
        free(d21);
        free(d22);
        }
    }
    void add(double **a, double **b, int size,double **c)
    {
        int i,j;
        for(i=0;i<size;i++){
            for(j=0;j<size;j++){
                c[i][j] = a[i][j] + b[i][j];
            }
        }
    }

    void sub(double **a,double **b,int size,double **c)
    {
        int i,j;
        for(i=0;i<size;i++){
                    for(j=0;j<size;j++){
                            c[i][j]= a[i][j] - b[i][j];
                    }
            }


    }
    void matrixProductSquareMatricesNoCheck(double **first,double**second,int dimension,double ** multiply)
    {

        int c, d, k;
        double sum = 0;
        for ( c = 0 ; c < dimension ; c++ )
        {
            for ( d = 0 ; d < dimension ; d++ )
            {
                for ( k = 0 ; k < dimension ; k++ )
                {
                    sum += first[c][k] * second[k][d];
                }
                multiply[c][d] = sum;
                sum = 0.;
            }
        }
    }


    void choleskyDecompositionSinglePointer(double *a, int n, double* p)
    {
        // adapted from http://www.math.hawaii.edu/~williamdemeo/C/stat243/reports/hw3/hw3/node6.html
        int i,j,k;
             for(j=0;j<n;j++)
                  p[j] = a[n*j+j];
             for(j=0;j<n;j++)
             {
                  for(k=0;k<j;k++)
                       p[j] -= a[n*k+j]*a[n*k+j];
                  p[j] = sqrt(p[j]);
                  for(i=j+1;i<n;i++)
                  {
                       for(k=0;k<j;k++)
                            a[n*j+i] -= a[n*k+i]*a[n*k+j];
                       a[n*j+i]/=p[j];
                  }
             }
    }



    void choleskyDecompositionTriangularMatrix(double **a, int n, bool isLowerMatrix)
    {
        // input double pointer (square matrix: symmetric and positive definite)
        // n: matrix dimension (n x n)
        // isLowerMatrix: 1) if true: lower triangular matrix 2) if false: upper triangular matrix
        double* diagonalElementsCholesky =(double*)calloc(n, sizeof(double));
        double* aLinear =(double*)calloc(n*n, sizeof(double));
        int counter = 0;
        for (int i=0;i<n;i++)
        {
            diagonalElementsCholesky[i] = NODATA;
            for (int j=0;j<n;j++)
            {
                aLinear[counter]= a[i][j];
                counter++;
            }
        }
        matricial::choleskyDecompositionSinglePointer(aLinear,n,diagonalElementsCholesky);
        counter = 0;
        if (isLowerMatrix)
        {
            for (int i=0;i<n;i++)
            {
                for (int j=0;j<n;j++)
                {
                    a[j][i]= aLinear[counter]; // for lower output matrix
                    counter++;
                }
                a[i][i]= diagonalElementsCholesky[i];
            }

            for (int i=0;i<n;i++)
            {
                    for (int j=i+1;j<n;j++) a[i][j]=0.;
            }
        }
        else
        {
            for (int i=0;i<n;i++)
            {
                for (int j=0;j<n;j++)
                {
                    a[i][j]= aLinear[counter]; // for upper output matrix
                    counter++;
                }
                a[i][i]= diagonalElementsCholesky[i];
            }

            for (int i=0;i<n;i++)
            {
                    for (int j=0;j<i;j++) a[i][j]=0.;
            }
        }
        free(diagonalElementsCholesky);
        free(aLinear);
    }

    void transposedSquareMatrix(double** a, int n)
    {
        double** b = (double**)calloc(n, sizeof(double*));
        for (int i=0;i<n;i++)
            b[i]= (double*)calloc(n, sizeof(double));

        for (int i=0;i<n;i++)
        {
            for (int j=0;j<n;j++)
                b[i][j]=a[j][i];
        }
        for (int i=0;i<n;i++)
        {
            for (int j=0;j<n;j++)
                a[i][j] = b[i][j];
        }

        for (int i=0;i<n;i++)
            free(b[i]);
        free(b);
    }

    void transposedMatrix(double **inputMatrix, int nrRows, int nrColumns, double **outputMatrix)
    {
        for (int i=0;i<nrRows;i++)
        {
            for (int j=0;j<nrColumns;j++)
            {
                outputMatrix[j][i] = inputMatrix[i][j];
            }
        }
    }

    void minorMatrix(double** b,double** a,int i,int n)
    {
        //	calculate minor of matrix OR build new matrix : k-had = minor
        int j,l,h=0,k=0;
        for(l=1;l<n;l++)
            for( j=0;j<n;j++)
            {
                if(j == i)
                    continue;
                b[h][k] = a[l][j];
                k++;
                if(k == (n-1))
                {
                    h++;
                    k=0;
                }
            }
    }


    double determinant(double** a,int n)
    {
        //	calculate determinant of matrix
        int i;
        double sum=0;
        if (n == 1)
            return a[0][0];
        else if(n == 2)
            return (a[0][0]*a[1][1]-a[0][1]*a[1][0]);
        else
        {
            double** b = (double**)calloc(n, sizeof(double*));
            for (int i=0;i<n;i++)
                b[i]= (double*)calloc(n, sizeof(double));
            for(i=0;i<n;i++)
            {
                matricial::minorMatrix(b,a,i,n);	// read function
                sum = (sum + a[0][i]*pow(-1,i)*matricial::determinant(b,(n-1)));	// sum = determinte matrix
            }
            for (int i=0;i<n;i++)
                free(b[i]);
            free(b);
            return sum;
        }
    }

    void cofactor(double** a,double** d,int n,double determinantOfMatrix)
    {
        double** b;
        double** c;
        b = (double**)calloc(n, sizeof(double*));
        c = (double**)calloc(n, sizeof(double*));
        for (int i=0;i<n;i++)
        {
            b[i]= (double*)calloc(n, sizeof(double));
            c[i]= (double*)calloc(n, sizeof(double));
        }
        //int l,h,m,k,i,j;
        int m,k;
        for (int h=0;h<n;h++)
            for (int l=0;l<n;l++)
            {
                m=0;
                k=0;
                for (int i=0;i<n;i++)
                    for (int j=0;j<n;j++)
                        if (i != h && j != l){
                            b[m][k]=a[i][j];
                            if (k<(n-2))
                                k++;
                            else{
                                k=0;
                                m++;
                            }
                        }
                c[h][l] = pow(-1,(h+l))*matricial::determinant(b,(n-1));	// c = cofactor Matrix
            }
        matricial::transposedMatrix(c,n,n,d);
        for (int h=0;h<n;h++)
        {
            for (int l=0;l<n;l++)
            {
                d[h][l] /= determinantOfMatrix;
            }
        }

        for (int h=0;h<n;h++)
        {
            free(b[h]);
            free(c[h]);
        }
        free(b);
        free(c);
    }

    void inverse(double** a,double** d,int n)
    {
        double determinantOfMatrix;
        determinantOfMatrix = determinant(a,n);
        //	calculate inverse of matrix
        if(determinantOfMatrix == 0)
            printf("\nInverse of Entered Matrix is not possible\n");
        else if(n == 1)
            d[0][0] = 1/a[0][0];
        else
            matricial::cofactor(a,d,n,determinantOfMatrix);
    }

    bool inverseGaussJordan(double** a,double** d,int n)
    {
        if (fabs(determinant(a,n))<EPSILON)
        {
            printf("\nInverse of Entered Matrix is not possible\n");
            return false;
        }
        for (int i=0;i<n;i++)
        {
            d[i][i]=1;
            for (int j=i+1;j<n;j++)
            {
                d[j][i] = d[i][j] = 0.;
            }
        }
        for (int i=0;i<n;i++)
        {
            if (a[i][i] == 0)
            {
                printf("\nInverse of Entered Matrix is not possible\n");
                return false;
            }
            double ratio;
            for(int j=0;j<n;j++)
            {
               if(i!=j)
               {
                    ratio = a[j][i]/a[i][i];
                    for(int k=0;k<n;k++)
                    {
                        a[j][k] = a[j][k] - ratio*a[i][k];
                        d[j][k] = d[j][k] - ratio*d[i][k];
                    }
               }
            }
        }

        for (int i=0;i<n;i++)
        {
            for (int j=0;j<n;j++)
            {
                a[i][j] = a[i][j]/a[i][i];
                d[i][j] = d[i][j]/a[i][i];
            }
        }
        return true;
    }

    int eigenSystemMatrix2x2(double** a, double* eigenvalueA, double** eigenvectorA, int n)
    {
        if (n != 2) return PARAMETER_ERROR;
        double b,c;
        b = - a[0][0] - a[1][1];
        c = (a[0][0] * a[1][1] - a[0][1] * a[1][0]);
        if ((b*b - 4*c) < 0) return PARAMETER_ERROR;

        if (fabs(a[0][1]) < EPSILON && fabs(a[1][0]) < EPSILON)
        {
            eigenvalueA[0]= a[0][0];
            eigenvalueA[1]= a[1][1];
            eigenvectorA[1][0] = 1;
            eigenvectorA[0][0] = 0;
            eigenvectorA[0][1] = 0;
            eigenvectorA[1][1] = 1;
            return 0;
        }
        if (fabs(a[0][1]) < EPSILON)
        {
            eigenvalueA[0]= a[0][0];
            eigenvalueA[1]= a[1][1];
            eigenvectorA[1][0] = 1;
            eigenvectorA[0][0] = (eigenvalueA[0] - a[1][1]) * eigenvectorA[1][0];
            eigenvectorA[0][1] = 0;
            eigenvectorA[1][1] = 1;
            return 0;
        }
        if (fabs(a[1][0]) < EPSILON)
        {
            eigenvalueA[0]= a[0][0];
            eigenvalueA[1]= a[1][1];
            eigenvectorA[0][0] = 1;
            eigenvectorA[1][0] = 0;
            eigenvectorA[0][1] = 1;
            eigenvectorA[1][1] = (eigenvalueA[1] - a[0][0]) * eigenvectorA[0][1];
            return 0;

        }
        eigenvalueA[0] = (-b + sqrt(b*b - 4*c))/2;
        eigenvalueA[1] = (-b - sqrt(b*b - 4*c))/2;
        for (int i=0;i<2;i++)
        {
            eigenvectorA[0][i] = 1;
            eigenvectorA[1][i] = (eigenvalueA[i]-a[0][0])/a[0][1];
        }
        return 0;
    }

    bool linearSystemResolutionByCramerMethod(double* constantTerm, double** coefficientMatrix, int matrixSize, double* roots)
    {

        double denominator;
        denominator = determinant(coefficientMatrix,matrixSize);
        if (fabs(denominator)<EPSILON) return false;

        double** numerator = (double**)calloc(matrixSize, sizeof(double*));
        for(int i = 0;i<matrixSize;i++)
        {
            numerator[i] = (double*)calloc(matrixSize, sizeof(double));
            for(int j = 0;j<matrixSize;j++)
            {
                numerator[i][j] = coefficientMatrix[i][j];
            }
        }

        for(int counterRoot = 0;counterRoot<matrixSize; counterRoot++)
        {
            /*for(int i = 0;i<matrixSize;i++)
            {
                for(int j = 0;j<matrixSize;j++)
                {
                    numerator[j][i] = coefficientMatrix[j][i];
                }
            }*/
            for(int j = 0;j<matrixSize;j++)
            {
                numerator[j][counterRoot] = constantTerm[j];
            }
            roots[counterRoot] = determinant(numerator,matrixSize)/denominator;
            for(int j = 0;j<matrixSize;j++)
            {
                numerator[j][counterRoot] = coefficientMatrix[j][counterRoot];
            }
        }
        for(int i = 0;i<matrixSize;i++)
        {
            free(numerator[i]);
        }
        free(numerator);

        return true;
    }
}


namespace distribution
{
    float normalGauss(TfunctionInput fInput)
    {
        return float(pow((2*PI*fInput.par[1]*fInput.par[1]),-0.5)*exp(-0.5*pow((fInput.x-fInput.par[0])/fInput.par[1],2)));
    }
}

namespace myrandom {

double cauchyRandom(double gamma)
{
    double temp = (double) rand() / (RAND_MAX);
    return statistics::functionQuantileCauchy(gamma, 0, temp);
}


//----------------------------------------------------------------------
    // Generate a standard normally-distributed random variable
    // (See Numerical Recipes in Pascal W. H. Press, et al. 1989 p. 225)
    //----------------------------------------------------------------------
    float normalRandom(int *gasDevIset,float *gasDevGset)
    {
        float fac = 0;
        float r = 0;
        float v1, v2, normalRandom;
        float temp;

        if (*gasDevIset == 0) //We don't have an extra deviate
        {
            do
            {
                temp = (float) rand() / (RAND_MAX);
                v1 = 2*temp - 1;
                temp = (float) rand() / (RAND_MAX);
                v2 = 2*temp - 1;
                r = v1 * v1 + v2 * v2;
            } while ( (r>=1) | (r==0) ); // see if they are in the unit circle, and if they are not, try again.
            // Box-Muller transformation to get two normal deviates. Return one and save the other for next time.
            fac = float(sqrt(-2 * log(r) / r));
            *gasDevGset = v1 * fac; //Gaussian random deviates
            normalRandom = v2 * fac;
            *gasDevIset = 1; //set the flag
        }
        // We have already an extra deviate
        else
        {
            *gasDevIset = 0; //unset the flag
            normalRandom = *gasDevGset;
        }
        return normalRandom;
    }

    double normalRandom(int *gasDevIset,double *gasDevGset)
    {
        double fac = 0;
        double r = 0;
        double v1, v2, normalRandom;
        double temp;
        if (*gasDevIset == 0) //We don't have an extra deviate
        {
            do
            {
                temp = (double) rand() / (RAND_MAX);
                v1 = 2*temp - 1;

                temp = (double) rand() / (RAND_MAX);
                v2 = 2*temp - 1;
                r = v1 * v1 + v2 * v2;
            } while ( (r>=1) | (r==0) ); // see if they are in the unit circle, and if they are not, try again.
            // Box-Muller transformation to get two normal deviates. Return one and save the other for next time.
            fac = sqrt(-2 * log(r) / r);
            *gasDevGset = v1 * fac; //Gaussian random deviates
            normalRandom = v2 * fac;
            *gasDevIset = 1; //set the flag
        }
        // We have already an extra deviate
        else
        {
            *gasDevIset = 0; //unset the flag
            normalRandom = *gasDevGset;
        }
        return normalRandom;
    }

    double normalRandomLongSeries(int *gasDevIset, double *gasDevGset, int *randomNumberInitial)
    {
        double fac = 0;
        double r = 0;
        double v1, v2, normalRandom;
        double temp;
        int randomNumber;
        //clock_t time0;
        //clock_t time1;
        //time1 = time0 = clock();
        if (*gasDevIset == 0) //We don't have an extra deviate
        {

            do
            {
                randomNumber = myrandom::getUniformallyDistributedRandomNumber(randomNumberInitial);
                temp = (double) randomNumber / (RAND_MAX);
                v1 = 2*temp - 1;

                randomNumber = myrandom::getUniformallyDistributedRandomNumber(randomNumberInitial);
                temp = (double) randomNumber / (RAND_MAX);
                v2 = 2*temp - 1;
                r = v1 * v1 + v2 * v2;
            } while ( (r>=1) | (r==0) ); // see if they are in the unit circle, and if they are not, try again.
            // Box-Muller transformation to get two normal deviates. Return one and save the other for next time.
            fac = sqrt(-2 * log(r) / r);
            *gasDevGset = v1 * fac; //Gaussian random deviates
            normalRandom = v2 * fac;
            *gasDevIset = 1; //set the flag
        }
        // We have already an extra deviate
        else
        {
            *gasDevIset = 0; //unset the flag
            normalRandom = *gasDevGset;
        }
        return normalRandom;
    }

    int getUniformallyDistributedRandomNumber(int* randomNumberInitial)
    {
        int randomNumber = rand();
        if (randomNumber != *randomNumberInitial)
        {
            return randomNumber;
        }
        else
        {
            int firstRandom;
            //int counter = 0;
            do
            {
                //counter++;
                srand(unsigned(time(0)));
                firstRandom = rand();
            }
            while(firstRandom == *randomNumberInitial);
            //printf("seed estratto %d\n",counter);
            *randomNumberInitial = firstRandom;
            return firstRandom;
        }
    }
}

namespace statistics
{

    double ERF(double x, double accuracy) // error function
    {

        return (1.12837916709551*double(integration::simpsonRule(errorFunctionPrimitive,0.,x,accuracy))); // the constant in front of integration is equal to 2*pow(PI,-0.5)
    }

    double ERFC(double x, double accuracy) // error function
    {
        return (1. - ERF(x, accuracy));
    }

    double tabulatedERFC(double x)
    {
        return (1. - tabulatedERF(x));
    }

    double tabulatedERF(double x)
    {
        //double output;
        //double variable;
        //variable = fabs(x);
        //double* extendedValueX = (double *) calloc(400, sizeof(double));
        //double* extendedValueY = (double *) calloc(400, sizeof(double));
        double valueY[400] = {0.000000000,0.011283000,0.022565000,0.033841000,0.045111000,0.056372000,0.067622000,0.078858000,0.090078000,0.101282000,0.112463000,0.123623000,0.134758000,0.145867000,0.156947000	,0.167996000	,0.179012000	,0.189992000	,0.200936000	,0.211840000	,0.222703000	,0.233522000	,0.244296000	,0.255023000	,0.265700000	,0.276326000	,0.286900000	,0.297418000	,0.307880000	,0.318283000	,0.328627000	,0.338908000	,0.349126000	,0.359279000	,0.369365000	,0.379382000	,0.389330000	,0.399206000	,0.409009000	,0.418739000	,0.428392000	,0.437969000	,0.447468000	,0.456887000	,0.466225000	,0.475482000	,0.484655000	,0.493745000	,0.502750000	,0.511668000	,0.520500000	,0.529244000	,0.537899000	,0.546464000	,0.554939000	,0.563323000	,0.571616000	,0.579816000	,0.587923000	,0.595936000	,0.603856000	,0.611681000	,0.619411000	,0.627046000	,0.634586000	,0.642029000	,0.649377000	,0.656628000	,0.663782000	,0.670840000	,0.677801000	,0.684666000	,0.691433000	,0.698104000	,0.704678000	,0.711156000	,0.717537000	,0.723822000	,0.730010000	,0.736103000	,0.742101000	,0.748003000	,0.753811000	,0.759524000	,0.765143000	,0.770668000	,0.776100000	,0.781440000	,0.786687000	,0.791843000	,0.796908000	,0.801883000	,0.806768000	,0.811564000	,0.816271000	,0.820891000	,0.825424000	,0.829870000	,0.834232000	,0.838508000	,0.842701000	,0.846810000	,0.850838000	,0.854784000	,0.858650000	,0.862436000	,0.866144000	,0.869773000	,0.873326000	,0.876803000	,0.880205000	,0.883533000	,0.886788000	,0.889971000	,0.893082000	,0.896124000	,0.899096000	,0.902000000	,0.904837000	,0.907608000	,0.910314000	,0.912956000	,0.915534000	,0.918050000	,0.920505000	,0.922900000	,0.925236000	,0.927514000	,0.929734000	,0.931899000	,0.934008000	,0.936063000	,0.938065000	,0.940015000	,0.941914000	,0.943762000	,0.945561000	,0.947312000	,0.949016000	,0.950673000	,0.952285000	,0.953852000	,0.955376000	,0.956857000	,0.958297000	,0.959695000	,0.961054000	,0.962373000	,0.963654000	,0.964898000	,0.966105000	,0.967277000	,0.968413000	,0.969516000	,0.970586000	,0.971623000	,0.972628000	,0.973603000	,0.974547000	,0.975462000	,0.976348000	,0.977207000	,0.978038000	,0.978843000	,0.979622000	,0.980376000	,0.981105000	,0.981810000	,0.982493000	,0.983153000	,0.983790000	,0.984407000	,0.985003000	,0.985578000	,0.986135000	,0.986672000	,0.987190000	,0.987691000	,0.988174000	,0.988641000	,0.989091000	,0.989525000	,0.989943000	,0.990347000	,0.990736000	,0.991111000	,0.991472000	,0.991821000	,0.992156000	,0.992479000	,0.992790000	,0.993090000	,0.993378000	,0.993656000	,0.993923000	,0.994179000	,0.994426000	,0.994664000	,0.994892000	,0.995111000	,0.995322000	,0.995525000	,0.995719000	,0.995906000	,0.996086000	,0.996258000	,0.996423000	,0.996582000	,0.996734000	,0.996880000	,0.997021000	,0.997155000	,0.997284000	,0.997407000	,0.997525000	,0.997639000	,0.997747000	,0.997851000	,0.997951000	,0.998046000	,0.998137000	,0.998224000	,0.998308000	,0.998388000	,0.998464000	,0.998537000	,0.998607000	,0.998674000	,0.998738000	,0.998799000	,0.998857000	,0.998912000	,0.998966000	,0.999016000	,0.999065000	,0.999111000	,0.999155000	,0.999197000	,0.999237000	,0.999275000	,0.999311000	,0.999346000	,0.999379000	,0.999411000	,0.999441000	,0.999469000	,0.999497000	,0.999523000	,0.999547000	,0.999571000	,0.999593000	,0.999614000	,0.999635000	,0.999654000	,0.999672000	,0.999689000	,0.999706000	,0.999722000	,0.999736000	,0.999751000	,0.999764000	,0.999777000	,0.999789000	,0.999800000	,0.999811000	,0.999822000	,0.999831000	,0.999841000	,0.999849000	,0.999858000	,0.999866000	,0.999873000	,0.999880000	,0.999887000	,0.999893000	,0.999899000	,0.999905000	,0.999910000	,0.999916000	,0.999920000	,0.999925000	,0.999929000	,0.999933000	,0.999937000	,0.999941000	,0.999944000	,0.999948000	,0.999951000	,0.999954000	,0.999956000	,0.999959000	,0.999961000	,0.999964000	,0.999966000	,0.999968000	,0.999970000	,0.999972000	,0.999973000	,0.999975000	,0.999977000	,0.999977910	,0.999979260	,0.999980530	,0.999981730	,0.999982860	,0.999983920	,0.999984920	,0.999985860	,0.999986740	,0.999987570	,0.999988350	,0.999989080	,0.999989770	,0.999990420	,0.999991030	,0.999991600	,0.999992140	,0.999992640	,0.999993110	,0.999993560	,0.999993970	,0.999994360	,0.999994730	,0.999995070	,0.999995400	,0.999995700	,0.999995980	,0.999996240	,0.999996490	,0.999996720	,0.999996940	,0.999997150	,0.999997340	,0.999997510	,0.999997680	,0.999997838	,0.999997983	,0.999998120	,0.999998247	,0.999998367	,0.999998478	,0.999998582	,0.999998679	,0.999998770	,0.999998855	,0.999998934	,0.999999008	,0.999999077	,0.999999141	,0.999999201	,0.999999257	,0.999999309	,0.999999358	,0.999999403	,0.999999445	,0.999999485	,0.999999521	,0.999999555	,0.999999587	,0.999999617	,0.999999644	,0.999999670	,0.999999694	,0.999999716	,0.999999736	,0.999999756	,0.999999773	,0.999999790	,0.999999805,0.999999820,0.999999833,0.999999845,0.999999857,0.999999867,0.999999877,0.999999886,0.999999895,0.999999903,0.999999910,0.999999917,0.999999923,0.999999929,0.999999934,0.999999939,0.999999944,0.999999948,0.999999952,0.999999956,0.999999959,0.999999962,0.999999965,0.999999968,0.999999970,0.999999973,0.999999975,0.999999977,0.999999979,0.999999980,0.999999982,0.999999983};
        double valueX[400];
        for (int i=0;i<400;i++)
        {
            valueX[i] = i*0.01;
            //extendedValueY[i] = valueY[i];
        }
        //output = interpolation::linearInterpolation(fabs(x),valueX,valueY,400);
        //free(extendedValueX);
        //free(extendedValueY);
        return sgnVariable(x)*interpolation::linearInterpolation(fabs(x),valueX,valueY,400);
        /*if (x < 0)
            return -output;
        else
            return output;*/
    }

    double inverseERF(double value, double accuracy)
    {

        if (value >=1 || value <= -1)
        {
            return PARAMETER_ERROR;
        }
        double root;

        if (fabs(value) < EPSILON)
        {
            return 0.;
        }
        else if (value  >= EPSILON)
        {
            double leftBound, rightBound;
            leftBound = 0.;
            rightBound = 100.;
            do
            {
                root = ERF((rightBound+leftBound)/2,accuracy);
                if (root < value)
                {
                    leftBound = (rightBound+leftBound)/2;
                }
                else
                {
                    rightBound = (rightBound+leftBound)/2;
                }
            } while(fabs(leftBound - rightBound) > accuracy);

            return (rightBound+leftBound)/2;
        }
        else
        {
            double leftBound, rightBound;
            leftBound = -100.;
            rightBound = 0.;
            do
            {
                root = ERF((rightBound+leftBound)/2,accuracy);
                if (root < value)
                {
                    leftBound = (rightBound+leftBound)/2;
                }
                else
                {
                    rightBound = (rightBound+leftBound)/2;
                }
            } while(fabs(leftBound - rightBound) > accuracy);

            return (rightBound+leftBound)/2;
        }
    }


    double inverseERFC(double value, double accuracy)
    {

        if (value >=2 || value <= 0)
        {
            return PARAMETER_ERROR;
        }

        double root;

        if (fabs(value-1) <= EPSILON)
        {
            return 0. ;
        }
        else if (value  < 1)
        {
            double leftBound, rightBound;
            leftBound = 0.;
            rightBound = 100.;
            do
            {
                root = ERFC((rightBound+leftBound)/2,accuracy);
                if (root > value)
                {
                    leftBound = (rightBound+leftBound)/2;
                }
                else
                {
                    rightBound = (rightBound+leftBound)/2;
                }
            } while(fabs(leftBound - rightBound) > accuracy);
            return (rightBound+leftBound)/2;
        }
        else
        {
            double leftBound, rightBound;
            leftBound = -100.;
            rightBound = 0.;
            do
            {
                root = ERFC((rightBound+leftBound)/2,accuracy);
                if (root > value)
                {
                    leftBound = (rightBound+leftBound)/2;
                }
                else
                {
                    rightBound = (rightBound+leftBound)/2;
                }
            } while(fabs(leftBound - rightBound) > accuracy);
            return (rightBound+leftBound)/2;
        }
    }

    double inverseTabulatedERF(double value)
    {
        // precision on the third digit after dot
        if (fabs(value) >= 1) return PARAMETER_ERROR;

        double output = 0;
        double variable;
        variable = fabs(value);
        double valueY[400] = {0.000000000,0.011283000,0.022565000,0.033841000,0.045111000,0.056372000,0.067622000,0.078858000,0.090078000,0.101282000,0.112463000,0.123623000,0.134758000,0.145867000,0.156947000	,0.167996000	,0.179012000	,0.189992000	,0.200936000	,0.211840000	,0.222703000	,0.233522000	,0.244296000	,0.255023000	,0.265700000	,0.276326000	,0.286900000	,0.297418000	,0.307880000	,0.318283000	,0.328627000	,0.338908000	,0.349126000	,0.359279000	,0.369365000	,0.379382000	,0.389330000	,0.399206000	,0.409009000	,0.418739000	,0.428392000	,0.437969000	,0.447468000	,0.456887000	,0.466225000	,0.475482000	,0.484655000	,0.493745000	,0.502750000	,0.511668000	,0.520500000	,0.529244000	,0.537899000	,0.546464000	,0.554939000	,0.563323000	,0.571616000	,0.579816000	,0.587923000	,0.595936000	,0.603856000	,0.611681000	,0.619411000	,0.627046000	,0.634586000	,0.642029000	,0.649377000	,0.656628000	,0.663782000	,0.670840000	,0.677801000	,0.684666000	,0.691433000	,0.698104000	,0.704678000	,0.711156000	,0.717537000	,0.723822000	,0.730010000	,0.736103000	,0.742101000	,0.748003000	,0.753811000	,0.759524000	,0.765143000	,0.770668000	,0.776100000	,0.781440000	,0.786687000	,0.791843000	,0.796908000	,0.801883000	,0.806768000	,0.811564000	,0.816271000	,0.820891000	,0.825424000	,0.829870000	,0.834232000	,0.838508000	,0.842701000	,0.846810000	,0.850838000	,0.854784000	,0.858650000	,0.862436000	,0.866144000	,0.869773000	,0.873326000	,0.876803000	,0.880205000	,0.883533000	,0.886788000	,0.889971000	,0.893082000	,0.896124000	,0.899096000	,0.902000000	,0.904837000	,0.907608000	,0.910314000	,0.912956000	,0.915534000	,0.918050000	,0.920505000	,0.922900000	,0.925236000	,0.927514000	,0.929734000	,0.931899000	,0.934008000	,0.936063000	,0.938065000	,0.940015000	,0.941914000	,0.943762000	,0.945561000	,0.947312000	,0.949016000	,0.950673000	,0.952285000	,0.953852000	,0.955376000	,0.956857000	,0.958297000	,0.959695000	,0.961054000	,0.962373000	,0.963654000	,0.964898000	,0.966105000	,0.967277000	,0.968413000	,0.969516000	,0.970586000	,0.971623000	,0.972628000	,0.973603000	,0.974547000	,0.975462000	,0.976348000	,0.977207000	,0.978038000	,0.978843000	,0.979622000	,0.980376000	,0.981105000	,0.981810000	,0.982493000	,0.983153000	,0.983790000	,0.984407000	,0.985003000	,0.985578000	,0.986135000	,0.986672000	,0.987190000	,0.987691000	,0.988174000	,0.988641000	,0.989091000	,0.989525000	,0.989943000	,0.990347000	,0.990736000	,0.991111000	,0.991472000	,0.991821000	,0.992156000	,0.992479000	,0.992790000	,0.993090000	,0.993378000	,0.993656000	,0.993923000	,0.994179000	,0.994426000	,0.994664000	,0.994892000	,0.995111000	,0.995322000	,0.995525000	,0.995719000	,0.995906000	,0.996086000	,0.996258000	,0.996423000	,0.996582000	,0.996734000	,0.996880000	,0.997021000	,0.997155000	,0.997284000	,0.997407000	,0.997525000	,0.997639000	,0.997747000	,0.997851000	,0.997951000	,0.998046000	,0.998137000	,0.998224000	,0.998308000	,0.998388000	,0.998464000	,0.998537000	,0.998607000	,0.998674000	,0.998738000	,0.998799000	,0.998857000	,0.998912000	,0.998966000	,0.999016000	,0.999065000	,0.999111000	,0.999155000	,0.999197000	,0.999237000	,0.999275000	,0.999311000	,0.999346000	,0.999379000	,0.999411000	,0.999441000	,0.999469000	,0.999497000	,0.999523000	,0.999547000	,0.999571000	,0.999593000	,0.999614000	,0.999635000	,0.999654000	,0.999672000	,0.999689000	,0.999706000	,0.999722000	,0.999736000	,0.999751000	,0.999764000	,0.999777000	,0.999789000	,0.999800000	,0.999811000	,0.999822000	,0.999831000	,0.999841000	,0.999849000	,0.999858000	,0.999866000	,0.999873000	,0.999880000	,0.999887000	,0.999893000	,0.999899000	,0.999905000	,0.999910000	,0.999916000	,0.999920000	,0.999925000	,0.999929000	,0.999933000	,0.999937000	,0.999941000	,0.999944000	,0.999948000	,0.999951000	,0.999954000	,0.999956000	,0.999959000	,0.999961000	,0.999964000	,0.999966000	,0.999968000	,0.999970000	,0.999972000	,0.999973000	,0.999975000	,0.999977000	,0.999977910	,0.999979260	,0.999980530	,0.999981730	,0.999982860	,0.999983920	,0.999984920	,0.999985860	,0.999986740	,0.999987570	,0.999988350	,0.999989080	,0.999989770	,0.999990420	,0.999991030	,0.999991600	,0.999992140	,0.999992640	,0.999993110	,0.999993560	,0.999993970	,0.999994360	,0.999994730	,0.999995070	,0.999995400	,0.999995700	,0.999995980	,0.999996240	,0.999996490	,0.999996720	,0.999996940	,0.999997150	,0.999997340	,0.999997510	,0.999997680	,0.999997838	,0.999997983	,0.999998120	,0.999998247	,0.999998367	,0.999998478	,0.999998582	,0.999998679	,0.999998770	,0.999998855	,0.999998934	,0.999999008	,0.999999077	,0.999999141	,0.999999201	,0.999999257	,0.999999309	,0.999999358	,0.999999403	,0.999999445	,0.999999485	,0.999999521	,0.999999555	,0.999999587	,0.999999617	,0.999999644	,0.999999670	,0.999999694	,0.999999716	,0.999999736	,0.999999756	,0.999999773	,0.999999790	,0.999999805,0.999999820,0.999999833,0.999999845,0.999999857,0.999999867,0.999999877,0.999999886,0.999999895,0.999999903,0.999999910,0.999999917,0.999999923,0.999999929,0.999999934,0.999999939,0.999999944,0.999999948,0.999999952,0.999999956,0.999999959,0.999999962,0.999999965,0.999999968,0.999999970,0.999999973,0.999999975,0.999999977,0.999999979,0.999999980,0.999999982,0.999999983};
        double* extendedValueX = (double *) calloc(1000, sizeof(double));
        double* extendedValueY = (double *) calloc(1000, sizeof(double));
        //double* secondDerivative = (double *) calloc(1000, sizeof(double));
        for (int i=0;i<1000;i++)
        {
            extendedValueX[i] = i*0.01;
            if (i > 399)
            {
                extendedValueY[i] = 1 - exp(-extendedValueX[i]*extendedValueX[i])/(sqrt(PI)*variable);
                if (extendedValueY[i] < extendedValueY[i-1]) extendedValueY[i] = extendedValueY[i-1];
            }
            else
            {
                extendedValueY[i] = valueY[i];
            }
        }
        output = interpolation::linearInterpolation(variable,extendedValueY,extendedValueX,1000);

        free(extendedValueX);
        free(extendedValueY);
        if (value < 0)
            return -output;
        else
            return output;

    }


    double inverseTabulatedERFC(double value)
    {
        // precision on the third digit after dots
        if (value >=2 || value <= 0)
        {
            return PARAMETER_ERROR;
        }
        return statistics::inverseTabulatedERF(1-value);
    }

    double functionCDFCauchy(double gamma,double x0, double x)
    {
        double result = 0;
        result = 0.5 + 1 / PI * atan((x-x0)/gamma);
        return result;
    }

    double functionPDFCauchy(double gamma,double x0, double x)
    {
        double result = 0;
        result = 1.0/(PI*gamma*(1+ ((x-x0)*(x-x0)/(gamma*gamma))));
        return result;
    }

    double functionQuantileCauchy(double gamma,double x0, double F)
    {
        double result = 0;
        result = x0 + gamma*tan(PI*(F-0.5));
        return result;
    }
}

