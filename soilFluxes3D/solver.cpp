/*!
    \name solver.cpp
    \copyright (C) 2011 Fausto Tomei, Gabriele Antolini, Antonio Volta,
                        Alberto Pistocchi, Marco Bittelli

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

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <thread>

#include "basicMath.h"
#include "header/types.h"
#include "header/solver.h"


double distance(unsigned long i, unsigned long j)
{
    return sqrt(square(fabs(double(nodeList[i].x - nodeList[j].x)))
                + square(fabs(double(nodeList[i].y - nodeList[j].y)))
                + square(fabs(double(nodeList[i].z - nodeList[j].z))));
}


double distance2D(unsigned long i, unsigned long j)
{
    return sqrt(square(fabs(double(nodeList[i].x - nodeList[j].x)))
                + square(fabs(double(nodeList[i].y - nodeList[j].y))));
}

double arithmeticMean(double v1, double v2)
{
    return (v1 + v2) * 0.5;
}

double logarithmicMean(double v1, double v2)
{
    if (v1 == v2)
    {
        return v1;
    }
    else
    {
        return (v1 - v2) / log(v1/v2);
    }
}

double geometricMean(double v1, double v2)
{
    double sign = v1 / fabs(v1);
    return sign * sqrt(v1 * v2);
}

double computeMean(double v1, double v2)
{
    if (myParameters.meanType == MEAN_LOGARITHMIC)
        return logarithmicMean(v1, v2);
    else if (myParameters.meanType == MEAN_GEOMETRIC)
        return geometricMean(v1, v2);
    else
        // default: logarithmic
        return logarithmicMean(v1, v2);
}


TlinkedNode* getLink(long i, long j)
{
    if (nodeList[i].up.index == j)
        return &(nodeList[i].up);

    if (nodeList[i].down.index == j)
        return &(nodeList[i].down);

    for (short l = 0; l < myStructure.nrLateralLinks; l++)
    {
         if (nodeList[i].lateral[l].index == j)
             return &(nodeList[i].lateral[l]);
    }

    return nullptr;
}


int getMaxIterationsNr(int approximationNr)
{
    int maxIterationsNr = int((approximationNr + 1) * (float(myParameters.maxIterationsNumber)
                                                      / float(myParameters.maxApproximationsNumber)));
    return MAXVALUE(20, maxIterationsNr);
}


double GaussSeidelHeat()
{
    double delta, new_x, norma_inf = 0.;
    short j;

    for (long i = 1; i < myStructure.nrNodes; i++)
        if (! nodeList[i].isSurface)
        {
            if (A[i][0].val != 0.)
            {
                j = 1;
                new_x = b[i];
                while ((A[i][j].index != NOLINK) && (j < myStructure.maxNrColumns))
                {
                    new_x -= A[i][j].val * X[A[i][j].index];
                    j++;
                }

                delta = fabs(new_x - X[i]);
                X[i] = new_x;

                if (delta > norma_inf)
                    norma_inf = delta;
            }
        }

    return norma_inf;
}


double GaussSeidelWater()
{
    double currentNorm = 0.;
    double infinityNorm = 0.;

    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        double newX = b[i];
        short j = 1;
        while ((A[i][j].index != NOLINK) && (j < myStructure.maxNrColumns))
        {
            newX -= A[i][j].val * X[A[i][j].index];
            j++;
        }

        // surface check (H cannot go below z)
        if (nodeList[i].isSurface && newX < nodeList[i].z)
        {
            newX = nodeList[i].z;
        }

        currentNorm = fabs(newX - X[i]);
        X[i] = newX;

        // water potential [m]
        double psi = fabs(newX - nodeList[i].z);
        if (psi > 1)
        {
            // normalized if psi > 1m
            currentNorm /= psi;
        }

        if (currentNorm > infinityNorm)
            infinityNorm = currentNorm;
    }

    return infinityNorm;
}


//-------------  THREADS -------------
void GaussSeidelThread(long start, long end, double *infinityNorm)
{
    *infinityNorm = 0.;

    for (long i = start; i <= end; i++)
    {
        double newX = b[i];
        short j = 1;
        while ((A[i][j].index != NOLINK) && (j < myStructure.maxNrColumns))
        {
            newX -= A[i][j].val * X[A[i][j].index];
            j++;
        }

        // surface check (H cannot go below z)
        if (nodeList[i].isSurface && newX < nodeList[i].z)
        {
            newX = nodeList[i].z;
        }

        double currentNorm = fabs(newX - X[i]);
        X[i] = newX;

        // water potential [m]
        double psi = fabs(newX - nodeList[i].z);
        if (psi > 1)
        {
            // normalized if psi > 1m
            currentNorm /= psi;
        }

        if (currentNorm > *infinityNorm)
            *infinityNorm = currentNorm;
    }
}


double iterationThreads()
{
    int nrThreads = myParameters.threadsNumber;
    int lastThread = nrThreads - 1;
    long step = myStructure.nrNodes / nrThreads;

    std::vector<std::thread> threadVector(nrThreads);
    std::vector<double> normVector(nrThreads);

    for (int n = 0; n < nrThreads; n++)
    {
        long start = n * step;
        long end = (n+1) * step - 1;

        if (n == lastThread)
            end = myStructure.nrNodes-1;

        threadVector[n] = std::thread(GaussSeidelThread, start, end, &(normVector[n]));
    }

    // wait threads
    for (auto& th : threadVector) {
        th.join();
    }

    // compute norm
    double infinityNorm = normVector[0];
    for (int n = 1; n < myParameters.threadsNumber; n++)
    {
        if (normVector[n] > infinityNorm)
            infinityNorm = normVector[n];
    }

    return infinityNorm;
}


//-------------  SOLVER -------------

bool solveLinearSystem(int approximation, double residualTolerance, int computationType)
{
    double currentNorm = 1.0;
    double bestNorm = currentNorm;

    int maxIterationsNr = getMaxIterationsNr(approximation);
    int iteration = 0;

    while ((currentNorm > residualTolerance) && (iteration < maxIterationsNr))
	{
        if (computationType == PROCESS_HEAT)
            currentNorm = GaussSeidelHeat();

        else if (computationType == PROCESS_WATER)
        {
            if (myParameters.threadsNumber == 1)
            {
                currentNorm = GaussSeidelWater();
            }
            else
            {
                // parallel computing
                currentNorm = iterationThreads();
            }

            if (currentNorm > (bestNorm * 10.0))
            {
                // non-convergent system
                return false;
            }
            else if (currentNorm < bestNorm)
            {
                bestNorm = currentNorm;
            }
        }

        iteration++;
	}

    return true;
}

