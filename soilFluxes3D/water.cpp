/*!
    \name water.cpp
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
#include <float.h>

#include "physics.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "header/types.h"
#include "header/water.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/balance.h"
#include "header/boundary.h"
#include "header/heat.h"


/*!
 * \brief getWaterExchange
 * \param link TlinkedNode pointer
 * \param deltaT [s]
 * \return water exchange [m3] between node i and Link
 */
double getWaterExchange(long i, TlinkedNode *link, double deltaT)
{
    if (link != nullptr)
    {
		double matrixValue = getMatrixValue(i, link);
        return matrixValue * (nodeList[i].H - nodeList[link->index].H) * deltaT;
    }
	else
        return double(INDEX_ERROR);
}


/*!
 * \brief runoff
 * Manning equation
 * Qij=((Hi+Hj-zi-zj)/2)^(5/3) * Sij / roughness * sqrt(abs(Hi-Hj)/Lij) * sgn(Hi-Hj)
 * \param link      linked node pointer
 * \param deltaT    [s]
 */
double runoff(long i, long j, TlinkedNode *link, double deltaT, unsigned approximationNr)
{
    double Hi, Hj;
    if (approximationNr == 0)
    {
        double flux_i = (nodeList[i].Qw * deltaT) / nodeList[i].volume_area;
        double flux_j = (nodeList[j].Qw * deltaT) / nodeList[j].volume_area;
        Hi = nodeList[i].oldH + flux_i * 0.5;
        Hj = nodeList[j].oldH + flux_j * 0.5;
    }
    else
    {
        /*
        // avg value
        Hi = (nodeList[i].H + nodeList[i].oldH) * 0.5;
        Hj = (nodeList[j].H + nodeList[j].oldH) * 0.5;
        */
        Hi = (nodeList[i].H);
        Hj = (nodeList[j].H);
    }

    double dH = fabs(Hi - Hj);
    if (dH < 0.0001)
        return 0.;

    double zi = nodeList[i].z + nodeList[i].pond;
    double zj = nodeList[j].z + nodeList[j].pond;

    double Hmax = std::max(Hi, Hj);
    double zmax = std::max(zi, zj);
    double Hs = Hmax - zmax;
    if (Hs < 0.0001)
        return 0.;

    // Land depression
    if ((Hi > Hj && zi < zj) || (Hj > Hi && zj < zi))
    {
        Hs = std::min(Hs, dH);
    }

    double cellDistance = distance2D(i, j);
    double slope = dH / cellDistance;
    if (slope < EPSILON)
        return 0.;

    double roughness = (nodeList[i].Soil->roughness + nodeList[j].Soil->roughness) * 0.5;

    // Manning equation
    double v = pow(Hs, 2./3.) * sqrt(slope) / roughness;                // [m s-1]
    CourantWater = std::max(CourantWater, v * deltaT / cellDistance);

    double flowArea = link->area * Hs;                                  // [m2]
    return v * flowArea / dH;
}


double infiltration(long sup, long inf, TlinkedNode *link, double deltaT)
{
    double cellDistance = nodeList[sup].z - nodeList[inf].z;

    // unsaturated
    if (nodeList[inf].H < nodeList[sup].z)
    {
        double surfaceH = (nodeList[sup].H + nodeList[sup].oldH) * 0.5;
        double soilH = (nodeList[inf].H + nodeList[inf].oldH) * 0.5;

        // surface avg water content [m]
        double surfaceWater = std::max(surfaceH - nodeList[sup].z, 0.);              // [m]

        // precipitation: positive
        // evaporation: negative
        double precOrEvapRate = nodeList[sup].Qw / nodeList[sup].volume_area;       // [m s-1]

        // maximum water infiltration rate [m/s]
        double maxInfRate = surfaceWater / deltaT + precOrEvapRate;                 // [m s-1]
        if (maxInfRate < DBL_EPSILON)
            return 0.;

        double dH = surfaceH - soilH;                                               // [m]
        double maxK = maxInfRate * (cellDistance / dH);                             // [m s-1]

        // first soil layer: mean between current k and k_sat
        double meanK = computeMean(nodeList[inf].Soil->K_sat, nodeList[inf].k);

        if (nodeList[inf].boundary != nullptr)
        {
            if (nodeList[inf].boundary->type == BOUNDARY_URBAN)
            {
                // TODO improve with external parameters
                meanK *= 0.1;
            }
            else if (nodeList[inf].boundary->type == BOUNDARY_ROAD)
            {
                meanK = 0.0;
            }
        }

        double k = std::min(meanK, maxK);
        return (k * link->area) / cellDistance;
    }
    else
    {
        // saturated
        if (nodeList[inf].boundary != nullptr)
        {
            if (nodeList[inf].boundary->type == BOUNDARY_URBAN)
            {
                // TODO improve with external parameters
                return(nodeList[inf].Soil->K_sat * 0.1 * link->area) / cellDistance;
            }
            else if (nodeList[inf].boundary->type == BOUNDARY_ROAD)
            {
                return 0.;
            }
        }

        return(nodeList[inf].Soil->K_sat * link->area) / cellDistance;
    }
}


double redistribution(long i, TlinkedNode *link, int linkType)
{
    double cellDistance;
    double k1 = nodeList[i].k;
    double k2 = nodeList[(*link).index].k;

    /*! horizontal */
    if (linkType == LATERAL)
    {
        cellDistance = distance(i, (*link).index);
        k1 *= myParameters.k_lateral_vertical_ratio;
        k2 *= myParameters.k_lateral_vertical_ratio;
    }
    else
    {
        cellDistance = fabs(nodeList[i].z - nodeList[(*link).index].z);
    }
    double k = computeMean(k1, k2);

    return (k * link->area) / cellDistance;
}


bool computeFlux(long i, int matrixIndex, TlinkedNode *link, double deltaT, unsigned approximationNr, int linkType)
{
    if ((*link).index == NOLINK)
        return false;

    double val;
    long j = (*link).index;

    if (nodeList[i].isSurface)
    {
		if (nodeList[j].isSurface)
            val = runoff(i, j, link, deltaT, approximationNr);
        else
            val = infiltration(i, j, link, deltaT);
    }
    else
    {
        if (nodeList[j].isSurface)
            val = infiltration(j, i, link, deltaT);
        else
            val = redistribution(i, link, linkType);
    }

    A[i][matrixIndex].index = j;
    A[i][matrixIndex].val = val;

    if (myStructure.computeHeat && ! nodeList[i].isSurface && ! nodeList[j].isSurface)
    {
        if (myStructure.computeHeatVapor)
        {
            double vaporThermal;
            vaporThermal = ThermalVaporFlux(i, link, PROCESS_WATER, NODATA, NODATA) / WATER_DENSITY;
            invariantFlux[i] += vaporThermal;
        }

        double liquidThermal;
        liquidThermal = ThermalLiquidFlux(i, link, PROCESS_WATER, NODATA, NODATA);
        invariantFlux[i] += liquidThermal;
    }

    return true;
}


/*! ------------------ parallel computing -------------------------*/
void computeCapacity_thread(unsigned long start, unsigned long end)
{
    for (unsigned long i = start; i <= end; i++)
    {
        invariantFlux[i] = 0.;
        if (! nodeList[i].isSurface)
        {
            // hydraulic conductivity
            nodeList[i].k = computeK(i);

            // theta derivative
            double dThetadH = dTheta_dH(i);
            C[i] = nodeList[i].volume_area  * dThetadH;

            // vapor capacity term
            if (myStructure.computeHeat && myStructure.computeHeatVapor)
            {
                double avgTemperature = getTMean(i);
                double dthetavdh = dThetav_dH(i, avgTemperature, dThetadH);
                C[i] += nodeList[i].volume_area  * dthetavdh;
            }
        }
    }
}


void computeMatrixElements_thread(unsigned long start, unsigned long end, unsigned approximationNr, double deltaT)
{
    for (unsigned long i = start; i <= end; i++)
    {
        // UP
        short j = 1;
        if (computeFlux(i, j, &(nodeList[i].up), deltaT, approximationNr, UP))
            j++;

        // LATERAL
        for (short l = 0; l < myStructure.nrLateralLinks; l++)
        {
            if (computeFlux(i, j, &(nodeList[i].lateral[l]), deltaT, approximationNr, LATERAL))
                j++;
        }

        // DOWN
        if (computeFlux(i, j, &(nodeList[i].down), deltaT, approximationNr, DOWN))
            j++;

        // void elements
        while (j < myStructure.maxNrColumns)
            A[i][j++].index = NOLINK;

        j = 1;
        double sum = 0.0;
        while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
        {
            sum += A[i][j].val;
            A[i][j].val *= -1.0;
            j++;
        }

        // diagonal
        A[i][0].val = C[i]/deltaT + sum;

        // b vector (vector of constant terms)
        b[i] = ((C[i] / deltaT) * nodeList[i].oldH) + nodeList[i].Qw + invariantFlux[i];

        // preconditioning
        j = 1;
        while ((A[i][j].index != NOLINK) && (j < myStructure.maxNrColumns))
        {
            A[i][j++].val /= A[i][0].val;
        }
        b[i] /= A[i][0].val;
    }
}


bool waterFlowComputation(double deltaT)
 {
     // initialize the indices on the diagonal
     for (int i = 0; i < myStructure.nrNodes; i++)
     {
         A[i][0].index = i;
     }

     int nrThreads = myParameters.threadsNumber;
     int lastThread = nrThreads - 1;
     long step = myStructure.nrNodes / nrThreads;
     std::vector<std::thread> threadVector;

     bool isValidStep = false;
     unsigned approximationNr = 0;
     do
     {
        // compute capacity term
        for (int n = 0; n < nrThreads; n++)
        {
            unsigned long start = n * step;
            unsigned long end = (n+1) * step - 1;

            if (n == lastThread)
                end = myStructure.nrNodes-1;

            threadVector.push_back(std::thread(computeCapacity_thread, start, end));
        }
        // wait threads
        for (auto& th : threadVector) {
            th.join();
        }
        threadVector.clear();

        // update boundary conditions
        updateBoundaryWater(deltaT);

        CourantWater = 0.0;

        // compute matrix elements
        for (int n = 0; n < nrThreads; n++)
        {
            unsigned long start = n * step;
            unsigned long end = (n+1) * step - 1;

            if (n == lastThread)
            {
                end = myStructure.nrNodes-1;
            }

            threadVector.push_back(std::thread(computeMatrixElements_thread, start, end, approximationNr, deltaT));
        }
        // wait threads
        for (auto& th : threadVector) {
            th.join();
        }
        threadVector.clear();

        // check Courant
        if (CourantWater > 1. && deltaT > myParameters.delta_t_min)
        {
            myParameters.current_delta_t = std::max(myParameters.current_delta_t / CourantWater, myParameters.delta_t_min);
            if (myParameters.current_delta_t > 1.)
            {
                myParameters.current_delta_t = floor(myParameters.current_delta_t);
            }
            setForcedHalvedTime(true);
            return false;
        }

        if (! solveLinearSystem(approximationNr, myParameters.ResidualTolerance, PROCESS_WATER))
        {
            if (deltaT > myParameters.delta_t_min)
            {
                halveTimeStep();
                setForcedHalvedTime(true);
                return false;
            }
        }

        /*! set new potential and compute new degree of saturation */
        for (int i = 0; i < myStructure.nrNodes; i++)
        {
            nodeList[i].H = X[i];
            if (! nodeList[i].isSurface)
            {
                nodeList[i].Se = computeSe(unsigned(i));
            }
        }

        // check water balance
        isValidStep = waterBalance(deltaT, approximationNr);

        if (getForcedHalvedTime())
            return false;
    }
    while ((! isValidStep) && (++approximationNr < unsigned(myParameters.maxApproximationsNumber)));

    return isValidStep;
 }



/*!
  * \brief computes water fluxes in the assigned period.
  * We assume that, by means of maxTime, we are sure to not exit from meteorology of the assigned hour
  * \param maxTime [s] maximum period for computation (max 3600 s)
  * \param acceptedTime [s] current seconds for simulation step
  * \return
  */
bool computeWaterFluxes(double maxTime, double *acceptedTime)
{
     bool isStepOK = false;

     while (! isStepOK)
     {
        *acceptedTime = std::min(myParameters.current_delta_t, maxTime);

        /*! save the instantaneous H values - Prepare the solutions vector (X = H) */
        for (long n = 0; n < myStructure.nrNodes; n++)
        {
            nodeList[n].oldH = nodeList[n].H;
            X[n] = nodeList[n].H;
        }

        /*! assign Theta_e
            for the surface nodes C = area */
        for (long n = 0; n < myStructure.nrNodes; n++)
        {
            if (nodeList[n].isSurface)
                C[n] = nodeList[n].volume_area;
            else
                nodeList[n].Se = computeSe(unsigned(n));
        }

        /*! update boundary conditions */
        updateConductance();
        updateBoundaryWater(*acceptedTime);

        isStepOK = waterFlowComputation(*acceptedTime);

        if (! isStepOK) restoreWater();
    }
    return isStepOK;
}


void restoreWater()
{
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        nodeList[n].H = nodeList[n].oldH;
    }
}
