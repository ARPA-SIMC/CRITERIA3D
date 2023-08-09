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

#include "physics.h"
#include "commonConstants.h"
#include "header/types.h"
#include "header/water.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/balance.h"
#include "header/boundary.h"
#include "header/heat.h"


/*!
 * \brief [m^3] water flow between node i and Link
 * \param i
 * \param link TlinkedNode pointer
 * \param deltaT
 * \return result
 */
double getWaterExchange(long i, TlinkedNode *link, double deltaT)
{
    if (link != nullptr)
        {
		double matrixValue = getMatrixValue(i, link);
		double flow = matrixValue * (myNode[i].H - myNode[link->index].H) * deltaT;
        return (flow);
        }
	else
		return(double(INDEX_ERROR));
}


/*!
 * \brief runoff
 * Manning formulation
 * Qij=((Hi+Hj-zi-zj)/2)^(5/3) * Sij / roughness * sqrt(abs(Hi-Hj)/Lij) * sgn(Hi-Hj)
 * \param i
 * \param j
 * \param link TlinkedNode pointer
 * \param deltaT
 * \param approximationNr
 * \return result
 */
double runoff(long i, long j, TlinkedNode *link, double deltaT, unsigned long approximationNr)
{
    double Hi, Hj;
    double const EPSILON_mm = 0.0001;

    if (approximationNr == 0)
    {
        double flux_i = (myNode[i].Qw * deltaT) / myNode[i].volume_area;
        double flux_j = (myNode[j].Qw * deltaT) / myNode[j].volume_area;
        Hi = myNode[i].oldH + flux_i;
        Hj = myNode[j].oldH + flux_j;
    }
    else
    {
		
		Hi = myNode[i].H;
		Hj = myNode[j].H;
		/*
		Hi = (myNode[i].H + myNode[i].oldH) / 2.0;
        Hj = (myNode[j].H + myNode[j].oldH) / 2.0;
		*/
    }


    double H = MAXVALUE(Hi, Hj);
    double z = MAXVALUE(myNode[i].z + myNode[i].Soil->Pond, myNode[j].z + myNode[j].Soil->Pond);
    double Hs = H - z;
    if (Hs <= 0.) return(0.);

    double dH = fabs(Hi - Hj);
    double cellDistance = distance2D(i,j);
    if ((dH/cellDistance) < EPSILON_mm) return(0.);

    double roughness = (myNode[i].Soil->Roughness + myNode[j].Soil->Roughness) / 2.;

    //Manning
    double v = pow(Hs, 2./3.) * sqrt(dH/cellDistance) / roughness;
    double flowArea = link->area * Hs;

    Courant = MAXVALUE(Courant, v * deltaT / cellDistance);
    return (v * flowArea) / dH;
}



double infiltration(long sup, long inf, TlinkedNode *link, double deltaT)
{
 double cellDistance = (myNode[sup].z - myNode[inf].z) * 2.0;

 /*! unsaturated */
 if (myNode[inf].H < myNode[sup].z)
        {
        /*! surface water content [m] */
        // double surfaceH = (myNode[sup].H + myNode[sup].oldH) * 0.5;
		double surfaceH = myNode[sup].H;

        /*! maximum water infiltration rate [m/s] */
        double maxInfiltrationRate = (surfaceH - myNode[sup].z) / deltaT;
        if (maxInfiltrationRate <= 0.0) return(0.0);

        /*! first soil layer: mean between current k and k_sat */
        double meanK = computeMean(myNode[inf].k, myNode[inf].Soil->K_sat);

        double dH = myNode[sup].H - myNode[inf].H;
        double maxK = maxInfiltrationRate * (cellDistance / dH);

        double k = MINVALUE(meanK , maxK);

        return (k * link->area) / cellDistance;
        }

 /*! saturated */
 else
    {
        return(myNode[inf].Soil->K_sat * link->area) / cellDistance;
    }

}


double redistribution(long i, TlinkedNode *link, int linkType)
{
    double cellDistance;
    double k1 = myNode[i].k;
    double k2 = myNode[(*link).index].k;

    /*! horizontal */
    if (linkType == LATERAL)
    {
        cellDistance = distance(i, (*link).index);
        k1 *= myParameters.k_lateral_vertical_ratio;
        k2 *= myParameters.k_lateral_vertical_ratio;
    }
    else
    {
        cellDistance = fabs(myNode[i].z - myNode[(*link).index].z);
    }
    double k = computeMean(k1, k2);

    return (k * link->area) / cellDistance;
}



bool computeFlux(long i, int matrixIndex, TlinkedNode *link, double deltaT, unsigned long myApprox, int linkType)
{
    if ((*link).index == NOLINK) return false;

    double val;
    long j = (*link).index;

    if (myNode[i].isSurface)
    {
		if (myNode[j].isSurface)
			val = runoff(i, j, link, deltaT, myApprox);
        else
            val = infiltration(i, j, link, deltaT);
    }
    else
    {
        if (myNode[j].isSurface)
            val = infiltration(j, i, link, deltaT);
        else
            val = redistribution(i, link, linkType);
    }

    A[i][matrixIndex].index = j;
    A[i][matrixIndex].val = val;

    if (myStructure.computeHeat &&
        ! myNode[i].isSurface && ! myNode[j].isSurface)
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


bool waterFlowComputation(double deltaT)
 {
     bool isValidStep;
     long i;
     double dThetadH, dthetavdh;
     double avgTemperature;

     int approximationNr = 0;
     do
     {
        Courant = 0.0;
        if (approximationNr == 0)
        {
            for (i = 0; i < myStructure.nrNodes; i++)
            {
                A[i][0].index = i;
            }
        }

        /*! hydraulic conductivity and theta derivative */
        for (i = 0; i < myStructure.nrNodes; i++)
        {
            invariantFlux[i] = 0.;
            if (!myNode[i].isSurface)
            {
                myNode[i].k = computeK(unsigned(i));
                dThetadH = dTheta_dH(unsigned(i));
                 C[i] = myNode[i].volume_area  * dThetadH;

                 // vapor capacity term
                 if (myStructure.computeHeat && myStructure.computeHeatVapor)
                 {
                     avgTemperature = getTMean(i);
                     dthetavdh = dThetav_dH(unsigned(i), avgTemperature, dThetadH);
                     C[i] += myNode[i].volume_area  * dthetavdh;
                 }
            }
        }

        // update boundary conditions
        // updateBoundaryWater(deltaT);

        /*! computes the matrix elements */
        for (i = 0; i < myStructure.nrNodes; i++)
        {
            short j = 1;
            if (computeFlux(i, j, &(myNode[i].up), deltaT, approximationNr, UP)) j++;
            for (short l = 0; l < myStructure.nrLateralLinks; l++)
                    if (computeFlux(i, j, &(myNode[i].lateral[l]), deltaT, approximationNr, LATERAL)) j++;
            if (computeFlux(i, j, &(myNode[i].down), deltaT, approximationNr, DOWN)) j++;

            /*! closure */
            while (j < myStructure.maxNrColumns) A[i][j++].index = NOLINK;

            j = 1;
            double sum = 0.;
            while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
            {
                sum += A[i][j].val;
                A[i][j].val *= -1.0;
                j++;
            }

            /*! sum of the diagonal elements */
            A[i][0].val = C[i]/deltaT + sum;

            /*! b vector(vector of constant terms) */
            b[i] = ((C[i] / deltaT) * myNode[i].oldH) + myNode[i].Qw + invariantFlux[i];

            /*! preconditioning */
            j = 1;
            while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
                    A[i][j++].val /= A[i][0].val;
            b[i] /= A[i][0].val;
        }

        if (Courant > 1.0)
            if (deltaT > myParameters.delta_t_min)
            {
                halveTimeStep();
                setForcedHalvedTime(true);
                return false;
            }

        if (! GaussSeidelRelaxation(approximationNr, myParameters.ResidualTolerance, PROCESS_WATER))
            if (deltaT > myParameters.delta_t_min)
            {
                halveTimeStep();
                setForcedHalvedTime(true);
                return (false);
            }

        /*! set new potential - compute new degree of saturation */
        for (i = 0; i < myStructure.nrNodes; i++)
        {
            myNode[i].H = X[i];
            if (!myNode[i].isSurface)
                myNode[i].Se = computeSe(unsigned(i));
        }

        /*! water balance */
        isValidStep = waterBalance(deltaT, approximationNr);
        if (getForcedHalvedTime()) return (false);
        }
    while ((!isValidStep) && (++approximationNr < myParameters.maxApproximationsNumber));

    return isValidStep;
 }



/*!
  * \brief computes water balance in the assigned period.
  * We assume that, by means of maxTime, we are sure to not exit from meteorology of the assigned hour
  * \param maxTime [s] maximum period for computation (max 3600 s)
  * \param acceptedTime [s] current seconds for simulation step
  * \return
  */
bool computeWater(double maxTime, double *acceptedTime)
{
     bool isStepOK = false;

     while (!isStepOK)
     {
        *acceptedTime = MINVALUE(myParameters.current_delta_t, maxTime);

        /*! save the instantaneous H values - Prepare the solutions vector (X = H) */
        for (long n = 0; n < myStructure.nrNodes; n++)
                {
                myNode[n].oldH = myNode[n].H;
                X[n] = myNode[n].H;
                }

        /*! assign Theta_e
            for the surface nodes C = area */
        for (long n = 0; n < myStructure.nrNodes; n++)
        {
            if (myNode[n].isSurface)
                C[n] = myNode[n].volume_area;
            else
                myNode[n].Se = computeSe(unsigned(n));
        }

        /*! update boundary conditions */
        updateBoundary();
        updateBoundaryWater(*acceptedTime);

        isStepOK = waterFlowComputation(*acceptedTime);

        if (!isStepOK) restoreWater();
    }
    return (isStepOK);
}


void restoreWater()
{
    for (long n = 0; n < myStructure.nrNodes; n++)
         myNode[n].H = myNode[n].oldH;
}
