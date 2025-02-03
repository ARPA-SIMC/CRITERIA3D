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
#include "basicMath.h"
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
		double flow = matrixValue * (nodeList[i].H - nodeList[link->index].H) * deltaT;
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

    if (approximationNr == 0)
    {
        double flux_i = (nodeList[i].Qw * deltaT) / nodeList[i].volume_area;
        double flux_j = (nodeList[j].Qw * deltaT) / nodeList[j].volume_area;
        Hi = nodeList[i].oldH + flux_i;
        Hj = nodeList[j].oldH + flux_j;
    }
    else
    {
        Hi = nodeList[i].H;
        Hj = nodeList[j].H;

        /*
		Hi = (nodeList[i].H + nodeList[i].oldH) / 2.0;
        Hj = (nodeList[j].H + nodeList[j].oldH) / 2.0;
        */
    }

    double H = MAXVALUE(Hi, Hj);
    double z = MAXVALUE(nodeList[i].z + nodeList[i].pond, nodeList[j].z + nodeList[j].pond);
    double Hs = H - z;
    if (Hs <= 0.)
        return 0.;

    double dH = fabs(Hi - Hj);
    double cellDistance = distance2D(i,j);
    if ((dH/cellDistance) < EPSILON)
        return 0.;

    double roughness = (nodeList[i].Soil->Roughness + nodeList[j].Soil->Roughness) / 2.;

    //Manning
    double v = pow(Hs, 2./3.) * sqrt(dH/cellDistance) / roughness;
    double flowArea = link->area * Hs;

    Courant = MAXVALUE(Courant, v * deltaT / cellDistance);
    return (v * flowArea) / dH;
}


double infiltration(long sup, long inf, TlinkedNode *link, double deltaT)
{
    double cellDistance = nodeList[sup].z - nodeList[inf].z;

    /*! unsaturated */
    if (nodeList[inf].H < nodeList[sup].z)
    {
        /*! surface water content [m] */
        //double surfaceH = (nodeList[sup].H + nodeList[sup].oldH) * 0.5;

        double initialSurfaceWater = MAXVALUE(nodeList[sup].oldH - nodeList[sup].z, 0);     // [m]

        // precipitation: positive  -  evaporation: negative
        double precOrEvapRate = nodeList[sup].Qw / nodeList[sup].volume_area;               // [m s-1]

        /*! maximum water infiltration rate [m/s] */
        double maxInfiltrationRate = initialSurfaceWater / deltaT + precOrEvapRate;         // [m s-1]
        if (maxInfiltrationRate <= 0)
            return 0.;

        double dH = nodeList[sup].H - nodeList[inf].H;
        double maxK = maxInfiltrationRate * (cellDistance / dH);                            // [m s-1]

        /*! first soil layer: mean between current k and k_sat */
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

        double k = MINVALUE(meanK, maxK);
        return (k * link->area) / cellDistance;
    }
    else
    {
        /*! saturated */
        if (nodeList[inf].boundary != nullptr)
        {
            if (nodeList[inf].boundary->type == BOUNDARY_URBAN)
            {
                // TODO check?
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


bool computeFlux(long i, int matrixIndex, TlinkedNode *link, double deltaT, unsigned long myApprox, int linkType)
{
    if ((*link).index == NOLINK) return false;

    double val;
    long j = (*link).index;

    if (nodeList[i].isSurface)
    {
		if (nodeList[j].isSurface)
			val = runoff(i, j, link, deltaT, myApprox);
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

    if (myStructure.computeHeat &&
        ! nodeList[i].isSurface && ! nodeList[j].isSurface)
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
     bool isValidStep = false;

     int approximationNr = 0;
     do
     {
        Courant = 0.0;
        if (approximationNr == 0)
        {
            // diagonal indexes
            for (int i = 0; i < myStructure.nrNodes; i++)
            {
                A[i][0].index = i;
            }
        }

        /*! hydraulic conductivity and theta derivative */
        for (unsigned i = 0; i < unsigned(myStructure.nrNodes); i++)
        {
            invariantFlux[i] = 0.;
            if (! nodeList[i].isSurface)
            {
                nodeList[i].k = computeK(i);
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

        // update boundary conditions
        //updateBoundaryWater(deltaT);

        /*! computes the matrix elements */
        for (int i = 0; i < myStructure.nrNodes; i++)
        {
            short j = 1;
            if (computeFlux(i, j, &(nodeList[i].up), deltaT, approximationNr, UP))
                j++;
            for (short l = 0; l < myStructure.nrLateralLinks; l++)
            {
                if (computeFlux(i, j, &(nodeList[i].lateral[l]), deltaT, approximationNr, LATERAL))
                    j++;
            }
            if (computeFlux(i, j, &(nodeList[i].down), deltaT, approximationNr, DOWN))
                j++;

            /*! closure */
            while (j < myStructure.maxNrColumns)
                A[i][j++].index = NOLINK;

            j = 1;
            double sum = 0.;
            while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
            {
                sum += A[i][j].val;
                A[i][j].val *= -1.0;
                j++;
            }

            /*! diagonal */
            A[i][0].val = C[i]/deltaT + sum;

            /*! b vector (vector of constant terms) */
            b[i] = ((C[i] / deltaT) * nodeList[i].oldH) + nodeList[i].Qw + invariantFlux[i];

            /*! preconditioning */
            j = 1;
            while ((A[i][j].index != NOLINK) && (j < myStructure.maxNrColumns))
            {
                A[i][j++].val /= A[i][0].val;
            }
            b[i] /= A[i][0].val;
        }

        if (Courant > 1.0 && deltaT > myParameters.delta_t_min)
        {
            halveTimeStep();
            setForcedHalvedTime(true);
            return false;
        }

        if (! solver(approximationNr, myParameters.ResidualTolerance, PROCESS_WATER))
        {
            if (deltaT > myParameters.delta_t_min)
            {
                halveTimeStep();
                setForcedHalvedTime(true);
                return false;
            }
        }

        /*! set new potential - compute new degree of saturation */
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
    while ((! isValidStep) && (++approximationNr < myParameters.maxApproximationsNumber));

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
    return (isStepOK);
}


void restoreWater()
{
    for (long n = 0; n < myStructure.nrNodes; n++)
         nodeList[n].H = nodeList[n].oldH;
}
