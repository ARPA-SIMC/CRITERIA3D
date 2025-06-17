/*!
    \name boundary.cpp
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
#include <iostream>
#include <algorithm>
#include <float.h>

#include "physics.h"
#include "commonConstants.h"
#include "header/types.h"
#include "header/solver.h"
#include "header/soilPhysics.h"
#include "header/boundary.h"
#include "header/soilFluxes3D.h"
#include "header/water.h"
#include "header/heat.h"



void initializeBoundary(Tboundary *myBoundary, int myType, float slope, float boundaryArea)
{
    (*myBoundary).type = short(myType);
	(*myBoundary).slope = slope;
    (*myBoundary).boundaryArea = boundaryArea;
    (*myBoundary).waterFlow = 0.;
    (*myBoundary).sumBoundaryWaterFlow = 0;
	(*myBoundary).prescribedTotalPotential = NODATA;

    if (myStructure.computeHeat)
    {
        (*myBoundary).Heat = new(TboundaryHeat);

        //surface characteristics
        (*myBoundary).Heat->heightWind = NODATA;
        (*myBoundary).Heat->heightTemperature = NODATA;
        (*myBoundary).Heat->roughnessHeight = NODATA;
        (*myBoundary).Heat->aerodynamicConductance = NODATA;
        (*myBoundary).Heat->soilConductance = NODATA;

        //atmospheric variables
        (*myBoundary).Heat->temperature = NODATA;
        (*myBoundary).Heat->relativeHumidity = NODATA;
        (*myBoundary).Heat->windSpeed = NODATA;
        (*myBoundary).Heat->netIrradiance = NODATA;

        //surface energy fluxes
        (*myBoundary).Heat->radiativeFlux = 0;
        (*myBoundary).Heat->latentFlux = 0;
        (*myBoundary).Heat->sensibleFlux = 0;
        (*myBoundary).Heat->advectiveHeatFlux = 0.;

        //bottom boundary
        (*myBoundary).Heat->fixedTemperature = NODATA;
        (*myBoundary).Heat->fixedTemperatureDepth = NODATA;
    }
    else (*myBoundary).Heat = nullptr;
}

double computeSoilSurfaceResistance(double thetaTop)
{	// soil surface resistance (s m-1)
    // Van De Griend and Owe (1994)
    const double THETAMIN = 0.15;
    double surfaceResistance = 10 * exp(0.3563 * (THETAMIN - thetaTop) * 100);
    return surfaceResistance;
}

double computeSoilSurfaceResistanceCG(double theta, double thetaSat)
{	// soil surface resistance (s m-1)
    // Camillo and Gurney (1986)
    return (-805 + 4140 * (thetaSat - theta));
}

/*!
 * \brief atmospheric sensible heat flux
 * \param i
 * \return sensible heat (W m-2)
 */
double computeAtmosphericSensibleFlux(long i)
{
    if (nodeList[i].boundary->Heat == nullptr || ! nodeList[nodeList[i].up.index].isSurface)
        return 0;

    double myPressure = pressureFromAltitude(double(nodeList[i].z));

    double myDeltaT = nodeList[i].boundary->Heat->temperature - nodeList[i].extra->Heat->T;

    double myCvAir = airVolumetricSpecificHeat(myPressure, nodeList[i].boundary->Heat->temperature);

    return (myCvAir * myDeltaT * nodeList[i].boundary->Heat->aerodynamicConductance);
}

/*!
 * \brief boundary vapor flux (evaporation/condensation)
 * \param i
 * \return vapor flux (kg m-2 s-1)
 */
double computeAtmosphericLatentFlux(long i)
{
    if (nodeList[i].boundary->Heat == nullptr || ! nodeList[nodeList[i].up.index].isSurface)
        return 0;

    double PressSat, ConcVapSat, BoundaryVapor;

    PressSat = saturationVaporPressure(nodeList[i].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, nodeList[i].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (nodeList[i].boundary->Heat->relativeHumidity / 100.);

    // kg m-3
    double myDeltaVapor = BoundaryVapor - soilFluxes3D::getNodeVapor(i);

    // m s-1
    double myTotalConductance = 1./((1./nodeList[i].boundary->Heat->aerodynamicConductance) + (1. / nodeList[i].boundary->Heat->soilConductance));

    // kg m-2 s-1
    double myVaporFlow = myDeltaVapor * myTotalConductance;

    return myVaporFlow;
}

/*!
 * \brief boundary vapor flux from surface water
 * \param i
 * \return vapor flux (kg m-2 s-1)
 */
double computeAtmosphericLatentFluxSurfaceWater(long i)
{
    if (! nodeList[i].isSurface)
        return 0.;

    if (nodeList[i].down.index == NOLINK)
        return 0.;

    long downIndex = nodeList[i].down.index;

    if (nodeList[downIndex].boundary->Heat == nullptr || nodeList[downIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return 0.;

    double PressSat, ConcVapSat, BoundaryVapor;

    // atmospheric vapor content (kg m-3)
    PressSat = saturationVaporPressure(nodeList[downIndex].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, nodeList[downIndex].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (nodeList[downIndex].boundary->Heat->relativeHumidity / 100.);

    // surface water vapor content (kg m-3) (assuming water temperature is the same of atmosphere)
    double myDeltaVapor = BoundaryVapor - ConcVapSat;

    // kg m-2 s-1
    // using aerodynamic conductance of index below (boundary for heat)
    double myVaporFlow = myDeltaVapor * nodeList[downIndex].boundary->Heat->aerodynamicConductance;

    return myVaporFlow;
}

/*!
 * \brief atmospheric latent heat flux (evaporation/condensation)
 * \param i
 * \return latent flux (W)
 */
double computeAtmosphericLatentHeatFlux(long i)
{
    if (nodeList[i].boundary->Heat == nullptr || ! nodeList[nodeList[i].up.index].isSurface)
        return 0;

    double latentHeatFlow = 0.;

    // J kg-1
    double lambda = latentHeatVaporization(nodeList[i].extra->Heat->T - ZEROCELSIUS);
    // waterFlow: vapor sink source (m3 s-1)
    latentHeatFlow = nodeList[i].boundary->waterFlow * WATER_DENSITY * lambda;

    return latentHeatFlow;
}


// [-]
double getSurfaceWaterFraction(int i)
{
    if (! nodeList[i].isSurface)
        return 0.0;
    else
    {
        double h = std::max(nodeList[i].H - nodeList[i].z, 0.);       // [m]
        double h0 = std::max(double(nodeList[i].pond), 0.001);        // [m]
        return std::min(h / h0, 1.);
    }
}


void updateConductance()
{
    if (myStructure.computeHeat)
    {
        for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (nodeList[i].boundary != nullptr)
            {
                if (nodeList[i].extra->Heat != nullptr)
                {
                    if (nodeList[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                    {
                        // update aerodynamic conductance
                        nodeList[i].boundary->Heat->aerodynamicConductance =
                                aerodynamicConductance(nodeList[i].boundary->Heat->heightTemperature,
                                    nodeList[i].boundary->Heat->heightWind,
                                    nodeList[i].extra->Heat->T,
                                    nodeList[i].boundary->Heat->roughnessHeight,
                                    nodeList[i].boundary->Heat->temperature,
                                    nodeList[i].boundary->Heat->windSpeed);

                        if (myStructure.computeWater)
                        {
                            // update soil surface conductance
                            double theta = theta_from_sign_Psi(nodeList[i].H - nodeList[i].z, i);
                            nodeList[i].boundary->Heat->soilConductance = 1./ computeSoilSurfaceResistance(theta);
                        }
                    }
                }
            }
        }
    }
}


void updateBoundaryWater (double deltaT)
{
    double const EPSILON_RUNOFF = 0.001;          // [m] 1 mm

    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        // initialize: water sink-source
        nodeList[i].Qw = nodeList[i].waterSinkSource;

        if (nodeList[i].boundary != nullptr)
        {
            nodeList[i].boundary->waterFlow = 0.;

            if (nodeList[i].boundary->type == BOUNDARY_RUNOFF)
            {
                double avgH = (nodeList[i].H + nodeList[i].oldH) * 0.5;        // [m]

                // Surface water available for runoff [m]
                double hs = std::max(avgH - (nodeList[i].z + nodeList[i].pond), 0.);
                if (hs > EPSILON_RUNOFF)
                {
                    double maxFlow = (hs * nodeList[i].volume_area) / deltaT;         // [m3 s-1] maximum flow available during the time step
                    // Manning equation
                    double v = pow(hs, 2./3.) * sqrt(nodeList[i].boundary->slope) / nodeList[i].Soil->roughness;
                    // on the surface boundaryArea is a side [m]
                    double flow = nodeList[i].boundary->boundaryArea * hs * v;        // [m3 s-1]
                    nodeList[i].boundary->waterFlow = -std::min(flow, maxFlow);
                }
            }
            else if (nodeList[i].boundary->type == BOUNDARY_FREEDRAINAGE)
            {
                // Darcy unit gradient
                // dH=dz=L  -> dH/L=1
                nodeList[i].boundary->waterFlow = -nodeList[i].k * nodeList[i].up.area;
            }

            else if (nodeList[i].boundary->type == BOUNDARY_FREELATERALDRAINAGE)
            {
                // Darcy gradient = slope
                // dH=dz slope=dz/L -> dH/L=slope
                nodeList[i].boundary->waterFlow = -nodeList[i].k * myParameters.k_lateral_vertical_ratio
                                            * nodeList[i].boundary->boundaryArea * nodeList[i].boundary->slope;
            }

            else if (nodeList[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
            {
                // water table
                double L = 1.0;                             // [m]
                double boundaryZ = nodeList[i].z - L;       // [m]
                double boundaryK;                           // [m s-1]

                if (nodeList[i].boundary->prescribedTotalPotential >= boundaryZ)
                {
                    // saturated
                    boundaryK = nodeList[i].Soil->K_sat;
                }
                else
                {
                    // unsaturated
                    double boundaryPsi = fabs(nodeList[i].boundary->prescribedTotalPotential - boundaryZ);
                    double boundarySe = computeSefromPsi_unsat(boundaryPsi, nodeList[i].Soil);
                    boundaryK = computeWaterConductivity(boundarySe, nodeList[i].Soil);
                }

                double meanK = computeMean(nodeList[i].k, boundaryK);
                double dH = nodeList[i].boundary->prescribedTotalPotential - nodeList[i].H;
                nodeList[i].boundary->waterFlow = meanK * nodeList[i].boundary->boundaryArea * (dH / L);
            }

            else if (nodeList[i].boundary->type == BOUNDARY_HEAT_SURFACE)
            {
                if (myStructure.computeHeat && myStructure.computeHeatVapor)
                {
                    long upIndex = NOLINK;

                    double surfaceWaterFraction = 0.;
                    if (nodeList[i].up.index != NOLINK)
                    {
                        upIndex = nodeList[i].up.index;
                        surfaceWaterFraction = getSurfaceWaterFraction(upIndex);
                    }

                    double evapFromSoil = computeAtmosphericLatentFlux(i) / WATER_DENSITY * nodeList[i].up.area;

                    // surface water
                    if (surfaceWaterFraction > 0. && upIndex != NOLINK)
                    {
                        double waterVolume = (nodeList[upIndex].H - nodeList[upIndex].z) * nodeList[upIndex].volume_area;
                        double evapFromSurface = computeAtmosphericLatentFluxSurfaceWater(upIndex) / WATER_DENSITY * nodeList[i].up.area;

                        evapFromSoil *= (1. - surfaceWaterFraction);
                        evapFromSurface *= surfaceWaterFraction;

                        evapFromSurface = std::max(evapFromSurface, -waterVolume / deltaT);

                        if (nodeList[upIndex].boundary != nullptr)
                        {
                            nodeList[upIndex].boundary->waterFlow = evapFromSurface;
                        }
                        else
                        {
                            nodeList[upIndex].Qw += evapFromSurface;
                        }
                    }

                    if (evapFromSoil < 0.)
                    {
                        evapFromSoil = std::max(evapFromSoil, -(theta_from_Se(i) - nodeList[i].Soil->Theta_r) * nodeList[i].volume_area / deltaT);
                    }
                    else
                    {
                        evapFromSoil = std::min(evapFromSoil, (nodeList[i].Soil->Theta_s - nodeList[i].Soil->Theta_r) * nodeList[i].volume_area / deltaT);
                    }

                    nodeList[i].boundary->waterFlow = evapFromSoil;
                }
            }

            // check epsilon
            if (abs(nodeList[i].boundary->waterFlow) > DBL_EPSILON)
            {
                nodeList[i].Qw += nodeList[i].boundary->waterFlow;
            }
            else
            {
                nodeList[i].boundary->waterFlow = 0.;
            }
        }
    }

	// Culvert
	if (myCulvert.index != NOLINK)
	{
		long i = myCulvert.index;
		double waterLevel = 0.5 * (nodeList[i].H + nodeList[i].oldH) - nodeList[i].z;		// [m]

        double flow = 0.0;                                                          // [m3 s-1]

		if (waterLevel >= myCulvert.height * 1.5)
		{
			// pressure flow - Hazen-Williams equation
			double equivalentDiameter = sqrt((4. * myCulvert.width * myCulvert.height) / PI);
			// roughness = 70 (rough concrete)
            flow = (70.0 * pow(myCulvert.slope, 0.54) * pow(equivalentDiameter, 2.63)) / 3.591;

		}
		else if (waterLevel > myCulvert.height)
		{
			// mixed flow: open channel - pressure flow
            double wettedPerimeter = myCulvert.width + 2.* myCulvert.height;                // [m]
            double hydraulicRadius = nodeList[i].boundary->boundaryArea / wettedPerimeter;	// [m]

            // maximum Manning flow [m3 s-1]
            double ManningFlow = (nodeList[i].boundary->boundaryArea / myCulvert.roughness)
                                * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2./3.);

			// pressure flow - Hazen-Williams equation - roughness = 70
			double equivalentDiameter = sqrt((4. * myCulvert.width * myCulvert.height) / PI);
            double pressureFlow = (70.0 * pow(myCulvert.slope, 0.54) * pow(equivalentDiameter, 2.63)) / 3.591;

			double weight = (waterLevel - myCulvert.height) / (myCulvert.height * 0.5);
			flow = weight * pressureFlow + (1. - weight) * ManningFlow;

		}
        else if (waterLevel > nodeList[i].pond)
		{
			// open channel flow
            double boundaryArea = myCulvert.width * waterLevel;					// [m2]
            double wettedPerimeter = myCulvert.width + 2.0 * waterLevel;        // [m]
            double hydraulicRadius = boundaryArea / wettedPerimeter;			// [m]

            // Manning equation [m3 s-1]
            flow = (boundaryArea / myCulvert.roughness) * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2./3.);
		}

		// set boundary
		nodeList[i].boundary->waterFlow = -flow;
		nodeList[i].Qw += nodeList[i].boundary->waterFlow;
	}
}


bool updateBoundaryHeat(double timeStep, double &reducedTimeStep)
{
    double CourantHeatBoundary = 0;

    for (long i = 1; i < myStructure.nrNodes; i++)
    {
        if (isHeatNode(i))
        {
            nodeList[i].extra->Heat->Qh = nodeList[i].extra->Heat->sinkSource;

            if (nodeList[i].boundary != nullptr)
            {
                if (nodeList[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                {
                    nodeList[i].boundary->Heat->advectiveHeatFlux = 0.;
                    nodeList[i].boundary->Heat->sensibleFlux = 0.;
                    nodeList[i].boundary->Heat->latentFlux = 0.;
                    nodeList[i].boundary->Heat->radiativeFlux = 0.;

                    if (nodeList[i].boundary->Heat->netIrradiance != NODATA)
                        nodeList[i].boundary->Heat->radiativeFlux = nodeList[i].boundary->Heat->netIrradiance;

                    nodeList[i].boundary->Heat->sensibleFlux += computeAtmosphericSensibleFlux(i);

                    if (myStructure.computeWater && myStructure.computeHeatVapor)
                        nodeList[i].boundary->Heat->latentFlux += computeAtmosphericLatentHeatFlux(i) / nodeList[i].up.area;

                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        double advTemperature = nodeList[i].boundary->Heat->temperature;

                        // advective heat from rain
                        double waterFlux = nodeList[i].up.linkedExtra->heatFlux->waterFlux;
                        if (waterFlux > 0.)
                        {
                            nodeList[i].boundary->Heat->advectiveHeatFlux = waterFlux * HEAT_CAPACITY_WATER * advTemperature / nodeList[i].up.area;
                        }

                        // advective heat from evaporation/condensation
                        if (nodeList[i].boundary->waterFlow < 0.)
                            advTemperature = nodeList[i].extra->Heat->T;

                        nodeList[i].boundary->Heat->advectiveHeatFlux += nodeList[i].boundary->waterFlow * WATER_DENSITY * HEAT_CAPACITY_WATER_VAPOR * advTemperature / nodeList[i].up.area;
                    }

                    nodeList[i].extra->Heat->Qh += nodeList[i].up.area * (nodeList[i].boundary->Heat->radiativeFlux +
                                                                      nodeList[i].boundary->Heat->sensibleFlux +
                                                                      nodeList[i].boundary->Heat->latentFlux +
                                                                      nodeList[i].boundary->Heat->advectiveHeatFlux);

                    // [J m-3 K-1]
                    double heatCapacity = SoilHeatCapacity(i, nodeList[i].oldH, nodeList[i].extra->Heat->oldT);
                    // TODO [K] ?
                    double currentCourant = fabs(nodeList[i].extra->Heat->Qh) * timeStep / (heatCapacity * nodeList[i].volume_area);
                    CourantHeatBoundary = std::max(CourantHeatBoundary, currentCourant);
                }
                else if (nodeList[i].boundary->type == BOUNDARY_FREEDRAINAGE || nodeList[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
                {
                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        double waterFlux = nodeList[i].boundary->waterFlow;

                        double advTemperature = nodeList[i].boundary->Heat->fixedTemperature;
                        if (waterFlux < 0)
                        {
                            advTemperature = nodeList[i].extra->Heat->T;
                        }

                        nodeList[i].boundary->Heat->advectiveHeatFlux = waterFlux * HEAT_CAPACITY_WATER * advTemperature / nodeList[i].up.area;

                        nodeList[i].extra->Heat->Qh += nodeList[i].up.area * nodeList[i].boundary->Heat->advectiveHeatFlux;
                    }

                    if (nodeList[i].boundary->Heat->fixedTemperature != NODATA)
                    {
                        double avgH = getHMean(i);
                        double boundaryHeatConductivity = SoilHeatConductivity(i, nodeList[i].extra->Heat->T, avgH - nodeList[i].z);
                        double deltaT = nodeList[i].boundary->Heat->fixedTemperature - nodeList[i].extra->Heat->T;
                        nodeList[i].extra->Heat->Qh += boundaryHeatConductivity * deltaT / nodeList[i].boundary->Heat->fixedTemperatureDepth * nodeList[i].up.area;
                    }
                }
            }
        }
    }

    if (CourantHeatBoundary > 1. && timeStep > myParameters.delta_t_min)
    {
        reducedTimeStep = std::max(timeStep / CourantHeatBoundary, myParameters.delta_t_min);
        if (reducedTimeStep > 1.)
        {
            reducedTimeStep = floor(reducedTimeStep);
        }

        return false;
    }

    return true;
}
