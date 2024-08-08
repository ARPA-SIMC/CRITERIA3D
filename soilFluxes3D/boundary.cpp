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

#include "physics.h"
#include "commonConstants.h"
#include "header/types.h"
#include "header/solver.h"
#include "header/soilPhysics.h"
#include "header/boundary.h"
#include "header/soilFluxes3D.h"
#include "header/water.h"
#include "header/heat.h"

#include <iostream>

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

double computeSoilSurfaceResistance(double myThetaTop)
{	// soil surface resistance (s m-1)
    // Van De Griend and Owe (1994)
    const double THETAMIN = 0.15;
    return (10 * exp(0.3563 * (THETAMIN - myThetaTop) * 100));
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
    if (nodeListPtr[i].boundary->Heat == nullptr || ! nodeListPtr[nodeListPtr[i].up.index].isSurface)
        return 0;

    double myPressure = pressureFromAltitude(double(nodeListPtr[i].z));

    double myDeltaT = nodeListPtr[i].boundary->Heat->temperature - nodeListPtr[i].extra->Heat->T;

    double myCvAir = airVolumetricSpecificHeat(myPressure, nodeListPtr[i].boundary->Heat->temperature);

    return (myCvAir * myDeltaT * nodeListPtr[i].boundary->Heat->aerodynamicConductance);
}

/*!
 * \brief boundary vapor flux (evaporation/condensation)
 * \param i
 * \return vapor flux (kg m-2 s-1)
 */
double computeAtmosphericLatentFlux(long i)
{
    if (nodeListPtr[i].boundary->Heat == nullptr || ! nodeListPtr[nodeListPtr[i].up.index].isSurface)
        return 0;

    double PressSat, ConcVapSat, BoundaryVapor;

    PressSat = saturationVaporPressure(nodeListPtr[i].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, nodeListPtr[i].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (nodeListPtr[i].boundary->Heat->relativeHumidity / 100.);

    // kg m-3
    double myDeltaVapor = BoundaryVapor - soilFluxes3D::getNodeVapor(i);

    // m s-1
    double myTotalConductance = 1./((1./nodeListPtr[i].boundary->Heat->aerodynamicConductance) + (1. / nodeListPtr[i].boundary->Heat->soilConductance));

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
    if (! nodeListPtr[i].isSurface) return 0.;
    if (&(nodeListPtr[i].down) == nullptr) return 0.;

    long downIndex = nodeListPtr[i].down.index;

    if (nodeListPtr[downIndex].boundary->Heat == nullptr || nodeListPtr[downIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return 0.;

    double PressSat, ConcVapSat, BoundaryVapor;

    // atmospheric vapor content (kg m-3)
    PressSat = saturationVaporPressure(nodeListPtr[downIndex].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, nodeListPtr[downIndex].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (nodeListPtr[downIndex].boundary->Heat->relativeHumidity / 100.);

    // surface water vapor content (kg m-3) (assuming water temperature is the same of atmosphere)
    double myDeltaVapor = BoundaryVapor - ConcVapSat;

    // kg m-2 s-1
    // using aerodynamic conductance of index below (boundary for heat)
    double myVaporFlow = myDeltaVapor * nodeListPtr[downIndex].boundary->Heat->aerodynamicConductance;

    return myVaporFlow;
}

/*!
 * \brief atmospheric latent heat flux (evaporation/condensation)
 * \param i
 * \return latent flux (W)
 */
double computeAtmosphericLatentHeatFlux(long i)
{
    if (nodeListPtr[i].boundary->Heat == nullptr || ! nodeListPtr[nodeListPtr[i].up.index].isSurface)
        return 0;

    double latentHeatFlow = 0.;

    // J kg-1
    double lambda = latentHeatVaporization(nodeListPtr[i].extra->Heat->T - ZEROCELSIUS);
    // waterFlow: vapor sink source (m3 s-1)
    latentHeatFlow = nodeListPtr[i].boundary->waterFlow * WATER_DENSITY * lambda;

    return latentHeatFlow;
}

double getSurfaceWaterFraction(int i)
{
    if (! nodeListPtr[i].isSurface)
        return 0.0;
    else
    {
        double h = std::max(nodeListPtr[i].H - nodeListPtr[i].z, 0.);       // [m]
        double h0 = std::max(double(nodeListPtr[i].pond), 0.001);           // [m]
        return h / h0;
    }
}

void updateConductance()
{
    if (myStructure.computeHeat)
    {
        for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (nodeListPtr[i].boundary != nullptr)
            {
                if (nodeListPtr[i].extra->Heat != nullptr)
                {
                    if (nodeListPtr[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                    {
                        // update aerodynamic conductance
                        nodeListPtr[i].boundary->Heat->aerodynamicConductance =
                                aerodynamicConductance(nodeListPtr[i].boundary->Heat->heightTemperature,
                                    nodeListPtr[i].boundary->Heat->heightWind,
                                    nodeListPtr[i].extra->Heat->T,
                                    nodeListPtr[i].boundary->Heat->roughnessHeight,
                                    nodeListPtr[i].boundary->Heat->temperature,
                                    nodeListPtr[i].boundary->Heat->windSpeed);

                        if (myStructure.computeWater)
                        {
                            // update soil surface conductance
                            double theta = theta_from_sign_Psi(nodeListPtr[i].H - nodeListPtr[i].z, i);
                            nodeListPtr[i].boundary->Heat->soilConductance = 1./ computeSoilSurfaceResistance(theta);
                        }
                    }
                }
            }
        }
    }
}


void updateBoundaryWater (double deltaT)
{
    double const EPSILON_METER = 0.0001;          // [m] 0.1 mm

    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        // water sink-source
        nodeListPtr[i].Qw = nodeListPtr[i].waterSinkSource;

        if (nodeListPtr[i].boundary != nullptr)
        {
            nodeListPtr[i].boundary->waterFlow = 0.;

            if (nodeListPtr[i].boundary->type == BOUNDARY_RUNOFF)
            {
                double avgH = (nodeListPtr[i].H + nodeListPtr[i].oldH) * 0.5;        // [m]

                // Surface water available for runoff [m]
                double hs = MAXVALUE(avgH - (nodeListPtr[i].z + nodeListPtr[i].pond), 0.);
                if (hs > EPSILON_METER)
                {
                    double maxFlow = (hs * nodeListPtr[i].volume_area) / deltaT;         // [m3 s-1] maximum flow available during the time step
                    // Manning equation
                    double v = (1. / nodeListPtr[i].Soil->Roughness) * pow(hs, 2./3.) * sqrt(nodeListPtr[i].boundary->slope);
                    // on the surface boundaryArea is a side [m]
                    double flow = nodeListPtr[i].boundary->boundaryArea * hs * v;        // [m3 s-1]
                    nodeListPtr[i].boundary->waterFlow = -MINVALUE(flow, maxFlow);
                }
            }
            else if (nodeListPtr[i].boundary->type == BOUNDARY_FREEDRAINAGE)
            {
                // Darcy unit gradient
                // dH=dz=L  -> dH/L=1
                nodeListPtr[i].boundary->waterFlow = -nodeListPtr[i].k * nodeListPtr[i].up.area;
            }

            else if (nodeListPtr[i].boundary->type == BOUNDARY_FREELATERALDRAINAGE)
            {
                // Darcy gradient = slope
                // dH=dz slope=dz/L -> dH/L=slope
                nodeListPtr[i].boundary->waterFlow = -nodeListPtr[i].k * myParameters.k_lateral_vertical_ratio
                                            * nodeListPtr[i].boundary->boundaryArea * nodeListPtr[i].boundary->slope;
            }

            else if (nodeListPtr[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
            {
                // water table
                double L = 1.0;                         // [m]
                double boundaryZ = nodeListPtr[i].z - L;     // [m]
                double boundaryK;                       // [m s-1]

                if (nodeListPtr[i].boundary->prescribedTotalPotential >= boundaryZ)
                {
                    // saturated
                    boundaryK = nodeListPtr[i].Soil->K_sat;
                }
                else
                {
                    // unsaturated
                    double boundaryPsi = fabs(nodeListPtr[i].boundary->prescribedTotalPotential - boundaryZ);
                    double boundarySe = computeSefromPsi_unsat(boundaryPsi, nodeListPtr[i].Soil);
                    boundaryK = computeWaterConductivity(boundarySe, nodeListPtr[i].Soil);
                }

                double meanK = computeMean(nodeListPtr[i].k, boundaryK);
                double dH = nodeListPtr[i].boundary->prescribedTotalPotential - nodeListPtr[i].H;
                nodeListPtr[i].boundary->waterFlow = meanK * nodeListPtr[i].boundary->boundaryArea * (dH / L);
            }

            else if (nodeListPtr[i].boundary->type == BOUNDARY_HEAT_SURFACE)
            {
                if (myStructure.computeHeat && myStructure.computeHeatVapor)
                {
                    long upIndex;

                    double surfaceWaterFraction = 0.;
                    if (&(nodeListPtr[i].up) != nullptr)
                    {
                        upIndex = nodeListPtr[i].up.index;
                        surfaceWaterFraction = getSurfaceWaterFraction(upIndex);
                    }

                    double evapFromSoil = computeAtmosphericLatentFlux(i) / WATER_DENSITY * nodeListPtr[i].up.area;

                    // surface water
                    if (surfaceWaterFraction > 0.)
                    {
                        double waterVolume = (nodeListPtr[upIndex].H - nodeListPtr[upIndex].z) * nodeListPtr[upIndex].volume_area;
                        double evapFromSurface = computeAtmosphericLatentFluxSurfaceWater(upIndex) / WATER_DENSITY * nodeListPtr[i].up.area;

                        evapFromSoil *= (1. - surfaceWaterFraction);
                        evapFromSurface *= surfaceWaterFraction;

                        evapFromSurface = MAXVALUE(evapFromSurface, -waterVolume / deltaT);

                        if (nodeListPtr[upIndex].boundary != nullptr)
                            nodeListPtr[upIndex].boundary->waterFlow = evapFromSurface;
                        else
                            nodeListPtr[upIndex].Qw += evapFromSurface;

                    }

                    if (evapFromSoil < 0.)
                        evapFromSoil = MAXVALUE(evapFromSoil, -(theta_from_Se(i) - nodeListPtr[i].Soil->Theta_r) * nodeListPtr[i].volume_area / deltaT);
                    else
                        evapFromSoil = MINVALUE(evapFromSoil, (nodeListPtr[i].Soil->Theta_s - nodeListPtr[i].Soil->Theta_r) * nodeListPtr[i].volume_area / deltaT);

                    nodeListPtr[i].boundary->waterFlow = evapFromSoil;
                }
            }            

            nodeListPtr[i].Qw += nodeListPtr[i].boundary->waterFlow;
        }
    }

	// Culvert
	if (myCulvert.index != NOLINK)
	{
		long i = myCulvert.index;
		double waterLevel = 0.5 * (nodeListPtr[i].H + nodeListPtr[i].oldH) - nodeListPtr[i].z;		// [m]

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
            double hydraulicRadius = nodeListPtr[i].boundary->boundaryArea / wettedPerimeter;	// [m]

            // maximum Manning flow [m3 s-1]
            double ManningFlow = (nodeListPtr[i].boundary->boundaryArea / myCulvert.roughness)
                                * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2. / 3.);

			// pressure flow - Hazen-Williams equation - roughness = 70
			double equivalentDiameter = sqrt((4. * myCulvert.width * myCulvert.height) / PI);
            double pressureFlow = (70.0 * pow(myCulvert.slope, 0.54) * pow(equivalentDiameter, 2.63)) / 3.591;

			double weight = (waterLevel - myCulvert.height) / (myCulvert.height * 0.5);
			flow = weight * pressureFlow + (1. - weight) * ManningFlow;

		}
        else if (waterLevel > nodeListPtr[i].pond)
		{
			// open channel flow
            double boundaryArea = myCulvert.width * waterLevel;					// [m^2]
            double wettedPerimeter = myCulvert.width + 2.0 * waterLevel;        // [m]
            double hydraulicRadius = boundaryArea / wettedPerimeter;			// [m]

			// Manning equation [m^3 s^-1] 
            flow = (boundaryArea / myCulvert.roughness) * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2./3.);
		}

		// set boundary
		nodeListPtr[i].boundary->waterFlow = -flow;
		nodeListPtr[i].Qw += nodeListPtr[i].boundary->waterFlow;
	}
}


void updateBoundaryHeat()
{
    double myWaterFlux, advTemperature, heatFlux;

    for (long i = 1; i < myStructure.nrNodes; i++)
    {
        if (isHeatNode(i))
        {
            nodeListPtr[i].extra->Heat->Qh = nodeListPtr[i].extra->Heat->sinkSource;

            if (nodeListPtr[i].boundary != nullptr)
            {
                if (nodeListPtr[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                {
                    nodeListPtr[i].boundary->Heat->advectiveHeatFlux = 0.;
                    nodeListPtr[i].boundary->Heat->sensibleFlux = 0.;
                    nodeListPtr[i].boundary->Heat->latentFlux = 0.;
                    nodeListPtr[i].boundary->Heat->radiativeFlux = 0.;

                    if (nodeListPtr[i].boundary->Heat->netIrradiance != NODATA)
                        nodeListPtr[i].boundary->Heat->radiativeFlux = nodeListPtr[i].boundary->Heat->netIrradiance;

                    nodeListPtr[i].boundary->Heat->sensibleFlux += computeAtmosphericSensibleFlux(i);

                    if (myStructure.computeWater && myStructure.computeHeatVapor)
                        nodeListPtr[i].boundary->Heat->latentFlux += computeAtmosphericLatentHeatFlux(i) / nodeListPtr[i].up.area;

                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        // advective heat from rain
                        myWaterFlux = nodeListPtr[i].up.linkedExtra->heatFlux->waterFlux;
                        if (myWaterFlux > 0.)
                        {
                            advTemperature = nodeListPtr[i].boundary->Heat->temperature;
                            heatFlux =  myWaterFlux * HEAT_CAPACITY_WATER * advTemperature / nodeListPtr[i].up.area;
                            nodeListPtr[i].boundary->Heat->advectiveHeatFlux += heatFlux;
                        }

                        // advective heat from evaporation/condensation
                        if (nodeListPtr[i].boundary->waterFlow < 0.)
                            advTemperature = nodeListPtr[i].extra->Heat->T;
                        else
                            advTemperature = nodeListPtr[i].boundary->Heat->temperature;

                        nodeListPtr[i].boundary->Heat->advectiveHeatFlux += nodeListPtr[i].boundary->waterFlow * WATER_DENSITY * HEAT_CAPACITY_WATER_VAPOR * advTemperature / nodeListPtr[i].up.area;

                    }

                    nodeListPtr[i].extra->Heat->Qh += nodeListPtr[i].up.area * (nodeListPtr[i].boundary->Heat->radiativeFlux +
                                                                      nodeListPtr[i].boundary->Heat->sensibleFlux +
                                                                      nodeListPtr[i].boundary->Heat->latentFlux +
                                                                      nodeListPtr[i].boundary->Heat->advectiveHeatFlux);
                }
                else if (nodeListPtr[i].boundary->type == BOUNDARY_FREEDRAINAGE ||
                         nodeListPtr[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
                {
                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        myWaterFlux = nodeListPtr[i].boundary->waterFlow;

                        if (myWaterFlux < 0)
                            advTemperature = nodeListPtr[i].extra->Heat->T;
                        else
                            advTemperature = nodeListPtr[i].boundary->Heat->fixedTemperature;

                        heatFlux =  myWaterFlux * HEAT_CAPACITY_WATER * advTemperature / nodeListPtr[i].up.area;
                        nodeListPtr[i].boundary->Heat->advectiveHeatFlux = heatFlux;

                        nodeListPtr[i].extra->Heat->Qh += nodeListPtr[i].up.area * nodeListPtr[i].boundary->Heat->advectiveHeatFlux;
                    }

                    if (nodeListPtr[i].boundary->Heat->fixedTemperature != NODATA)
                    {
                        double avgH = getHMean(i);
                        double boundaryHeatConductivity = SoilHeatConductivity(i, nodeListPtr[i].extra->Heat->T, avgH - nodeListPtr[i].z);
                        double deltaT = nodeListPtr[i].boundary->Heat->fixedTemperature - nodeListPtr[i].extra->Heat->T;
                        nodeListPtr[i].extra->Heat->Qh += boundaryHeatConductivity * deltaT / nodeListPtr[i].boundary->Heat->fixedTemperatureDepth * nodeListPtr[i].up.area;
                    }
                }
            }
        }
    }
}
