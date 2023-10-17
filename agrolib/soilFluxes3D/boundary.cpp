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
    if (myNode[i].boundary->Heat == nullptr || ! myNode[myNode[i].up.index].isSurface)
        return 0;

    double myPressure = pressureFromAltitude(double(myNode[i].z));

    double myDeltaT = myNode[i].boundary->Heat->temperature - myNode[i].extra->Heat->T;

    double myCvAir = airVolumetricSpecificHeat(myPressure, myNode[i].boundary->Heat->temperature);

    return (myCvAir * myDeltaT * myNode[i].boundary->Heat->aerodynamicConductance);
}

/*!
 * \brief boundary vapor flux (evaporation/condensation)
 * \param i
 * \return vapor flux (kg m-2 s-1)
 */
double computeAtmosphericLatentFlux(long i)
{
    if (myNode[i].boundary->Heat == nullptr || ! myNode[myNode[i].up.index].isSurface)
        return 0;

    double PressSat, ConcVapSat, BoundaryVapor;

    PressSat = saturationVaporPressure(myNode[i].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, myNode[i].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (myNode[i].boundary->Heat->relativeHumidity / 100.);

    // kg m-3
    double myDeltaVapor = BoundaryVapor - soilFluxes3D::getNodeVapor(i);

    // m s-1
    double myTotalConductance = 1./((1./myNode[i].boundary->Heat->aerodynamicConductance) + (1. / myNode[i].boundary->Heat->soilConductance));

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
    if (! myNode[i].isSurface) return 0.;
    if (&(myNode[i].down) == nullptr) return 0.;

    long downIndex = myNode[i].down.index;

    if (myNode[downIndex].boundary->Heat == nullptr || myNode[downIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return 0.;

    double PressSat, ConcVapSat, BoundaryVapor;

    // atmospheric vapor content (kg m-3)
    PressSat = saturationVaporPressure(myNode[downIndex].boundary->Heat->temperature - ZEROCELSIUS);
    ConcVapSat = vaporConcentrationFromPressure(PressSat, myNode[downIndex].boundary->Heat->temperature);
    BoundaryVapor = ConcVapSat * (myNode[downIndex].boundary->Heat->relativeHumidity / 100.);

    // surface water vapor content (kg m-3) (assuming water temperature is the same of atmosphere)
    double myDeltaVapor = BoundaryVapor - ConcVapSat;

    // kg m-2 s-1
    // using aerodynamic conductance of index below (boundary for heat)
    double myVaporFlow = myDeltaVapor * myNode[downIndex].boundary->Heat->aerodynamicConductance;

    return myVaporFlow;
}

/*!
 * \brief atmospheric latent heat flux (evaporation/condensation)
 * \param i
 * \return latent flux (W)
 */
double computeAtmosphericLatentHeatFlux(long i)
{
    if (myNode[i].boundary->Heat == nullptr || ! myNode[myNode[i].up.index].isSurface)
        return 0;

    double latentHeatFlow = 0.;

    // J kg-1
    double lambda = latentHeatVaporization(myNode[i].extra->Heat->T - ZEROCELSIUS);
    // waterFlow: vapor sink source (m3 s-1)
    latentHeatFlow = myNode[i].boundary->waterFlow * WATER_DENSITY * lambda;

    return latentHeatFlow;
}

double getSurfaceWaterFraction(int i)
{
    if (! myNode[i].isSurface)
        return 0.0;
    else
    {
        double h = MAXVALUE(myNode[i].H - double(myNode[i].z), 0);
        return 1.0 - MAXVALUE(0.0, myNode[i].Soil->Pond - h) / myNode[i].Soil->Pond;
    }
}

void updateBoundary()
{
    for (long i = 0; i < myStructure.nrNodes; i++)
        if (myNode[i].boundary != nullptr)
            if (myStructure.computeHeat)
                if (myNode[i].extra->Heat != nullptr)
                    if (myNode[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                    {
                        // update aerodynamic conductance
                        myNode[i].boundary->Heat->aerodynamicConductance =
                                aerodynamicConductance(myNode[i].boundary->Heat->heightTemperature,
                                    myNode[i].boundary->Heat->heightWind,
                                    myNode[i].extra->Heat->T,
                                    myNode[i].boundary->Heat->roughnessHeight,
                                    myNode[i].boundary->Heat->temperature,
                                    myNode[i].boundary->Heat->windSpeed);

                        if (myStructure.computeWater)
                            // update soil surface conductance
                        {
                            double theta = theta_from_sign_Psi(myNode[i].H - myNode[i].z, i);
                            myNode[i].boundary->Heat->soilConductance = 1./ computeSoilSurfaceResistance(theta);
                        }
                    }
}


void updateBoundaryWater (double deltaT)
{
    double const EPSILON_METER = 0.0001;          // [m] 0.1 mm

    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        // water sink-source
        myNode[i].Qw = myNode[i].waterSinkSource;

        if (myNode[i].boundary != nullptr)
        {
            myNode[i].boundary->waterFlow = 0.;

            if (myNode[i].boundary->type == BOUNDARY_RUNOFF)
            {
                double avgH = (myNode[i].H + myNode[i].oldH) * 0.5;        // [m]
                // Surface water available for runoff [m]
                double hs = MAXVALUE(avgH - (myNode[i].z + myNode[i].Soil->Pond), 0.0);
                if (hs > EPSILON_METER)
                {
                    double maxFlow = (hs * myNode[i].volume_area) / deltaT;         // [m3 s-1] maximum flow available during the time step
                    // Manning equation
                    double v = (1. / myNode[i].Soil->Roughness) * pow(hs, 2./3.) * sqrt(myNode[i].boundary->slope);
                    // on the surface boundaryArea is a side [m]
                    double flow = myNode[i].boundary->boundaryArea * hs * v;        // [m3 s-1]
                    myNode[i].boundary->waterFlow = -MINVALUE(flow, maxFlow);
                }
            }
            else if (myNode[i].boundary->type == BOUNDARY_FREEDRAINAGE)
            {
                // Darcy unit gradient
                // dH=dz=L  -> dH/L=1
                myNode[i].boundary->waterFlow = -myNode[i].k * myNode[i].up.area;
            }

            else if (myNode[i].boundary->type == BOUNDARY_FREELATERALDRAINAGE)
            {
                // Darcy gradient = slope
                // dH=dz slope=dz/L -> dH/L=slope
                myNode[i].boundary->waterFlow = -myNode[i].k * myParameters.k_lateral_vertical_ratio
                                            * myNode[i].boundary->boundaryArea * myNode[i].boundary->slope;
            }

            else if (myNode[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
            {
                // water table
                double L = 1.0;                         // [m]
                double boundaryZ = myNode[i].z - L;     // [m]
                double boundaryK;                       // [m s-1]

                if (myNode[i].boundary->prescribedTotalPotential >= boundaryZ)
                {
                    // saturated
                    boundaryK = myNode[i].Soil->K_sat;
                }
                else
                {
                    // unsaturated
                    double boundaryPsi = fabs(myNode[i].boundary->prescribedTotalPotential - boundaryZ);
                    double boundarySe = computeSefromPsi_unsat(boundaryPsi, myNode[i].Soil);
                    boundaryK = computeWaterConductivity(boundarySe, myNode[i].Soil);
                }

                double meanK = computeMean(myNode[i].k, boundaryK);
                double dH = myNode[i].boundary->prescribedTotalPotential - myNode[i].H;
                myNode[i].boundary->waterFlow = meanK * myNode[i].boundary->boundaryArea * (dH / L);
            }

            else if (myNode[i].boundary->type == BOUNDARY_HEAT_SURFACE)
            {
                if (myStructure.computeHeat && myStructure.computeHeatVapor)
                {
                    long upIndex;

                    double surfaceWaterFraction = 0.;
                    if (&(myNode[i].up) != nullptr)
                    {
                        upIndex = myNode[i].up.index;
                        surfaceWaterFraction = getSurfaceWaterFraction(upIndex);
                    }

                    double evapFromSoil = computeAtmosphericLatentFlux(i) / WATER_DENSITY * myNode[i].up.area;

                    // surface water
                    if (surfaceWaterFraction > 0.)
                    {
                        double waterVolume = (myNode[upIndex].H - myNode[upIndex].z) * myNode[upIndex].volume_area;
                        double evapFromSurface = computeAtmosphericLatentFluxSurfaceWater(upIndex) / WATER_DENSITY * myNode[i].up.area;

                        evapFromSoil *= (1. - surfaceWaterFraction);
                        evapFromSurface *= surfaceWaterFraction;

                        evapFromSurface = MAXVALUE(evapFromSurface, -waterVolume / deltaT);

                        if (myNode[upIndex].boundary != nullptr)
                            myNode[upIndex].boundary->waterFlow = evapFromSurface;
                        else
                            myNode[upIndex].Qw += evapFromSurface;

                    }

                    if (evapFromSoil < 0.)
                        evapFromSoil = MAXVALUE(evapFromSoil, -(theta_from_Se(i) - myNode[i].Soil->Theta_r) * myNode[i].volume_area / deltaT);
                    else
                        evapFromSoil = MINVALUE(evapFromSoil, (myNode[i].Soil->Theta_s - myNode[i].Soil->Theta_r) * myNode[i].volume_area / deltaT);

                    myNode[i].boundary->waterFlow = evapFromSoil;
                }
            }            

            myNode[i].Qw += myNode[i].boundary->waterFlow;
        }
    }

	// Culvert
	if (myCulvert.index != NOLINK)
	{
		long i = myCulvert.index;
		double waterLevel = 0.5 * (myNode[i].H + myNode[i].oldH) - myNode[i].z;		// [m]

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
            double hydraulicRadius = myNode[i].boundary->boundaryArea / wettedPerimeter;	// [m]

            // maximum Manning flow [m3 s-1]
            double ManningFlow = (myNode[i].boundary->boundaryArea / myCulvert.roughness)
                                * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2. / 3.);

			// pressure flow - Hazen-Williams equation - roughness = 70
			double equivalentDiameter = sqrt((4. * myCulvert.width * myCulvert.height) / PI);
            double pressureFlow = (70.0 * pow(myCulvert.slope, 0.54) * pow(equivalentDiameter, 2.63)) / 3.591;

			double weight = (waterLevel - myCulvert.height) / (myCulvert.height * 0.5);
			flow = weight * pressureFlow + (1. - weight) * ManningFlow;

		}
		else if (waterLevel > myNode[i].Soil->Pond)
		{
			// open channel flow
            double boundaryArea = myCulvert.width * waterLevel;					// [m^2]
            double wettedPerimeter = myCulvert.width + 2.0 * waterLevel;        // [m]
            double hydraulicRadius = boundaryArea / wettedPerimeter;			// [m]

			// Manning equation [m^3 s^-1] 
            flow = (boundaryArea / myCulvert.roughness) * sqrt(myCulvert.slope) * pow(hydraulicRadius, 2./3.);
		}

		// set boundary
		myNode[i].boundary->waterFlow = -flow;
		myNode[i].Qw += myNode[i].boundary->waterFlow;
	}
}


void updateBoundaryHeat()
{
    double myWaterFlux, advTemperature, heatFlux;

    for (long i = 1; i < myStructure.nrNodes; i++)
    {
        if (isHeatNode(i))
        {
            myNode[i].extra->Heat->Qh = myNode[i].extra->Heat->sinkSource;

            if (myNode[i].boundary != nullptr)
            {
                if (myNode[i].boundary->type == BOUNDARY_HEAT_SURFACE)
                {
                    myNode[i].boundary->Heat->advectiveHeatFlux = 0.;
                    myNode[i].boundary->Heat->sensibleFlux = 0.;
                    myNode[i].boundary->Heat->latentFlux = 0.;
                    myNode[i].boundary->Heat->radiativeFlux = 0.;

                    if (myNode[i].boundary->Heat->netIrradiance != NODATA)
                        myNode[i].boundary->Heat->radiativeFlux = myNode[i].boundary->Heat->netIrradiance;

                    myNode[i].boundary->Heat->sensibleFlux += computeAtmosphericSensibleFlux(i);

                    if (myStructure.computeWater && myStructure.computeHeatVapor)
                        myNode[i].boundary->Heat->latentFlux += computeAtmosphericLatentHeatFlux(i) / myNode[i].up.area;

                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        // advective heat from rain
                        myWaterFlux = myNode[i].up.linkedExtra->heatFlux->waterFlux;
                        if (myWaterFlux > 0.)
                        {
                            advTemperature = myNode[i].boundary->Heat->temperature;
                            heatFlux =  myWaterFlux * HEAT_CAPACITY_WATER * advTemperature / myNode[i].up.area;
                            myNode[i].boundary->Heat->advectiveHeatFlux += heatFlux;
                        }

                        // advective heat from evaporation/condensation
                        if (myNode[i].boundary->waterFlow < 0.)
                            advTemperature = myNode[i].extra->Heat->T;
                        else
                            advTemperature = myNode[i].boundary->Heat->temperature;

                        myNode[i].boundary->Heat->advectiveHeatFlux += myNode[i].boundary->waterFlow * WATER_DENSITY * HEAT_CAPACITY_WATER_VAPOR * advTemperature / myNode[i].up.area;

                    }

                    myNode[i].extra->Heat->Qh += myNode[i].up.area * (myNode[i].boundary->Heat->radiativeFlux +
                                                                      myNode[i].boundary->Heat->sensibleFlux +
                                                                      myNode[i].boundary->Heat->latentFlux +
                                                                      myNode[i].boundary->Heat->advectiveHeatFlux);
                }
                else if (myNode[i].boundary->type == BOUNDARY_FREEDRAINAGE ||
                         myNode[i].boundary->type == BOUNDARY_PRESCRIBEDTOTALPOTENTIAL)
                {
                    if (myStructure.computeWater && myStructure.computeHeatAdvection)
                    {
                        myWaterFlux = myNode[i].boundary->waterFlow;

                        if (myWaterFlux < 0)
                            advTemperature = myNode[i].extra->Heat->T;
                        else
                            advTemperature = myNode[i].boundary->Heat->fixedTemperature;

                        heatFlux =  myWaterFlux * HEAT_CAPACITY_WATER * advTemperature / myNode[i].up.area;
                        myNode[i].boundary->Heat->advectiveHeatFlux = heatFlux;

                        myNode[i].extra->Heat->Qh += myNode[i].up.area * myNode[i].boundary->Heat->advectiveHeatFlux;
                    }

                    if (myNode[i].boundary->Heat->fixedTemperature != NODATA)
                    {
                        double avgH = getHMean(i);
                        double boundaryHeatConductivity = SoilHeatConductivity(i, myNode[i].extra->Heat->T, avgH - myNode[i].z);
                        double deltaT = myNode[i].boundary->Heat->fixedTemperature - myNode[i].extra->Heat->T;
                        myNode[i].extra->Heat->Qh += boundaryHeatConductivity * deltaT / myNode[i].boundary->Heat->fixedTemperatureDepth * myNode[i].up.area;
                    }
                }
            }
        }
    }
}
