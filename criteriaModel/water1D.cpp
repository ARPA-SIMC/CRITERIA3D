/*!
    \copyright 2018 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <math.h>

#include "commonConstants.h"
#include "water1D.h"
#include "criteriaModel.h"
#include "soil.h"


/*!
 * \brief Initialize water content
 * assign two different initial available water
 * in the ploughed soil and in the deep soil
 */
void initializeWater(CriteriaModel* myCase)
{
    // TODO water content as function of month
    myCase->soilLayers[0].waterContent = 0.0;
    for (unsigned int i = 1; i < myCase->nrLayers; i++)
    {
        if (myCase->soilLayers[i].depth <= myCase->depthPloughedSoil)
            myCase->soilLayers[i].waterContent = soil::getWaterContentFromAW(myCase->initialAW[0], &(myCase->soilLayers[i]));
        else
            myCase->soilLayers[i].waterContent = soil::getWaterContentFromAW(myCase->initialAW[1], &(myCase->soilLayers[i]));
    }
}


/*!
 * \brief Water infiltration and redistribution 1D
 * \param myCase
 * \param myError
 * \param prec      [mm]
 * \param surfaceIrrigation [mm]
 * \author Margot van Soetendael
 * \note P.M.Driessen, 1986, "The water balance of soil"
 */
bool computeInfiltration(CriteriaModel* myCase, double prec, double surfaceIrrigation)
{
    // TODO extend to geometric soilLayers
    unsigned int reached;                   // [-] index of reached soilLayers for surpuls water
    double avgPloughSatDegree = NODATA;     // [-] average degree of saturation ploughed soil
    double fluxLayer = NODATA;              // [mm]
    double residualFlux = NODATA;           // [mm]
    double localFlux = NODATA;              // [mm]
    double waterSurplus = NODATA;           // [mm]
    double waterDeficit = NODATA;           // [mm]
    double localWater = NODATA;             // [mm]
    double distrH2O = NODATA;               // [mm] la quantità di acqua (=imax dello strato sotto) che potrebbe saturare il profilo sotto lo strato in surplus

    // Assign precipitation (surface pond)
    myCase->soilLayers[0].waterContent += (prec + surfaceIrrigation);

    // Initialize fluxes
    for (unsigned int i = 0; i < myCase->nrLayers; i++)
    {
        myCase->soilLayers[i].flux = 0.0;
        myCase->soilLayers[i].maxInfiltration = 0.0;
    }

    // Average degree of saturation (ploughed soil)
    unsigned int i = 1;
    unsigned int nrPloughLayers = 0;
    avgPloughSatDegree = 0;
    while (i < myCase->nrLayers && myCase->soilLayers[i].depth < myCase->depthPloughedSoil)
    {
        nrPloughLayers++;
        avgPloughSatDegree += (myCase->soilLayers[i].waterContent / myCase->soilLayers[i].SAT);
        i++;
    }
    avgPloughSatDegree /= nrPloughLayers;

    // Maximum infiltration - due to gravitational force and permeability (Driessen 1986, eq.34)
    for (i = 1; i < myCase->nrLayers; i++)
    {
        myCase->soilLayers[i].maxInfiltration = 10 * myCase->soilLayers[i].horizon->Driessen.gravConductivity;

        if (myCase->soilLayers[i].depth < myCase->depthPloughedSoil)
        {
            myCase->soilLayers[i].maxInfiltration += 10 * (1 - avgPloughSatDegree) * myCase->soilLayers[i].horizon->Driessen.maxSorptivity;
        }
    }

    myCase->soilLayers[0].maxInfiltration = MINVALUE(myCase->soilLayers[0].waterContent, myCase->soilLayers[1].maxInfiltration);
    myCase->soilLayers[0].flux = 0;

    for (int layerIndex = signed(myCase->nrLayers)-1; layerIndex >= 0; layerIndex--)
    {
        unsigned int l = unsigned(layerIndex);

        // find soilLayers in water surplus
        if (myCase->soilLayers[l].waterContent > myCase->soilLayers[l].critical)
        {
            fluxLayer = MINVALUE(myCase->soilLayers[l].maxInfiltration, myCase->soilLayers[l].waterContent - myCase->soilLayers[l].critical);
            myCase->soilLayers[l].flux += fluxLayer;
            myCase->soilLayers[l].waterContent -= fluxLayer;

            // TODO translate comment
            // cerca il punto di arrivo del fronte
            // saturando virtualmente il profilo sottostante con la quantità Imax
            // tiene conto degli Imax  e dei flussi già passati dagli strati sottostanti prendendo il minimo
            // ogni passo toglie la parte che va a saturare lo strato
            if (l == (myCase->nrLayers-1))
                reached = l;
            else
            {
                distrH2O = myCase->soilLayers[l+1].maxInfiltration;
                i = l+1;
                while ((i < myCase->nrLayers-1) && (distrH2O > 0.0))
                {
                    distrH2O -= (myCase->soilLayers[i].SAT - myCase->soilLayers[i].waterContent);
                    distrH2O = MINVALUE(distrH2O, myCase->soilLayers[i].maxInfiltration);
                    if (distrH2O > 0.0) i++;
                }
                reached = i;
            }

            if ((l == reached) && (l < (myCase->nrLayers -1)))
                    reached += 1;

            while ((reached < (myCase->nrLayers -1))
                   && (myCase->soilLayers[reached].waterContent >= myCase->soilLayers[reached].SAT))
                    reached += 1;

            // move water and compute fluxes
            for (i = l+1; i <= reached; i++)
            {
                // TODO translate comment
                // define fluxLayer in base allo stato idrico dello strato sottostante
                // sotto Field Capacity tolgo il deficit al fluxLayer,
                // in water surplus, aggiungo il surplus al fluxLayer
                if (myCase->soilLayers[i].waterContent > myCase->soilLayers[i].critical)
                {
                    // soilLayers in water surplus (critical point: usually is FC)
                    waterSurplus = myCase->soilLayers[i].waterContent - myCase->soilLayers[i].critical;
                    fluxLayer += waterSurplus;
                    myCase->soilLayers[i].waterContent -= waterSurplus;
                }
                else
                {
                    // soilLayers before critical point
                    waterDeficit = myCase->soilLayers[i].critical - myCase->soilLayers[i].waterContent;
                    localWater = MINVALUE(fluxLayer, waterDeficit);
                    fluxLayer -= localWater;
                    myCase->soilLayers[i].waterContent += localWater;
                    if (fluxLayer <= 0.0) break;
                }

                residualFlux = myCase->soilLayers[i].maxInfiltration - myCase->soilLayers[i].flux;
                residualFlux = MAXVALUE(residualFlux, 0.0);

                if (residualFlux >= fluxLayer)
                {
                    myCase->soilLayers[i].flux += fluxLayer;
                }
                else
                {
                    // local surplus (localflux)
                    localFlux = fluxLayer - residualFlux;
                    fluxLayer = residualFlux;
                    myCase->soilLayers[i].flux += fluxLayer;

                    // surplus management
                    if (localFlux <= (myCase->soilLayers[i].SAT - myCase->soilLayers[i].waterContent))
                    {
                        // available space for water in the soilLayers
                        myCase->soilLayers[i].waterContent += localFlux;
                        localFlux = 0;
                    }
                    else
                    {
                        // not enough space for water, upper soilLayers are involved
                        unsigned int j;
                        for (j = i; j >= l + 1; j--)
                        {
                            if (localFlux <= 0.0) break;

                            localWater = MINVALUE(localFlux, myCase->soilLayers[j].SAT - myCase->soilLayers[j].waterContent);
                            if (j < i)
                                myCase->soilLayers[j].flux -= localFlux;

                            localFlux -= localWater;
                            myCase->soilLayers[j].waterContent += localWater;
                        }

                        // residual water
                        if ((localFlux > 0.0) && (j == l))
                        {
                            myCase->soilLayers[l].waterContent += localFlux;
                            myCase->soilLayers[l].flux -= localFlux;
                        }
                    }
                }
            } // end cycle l+1-->reached soilLayers

            // drainage
            if ((reached == myCase->nrLayers-1) && (fluxLayer > 0))
            {
                myCase->output.dailyDrainage += fluxLayer;
                fluxLayer = 0;
            }

            // surplus distribution (saturated soilLayers)
            i = reached;
            while ((fluxLayer > 0) && (i >= l+1))
            {
                localWater = MINVALUE(fluxLayer, myCase->soilLayers[i].SAT - myCase->soilLayers[i].waterContent);
                fluxLayer -= localWater;
                myCase->soilLayers[i].flux -= localWater;
                myCase->soilLayers[i].waterContent += localWater;
                i--;
            }

            // first soilLayers (pond on surface)
            if ((fluxLayer != 0.) && (i == l))
            {
                myCase->soilLayers[l].waterContent += fluxLayer;
                myCase->soilLayers[l].flux -= fluxLayer;
            }

        }  // end if surplus soilLayers
    }

    return true;
}


/*!
 * \brief compute capillary rise due to watertable
 * \param myCase
 * \param waterTableDepth [m]
 * \return
 */
bool computeCapillaryRise(CriteriaModel* myCase, double waterTableDepth)
{
    double psi, previousPsi;             // [kPa] water potential
    double he_boundary;                  // [kPa] air entry point boundary soilLayers
    double k_psi;                        // [cm/d] water conductivity
    double dz, dPsi;                     // [m]
    double capillaryRise;                // [mm]
    double maxCapillaryRise;             // [mm]
    double capillaryRiseSum = 0;         // [mm]

    unsigned int lastLayer = myCase->nrLayers - 1;
    const double REDUCTION_FACTOR = 0.5;

    // NO WaterTable, wrong data or watertable too depth
    if ( (int(waterTableDepth) == int(NODATA))
      || (waterTableDepth <= 0)
      || (waterTableDepth > (myCase->soilLayers[lastLayer].depth + 8)))
    {
        //re-initialize threshold for vertical drainage
        for (unsigned int i = 1; i < myCase->nrLayers; i++)
            myCase->soilLayers[i].critical = myCase->soilLayers[i].FC;

        return false;
    }

    // search boundary soilLayers: first soilLayers over watertable
    // depth is assigned at center soilLayers
    unsigned int first = lastLayer;
    if (waterTableDepth < myCase->mySoil.totalDepth)
    {
        while ((first > 1) && (waterTableDepth <= myCase->soilLayers[first].depth))
            first--;
    }

    unsigned int boundaryLayer = first;

    // soilLayers below watertable: saturated
    if (boundaryLayer < lastLayer)
    {
        for (unsigned int i = boundaryLayer+1; i <= lastLayer; i++)
        {
            myCase->soilLayers[i].critical = myCase->soilLayers[i].SAT;

            if (myCase->soilLayers[i].waterContent < myCase->soilLayers[i].SAT)
            {
                capillaryRiseSum += (myCase->soilLayers[i].SAT - myCase->soilLayers[i].waterContent);
                myCase->soilLayers[i].waterContent = myCase->soilLayers[i].SAT;
            }
        }
    }

    // air entry point of boundary soilLayers
    he_boundary = myCase->soilLayers[boundaryLayer].horizon->vanGenuchten.he;       // [kPa]

    // above watertable: assign water content threshold for vertical drainage
    for (unsigned int i = 1; i <= boundaryLayer; i++)
    {
        dz = (waterTableDepth - myCase->soilLayers[i].depth);                       // [m]
        psi = soil::metersTokPa(dz) + he_boundary;                              // [kPa]

        myCase->soilLayers[i].critical = soil::getWaterContentFromPsi(psi, &(myCase->soilLayers[i]));

        if (myCase->soilLayers[i].critical < myCase->soilLayers[i].FC)
        {
            myCase->soilLayers[i].critical = myCase->soilLayers[i].FC;
        }
    }

    // above watertable: capillary rise
    previousPsi = soil::getWaterPotential(&(myCase->soilLayers[boundaryLayer]));
    for (unsigned int i = boundaryLayer; i > 0; i--)
    {
        psi = soil::getWaterPotential(&(myCase->soilLayers[i]));                // [kPa]

        if (i < boundaryLayer && psi < previousPsi)
            break;

        dPsi = soil::kPaToMeters(psi - he_boundary);                        // [m]
        dz = waterTableDepth - myCase->soilLayers[i].depth;                     // [m]

        if (dPsi > dz)
        {
            k_psi = soil::getWaterConductivity(&(myCase->soilLayers[i]));       // [cm day-1]

            k_psi *= REDUCTION_FACTOR * 10.;                                // [mm day-1]

            capillaryRise = k_psi * ((dPsi / dz) - 1);                      // [mm day-1]

            maxCapillaryRise = myCase->soilLayers[i].critical - myCase->soilLayers[i].waterContent;
            capillaryRise = MINVALUE(capillaryRise, maxCapillaryRise);

            // update water contet
            myCase->soilLayers[i].waterContent += capillaryRise;
            capillaryRiseSum += capillaryRise;

            previousPsi = soil::getWaterPotential(&(myCase->soilLayers[i]));     // [kPa]
        }
        else
        {
            previousPsi = psi;
        }
    }

    myCase->output.dailyCapillaryRise = capillaryRiseSum;
    return true;
}


/*!
 * \brief compute surface runoff [mm]
 * \param myCase
 * \return
 */
bool computeSurfaceRunoff(CriteriaModel* myCase)
{
    double clodHeight;           // [mm] effective height of clod
    double roughness;            // [mm]

    // TODO taking into account tillage and others operations
    if (myCase->myCrop.isPluriannual())
        clodHeight = 0.0;
    else
        clodHeight = 5.0;

    roughness = myCase->myCrop.maxSurfacePuddle + clodHeight;

    if (myCase->soilLayers[0].waterContent > roughness)
    {
       myCase->output.dailySurfaceRunoff = myCase->soilLayers[0].waterContent - roughness;
       myCase->soilLayers[0].waterContent = roughness;
    }
    else
       myCase->output.dailySurfaceRunoff = 0.0;

    return true;
}


/*!
 * \brief Compute lateral drainage
 * \param myCase
 * \note P.M.Driessen, 1986, eq.58
 * \return
 */
bool computeLateralDrainage(CriteriaModel* myCase)
{
    double satFactor;                       // [-]
    double hydrHead;                        // [m]
    double waterSurplus;                    // [mm]
    double layerDrainage;                   // [mm]
    double maxDrainage;                     // [mm]
    const double drainRadius = 0.25;        // [m]
    const double drainDepth = 1.0;          // [m]
    const double fieldWidth = 100.0;        // [m]

    for (unsigned int i = 1; i < myCase->nrLayers; i++)
    {
        if (myCase->soilLayers[i].depth > drainDepth)
            break;

        waterSurplus = myCase->soilLayers[i].waterContent - myCase->soilLayers[i].critical;               // [mm]
        if (waterSurplus > 0.0)
        {
            satFactor = waterSurplus / (myCase->soilLayers[i].SAT - myCase->soilLayers[i].critical);      // [-]

            hydrHead = satFactor * (drainDepth - myCase->soilLayers[i].depth);                       // [m]

            maxDrainage =  10 * myCase->soilLayers[i].horizon->Driessen.k0 * hydrHead /
                    (hydrHead + (fieldWidth / PI) * log(fieldWidth / (PI * drainRadius)));      // [mm]

            layerDrainage = MINVALUE(waterSurplus, maxDrainage);                                // [mm]

            myCase->soilLayers[i].waterContent -= layerDrainage;
            myCase->output.dailyLateralDrainage += layerDrainage;
        }
    }

    return true;
}


/*!
 * \brief getSoilWaterContent
 * \param myCase
 * \return sum of water content (mm) in the first meter of soil
 */
double getSoilWaterContent(CriteriaModel* myCase)
{
    const double maxDepth = 1.0;            // [m]
    double lowerDepth, upperDepth;          // [m]
    double depthRatio;                      // [-]
    double waterContent = 0.0;              // [mm]

    for (unsigned int i = 1; i < myCase->nrLayers; i++)
    {
        lowerDepth = myCase->soilLayers[i].depth + myCase->soilLayers[i].thickness * 0.5;
        if (lowerDepth < maxDepth)
            waterContent += myCase->soilLayers[i].waterContent;
        else
        {
            upperDepth = myCase->soilLayers[i].depth - myCase->soilLayers[i].thickness * 0.5;
            depthRatio = (maxDepth - upperDepth) / myCase->soilLayers[i].thickness;
            waterContent += myCase->soilLayers[i].waterContent * depthRatio;
            break;
        }
    }

    return waterContent;
}

