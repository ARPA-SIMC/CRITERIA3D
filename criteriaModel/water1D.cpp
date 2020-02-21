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
    myCase->layers[0].waterContent = 0.0;
    for (unsigned int i = 1; i < myCase->nrLayers; i++)
    {
        if (myCase->layers[i].depth <= myCase->depthPloughedSoil)
            myCase->layers[i].waterContent = soil::getWaterContentFromAW(myCase->initialAW[0], &(myCase->layers[i]));
        else
            myCase->layers[i].waterContent = soil::getWaterContentFromAW(myCase->initialAW[1], &(myCase->layers[i]));
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
bool computeInfiltration(CriteriaModel* myCase, float prec, float surfaceIrrigation)
{
    // TODO extend to geometric layers
    unsigned int reached;                   // [-] index of reached layers for surpuls water
    double avgPloughSatDegree = NODATA;     // [-] average degree of saturation ploughed soil
    double fluxLayer = NODATA;              // [mm]
    double residualFlux = NODATA;           // [mm]
    double localFlux = NODATA;              // [mm]
    double waterSurplus = NODATA;           // [mm]
    double waterDeficit = NODATA;           // [mm]
    double localWater = NODATA;             // [mm]
    double distrH2O = NODATA;               // [mm] la quantità di acqua (=imax dello strato sotto) che potrebbe saturare il profilo sotto lo strato in surplus

    // Assign precipitation (surface pond)
    myCase->layers[0].waterContent += double(prec + surfaceIrrigation);

    // Initialize fluxes
    for (unsigned int i = 0; i < myCase->nrLayers; i++)
    {
        myCase->layers[i].flux = 0.0;
        myCase->layers[i].maxInfiltration = 0.0;
    }

    // Average degree of saturation (ploughed soil)
    unsigned int i = 1;
    unsigned int nrPloughLayers = 0;
    avgPloughSatDegree = 0;
    while (i < myCase->nrLayers && myCase->layers[i].depth < myCase->depthPloughedSoil)
    {
        nrPloughLayers++;
        avgPloughSatDegree += (myCase->layers[i].waterContent / myCase->layers[i].SAT);
        i++;
    }
    avgPloughSatDegree /= nrPloughLayers;

    // Maximum infiltration - due to gravitational force and permeability (Driessen 1986, eq.34)
    for (i = 1; i < myCase->nrLayers; i++)
    {
        myCase->layers[i].maxInfiltration = 10 * myCase->layers[i].horizon->Driessen.gravConductivity;

        if (myCase->layers[i].depth < myCase->depthPloughedSoil)
        {
            myCase->layers[i].maxInfiltration += 10 * (1 - avgPloughSatDegree) * myCase->layers[i].horizon->Driessen.maxSorptivity;
        }
    }

    myCase->layers[0].maxInfiltration = MINVALUE(myCase->layers[0].waterContent, myCase->layers[1].maxInfiltration);
    myCase->layers[0].flux = 0;

    for (int layerIndex = signed(myCase->nrLayers)-1; layerIndex >= 0; layerIndex--)
    {
        unsigned int l = unsigned(layerIndex);

        // find layers in water surplus
        if (myCase->layers[l].waterContent > myCase->layers[l].critical)
        {
            fluxLayer = MINVALUE(myCase->layers[l].maxInfiltration, myCase->layers[l].waterContent - myCase->layers[l].critical);
            myCase->layers[l].flux += fluxLayer;
            myCase->layers[l].waterContent -= fluxLayer;

            // TODO translate comment
            // cerca il punto di arrivo del fronte
            // saturando virtualmente il profilo sottostante con la quantità Imax
            // tiene conto degli Imax  e dei flussi già passati dagli strati sottostanti prendendo il minimo
            // ogni passo toglie la parte che va a saturare lo strato
            if (l == (myCase->nrLayers-1))
                reached = l;
            else
            {
                distrH2O = myCase->layers[l+1].maxInfiltration;
                i = l+1;
                while ((i < myCase->nrLayers-1) && (distrH2O > 0.0))
                {
                    distrH2O -= (myCase->layers[i].SAT - myCase->layers[i].waterContent);
                    distrH2O = MINVALUE(distrH2O, myCase->layers[i].maxInfiltration);
                    if (distrH2O > 0.0) i++;
                }
                reached = i;
            }

            if ((l == reached) && (l < (myCase->nrLayers -1)))
                    reached += 1;

            while ((reached < (myCase->nrLayers -1))
                   && (myCase->layers[reached].waterContent >= myCase->layers[reached].SAT))
                    reached += 1;

            // move water and compute fluxes
            for (i = l+1; i <= reached; i++)
            {
                // TODO translate comment
                // define fluxLayer in base allo stato idrico dello strato sottostante
                // sotto Field Capacity tolgo il deficit al fluxLayer,
                // in water surplus, aggiungo il surplus al fluxLayer
                if (myCase->layers[i].waterContent > myCase->layers[i].critical)
                {
                    // layers in water surplus (critical point: usually is FC)
                    waterSurplus = myCase->layers[i].waterContent - myCase->layers[i].critical;
                    fluxLayer += waterSurplus;
                    myCase->layers[i].waterContent -= waterSurplus;
                }
                else
                {
                    // layers before critical point
                    waterDeficit = myCase->layers[i].critical - myCase->layers[i].waterContent;
                    localWater = MINVALUE(fluxLayer, waterDeficit);
                    fluxLayer -= localWater;
                    myCase->layers[i].waterContent += localWater;
                    if (fluxLayer <= 0.0) break;
                }

                residualFlux = myCase->layers[i].maxInfiltration - myCase->layers[i].flux;
                residualFlux = MAXVALUE(residualFlux, 0.0);

                if (residualFlux >= fluxLayer)
                {
                    myCase->layers[i].flux += fluxLayer;
                }
                else
                {
                    // local surplus (localflux)
                    localFlux = fluxLayer - residualFlux;
                    fluxLayer = residualFlux;
                    myCase->layers[i].flux += fluxLayer;

                    // surplus management
                    if (localFlux <= (myCase->layers[i].SAT - myCase->layers[i].waterContent))
                    {
                        // available space for water in the layers
                        myCase->layers[i].waterContent += localFlux;
                        localFlux = 0;
                    }
                    else
                    {
                        // not enough space for water, upper layers are involved
                        unsigned int j;
                        for (j = i; j >= l + 1; j--)
                        {
                            if (localFlux <= 0.0) break;

                            localWater = MINVALUE(localFlux, myCase->layers[j].SAT - myCase->layers[j].waterContent);
                            if (j < i)
                                myCase->layers[j].flux -= localFlux;

                            localFlux -= localWater;
                            myCase->layers[j].waterContent += localWater;
                        }

                        // residual water
                        if ((localFlux > 0.0) && (j == l))
                        {
                            myCase->layers[l].waterContent += localFlux;
                            myCase->layers[l].flux -= localFlux;
                        }
                    }
                }
            } // end cycle l+1-->reached layers

            // drainage
            if ((reached == myCase->nrLayers-1) && (fluxLayer > 0))
            {
                myCase->output.dailyDrainage += fluxLayer;
                fluxLayer = 0;
            }

            // surplus distribution (saturated layers)
            i = reached;
            while ((fluxLayer > 0) && (i >= l+1))
            {
                localWater = MINVALUE(fluxLayer, myCase->layers[i].SAT - myCase->layers[i].waterContent);
                fluxLayer -= localWater;
                myCase->layers[i].flux -= localWater;
                myCase->layers[i].waterContent += localWater;
                i--;
            }

            // first layers (pond on surface)
            if ((fluxLayer != 0.) && (i == l))
            {
                myCase->layers[l].waterContent += fluxLayer;
                myCase->layers[l].flux -= fluxLayer;
            }

        }  // end if surplus layers
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
    double he_boundary;                  // [kPa] air entry point boundary layers
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
      || (waterTableDepth > (myCase->layers[lastLayer].depth + 8)))
    {
        //re-initialize threshold for vertical drainage
        for (unsigned int i = 1; i < myCase->nrLayers; i++)
            myCase->layers[i].critical = myCase->layers[i].FC;

        return false;
    }

    // search boundary layers: first layers over watertable
    // depth is assigned at center layers
    int first = signed(lastLayer);
    if (waterTableDepth < myCase->mySoil.totalDepth)
    {
        while ((first > 1) && (waterTableDepth <= myCase->layers[unsigned(first)].depth))
            first--;
    }

    unsigned int boundaryLayer = unsigned(first);

    // layers below watertable: saturated
    if (boundaryLayer < lastLayer)
    {
        for (unsigned int i = boundaryLayer+1; i <= lastLayer; i++)
        {
            myCase->layers[i].critical = myCase->layers[i].SAT;

            if (myCase->layers[i].waterContent < myCase->layers[i].SAT)
            {
                capillaryRiseSum += (myCase->layers[i].SAT - myCase->layers[i].waterContent);
                myCase->layers[i].waterContent = myCase->layers[i].SAT;
            }
        }
    }

    // air entry point of boundary layers
    he_boundary = myCase->layers[boundaryLayer].horizon->vanGenuchten.he;       // [kPa]

    // above watertable: assign water content threshold for vertical drainage
    for (unsigned int i = 1; i <= boundaryLayer; i++)
    {
        dz = (waterTableDepth - myCase->layers[i].depth);                       // [m]
        psi = soil::metersTokPa(dz) + he_boundary;                              // [kPa]

        myCase->layers[i].critical = soil::getWaterContentFromPsi(psi, &(myCase->layers[i]));

        if (myCase->layers[i].critical < myCase->layers[i].FC)
        {
            myCase->layers[i].critical = myCase->layers[i].FC;
        }
    }

    // above watertable: capillary rise
    previousPsi = soil::getWaterPotential(&(myCase->layers[boundaryLayer]));
    for (unsigned int i = boundaryLayer; i > 0; i--)
    {
        psi = soil::getWaterPotential(&(myCase->layers[i]));                // [kPa]

        if (i < boundaryLayer && psi < previousPsi)
            break;

        dPsi = soil::kPaToMeters(psi - he_boundary);                        // [m]
        dz = waterTableDepth - myCase->layers[i].depth;                     // [m]

        if (dPsi > dz)
        {
            k_psi = soil::getWaterConductivity(&(myCase->layers[i]));       // [cm day-1]

            k_psi *= REDUCTION_FACTOR * 10.;                                // [mm day-1]

            capillaryRise = k_psi * ((dPsi / dz) - 1);                      // [mm day-1]

            maxCapillaryRise = myCase->layers[i].critical - myCase->layers[i].waterContent;
            capillaryRise = MINVALUE(capillaryRise, maxCapillaryRise);

            // update water contet
            myCase->layers[i].waterContent += capillaryRise;
            capillaryRiseSum += capillaryRise;

            previousPsi = soil::getWaterPotential(&(myCase->layers[i]));     // [kPa]
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

    if (myCase->layers[0].waterContent > roughness)
    {
       myCase->output.dailySurfaceRunoff = myCase->layers[0].waterContent - roughness;
       myCase->layers[0].waterContent = roughness;
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
        if (myCase->layers[i].depth > drainDepth)
            break;

        waterSurplus = myCase->layers[i].waterContent - myCase->layers[i].critical;               // [mm]
        if (waterSurplus > 0.0)
        {
            satFactor = waterSurplus / (myCase->layers[i].SAT - myCase->layers[i].critical);      // [-]

            hydrHead = satFactor * (drainDepth - myCase->layers[i].depth);                       // [m]

            maxDrainage =  10 * myCase->layers[i].horizon->Driessen.k0 * hydrHead /
                    (hydrHead + (fieldWidth / PI) * log(fieldWidth / (PI * drainRadius)));      // [mm]

            layerDrainage = MINVALUE(waterSurplus, maxDrainage);                                // [mm]

            myCase->layers[i].waterContent -= layerDrainage;
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
        lowerDepth = myCase->layers[i].depth + myCase->layers[i].thickness * 0.5;
        if (lowerDepth < maxDepth)
            waterContent += myCase->layers[i].waterContent;
        else
        {
            upperDepth = myCase->layers[i].depth - myCase->layers[i].thickness * 0.5;
            depthRatio = (maxDepth - upperDepth) / myCase->layers[i].thickness;
            waterContent += myCase->layers[i].waterContent * depthRatio;
            break;
        }
    }

    return waterContent;
}

