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
#include <algorithm>

#include "commonConstants.h"
#include "water1D.h"
#include "soil.h"
#include "basicMath.h"
#include "crop.h"
#include "soilFluxes3D.h"


/*!
 * \brief Initialize soil water content
 * \param soilLayers
 */
void initializeWater(std::vector<soil::Crit1DLayer> &soilLayers)
{
    // TODO water content as function of month
    double initialAW = 0.8;             /*!<  [-] fraction of available Water  */

    soilLayers[0].waterContent = 0.0;
    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        soilLayers[i].waterContent = soil::getWaterContentFromAW(initialAW, soilLayers[i]);
    }
}


/*!
 * \brief Water infiltration and redistribution 1D
 * \param soilLayers
 * \param inputWater        [mm] precipitation + irrigation
 * \param ploughedSoilDepth [m]  depth of the ploughed soil
 * \return drainage         [mm] deep drainage
 * \author Margot van Soetendael
 * \note P.M.Driessen, 1986, "The water balance of soil"
 */
double computeInfiltration(std::vector<soil::Crit1DLayer> &soilLayers, double inputWater, double ploughedSoilDepth)
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
    double drainage = 0;                    // [mm]

    unsigned nrLayers = unsigned(soilLayers.size());

    // Assign precipitation (surface pond)
    soilLayers[0].waterContent += inputWater;

    // Initialize fluxes
    for (unsigned int i = 0; i < nrLayers; i++)
    {
        soilLayers[i].flux = 0.0;
        soilLayers[i].maxInfiltration = 0.0;
    }

    // Average degree of saturation (ploughed soil)
    unsigned int i = 1;
    unsigned int nrPloughLayers = 0;
    avgPloughSatDegree = 0;
    while (i < nrLayers && soilLayers[i].depth < ploughedSoilDepth)
    {
        nrPloughLayers++;
        avgPloughSatDegree += (soilLayers[i].waterContent / soilLayers[i].SAT);
        i++;
    }
    avgPloughSatDegree /= nrPloughLayers;

    // Maximum infiltration - due to gravitational force and permeability (Driessen 1986, eq.34)
    for (i = 1; i < nrLayers; i++)
    {
        soilLayers[i].maxInfiltration = 10 * soilLayers[i].horizonPtr->Driessen.gravConductivity;

        if (soilLayers[i].depth < ploughedSoilDepth)
        {
            soilLayers[i].maxInfiltration += 10 * (1 - avgPloughSatDegree) * soilLayers[i].horizonPtr->Driessen.maxSorptivity;
        }
    }

    soilLayers[0].maxInfiltration = MINVALUE(soilLayers[0].waterContent, soilLayers[1].maxInfiltration);
    soilLayers[0].flux = 0;

    for (int layerIndex = signed(nrLayers)-1; layerIndex >= 0; layerIndex--)
    {
        unsigned int l = unsigned(layerIndex);

        // find soilLayers in water surplus
        if (soilLayers[l].waterContent > soilLayers[l].critical)
        {
            fluxLayer = MINVALUE(soilLayers[l].maxInfiltration, soilLayers[l].waterContent - soilLayers[l].critical);
            soilLayers[l].flux += fluxLayer;
            soilLayers[l].waterContent -= fluxLayer;

            // TODO translate comment
            // cerca il punto di arrivo del fronte
            // saturando virtualmente il profilo sottostante con la quantità Imax
            // tiene conto degli Imax  e dei flussi già passati dagli strati sottostanti prendendo il minimo
            // ogni passo toglie la parte che va a saturare lo strato
            if (l == (nrLayers-1))
                reached = l;
            else
            {
                distrH2O = soilLayers[l+1].maxInfiltration;
                i = l+1;
                while ((i < nrLayers-1) && (distrH2O > 0.0))
                {
                    distrH2O -= (soilLayers[i].SAT - soilLayers[i].waterContent);
                    distrH2O = MINVALUE(distrH2O, soilLayers[i].maxInfiltration);
                    if (distrH2O > 0.0) i++;
                }
                reached = i;
            }

            if ((l == reached) && (l < (nrLayers -1)))
                    reached += 1;

            while ((reached < (nrLayers -1))
                   && (soilLayers[reached].waterContent >= soilLayers[reached].SAT))
                    reached += 1;

            // move water and compute fluxes
            for (i = l+1; i <= reached; i++)
            {
                // TODO translate comment
                // define fluxLayer in base allo stato idrico dello strato sottostante
                // sotto Field Capacity tolgo il deficit al fluxLayer,
                // in water surplus, aggiungo il surplus al fluxLayer
                if (soilLayers[i].waterContent > soilLayers[i].critical)
                {
                    // layers above critical point (it depends on watertable depth: default is FC)
                    waterSurplus = soilLayers[i].waterContent - soilLayers[i].critical;
                    fluxLayer += waterSurplus;
                    soilLayers[i].waterContent -= waterSurplus;
                }
                else
                {
                    // layers before critical point
                    waterDeficit = soilLayers[i].critical - soilLayers[i].waterContent;
                    localWater = MINVALUE(fluxLayer, waterDeficit);
                    fluxLayer -= localWater;
                    soilLayers[i].waterContent += localWater;
                    if (fluxLayer <= 0.0) break;
                }

                residualFlux = soilLayers[i].maxInfiltration - soilLayers[i].flux;
                residualFlux = MAXVALUE(residualFlux, 0.0);

                if (residualFlux >= fluxLayer)
                {
                    soilLayers[i].flux += fluxLayer;
                }
                else
                {
                    // local surplus (localflux)
                    localFlux = fluxLayer - residualFlux;
                    fluxLayer = residualFlux;
                    soilLayers[i].flux += fluxLayer;

                    // surplus management
                    if (localFlux <= (soilLayers[i].SAT - soilLayers[i].waterContent))
                    {
                        // there is available space for water in the layer
                        soilLayers[i].waterContent += localFlux;
                        localFlux = 0;
                    }
                    else
                    {
                        // not enough space for water, upper layers are involved
                        unsigned int j;
                        for (j = i; j >= l + 1; j--)
                        {
                            if (localFlux <= 0.0) break;

                            localWater = MINVALUE(localFlux, soilLayers[j].SAT - soilLayers[j].waterContent);
                            if (j < i)
                                soilLayers[j].flux -= localFlux;

                            localFlux -= localWater;
                            soilLayers[j].waterContent += localWater;
                        }

                        // residual water
                        if ((localFlux > 0.0) && (j == l))
                        {
                            soilLayers[l].waterContent += localFlux;
                            soilLayers[l].flux -= localFlux;
                        }
                    }
                }
            }

            // drainage
            if ((reached == nrLayers-1) && (fluxLayer > 0))
            {
                drainage += fluxLayer;
                fluxLayer = 0;
            }

            // surplus distribution (saturated layers)
            i = reached;
            while ((fluxLayer > 0) && (i >= l+1))
            {
                localWater = MINVALUE(fluxLayer, soilLayers[i].SAT - soilLayers[i].waterContent);
                fluxLayer -= localWater;
                soilLayers[i].flux -= localWater;
                soilLayers[i].waterContent += localWater;
                i--;
            }

            // first layer (pond on surface)
            if ((fluxLayer != 0.) && (i == l))
            {
                soilLayers[l].waterContent += fluxLayer;
                soilLayers[l].flux -= fluxLayer;
            }

        }
    }

    // update water potential
    for (unsigned i=1; i < nrLayers; i++)
    {
        soilLayers[i].waterPotential = soilLayers[i].getWaterPotential();                      // [kPa]
    }

    return drainage;
}


/*!
 * \brief compute capillary rise due to watertable
 * \param soilLayers
 * \param waterTableDepth [m]
 * \return capillary rise
 */
double computeCapillaryRise(std::vector<soil::Crit1DLayer> &soilLayers, double waterTableDepth)
{
    double psi, previousPsi;             // [kPa] water potential
    double he_boundary;                  // [kPa] air entry point boundary soilLayers
    double k_psi;                        // [cm day-1] water conductivity
    double dz, dPsi;                     // [m]

    double capillaryRiseSum = 0;         // [mm day-1]

    const double REDUCTION_FACTOR = 0.5;

    unsigned nrLayers = unsigned(soilLayers.size());
    unsigned int lastLayer = nrLayers-1;
    if (nrLayers == 0) return 0;

    // No WaterTable, wrong data or watertable too depth (6 meters)
    if ( isEqual(waterTableDepth, NODATA) || waterTableDepth <= 0
            || waterTableDepth > (soilLayers[lastLayer].depth + 6) )
    {
        // re-initialize threshold for vertical drainage
        for (unsigned int i = 1; i < nrLayers; i++)
            soilLayers[i].critical = soilLayers[i].FC;

        return 0;
    }

    // search boundary layer: first soil layer over watertable
    // depth is assigned at center of layer
    unsigned int boundaryLayer = lastLayer;
    while ((boundaryLayer > 1) && (waterTableDepth <= soilLayers[boundaryLayer].depth))
            boundaryLayer--;

    // layer below watertable: saturated
    if (boundaryLayer < lastLayer)
    {
        for (unsigned int i = boundaryLayer+1; i <= lastLayer; i++)
        {
            soilLayers[i].critical = soilLayers[i].SAT;

            if (soilLayers[i].waterContent < soilLayers[i].SAT)
            {
                capillaryRiseSum += (soilLayers[i].SAT - soilLayers[i].waterContent);
                soilLayers[i].waterContent = soilLayers[i].SAT;
            }
        }
    }

    // air entry point of boundary layer
    he_boundary = soilLayers[boundaryLayer].horizonPtr->vanGenuchten.he;       // [kPa]

    // above watertable: assign water content threshold for vertical drainage
    for (unsigned int i = 1; i <= boundaryLayer; i++)
    {
        dz = (waterTableDepth - soilLayers[i].depth);                       // [m]
        psi = soil::metersTokPa(dz) + he_boundary;                          // [kPa]

        soilLayers[i].critical = soil::getWaterContentFromPsi(psi, soilLayers[i]);

        if (soilLayers[i].critical < soilLayers[i].FC)
        {
            soilLayers[i].critical = soilLayers[i].FC;
        }
    }

    // above watertable: capillary rise
    previousPsi = soilLayers[boundaryLayer].getWaterPotential();
    for (unsigned int i = boundaryLayer; i > 0; i--)
    {
        psi = soilLayers[i].getWaterPotential();                        // [kPa]

        if (i < boundaryLayer && psi < previousPsi)
            break;

        dPsi = soil::kPaToMeters(psi - he_boundary);                    // [m]
        dz = waterTableDepth - soilLayers[i].depth;                     // [m]

        if (dPsi > dz)
        {
            // [cm day-1]
            k_psi = soilLayers[i].getWaterConductivity();
            // [mm day-1]
            k_psi *= REDUCTION_FACTOR * 10.;
            // [mm day-1]
            double capillaryRise = k_psi * ((dPsi / dz) - 1);
            // [mm day-1]
            double maxCapillaryRise = soilLayers[i].critical - soilLayers[i].waterContent;

            capillaryRise = MINVALUE(capillaryRise, maxCapillaryRise);

            // update water contet
            soilLayers[i].waterContent += capillaryRise;
            capillaryRiseSum += capillaryRise;

            previousPsi = soilLayers[i].getWaterPotential();            // [kPa]
        }
        else
        {
            previousPsi = psi;
        }
    }

    return capillaryRiseSum;
}


double computeEvaporation(std::vector<soil::Crit1DLayer> &soilLayers, double maxEvaporation)
{
    // surface evaporation
    double surfaceEvaporation = MINVALUE(maxEvaporation, soilLayers[0].waterContent);
    soilLayers[0].waterContent -= surfaceEvaporation;

    double actualEvaporation = surfaceEvaporation;

    double residualEvaporation = maxEvaporation - surfaceEvaporation;
    if (residualEvaporation < EPSILON)
    {
        return actualEvaporation;
    }

    // soil evaporation
    int nrEvapLayers = int(floor(MAX_EVAPORATION_DEPTH / soilLayers[1].thickness)) +1;
    nrEvapLayers = std::min(nrEvapLayers, int(soilLayers.size()-1));

    std::vector<double> coeffEvap;
    coeffEvap.resize(nrEvapLayers);

    double minDepth = soilLayers[1].depth + soilLayers[1].thickness / 2;
    double sumCoeff = 0;
    for (int i=1; i <= nrEvapLayers; i++)
    {
        double layerDepth = soilLayers[i].depth + soilLayers[i].thickness / 2.0;

        double coeffDepth = MAXVALUE((layerDepth - minDepth) / (MAX_EVAPORATION_DEPTH - minDepth), 0);
        // evaporation coefficient: 1 at depthMin, ~0.1 at maximum depth for evaporation
        coeffEvap[i-1] = exp(-2 * coeffDepth);

        sumCoeff += coeffEvap[i-1];
    }

    // normalize
    std::vector<double> coeffThreshold;
    coeffThreshold.resize(nrEvapLayers);
    for (int i=0; i < nrEvapLayers; i++)
    {
        coeffThreshold[i] = (1.0 - coeffEvap[i]) * 0.5;
        coeffEvap[i] /= sumCoeff;
    }

    bool isWaterSupply = true;
    double sumEvap, evapLayerThreshold, evapLayer;
    int nrIteration = 0;
    while ((residualEvaporation > EPSILON) && (isWaterSupply == true) && nrIteration < 3)
    {
        sumEvap = 0.0;

        for (int i=1; i <= nrEvapLayers; i++)
        {
            evapLayer = residualEvaporation * coeffEvap[i-1];
            evapLayerThreshold = soilLayers[i].HH + (soilLayers[i].FC - soilLayers[i].HH) * coeffThreshold[i-1];

            if (soilLayers[i].waterContent > (evapLayerThreshold + evapLayer))
            {
                isWaterSupply = true;
            }
            else
            {
                if (soilLayers[i].waterContent > evapLayerThreshold)
                {
                    evapLayer = soilLayers[i].waterContent - evapLayerThreshold;
                }
                else
                {
                    evapLayer = 0.0;
                }

                isWaterSupply = false;
            }

            soilLayers[i].waterContent -= evapLayer;
            sumEvap += evapLayer;
        }

        residualEvaporation -= sumEvap;
        actualEvaporation  += sumEvap;
        nrIteration++;
    }

    return actualEvaporation;
}


/*!
 * \brief compute surface runoff [mm]
 */
double computeSurfaceRunoff(const Crit3DCrop &myCrop, std::vector<soil::Crit1DLayer> &soilLayers)
{
    double surfaceRunoff;           // [mm]
    double maxSurfaceWater;         // [mm]

    maxSurfaceWater = myCrop.getSurfaceWaterPonding();
    if (soilLayers[0].waterContent > maxSurfaceWater)
    {
       surfaceRunoff = soilLayers[0].waterContent - maxSurfaceWater;
       soilLayers[0].waterContent = maxSurfaceWater;
    }
    else
       surfaceRunoff = 0.0;

    return surfaceRunoff;
}


/*!
 * \brief Compute lateral drainage
 * \param soilLayers
 * \note P.M.Driessen, 1986, eq.58
 * \return lateralDrainage
 */
double computeLateralDrainage(std::vector<soil::Crit1DLayer> &soilLayers)
{
    double satFactor;                       // [-]
    double hydrHead;                        // [m]
    double waterSurplus;                    // [mm]
    double layerDrainage;                   // [mm]
    double maxDrainage;                     // [mm]

    const double drainRadius = 0.25;        // [m]
    const double drainDepth = 1.0;          // [m] depth of drain
    const double fieldWidth = 100.0;        // [m]

    double lateralDrainageSum = 0;          // [mm]

    unsigned nrLayers = unsigned(soilLayers.size());

    for (unsigned int i = 1; i < nrLayers; i++)
    {
        if (soilLayers[i].depth > drainDepth)
            break;

        waterSurplus = soilLayers[i].waterContent - soilLayers[i].critical;                 // [mm]
        if (waterSurplus > 0.0)
        {
            satFactor = waterSurplus / (soilLayers[i].SAT - soilLayers[i].critical);        // [-]

            hydrHead = satFactor * (drainDepth - soilLayers[i].depth);                      // [m]

            maxDrainage =  10 * soilLayers[i].horizonPtr->Driessen.k0 * hydrHead /
                    (hydrHead + (fieldWidth / PI) * log(fieldWidth / (PI * drainRadius)));      // [mm]

            layerDrainage = MINVALUE(waterSurplus, maxDrainage);                                // [mm]

            soilLayers[i].waterContent -= layerDrainage;
            lateralDrainageSum += layerDrainage;
        }
    }

    return lateralDrainageSum;
}


/*!
 * \name assignOptimalIrrigation
 * \brief assign subirrigation to restore field capacity in the root zone
 * \param soilLayers, lastRootLayer, irrigationMax
 * \return irrigation [mm]
 */
double assignOptimalIrrigation(std::vector<soil::Crit1DLayer> &soilLayers, unsigned int lastRootLayer, double irrigationMax)
{
    double residualIrrigation = irrigationMax;
    unsigned int nrLayers = unsigned(soilLayers.size());

    unsigned int i=0;
    while ((i < nrLayers) && (i <= lastRootLayer) && (residualIrrigation > 0))
    {
        if (soilLayers[i].waterContent < soilLayers[i].FC)
        {
            double deficit = soilLayers[i].FC - soilLayers[i].waterContent;
            deficit = MINVALUE(deficit, residualIrrigation);

            soilLayers[i].waterContent += deficit;
            residualIrrigation -= deficit;
        }
        i++;
    }

    return (irrigationMax - residualIrrigation);
}


/*!
 * \brief getSoilWaterContentSum
 * \param soilLayers
 * \param computationDepth = computation soil depth [cm]
 * \return sum of water content from zero to computationSoilDepth [mm]
 */
double getSoilWaterContentSum(const std::vector<soil::Crit1DLayer> &soilLayers, double computationDepth)
{
    computationDepth /= 100;                // [cm] --> [m]
    double lowerDepth, upperDepth;          // [m]
    double waterContentSum = 0;             // [mm]

    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        lowerDepth = soilLayers[i].depth + soilLayers[i].thickness * 0.5;

        if (lowerDepth < computationDepth)
        {
            waterContentSum += soilLayers[i].waterContent;
        }
        else
        {
            upperDepth = soilLayers[i].depth - soilLayers[i].thickness * 0.5;
            double depthFraction = (computationDepth - upperDepth) / soilLayers[i].thickness;
            return waterContentSum + soilLayers[i].waterContent * depthFraction;
        }
    }

    return waterContentSum;
}


/*!
 * \brief getReadilyAvailableWater
 * \param myCrop
 * \param soilLayers
 * \return sum of readily available water in the rooting zone [mm]
 */
double getReadilyAvailableWater(const Crit3DCrop &myCrop, const std::vector<soil::Crit1DLayer> &soilLayers)
{
    if (! myCrop.isLiving) return NODATA;
    if (myCrop.roots.rootDepth <= myCrop.roots.rootDepthMin) return NODATA;
    if (myCrop.roots.firstRootLayer == NODATA) return NODATA;

    double sumRAW = 0.0;
    for (unsigned int i = unsigned(myCrop.roots.firstRootLayer); i <= unsigned(myCrop.roots.lastRootLayer); i++)
    {
        double thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(myCrop.psiLeaf), *(soilLayers[i].horizonPtr));
        // [mm]
        double cropWP = thetaWP * soilLayers[i].thickness * soilLayers[i].soilFraction * 1000.;
        // [mm]
        double threshold = soilLayers[i].FC - myCrop.fRAW * (soilLayers[i].FC - cropWP);

        double layerRAW = (soilLayers[i].waterContent - threshold);

        double layerMaxDepth = soilLayers[i].depth + soilLayers[i].thickness / 2.0;
        if (myCrop.roots.rootDepth < layerMaxDepth)
        {
                layerRAW *= (myCrop.roots.rootDepth - layerMaxDepth) / soilLayers[i].thickness;
        }

        sumRAW += layerRAW;
    }

    return sumRAW;
}

