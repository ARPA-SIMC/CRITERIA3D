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
#include "crit3dDate.h"
#include "development.h"
#include "soil.h"
#include "criteriaModel.h"
#include "croppingSystem.h"
#include "root.h"



void initializeCrop(CriteriaModel* myCase, int currentDoy)
{
    myCase->myCrop.initialize(myCase->meteoPoint.latitude, myCase->nrLayers, myCase->mySoil.totalDepth, currentDoy);
}


bool updateCrop(CriteriaModel* myCase, Crit3DDate myDate, float tmin, float tmax, double waterTableDepth, QString *myError)
{
    std::string errorString;

    if ( !myCase->myCrop.dailyUpdate(myDate, myCase->meteoPoint.latitude, signed(myCase->nrLayers),
                                    myCase->mySoil.totalDepth, tmin, tmax, waterTableDepth, &errorString))
    {
        *myError = QString::fromStdString(errorString);
        return false;
    }

    if ( !root::computeRootDensity(&(myCase->myCrop), myCase->layers, signed(myCase->nrLayers), myCase->mySoil.totalDepth))
    {
        *myError = "Error in updating roots for crop " + QString::fromStdString(myCase->myCrop.idCrop);
        return false;
    }

    return true;
}


bool cropWaterDemand(CriteriaModel* myCase)
{
    double Kc;                  // crop coefficient
    double TC;                  // turbulence coefficient
    double ke = 0.6;            // light extinction factor
    const double maxEvapRatio = 0.66;

    if (myCase->myCrop.idCrop == "" || ! myCase->myCrop.isLiving || myCase->myCrop.LAI < EPSILON)
    {
        myCase->output.dailyMaxEvaporation = myCase->output.dailyEt0 * maxEvapRatio;
        myCase->output.dailyMaxTranspiration = 0.0;
        myCase->output.dailyKc = 0.0;
    }
    else
    {
        Kc = 1 - exp(-ke * myCase->myCrop.LAI);
        TC = 1 + (myCase->myCrop.kcMax - 1.0) * Kc;
        myCase->output.dailyKc = TC * Kc;
        myCase->output.dailyMaxEvaporation = myCase->output.dailyEt0 * maxEvapRatio * (1.0 - Kc);
        myCase->output.dailyMaxTranspiration = myCase->output.dailyEt0 * myCase->output.dailyKc;
    }

    return true;
}


/*!
 * \brief getCropReadilyAvailableWater
 * \return sum of readily available water (mm) in the rooting zone
 */
double getCropReadilyAvailableWater(CriteriaModel* myCase)
{
    if (! myCase->myCrop.isLiving) return 0.;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return 0.;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return 0.;

    double depth, threshold, layerRAW;

    double sumRAW = 0.0;
    for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
    {
        threshold = myCase->layers[i].FC - myCase->myCrop.fRAW * (myCase->layers[i].FC - myCase->layers[i].WP);

        layerRAW = (myCase->layers[i].waterContent - threshold);

        depth = myCase->layers[i].depth + myCase->layers[i].thickness / 2.0;

        if (myCase->myCrop.roots.rootDepth < depth)
                layerRAW *= (myCase->myCrop.roots.rootDepth - depth) / myCase->layers[i].thickness;

        sumRAW += layerRAW;
    }

    return sumRAW;
}


/*!
 * \brief getTotalReadilyAvailableWater
 * \return sum of readily available water (mm)
 * \note take into account at minimum the forst meter f soil and the surface water
 */
double getTotalReadilyAvailableWater(CriteriaModel* myCase)
{
    if (! myCase->myCrop.isLiving) return NODATA;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return NODATA;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return NODATA;

    double threshold;
    int lastLayer = 0;

    while (unsigned(lastLayer) < (myCase->nrLayers-1) && myCase->layers[unsigned(lastLayer)].depth < 1.0)
            lastLayer++;

    lastLayer = MAXVALUE(lastLayer, myCase->myCrop.roots.lastRootLayer);

    // surface water
    double RAW = myCase->layers[0].waterContent;

    for (unsigned int i = 1; i <= unsigned(lastLayer); i++)
    {
        if (signed(i) < myCase->myCrop.roots.firstRootLayer)
            threshold = myCase->layers[i].FC;
        else
            // rooting zone
            threshold = myCase->layers[i].FC - myCase->myCrop.fRAW * (myCase->layers[i].FC - myCase->layers[i].WP);

        if(myCase->layers[i].waterContent > threshold)
            RAW += (myCase->layers[i].waterContent - threshold);
    }

    return RAW;
}


float cropIrrigationDemand(CriteriaModel* myCase, int doy, float currentPrec, float nextPrec)
{
    // update days since last irrigation
    if (myCase->myCrop.daysSinceIrrigation != NODATA)
        myCase->myCrop.daysSinceIrrigation++;

    // check irrigated crop
    if (myCase->myCrop.idCrop == ""
        || ! myCase->myCrop.isLiving
        || int(myCase->myCrop.irrigationVolume) == int(NODATA)
        || int(myCase->myCrop.irrigationVolume) == 0)
        return 0;

    // check irrigation period
    if (myCase->myCrop.doyStartIrrigation != NODATA && myCase->myCrop.doyEndIrrigation != NODATA)
    {
        if (doy < myCase->myCrop.doyStartIrrigation ||
            doy > myCase->myCrop.doyEndIrrigation) return 0;
    }
    if (myCase->myCrop.degreeDaysStartIrrigation != NODATA && myCase->myCrop.degreeDaysEndIrrigation != NODATA)
    {
        if (myCase->myCrop.degreeDays < myCase->myCrop.degreeDaysStartIrrigation ||
            myCase->myCrop.degreeDays > myCase->myCrop.degreeDaysEndIrrigation) return 0;
    }

    // check forecast
    double waterNeeds = myCase->myCrop.irrigationVolume / myCase->myCrop.irrigationShift;
    double todayWater = double(currentPrec) + myCase->layers[0].waterContent;
    double twoDaysWater = todayWater + double(nextPrec);

    if (todayWater > waterNeeds) return 0;
    if (twoDaysWater > 2*waterNeeds) return 0;

    // check water stress (before infiltration)
    double threshold = 1. - myCase->myCrop.stressTolerance;
    double waterStress = cropTranspiration(myCase, true);
    if (waterStress <= threshold) return 0;

    // check irrigation shift
    if (myCase->myCrop.daysSinceIrrigation != NODATA)
    {
        if (myCase->myCrop.daysSinceIrrigation < myCase->myCrop.irrigationShift)
            return 0;
    }

    // all check passed --> IRRIGATION

    // reset irrigation shift
    myCase->myCrop.daysSinceIrrigation = 0;

    if (myCase->optimizeIrrigation)
    {
        return float(MINVALUE(getCropWaterDeficit(myCase), myCase->myCrop.irrigationVolume));
    }
    else
    {
        return float(myCase->myCrop.irrigationVolume);
        //return float(MAXVALUE(int(myCase->output.dailyMaxTranspiration), myCase->myCrop.irrigationVolume));
    }
}


bool optimalIrrigation(CriteriaModel* myCase, float myIrrigation)
{
    float myDeficit;
    float residualIrrigation = myIrrigation;

    unsigned int i=0;
    while (i < myCase->nrLayers && residualIrrigation > 0)
    {
        if (myCase->layers[i].waterContent < myCase->layers[i].FC)
        {
            myDeficit = float(myCase->layers[i].FC - myCase->layers[i].waterContent);
            myDeficit = MINVALUE(myDeficit, residualIrrigation);

            myCase->layers[i].waterContent += double(myDeficit);
            residualIrrigation -= myDeficit;
        }
        i++;
    }

    myCase->output.dailyIrrigation = double(myIrrigation - residualIrrigation);
    return true;
}


bool evaporation(CriteriaModel* myCase)
{
    // evaporation on surface
    double evaporationOpenWater = MINVALUE(myCase->output.dailyMaxEvaporation, myCase->layers[0].waterContent);
    myCase->layers[0].waterContent -= evaporationOpenWater;
    myCase->output.dailyEvaporation = evaporationOpenWater;

    double residualEvaporation = myCase->output.dailyMaxEvaporation - evaporationOpenWater;
    if (residualEvaporation < EPSILON)
        return true;

    // evaporation on soil
    unsigned int lastLayerEvap = unsigned(floor(MAX_EVAPORATION_DEPTH / myCase->layerThickness)) +1;
    double* coeffEvap = new double[unsigned(lastLayerEvap)];
    double layerDepth, coeffDepth;

    double sumCoeff = 0;
    double minDepth = myCase->layers[1].depth + myCase->layers[1].thickness / 2;
    for (unsigned int i=1; i <= lastLayerEvap; i++)
    {
        layerDepth = myCase->layers[i].depth + myCase->layers[i].thickness / 2.0;

        coeffDepth = MAXVALUE((layerDepth - minDepth) / (MAX_EVAPORATION_DEPTH - minDepth), 0);
        // values = 1 a depthMin, ~0.1 a depthMax
        coeffEvap[i-1] = exp(-2 * coeffDepth);

        coeffEvap[i-1] = MINVALUE(1.0, exp((-layerDepth * 2.0) / MAX_EVAPORATION_DEPTH));
        sumCoeff += coeffEvap[i-1];
    }

    bool isWaterSupply = true;
    double sumEvap, evapLayerThreshold, evapLayer;
    while ((residualEvaporation > EPSILON) && (isWaterSupply == true))
    {
        isWaterSupply = false;
        sumEvap = 0.0;

        for (unsigned int i=1; i<=lastLayerEvap; i++)
        {
            evapLayerThreshold = myCase->layers[i].FC - coeffEvap[i-1] * (myCase->layers[i].FC - myCase->layers[i].HH);
            evapLayer = (coeffEvap[i-1] / sumCoeff) * residualEvaporation;

            if (myCase->layers[i].waterContent > (evapLayerThreshold + evapLayer))
                isWaterSupply = true;
            else if (myCase->layers[i].waterContent > evapLayerThreshold)
                evapLayer = myCase->layers[i].waterContent - evapLayerThreshold;
            else
                evapLayer = 0.0;

            myCase->layers[i].waterContent -= evapLayer;
            sumEvap += evapLayer;
        }

        residualEvaporation -= sumEvap;
        myCase->output.dailyEvaporation  += sumEvap;
    }

    delete[] coeffEvap;

    return true;
}


// return total daily transpiration [mm]
// or percentage of water stress (if getWaterStress = true)
double cropTranspiration(CriteriaModel* myCase, bool getWaterStress)
{
    // check
    if (myCase->myCrop.idCrop == "") return 0.0;
    if (! myCase->myCrop.isLiving) return 0.0;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return 0.0;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return 0.0;

    double thetaWP;                                 // [m3 m-3] volumetric water content at Wilting Point
    double cropWP;                                  // [mm] wilting point specific for crop
    double surplusThreshold;                        // [mm] water surplus stress threshold
    double waterScarcityThreshold;                  // [mm] water scarcity stress threshold
    double WSS;                                     // [] water surplus stress

    double TRs=0.0;                                 // [mm] actual transpiration with only water scarsity stress
    double TRe=0.0;                                 // [mm] actual transpiration with only water surplus stress
    double totRootDensityWithoutStress = 0.0;       // [-]
    double stress = 0.0;                            // [-]
    double redistribution = 0.0;                    // [mm]

    if (myCase->output.dailyMaxTranspiration < EPSILON)
        return 0;

    // initialize
    bool* isLayerStressed = new bool[myCase->nrLayers];
    double* layerTranspiration = new double[myCase->nrLayers];
    for (unsigned int i = 0; i < myCase->nrLayers; i++)
    {
        isLayerStressed[i] = false;
        layerTranspiration[i] = 0;
    }

    // water surplus
    if (myCase->myCrop.isWaterSurplusResistant())
        WSS = 0.0;
    else
        WSS = 0.5;

    for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
    {
        // [mm]
        surplusThreshold = myCase->layers[i].SAT - (WSS * (myCase->layers[i].SAT - myCase->layers[i].FC));

        thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(myCase->myCrop.psiLeaf), myCase->layers[i].horizon);
        // [mm]
        cropWP = thetaWP * myCase->layers[i].thickness * myCase->layers[i].soilFraction * 1000.0;

        // [mm]
        waterScarcityThreshold = myCase->layers[i].FC - myCase->myCrop.fRAW * (myCase->layers[i].FC - cropWP);

        if (myCase->layers[i].waterContent > surplusThreshold)
        {
            //WATER SURPLUS
            layerTranspiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i] *
                                ((myCase->layers[i].SAT - myCase->layers[i].waterContent) / (myCase->layers[i].SAT - surplusThreshold));

            TRe += layerTranspiration[i];
            TRs += myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else if (myCase->layers[i].waterContent < waterScarcityThreshold)
        {
            //WATER SCARSITY
            if (myCase->layers[i].waterContent <= cropWP)
            {
                layerTranspiration[i] = 0;
            }
            else
            {
                layerTranspiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i] *
                                     ((myCase->layers[i].waterContent - cropWP) / (waterScarcityThreshold - cropWP));
            }

            TRs += layerTranspiration[i];
            TRe += myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else
        {
            //normal conditions
            layerTranspiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];

            TRs += layerTranspiration[i];
            TRe += layerTranspiration[i];

            if ((myCase->layers[i].waterContent - layerTranspiration[i]) > waterScarcityThreshold)
            {
                isLayerStressed[i] = false;
                totRootDensityWithoutStress +=  myCase->myCrop.roots.rootDensity[i];
            }
            else
            {
                isLayerStressed[i] = true;
            }
        }
    }

    // Hydraulic redistribution
    // the movement of water from moist to dry soil through plant roots
    // TODO add numerical process
    if (myCase->output.dailyMaxTranspiration > 0 && totRootDensityWithoutStress > 0)
    {
        stress = 1.0 - (TRs / myCase->output.dailyMaxTranspiration);

        if (stress > EPSILON)
        {
            redistribution = MINVALUE(stress, totRootDensityWithoutStress * 0.5) * myCase->output.dailyMaxTranspiration;
            // maximum 1.6 mm (Neumann at al. values span from 0 to 3.2)
            // redistribution = MINVALUE(redistribution, 1.6);

            for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
            {
                if (! isLayerStressed[i])
                {
                    double addTransp = redistribution * (myCase->myCrop.roots.rootDensity[i] / totRootDensityWithoutStress);
                    layerTranspiration[i] += addTransp;
                    TRs += addTransp;
                    TRe += addTransp;
                }
            }
        }
    }

    if (getWaterStress)
    {
        // return water stress
        return 1.0 - (TRs / myCase->output.dailyMaxTranspiration);
    }

    // update water content
    double dailyTranspiration = 0;
    for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
    {
        myCase->layers[unsigned(i)].waterContent -= layerTranspiration[i];
        dailyTranspiration += layerTranspiration[i];
    }

    return dailyTranspiration;

}



/*!
 * \brief getCropWaterDeficit
 * \param myCase
 * \return sum of water deficit (mm) in the rooting zone
 */
double getCropWaterDeficit(CriteriaModel* myCase)
{
    //check
    if (! myCase->myCrop.isLiving) return NODATA;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return NODATA;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return NODATA;

    double waterDeficit = 0.0;
    for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
        waterDeficit += myCase->layers[unsigned(i)].FC - myCase->layers[unsigned(i)].waterContent;

    return MAXVALUE(waterDeficit, 0.0);
}

