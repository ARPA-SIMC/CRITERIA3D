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


// initialization of crop
void initializeCrop(CriteriaModel* myCase, int currentDoy)
{    
    // initialize root density
    if (myCase->myCrop.roots.rootDensity != nullptr) delete[] myCase->myCrop.roots.rootDensity;
    myCase->myCrop.roots.rootDensity = new double[unsigned(myCase->nrLayers)];

    // initialize root depth
    myCase->myCrop.roots.rootDepth = 0;

    // initialize transpiration
    if (myCase->myCrop.roots.transpiration != nullptr) delete[] myCase->myCrop.roots.transpiration;
    myCase->myCrop.roots.transpiration = new double[unsigned(myCase->nrLayers)];

    // root max depth
    if (myCase->myCrop.roots.rootDepthMax > myCase->mySoil.totalDepth)
        myCase->myCrop.roots.rootDepthMax = myCase->mySoil.totalDepth;

    myCase->myCrop.degreeDays = 0;

    if (myCase->meteoPoint.latitude > 0)
        myCase->myCrop.doyStartSenescence = 305;
    else
        myCase->myCrop.doyStartSenescence = 120;

    myCase->myCrop.LAIstartSenescence = NODATA;
    myCase->myCrop.currentSowingDoy = NODATA;

    myCase->myCrop.daysSinceIrrigation = NODATA;

    // is crop living?
    if (myCase->myCrop.isPluriannual())
        myCase->myCrop.isLiving = true;
    else
    {
        myCase->myCrop.isLiving = myCase->myCrop.isInsideTypicalCycle(currentDoy);

        if (myCase->myCrop.isLiving == true)
            myCase->myCrop.currentSowingDoy = myCase->myCrop.sowingDoy;
    }

    // reset crop
    myCase->myCrop.resetCrop(myCase->nrLayers);
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



bool updateLAI(CriteriaModel* myCase, int myDoy)
{
    double degreeDaysLai = 0;
    double myLai = 0;

    if (! myCase->myCrop.isPluriannual())
    {
        if (! myCase->myCrop.isEmerged)
        {
            if (myCase->myCrop.degreeDays < myCase->myCrop.degreeDaysEmergence)
                return true;
            else if (myDoy - myCase->myCrop.sowingDoy >= MIN_EMERGENCE_DAYS)
            {
                myCase->myCrop.isEmerged = true;
                degreeDaysLai = myCase->myCrop.degreeDays - myCase->myCrop.degreeDaysEmergence;
            }
            else
                return true;
        }
        else
        {
            degreeDaysLai = myCase->myCrop.degreeDays - myCase->myCrop.degreeDaysEmergence;
        }

        if (degreeDaysLai > 0)
            myLai = leafDevelopment::getLAICriteria(&(myCase->myCrop), degreeDaysLai);
    }
    else
    {
        if (myCase->myCrop.type == GRASS)
            // grass cut
            if (myCase->myCrop.degreeDays >= myCase->myCrop.degreeDaysIncrease)
                myCase->myCrop.resetCrop(myCase->nrLayers);

        if (myCase->myCrop.degreeDays > 0)
            myLai = leafDevelopment::getLAICriteria(&(myCase->myCrop), myCase->myCrop.degreeDays);
        else
            myLai = myCase->myCrop.LAImin;

        bool inSenescence;
        if (myCase->meteoPoint.latitude > 0)
            inSenescence = (myDoy >= myCase->myCrop.doyStartSenescence);
        else
            inSenescence = ((myDoy >= myCase->myCrop.doyStartSenescence) && (myDoy < 182));

        if (inSenescence)
        {
            if (myDoy == myCase->myCrop.doyStartSenescence || int(myCase->myCrop.LAIstartSenescence) == int(NODATA))
                myCase->myCrop.LAIstartSenescence = myLai;
            else
                myLai = leafDevelopment::getLAISenescence(myCase->myCrop.LAImin,
                        myCase->myCrop.LAIstartSenescence, myDoy-myCase->myCrop.doyStartSenescence);
        }

        if (myCase->myCrop.type == FRUIT_TREE)
            myLai += myCase->myCrop.LAIgrass;
    }

    myCase->myCrop.LAI = myLai;

    return true;
}


bool updateRoots(CriteriaModel* myCase)
{
    root::computeRootDepth(&(myCase->myCrop), myCase->mySoil.totalDepth, myCase->myCrop.degreeDays, myCase->output.dailyWaterTable);
    return root::computeRootDensity(&(myCase->myCrop), myCase->layer, myCase->nrLayers, myCase->mySoil.totalDepth);
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
    for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
    {
        threshold = myCase->layer[i].FC - myCase->myCrop.fRAW * (myCase->layer[i].FC - myCase->layer[i].WP);

        layerRAW = (myCase->layer[i].waterContent - threshold);

        depth = myCase->layer[i].depth + myCase->layer[i].thickness / 2.0;

        if (myCase->myCrop.roots.rootDepth < depth)
                layerRAW *= (myCase->myCrop.roots.rootDepth - depth) / myCase->layer[i].thickness;

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

    while ((lastLayer < (myCase->nrLayers-1)) && (myCase->layer[lastLayer].depth < 1.0))
        lastLayer++;

    lastLayer = MAXVALUE(lastLayer, myCase->myCrop.roots.lastRootLayer);

    // surface water
    double RAW = myCase->layer[0].waterContent;

    for (int i = 1; i <= lastLayer; i++)
    {
        if (i < myCase->myCrop.roots.firstRootLayer)
            threshold = myCase->layer[i].FC;
        else
            // rooting zone
            threshold = myCase->layer[i].FC - myCase->myCrop.fRAW * (myCase->layer[i].FC - myCase->layer[i].WP);

        if(myCase->layer[i].waterContent > threshold)
            RAW += (myCase->layer[i].waterContent - threshold);
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
    double todayWater = double(currentPrec) + myCase->layer[0].waterContent;
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

    int i=0;
    while (i < myCase->nrLayers && residualIrrigation > 0)
    {
        if (myCase->layer[i].waterContent < myCase->layer[i].FC)
        {
            myDeficit = float(myCase->layer[i].FC - myCase->layer[i].waterContent);
            myDeficit = MINVALUE(myDeficit, residualIrrigation);

            myCase->layer[i].waterContent += double(myDeficit);
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
    double evaporationOpenWater = MINVALUE(myCase->output.dailyMaxEvaporation, myCase->layer[0].waterContent);
    myCase->layer[0].waterContent -= evaporationOpenWater;
    myCase->output.dailyEvaporation = evaporationOpenWater;

    double residualEvaporation = myCase->output.dailyMaxEvaporation - evaporationOpenWater;
    if (residualEvaporation < EPSILON)
        return true;

    // evaporation on soil
    int lastLayerEvap = int(floor(MAX_EVAPORATION_DEPTH / myCase->layerThickness)) +1;
    double* coeffEvap = new double[unsigned(lastLayerEvap)];
    double layerDepth, coeffDepth;

    double sumCoeff = 0;
    double minDepth = myCase->layer[1].depth + myCase->layer[1].thickness / 2;
    for (int i=1; i <= lastLayerEvap; i++)
    {
        layerDepth = myCase->layer[i].depth + myCase->layer[i].thickness / 2.0;

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

        for (int i=1; i<=lastLayerEvap; i++)
        {
            evapLayerThreshold = myCase->layer[i].FC - coeffEvap[i-1] * (myCase->layer[i].FC - myCase->layer[i].HH);
            evapLayer = (coeffEvap[i-1] / sumCoeff) * residualEvaporation;

            if (myCase->layer[i].waterContent > (evapLayerThreshold + evapLayer))
                isWaterSupply = true;
            else if (myCase->layer[i].waterContent > evapLayerThreshold)
                evapLayer = myCase->layer[i].waterContent - evapLayerThreshold;
            else
                evapLayer = 0.0;

            myCase->layer[i].waterContent -= evapLayer;
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
    //check
    if (myCase->myCrop.idCrop == "") return 0.0;
    if (! myCase->myCrop.isLiving) return 0.0;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return 0.0;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return 0.0;

    double thetaWP;                                 // [m3 m-3] volumetric water content at Wilting Point
    double soilThickness;                           // [mm] thickness of soil layer
    double surplusThreshold;                        // [mm] water surplus stress threshold
    double waterScarcityThreshold;                  // [mm] water scarcity stress threshold
    double cropWP;                                  // [mm] wilting point specific for crop
    double WSS;                                     // [] water surplus stress

    double TRs=0.0;                                 // [mm] actual transpiration with only water scarsity stress
    double TRe=0.0;                                 // [mm] actual transpiration with only water surplus stress
    double totRootDensityWithoutStress = 0.0;       // [-]
    double stress = 0.0;                            // [-]
    double redistribution = 0.0;                    // [mm]

    // initialize layer transpiration
    for (int i = 0; i < myCase->nrLayers; i++)
        myCase->myCrop.roots.transpiration[i] = 0.0;

    if (myCase->output.dailyMaxTranspiration < EPSILON)
        return 0;

    // initialize stressed layers
    bool* isLayerStressed = new bool[unsigned(myCase->nrLayers)];
    for (int i = 0; i < myCase->nrLayers; i++)
        isLayerStressed[i] = false;

    // deactivated water surplus
    if (myCase->myCrop.isWaterSurplusResistant())
        WSS = 0.0;
    else
        WSS = 0.0;

    for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
    {
        surplusThreshold = myCase->layer[i].SAT - (WSS * (myCase->layer[i].SAT - myCase->layer[i].FC));

        soilThickness = myCase->layer[i].thickness * myCase->layer[i].soilFraction * 1000.0;    // [mm]

        thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(myCase->myCrop.psiLeaf), myCase->layer[i].horizon);

        cropWP = thetaWP * soilThickness;                                                         // [mm]

        waterScarcityThreshold = myCase->layer[i].FC - myCase->myCrop.fRAW * (myCase->layer[i].FC - cropWP);

        if (myCase->layer[i].waterContent > surplusThreshold)
        {
            //WATER SURPLUS
            myCase->myCrop.roots.transpiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i] *
                    (myCase->layer[i].SAT - myCase->layer[i].waterContent) / (myCase->layer[i].SAT - surplusThreshold);

            TRe += myCase->myCrop.roots.transpiration[i];
            TRs += myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else if (myCase->layer[i].waterContent < waterScarcityThreshold)
        {
            //WATER SCARSITY
            myCase->myCrop.roots.transpiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i] *
                    (myCase->layer[i].waterContent - cropWP) / (waterScarcityThreshold - cropWP);

            TRs += myCase->myCrop.roots.transpiration[i];
            TRe += myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else
        {
            //normal conditions
            myCase->myCrop.roots.transpiration[i] = myCase->output.dailyMaxTranspiration * myCase->myCrop.roots.rootDensity[i];

            TRs += myCase->myCrop.roots.transpiration[i];
            TRe += myCase->myCrop.roots.transpiration[i];

            if ((myCase->layer[i].waterContent - myCase->myCrop.roots.transpiration[i]) > waterScarcityThreshold)
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
    double value;
    if (myCase->output.dailyMaxTranspiration > 0)
    {
        stress = 1.0 - (TRs / myCase->output.dailyMaxTranspiration);

        // at least 20% of roots moist
        if ((stress > EPSILON) && (totRootDensityWithoutStress > 0.2))
        {
            redistribution = MINVALUE(stress, totRootDensityWithoutStress) * myCase->output.dailyMaxTranspiration;
            // maximum 1.6 mm (Neumann at al. values span from 0 to 3.2)
            redistribution = MINVALUE(redistribution, 1.6);

            for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
            {
                if (! isLayerStressed[i])
                {
                    value = redistribution * (myCase->myCrop.roots.rootDensity[i] / totRootDensityWithoutStress);
                    myCase->myCrop.roots.transpiration[i] += value;
                    TRs += value;
                    TRe += value;
                }
            }
        }
    }

    delete[] isLayerStressed;

    if (getWaterStress)
    {
        // return water stress
        return 1.0 - (TRs / myCase->output.dailyMaxTranspiration);
    }

    // update water content
    double dailyTranspiration = 0.0;
    for (int i = myCase->myCrop.roots.firstRootLayer; i <= myCase->myCrop.roots.lastRootLayer; i++)
    {
        myCase->layer[i].waterContent -= myCase->myCrop.roots.transpiration[i];
        dailyTranspiration += myCase->myCrop.roots.transpiration[i];
    }

    return dailyTranspiration;

}


bool updateCrop(CriteriaModel* myCase, QString* myError, Crit3DDate myDate,
                float tmin, float tmax, float waterTableDepth)
{
    *myError = "";

    if (myCase->myCrop.idCrop == "")
        return false;

    // check start/end crop cycle (update isLiving)
    if (myCase->myCrop.needReset(myDate, float(myCase->meteoPoint.latitude), waterTableDepth))
    {
        myCase->myCrop.resetCrop(myCase->nrLayers);
    }

    if (myCase->myCrop.isLiving)
    {
        int currentDoy = getDoyFromDate(myDate);

        // update degree days
        myCase->myCrop.degreeDays += computeDegreeDays(double(tmin), double(tmax), myCase->myCrop.thermalThreshold, myCase->myCrop.upperThermalThreshold);

        // update LAI
        if (! updateLAI(myCase, currentDoy))
        {
            *myError = "Error in updating LAI for crop " + QString::fromStdString(myCase->myCrop.idCrop);
            return false;
        }

        // update roots
        if (! updateRoots(myCase))
        {
            *myError = "Error in updating roots for crop " + QString::fromStdString(myCase->myCrop.idCrop);
            return false;
        }
    }

    return true;
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
        waterDeficit += myCase->layer[i].FC - myCase->layer[i].waterContent;

    return MAXVALUE(waterDeficit, 0.0);
}

