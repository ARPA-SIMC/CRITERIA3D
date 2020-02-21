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



/*!
 * \brief getCropReadilyAvailableWater
 * \return sum of readily available water (mm) in the rooting zone
 */
double getCropReadilyAvailableWater(CriteriaModel* myCase)
{
    if (! myCase->myCrop.isLiving) return 0.;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return 0.;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return 0.;

    double sumRAW = 0.0;
    for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
    {
        double thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(myCase->myCrop.psiLeaf), myCase->layers[i].horizon);
        // [mm]
        double cropWP = thetaWP * myCase->layers[i].thickness * myCase->layers[i].soilFraction * 1000.0;
        // [mm]
        double threshold = myCase->layers[i].FC - myCase->myCrop.fRAW * (myCase->layers[i].FC - cropWP);

        double layerRAW = (myCase->layers[i].waterContent - threshold);

        double layerMaxDepth = myCase->layers[i].depth + myCase->layers[i].thickness / 2.0;
        if (myCase->myCrop.roots.rootDepth < layerMaxDepth)
        {
                layerRAW *= (myCase->myCrop.roots.rootDepth - layerMaxDepth) / myCase->layers[i].thickness;
        }

        sumRAW += layerRAW;
    }

    return sumRAW;
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


bool computeEvaporation(CriteriaModel* myCase)
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


/*!
 * \brief getSoilWaterDeficit
 * \param myCase
 * \return sum of water deficit (mm) in the first meter of soil
 */
double getSoilWaterDeficit(CriteriaModel* myCase)
{
    // surface water content
    double waterDeficit = -myCase->layers[0].waterContent;

    for (unsigned int i = 1; i <= myCase->nrLayers; i++)
    {
        if (myCase->layers[i].depth > 1)
            return waterDeficit;

        waterDeficit += myCase->layers[unsigned(i)].FC - myCase->layers[unsigned(i)].waterContent;
    }

    return waterDeficit;
}

