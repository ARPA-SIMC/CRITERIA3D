#include "commonConstants.h"
#include "basicMath.h"
#include "gis.h"
#include "vine3DProject.h"
#include "soilFluxes3D.h"
#include "waterBalance.h"

#include <math.h>
#include <vector>


Crit3DWaterBalanceMaps::Crit3DWaterBalanceMaps()
{
    initialize();
}

Crit3DWaterBalanceMaps::Crit3DWaterBalanceMaps(const gis::Crit3DRasterGrid &myDEM)
{
    initializeWithDEM(myDEM);
}

void Crit3DWaterBalanceMaps::initialize()
{
    bottomDrainageMap = new gis::Crit3DRasterGrid;
    waterInflowMap = new gis::Crit3DRasterGrid;
}

void Crit3DWaterBalanceMaps::initializeWithDEM(const gis::Crit3DRasterGrid &myDEM)
{
    initialize();
    bottomDrainageMap->initializeGrid(myDEM);
    waterInflowMap->initializeGrid(myDEM);
}


gis::Crit3DRasterGrid* Crit3DWaterBalanceMaps::getMapFromVar(criteria3DVariable myVar)
{
    if (myVar == bottomDrainage)
        return bottomDrainageMap;
    else if (myVar == waterInflow)
        return waterInflowMap;
    else
        return nullptr;
}



double getSoilVar(Vine3DProject* myProject, int soilIndex, int layerIndex, soil::soilVariable myVar)
{
    int horizonIndex = soil::getHorizonIndex(myProject->soilList[soilIndex], myProject->layerDepth[unsigned(layerIndex)]);
    if (horizonIndex == NODATA)
        return NODATA;

    if (myVar == soil::soilWaterPotentialWP)
    {
        // [m]
        return myProject->soilList[soilIndex].horizon[horizonIndex].wiltingPoint / GRAVITY;
    }
    else if (myVar == soil::soilWaterPotentialFC)
    {
        // [m]
        return myProject->soilList[soilIndex].horizon[horizonIndex].fieldCapacity / GRAVITY;
    }
    else if (myVar == soil::soilWaterContentFC)
    {
        // [m^3 m^-3]
        return myProject->soilList[soilIndex].horizon[horizonIndex].waterContentFC;
    }
    else if (myVar == soil::soilWaterContentSat)
    {
        // [m^3 m^-3]
        return myProject->soilList[soilIndex].horizon[horizonIndex].vanGenuchten.thetaS;
    }
    else if (myVar == soil::soilWaterContentWP)
    {
        if (myProject->cultivar.size() > 0)
        {
            double signPsiLeaf = - myProject->cultivar[0].parameterWangLeuning.psiLeaf;         // [kPa]
            // [m^3 m^-3]
            return soil::thetaFromSignPsi(signPsiLeaf, myProject->soilList[soilIndex].horizon[horizonIndex]);
        }
        else
            return NODATA;
    }
    else
    {
        return NODATA;
    }
}


std::vector<double> getSoilVarProfile(Vine3DProject* myProject, int row, int col, soil::soilVariable myVar)
{
    std::vector<double> varProfile;
    varProfile.resize(myProject->nrLayers);

    for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        varProfile[layerIndex] = NODATA;

    int soilIndex = myProject->getSoilIndex(row, col);
    if (soilIndex != NODATA)
    {
        for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        {
            if (myVar == soil::soilWaterPotentialFC || myVar == soil::soilWaterPotentialWP
                || myVar == soil::soilWaterContentSat  || myVar == soil::soilWaterContentFC
                || myVar == soil::soilWaterContentWP)
            {
                varProfile[layerIndex] = getSoilVar(myProject, soilIndex, layerIndex, myVar);
            }
        }
    }

    return varProfile;
}


std::vector<double> getCriteria3DVarProfile(Vine3DProject* myProject, int row, int col, criteria3DVariable myVar)
{
    std::vector<double> varProfile;
    varProfile.resize(myProject->nrLayers);

    for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        varProfile[layerIndex] = NODATA;

    for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
    {
        long nodeIndex = myProject->indexMap.at(layerIndex).value[row][col];

        if (nodeIndex != myProject->indexMap.at(layerIndex).header->flag)
        {
             varProfile[layerIndex] = getCriteria3DVar(myVar, nodeIndex);
        }
    }

    return varProfile;
}



bool setCriteria3DVarMap(int myLayerIndex, Vine3DProject* myProject, criteria3DVariable myVar,
                        gis::Crit3DRasterGrid* myCriteria3DMap)
{
    long nodeIndex;

    for (int row = 0; row < myProject->indexMap.at(myLayerIndex).header->nrRows; row++)
        for (int col = 0; col < myProject->indexMap.at(myLayerIndex).header->nrCols; col++)
        {
            nodeIndex = myProject->indexMap.at(myLayerIndex).value[row][col];
            if (nodeIndex != myProject->indexMap.at(myLayerIndex).header->flag)
            {
                if (! setCriteria3DVar(myVar, nodeIndex, myCriteria3DMap->value[row][col])) return false;
            }
        }

    return true;
}


bool getCriteria3DVarMap(Vine3DProject* myProject, criteria3DVariable myVar,
                        int layerIndex, gis::Crit3DRasterGrid* criteria3DMap)
{
    long nodeIndex;
    double myValue;

    criteria3DMap->initializeGrid(myProject->indexMap.at(layerIndex));

    for (int row = 0; row < myProject->indexMap.at(layerIndex).header->nrRows; row++)
        for (int col = 0; col < myProject->indexMap.at(layerIndex).header->nrCols; col++)
        {
            nodeIndex = myProject->indexMap.at(layerIndex).value[row][col];
            if (nodeIndex != myProject->indexMap.at(layerIndex).header->flag)
            {
                myValue = getCriteria3DVar(myVar, nodeIndex);

                if (myValue == NODATA)
                    criteria3DMap->value[row][col] = criteria3DMap->header->flag;
                else
                    criteria3DMap->value[row][col] = myValue;
            }
            else
                criteria3DMap->value[row][col] = criteria3DMap->header->flag;
        }

    return true;
}

bool getSoilSurfaceMoisture(Vine3DProject* myProject, gis::Crit3DRasterGrid* outputMap, double lowerDepth)
{
    int lastLayer;
    long nodeIndex;
    double waterContent, sumWater = 0.0, wiltingPoint, minWater = 0.0, saturation, maxWater = 0.0;
    double soilSurfaceMoisture;
    int layer;

    lastLayer = myProject->getSoilLayerIndex(lowerDepth);

    for (int row = 0; row < myProject->indexMap.at(0).header->nrRows; row++)
        for (int col = 0; col < myProject->indexMap.at(0).header->nrCols; col++)
        {
            outputMap->value[row][col] = outputMap->header->flag;

            if (int(myProject->indexMap.at(0).value[row][col]) != int(myProject->indexMap.at(0).header->flag))
            {
                for (layer = 0; layer <= lastLayer; layer++)
                {
                    nodeIndex = long(myProject->indexMap.at(size_t(layer)).value[row][col]);

                    if (layer == 0)
                    {
                        sumWater = soilFluxes3D::getWaterContent(nodeIndex);
                        minWater = 0.0;
                        maxWater = 0.0;
                    }
                    else
                    {
                        if (int(myProject->indexMap.at(size_t(layer)).value[row][col]) != int(myProject->indexMap.at(size_t(layer)).header->flag))
                        {
                            waterContent = soilFluxes3D::getWaterContent(nodeIndex);                        //[m^3 m^-3]
                            sumWater += waterContent * myProject->layerThickness.at(size_t(layer));
                            wiltingPoint = getSoilVar(myProject, 0, layer, soil::soilWaterContentWP);       //[m^3 m^-3]
                            minWater += wiltingPoint * myProject->layerThickness.at(size_t(layer));         //[m]
                            saturation = getSoilVar(myProject, 0, layer, soil::soilWaterContentSat);        //[m^3 m^-3]
                            maxWater += saturation * myProject->layerThickness.at(size_t(layer));
                        }
                    }
                }
                soilSurfaceMoisture = 100 * ((sumWater-minWater) / (maxWater-minWater));
                soilSurfaceMoisture = MINVALUE(MAXVALUE(soilSurfaceMoisture, 0), 100);
                outputMap->value[row][col] = float(soilSurfaceMoisture);
            }
         }

    return true;
}

// return map of available water content in the root zone [mm]
bool getRootZoneAWCmap(Vine3DProject* myProject, gis::Crit3DRasterGrid* outputMap)
{
    long nodeIndex;
    double awc, thickness, sumAWC;
    int soilIndex, horizonIndex;

    for (int row = 0; row < outputMap->header->nrRows; row++)
        for (int col = 0; col < outputMap->header->nrCols; col++)
        {
            //initialize
            outputMap->value[row][col] = outputMap->header->flag;

            if (myProject->indexMap.at(0).value[row][col] != myProject->indexMap.at(0).header->flag)
            {
                sumAWC = 0.0;
                soilIndex = myProject->getSoilIndex(row, col);
                int caseIndex = myProject->getModelCaseIndex(row,col);

                if (soilIndex != NODATA && caseIndex != NODATA)
                {
                    for (unsigned int layer = 1; layer < myProject->nrLayers; layer++)
                    {
                        nodeIndex = myProject->indexMap.at(layer).value[row][col];

                        if (nodeIndex != myProject->indexMap.at(layer).header->flag)
                        {
                            if (myProject->grapevine.getRootDensity(&(myProject->modelCases[caseIndex]), layer) > 0.0)
                            {
                                awc = soilFluxes3D::getAvailableWaterContent(nodeIndex);  //[m3 m-3]
                                if (awc != NODATA)
                                {
                                    thickness = myProject->layerThickness[layer] * 1000.0;  //[mm]
                                    horizonIndex = soil::getHorizonIndex(myProject->soilList[soilIndex], myProject->layerDepth[layer]);
                                    sumAWC += (awc * thickness);         //[mm]
                                }
                            }
                        }
                    }
                }
                outputMap->value[row][col] = sumAWC;
            }
        }

    return true;
}


bool getCriteria3DIntegrationMap(Vine3DProject* myProject, criteria3DVariable myVar,
                       double upperDepth, double lowerDepth, gis::Crit3DRasterGrid* criteria3DMap)
{
    if (upperDepth > myProject->computationSoilDepth) return false;
    lowerDepth = MINVALUE(lowerDepth, myProject->computationSoilDepth);

    if (upperDepth == lowerDepth)
    {
        int layerIndex = myProject->getSoilLayerIndex(upperDepth);
        if (layerIndex == INDEX_ERROR) return false;
        return getCriteria3DVarMap(myProject, myVar, layerIndex, criteria3DMap);
    }

    int firstIndex, lastIndex;
    double firstThickness, lastThickness;
    firstIndex = myProject->getSoilLayerIndex(upperDepth);
    lastIndex = myProject->getSoilLayerIndex(lowerDepth);
    if ((firstIndex == INDEX_ERROR)||(lastIndex == INDEX_ERROR))
        return false;
    firstThickness = myProject->getSoilLayerBottom(firstIndex) - upperDepth;
    lastThickness = lowerDepth - myProject->getSoilLayerTop(lastIndex);

    long nodeIndex;
    double myValue, sumValues;
    double thickCoeff, sumCoeff;
    int soilIndex;

    for (int row = 0; row < myProject->indexMap.at(0).header->nrRows; row++)
        for (int col = 0; col < myProject->indexMap.at(0).header->nrCols; col++)
        {
            criteria3DMap->value[row][col] = criteria3DMap->header->flag;

            if (myProject->indexMap.at(0).value[row][col] != myProject->indexMap.at(0).header->flag)
            {
                sumValues = 0.0;
                sumCoeff = 0.0;

                soilIndex = myProject->getSoilIndex(row, col);

                if (soilIndex != NODATA)
                {
                    for (int i = firstIndex; i <= lastIndex; i++)
                    {
                        nodeIndex = myProject->indexMap.at(i).value[row][col];
                        if (nodeIndex != long(myProject->indexMap.at(size_t(i)).header->flag))
                        {
                            myValue = getCriteria3DVar(myVar, nodeIndex);

                            if (myValue != NODATA)
                            {
                                if (i == firstIndex)
                                    thickCoeff = firstThickness;
                                else if (i == lastIndex)
                                    thickCoeff = lastThickness;
                                else
                                    thickCoeff = myProject->layerThickness[i];

                                sumValues += (myValue * thickCoeff);
                                sumCoeff += thickCoeff;
                            }
                        }
                    }
                    criteria3DMap->value[row][col] = sumValues / sumCoeff;
                }
            }
        }

    return true;
}


bool saveWaterBalanceCumulatedOutput(Vine3DProject* myProject, QDate myDate, criteria3DVariable myVar,
                            QString varName, QString notes, QString outputPath)
{
    QString outputFilename = outputPath + getOutputNameDaily(varName, notes, myDate);
    std::string errorStr;
    gis::Crit3DRasterGrid* myMap = myProject->outputWaterBalanceMaps->getMapFromVar(myVar);
    if (! gis::writeEsriGrid(outputFilename.toStdString(), myMap, errorStr))
    {
         myProject->logError(QString::fromStdString(errorStr));
         return false;
    }

    return true;
}


bool saveWaterBalanceOutput(Vine3DProject* myProject, QDate myDate, criteria3DVariable myVar,
                            QString varName, QString notes, QString outputPath,
                            double upperDepth, double lowerDepth)
{
    gis::Crit3DRasterGrid* myMap = new gis::Crit3DRasterGrid();
    myMap->initializeGrid(myProject->indexMap.at(0));

    if (myVar == soilSurfaceMoisture)
    {
        if (! getSoilSurfaceMoisture(myProject, myMap, lowerDepth))
            return false;
    }
    else if(myVar == availableWaterContent)
    {
        if (! getRootZoneAWCmap(myProject, myMap))
            return false;
    }
    else
    {
        if (! getCriteria3DIntegrationMap(myProject, myVar, upperDepth, lowerDepth, myMap))
            return false;
    }

    QString outputFilename = outputPath + getOutputNameDaily(varName, notes, myDate);
    std::string errorStr;
    if (! gis::writeEsriGrid(outputFilename.toStdString(), myMap, errorStr))
    {
         myProject->logError(QString::fromStdString(errorStr));
         return false;
    }

    myMap->clear();

    return true;
}



bool loadWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString statePath, criteria3DVariable myVar)
{
    std::string errorStr;
    QString myMapName;

    gis::Crit3DRasterGrid myMap;

    QString myPrefix = getDailyPrefixFromVar(myDate, myVar);

    for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
    {
        myMapName = statePath + myPrefix + QString::number(layerIndex);
        if (! gis::readEsriGrid(myMapName.toStdString(), &myMap, errorStr))
        {
            myProject->logError(QString::fromStdString(errorStr));
            return false;
        }
        else
            if (!setCriteria3DVarMap(layerIndex, myProject, myVar, &myMap))
                return false;
    }
    myMap.clear();
    return true;
}


bool saveWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString statePath, criteria3DVariable myVar)
{
    gis::Crit3DRasterGrid* myMap;
    myMap = new gis::Crit3DRasterGrid();
    myMap->initializeGrid(myProject->indexMap.at(0));

    QString myPrefix = getDailyPrefixFromVar(myDate, myVar);

    for (unsigned int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        if (getCriteria3DVarMap(myProject, myVar, layerIndex, myMap))
        {
            QString myOutputMapName = statePath + myPrefix + QString::number(layerIndex);

            std::string errorStr;
            if (! gis::writeEsriGrid(myOutputMapName.toStdString(), myMap, errorStr))
            {
                myProject->logError(QString::fromStdString(errorStr));
                return false;
            }
        }
    myMap->clear();

    return true;
}


