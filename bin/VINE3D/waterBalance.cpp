#include <math.h>
#include <vector>
#include "commonConstants.h"
#include "gis.h"
#include "dataHandler.h"
#include "vine3DProject.h"
#include "soilFluxes3D.h"
#include "waterBalance.h"


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

void resetWaterBalanceMap(Vine3DProject* myProject)
{
    myProject->outputWaterBalanceMaps->bottomDrainageMap->setConstantValueWithBase(0, myProject->DEM);
    myProject->outputWaterBalanceMaps->waterInflowMap->setConstantValueWithBase(0, myProject->DEM);
}


void updateWaterBalanceMaps(Vine3DProject* myProject)
{
    long row, col;
    long nodeIndex;
    int layer, soilIndex;
    double flow, flow_mm;
    double area;

    area = pow(myProject->outputWaterBalanceMaps->bottomDrainageMap->header->cellSize, 2);

    for (row = 0; row < myProject->outputWaterBalanceMaps->bottomDrainageMap->header->nrRows; row++)
        for (col = 0; col < myProject->outputWaterBalanceMaps->bottomDrainageMap->header->nrCols; col++)
            if (int(myProject->indexMap.at(0).value[row][col]) != int(myProject->indexMap.at(0).header->flag))
            {
                soilIndex = myProject->getSoilIndex(row,col);
                layer = 1;
                do
                {
                    nodeIndex = long(myProject->indexMap.at(size_t(layer)).value[row][col]);
                    flow = soilFluxes3D::getSumLateralWaterFlowIn(nodeIndex);
                    myProject->outputWaterBalanceMaps->waterInflowMap->value[row][col] += float(flow * 1000); //liters

                    layer++;
                } while (layer < myProject->nrLayers && myProject->isWithinSoil(soilIndex, myProject->layerDepth.at(size_t(layer))));

                nodeIndex = long(myProject->indexMap.at(size_t(--layer)).value[row][col]);

                flow = soilFluxes3D::getBoundaryWaterFlow(nodeIndex); //m3
                flow_mm = flow * 1000 / area;
                myProject->outputWaterBalanceMaps->bottomDrainageMap->value[row][col] -= float(flow_mm);
            }
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
    int horizonIndex = soil::getHorizonIndex(&(myProject->soilList[soilIndex]),
                                          myProject->layerDepth[unsigned(layerIndex)]);

    if (int(horizonIndex) == int(NODATA)) return NODATA;

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
        double signPsiLeaf = - myProject->cultivar->parameterWangLeuning.psiLeaf;         // [kPa]
        // [m^3 m^-3]
        return soil::thetaFromSignPsi(signPsiLeaf, &(myProject->soilList[soilIndex].horizon[horizonIndex]));
    }
    else
    {
        return NODATA;
    }
}


double* getSoilVarProfile(Vine3DProject* myProject, int row, int col, soil::soilVariable myVar)
{
    double* myProfile = static_cast<double*> (calloc(size_t(myProject->nrLayers), sizeof(double)));

    for (int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        myProfile[layerIndex] = NODATA;

    int nodeIndex;
    int soilIndex = myProject->getSoilIndex(row, col);

    for (int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
    {
        nodeIndex = int(myProject->indexMap.at(size_t(layerIndex)).value[row][col]);

        if (nodeIndex != int(myProject->indexMap.at(size_t(layerIndex)).header->flag))
        {
            if (myVar == soil::soilWaterPotentialFC || myVar == soil::soilWaterPotentialWP
                || myVar == soil::soilWaterContentSat  || myVar == soil::soilWaterContentFC
                || myVar == soil::soilWaterContentWP)
            {
                myProfile[layerIndex] = getSoilVar(myProject, soilIndex, layerIndex, myVar);
            }
        }
    }

    return myProfile;
}


double* getCriteria3DVarProfile(Vine3DProject* myProject, int row, int col, criteria3DVariable myVar)
{
    double* myProfile = (double *) calloc(myProject->nrLayers, sizeof(double));

    for (int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        myProfile[layerIndex] = NODATA;

    long nodeIndex, layerIndex;

    for (layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
    {
        nodeIndex = myProject->indexMap.at(layerIndex).value[row][col];

        if (nodeIndex != myProject->indexMap.at(layerIndex).header->flag)
        {
             myProfile[layerIndex] = getCriteria3DVar(myVar, nodeIndex);
        }
    }

    return myProfile;
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
                            waterContent = soilFluxes3D::getWaterContent(nodeIndex);              //[m^3 m^-3]
                            sumWater += waterContent * myProject->layerThickness.at(size_t(layer));
                            wiltingPoint = getSoilVar(myProject, 0, layer, soil::soilWaterContentWP); //[m^3 m^-3]
                            minWater += wiltingPoint * myProject->layerThickness.at(size_t(layer));        //[m]
                            saturation = getSoilVar(myProject, 0, layer, soil::soilWaterContentSat);       //[m^3 m^-3]
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
    int modelCaseIndex;

    for (int row = 0; row < outputMap->header->nrRows; row++)
        for (int col = 0; col < outputMap->header->nrCols; col++)
        {
            //initialize
            outputMap->value[row][col] = outputMap->header->flag;

            if (myProject->indexMap.at(0).value[row][col] != myProject->indexMap.at(0).header->flag)
            {
                sumAWC = 0.0;
                soilIndex = myProject->getSoilIndex(row, col);
                modelCaseIndex = myProject->getModelCaseIndex(row,col);

                if (soilIndex != NODATA)
                {
                    for (int layer = 1; layer < myProject->nrLayers; layer++)
                    {
                        nodeIndex = myProject->indexMap.at(layer).value[row][col];

                        if (nodeIndex != myProject->indexMap.at(layer).header->flag)
                        {
                            if (myProject->grapevine.getRootDensity(&(myProject->modelCases[modelCaseIndex]), layer) > 0.0)
                            {
                                awc = soilFluxes3D::getAvailableWaterContent(nodeIndex);  //[m3 m-3]
                                if (awc != NODATA)
                                {
                                    thickness = myProject->layerThickness[layer] * 1000.0;  //[mm]
                                    horizonIndex = soil::getHorizonIndex(&(myProject->soilList[soilIndex]),
                                                                         myProject->layerDepth[layer]);
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
    if (upperDepth > myProject->soilDepth) return false;
    lowerDepth = MINVALUE(lowerDepth, myProject->soilDepth);

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
                            QString varName, QString notes, QString outputPath, QString myArea)
{
    QString outputFilename = outputPath + getOutputNameDaily(varName, myArea, notes, myDate);
    std::string myErrorString;
    gis::Crit3DRasterGrid* myMap = myProject->outputWaterBalanceMaps->getMapFromVar(myVar);
    if (! gis::writeEsriGrid(outputFilename.toStdString(), myMap, &myErrorString))
    {
         myProject->logError(QString::fromStdString(myErrorString));
         return false;
    }

    return true;
}


bool saveWaterBalanceOutput(Vine3DProject* myProject, QDate myDate, criteria3DVariable myVar,
                            QString varName, QString notes, QString outputPath, QString myArea,
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

    QString outputFilename = outputPath + getOutputNameDaily(varName, myArea, notes, myDate);
    std::string myErrorString;
    if (! gis::writeEsriGrid(outputFilename.toStdString(), myMap, &myErrorString))
    {
         myProject->logError(QString::fromStdString(myErrorString));
         return false;
    }

    myMap->clear();

    return true;
}



bool loadWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString myArea, QString statePath, criteria3DVariable myVar)
{
    std::string myErrorString;
    QString myMapName;

    gis::Crit3DRasterGrid myMap;

    QString myPrefix = getDailyPrefixFromVar(myDate, myArea, myVar);

    for (int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
    {
        myMapName = statePath + myPrefix + QString::number(layerIndex);
        if (! gis::readEsriGrid(myMapName.toStdString(), &myMap, &myErrorString))
        {
            myProject->logError(QString::fromStdString(myErrorString));
            return false;
        }
        else
            if (!setCriteria3DVarMap(layerIndex, myProject, myVar, &myMap))
                return false;
    }
    myMap.clear();
    return true;
}


bool saveWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString myArea, QString statePath, criteria3DVariable myVar)
{
    std::string myErrorString;
    gis::Crit3DRasterGrid* myMap;
    myMap = new gis::Crit3DRasterGrid();
    myMap->initializeGrid(myProject->indexMap.at(0));

    QString myPrefix = getDailyPrefixFromVar(myDate, myArea, myVar);

    for (int layerIndex = 0; layerIndex < myProject->nrLayers; layerIndex++)
        if (getCriteria3DVarMap(myProject, myVar, layerIndex, myMap))
        {
            QString myOutputMapName = statePath + myPrefix + QString::number(layerIndex);
            if (! gis::writeEsriGrid(myOutputMapName.toStdString(), myMap, &myErrorString))
            {
                myProject->logError(QString::fromStdString(myErrorString));
                return false;
            }
        }
    myMap->clear();

    return true;
}


