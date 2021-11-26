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
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include "project3D.h"
#include "soilFluxes3D.h"
#include "soilDbTools.h"
#include "math.h"
#include "utilities.h"
#include <QUuid>
#include <QApplication>


Project3D::Project3D() : Project()
{
    initializeProject3D();
}

void Project3D::initializeProject3D()
{
    soilDbFileName = "";
    cropDbFileName = "";
    soilMapFileName = "";

    // default
    soilDepth = 1.0;            // [m]
    minThickness = 0.02;        // [m]
    maxThickness = 0.1;         // [m]
    thickFactor = 1.5;

    nrSoils = 0;
    nrLayers = 0;
    nrNodes = 0;
    nrLateralLink = 8;

    totalPrecipitation = 0;
    totalEvaporation = 0;
    totalTranspiration = 0;

    setCurrentFrequency(hourly);
}


void Project3D::clearWaterBalance3D()
{
    soilFluxes3D::cleanMemory();
    waterSinkSource.clear();

    layerThickness.clear();
    layerDepth.clear();

    for (unsigned int i = 0; i < indexMap.size(); i++)
    {
        indexMap[i].clear();
    }
    indexMap.clear();

    boundaryMap.clear();
    soilIndexMap.clear();

    isCriteria3DInitialized = false;
}


void Project3D::clearProject3D()
{
    clearWaterBalance3D();

    for (unsigned int i = 0; i < soilList.size(); i++)
    {
        soilList[i].cleanSoil();
    }
    soilList.clear();

    clearProject();
}


bool Project3D::initializeWaterBalance3D()
{
    logInfo("\nInitialize Waterbalance...");
    QString myError;

    // Layers depth
    computeNrLayers();
    setLayersDepth();
    logInfo("nr of layers: " + QString::number(nrLayers));

    // Index map
    if (! setIndexMaps()) return false;
    logInfo("nr of nodes: " + QString::number(nrNodes));

    waterSinkSource.resize(nrNodes);

    // Boundary
    if (!setLateralBoundary()) return false;
    logInfo("Lateral boundary computed");

    // Initiale soil fluxes
    int myResult = soilFluxes3D::initialize(long(nrNodes), int(nrLayers), nrLateralLink, true, false, false);
    if (isCrit3dError(myResult, &myError))
    {
        logError("initializeWaterBalance3D:" + myError);
        return false;
    }
    logInfo("Memory initialized");

    // Set properties for all voxels
    if (! setCrit3DSurfaces()) return false;

    if (! setCrit3DSoils()) return false;
    logInfo("Soils initialized");

    if (! setCrit3DTopography()) return false;
    logInfo("Topology initialized");

    if (! setCrit3DNodeSoil()) return false;
    logInfo("Soils initialized");

    soilFluxes3D::setNumericalParameters(6, 600, 200, 10, 12, 3);   // precision
    //soilFluxes3D::setNumericalParameters(30, 1800, 100, 10, 12, 2);  // speedy
    //soilFluxes3D::setNumericalParameters(300, 3600, 100, 10, 12, 1);   // very speedy (high error)
    soilFluxes3D::setHydraulicProperties(MODIFIEDVANGENUCHTEN, MEAN_LOGARITHMIC, 10.0);

    logInfo("3D water balance initialized");
    return true;
}


bool Project3D::loadSoilDatabase(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing Soil DB filename");
        return false;
    }

    soilDbFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_SOIL);

    if (! loadAllSoils(fileName, &(soilList), texturalClassList, &fittingOptions, &errorString))
    {
        logError();
        return false;
    }
    nrSoils = unsigned(soilList.size());

    logInfo("Soil database = " + fileName);
    return true;
}


bool Project3D::loadCropDatabase(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing Crop DB filename");
        return false;
    }
    cropDbFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_SOIL);

    QSqlDatabase dbCrop;
    dbCrop = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbCrop.setDatabaseName(fileName);

    if (!dbCrop.open())
    {
       logError("Connection with crop database fail");
       return false;
    }

    // TODO: Load crop parameters
    logInfo("Crop database = " + fileName);
    return true;
}


void Project3D::computeNrLayers()
 {
    double nextThickness;
    double prevThickness = minThickness;
    double depth = minThickness * 0.5;

    nrLayers = 1;
    while (depth < soilDepth)
    {
        nextThickness = MINVALUE(maxThickness, prevThickness * thickFactor);
        depth = depth + (prevThickness + nextThickness) * 0.5;
        prevThickness = nextThickness;
        nrLayers++;
    }
}


// set thickness and depth (center) of layers [m]
void Project3D::setLayersDepth()
{
    unsigned int lastLayer = nrLayers-1;
    layerDepth.resize(nrLayers);
    layerThickness.resize(nrLayers);

    layerDepth[0] = 0.0;
    layerThickness[0] = 0.0;
    layerThickness[1] = minThickness;
    layerDepth[1] = minThickness * 0.5;

    for (unsigned int i = 2; i < nrLayers; i++)
    {
        if (i == lastLayer)
        {
            layerThickness[i] = soilDepth - (layerDepth[i-1] + layerThickness[i-1] / 2.0);
        }
        else
        {
            layerThickness[i] = MINVALUE(maxThickness, layerThickness[i-1] * thickFactor);
        }

        layerDepth[i] = layerDepth[i-1] + (layerThickness[i-1] + layerThickness[i]) * 0.5;
    }
}


bool Project3D::setIndexMaps()
{
    // check
    if (!DEM.isLoaded || !soilIndexMap.isLoaded || nrSoils == 0)
    {
        if (!DEM.isLoaded)
            logError("Missing Digital Elevation Model.");
        else if (!soilIndexMap.isLoaded)
            logError("Missing soil map.");
        else if (nrSoils == 0)
            logError("Missing soil properties.");
        return false;
    }

    indexMap.resize(nrLayers);

    unsigned long currentIndex = 0;
    for (unsigned int i = 0; i < nrLayers; i++)
    {
        indexMap.at(i).initializeGrid(DEM);

        for (int row = 0; row < indexMap.at(i).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(i).header->nrCols; col++)
            {
                if (int(DEM.value[row][col]) != int(DEM.header->flag))
                {
                    int soilIndex = getSoilIndex(row, col);
                    if (isWithinSoil(soilIndex, layerDepth.at(i)))
                    {
                        indexMap.at(i).value[row][col] = currentIndex;
                        currentIndex++;
                    }
                }
            }
        }
    }

    nrNodes = currentIndex;
    return (currentIndex > 0);
}


bool Project3D::setLateralBoundary()
{
    if (! this->DEM.isLoaded)
    {
        logError("Missing Digital Elevation Model.");
        return false;
    }

    boundaryMap.initializeGrid(this->DEM);
    for (int row = 0; row < boundaryMap.header->nrRows; row++)
    {
        for (int col = 0; col < boundaryMap.header->nrCols; col++)
        {
            if (gis::isBoundary(this->DEM, row, col))
            {
                // or: if(isMinimum)
                if (! gis::isStrictMaximum(this->DEM, row, col))
                {
                    boundaryMap.value[row][col] = BOUNDARY_RUNOFF;
                }
            }
        }
    }

    return true;
}


bool Project3D::setCrit3DSurfaces()
{
    QString myError;
    int result;

    // TODO: read soilUse parameters
    int nrSurfaces = 1;
    double ManningRoughness = 0.24;         // [s m^-1/3]
    double surfacePond = 0.002;             // [m]

    for (int surfaceIndex = 0; surfaceIndex < nrSurfaces; surfaceIndex++)
    {
        result = soilFluxes3D::setSurfaceProperties(surfaceIndex, ManningRoughness, surfacePond);
        if (isCrit3dError(result, &myError))
        {
            errorString = "setCrit3DSurfaces: " + myError
                           + "\n surface nr:" + QString::number(surfaceIndex);
            return false;
        }
    }

    return true;
}


// ThetaS and ThetaR already corrected for coarse fragments
bool Project3D::setCrit3DSoils()
{
    soil::Crit3DHorizon* myHorizon;
    QString myError;
    int result;

    for (unsigned int soilIndex = 0; soilIndex < nrSoils; soilIndex++)
    {
        for (unsigned int horizIndex = 0; horizIndex < soilList[soilIndex].nrHorizons; horizIndex++)
        {
            myHorizon = &(soilList[soilIndex].horizon[horizIndex]);
            if ((myHorizon->texture.classUSDA > 0) && (myHorizon->texture.classUSDA <= 12))
            {
                result = soilFluxes3D::setSoilProperties(signed(soilIndex), signed(horizIndex),
                     myHorizon->vanGenuchten.alpha * GRAVITY,               // [kPa-1] -> [m-1]
                     myHorizon->vanGenuchten.n,
                     myHorizon->vanGenuchten.m,
                     myHorizon->vanGenuchten.he / GRAVITY,                  // [kPa] -> [m]
                     myHorizon->vanGenuchten.thetaR,
                     myHorizon->vanGenuchten.thetaS,
                     myHorizon->waterConductivity.kSat / DAY_SECONDS / 100, // [cm/d] -> [m/s]
                     myHorizon->waterConductivity.l,
                     myHorizon->organicMatter,
                     double(myHorizon->texture.clay));

                 if (isCrit3dError(result, &myError))
                 {
                     errorString = "setCrit3DSoils: " + myError
                                + "\n soil code: " + QString::fromStdString(soilList[unsigned(soilIndex)].code)
                                + " horizon nr: " + QString::number(horizIndex);
                     return false;
                 }
            }
        }
    }

    return true;
}


bool Project3D::setCrit3DTopography()
{
    double x, y;
    float z, lateralArea, slope;
    double area, volume;
    long index, linkIndex, soilIndex;
    int myResult;
    QString myError;

    for (size_t layer = 0; layer < nrLayers; layer++)
    {
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                index = long(indexMap.at(layer).value[row][col]);

                if (index != long(indexMap.at(layer).header->flag))
                {
                    gis::getUtmXYFromRowCol(DEM, row, col, &x, &y);
                    area = DEM.header->cellSize * DEM.header->cellSize;
                    slope = radiationMaps->slopeMap->value[row][col] / 100;
                    z = DEM.value[row][col] - float(layerDepth[layer]);
                    volume = area * layerThickness[layer];

                    soilIndex = getSoilIndex(row, col);

                    // surface
                    if (layer == 0)
                    {
                        lateralArea = float(DEM.header->cellSize);

                        if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF)
                            myResult = soilFluxes3D::setNode(index, float(x), float(y), z, area, true, true, BOUNDARY_RUNOFF, slope, DEM.header->cellSize);
                        else
                            myResult = soilFluxes3D::setNode(index, float(x), float(y), z, area, true, false, BOUNDARY_NONE, 0, 0);
                    }
                    // sub-surface
                    else
                    {
                        lateralArea = float(DEM.header->cellSize * layerThickness[layer]);

                        // last project layer or last soil layer
                        if (layer == (nrLayers - 1) || ! isWithinSoil(soilIndex, layerDepth.at(size_t(layer+1))))
                            myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, true, BOUNDARY_FREEDRAINAGE, 0, area);
                        else
                        {
                            if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF)
                            {
                                float boundaryArea = DEM.header->cellSize * layerThickness[layer];
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, true, BOUNDARY_FREELATERALDRAINAGE, slope, boundaryArea);
                            }
                            else
                            {
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, false, BOUNDARY_NONE, 0, 0);
                            }
                        }
                    }

                    // check error
                    if (isCrit3dError(myResult, &myError))
                    {
                        errorString = "setTopography:" + myError + " in layer nr:" + QString::number(layer);
                        return(false);
                    }

                    // up link
                    if (layer > 0)
                    {
                        linkIndex = long(indexMap.at(layer - 1).value[row][col]);

                        if (linkIndex != long(indexMap.at(layer - 1).header->flag))
                        {
                            myResult = soilFluxes3D::setNodeLink(index, linkIndex, UP, float(area));
                            if (isCrit3dError(myResult, &myError))
                            {
                                errorString = "setNodeLink:" + myError + " in layer nr:" + QString::number(layer);
                                return(false);
                            }
                        }
                    }

                    // down link
                    if (layer < (nrLayers - 1) && isWithinSoil(soilIndex, layerDepth.at(size_t(layer + 1))))
                    {
                        linkIndex = long(indexMap.at(layer + 1).value[row][col]);

                        if (linkIndex != long(indexMap.at(layer + 1).header->flag))
                        {
                            myResult = soilFluxes3D::setNodeLink(index, linkIndex, DOWN, float(area));

                            if (isCrit3dError(myResult, &myError))
                            {
                                errorString = "setNodeLink:" + myError + " in layer nr:" + QString::number(layer);
                                return(false);
                            }
                        }
                    }

                    // lateral links
                    for (int i=-1; i <= 1; i++)
                    {
                        for (int j=-1; j <= 1; j++)
                        {
                            if ((i != 0)||(j != 0))
                            {
                                if (! gis::isOutOfGridRowCol(row+i, col+j, indexMap.at(layer)))
                                {
                                    linkIndex = long(indexMap.at(layer).value[row+i][col+j]);
                                    if (linkIndex != long(indexMap.at(layer).header->flag))
                                    {
                                        myResult = soilFluxes3D::setNodeLink(index, linkIndex, LATERAL, float(lateralArea / 2));
                                        if (isCrit3dError(myResult, &myError))
                                        {
                                            errorString = "setNodeLink:" + myError + " in layer nr:" + QString::number(layer);
                                            return(false);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

   return true;
}


bool Project3D::setCrit3DNodeSoil()
{
    long index;
    int soilIndex, horizonIndex, myResult;
    QString myError;

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                index = long(indexMap.at(layer).value[row][col]);
                if (index != long(indexMap.at(layer).header->flag))
                {
                    soilIndex = getSoilIndex(row, col);

                    if (layer == 0)
                    {
                        // surface
                        myResult = soilFluxes3D::setNodeSurface(index, 0);
                    }
                    else
                    {
                        // sub-surface
                        horizonIndex = soil::getHorizonIndex(&(soilList[unsigned(soilIndex)]), layerDepth[layer]);
                        if (horizonIndex == NODATA)
                        {
                            logError("function setCrit3DNodeSoil: \nno horizon definition in soil "
                                    + QString::fromStdString(soilList[unsigned(soilIndex)].code) + " depth: " + QString::number(layerDepth[layer])
                                    +"\nCheck soil totalDepth");
                            return false;
                        }

                        myResult = soilFluxes3D::setNodeSoil(index, soilIndex, horizonIndex);

                        // check error
                        if (isCrit3dError(myResult, &myError))
                        {
                            logError("setCrit3DNodeSoil:" + myError + " in soil nr: " + QString::number(soilIndex)
                                    + " horizon nr:" + QString::number(horizonIndex));
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}


bool Project3D::initializeSoilMoisture(int month)
{
    int crit3dResult = CRIT3D_OK;
    QString error;
    long index, soilIndex, horizonIndex;
    double moistureIndex, waterPotential;
    double fieldCapacity;                    // [m]
    double  wiltingPoint = -160.0;           // [m]
    double dry = wiltingPoint / 3.0;         // [m] dry potential

    // [0-1] min: august  max: february
    moistureIndex = fabs(1 - (month - 2) / 6);
    moistureIndex = MAXVALUE(moistureIndex, 0.001);
    moistureIndex = log(moistureIndex) / log(0.001);

    logInfo("Initialize soil moisture");

    for (unsigned int layer = 0; layer < nrLayers; layer++)
    {
        for (int row = 0; row < indexMap.at(size_t(layer)).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(size_t(layer)).header->nrCols; col++)
            {
                index = long(indexMap.at(size_t(layer)).value[row][col]);
                if (index != long(indexMap.at(size_t(layer)).header->flag))
                {
                    if (layer == 0)
                    {
                        // surface
                        soilFluxes3D::setWaterContent(index, 0.0);
                    }
                    else
                    {
                        soilIndex = getSoilIndex(row, col);
                        if (soilIndex != NODATA)
                        {
                            horizonIndex = soil::getHorizonIndex(&(soilList[unsigned(soilIndex)]), layerDepth[size_t(layer)]);
                            if (horizonIndex != NODATA)
                            {
                                fieldCapacity = soilList[unsigned(soilIndex)].horizon[unsigned(horizonIndex)].fieldCapacity;
                                waterPotential = fieldCapacity - moistureIndex * (fieldCapacity-dry);
                                crit3dResult = soilFluxes3D::setMatricPotential(index, waterPotential);
                            }
                        }
                    }

                    if (isCrit3dError(crit3dResult, &error))
                    {
                        logError("initializeSoilMoisture:" + error);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


int Project3D::getSoilIndex(long row, long col)
{
    if ( !soilIndexMap.isLoaded)
        return NODATA;

    int soilIndex = int(soilIndexMap.getValueFromRowCol(row, col));

    if (soilIndex == int(soilIndexMap.header->flag))
        return NODATA;

    return soilIndex;
}


bool Project3D::isWithinSoil(int soilIndex, double depth)
{
    if (soilIndex == int(NODATA) || soilIndex >= int(soilList.size())) return false;

    // check if depth is lower than lowerDepth of last horizon
    unsigned int lastHorizon = soilList[unsigned(soilIndex)].nrHorizons -1;
    double lowerDepth = soilList[unsigned(soilIndex)].horizon[lastHorizon].lowerDepth;

    return (depth <= lowerDepth);
}


// upper depth of soil layer [m]

double Project3D::getSoilLayerTop(unsigned int i)
{
    return layerDepth[i] - layerThickness[i] / 2.0;
}

// lower depth of soil layer [m]
double Project3D::getSoilLayerBottom(unsigned int i)
{
    return layerDepth[i] + layerThickness[i] / 2.0;
}

// index of soil layer for the current depth
int Project3D::getSoilLayerIndex(double depth)
{
    unsigned int i= 0;
    while (depth > getSoilLayerBottom(i))
    {
        if (i == nrLayers-1)
        {
            logError("getSoilLayerIndex: wrong soil depth.");
            return INDEX_ERROR;
        }
        i++;
    }

    return signed(i);
}


bool Project3D::saveHourlyMeteoOutput(meteoVariable myVar, const QString& myPath, QDateTime myTime, const QString& myArea)
{
    gis::Crit3DRasterGrid* myRaster = getHourlyMeteoRaster(myVar);
    if (myRaster == nullptr) return false;

    QString fileName = getOutputNameHourly(myVar, myTime, myArea);
    QString outputFileName = myPath + fileName;

    std::string errStr;
    if (! gis::writeEsriGrid(outputFileName.toStdString(), myRaster, &errStr))
    {
        logError(QString::fromStdString(errStr));
        return false;
    }
    else
        return true;
}


bool Project3D::aggregateAndSaveDailyMap(meteoVariable myVar, aggregationMethod myAggregation, const Crit3DDate& myDate,
                              const QString& dailyPath, const QString& hourlyPath, const QString& myArea)
{
    std::string myError;
    int myTimeStep = int(3600. / meteoSettings->getHourlyIntervals());
    Crit3DTime myTimeIni(myDate, myTimeStep);
    Crit3DTime myTimeFin(myDate.addDays(1), 0.);

    gis::Crit3DRasterGrid* myMap = new gis::Crit3DRasterGrid();
    myMap->initializeGrid(DEM);
    gis::Crit3DRasterGrid* myAggrMap = new gis::Crit3DRasterGrid();
    myAggrMap->initializeGrid(DEM);

    long myRow, myCol;
    int nrAggrMap = 0;

    for (Crit3DTime myTime = myTimeIni; myTime<=myTimeFin; myTime=myTime.addSeconds(myTimeStep))
    {
        QString hourlyFileName = getOutputNameHourly(myVar, getQDateTime(myTime), myArea);
        if (gis::readEsriGrid((hourlyPath + hourlyFileName).toStdString(), myMap, &myError))
        {
            if (myTime == myTimeIni)
            {
                for (myRow = 0; myRow < myAggrMap->header->nrRows; myRow++)
                    for (myCol = 0; myCol < myAggrMap->header->nrCols; myCol++)
                        myAggrMap->value[myRow][myCol] = myMap->value[myRow][myCol];

                nrAggrMap++;
            }
            else
            {
                if (myAggregation == aggrMin)
                    gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationMin);
                else if (myAggregation == aggrMax)
                    gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationMax);
                else if (myAggregation == aggrSum || myAggregation == aggrAverage || myAggregation == aggrIntegral)
                    gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationSum);
                else
                {
                    logError("wrong aggregation type in function 'aggregateAndSaveDailyMap'");
                    return(false);
                }
                nrAggrMap++;
            }
        }
    }

    if (myAggregation == aggrAverage)
        gis::mapAlgebra(myAggrMap, nrAggrMap, myAggrMap, operationDivide);
    else if (myAggregation == aggrSum)
    {
        if (myVar == globalIrradiance || myVar == directIrradiance || myVar == diffuseIrradiance || myVar == reflectedIrradiance)
            gis::mapAlgebra(myAggrMap, float(myTimeStep / 1000000.0), myAggrMap, operationProduct);
    }

    meteoVariable dailyVar = getDailyMeteoVarFromHourly(myVar, myAggregation);
    QString varName = QString::fromStdString(MapDailyMeteoVarToString.at(dailyVar));

    QString filename = getOutputNameDaily(varName, myArea , "", getQDate(myDate));

    QString outputFileName = dailyPath + filename;
    bool isOk = gis::writeEsriGrid(outputFileName.toStdString(), myAggrMap, &myError);

    myMap->clear();
    myAggrMap->clear();

    if (! isOk)
    {
        logError("aggregateMapToDaily: " + QString::fromStdString(myError));
        return false;
    }

    return true;
}


// compute evaporation [mm]
double Project3D::computeEvaporation(int row, int col, double lai)
{
    double depthCoeff, thickCoeff, layerCoeff;
    double residualEvap, layerEvap, availableWater, flow;

    double const MAX_PROF_EVAPORATION = 0.2;           //[m]
    int lastEvapLayer = getSoilLayerIndex(MAX_PROF_EVAPORATION);
    double area = DEM.header->cellSize * DEM.header->cellSize;

    //E0 [mm]
    double et0 = double(hourlyMeteoMaps->mapHourlyET0->value[row][col]);
    double potentialEvaporation = getMaxEvaporation(et0, lai);
    double realEvap = 0;

    for (unsigned int layer=0; layer <= unsigned(lastEvapLayer); layer++)
    {
        long nodeIndex = long(indexMap.at(layer).value[row][col]);

        // layer coefficient
        if (layer == 0)
        {
            // surface: [m] water level
            availableWater = getCriteria3DVar(availableWaterContent, nodeIndex);
            layerCoeff = 1;
        }
        else
        {
            // sub-surface: [m^3 m^-3]
            availableWater = getCriteria3DVar(availableWaterContent, nodeIndex);
            availableWater *= layerThickness[layer];                // [m]

            depthCoeff = layerDepth[layer] / MAX_PROF_EVAPORATION;
            thickCoeff = layerThickness[layer] / 0.04;
            layerCoeff = exp(-EULER * depthCoeff) * thickCoeff;
        }

        // [m]->[mm]
        availableWater *= 1000;

        residualEvap = potentialEvaporation - realEvap;
        layerEvap = MINVALUE(potentialEvaporation * layerCoeff, residualEvap);
        layerEvap = MINVALUE(layerEvap, availableWater);

        if (layerEvap > 0)
        {
            realEvap += layerEvap;
            flow = area * (layerEvap / 1000);                               // [m3/h]
            waterSinkSource.at(unsigned(nodeIndex)) -= (flow / 3600);       // [m3/s]
        }
    }

    return realEvap;
}


// input: timeStep [s]
void Project3D::computeWaterBalance3D(double timeStep)
{
    double previousWaterContent = soilFluxes3D::getTotalWaterContent();
    logInfo("total water [m^3]: " + QString::number(previousWaterContent));

    logInfo("precipitation [m^3]: " + QString::number(totalPrecipitation));
    logInfo("evaporation [m^3]: " + QString::number(-totalEvaporation));
    logInfo("transpiration [m^3]: " + QString::number(-totalTranspiration));

    soilFluxes3D::initializeBalance();

    logInfo("Compute water flow");
    soilFluxes3D::computePeriod(timeStep);

    double runoff = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_RUNOFF);
    logInfo("runoff [m^3]: " + QString::number(runoff));

    double freeDrainage = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_FREEDRAINAGE);
    logInfo("free drainage [m^3]: " + QString::number(freeDrainage));

    double lateralDrainage = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_FREELATERALDRAINAGE);
    logInfo("lateral drainage [m^3]: " + QString::number(lateralDrainage));

    double currentWaterContent = soilFluxes3D::getTotalWaterContent();
    double forecastWaterContent = previousWaterContent + runoff + freeDrainage + lateralDrainage
                                  + totalPrecipitation - totalEvaporation - totalTranspiration;
    double massBalanceError = currentWaterContent - forecastWaterContent;
    logInfo("Mass balance error [m^3]: " + QString::number(massBalanceError));
}


bool Project3D::computeCrop(QDateTime myTime)
{
    logInfo("Compute crop");

    for (long row = 0; row < DEM.header->nrRows ; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                // TODO read crop
                // compute LAI and kc
                // state variables
            }
        }
    }

    return true;
}


bool Project3D::interpolateHourlyMeteoVar(meteoVariable myVar, const QDateTime& myTime)
{
    if (myVar == airRelHumidity && interpolationSettings.getUseDewPoint())
    {
        if (interpolationSettings.getUseInterpolatedTForRH())
            passInterpolatedTemperatureToHumidityPoints(getCrit3DTime(myTime), meteoSettings);

        // TODO check on airTemperatureMap
        if (! interpolationDem(airDewTemperature, getCrit3DTime(myTime), hourlyMeteoMaps->mapHourlyTdew))
            return false;

        if (! hourlyMeteoMaps->computeRelativeHumidityMap(hourlyMeteoMaps->mapHourlyRelHum))
            return false;
    }
    else
    {
        gis::Crit3DRasterGrid* myRaster = getHourlyMeteoRaster(myVar);
        if (myRaster == nullptr) return false;

        if (! interpolationDemMain(myVar, getCrit3DTime(myTime), myRaster))
        {
            QString timeStr = myTime.toString("yyyy-MM-dd hh:mm");
            QString varStr = QString::fromStdString(MapHourlyMeteoVarToString.at(myVar));
            errorString = "Error in interpolation of " + varStr + " at time: " + timeStr;
            return false;
        }
    }

    return true;
}


bool Project3D::interpolateAndSaveHourlyMeteo(meteoVariable myVar, const QDateTime& myTime,
                                              const QString& outputPath, bool isSaveOutput)
{
    if (! interpolateHourlyMeteoVar(myVar, myTime))
        return false;

    if (isSaveOutput)
        return saveHourlyMeteoOutput(myVar, outputPath, myTime, "");
    else
        return true;
}


bool Project3D::computeWaterSinkSource()
{
    long surfaceIndex, nodeIndex;
    double prec, waterSource;
    double flow;
    int myResult;
    QString myError;

    //initialize
    totalPrecipitation = 0;
    totalEvaporation = 0;
    totalTranspiration = 0;

    for (unsigned long i = 0; i < nrNodes; i++)
        waterSinkSource.at(size_t(i)) = 0.0;

    double area = DEM.header->cellSize * DEM.header->cellSize;

    //precipitation - irrigation
    for (long row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (long col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            surfaceIndex = long(indexMap.at(0).value[row][col]);
            if (surfaceIndex != long(indexMap.at(0).header->flag))
            {
                waterSource = 0;
                prec = double(hourlyMeteoMaps->mapHourlyPrec->value[row][col]);
                if (int(prec) != int(hourlyMeteoMaps->mapHourlyPrec->header->flag)) waterSource += prec;

                if (waterSource > 0)
                {
                    flow = area * (waterSource / 1000);                         // [m3/h]
                    totalPrecipitation += flow;
                    waterSinkSource[unsigned(surfaceIndex)] += flow / 3600;     // [m3/s]
                }
            }
        }
    }

    //Evaporation
    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            surfaceIndex = long(indexMap.at(0).value[row][col]);
            if (surfaceIndex != long(indexMap.at(0).header->flag))
            {
                // TODO read LAI
                double lai = 0;

                double realEvap = computeEvaporation(row, col, lai);        // [mm]
                flow = area * (realEvap / 1000.0);                          // [m3/h]
                totalEvaporation += flow;
            }
        }
    }

    //crop transpiration
    for (unsigned int layerIndex=1; layerIndex < nrLayers; layerIndex++)
    {
        for (long row = 0; row < indexMap.at(size_t(layerIndex)).header->nrRows; row++)
        {
            for (long col = 0; col < indexMap.at(size_t(layerIndex)).header->nrCols; col++)
            {
                nodeIndex = long(indexMap.at(size_t(layerIndex)).value[row][col]);
                if (nodeIndex != long(indexMap.at(size_t(layerIndex)).header->flag))
                {
                    // TO DO: transpiration
                    /*
                    float transp = outputPlantMaps->transpirationLayerMaps[layerIndex]->value[row][col];
                    if (int(transp) != int(outputPlantMaps->transpirationLayerMaps[layerIndex]->header->flag))
                    {
                        flow = area * (transp / 1000.0);                            //[m^3/h]
                        totalTranspiration += flow;
                        waterSinkSource.at(unsigned(nodeIndex)) -= flow / 3600.0;   //[m^3/s]
                    }
                    */
                }
            }
        }
    }

    for (unsigned long i = 0; i < nrNodes; i++)
    {
        myResult = soilFluxes3D::setWaterSinkSource(signed(i), waterSinkSource.at(i));
        if (isCrit3dError(myResult, &myError))
        {
            logError("waterBalanceSinkSource:" + myError);
            return false;
        }
    }

    return true;
}





// ------------------------- other functions -------------------------


bool isCrit3dError(int result, QString* error)
{
    if (result == CRIT3D_OK) return false;

    switch (result)
    {
    case INDEX_ERROR:
        *error = "index error";
        break;
    case MEMORY_ERROR:
        *error = "memory error";
        break;
    case TOPOGRAPHY_ERROR:
        *error = "topography error";
        break;
    case BOUNDARY_ERROR:
        *error = "boundary error";
        break;
    case PARAMETER_ERROR:
        *error = "parameter error";
        break;
    default:
        *error = "parameter error";
    }

    return true;
}


double getCriteria3DVar(criteria3DVariable myVar, long nodeIndex)
{
    double crit3dVar;

    if (myVar == waterContent)
    {
        crit3dVar = soilFluxes3D::getWaterContent(nodeIndex);
    }
    else if (myVar == availableWaterContent)
    {
        crit3dVar = soilFluxes3D::getAvailableWaterContent(nodeIndex);
    }
    else if (myVar == waterTotalPotential)
    {
        crit3dVar = soilFluxes3D::getTotalPotential(nodeIndex);
    }
    else if (myVar == waterMatricPotential)
    {
        crit3dVar = soilFluxes3D::getMatricPotential(nodeIndex);
    }
    else if (myVar == degreeOfSaturation)
    {
        crit3dVar = soilFluxes3D::getDegreeOfSaturation(nodeIndex);
    }
    else if (myVar == waterInflow)
    {
        crit3dVar = soilFluxes3D::getSumLateralWaterFlowIn(nodeIndex) * 1000;
    }
    else if (myVar == waterOutflow)
    {
        crit3dVar = soilFluxes3D::getSumLateralWaterFlowOut(nodeIndex) * 1000;
    }
    else if (myVar == waterDeficit)
    {
        // TODO leggere horizon per field capacity
        double fieldCapacity = 3.0;
        crit3dVar = soilFluxes3D::getWaterDeficit(nodeIndex, fieldCapacity);
    }
    else
    {
        crit3dVar = MISSING_DATA_ERROR;
    }

    if (crit3dVar == INDEX_ERROR || crit3dVar == MEMORY_ERROR
        || crit3dVar == MISSING_DATA_ERROR || crit3dVar == TOPOGRAPHY_ERROR) {
        return NODATA;
    }
    else {
        return crit3dVar;
    }
}


bool setCriteria3DVar(criteria3DVariable myVar, long nodeIndex, double myValue)
{
    int myResult = MISSING_DATA_ERROR;

    if (myVar == waterContent)
    {
        // TODO check: skeleton
        myResult = soilFluxes3D::setWaterContent(nodeIndex, myValue);
    }
    else if (myVar == waterMatricPotential)
    {
        myResult = soilFluxes3D::setMatricPotential(nodeIndex, myValue);
    }

    return (myResult != INDEX_ERROR && myResult != MEMORY_ERROR && myResult != MISSING_DATA_ERROR &&
            myResult != TOPOGRAPHY_ERROR);
}


QString getOutputNameDaily(QString varName, QString strArea, QString notes, QDate myDate)
{
    QString myStr = varName;
    if (strArea != "")
        myStr += "_" + strArea;
    if (notes != "")
        myStr += "_" + notes;

    return myStr + "_" + myDate.toString("yyyyMMdd");
}


QString getOutputNameHourly(meteoVariable hourlyVar, QDateTime myTime, QString myArea)
{
    std::string varName = MapHourlyMeteoVarToString.at(hourlyVar);
    QString myStr = QString::fromStdString(varName);
    if (myArea != "")
        myStr += "_" + myArea;

    return myStr + myTime.toString("yyyyMMddThhmm");
}


bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, gis::Crit3DRasterGrid* myGrid)
{
    QString fileName = hourlyPath + getOutputNameHourly(myVar, myTime, myArea);
    std::string error;

    if (gis::readEsriGrid(fileName.toStdString(), myGrid, &error))
        return true;
    else
        return false;
}


float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, int row, int col)
{
    gis::Crit3DRasterGrid* myGrid = new gis::Crit3DRasterGrid();
    QString fileName = hourlyPath + getOutputNameHourly(myVar, myTime, myArea);
    std::string error;

    if (gis::readEsriGrid(fileName.toStdString(), myGrid, &error))
        if (myGrid->value[row][col] != myGrid->header->flag)
            return myGrid->value[row][col];

    return NODATA;
}


QString getDailyPrefixFromVar(QDate myDate, QString myArea, criteria3DVariable myVar)
{
    QString fileName = myDate.toString("yyyyMMdd");
    if (myArea != "")
        fileName += "_" + myArea;

    if (myVar == waterContent)
        fileName += "_WaterContent_";
    if (myVar == availableWaterContent)
        fileName += "_availableWaterContent_";
    else if(myVar == waterMatricPotential)
        fileName += "_MP_";
    else if(myVar == degreeOfSaturation)
        fileName += "_degreeOfSaturation_";
    else
        return "";

    return fileName;
}


double getMaxEvaporation(double ET0, double LAI)
{
    const double ke = 0.6;   //[-] light extinction factor
    const double maxEvaporationRatio = 0.66;

    double Kc = exp(-ke * LAI);
    return(ET0 * Kc * maxEvaporationRatio);
}

