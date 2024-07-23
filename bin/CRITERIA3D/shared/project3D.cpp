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

#include "commonConstants.h"
#include "basicMath.h"
#include "cropDbTools.h"
#include "project3D.h"
#include "soilFluxes3D.h"
#include "soilDbTools.h"

#include "math.h"
#include "utilities.h"
#include "root.h"
#include "gis.h"
#include "soil.h"
#include "meteo.h"

#include <QUuid>
#include <QApplication>
#include <algorithm>


WaterFluxesParameters::WaterFluxesParameters()
{
    initialize();
}

void WaterFluxesParameters::initialize()
{
    freeCatchmentRunoff = true;
    freeLateralDrainage = true;
    freeBottomDrainage = true;

    computeOnlySurface = false;
    computeAllSoilDepth = true;

    initialWaterPotential = -3.0;           // [m] default: field capacity
    imposedComputationDepth = 0.3;          // [m]
    horizVertRatioConductivity = 10.0;      // [-] default: ten times

    modelAccuracy = 3;                      // [-] default: error on the third digit
}


Crit3DProcesses::Crit3DProcesses()
{
    initialize();
}


void Crit3DProcesses::initialize()
{
    computeMeteo = false;
    computeRadiation = false;
    computeWater = false;
    computeSlopeStability = false;
    computeCrop = false;
    computeSnow = false;
    computeSolutes = false;
    computeHeat = false;
    computeAdvectiveHeat = false;
    computeLatentHeat = false;
}

void Crit3DProcesses::setComputeCrop(bool value)
{
    computeCrop = value;

    // prerequisites
    if (computeCrop)
    {
        computeMeteo = true;
        computeRadiation = true;
    }
}

void Crit3DProcesses::setComputeSnow(bool value)
{
    computeSnow = value;

    // prerequisites
    if (computeSnow)
    {
        computeMeteo = true;
        computeRadiation = true;
    }
}

void Crit3DProcesses::setComputeWater(bool value)
{
    computeWater = value;

    // prerequisites
    if (computeWater)
    {
        computeMeteo = true;
        computeRadiation = true;
        computeSlopeStability = true;
    }
}


Project3D::Project3D() : Project()
{
    initializeProject3D();
}


void Project3D::initializeProject3D()
{
    initializeProject();

    isCriteria3DInitialized = false;
    isCropInitialized = false;
    showEachTimeStep = false;
    increaseSlope = false;

    modelPause = false;
    modelStop = false;

    waterFluxesParameters.initialize();

    texturalClassList.resize(13);
    geotechnicsClassList.resize(19);

    soilDbFileName = "";
    cropDbFileName = "";
    soilMapFileName = "";
    landUseMapFileName = "";

    computationSoilDepth = 0.0;     // [m]
    minThickness = 0.02;            // [m] default: 2 cm
    maxThickness = 0.1;             // [m] default: 10 cm
    maxThicknessDepth = 0.40;       // [m] default: 40 cm
    thicknessGrowthFactor = 1.2;    // [-]

    nrSoils = 0;
    nrLayers = 0;
    nrNodes = 0;
    nrLateralLink = 8;

    currentSeconds = 0;
    previousTotalWaterContent = 0;

    totalPrecipitation = 0;
    totalEvaporation = 0;
    totalTranspiration = 0;

    setCurrentFrequency(hourly);
}


void Project3D::clearProject3D()
{
    clearWaterBalance3D();

    for (unsigned int i = 0; i < soilList.size(); i++)
    {
        soilList[i].cleanSoil();
    }
    soilList.clear();

    soilMap.clear();
    landUseMap.clear();

    landUnitList.clear();
    cropList.clear();

    processes.initialize();

    clearProject();
}


void Project3D::clearWaterBalance3D()
{
    soilFluxes3D::cleanMemory();

    layerThickness.clear();
    layerDepth.clear();

    waterSinkSource.clear();

    for (unsigned int i = 0; i < indexMap.size(); i++)
    {
        indexMap[i].clear();
    }
    indexMap.clear();
    soilIndexMap.clear();

    boundaryMap.clear();
    criteria3DMap.clear();

    soilIndexList.clear();

    isCriteria3DInitialized = false;
}


bool Project3D::loadProject3DSettings()
{
    projectSettings->beginGroup("project");

    // soil db
    soilDbFileName = projectSettings->value("soil_db").toString();
    if (soilDbFileName == "")
    {
        soilDbFileName = projectSettings->value("db_soil").toString();
    }

    // crop db
    cropDbFileName = projectSettings->value("crop_db").toString();
    if (cropDbFileName == "")
    {
        cropDbFileName = projectSettings->value("db_crop").toString();
    }

    // soil map
    soilMapFileName = projectSettings->value("soil_map").toString();

    // land use map
    landUseMapFileName = projectSettings->value("landuse_map").toString();
    if (landUseMapFileName.isEmpty())
    {
        // old vine3D version
        landUseMapFileName = projectSettings->value("modelCaseMap").toString();
    }

    projectSettings->endGroup();

    return true;
}


bool Project3D::initialize3DModel()
{
    logInfo("Initialize 3D model...");

    // check soil
    if (! soilMap.isLoaded || soilList.size() == 0)
    {
        logInfo("WARNING: soil map or soil db is missing: only surface fluxes will be computed.");
        waterFluxesParameters.computeOnlySurface = true;
    }

    // check crop
    if (processes.computeCrop)
    {
        if (! landUseMap.isLoaded || landUnitList.empty())
        {
            logWarning("Land use map or crop db is missing.\nCrop computation will be deactivated.");
            processes.setComputeCrop(false);

            // use default crop per surface properties
            landUnitList.clear();
            Crit3DLandUnit deafultLandUnit;
            landUnitList.push_back(deafultLandUnit);
        }
    }

    // set computation depth
    if (waterFluxesParameters.computeOnlySurface)
    {
        computationSoilDepth = 0;
    }
    else
    {
        if (waterFluxesParameters.computeAllSoilDepth)
        {
            for (unsigned int i = 0; i < nrSoils; i++)
            {
                if (soilIndexList.contains(i))
                {
                    computationSoilDepth = std::max(computationSoilDepth, soilList[i].totalDepth);
                }
            }
        }
        else
        {
            computationSoilDepth = waterFluxesParameters.imposedComputationDepth;
        }
    }
    logInfo("Computation depth: " + QString::number(computationSoilDepth) + " m");

    // set layers depth
    setSoilLayers();

    if (! setLayersDepth())
    {
        logError();
        return false;
    }
    logInfo("Nr of layers: " + QString::number(nrLayers));

    // set nr of nodes
    setIndexMaps();
    logInfo("Nr of nodes: " + QString::number(nrNodes));
    if (nrNodes == 0)
    {
        logError("Missing information to assign nodes.");
        return false;
    }

    waterSinkSource.resize(nrNodes);

    // set boundary
    if (! setLateralBoundary()) return false;
    logInfo("Lateral boundary computed");

    // initialize soil fluxes
    int myResult = soilFluxes3D::initialize(long(nrNodes), int(nrLayers), nrLateralLink, true, false, false);
    if (isCrit3dError(myResult, errorString))
    {
        logError("initialize3DModel:" + errorString);
        return false;
    }
    logInfo("Memory initialized");

    // set properties for soil surface (roughness, pond)
    if (! setCrit3DSurfaces())
    {
        logError();
        return false;
    }
    logInfo("Soil surface parameters initialized");

    if (! setCrit3DSoils())
    {
        logError();
        return false;
    }
    if (nrSoils > 0)
    {
        logInfo("Soils parameters initialized");
    }

    if (! setCrit3DTopography())
    {
        logError();
        return false;
    }
    logInfo("Topology initialized");

    if (! setCrit3DNodeSoil())
    {
        logError();
        return false;
    }
    logInfo("Node properties initialized");

    soilFluxes3D::setHydraulicProperties(MODIFIEDVANGENUCHTEN, MEAN_LOGARITHMIC, waterFluxesParameters.horizVertRatioConductivity);

    double vmax = 10.0;                                         // [m s-1]
    if (waterFluxesParameters.modelAccuracy < 3)
        vmax = 5.0;

    double minimumDeltaT = DEM.header->cellSize / vmax;         // [m]

    // Mass Balance Ratio precision (digit at which error is accepted)
    int digitMBR = waterFluxesParameters.modelAccuracy;
    soilFluxes3D::setNumericalParameters(minimumDeltaT, 3600, 100, 10, 12, digitMBR);

    if (! initializeMatricPotential(waterFluxesParameters.initialWaterPotential))  // [m]
    {
        logError();
        return false;
    }

    logInfo("3D water balance initialized");
    return true;
}


bool Project3D::loadSoilMap(QString fileName)
{
    if (fileName.isEmpty())
    {
        logError("Missing soil map");
        return false;
    }

    fileName = getCompleteFileName(fileName, PATH_GEO);

    std::string errorStr;
    if (! gis::openRaster(fileName.toStdString(), &soilMap, gisSettings.utmZone, errorStr))
    {
        logError("Loading soil map failed: " + fileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    gis::updateMinMaxRasterGrid(&(soilMap));

    soilMapFileName = fileName;
    logInfo("Soil map = " + soilMapFileName);

    return true;
}


bool Project3D::setSoilIndexMap()
{
    if (! DEM.isLoaded)
    {
        errorString = "Missing DEM.";
        return false;
    }
    if (! soilMap.isLoaded)
    {
        errorString = "Missing soil map.";
        return false;
    }
    if (soilList.size() == 0)
    {
        errorString = "Missing soil properties.";
        return false;
    }

    double x, y;
    soilIndexMap.initializeGrid(*(DEM.header));
    soilIndexList.clear();

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (! isEqual(DEM.value[row][col], DEM.header->flag))
            {
                DEM.getXY(row, col, x, y);
                int soilIndex = getSoilListIndex(x, y);
                if (soilIndex != NODATA)
                {
                    soilIndexMap.value[row][col] = float(soilIndex);
                    if (! soilIndexList.contains(soilIndex))
                    {
                        soilIndexList.append(soilIndex);
                    }
                }
            }
        }
    }

    soilIndexMap.isLoaded = true;
    return true;
}


void Project3D::setIndexMaps()
{
    indexMap.resize(nrLayers);

    unsigned long currentIndex = 0;
    for (unsigned int layer = 0; layer < nrLayers; layer++)
    {
        indexMap.at(layer).initializeGrid(DEM);

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                if (! isEqual(DEM.value[row][col], DEM.header->flag))
                {
                    bool checkIndex = false;
                    if (layer == 0)
                    {
                        // surface (check only land use)
                        if (getLandUnitIndexRowCol(row, col) != NODATA)
                            checkIndex = true;
                    }
                    else
                    {
                        // sub-surface (check land use and soil)
                        if (getLandUnitIndexRowCol(row, col) != NODATA)
                        {
                            int soilIndex = getSoilIndex(row, col);
                            if (isWithinSoil(soilIndex, layerDepth.at(layer)))
                            {
                                checkIndex = true;
                            }
                        }
                    }

                    if (checkIndex)
                    {
                        indexMap.at(layer).value[row][col] = currentIndex;
                        currentIndex++;
                    }
                }
            }
        }
    }

    nrNodes = currentIndex;
}


bool Project3D::setLateralBoundary()
{
    if (! DEM.isLoaded)
    {
        logError("Missing Digital Elevation Model.");
        return false;
    }

    boundaryMap.initializeGrid(indexMap[0]);
    for (int row = 0; row < boundaryMap.header->nrRows; row++)
    {
        for (int col = 0; col < boundaryMap.header->nrCols; col++)
        {
            if (gis::isBoundaryRunoff(indexMap[0], *(radiationMaps->aspectMap), row, col))
            {
                boundaryMap.value[row][col] = BOUNDARY_RUNOFF;
            }
        }
    }

    return true;
}


bool Project3D::setCrit3DSurfaces()
{
    if (landUnitList.empty())
    {
        logInfo("WARNING! Missing land unit list: default (fallow) will be used.");
        Crit3DLandUnit deafultLandUnit;
        landUnitList.push_back(deafultLandUnit);
    }

    for (int i = 0; i < int(landUnitList.size()); i++)
    {
        int result = soilFluxes3D::setSurfaceProperties(i, landUnitList[i].roughness, landUnitList[i].pond);
        if (isCrit3dError(result, errorString))
        {
            errorString = "Error in setSurfaceProperties: " + errorString + "\n"
                           + "Unit nr:" + QString::number(i);
            return false;
        }
    }

    return true;
}


// thetaS and thetaR are already corrected for coarse fragments
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

                 if (isCrit3dError(result, myError))
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
    float lateralArea;
    long linkIndex, soilIndex;
    int myResult;
    QString myError;

    double area = DEM.header->cellSize * DEM.header->cellSize;

    for (size_t layer = 0; layer < nrLayers; layer++)
    {
        double volume = area * layerThickness[layer];
        if (layer == 0)
            lateralArea = DEM.header->cellSize;
        else
            lateralArea = DEM.header->cellSize * layerThickness[layer];

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = long(indexMap.at(layer).value[row][col]);
                long flag = long(indexMap.at(layer).header->flag);
                if (index != flag)
                {
                    DEM.getXY(row, col, x, y);
                    float slopeDegree = radiationMaps->slopeMap->value[row][col];
                    float boundarySlope = tan(slopeDegree * DEG_TO_RAD);
                    float z = DEM.value[row][col] - float(layerDepth[layer]);

                    soilIndex = getSoilIndex(row, col);

                    if (layer == 0)
                    {
                        // SURFACE
                        if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF && waterFluxesParameters.freeCatchmentRunoff)
                        {
                            float boundaryArea = DEM.header->cellSize;
                            myResult = soilFluxes3D::setNode(index, float(x), float(y), z, area, true, true, BOUNDARY_RUNOFF, boundarySlope, boundaryArea);
                        }
                        else
                        {
                            myResult = soilFluxes3D::setNode(index, float(x), float(y), z, area, true, false, BOUNDARY_NONE, 0, 0);
                        }
                    }
                    else
                    {
                        // LAST SOIL LAYER
                        if (layer == (nrLayers - 1) || ! isWithinSoil(soilIndex, layerDepth.at(size_t(layer+1))))
                        {
                            if (waterFluxesParameters.freeLateralDrainage)
                            {
                                float boundaryArea = area;
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, true, BOUNDARY_FREEDRAINAGE, 0, boundaryArea);
                            }
                            else
                            {
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, false, BOUNDARY_NONE, 0, 0);
                            }
                        }
                        else
                        {
                            // SUB-SURFACE
                            if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF)
                            {
                                float boundaryArea = lateralArea;
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, true, BOUNDARY_FREELATERALDRAINAGE, boundarySlope, boundaryArea);
                            }
                            else
                            {
                                myResult = soilFluxes3D::setNode(index, float(x), float(y), z, volume, false, false, BOUNDARY_NONE, 0, 0);
                            }
                        }
                    }

                    // check error
                    if (isCrit3dError(myResult, myError))
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
                            if (isCrit3dError(myResult, myError))
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

                            if (isCrit3dError(myResult, myError))
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
                                        myResult = soilFluxes3D::setNodeLink(index, linkIndex, LATERAL, lateralArea * 0.5);
                                        if (isCrit3dError(myResult, myError))
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


bool Project3D::initializeMatricPotential(float psi)
{
    long index;
    int myResult;
    QString error;

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                index = long(indexMap.at(layer).value[row][col]);
                if (index != long(indexMap.at(layer).header->flag))
                {
                    if (layer == 0)
                        myResult = soilFluxes3D::setMatricPotential(index, 0);
                    else
                        myResult = soilFluxes3D::setMatricPotential(index, psi);

                    if (isCrit3dError(myResult, error))
                    {
                        errorString = "setMatricPotential: " + error + " in row:"
                                    + QString::number(row) + " col:" + QString::number(col);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


bool Project3D::setCrit3DNodeSoil()
{
    int soilIndex, horizonIndex, myResult;

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = long(indexMap.at(layer).value[row][col]);
                if (index != long(indexMap.at(layer).header->flag))
                {
                    if (layer == 0)
                    {
                        // surface
                        int landUnitIndex = getLandUnitIndexRowCol(row, col);

                        if (landUnitIndex != NODATA)
                            soilFluxes3D::setNodeSurface(index, landUnitIndex);
                        else
                        {
                            errorString = "Wrong surface definition in row, col: "
                                        + QString::number(row) + "," + QString::number(col);
                            return false;
                        }
                    }
                    else
                    {
                        // sub-surface
                        soilIndex = getSoilIndex(row, col);

                        horizonIndex = soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[layer]);
                        if (horizonIndex == NODATA)
                        {
                            errorString = "function setCrit3DNodeSoil:\n No horizon definition in soil "
                                        + QString::fromStdString(soilList[unsigned(soilIndex)].code)
                                        + " depth: " + QString::number(layerDepth[layer])
                                        + "\nCheck soil totalDepth";
                            return false;
                        }

                        myResult = soilFluxes3D::setNodeSoil(index, soilIndex, horizonIndex);

                        // check error
                        if (isCrit3dError(myResult, errorString))
                        {
                            errorString = "setCrit3DNodeSoil:" + errorString + " in soil nr: " + QString::number(soilIndex)
                                            + " horizon nr:" + QString::number(horizonIndex);
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
                            horizonIndex = soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[size_t(layer)]);
                            if (horizonIndex != NODATA)
                            {
                                fieldCapacity = soilList[unsigned(soilIndex)].horizon[unsigned(horizonIndex)].fieldCapacity;
                                waterPotential = fieldCapacity - moistureIndex * (fieldCapacity-dry);
                                crit3dResult = soilFluxes3D::setMatricPotential(index, waterPotential);
                            }
                        }
                    }

                    if (isCrit3dError(crit3dResult, errorString))
                    {
                        logError();
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


/*! \brief runModel
 *  \param totalTimeStep [s]
 */
void Project3D::runModel(double totalTimeStep, bool isRestart)
{
    if (! isRestart)
    {
        currentSeconds = 0;                 // [s]
        soilFluxes3D::initializeBalance();

        previousTotalWaterContent = soilFluxes3D::getTotalWaterContent();       // [m3]

        logInfo("total water [m3]: " + QString::number(previousTotalWaterContent));
        logInfo("precipitation [m3]: " + QString::number(totalPrecipitation));
        logInfo("evaporation [m3]: " + QString::number(-totalEvaporation));
        logInfo("transpiration [m3]: " + QString::number(-totalTranspiration));
        logInfo("Compute water flow...");
    }

    double minimumShowTime = 1;         // [s]
    int currentStep = 0;
    while (currentSeconds < totalTimeStep)
    {
        currentSeconds += soilFluxes3D::computeStep(totalTimeStep - currentSeconds);

        if (modelPause)
        {
            emit updateOutputSignal();
            return;
        }

        if (showEachTimeStep)
        {
            if (currentSeconds < totalTimeStep && int(currentSeconds / minimumShowTime) > currentStep)
            {
                currentStep = int(currentSeconds / minimumShowTime);
                emit updateOutputSignal();
            }
        }
    }

    // refresh
    emit updateOutputSignal();

    double runoff = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_RUNOFF);
    logInfo("runoff [m^3]: " + QString::number(runoff));

    double freeDrainage = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_FREEDRAINAGE);
    logInfo("free drainage [m^3]: " + QString::number(freeDrainage));

    double lateralDrainage = soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_FREELATERALDRAINAGE);
    logInfo("lateral drainage [m^3]: " + QString::number(lateralDrainage));

    double currentWaterContent = soilFluxes3D::getTotalWaterContent();
    double forecastWaterContent = previousTotalWaterContent + runoff + freeDrainage + lateralDrainage
                                  + totalPrecipitation - totalEvaporation - totalTranspiration;
    double massBalanceError = currentWaterContent - forecastWaterContent;
    logInfo("Mass balance error [m^3]: " + QString::number(massBalanceError));
}


// ----------------------------------------- CROP and LAND USE -----------------------------------

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

    // land unit list
    if (! loadLandUnitList(dbCrop, landUnitList, errorString))
    {
       logError("Error in reading land_units table\n" + errorString);
       return false;
    }

    // crop list (same index of landUnitsList)
    cropList.resize(landUnitList.size());
    for (int i = 0; i < int(landUnitList.size()); i++)
    {
        if (landUnitList[i].idCrop == "") continue;

        if (! loadCropParameters(dbCrop, landUnitList[i].idCrop, cropList[i], errorString))
        {
            QString infoStr = "Error in reading crop data: " + landUnitList[i].idCrop;
            logError(infoStr + "\n" + errorString);
            return false;
        }
    }

    logInfo("Crop/landUse database = " + fileName);
    return true;
}


int Project3D::getLandUnitIdUTM(double x, double y)
{
    if (! landUseMap.isLoaded)
        return NODATA;

    int id = int(gis::getValueFromXY(landUseMap, x, y));

    if (id == int(landUseMap.header->flag))
    {
        return NODATA;
    }
    else
    {
        return id;
    }
}


int Project3D::getLandUnitIdGeo(double lat, double lon)
{
    double x, y;
    gis::latLonToUtmForceZone(gisSettings.utmZone, lat, lon, &x, &y);

    return getLandUnitIdUTM(x, y);
}


int Project3D::getLandUnitIndexRowCol(int row, int col)
{
    if (! landUseMap.isLoaded || landUnitList.empty() || landUnitList.size() == 1)
    {
        // default
        return 0;
    }

    double x, y;
    DEM.getXY(row, col, x, y);

    int id = getLandUnitIdUTM(x, y);
    if (id == NODATA)
    {
        return NODATA;
    }

    return getLandUnitIndex(landUnitList, id);
}


// ------------------------------------ SOIL --------------------------------------

bool Project3D::loadSoilDatabase(QString fileName)
{
    if (fileName == "")
    {
        errorString = "Missing Soil DB filename";
        return false;
    }

    soilDbFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_SOIL);

    if (! loadAllSoils(fileName, soilList, texturalClassList, geotechnicsClassList, fittingOptions, errorString))
    {
        return false;
    }
    if (! errorString.isEmpty())
    {
        logWarning();
    }
    nrSoils = unsigned(soilList.size());

    logInfo("Soil database = " + fileName);
    return true;
}


void Project3D::setProgressionFactor()
{
    if (minThickness == maxThickness)
    {
        thicknessGrowthFactor = 1.0;
        return;
    }

    double factor = 1.01;
    double bestFactor = factor;
    double bestError = 99;
    while (factor <= 2.0)
    {
        double currentThickness = minThickness;
        double currentDepth = minThickness * 0.5;
        while (currentThickness < maxThickness)
        {
            double nextThickness = currentThickness * factor;
            currentDepth += (currentThickness + nextThickness) * 0.5;
            currentThickness = nextThickness;
        }

        double upperDepth = currentDepth - currentThickness * 0.5;
        double error = fabs(upperDepth - maxThicknessDepth);
        if (error < bestError)
        {
            bestError = error;
            bestFactor = factor;
        }

        factor += 0.01;
    }

    thicknessGrowthFactor = bestFactor;
}


void Project3D::setSoilLayers()
 {
    nrLayers = 1;
    if (computationSoilDepth <= 0)
        return;

    setProgressionFactor();

    nrLayers++;
    double currentThickness = minThickness;
    double currentDepth = minThickness;

    while ((computationSoilDepth - currentDepth) > minThickness)
    {
        nrLayers++;
        double nextThickness = std::min(maxThickness, currentThickness * thicknessGrowthFactor);
        currentDepth += nextThickness;
        currentThickness = nextThickness;
    }
}


// set thickness and depth (center) of layers [m]
bool Project3D::setLayersDepth()
{
    if (nrLayers == 0)
    {
        errorString = "Soil layers not defined.";
        return false;
    }

    unsigned int lastLayer = nrLayers-1;
    layerDepth.resize(nrLayers);
    layerThickness.resize(nrLayers);

    layerDepth[0] = 0.0;
    layerThickness[0] = 0.0;

    if (nrLayers == 1)
        return true;

    layerThickness[1] = minThickness;
    layerDepth[1] = minThickness * 0.5;
    double currentDepth = minThickness;

    for (unsigned int i = 2; i <= lastLayer; i++)
    {
        if (i == lastLayer)
        {
            layerThickness[i] = computationSoilDepth - currentDepth;
        }
        else
        {
            layerThickness[i] = std::min(maxThickness, layerThickness[i-1] * thicknessGrowthFactor);
        }

        layerDepth[i] = currentDepth + layerThickness[i] * 0.5;
        currentDepth += layerThickness[i];
    }

    return true;
}


int Project3D::getSoilMapId(double x, double y)
{
    if (! soilMap.isLoaded)
        return NODATA;

    float soilMapValue = gis::getValueFromXY(soilMap, x, y);

    if (isEqual(soilMapValue, soilMap.header->flag))
    {
        return NODATA;
    }
    else
    {
        return int(soilMapValue);
    }
}


int Project3D::getSoilListIndex(double x, double y)
{
    int idSoil = getSoilMapId(x, y);

    if (idSoil == NODATA)
        return NODATA;

    for (int index = 0; index < int(soilList.size()); index++)
    {
        if (soilList[index].id == idSoil)
        {
            return index;
        }
    }

    return NODATA;
}


QString Project3D::getSoilCode(double x, double y)
{
    int idSoil = getSoilMapId(x, y);
    if (idSoil == NODATA)
        return "";

    for (unsigned int i = 0; i < soilList.size(); i++)
    {
        if (soilList[i].id == idSoil)
        {
            return QString::fromStdString(soilList[i].code);
        }
    }

    return "NOT FOUND";
}


int Project3D::getSoilIndex(long row, long col)
{
    if (! soilIndexMap.isLoaded)
        return NODATA;

    float soilIndex = soilIndexMap.getValueFromRowCol(row, col);

    if (isEqual(soilIndex,soilIndexMap.header->flag))
    {
        return NODATA;
    }
    else
    {
        return int(soilIndex);
    }
}


bool Project3D::isWithinSoil(int soilIndex, double depth)
{
    if (soilIndex == int(NODATA) || soilIndex >= int(soilList.size()))
        return false;

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

// soil layer index from soil depth
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


// ------------------------------------ INPUT MAP ------------------------------------------

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
                                              const QString& outputPath, bool isSaveOutputRaster)
{
    if (! interpolateHourlyMeteoVar(myVar, myTime))
        return false;

    if (isSaveOutputRaster)
        return saveHourlyMeteoOutput(myVar, outputPath, myTime);
    else
        return true;
}


// ----------------------------------------- OUTPUT MAP ------------------------------------------

bool Project3D::computeCriteria3DMap(gis::Crit3DRasterGrid &outputRaster, criteria3DVariable var, int layerIndex)
{
    if (var == minimumFactorOfSafety)
    {
        return computeMinimumFoS(outputRaster);
    }

    // check layer
    if (layerIndex >= int(indexMap.size()) || layerIndex == NODATA)
    {
        errorString = "Layer is not defined: " + QString::number(layerIndex);
        return false;
    }

    outputRaster.initializeGrid(indexMap.at(layerIndex));

    // compute map
    for (int row = 0; row < indexMap.at(layerIndex).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(layerIndex).header->nrCols; col++)
        {
            long nodeIndex = indexMap.at(layerIndex).value[row][col];
            if (nodeIndex != indexMap.at(layerIndex).header->flag)
            {
                double value;
                if (var == factorOfSafety)
                {
                    value = computeFactorOfSafety(row, col, layerIndex);
                }
                else
                {
                    value = getCriteria3DVar(var, nodeIndex);
                }

                if (value == NODATA)
                {
                    outputRaster.value[row][col] = outputRaster.header->flag;
                }
                else
                {
                    // surface: water height
                    if (var == volumetricWaterContent && layerIndex == 0)
                    {
                        value *= 1000;          // [m] -> [mm]
                    }

                    outputRaster.value[row][col] = value;
                }
            }
            else
            {
                outputRaster.value[row][col] = outputRaster.header->flag;
            }
        }
    }

    gis::updateMinMaxRasterGrid(&outputRaster);
    return true;
}


bool Project3D::computeMinimumFoS(gis::Crit3DRasterGrid &outputRaster)
{
    outputRaster.initializeGrid(indexMap.at(0));

    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            double minimumValue = NODATA;
            for (unsigned int layer = 1; layer < nrLayers; layer++)
            {
                double currentValue = computeFactorOfSafety(row, col, layer);
                if (! isEqual(currentValue, NODATA))
                    if (isEqual(minimumValue, NODATA) || currentValue < minimumValue)
                        minimumValue = currentValue;
            }

            if (! isEqual(minimumValue, NODATA))
            {
                outputRaster.value[row][col] = minimumValue;
            }
        }
    }

    gis::updateMinMaxRasterGrid(&outputRaster);

    return true;
}


bool Project3D::saveHourlyMeteoOutput(meteoVariable myVar, const QString& myPath, QDateTime myTime)
{
    gis::Crit3DRasterGrid* myRaster = getHourlyMeteoRaster(myVar);
    if (myRaster == nullptr) return false;

    QString fileName = getOutputNameHourly(myVar, myTime);
    QString outputFileName = myPath + fileName;

    std::string errStr;
    if (! gis::writeEsriGrid(outputFileName.toStdString(), myRaster, errStr))
    {
        logError(QString::fromStdString(errStr));
        return false;
    }
    else
        return true;
}


bool Project3D::aggregateAndSaveDailyMap(meteoVariable myVar, aggregationMethod myAggregation, const Crit3DDate& myDate,
                              const QString& dailyPath, const QString& hourlyPath)
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
        QString hourlyFileName = getOutputNameHourly(myVar, getQDateTime(myTime));
        if (gis::readEsriGrid((hourlyPath + hourlyFileName).toStdString(), myMap, myError))
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

    QString filename = getOutputNameDaily(varName , "", getQDate(myDate));

    QString outputFileName = dailyPath + filename;
    bool isOk = gis::writeEsriGrid(outputFileName.toStdString(), myAggrMap, myError);

    myMap->clear();
    myAggrMap->clear();

    if (! isOk)
    {
        logError("aggregateMapToDaily: " + QString::fromStdString(myError));
        return false;
    }

    return true;
}



// ----------------------------------  SINK / SOURCE -----------------------------------

bool Project3D::setSinkSource()
{
    for (unsigned long i = 0; i < nrNodes; i++)
    {
        int myResult = soilFluxes3D::setWaterSinkSource(signed(i), waterSinkSource.at(i));
        if (isCrit3dError(myResult, errorString))
        {
            errorString = "waterBalanceSinkSource:" + errorString;
            return false;
        }
    }

    return true;
}

/*! \brief getCoveredSurfaceFraction
 *  \param lai: leaf area index [m2 m-2]
 *  \ref Liangxia Zhang, Zhongmin Hu, Jiangwen Fan, Decheng Zhou & Fengpei Tang, 2014
 *  A meta-analysis of the canopy light extinction coefficient in terrestrial ecosystems
 *  "Cropland had the highest value of K (0.62), followed by broadleaf forest (0.59)
 *  shrubland (0.56), grassland (0.50), and needleleaf forest (0.45)"
 *  \return covered surface fraction [-]
 */
double getCoveredSurfaceFraction(double lai)
{
    if (lai < EPSILON) return 0;

    double k = 0.6;             // [-] light extinction coefficient
    return 1 - exp(-k * lai);
}


/*! \brief getPotentialEvaporation
 *  \param ET0: potential evapo-transpiration [mm]
 *  \param lai: leaf area index [m2 m-2]
 *  \return maximum soil evaporation [mm]
 */
double getPotentialEvaporation(double ET0, double lai)
{
    double evapMax = ET0 * (1.0 - getCoveredSurfaceFraction(lai));
    // TODO check evaporation on free water
    return evapMax * 0.67;
}


/*! \brief getMaxCropTranspiration
 *  \param ET0: potential evapo-transpiration [mm]
 *  \param lai: leaf area index [m2 m-2]
 *  \param maxKc: maximum crop coefficient [-]
 *  \return maximum crop transpiration [mm]
 */
double getPotentialTranspiration(double ET0, double lai, double kcMax)
{
    double covSurfFraction = getCoveredSurfaceFraction(lai);
    double kcFactor = 1 + (kcMax - 1) * covSurfFraction;
    return ET0 * covSurfFraction * kcFactor;
}


/*! \brief assignEvaporation
 *  assign soil evaporation with a decrescent rate from surface to MAX_EVAPORATION_DEPTH
 *  \param row, col
 *  \param lai: leaf area index [m2 m-2]
 *  \return actual evaporation on soil column [mm]
 */
double Project3D::assignEvaporation(int row, int col, double lai, int soilIndex)
{
    int lastEvapLayer;
    if (soilIndex == NODATA)
    {
        // no soil, only surface
        lastEvapLayer = 0;
    }
    else
    {
        if (computationSoilDepth >= MAX_EVAPORATION_DEPTH)
        {
            lastEvapLayer = getSoilLayerIndex(MAX_EVAPORATION_DEPTH);
        }
        else
        {
           lastEvapLayer = getSoilLayerIndex(computationSoilDepth);
        }
    }

    double area = DEM.header->cellSize * DEM.header->cellSize;            // [m2]

    double et0 = double(hourlyMeteoMaps->mapHourlyET0->value[row][col]);        // [mm]
    double maxEvaporation = getPotentialEvaporation(et0, lai);                  // [mm]
    double actualEvaporationSum = 0;                                            // [mm]

    // assigns surface evaporation
    long surfaceNodeIndex = long(indexMap.at(0).value[row][col]);
    // [mm]
    double surfaceWater = getCriteria3DVar(volumetricWaterContent, surfaceNodeIndex) * 1000;
    double surfaceEvaporation = std::min(maxEvaporation, surfaceWater);

    // TODO surface evaporation out of numerical solution
    double surfaceFlow = area * (surfaceEvaporation / 1000);                    // [m3 h-1]
    waterSinkSource.at(unsigned(surfaceNodeIndex)) -= (surfaceFlow / 3600);     // [m3 s-1]

    actualEvaporationSum += surfaceEvaporation;
    double residualEvaporation = maxEvaporation - surfaceEvaporation;
    if (residualEvaporation < EPSILON || lastEvapLayer == 0)
    {
        return actualEvaporationSum;
    }

    // assign layers coefficient
    std::vector<double> evapCoeff;
    std::vector<double> layerCoeff;
    evapCoeff.resize(lastEvapLayer+1);
    layerCoeff.resize(lastEvapLayer+1);

    double coeffSum = 0;
    for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
    {
        double depthCoeff = std::max((layerDepth[layer] - layerDepth[1]) / (MAX_EVAPORATION_DEPTH - layerDepth[1]), 0.0);
        // evaporation coefficient: 1 at depthMin, ~0.1 at MAX_EVAPORATION_DEPTH
        evapCoeff[layer] = exp(-2 * depthCoeff);
        // modify by layer thickness
        layerCoeff[layer] = evapCoeff[layer] * (layerThickness[layer] / 0.04);
        coeffSum += layerCoeff[layer];
    }

    // normalize layer coefficients
    for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
    {
        layerCoeff[layer] /= coeffSum;
    }

    int nrIteration = 0;
    while (residualEvaporation > EPSILON && nrIteration < 3)
    {
        double iterEvaporation = 0;
        for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
        {
            long nodeIndex = long(indexMap.at(layer).value[row][col]);

            int horIndex = soilList[soilIndex].getHorizonIndex(layerDepth[layer]);
            if (horIndex != NODATA)
            {
                soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[horIndex];

                // TODO getHygroscopicHumidity
                double evapThreshold = horizon.waterContentWP + (1 - evapCoeff[layer]) * (horizon.waterContentFC - horizon.waterContentWP) * 0.5;
                double vwcAboveThreshold = std::max(getCriteria3DVar(volumetricWaterContent, nodeIndex) - evapThreshold, 0.0);   // [m3 m-3]

                // [mm]
                double evapAvailableWater = vwcAboveThreshold * layerThickness[layer] * 1000;

                double layerEvap = std::min(evapAvailableWater, residualEvaporation * layerCoeff[layer]);     // [mm]
                if (layerEvap > 0)
                {
                    actualEvaporationSum += layerEvap;
                    iterEvaporation += layerEvap;
                    double flow = area * (layerEvap / 1000);                        // [m3 h-1]
                    waterSinkSource.at(unsigned(nodeIndex)) -= (flow / 3600);       // [m3 s-1]
                }
            }
        }

        residualEvaporation -= iterEvaporation;
        nrIteration++;
    }

    return actualEvaporationSum;
}


/*! \brief assignTranspiration
 *  it computes the actual crop transpiration from the soil root zone
 *  and assigns to waterSinkSource
 *  \param currentLai: leaf area index [m2 m-2]
 *  \param currentDegreeDays: degree days sum [°C]
 *  \return actual transpiration [mm]
 */
double Project3D::assignTranspiration(int row, int col, double currentLai, double currentDegreeDays)
{
    double actualTranspiration = 0;                 // [mm]

    // check lai and degree days
    if (currentLai < EPSILON || isEqual(currentDegreeDays, NODATA))
    {
        return actualTranspiration;
    }

    // only surface
    if (nrLayers <= 1)
    {
        return actualTranspiration;
    }

    // check land unit
    int cropIndex = getLandUnitIndexRowCol(row, col);
    if (cropIndex == NODATA)
    {
        return actualTranspiration;
    }

    // check crop
    if (landUnitList[cropIndex].idCrop.isEmpty())
    {
        return actualTranspiration;
    }
    Crit3DCrop currentCrop = cropList[cropIndex];

    // check soil
    int soilIndex = int(soilIndexMap.value[row][col]);
    if (soilIndex == NODATA)
    {
        return actualTranspiration;
    }

    // maximum transpiration
    double et0 = double(hourlyMeteoMaps->mapHourlyET0->value[row][col]);        // [mm]
    double kcMax = currentCrop.kcMax;                                           // [-]
    double maxTranspiration = getPotentialTranspiration(et0, currentLai, kcMax);

    if (maxTranspiration < EPSILON)
    {
        return actualTranspiration;
    }

    // compute root lenght
    currentCrop.computeRootLength3D(currentDegreeDays, soilList[soilIndex].totalDepth);
    if (currentCrop.roots.currentRootLength <= 0)
    {
        return actualTranspiration;
    }

    // compute root density
    if (! root::computeRootDensity3D(currentCrop, soilList[soilIndex], nrLayers, layerDepth, layerThickness))
    {
        return actualTranspiration;
    }
    // check root layers
    if (currentCrop.roots.firstRootLayer == NODATA || currentCrop.roots.lastRootLayer == NODATA)
    {
        return actualTranspiration;
    }

    // initialize vectors
    std::vector<bool> isLayerStressed;
    std::vector<float> layerTranspiration;
    isLayerStressed.resize(nrLayers);
    layerTranspiration.resize(nrLayers);

    for (unsigned int i = 0; i < nrLayers; i++)
    {
        isLayerStressed[i] = false;
        layerTranspiration[i] = 0;
    }

    // set water surplus stress threshold (0: saturation 1: field capacity)
    double waterSurplusStressFraction = 0.5;                // [-]
    if (currentCrop.isWaterSurplusResistant())
    {
        waterSurplusStressFraction = 0.0;
    }

    double rootDensityWithoutStress = 0.0;                   // [-]
    int firstRootLayer = currentCrop.roots.firstRootLayer;
    int lastRootLayer = currentCrop.roots.lastRootLayer;
    double transpirationSubsetMax = 0;

    for (int layer = firstRootLayer; layer <= lastRootLayer; layer++)
    {
        long nodeIndex = long(indexMap.at(layer).value[row][col]);
        int horizonIndex = soilList[soilIndex].getHorizonIndex(layerDepth[layer]);
        soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[horizonIndex];

        // [m3 m-3]
        double volWaterContent = getCriteria3DVar(volumetricWaterContent, nodeIndex);
        double thetaSat = horizon.vanGenuchten.thetaS;
        double volWaterSurplusThreshold = thetaSat - waterSurplusStressFraction * (thetaSat - horizon.waterContentFC);
        double volWaterScarcityThreshold = horizon.waterContentFC - currentCrop.fRAW * (horizon.waterContentFC - horizon.waterContentWP);

        double ratio;
        if (volWaterContent <= horizon.waterContentWP)
        {
            // NO AVAILABLE WATER
            ratio = 0;
            isLayerStressed[layer] = true;
        }
        else if (volWaterContent < volWaterScarcityThreshold)
        {
            // WATER SCARSITY
            ratio = (volWaterContent - horizon.waterContentWP) / (volWaterScarcityThreshold - horizon.waterContentWP);
            isLayerStressed[layer] = true;
        }
        else if  ((volWaterContent - volWaterSurplusThreshold)  > EPSILON)
        {
            // WATER SURPLUS
            ratio = (thetaSat - volWaterContent) / (thetaSat - volWaterSurplusThreshold);
            isLayerStressed[layer] = true;
        }
        else
        {
            // NORMAL CONDITION
            ratio = 1;
            isLayerStressed[layer] = false;
            rootDensityWithoutStress += currentCrop.roots.rootDensity[layer];
        }

        layerTranspiration[layer] = maxTranspiration * currentCrop.roots.rootDensity[layer] * ratio;
        transpirationSubsetMax += maxTranspiration * currentCrop.roots.rootDensity[layer];
        actualTranspiration += layerTranspiration[layer];
    }

    // WATER STRESS [-]
    // transpirationSubsetMax = maxTranspiration when all soil profile is simulated
    // otherwise it refers to the subset of soil profile considered
    double waterStress = 1 - (actualTranspiration / transpirationSubsetMax);

    // Hydraulic redistribution: movement of water from moist to dry soil through plant roots
    if (waterStress > EPSILON && rootDensityWithoutStress > EPSILON)
    {
        // redistribution acts only on not stressed roots
        double redistribution = transpirationSubsetMax * std::min(waterStress, rootDensityWithoutStress);     // [mm]

        for (int layer = firstRootLayer; layer <= lastRootLayer; layer++)
            if (! isLayerStressed[layer])
                layerTranspiration[layer] += redistribution * (currentCrop.roots.rootDensity[layer] / rootDensityWithoutStress);
    }

    // assigns transpiration to water sink source
    double area = DEM.header->cellSize * DEM.header->cellSize;          // [m2]
    actualTranspiration = 0;
    for (int layer = firstRootLayer; layer <= lastRootLayer; layer++)
    {
        double flow = area * (layerTranspiration[layer] / 1000);        // [m3 h-1]
        long nodeIndex = long(indexMap.at(layer).value[row][col]);
        waterSinkSource.at(nodeIndex) -= (flow / 3600);                 // [m3 s-1]
        actualTranspiration += layerTranspiration[layer];               // [mm]
    }

    return actualTranspiration;
}


float Project3D::computeFactorOfSafety(int row, int col, unsigned int layerIndex)
{
    // check layer
    if (layerIndex >= nrLayers)
    {
        errorString = "Wrong layer nr.: " + QString::number(layerIndex);
        return NODATA;
    }

    // check node
    long nodeIndex = indexMap.at(layerIndex).value[row][col];
    if (nodeIndex == indexMap.at(layerIndex).header->flag)
    {
        return NODATA;
    }

    // check horizon
    int soilIndex = getSoilIndex(row, col);
    int horizonIndex = soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[layerIndex]);
    if (horizonIndex == NODATA)
    {
        return NODATA;
    }

    // slope angle [degrees]
    double slopeDegree = double(radiationMaps->slopeMap->getValueFromRowCol(row, col));
    if (increaseSlope)
    {
        // increase slope (max: 89 degrees)
        slopeDegree = std::min(slopeDegree * 1.5, 89.);
    }
    double slopeAngle = std::max(slopeDegree * DEG_TO_RAD, EPSILON);        // [rad]

    // friction angle [rad]
    double frictionAngle = soilList[unsigned(soilIndex)].horizon[horizonIndex].frictionAngle * DEG_TO_RAD;      // [rad]

    // friction effect [-]
    double tanAngle = std::max(EPSILON, tan(slopeAngle));
    double tanFrictionAngle = tan(frictionAngle);
    double frictionEffect =  tanFrictionAngle / tanAngle;

    // degree of saturation [-]
    double saturationDegree = soilFluxes3D::getDegreeOfSaturation(nodeIndex);
    if (saturationDegree == MEMORY_ERROR || saturationDegree == INDEX_ERROR)
    {
        return NODATA;
    }

    // matric potential (with sign) [kPa]
    double matricPotential = soilFluxes3D::getMatricPotential(nodeIndex) * GRAVITY;

    // suction stress [kPa]
    double suctionStress = matricPotential * saturationDegree;

    // effective cohesion [kPa]
    double effectiveCohesion = soilList[unsigned(soilIndex)].horizon[horizonIndex].effectiveCohesion;

    // unit weight - integration from zero to layerDepth
    double weightSum = 0;                                       // [kPa]
    for (unsigned int layer = 1; layer <= layerIndex; layer++)
    {
        long cuurentNode = indexMap.at(layer).value[row][col];
        if (cuurentNode != indexMap.at(layer).header->flag)
        {
            int currentHorizon = soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[layer]);
            if (currentHorizon != NODATA)
            {
                // [g cm-3] --> [Mg m-3]
                double bulkDensity = soilList[unsigned(soilIndex)].horizon[currentHorizon].bulkDensity;
                double waterContent = soilFluxes3D::getWaterContent(cuurentNode);
                // [kN m-3]
                double unitWeight = (bulkDensity + waterContent) * GRAVITY;
                // [kPa]
                weightSum += unitWeight * layerThickness[layer];
            }
        }
    }

    // TODO root cohesion [kPa] leggere da db e assegnare in base alla ratio di root density
    double rootCohesion = 0.;

    // cohesion effect [-]
    double cohesionEffect = 2 * (effectiveCohesion + rootCohesion) / (weightSum * sin(2*slopeAngle));

    // suction effect [-]
    double suctionEffect = (suctionStress * (tanAngle + 1/tanAngle) * tanFrictionAngle) / weightSum;

    // factor of safety [-]
    return frictionEffect + cohesionEffect - suctionEffect;
}


// ------------------------------ other functions ----------------------------------

bool isCrit3dError(int result, QString& error)
{
    if (result == CRIT3D_OK) return false;

    switch (result)
    {
    case INDEX_ERROR:
        error = "index error";
        break;
    case MEMORY_ERROR:
        error = "memory error";
        break;
    case TOPOGRAPHY_ERROR:
        error = "topography error";
        break;
    case BOUNDARY_ERROR:
        error = "boundary error";
        break;
    case PARAMETER_ERROR:
        error = "parameter error";
        break;
    default:
        error = "parameter error";
    }

    return true;
}


double getCriteria3DVar(criteria3DVariable myVar, long nodeIndex)
{
    double crit3dVar;

    if (myVar == volumetricWaterContent)
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

    if (myVar == volumetricWaterContent)
    {
        // TODO check: skeleton
        myResult = soilFluxes3D::setWaterContent(nodeIndex, myValue);
    }
    else if (myVar == waterMatricPotential)
    {
        myResult = soilFluxes3D::setMatricPotential(nodeIndex, myValue);
    }

    return (myResult != INDEX_ERROR && myResult != MEMORY_ERROR
            && myResult != MISSING_DATA_ERROR && myResult != TOPOGRAPHY_ERROR);
}


QString getOutputNameDaily(QString varName, QString notes, QDate myDate)
{
    QString myStr = varName;

    if (notes != "")
        myStr += "_" + notes;

    return myStr + "_" + myDate.toString("yyyyMMdd");
}


QString getOutputNameHourly(meteoVariable hourlyVar, QDateTime myTime)
{
    std::string varName = MapHourlyMeteoVarToString.at(hourlyVar);
    QString myStr = QString::fromStdString(varName);

    return myStr + myTime.toString("yyyyMMddThhmm");
}


bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, gis::Crit3DRasterGrid* myGrid)
{
    QString fileName = hourlyPath + getOutputNameHourly(myVar, myTime);
    std::string error;

    if (gis::readEsriGrid(fileName.toStdString(), myGrid, error))
        return true;
    else
        return false;
}


float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, int row, int col)
{
    gis::Crit3DRasterGrid* myGrid = new gis::Crit3DRasterGrid();
    QString fileName = hourlyPath + getOutputNameHourly(myVar, myTime);
    std::string error;

    if (gis::readEsriGrid(fileName.toStdString(), myGrid, error))
        if (myGrid->value[row][col] != myGrid->header->flag)
            return myGrid->value[row][col];

    return NODATA;
}


QString getDailyPrefixFromVar(QDate myDate, criteria3DVariable myVar)
{
    QString fileName = myDate.toString("yyyyMMdd");

    if (myVar == volumetricWaterContent)
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

