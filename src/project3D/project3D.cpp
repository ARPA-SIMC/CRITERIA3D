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
#include "project.h"
#include "soilDbTools.h"
#include <float.h>


#include "soilFluxes3D.h"
#include "math.h"
#include "utilities.h"
#include "root.h"
#include "gis.h"
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
    // boundary conditions
    freeCatchmentRunoff = true;
    freeLateralDrainage = true;
    freeBottomDrainage = true;

    // computation soil depth
    computeOnlySurface = false;
    computeAllSoilDepth = true;
    imposedComputationDepth = 1.0;          // [m]

    // soil layers thicknesses progression
    minSoilLayerThickness = 0.02;           // [m] default: 2 cm
    maxSoilLayerThickness = 0.10;           // [m] default: 10 cm
    maxSoilLayerThicknessDepth = 0.40;      // [m] default: 40 cm

    // initial conditions
    isInitialWaterPotential = true;
    initialWaterPotential = -2.0;           // [m] default: field capacity
    initialDegreeOfSaturation = 0.8;        // [-]

    // hydraulic conductivity horizontal / vertical ratio
    conductivityHorizVertRatio = 10.0;      // [-] default: ten times

    modelAccuracy = 3;                      // [-] default: error on the third digit
    numberOfThreads = 4;                    // [-] default: 4 parallel threads
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
    computeCrop = false;
    computeHydrall = false;
    computeSnow = false;
    computeSolutes = false;
    computeHeat = false;
    computeAdvectiveHeat = false;
    computeLatentHeat = false;
}


void Crit3DProcesses::setComputeHydrall(bool value)
{
    computeHydrall = value;

    // prerequisites
    if (computeHydrall)
    {
        computeCrop = true;
        computeWater = true;
        computeMeteo = true;
        computeRadiation = true;
    }
}

void Crit3DProcesses::setComputeRothC(bool value)
{
    computeRothC = value;

    //prerequisites
    if (computeRothC)
    {
        //computeCrop = true;
        //computeWater = true;
        computeMeteo = true;
        computeRadiation = true;
        //computeHydrall = true;
    }
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
    isSnowInitialized = false;
    isRothCInitialized = false;
    isHydrallInitialized = false;

    showEachTimeStep = false;
    increaseSlope = false;

    isModelRunning = false;
    isModelPaused = false;
    isModelStopped = false;

    texturalClassList.resize(13);
    geotechnicsClassList.resize(19);

    soilDbFileName = "";
    cropDbFileName = "";
    soilMapFileName = "";
    landUseMapFileName = "";
    treeCoverMapFileName = "";

    waterFluxesParameters.initialize();

    computationSoilDepth = 0.0;             // [m]
    soilLayerThicknessGrowthFactor = 1.2;   // [-]

    nrSoils = 0;
    nrLayers = 0;
    nrNodes = 0;
    nrSurfaceNodes = 0;
    nrLateralLink = 8;                  // lateral neighbours

    currentSeconds = 0;                 // [s]
    previousTotalWaterContent = 0;      // [m3]
    totalMassBalanceError = 0;          // [m3]

    totalPrecipitation = 0;
    totalEvaporation = 0;
    totalTranspiration = 0;

    // specific outputs
    waterContentDepth.clear();
    degreeOfSaturationDepth.clear();
    waterPotentialDepth.clear();
    factorOfSafetyDepth.clear();

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
    laiMap.clear();
    treeCoverMap.clear();

    landUnitList.clear();
    cropList.clear();

    processes.initialize();

    clearProject();
}


void Project3D::clearWaterBalance3D()
{
    soilFluxes3D::cleanSF3D();

    layerThickness.clear();
    layerDepth.clear();

    waterSinkSource.clear();

    for (unsigned int i = 0; i < indexMap.size(); i++)
    {
        indexMap[i].value.clear();
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
    if (projectSettings == nullptr)
    {
        logError("projectSettings is not loaded");
        return false;
    }

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

    treeCoverMapFileName = projectSettings->value("treecover_map").toString();

    projectSettings->endGroup();

    // output variables (optional)
    projectSettings->beginGroup("output");

    QList<QString> depthList = projectSettings->value("waterContent").toStringList();
    if (! setVariableDepth(depthList, waterContentDepth))
    {
        errorString = "Wrong water content depth in the settings file: " + projectSettings->fileName();
    }

    depthList = projectSettings->value("degreeOfSaturation").toStringList();
    if (! setVariableDepth(depthList, degreeOfSaturationDepth))
    {
        errorString = "Wrong degree of saturation depth in the settings file: " + projectSettings->fileName();
    }

    depthList = projectSettings->value("waterPotential").toStringList();
    if (! setVariableDepth(depthList, waterPotentialDepth))
    {
        errorString = "Wrong water potential depth in the settings file: " + projectSettings->fileName();
    }

    depthList = projectSettings->value("factorOfSafety").toStringList();
    if (! setVariableDepth(depthList, factorOfSafetyDepth))
    {
        errorString = "Wrong factor of safety depth in the settings file: " + projectSettings->fileName();
    }

    projectSettings->endGroup();

    return true;
}


bool Project3D::loadProject3DParameters()
{
    if (parametersSettings == nullptr)
    {
        logError("parameters are not loaded");
        return false;
    }

    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        if (group == "soilWaterFluxes")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("isInitialWaterPotential") && ! parametersSettings->value("isInitialWaterPotential").toString().isEmpty())
            {
                waterFluxesParameters.isInitialWaterPotential = parametersSettings->value("isInitialWaterPotential").toBool();
            }

            if (parametersSettings->contains("initialWaterPotential") && ! parametersSettings->value("initialWaterPotential").toString().isEmpty())
            {
                waterFluxesParameters.initialWaterPotential = parametersSettings->value("initialWaterPotential").toDouble();
            }

            if (parametersSettings->contains("initialDegreeOfSaturation") && ! parametersSettings->value("initialDegreeOfSaturation").toString().isEmpty())
            {
                waterFluxesParameters.initialDegreeOfSaturation = parametersSettings->value("initialDegreeOfSaturation").toDouble();
            }

            if (parametersSettings->contains("computeOnlySurface") && ! parametersSettings->value("computeOnlySurface").toString().isEmpty())
            {
                waterFluxesParameters.computeOnlySurface = parametersSettings->value("computeOnlySurface").toBool();
            }

            if (parametersSettings->contains("computeAllSoilDepth") && ! parametersSettings->value("computeAllSoilDepth").toString().isEmpty())
            {
                waterFluxesParameters.computeAllSoilDepth = parametersSettings->value("computeAllSoilDepth").toBool();
            }

            if (parametersSettings->contains("imposedComputationDepth") && ! parametersSettings->value("imposedComputationDepth").toString().isEmpty())
            {
                waterFluxesParameters.imposedComputationDepth = parametersSettings->value("imposedComputationDepth").toDouble();
            }

            if (parametersSettings->contains("conductivityHorizVertRatio") && ! parametersSettings->value("conductivityHorizVertRatio").toString().isEmpty())
            {
                waterFluxesParameters.conductivityHorizVertRatio = parametersSettings->value("conductivityHorizVertRatio").toDouble();
            }

            if (parametersSettings->contains("freeCatchmentRunoff") && ! parametersSettings->value("freeCatchmentRunoff").toString().isEmpty())
            {
                waterFluxesParameters.freeCatchmentRunoff = parametersSettings->value("freeCatchmentRunoff").toBool();
            }

            if (parametersSettings->contains("freeBottomDrainage") && ! parametersSettings->value("freeBottomDrainage").toString().isEmpty())
            {
                waterFluxesParameters.freeBottomDrainage = parametersSettings->value("freeBottomDrainage").toBool();
            }

            if (parametersSettings->contains("freeLateralDrainage") && ! parametersSettings->value("freeLateralDrainage").toString().isEmpty())
            {
                waterFluxesParameters.freeLateralDrainage = parametersSettings->value("freeLateralDrainage").toBool();
            }

            if (parametersSettings->contains("modelAccuracy") && ! parametersSettings->value("modelAccuracy").toString().isEmpty())
            {
                waterFluxesParameters.modelAccuracy = parametersSettings->value("modelAccuracy").toInt();
            }

            if (parametersSettings->contains("numberOfThreads") && ! parametersSettings->value("numberOfThreads").toString().isEmpty())
            {
                waterFluxesParameters.numberOfThreads = parametersSettings->value("numberOfThreads").toInt();
            }

            parametersSettings->endGroup();
        }

        if (group == "soilCracking")
        {
            parametersSettings->beginGroup(group);

            // TODO parametri soil crack

            parametersSettings->endGroup();
        }
    }

    return true;
}


bool Project3D::initialize3DModel()
{
    logInfo("Initialize 3D model...");

    clearWaterBalance3D();

    // soil
    if (! waterFluxesParameters.computeOnlySurface)
    {
        // it is necessary to reload the soils db (the fitting options may have changed)
        if (! loadSoilDatabase(soilDbFileName))
        {
            logError();
            return false;
        }

        if (! setSoilIndexMap())
        {
            logError();
            return false;
        }
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
        computationSoilDepth = 0.;
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

    // set runoff boundary
    if (! setLateralBoundary()) return false;
    logInfo("Lateral boundary computed");

    // initialize soil fluxes
    auto myResult = soilFluxes3D::initializeSF3D(static_cast<soilFluxes3D::SF3Duint_t>(nrNodes), static_cast<soilFluxes3D::u16_t>(nrLayers), nrLateralLink, true, false, false);
    std::string errorName = "";
    if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
    {
        logError("initializeFluxes:" + QString::fromStdString(errorName));
        return false;
    }
    logInfo("Memory initialized");

    // set properties for soil surface (roughness, pond)
    if (! setCrit3DSurfaces())
    {
        logError();
        return false;
    }
    logInfo("Surface parameters initialized");

    if (! waterFluxesParameters.computeOnlySurface)
    {
        if (! setCrit3DSoils())
        {
            logError();
            return false;
        }
        if (nrSoils > 0)
        {
            logInfo("Soils parameters initialized");
        }
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

    soilFluxes3D::setHydraulicProperties(soilFluxes3D::WRCModel::ModifiedVanGenuchten, soilFluxes3D::meanType_t::Logarithmic, waterFluxesParameters.conductivityHorizVertRatio);

    if (! setAccuracy())
    {
        logError();
        return false;
    }

    if (! initializeWaterContent())
    {
        logError();
        return false;
    }

    if (! initializeEvaporationCoefficient())
    {
        logError();
        return false;
    }

    totalMassBalanceError = 0;      // [m3]
    isCriteria3DInitialized = true;

    logInfo("3D water balance initialized");

    return true;
}


bool Project3D::setAccuracy()
{
    // maximum water velocity
    double vMax = 4 + 4 * waterFluxesParameters.modelAccuracy;                      // [m s-1]

    // minimum dT
    double minimumDeltaT = std::min(30.0, DEM.header->cellSize / vMax);             // [s]

    // Mass Balance Ratio precision (digit at which error is accepted)
    int massBalanceRatioDigit = waterFluxesParameters.modelAccuracy;
    int toleranceDigit = 7 + waterFluxesParameters.modelAccuracy;

    soilFluxes3D::setNumericalParameters(minimumDeltaT, 3600, 100, 10, toleranceDigit, massBalanceRatioDigit);

    // parallel computing
    waterFluxesParameters.numberOfThreads = soilFluxes3D::setThreadsNumber(waterFluxesParameters.numberOfThreads);

    return true;
}


bool Project3D::loadLandUseMap(const QString &fileName)
{
    if (fileName == "")
    {
        logError("Missing land use map filename");
        return false;
    }

    landUseMapFileName = getCompleteFileName(fileName, PATH_GEO);

    std::string errorStr;
    gis::Crit3DRasterGrid raster;
    if (! gis::openRaster(landUseMapFileName.toStdString(), &raster, gisSettings.utmZone, errorStr))
    {
        logError("Load land use map failed: " + landUseMapFileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    gis::resampleGrid(raster, &landUseMap, DEM.header, aggrPrevailing, 0);
    raster.clear();

    logInfo("Land use map = " + landUseMapFileName);
    return true;
}


bool Project3D::loadSoilMap(const QString &fileName)
{
    if (fileName.isEmpty())
    {
        logError("Missing soil map");
        return false;
    }

    soilMapFileName = getCompleteFileName(fileName, PATH_GEO);

    std::string errorStr;
    gis::Crit3DRasterGrid raster;
    if (! gis::openRaster(soilMapFileName.toStdString(), &raster, gisSettings.utmZone, errorStr))
    {
        logError("Loading soil map failed: " + soilMapFileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    gis::resampleGrid(raster, &soilMap, DEM.header, aggrPrevailing, 0);
    raster.clear();

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

    logInfo("Set soil index...");

    double x, y;
    soilIndexMap.initializeGrid(*(DEM.header));
    soilIndexList.clear();

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (isEqual(DEM.value[row][col], DEM.header->flag))
                continue;

            DEM.getXY(row, col, x, y);
            int soilIndex = getSoilListIndex(x, y);
            if (soilIndex == NODATA)
                continue;

            soilIndexMap.value[row][col] = static_cast<float>(soilIndex);
            if (!soilIndexList.contains(soilIndex))
            {
                soilIndexList.append(soilIndex);
            }
        }
    }

    gis::updateMinMaxRasterGrid(&soilIndexMap);
    soilIndexMap.isLoaded = true;
    return true;
}


void Project3D::setIndexMaps()      //Improvable
{
    indexMap.resize(nrLayers);

    nrSurfaceNodes = 0;

    unsigned long currentIndex = 0;
    for (unsigned int layer = 0; layer < nrLayers; layer++)
    {
        long noIndex = static_cast<long>(indexMap.at(layer).header->flag);

        indexMap.at(layer).initializeGrid(*(DEM.header));

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                if (isEqual(DEM.value[row][col], DEM.header->flag))
                    continue;

                bool checkIndex = false;
                if (layer == 0)
                {
                    // surface (check only land use)
                    if (getLandUnitIndexRowCol(row, col) != NODATA)
                    {
                        checkIndex = true;
                        nrSurfaceNodes++;
                    }
                }
                else
                {
                    // sub-surface (check land use and soil)
                    int luIndex = getLandUnitIndexRowCol(row, col);
                    if (luIndex != NODATA)
                    {
                        // ROAD is no soil
                        if (landUnitList.empty() || landUnitList[luIndex].landUseType != LANDUSE_ROAD)
                        {
                            int soilIndex = getSoilIndex(row, col);
                            if (isWithinSoil(soilIndex, layerDepth.at(layer)))
                                checkIndex = true;
                        }
                    }
                }

                if (checkIndex)
                {
                    indexMap.at(layer).value[row][col] = currentIndex;
                    currentIndex++;
                }
                else
                {
                    indexMap.at(layer).value[row][col] = noIndex;
                }
            }
        }
    }

    nrNodes = currentIndex;
}

bool Project3D::loadTreeCoverMap(const QString &fileName)
{
    if (fileName == "")
    {
        logError("Missing tree cover map filename");
        return false;
    }

    treeCoverMapFileName = getCompleteFileName(fileName, PATH_GEO);

    if (! QFile::exists(treeCoverMapFileName))
    {
        logError("The tree cover map doesn't exist: " + treeCoverMapFileName);
        return false;
    }

    std::string errorStr;
    gis::Crit3DRasterGrid raster;
    if (! gis::openRaster(treeCoverMapFileName.toStdString(), &raster, gisSettings.utmZone, errorStr))
    {
        logError("Load tree cover map failed: " + treeCoverMapFileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    gis::resampleGrid(raster, &treeCoverMap, DEM.header, aggrPrevailing, 0);

    logInfo("Tree cover map = " + treeCoverMapFileName);
    return true;
}


bool Project3D::setLateralBoundary()
{
    if (! DEM.isLoaded)
    {
        logError(ERROR_STR_MISSING_DEM);
        return false;
    }

    boundaryMap.initializeGrid(DEM);

    for (int row = 0; row < boundaryMap.header->nrRows; row++)
    {
        for (int col = 0; col < boundaryMap.header->nrCols; col++)
        {
            if (gis::isBoundaryRunoff(indexMap[0], DEM, *(radiationMaps->aspectMap), row, col))
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

    std::string errorName = "";
    for (int i = 0; i < int(landUnitList.size()); i++)
    {
        auto result = soilFluxes3D::setSurfaceProperties(i, landUnitList[i].roughness);

        if(soilFluxes3D::getSF3DerrorName(result, errorName))
        {
            errorString = "Error in setSurfaceProperties: " + QString::fromStdString(errorName) + "\n"
                           + "Unit nr:" + QString::number(i);
            return false;
        }
    }

    return true;
}


// thetaS and thetaR are already corrected for coarse fragments
bool Project3D::setCrit3DSoils()
{
    std::string errorName = "";

    for (unsigned int soilIndex = 0; soilIndex < nrSoils; soilIndex++)
    {
        for (unsigned int horizIndex = 0; horizIndex < soilList[soilIndex].nrHorizons; horizIndex++)
        {
            soil::Crit3DHorizon& myHorizon = soilList[soilIndex].horizon[horizIndex];
            if ((myHorizon.texture.classUSDA <= 0) || (myHorizon.texture.classUSDA > 12))
                continue;

            auto result = soilFluxes3D::setSoilProperties(static_cast<std::uint16_t>(soilIndex), static_cast<std::uint8_t>(horizIndex),
                                                             myHorizon.vanGenuchten.alpha * GRAVITY,                           // [kPa-1] -> [m-1]
                                                             myHorizon.vanGenuchten.n,
                                                             myHorizon.vanGenuchten.m,
                                                             myHorizon.vanGenuchten.he / GRAVITY,                              // [kPa] -> [m]
                                                             myHorizon.vanGenuchten.thetaR * myHorizon.getSoilFraction(),
                                                             myHorizon.vanGenuchten.thetaS * myHorizon.getSoilFraction(),
                                                             (myHorizon.waterConductivity.kSat * 0.01) / DAY_SECONDS,           // [cm/d] -> [m/s]
                                                             myHorizon.waterConductivity.l,
                                                             myHorizon.organicMatter,
                                                             static_cast<double>(myHorizon.texture.clay));

            if(soilFluxes3D::getSF3DerrorName(result, errorName))
            {
                errorString = "setCrit3DSoils: " + QString::fromStdString(errorName)
                                + "\n soil code: " + QString::fromStdString(soilList[unsigned(soilIndex)].code)
                                + " horizon nr: " + QString::number(horizIndex);
                return false;
            }
        }
    }

    return true;
}


bool Project3D::setCrit3DTopography()
{
    double area = DEM.header->cellSize * DEM.header->cellSize;
    std::string errorName = "";

    for (size_t layer = 0; layer < nrLayers; layer++)
    {
        double volume = area * layerThickness[layer];
        float lateralArea;
        long noIndex = static_cast<long>(indexMap.at(layer).header->flag);

        lateralArea = (layer == 0) ? DEM.header->cellSize : DEM.header->cellSize * layerThickness[layer];

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = static_cast<long>(indexMap.at(layer).value[row][col]);
                if (index == noIndex)
                    continue;

                double x, y;
                DEM.getXY(row, col, x, y);
                float slopeDegree = radiationMaps->slopeMap->value[row][col];
                float boundarySlope = tan(slopeDegree * DEG_TO_RAD);
                float z = DEM.value[row][col] - float(layerDepth[layer]);

                int soilIndex = getSoilIndex(row, col);
                auto myResult = soilFluxes3D::SF3Derror_t::SF3Dok;

                if (layer == 0)
                {
                    // SURFACE
                    if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF && waterFluxesParameters.freeCatchmentRunoff)
                    {
                        float boundaryArea = DEM.header->cellSize;
                        myResult = soilFluxes3D::setNode(index, x, y, z, area, true, soilFluxes3D::boundaryType_t::Runoff,
                                                         boundarySlope, boundaryArea);
                    }
                    else
                    {
                        myResult = soilFluxes3D::setNode(index, x, y, z, area, true, soilFluxes3D::boundaryType_t::NoBoundary);
                    }
                }
                else
                {
                    // LAST SOIL LAYER
                    if (layer == (nrLayers - 1) || ! isWithinSoil(soilIndex, layerDepth.at(size_t(layer+1))))
                    {
                        if (waterFluxesParameters.freeBottomDrainage)
                        {
                            float boundaryArea = area;
                            myResult = soilFluxes3D::setNode(index, x, y, z, volume, false,
                                                             soilFluxes3D::boundaryType_t::FreeDrainage, 0, boundaryArea);
                        }
                        else
                        {
                            myResult = soilFluxes3D::setNode(index, x, y, z, volume, false, soilFluxes3D::boundaryType_t::NoBoundary);
                        }
                    }
                    else
                    {
                        // SUB-SURFACE
                        if (int(boundaryMap.value[row][col]) == BOUNDARY_RUNOFF && waterFluxesParameters.freeLateralDrainage)
                        {
                            // TODO problema se è urban o road
                            float boundaryArea = lateralArea;
                            myResult = soilFluxes3D::setNode(index, x, y, z, volume, false, soilFluxes3D::boundaryType_t::FreeLateraleDrainage,
                                                             boundarySlope, boundaryArea);
                        }
                        else
                        {
                            auto boundaryType = soilFluxes3D::boundaryType_t::NoBoundary;

                            // check on urban or road land use
                            if (layer == 1 && ! landUnitList.empty())
                            {
                                int luIndex = getLandUnitIndexRowCol(row, col);
                                if (landUnitList[luIndex].landUseType == LANDUSE_ROAD)
                                {
                                    boundaryType = soilFluxes3D::boundaryType_t::Road;
                                }
                                if (landUnitList[luIndex].landUseType == LANDUSE_URBAN)
                                {
                                    boundaryType = soilFluxes3D::boundaryType_t::Urban;
                                }
                            }

                            myResult = soilFluxes3D::setNode(index, x, y, z, volume, false, boundaryType);
                        }
                    }
                }

                // check error
                if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                {
                    errorString = "setTopography:" + QString::fromStdString(errorName) + " in layer nr:" + QString::number(layer);
                    return false;
                }

                // up link
                if (layer > 0)
                {
                    long linkIndex = indexMap.at(layer - 1).value[row][col];
                    if (linkIndex == noIndex)
                        continue;

                    myResult = soilFluxes3D::setNodeLink(index, linkIndex, soilFluxes3D::linkType_t::Up, area);

                    if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                    {
                        errorString = "setNodeLink (up):" + QString::fromStdString(errorName) + " in layer nr:" + QString::number(layer);
                        return false;
                    }
                }

                // down link
                if (layer < (nrLayers - 1) && isWithinSoil(soilIndex, layerDepth.at(size_t(layer + 1))))
                {
                    long linkIndex = indexMap.at(layer + 1).value[row][col];
                    if (linkIndex == noIndex)
                        continue;

                    myResult = soilFluxes3D::setNodeLink(index, linkIndex, soilFluxes3D::linkType_t::Down, area);
                    if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                    {
                        errorString = "setNodeLink (down):" + QString::fromStdString(errorName)+ " in layer nr:" + QString::number(layer);
                        return false;
                    }
                }

                // lateral links
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        if((i == 0) && (j == 0))
                            continue;

                        if(indexMap.at(layer).isOutOfGrid(row+i, col+j))
                            continue;

                        long linkIndex = indexMap.at(layer).value[row+i][col+j];
                        if (linkIndex == noIndex)
                            continue;

                        // eight lateral nodes: each is assigned half a side (conceptual octagon)
                        myResult = soilFluxes3D::setNodeLink(index, linkIndex, soilFluxes3D::linkType_t::Lateral, lateralArea * 0.5);

                        if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                        {
                            errorString = "setNodeLink (lateral):" + QString::fromStdString(errorName)
                                            + " in layer nr:" + QString::number(layer);
                            return false;
                        }
                    }
                }
            }
        }
    }

   return true;
}


bool Project3D::initializeWaterContent()
{
    auto myResult = soilFluxes3D::SF3Derror_t::SF3Dok;
    std::string errorName = "";

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        long noIndex = static_cast<long>(indexMap.at(layer).header->flag);

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = indexMap.at(layer).value[row][col];
                if (index == noIndex)
                    continue;

                if (layer == 0)
                {
                    // surface
                    if (waterFluxesParameters.isInitialWaterPotential && waterFluxesParameters.initialWaterPotential > 0)
                    {
                        myResult = soilFluxes3D::setNodeMatricPotential(index, waterFluxesParameters.initialWaterPotential);

                    }
                    else
                    {
                        myResult = soilFluxes3D::setNodeMatricPotential(index, 0);
                    }
                }
                else
                {
                    // sub-surface
                    if (waterFluxesParameters.isInitialWaterPotential)
                    {
                        myResult = soilFluxes3D::setNodeMatricPotential(index, waterFluxesParameters.initialWaterPotential);
                    }
                    else
                    {
                        myResult = soilFluxes3D::setNodeDegreeOfSaturation(index, waterFluxesParameters.initialDegreeOfSaturation);
                    }
                }

                if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                {
                    errorString = "Function initializeWaterContent: " + QString::fromStdString(errorName) + "\n";
                    errorString += "In row:" + QString::number(row) + " col:" + QString::number(col);
                    return false;
                }
            }
        }
    }

    return true;
}


// assigns the soil to the nodes
bool Project3D::setCrit3DNodeSoil()
{
    int soilIndex, horizonIndex;
    auto myResult = soilFluxes3D::SF3Derror_t::SF3Dok;
    std::string errorName = "";

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        long noIndex = static_cast<long>(indexMap.at(layer).header->flag);

        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = indexMap.at(layer).value[row][col];
                if (index == noIndex)
                    continue;

                if (layer == 0)
                {
                    // surface
                    int unitIndex = getLandUnitIndexRowCol(row, col);

                    if (unitIndex != NODATA)
                    {
                        soilFluxes3D::setNodeSurface(index, unitIndex);

                        float currentPond = computeCurrentPond(row, col);
                        if (!isEqual(currentPond, NODATA))
                        {
                            soilFluxes3D::setNodePond(index, currentPond);
                        }
                        else
                        {
                            soilFluxes3D::setNodePond(index, landUnitList[unitIndex].pond);
                        }
                    }
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
                    if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                    {
                        errorString = "setCrit3DNodeSoil:" + QString::fromStdString(errorName) + " in soil nr: " + QString::number(soilIndex)
                                        + " horizon nr:" + QString::number(horizonIndex);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


bool Project3D::initializeSoilMoisture(int month)
{
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
    std::string errorName = "";

    for (unsigned int layer = 0; layer < nrLayers; layer++)
    {
        long noIndex = static_cast<long>(indexMap.at(layer).header->flag);

        for (int row = 0; row < indexMap.at(size_t(layer)).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(size_t(layer)).header->nrCols; col++)
            {
                index = indexMap.at(layer).value[row][col];
                if (index == noIndex)
                    continue;

                auto crit3dResult = soilFluxes3D::SF3Derror_t::SF3Dok;

                if (layer == 0)
                {
                    // surface
                    soilFluxes3D::setNodeWaterContent(index, 0.0);
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
                            crit3dResult = soilFluxes3D::setNodeMatricPotential(index, waterPotential);
                        }
                    }
                }

                if(soilFluxes3D::getSF3DerrorName(crit3dResult, errorName))
                {
                    logError();
                    return false;
                }
            }
        }
    }

    return true;
}


/*! \brief runWaterFluxes3DModel
 *  \param totalTimeStep [s]
 */
void Project3D::runWaterFluxes3DModel(double totalTimeStep, bool isRestart)
{
    if (!isRestart)
    {
        currentSeconds = 0;                                 // [s]
        soilFluxes3D::initializeBalance();

        previousTotalWaterContent = soilFluxes3D::getTotalWaterContent();

        logInfo("total water [m3]: " + QString::number(previousTotalWaterContent));
        if (processes.computeSnow)
			logInfo("precipitation/snowmelt [m3]: " + QString::number(totalPrecipitation));
        else
			logInfo("precipitation [m3]: " + QString::number(totalPrecipitation));
        logInfo("evaporation [m3]: " + QString::number(-totalEvaporation));
        logInfo("transpiration [m3]: " + QString::number(-totalTranspiration));
        logInfo("Compute water flow...");
    }

    double minimumShowTime = 0.5;         // [s]
    int lastShowStep = int(currentSeconds / minimumShowTime);
    double previuosSeconds = currentSeconds;

    while (currentSeconds < totalTimeStep)
    {
        currentSeconds += soilFluxes3D::computeStep(totalTimeStep - currentSeconds);

        if ((isModelPaused || isModelStopped) && currentSeconds < totalTimeStep)
        {
            if (modality == MODE_GUI)
                emit updateOutputSignal();
            return;
        }

        if (modality == MODE_GUI && showEachTimeStep)
        {
            int currentStep = int(currentSeconds / minimumShowTime);
            if (currentSeconds < totalTimeStep && currentStep > lastShowStep)
            {
                lastShowStep = currentStep;
                emit updateOutputSignal();
            }
        }

        if (currentSeconds < (totalTimeStep-300) && (currentSeconds - previuosSeconds) >= 600)
        {
            int minutes = int(currentSeconds / 60.);
            double seconds = currentSeconds - (minutes * 60);
            logInfo(QDateTime::currentDateTime().toString(Qt::ISODate)
                    + " minutes: " + QString::number(minutes) + "::" + QString::number(seconds));
            previuosSeconds = currentSeconds;
        }
    }

    // refresh
    if (modality == MODE_GUI)
        emit updateOutputSignal();

    double runoff = soilFluxes3D::getTotalBoundaryWaterFlow(soilFluxes3D::boundaryType_t::Runoff);
    logInfo("runoff [m3]: " + QString::number(runoff));

    double freeDrainage = soilFluxes3D::getTotalBoundaryWaterFlow(soilFluxes3D::boundaryType_t::FreeDrainage);
    logInfo("free drainage [m3]: " + QString::number(freeDrainage));

    double lateralDrainage = soilFluxes3D::getTotalBoundaryWaterFlow(soilFluxes3D::boundaryType_t::FreeLateraleDrainage);
    logInfo("lateral drainage [m3]: " + QString::number(lateralDrainage));

    double forecastWaterContent = previousTotalWaterContent + runoff + freeDrainage + lateralDrainage
                                  + totalPrecipitation - totalEvaporation - totalTranspiration;
    double currentWaterContent = soilFluxes3D::getTotalWaterContent();
    double massBalanceError = currentWaterContent - forecastWaterContent;

    logInfo("Mass balance error [m3]: " + QString::number(massBalanceError));
	double surfaceArea = DEM.header->cellSize * DEM.header->cellSize * nrSurfaceNodes;      // [m2]
    double error_mm = massBalanceError / surfaceArea * 1000;                                // [mm]
    logInfo("Mass balance error [mm]: " + QString::number(error_mm));

    totalMassBalanceError += massBalanceError;
    logInfo("Total mass balance error [m3]: " + QString::number(totalMassBalanceError));
}


// ----------------------------------------- CROP and LAND USE -----------------------------------

bool Project3D::loadCropDatabase(const QString &fileName)
{
    if (fileName == "")
    {
        logError("Missing Crop DB fileName");
        return false;
    }

    cropDbFileName = getCompleteFileName(fileName, PATH_SOIL);

    if (! QFile::exists(cropDbFileName))
    {
        logError("The crop DB doesn't exist: " + cropDbFileName);
        return false;
    }

    QSqlDatabase dbCrop;
    dbCrop = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbCrop.setDatabaseName(cropDbFileName);

    if (! dbCrop.open())
    {
       logError("Connection with crop database fail");
       return false;
    }

    // land unit list
    errorString = "";
    if (! loadLandUnitList(dbCrop, landUnitList, errorString))
    {
       logError("Error in reading land_units table\n" + errorString);
       return false;
    }
    else if (errorString != "")
    {
        logWarning("Warning in the land_units table of crop db: " + errorString + "\nThe default land use (FALLOW) will be used.");
        errorString = "";
    }

    // crop list (same index of landUnitsList)
    cropList.resize(landUnitList.size());
    for (int i = 0; i < int(landUnitList.size()); i++)
    {
        if (landUnitList[i].idCrop.isEmpty()) continue;

        if (! loadCropParameters(dbCrop, landUnitList[i].idCrop, cropList[i], errorString))
        {
            QString infoStr = "Error in reading crop data: " + landUnitList[i].idCrop;
            logError(infoStr + "\n" + errorString);
            return false;
        }
    }

    logInfo("Crop/landUse database = " + cropDbFileName);
    return true;
}


int Project3D::getLandUnitFromUtm(double x, double y)
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

    return getLandUnitFromUtm(x, y);
}


int Project3D::getLandUnitIndexRowCol(int row, int col)
{
    if (! landUseMap.isLoaded || landUnitList.empty() || landUnitList.size() == 1)
    {
        // default
        return 0;
    }

    // landuse map has same header of DEM
    int id = landUseMap.value[row][col];
    if (id == landUseMap.header->flag)
    {
        return NODATA;
    }

    return getLandUnitListIndex(id);
}


int Project3D::getTreeCoverIndexRowCol(int row, int col)
{
    if (! treeCoverMap.isLoaded)
    {
        // default
        return 0;
    }

    // treeCover map has same header of DEM
    int id = treeCoverMap.value[row][col];
    if (id == treeCoverMap.header->flag)
    {
        return NODATA;
    }

    return id;
}


int Project3D::getLandUnitListIndex(int id)
{
    for (int index = 0; index < int(landUnitList.size()); index++)
    {
        if (landUnitList[index].id == id)
            return index;
    }

    return NODATA;
}


bool Project3D::isCrop(int unitIndex)
{
    if (unitIndex == NODATA)
        return false;

    QString idCrop = landUnitList[unitIndex].idCrop.toUpper();

    if (idCrop.isEmpty() || idCrop == "BARE")
        return false;

    return true;
}


// ------------------------------------ SOIL --------------------------------------

bool Project3D::loadSoilDatabase(const QString &fileName)
{
    if (fileName == "")
    {
        errorString = "Missing Soil DB fileName";
        return false;
    }

    soilDbFileName = getCompleteFileName(fileName, PATH_SOIL);

    if (!loadAllSoils(soilDbFileName, soilList, texturalClassList, geotechnicsClassList, fittingOptions, errorString))
    {
        return false;
    }

    if (! errorString.isEmpty())
    {
        logWarning();
    }
    nrSoils = unsigned(soilList.size());

    logInfo("Soil database = " + soilDbFileName);
    return true;
}


void Project3D::setSoilLayers()
 {
    nrLayers = 1;
    if (computationSoilDepth <= 0)
        return;

    // set thicknessGrowthFactor
    if (waterFluxesParameters.minSoilLayerThickness == waterFluxesParameters.maxSoilLayerThickness)
    {
        soilLayerThicknessGrowthFactor = 1.0;
    }
    else
    {
        double factor = 1.01;
        double bestFactor = factor;
        double bestError = 99;
        while (factor <= 2.0)
        {
            double upperDepth = 0;
            double currentThickness = waterFluxesParameters.minSoilLayerThickness;
            double currentDepth = upperDepth + currentThickness * 0.5;
            while (currentThickness < waterFluxesParameters.maxSoilLayerThickness)
            {
                upperDepth += currentThickness;
                currentThickness = std::min(currentThickness * factor, waterFluxesParameters.maxSoilLayerThickness);
                currentDepth = upperDepth + currentThickness * 0.5;
            }

            double error = fabs(currentDepth - waterFluxesParameters.maxSoilLayerThicknessDepth);
            if (error < bestError)
            {
                bestError = error;
                bestFactor = factor;
            }

            factor += 0.01;
        }
        soilLayerThicknessGrowthFactor = bestFactor;
    }

    nrLayers++;
    double currentThickness = waterFluxesParameters.minSoilLayerThickness;
    double currentLowerDepth = waterFluxesParameters.minSoilLayerThickness;

    while ((computationSoilDepth - currentLowerDepth) > waterFluxesParameters.minSoilLayerThickness)
    {
        nrLayers++;
        double nextThickness = std::min(currentThickness * soilLayerThicknessGrowthFactor, waterFluxesParameters.maxSoilLayerThickness);
        currentLowerDepth += nextThickness;
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

    layerThickness[1] = waterFluxesParameters.minSoilLayerThickness;
    layerDepth[1] = waterFluxesParameters.minSoilLayerThickness * 0.5;
    double currentDepth = waterFluxesParameters.minSoilLayerThickness;

    for (unsigned int i = 2; i <= lastLayer; i++)
    {
        if (i == lastLayer)
        {
            layerThickness[i] = computationSoilDepth - currentDepth;
        }
        else
        {
            layerThickness[i] = std::min(waterFluxesParameters.maxSoilLayerThickness, layerThickness[i-1] * soilLayerThicknessGrowthFactor);
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


// soil layer index from soildepth [m]
int Project3D::getSoilLayerIndex(double depth)
{
    unsigned int layer = 0;
    while (depth > getSoilLayerBottom(layer))
    {
        if (layer == nrLayers-1)
        {
            errorString = "Wrong soil depth.";
            return NODATA;
        }
        layer++;
    }

    return layer;
}


float Project3D::computeCurrentPond(int row, int col)
{
    if (! radiationMaps->slopeMap->isLoaded)
        return NODATA;

    float slopeDegree = radiationMaps->slopeMap->getValueFromRowCol(row, col);

    if (isEqual(slopeDegree, radiationMaps->slopeMap->header->flag))
        return NODATA;

    double slope = tan(slopeDegree * DEG_TO_RAD);

    int unitIndex = getLandUnitIndexRowCol(row, col);
    if (unitIndex == NODATA)
        return NODATA;

    double maximumPond = landUnitList[unitIndex].pond;
    double soilMaximumPond = maximumPond / (slope + 1.);

    double currentInterception = 0;
    if (processes.computeCrop)
    {
        if (isCrop(unitIndex))
        {
            // distributes the pond between the soil and the interception
            soilMaximumPond *= 0.5;

            // landUnit list and crop list have the same index
            float currentLai = laiMap.value[row][col];
            if (! isEqual(currentLai, laiMap.header->flag))
            {
                currentInterception = maximumPond * 0.5 * (currentLai / cropList[unitIndex].LAImax);
            }
        }
    }

    return float(soilMaximumPond + currentInterception);
}


bool Project3D::dailyUpdatePond()
{
    if (! this->isCriteria3DInitialized || indexMap.empty() )
    {
        return false;
    }

    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            long nodeIndex = indexMap.at(0).value[row][col];
            if (nodeIndex == static_cast<long>(indexMap.at(0).header->flag))
                continue;

            float pond = computeCurrentPond(row, col);
            if (isEqual(pond, NODATA))
                continue;

            soilFluxes3D::setNodePond(nodeIndex, pond);
        }
    }

    return true;
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
        if (myRaster == nullptr)
            return false;

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

    outputRaster.initializeGrid(*(indexMap.at(layerIndex).header));

    // compute map
    for (int row = 0; row < indexMap.at(layerIndex).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(layerIndex).header->nrCols; col++)
        {
            long nodeIndex = indexMap.at(layerIndex).value[row][col];
            if (nodeIndex == static_cast<long>(indexMap.at(layerIndex).header->flag))
            {
                outputRaster.value[row][col] = outputRaster.header->flag;
                continue;
            }

            double value = (var == factorOfSafety) ? computeFactorOfSafety(row, col, layerIndex) : getCriteria3DVar(var, nodeIndex);

            if (value == NODATA)
            {
                outputRaster.value[row][col] = outputRaster.header->flag;
                continue;
            }

            // surface water level: from [m] to [mm]
            if (var == volumetricWaterContent && layerIndex == 0)
            {
                value *= 1000;          // [m] -> [mm]
            }

            outputRaster.value[row][col] = value;
        }
    }

    gis::updateMinMaxRasterGrid(&outputRaster);
    return true;
}


/*!
 * \brief getTotalSurfaceWaterContent
 * \return
 * wcSum: [m3] sum of surface water content
 * nrVoxels: [-] number of valid voxels
 */
bool Project3D::getTotalSurfaceWaterContent(double &wcSum, long &nrVoxels, int row0, int col0, int row1, int col1)
{
    errorString = "";
    if (! isCriteria3DInitialized)
    {
        errorString = ERROR_STR_INITIALIZE_3D;
        return false;
    }

    gis::Crit3DRasterHeader* header = indexMap.at(0).header;
    long flag = static_cast<long>(header->flag);
    double voxelArea = header->cellSize * header->cellSize;                 // [m2]

    // default: all map
    if (row1 == NODATA || col1 == NODATA)
    {
        row1 = header->nrRows - 1;
        col1 = header->nrCols - 1;
    }

    nrVoxels = 0;
    wcSum = 0.;
    for (int row = row0; row <= row1; row++)
    {
        for (int col = col0; col <= col1 ; col++)
        {
            long surfaceNodeIndex = indexMap.at(0).value[row][col];
            if (surfaceNodeIndex == flag)
                continue;

            double surfaceWC = getCriteria3DVar(volumetricWaterContent, surfaceNodeIndex);    // [m]

            if (isEqual(surfaceWC, NODATA))
                continue;

            wcSum += surfaceWC * voxelArea;        // [m3]
            nrVoxels++;
        }
    }

    return true;
}


/*!
 * \brief getTotalSoilWaterContent
 * \param
 * isMaximum: required water content at saturation
 * \return
 * wcSum: [m3] sum of soil water content
 * nrVoxels: [-] number of valid voxels
 */
bool Project3D::getTotalSoilWaterContent(double &wcSum, long &nrVoxels, bool isMaximum, int row0, int col0, int row1, int col1)
{
    errorString = "";
    if (! isCriteria3DInitialized)
    {
        errorString = ERROR_STR_INITIALIZE_3D;
        return false;
    }

    nrVoxels = 0;
    wcSum = 0.;
    gis::Crit3DRasterHeader* header = indexMap.at(0).header;
    long flag = static_cast<long>(header->flag);
    double voxelArea = header->cellSize * header->cellSize;                 // [m2]

    // default: all map
    if (row1 == NODATA || col1 == NODATA)
    {
        row1 = header->nrRows - 1;
        col1 = header->nrCols - 1;
    }

    for (unsigned layer = 1; layer < nrLayers; layer++)
    {
        double volume = voxelArea * layerThickness[layer];                  // [m3]
        double currentDepth = layerDepth[layer];

        for (int row = row0; row <= row1; row++)
        {
            for (int col = col0; col <= col1; col++)
            {
                long nodeIndex = indexMap.at(layer).value[row][col];
                if (nodeIndex == flag)
                    continue;

                int soilIndex = getSoilIndex(row, col);
                if (soilIndex == NODATA)
                    continue;

                int horizonIndex = soilList[soilIndex].getHorizonIndex(currentDepth);
                if (horizonIndex == NODATA)
                    continue;

                criteria3DVariable required3DVar = volumetricWaterContent;
                if (isMaximum)
                    required3DVar = maximumVolumetricWaterContent;

                double volWaterContent = getCriteria3DVar(required3DVar, nodeIndex);    // [m3 m-3]

                if (isEqual(volWaterContent, NODATA))
                    continue;

                double soilFraction = 1.0 - soilList[soilIndex].horizon[horizonIndex].coarseFragments;

                wcSum += volWaterContent * volume * soilFraction;                       // [m3]
                if (layer == 1)
                    nrVoxels++;
            }
        }
    }

    return true;
}


bool Project3D::computeMinimumFoS(gis::Crit3DRasterGrid &outputRaster)
{
    outputRaster.initializeGrid(*(indexMap.at(0).header));

    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            double minimumValue = NODATA;
            for (unsigned int layer = 1; layer < nrLayers; layer++)
            {
                double currentValue = computeFactorOfSafety(row, col, layer);
                if (isEqual(currentValue, NODATA))
                    continue;

                if (isEqual(minimumValue, NODATA) || currentValue < minimumValue)
                    minimumValue = currentValue;
            }

            if (!isEqual(minimumValue, NODATA))
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
                switch(myAggregation)
                {
                    case aggrMin:
                        gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationMin);
                        break;

                    case aggrMax:
                        gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationMax);
                        break;

                    case aggrSum:
                    case aggrAverage:
                    case aggrIntegral:
                        gis::mapAlgebra(myAggrMap, myMap, myAggrMap, operationSum);
                        break;

                    default:
                        logError("wrong aggregation type in function 'aggregateAndSaveDailyMap'");
                        return false;
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

// [m3 s-1]
bool Project3D::setSinkSource()
{
    std::string errorName = "";

    for (unsigned long i = 0; i < nrNodes; i++)
    {
        auto myResult = soilFluxes3D::setNodeWaterSinkSource(i, waterSinkSource[i]);

        if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
        {
            errorString = "Error in setWaterSinkSource: " + QString::fromStdString(errorName);
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
    //return evapMax * 0.67;   // TODO check evaporation on free water
    return evapMax;
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


bool Project3D::initializeEvaporationCoefficient()
{
    int lastEvapLayer = getSoilLayerIndex(MAX_EVAPORATION_DEPTH);
    if (computationSoilDepth < MAX_EVAPORATION_DEPTH)
    {
        lastEvapLayer = getSoilLayerIndex(computationSoilDepth);
    }

    // check
    if (lastEvapLayer == NODATA)
        return false;

    evapCoeff.clear();
    evapCoeff.resize(lastEvapLayer+1);
    layerEvapCoeff.clear();
    layerEvapCoeff.resize(lastEvapLayer+1);

    // assign layers coefficient
    double coeffSum = 0;
    for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
    {
        double depthCoeff = std::max((layerDepth[layer] - layerDepth[1]) / (MAX_EVAPORATION_DEPTH - layerDepth[1]), 0.0);
        // evaporation coefficient: 1 at depthMin, ~0.1 at MAX_EVAPORATION_DEPTH
        evapCoeff[layer] = exp(-2 * depthCoeff);
        // modify by layer thickness (normalized for a layer of 4 cm)
        layerEvapCoeff[layer] = evapCoeff[layer] * (layerThickness[layer] / 0.04);
        coeffSum += layerEvapCoeff[layer];
    }

    // normalize layer coefficients
    for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
    {
        layerEvapCoeff[layer] /= coeffSum;
    }

    return true;
}


/*! \brief assignEvaporation
 *  assign soil evaporation with a decrescent rate from surface to MAX_EVAPORATION_DEPTH
 *  \param row, col
 *  \param lai: leaf area index [m2 m-2]
 *  \return actual evaporation on soil column [mm]
 */
double Project3D::assignEvaporation(int row, int col, double lai, int soilIndex)
{
    // potential evaporation
    double et0 = double(hourlyMeteoMaps->mapHourlyET0->value[row][col]);        // [mm]
    double maxEvaporation = getPotentialEvaporation(et0, lai);                  // [mm]

    if (maxEvaporation < EPSILON)
        return 0.;

    // surface evaporation
    double actualEvaporationSum = 0;                                            // [mm]
    long surfaceNodeIndex = long(indexMap.at(0).value[row][col]);
    double surfaceWater = getCriteria3DVar(volumetricWaterContent, surfaceNodeIndex) * 1000;
    double surfaceEvaporation = std::min(maxEvaporation, surfaceWater);         // [mm]

    // TODO surface evaporation out of numerical solution
    double area = DEM.header->cellSize * DEM.header->cellSize;                  // [m2]
    double surfaceFlow = area * (surfaceEvaporation / 1000.) / 3600.;           // [m3 s-1]

    if (surfaceFlow <= DBL_EPSILON) {
        surfaceEvaporation = 0.;                                                // [mm]
    }
    else
    {
        waterSinkSource[surfaceNodeIndex] -= surfaceFlow;                       // [m3 s-1]
        actualEvaporationSum += surfaceEvaporation;                             // [mm]
    }

    double residualEvaporation = maxEvaporation - surfaceEvaporation;           // [mm]

    if (residualEvaporation < EPSILON || soilIndex == NODATA)
        return actualEvaporationSum;

    // soil evaporation
    int lastEvapLayer = (int)layerEvapCoeff.size() -1;
    int nrIteration = 0;
    while (residualEvaporation > EPSILON && nrIteration < 3)
    {
        double iterationEvapSum = 0;        // [mm]

        for (unsigned int layer=1; layer <= unsigned(lastEvapLayer); layer++)
        {
            long nodeIndex = long(indexMap.at(layer).value[row][col]);

            int horIndex = soilList[soilIndex].getHorizonIndex(layerDepth[layer]);
            if (horIndex == NODATA)
                continue;

            soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[horIndex];
            // [m3 m-3]
            double evapThreshold = horizon.waterContentHH + (1 - evapCoeff[layer]) * (horizon.waterContentFC - horizon.waterContentHH) * 0.5;
            // [m3 m-3]
            double layerWaterContent = getCriteria3DVar(volumetricWaterContent, nodeIndex) * horizon.getSoilFraction();
            // [m3 m-3]
            double wcAboveThreshold = std::max(layerWaterContent - evapThreshold, 0.0);
            // [mm]
            double evapAvailableWater = wcAboveThreshold * layerThickness[layer] * 1000.;
            // [mm]
            double layerEvaporation = std::min(evapAvailableWater, residualEvaporation * layerEvapCoeff[layer]);
            if (layerEvaporation > EPSILON)
            {
                double flow = area * (layerEvaporation / 1000.) / 3600.;        // [m3 s-1]

                waterSinkSource[nodeIndex] -= flow;                         // [m3 s-1]
                actualEvaporationSum += layerEvaporation;                   // [mm]
                iterationEvapSum += layerEvaporation;                       // [mm]
            }
        }

        residualEvaporation -= iterationEvapSum;
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
double Project3D::assignTranspiration(int row, int col, Crit3DCrop &currentCrop, double currentLai, double currentDegreeDays)
{
    // check lai and degree days
    if (currentLai < EPSILON || isEqual(currentDegreeDays, NODATA))
        return 0;       // [mm]

    // only surface
    if (nrLayers <= 1)
        return 0;

    // check soil
    int soilIndex = int(soilIndexMap.value[row][col]);
    if (soilIndex == NODATA)
    {
        // TODO FT improve (no soil)
        return 0;
    }

    // maximum transpiration
    double et0 = double(hourlyMeteoMaps->mapHourlyET0->value[row][col]);        // [mm]
    double kcMax = currentCrop.kcMax;                                           // [-]
    double maxTranspiration = getPotentialTranspiration(et0, currentLai, kcMax);

    if (maxTranspiration < EPSILON)
        return 0;

    // compute root lenght
    currentCrop.computeRootLength3D(currentDegreeDays, soilList[soilIndex].totalDepth);
    if (currentCrop.roots.currentRootLength <= 0)
        return 0;

    // compute root density
    if (! root::computeRootDensity3D(currentCrop, soilList[soilIndex], nrLayers, layerDepth, layerThickness))
        return 0;

    // check root layers
    if (currentCrop.roots.firstRootLayer == NODATA || currentCrop.roots.lastRootLayer == NODATA)
        return 0;

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
        waterSurplusStressFraction = 0;
    }

    double rootDensityWithoutStress = 0.0;                   // [-]
    int firstRootLayer = currentCrop.roots.firstRootLayer;
    int lastRootLayer = currentCrop.roots.lastRootLayer;
    double transpirationSubsetMax = 0; 
    double actualTranspiration = 0;             // [mm]

    for (int layer = firstRootLayer; layer <= lastRootLayer; layer++)
    {
        long nodeIndex = long(indexMap.at(layer).value[row][col]);
        if (isEqual(nodeIndex, indexMap.at(layer).header->flag))
            continue;

        int horizonIndex = soilList[soilIndex].getHorizonIndex(layerDepth[layer]);
        if (horizonIndex == NODATA)
            continue;

        soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[horizonIndex];

        // [m3 m-3]
        double volWaterContent = getCriteria3DVar(volumetricWaterContent, nodeIndex);
        double volWaterSurplusThreshold = horizon.waterContentSAT - waterSurplusStressFraction * (horizon.waterContentSAT - horizon.waterContentFC);
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
            ratio = (horizon.waterContentSAT - volWaterContent) / (horizon.waterContentSAT - volWaterSurplusThreshold);
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
        {
            if (! isLayerStressed[layer] && layerTranspiration[layer] > 0)
                layerTranspiration[layer] += redistribution * (currentCrop.roots.rootDensity[layer] / rootDensityWithoutStress);
        }
    }

    // assigns transpiration to water sink source
    double area = DEM.header->cellSize * DEM.header->cellSize;              // [m2]
    actualTranspiration = 0;
    for (int layer = firstRootLayer; layer <= lastRootLayer; layer++)
    {
        double flow = area * (layerTranspiration[layer] / 1000.) / 3600.;   // [m3 s-1]
        if (flow > DBL_EPSILON)
        {
            long nodeIndex = long(indexMap.at(layer).value[row][col]);
            if (! isEqual(nodeIndex, indexMap.at(layer).header->flag))
            {
                waterSinkSource.at(nodeIndex) -= flow;                      // [m3 s-1]
                actualTranspiration += layerTranspiration[layer];           // [mm]
            }
        }
    }

    return actualTranspiration;
}


/*!
     * \brief computeFactorOfSafety
     * \return factor of safety FoS [-]
     * if fos < 1 the slope is unstable
     */
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
    double saturationDegree = soilFluxes3D::getNodeDegreeOfSaturation(nodeIndex);

    if (saturationDegree == MEMORY_ERROR || saturationDegree == INDEX_ERROR)
    {
        return NODATA;
    }

    // matric potential (with sign) [kPa]
    double matricPotential = std::min(0.0, soilFluxes3D::getNodeMatricPotential(nodeIndex) * GRAVITY);

    // suction stress [kPa]
    double suctionStress = matricPotential * saturationDegree;

    // effective cohesion [kPa]
    double effectiveCohesion = soilList[unsigned(soilIndex)].horizon[horizonIndex].effectiveCohesion;

    // unit weight - integration from zero to layerDepth
    // [kPa]
    double weightSum = 0;

    // surface
    long currentNode = indexMap.at(0).value[row][col];
    if (currentNode != indexMap.at(0).header->flag)
    {
        // [m]
        double surfaceWater = soilFluxes3D::getNodeWaterContent(currentNode);
        if (surfaceWater > 0)
        {
            weightSum += (surfaceWater * GRAVITY);
        }
    }

    // sub-surface
    for (unsigned int layer = 1; layer <= layerIndex; layer++)
    {
        long currentNode = indexMap.at(layer).value[row][col];
        if (currentNode != indexMap.at(layer).header->flag)
        {
            int currentHorizon = soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[layer]);
            if (currentHorizon == NODATA)
                continue;

            // [g cm-3] --> [Mg m-3]
            double bulkDensity = soilList[unsigned(soilIndex)].horizon[currentHorizon].bulkDensity;
            double waterContent = soilFluxes3D::getNodeWaterContent(currentNode);
            // [kN m-3]
            double unitWeight = (bulkDensity + waterContent) * GRAVITY;
            // [kPa]
            weightSum += unitWeight * layerThickness[layer];
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
    if (result == CRIT3D_OK)
        return false;

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
        crit3dVar = soilFluxes3D::getNodeWaterContent(nodeIndex);
    }
    else if (myVar == maximumVolumetricWaterContent)
    {
        crit3dVar = soilFluxes3D::getNodeMaximumWaterContent(nodeIndex);
    }
    else if (myVar == availableWaterContent)
    {
        crit3dVar = soilFluxes3D::getNodeAvailableWaterContent(nodeIndex);
    }
    else if (myVar == waterTotalPotential)
    {
        crit3dVar = soilFluxes3D::getNodeTotalPotential(nodeIndex);
    }
    else if (myVar == waterMatricPotential)
    {
        crit3dVar = soilFluxes3D::getNodeMatricPotential(nodeIndex);
    }
    else if (myVar == degreeOfSaturation)
    {
        crit3dVar = soilFluxes3D::getNodeDegreeOfSaturation(nodeIndex);
    }
    else if (myVar == waterInflow)
    {
        crit3dVar = soilFluxes3D::getNodeSumLateralWaterFlowIn(nodeIndex) * 1000;
    }
    else if (myVar == waterOutflow)
    {
        crit3dVar = soilFluxes3D::getNodeSumLateralWaterFlowOut(nodeIndex) * 1000;
    }
    else if (myVar == waterDeficit)
    {
        // TODO leggere horizon per field capacity
        double fieldCapacity = 3.0;
        crit3dVar = soilFluxes3D::getNodeWaterDeficit(nodeIndex, fieldCapacity);
    }
    else if (myVar == surfacePond)
    {
        crit3dVar = soilFluxes3D::getNodePond(nodeIndex) * 1000;
    }
    else
    {
        crit3dVar = MISSING_DATA_ERROR;
    }

    // check result
    if (crit3dVar == INDEX_ERROR || crit3dVar == MEMORY_ERROR || crit3dVar == TOPOGRAPHY_ERROR || crit3dVar == MISSING_DATA_ERROR)
    {
        return NODATA;
    }
    else
    {
        return crit3dVar;
    }
}


bool setCriteria3DVar(criteria3DVariable myVar, long nodeIndex, double myValue)
{
    auto myResult = soilFluxes3D::SF3Derror_t::MissingDataError;

    if (myVar == volumetricWaterContent)
    {
        // TODO check: skeleton
        myResult = soilFluxes3D::setNodeWaterContent(nodeIndex, myValue);
    }
    else if (myVar == waterMatricPotential)
    {
        myResult = soilFluxes3D::setNodeMatricPotential(nodeIndex, myValue);
    }

    return myResult == soilFluxes3D::SF3Derror_t::SF3Dok;
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


bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, gis::Crit3DRasterGrid* myGrid)
{
    QString fileName = hourlyPath + getOutputNameHourly(myVar, myTime);
    std::string error;

    if (gis::readEsriGrid(fileName.toStdString(), myGrid, error))
        return true;
    else
        return false;
}


float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, int row, int col)
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

    switch(myVar)
    {
        case volumetricWaterContent:
            fileName += "_WaterContent_";
            break;
        case availableWaterContent:
            fileName += "_availableWaterContent_";
            break;
        case waterMatricPotential:
            fileName += "_MP_";
            break;
        case degreeOfSaturation:
            fileName += "_degreeOfSaturation_";
            break;
        default:
            return "";
    }

    return fileName;
}


bool setVariableDepth(const QList<QString>& depthList, std::vector<int>& variableDepth)
{
    std::size_t nrDepth = static_cast<std::size_t>(depthList.size());

    if (nrDepth == 0)
        return true;

    variableDepth.resize(nrDepth);
    for (unsigned i = 0; i < nrDepth; i++)
    {
        bool isOk;
        int depth = depthList[i].toInt(&isOk);
        if (!isOk || depth <= 0)
            return false;

        variableDepth[i] = depth;
    }

    return true;
}



