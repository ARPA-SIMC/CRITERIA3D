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
#include "utilities.h"
#include "criteria3DProject.h"
#include "gis.h"
#include "meteo.h"
#include "color.h"
#include "statistics.h"
#include "project3D.h"
#include "soilFluxes3D.h"

#include <QtSql>
#include <QPaintEvent>
#include <QVector3D>


Crit3DProject::Crit3DProject() : Project3D()
{
    _saveOutputRaster = false;
    _saveOutputPoints = false;
    _saveDailyState = false;
    _saveEndOfRunState = false;

    modelFirstTime.setTimeSpec(Qt::UTC);
    modelLastTime.setTimeSpec(Qt::UTC);
}


bool Crit3DProject::initializeCriteria3DModel()
{
    if (! check3DProject())
        return false;

    clearWaterBalance3D();

    // it is necessary to reload the soils db (the fitting options may have changed)
    if (! loadSoilDatabase(soilDbFileName))
        return false;

    if (! setSoilIndexMap())
        return false;

    if (! initialize3DModel())
    {
        clearWaterBalance3D();
        errorString += "\nCriteria3D model is not initialized.";
        return false;
    }

    isCriteria3DInitialized = true;
    return true;
}


bool Crit3DProject::initializeCropMaps()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    logInfo("Initialize crop...");

    // initialize LAI and degree days map to NODATA
    laiMap.initializeGrid(*(DEM.header));
    degreeDaysMap.initializeGrid(*(DEM.header));

    dailyTminMap.initializeGrid(*(DEM.header));
    dailyTmaxMap.initializeGrid(*(DEM.header));

    return true;
}


bool Crit3DProject::initializeCropWithClimateData()
{
    if (! processes.computeCrop)
        return false;

    if (! initializeCropMaps())
        return false;

    if (landUnitList.empty() || cropList.empty())
    {
        errorString = "missing crop db or land use map";
        return false;
    }

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    double degreeDays = 0;
                    int firstDoy = 1;
                    int lastDoy = currentDate.dayOfYear();

                    if (gisSettings.startLocation.latitude >= 0)
                    {
                        // Northern hemisphere
                        firstDoy = 1;
                    }
                    else
                    {
                        // Southern hemisphere
                        if (currentDate.dayOfYear() >= 182)
                        {
                            firstDoy = 182;
                        }
                        else
                        {
                            firstDoy = -183;
                        }
                    }

                    // daily cycle
                    for (int doy = firstDoy; doy <= lastDoy; doy++)
                    {
                        int currentDoy = doy;
                        int currentYear = currentDate.year();
                        if (currentDoy <= 0)
                        {
                            currentYear--;
                            currentDoy += 365;
                        }
                        Crit3DDate myDate = getDateFromDoy(currentYear, currentDoy);

                        float tmin = climateParameters.getClimateVar(dailyAirTemperatureMin, myDate.month,
                                                                     height, quality->getReferenceHeight());
                        float tmax = climateParameters.getClimateVar(dailyAirTemperatureMax, myDate.month,
                                                                     height, quality->getReferenceHeight());

                        double currentDD = cropList[index].getDailyDegreeIncrease(tmin, tmax, currentDoy);
                        if (! isEqual(currentDD, NODATA))
                        {
                            degreeDays += currentDD;
                        }
                    }

                    degreeDaysMap.value[row][col] = float(degreeDays);
                    laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDays, gisSettings.startLocation.latitude, currentDate.dayOfYear());
                }
            }
        }
    }

    logInfo("LAI initialized with climate data - doy: " + QString::number(currentDate.dayOfYear()));
    isCropInitialized = true;

    return true;
}


bool Crit3DProject::initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap)
{
    initializeCropMaps();

    if (! myDegreeMap.isLoaded)
    {
        errorString = "Wrong degree days map: crop cannot be initialized.";
        processes.setComputeCrop(false);
        return false;
    }

     if (! landUseMap.isLoaded || landUnitList.empty())
    {
        errorString = "Crop db or land use map is missing: crop cannot be initialized.";
        processes.setComputeCrop(false);
        return false;
    }

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // field unit list and crop list have the same index
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    double x, y;
                    gis::getUtmXYFromRowCol(*(DEM.header), row, col, &x, &y);
                    float currentDegreeDay = gis::getValueFromXY(myDegreeMap, x, y);
                    if (! isEqual(currentDegreeDay, myDegreeMap.header->flag))
                    {
                        degreeDaysMap.value[row][col] = currentDegreeDay;
                        laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDaysMap.value[row][col],
                                                         gisSettings.startLocation.latitude, currentDate.dayOfYear());
                    }
                }
            }
        }
    }

    logInfo("LAI initialized with degree days map.");
    isCropInitialized = true;
    return true;
}


void Crit3DProject::dailyUpdateCropMaps(const QDate &myDate)
{
    int firstDoy = 1;
    if (gisSettings.startLocation.latitude < 0)
    {
        // Southern hemisphere
        firstDoy = 182;
    }

    // reset the crop at the beginning of the new year
    if (myDate.dayOfYear() == firstDoy)
    {
         logInfo("Reset crop...");

        laiMap.emptyGrid();
        degreeDaysMap.emptyGrid();
    }

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            // is valid point
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // landUnit list and crop list have the same index
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    float tmin = dailyTminMap.value[row][col];
                    float tmax = dailyTmaxMap.value[row][col];
                    if (! isEqual(tmin, dailyTminMap.header->flag) && ! isEqual(tmax, dailyTmaxMap.header->flag))
                    {
                        double dailyDD = cropList[index].getDailyDegreeIncrease(tmin, tmax, currentDate.dayOfYear());
                        if (! isEqual(dailyDD, NODATA))
                        {
                            if (isEqual(degreeDaysMap.value[row][col], degreeDaysMap.header->flag))
                            {
                                degreeDaysMap.value[row][col] = float(dailyDD);
                            }
                            else
                            {
                                degreeDaysMap.value[row][col] += float(dailyDD);
                            }

                            laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDaysMap.value[row][col],
                                                            gisSettings.startLocation.latitude, currentDate.dayOfYear());
                        }
                    }
                }
            }
        }
    }

    // cleans daily temperature maps
    dailyTminMap.emptyGrid();
    dailyTmaxMap.emptyGrid();
}


/*!
 * \brief assignETreal
 * assigns soil evaporation and crop transpiration for the whole domain
 */
void Crit3DProject::assignETreal()
{
    totalEvaporation = 0;               // [m3 h-1]
    totalTranspiration = 0;             // [m3 h-1]

    double area = DEM.header->cellSize * DEM.header->cellSize;

    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                double utmX, utmY;
                DEM.getXY(row, col, utmX, utmY);
                int soilIndex = getSoilListIndex(utmX, utmY);

                float lai = laiMap.value[row][col];
                if (isEqual(lai, NODATA))
                {
                    lai = 0;
                }

                // assigns actual evaporation
                double actualEvap = assignEvaporation(row, col, lai, soilIndex);    // [mm h-1]
                double evapFlow = area * (actualEvap / 1000.);                      // [m3 h-1]
                totalEvaporation += evapFlow;                                       // [m3 h-1]

                // assigns actual transpiration
                if (lai > 0)
                {
                    float degreeDays = degreeDaysMap.value[row][col];
                    double actualTransp = assignTranspiration(row, col, lai, degreeDays);   // [mm h-1]
                    double traspFlow = area * (actualTransp / 1000.);                       // [m3 h-1]
                    totalTranspiration += traspFlow;                                        // [m3 h-1]
                }
            }
        }
    }
}


void Crit3DProject::assignPrecipitation()
{
    // initialize
    totalPrecipitation = 0;                 // [m3]

    double area = DEM.header->cellSize * DEM.header->cellSize;

    gis::Crit3DRasterGrid *snowFallMap, *snowMeltMap;
    if (processes.computeSnow)
    {
        snowFallMap = snowMaps.getSnowFallMap();
        snowMeltMap = snowMaps.getSnowMeltMap();
    }

    // precipitation
    for (long row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (long col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                float prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                if (! isEqual(prec, hourlyMeteoMaps->mapHourlyPrec->header->flag))
                {
                    float liquidWater = prec;
                    if (processes.computeSnow)
                    {
                        float currentSnowFall = snowFallMap->value[row][col];
                        float currentSnowMelt = snowMeltMap->value[row][col];
                        if (! isEqual(currentSnowFall, snowFallMap->header->flag)
                            && ! isEqual(currentSnowMelt, snowMeltMap->header->flag) )
                        {
                            liquidWater = prec - currentSnowFall + currentSnowMelt;
                        }
                    }
                    if (liquidWater > 0)
                    {
                        float surfaceWater = checkSoilCracking(row, col, liquidWater);

                        double flow = area * (surfaceWater / 1000.);        // [m3 h-1]
                        waterSinkSource[surfaceIndex] += flow / 3600.;      // [m3 s-1]
                        totalPrecipitation += flow;                         // [m3]
                    }
                }
            }
        }
    }
}


// Water infiltration into soil cracks
// input: rainfall [mm]
// returns the water remaining on the surface after infiltration into soil cracks
float Crit3DProject::checkSoilCracking(int row, int col, float precipitation)
{
    const double MAX_CRACKING_DEPTH = 0.6;              // [m]
    const double MIN_VOID_VOLUME = 0.15;                // [m3 m-3]
    const double MAX_VOID_VOLUME = 0.20;                // [m3 m-3]

    // check soil
    int soilIndex = getSoilIndex(row, col);
    if (soilIndex == NODATA)
        return precipitation;

    // check pond
    long surfaceNodeIndex = indexMap.at(0).value[row][col];
    double currentPond = getCriteria3DVar(surfacePond, surfaceNodeIndex);       // [mm]
    double minimumPond = currentPond;                                           // [mm]
    if (precipitation <= minimumPond)
        return precipitation;

    // check clay
    double maxDepth = std::min(computationSoilDepth, MAX_CRACKING_DEPTH);   // [m]
    bool isFineFraction = true;
    int lastFineHorizon = NODATA;
    int h = 0;
    while (soilList[soilIndex].horizon[h].upperDepth <= maxDepth && isFineFraction)
    {
        soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[h];

        double fineFraction = (horizon.texture.clay + horizon.texture.silt * 0.5) / 100 * (1 - horizon.coarseFragments);
        if (fineFraction < 0.5)
        {
            isFineFraction = false;
        }
        else
        {
            lastFineHorizon = h;
            h++;
        }
    }

    if (lastFineHorizon == NODATA)
        return precipitation;

    maxDepth = std::min(soilList[soilIndex].horizon[lastFineHorizon].lowerDepth, MAX_CRACKING_DEPTH);

    // clay horizon is too thin
    if (maxDepth < 0.2)
        return precipitation;

    // compute void volume
    double stepDepth = 0.05;                // [m]
    double currentDepth = stepDepth;        // [m]
    double voidVolumeSum = 0;
    int nrData = 0;
    while (currentDepth <= maxDepth )
    {
        int layerIndex = getSoilLayerIndex(currentDepth);
        long nodeIndex = indexMap.at(layerIndex).value[row][col];

        double VWC = getCriteria3DVar(volumetricWaterContent, nodeIndex);               // [m3 m-3]
        double maxVWC = getCriteria3DVar(maximumVolumetricWaterContent, nodeIndex);     // [m3 m-3]

        // TODO: coarse fragment
        voidVolumeSum += (maxVWC - VWC);

        currentDepth += stepDepth;
        nrData++;
    }

    double avgVoidVolume = voidVolumeSum / nrData;              // [m3 m-3]
    if (avgVoidVolume <= MIN_VOID_VOLUME)
        return precipitation;

    // THERE IS A SOIL CRACK
    double crackRatio = std::min(1.0, (avgVoidVolume - MIN_VOID_VOLUME) / (MAX_VOID_VOLUME - MIN_VOID_VOLUME));

    double maxInfiltration = precipitation * crackRatio;        // [mm]
    double surfaceWater = precipitation - maxInfiltration;      // [mm]
    surfaceWater = std::max(surfaceWater, minimumPond);
    double downWater = precipitation - surfaceWater;            // [mm]

    int lastLayer = getSoilLayerIndex(maxDepth);
    double area = DEM.header->cellSize * DEM.header->cellSize;  // [m2]

    // accumulation on the crack bottom (0.5 mm of water for each soil cm)
    for (int l = lastLayer; l > 0; l--)
    {
        if (downWater <= 0)
            break;

        double layerThick_cm = layerThickness[l] * 100;         // [cm]
        double layerWater = layerThick_cm * 0.5;                // [mm]
        layerWater = std::min(layerWater, downWater);

        long nodeIndex = indexMap.at(l).value[row][col];
        double flow = area * (layerWater / 1000.);              // [m3 h-1]
        waterSinkSource[nodeIndex] += flow / 3600.;             // [m3 s-1]
        totalPrecipitation += flow;                             // [m3]

        downWater -= layerWater;
    }

    if (downWater > 0)
    {
        return surfaceWater + downWater;
    }
    else
    {
        return surfaceWater;
    }
}


bool Crit3DProject::runModels(QDateTime firstTime, QDateTime lastTime, bool isRestart)
{
    if (! isRestart)
    {
        // create tables for output points
        if (isSaveOutputPoints())
        {
            if (! writeOutputPointsTables())
            {
                logError();
                return false;
            }
        }

        // initialize meteo maps
        if (processes.computeMeteo)
        {
            hourlyMeteoMaps->initialize();

            // load td maps if needed
            if (interpolationSettings.getUseTD())
            {
                logInfoGUI("Loading topographic distance maps...");
                if (! loadTopographicDistanceMaps(true, false))
                    return false;
            }
        }

        // initialize radiation maps
        if (processes.computeRadiation)
        {
            radiationMaps->initialize();
        }

        isModelRunning = true;
    }

    QDate firstDate = firstTime.date();
    QDate lastDate = lastTime.date();
    int hour1 = firstTime.time().hour();
    int hour2 = lastTime.time().hour();

    // cycle on days
    QString currentOutputPath;
    for (QDate myDate = firstDate; myDate <= lastDate; myDate = myDate.addDays(1))
    {
        setCurrentDate(myDate);

        if (processes.computeCrop)
        {
            dailyUpdateCropMaps(myDate);
        }
        if (processes.computeWater)
        {
            dailyUpdatePond();
        }

        if (isSaveOutputRaster())
        {
            // create directory for hourly raster output
            currentOutputPath = getProjectPath() + PATH_OUTPUT + myDate.toString("yyyy/MM/dd/");
            if (! QDir().mkpath(currentOutputPath))
            {
                logError("Creation of directory for hourly raster output failed:" + currentOutputPath);
                setSaveOutputRaster(false);
            }
        }

        // cycle on hours
        int firstHour = (myDate == firstDate) ? hour1 : 0;
        int lastHour = (myDate == lastDate) ? hour2 : 23;

        for (int hour = firstHour; hour <= lastHour; hour++)
        {
            setCurrentHour(hour);
            if (currentSeconds == 0 || currentSeconds == 3600)
                isRestart = false;

            if (! runModelHour(currentOutputPath, isRestart))
            {
                isModelRunning = false;
                logError();
                return false;
            }

            // output points
            if (isSaveOutputPoints() && currentSeconds == 3600)
            {
                if (! writeOutputPointsData())
                {
                    isModelRunning = false;
                    logError();
                    return false;
                }
            }

            if (isModelPaused || isModelStopped)
            {
                return true;
            }
        }

        if (isSaveDailyState())
        {
            saveModelsState();
        }
    }

    if (isSaveEndOfRunState())
    {
        saveModelsState();
    }

    isModelRunning = false;
    logInfoGUI("Computation is finished.");

    return true;
}


void Crit3DProject::setSaveOutputRaster(bool isSave)
{
    _saveOutputRaster = isSave;
}

bool Crit3DProject::isSaveOutputRaster()
{
    return _saveOutputRaster;
}

void Crit3DProject::setSaveOutputPoints(bool isSave)
{
    _saveOutputPoints = isSave;
}

// true if at least one point is active
bool Crit3DProject::isSaveOutputPoints()
{
    if (! _saveOutputPoints || outputPoints.empty())
        return false;

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
            return true;
    }

    return false;
}


bool Crit3DProject::loadCriteria3DProject(QString myFileName)
{
    if (myFileName == "") return(false);

    clear3DProject();
    initializeProject3D();

    //snowModel.initialize();

    if (! loadProjectSettings(myFileName))
        return false;

    if (! loadProject3DSettings())
        return false;

    if (! loadProject())
    {
        if (errorType != ERROR_DBGRID && errorType != ERROR_DBPOINT)
            return false;
    }

    if (meteoPointsLoaded)
    {
        meteoPointsDbFirstTime = findDbPointFirstTime();
    }

    if (! loadCriteria3DParameters())
    {
        return false;
    }

    // soil map and data
    if (soilMapFileName != "")
        loadSoilMap(soilMapFileName);

    if (soilDbFileName != "")
        loadSoilDatabase(soilDbFileName);

    // land use map and crop data
    if (landUseMapFileName != "") loadLandUseMap(landUseMapFileName);
    if (cropDbFileName != "") loadCropDatabase(cropDbFileName);

    if (projectName != "")
    {
        logInfo("Project " + projectName + " loaded");
    }

    isProjectLoaded = true;
    return isProjectLoaded;
}


bool Crit3DProject::loadCriteria3DParameters()
{
    QString fileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing parameters file: " + fileName);
        return false;
    }
    if (parametersSettings == nullptr)
    {
        logError("parametersSettings is null");
        return false;
    }
    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        if (group == "snow")
        {
            parametersSettings->beginGroup(group);
            if (parametersSettings->contains("tempMaxWithSnow") && !parametersSettings->value("tempMaxWithSnow").toString().isEmpty())
            {
                snowModel.snowParameters.tempMaxWithSnow = parametersSettings->value("tempMaxWithSnow").toDouble();
            }
            if (parametersSettings->contains("tempMinWithRain") && !parametersSettings->value("tempMinWithRain").toString().isEmpty())
            {
                snowModel.snowParameters.tempMinWithRain = parametersSettings->value("tempMinWithRain").toDouble();
            }
            if (parametersSettings->contains("snowWaterHoldingCapacity") && !parametersSettings->value("snowWaterHoldingCapacity").toString().isEmpty())
            {
                snowModel.snowParameters.snowWaterHoldingCapacity = parametersSettings->value("snowWaterHoldingCapacity").toDouble();
            }
            if (parametersSettings->contains("skinThickness") && !parametersSettings->value("skinThickness").toString().isEmpty())
            {
                snowModel.snowParameters.skinThickness = parametersSettings->value("skinThickness").toDouble();
            }
            if (parametersSettings->contains("snowVegetationHeight") && !parametersSettings->value("snowVegetationHeight").toString().isEmpty())
            {
                snowModel.snowParameters.snowVegetationHeight = parametersSettings->value("snowVegetationHeight").toDouble();
            }
            if (parametersSettings->contains("soilAlbedo") && !parametersSettings->value("soilAlbedo").toString().isEmpty())
            {
                snowModel.snowParameters.soilAlbedo = parametersSettings->value("soilAlbedo").toDouble();
            }
            if (parametersSettings->contains("snowSurfaceDampingDepth") && !parametersSettings->value("snowSurfaceDampingDepth").toString().isEmpty())
            {
                snowModel.snowParameters.snowSurfaceDampingDepth = parametersSettings->value("snowSurfaceDampingDepth").toDouble();
            }
            parametersSettings->endGroup();
        }

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

            parametersSettings->endGroup();
        }

        if (group == "soilCracking")
        {
            parametersSettings->beginGroup(group);

            // TODO

            parametersSettings->endGroup();

        }
    }
    return true;
}


bool Crit3DProject::writeCriteria3DParameters(bool isSnow, bool isWater, bool isSoilCrack)
{
    QString fileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing parametersSettings file: " + fileName);
        return false;
    }
    if (parametersSettings == nullptr)
    {
        logError("parametersSettings is null");
        return false;
    }

    if (isSnow)
    {
        parametersSettings->setValue("snow/tempMaxWithSnow", snowModel.snowParameters.tempMaxWithSnow);
        parametersSettings->setValue("snow/tempMinWithRain", snowModel.snowParameters.tempMinWithRain);
        parametersSettings->setValue("snow/snowWaterHoldingCapacity", snowModel.snowParameters.snowWaterHoldingCapacity);
        parametersSettings->setValue("snow/skinThickness", snowModel.snowParameters.skinThickness);
        parametersSettings->setValue("snow/snowVegetationHeight", snowModel.snowParameters.snowVegetationHeight);
        parametersSettings->setValue("snow/soilAlbedo", snowModel.snowParameters.soilAlbedo);
        parametersSettings->setValue("snow/snowSurfaceDampingDepth", snowModel.snowParameters.snowSurfaceDampingDepth);
    }

    if (isWater)
    {
        parametersSettings->setValue("soilWaterFluxes/isInitialWaterPotential", waterFluxesParameters.isInitialWaterPotential);
        parametersSettings->setValue("soilWaterFluxes/initialWaterPotential", waterFluxesParameters.initialWaterPotential);
        parametersSettings->setValue("soilWaterFluxes/initialDegreeOfSaturation", waterFluxesParameters.initialDegreeOfSaturation);

        parametersSettings->setValue("soilWaterFluxes/computeOnlySurface", waterFluxesParameters.computeOnlySurface);
        parametersSettings->setValue("soilWaterFluxes/computeAllSoilDepth", waterFluxesParameters.computeAllSoilDepth);
        parametersSettings->setValue("soilWaterFluxes/imposedComputationDepth", waterFluxesParameters.imposedComputationDepth);

        parametersSettings->setValue("soilWaterFluxes/conductivityHorizVertRatio", waterFluxesParameters.conductivityHorizVertRatio);

        parametersSettings->setValue("soilWaterFluxes/freeCatchmentRunoff", waterFluxesParameters.freeCatchmentRunoff);
        parametersSettings->setValue("soilWaterFluxes/freeBottomDrainage", waterFluxesParameters.freeBottomDrainage);
        parametersSettings->setValue("soilWaterFluxes/freeLateralDrainage", waterFluxesParameters.freeLateralDrainage);

        parametersSettings->setValue("soilWaterFluxes/modelAccuracy", waterFluxesParameters.modelAccuracy);
    }

    if (isSoilCrack)
    {
        // todo
        // parametersSettings->setValue("soilCracking/ ", );

    }

    parametersSettings->sync();

    return true;
}


bool Crit3DProject::check3DProject()
{
    if (!DEM.isLoaded || !meteoPointsLoaded)
    {
        if (! DEM.isLoaded)
            errorString = ERROR_STR_MISSING_DEM;
        else if (! meteoPointsLoaded)
            errorString =  ERROR_STR_MISSING_DB;
        return false;
    }

    return true;
}


double Crit3DProject::getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar)
{
    unsigned int hIndex = unsigned(soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[unsigned(layerIndex)]));

    if (myVar == soil::soilWaterPotentialWP)
        return soilList[unsigned(soilIndex)].horizon[hIndex].wiltingPoint;
    else if (myVar == soil::soilWaterPotentialFC)
        return soilList[unsigned(soilIndex)].horizon[hIndex].fieldCapacity;
    else if (myVar == soil::soilWaterContentFC)
        return soilList[unsigned(soilIndex)].horizon[hIndex].waterContentFC;
    else if (myVar == soil::soilWaterContentSat)
        return soilList[unsigned(soilIndex)].horizon[hIndex].vanGenuchten.thetaS;
    else if (myVar == soil::soilWaterContentWP)
    {
        double signPsiLeaf = -160;      //[m]
        return soil::thetaFromSignPsi(signPsiLeaf, soilList[unsigned(soilIndex)].horizon[hIndex]);
    }
    else
        return NODATA;
}


void Crit3DProject::clear3DProject()
{
    snowMaps.clear();

    dailyTminMap.clear();
    dailyTmaxMap.clear();

    degreeDaysMap.clear();

    clearGeometry();

    clearProject3D();
}


bool Crit3DProject::computeAllMeteoMaps(const QDateTime& myTime, bool showInfo)
{
    if (! this->DEM.isLoaded)
    {
        errorString = "Load a Digital Elevation Model (DEM) before.";
        return false;
    }
    if (this->hourlyMeteoMaps == nullptr)
    {
        errorString = "Meteo maps not initialized.";
        return false;
    }

    this->hourlyMeteoMaps->setComputed(false);

    if (showInfo)
    {
        setProgressBar("Computing air temperature...", 6);
    }

    if (! interpolateHourlyMeteoVar(airTemperature, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing air relative humidity...");
        updateProgressBar(1);
    }

    if (! interpolateHourlyMeteoVar(airRelHumidity, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing precipitation...");
        updateProgressBar(2);
    }

    if (! interpolateHourlyMeteoVar(precipitation, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing wind intensity...");
        updateProgressBar(3);
    }

    if (! interpolateHourlyMeteoVar(windScalarIntensity, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing global irradiance...");
        updateProgressBar(4);
    }

    if (! interpolateHourlyMeteoVar(globalIrradiance, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing ET0...");
        updateProgressBar(5);
    }

    if (! this->hourlyMeteoMaps->computeET0PMMap(this->DEM, this->radiationMaps))
        return false;

    if (showInfo) closeProgressBar();

    this->hourlyMeteoMaps->setComputed(true);

    return true;
}


void Crit3DProject::setAllHourlyMeteoMapsComputed(bool value)
{
    if (radiationMaps != nullptr)
        radiationMaps->setComputed(value);

    if (hourlyMeteoMaps != nullptr)
        hourlyMeteoMaps->setComputed(value);
}


bool Crit3DProject::saveDailyOutput(QDate myDate, const QString& hourlyPath)
{
    QString dailyPath = getProjectPath() + "OUTPUT/daily/" + myDate.toString("yyyy/MM/dd/");
    QDir myDir;

    if (! myDir.mkpath(dailyPath))
    {
        logError("Creation daily output directory failed." );
        return false;
    }
    else
    {
        logInfo("Aggregate daily meteo data");
        Crit3DDate crit3DDate = getCrit3DDate(myDate);

        aggregateAndSaveDailyMap(airTemperature, aggrMin, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airTemperature, aggrMax, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airTemperature, aggrAverage, crit3DDate, dailyPath,hourlyPath);
        aggregateAndSaveDailyMap(precipitation, aggrSum, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(referenceEvapotranspiration, aggrSum, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrMin, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrMax, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrAverage, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(globalIrradiance, aggrSum, crit3DDate, dailyPath, hourlyPath);

        removeDirectory(hourlyPath);

        // save crop output

        // save water balance output
    }

    return true;
}


bool Crit3DProject::initializeSnowModel()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    snowMaps.initializeSnowMaps(DEM, snowModel.snowParameters.skinThickness);

    return true;
}


void Crit3DProject::computeSnowPoint(int row, int col)
{
    snowMaps.setPoint(snowModel, row, col);

    double airT = hourlyMeteoMaps->mapHourlyTair->value[row][col];
    double prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
    double relHum = hourlyMeteoMaps->mapHourlyRelHum->value[row][col];
    double windInt = hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col];
    double globalRad = radiationMaps->globalRadiationMap->value[row][col];
    double beamRad = radiationMaps->beamRadiationMap->value[row][col];
    double transmissivity = radiationMaps->transmissivityMap->value[row][col];
    double clearSkyTrans = radSettings.getClearSky();
    double myWaterContent = 0;                              // [mm]

    snowModel.setSnowInputData(airT, prec, relHum, windInt, globalRad, beamRad, transmissivity, clearSkyTrans, myWaterContent);

    snowModel.computeSnowBrooksModel();

    snowMaps.updateMapRowCol(snowModel, row, col);
}


// it assumes that header of meteo and snow maps = header of DEM
bool Crit3DProject::computeSnowModel()
{
    // check
    if (! snowMaps.isInitialized)
    {
        logError("Initialize snow model before.");
        return false;
    }

    if (! hourlyMeteoMaps->getComputed())
    {
        logError("Missing meteo maps.");
        return false;
    }

    if (! radiationMaps->getComputed())
    {
        logError("Missing radiation map.");
        return false;
    }

    if (getComputeOnlyPoints())
    {
        for (unsigned int i = 0; i < outputPoints.size(); i++)
        {
            if (outputPoints[i].active)
            {
                double x = outputPoints[i].utm.x;
                double y = outputPoints[i].utm.y;

                int row, col;
                DEM.getRowCol(x, y, row, col);
                if (! gis::isOutOfGridRowCol(row, col, DEM))
                {
                    computeSnowPoint(row, col);
                }
            }
        }
    }
    else
    {
        for (int row = 0; row < DEM.header->nrRows; row++)
        {
            for (int col = 0; col < DEM.header->nrCols; col++)
            {
                if (! isEqual(DEM.value[row][col], DEM.header->flag))
                {
                    computeSnowPoint(row, col);
                }
                else
                {
                    snowMaps.flagMapRowCol(row, col);
                }
            }
        }

        snowMaps.updateRangeMaps();
    }

    return true;
}


bool Crit3DProject::updateDailyTemperatures()
{
    if (! dailyTminMap.isLoaded || ! dailyTmaxMap.isLoaded || ! hourlyMeteoMaps->mapHourlyTair->isLoaded)
        return false;

    for (long row = 0; row < dailyTminMap.header->nrRows; row++)
    {
        for (long col = 0; col < dailyTminMap.header->nrCols; col++)
        {
            float airT = hourlyMeteoMaps->mapHourlyTair->value[row][col];
            if (! isEqual(airT, hourlyMeteoMaps->mapHourlyTair->header->flag))
            {
                float currentTmin = dailyTminMap.value[row][col];
                if (isEqual (currentTmin, dailyTminMap.header->flag))
                {
                    dailyTminMap.value[row][col] = airT;
                }
                else
                {
                    dailyTminMap.value[row][col] = std::min(currentTmin, airT);
                }

                float currentTmax = dailyTmaxMap.value[row][col];
                if (isEqual (currentTmax, dailyTmaxMap.header->flag))
                {
                    dailyTmaxMap.value[row][col] = airT;
                }
                else
                {
                    dailyTmaxMap.value[row][col] = std::max(currentTmax, airT);
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::checkProcesses()
{
    if (! isProjectLoaded)
    {
        errorString = ERROR_STR_MISSING_PROJECT;
        return false;
    }

    if (! (processes.computeCrop || processes.computeWater || processes.computeSnow))
    {
        errorString = "Set active processes before.";
        return false;
    }

    if (processes.computeCrop || processes.computeWater)
    {
        if (! isCriteria3DInitialized)
        {
            errorString = "Initialize 3D model before";
            return false;
        }
    }

    if (processes.computeSnow)
    {
        if (! snowMaps.isInitialized)
        {
            if (! initializeSnowModel())
                return false;
        }
    }

    return true;
}


bool Crit3DProject::runModelHour(const QString& hourlyOutputPath, bool isRestart)
{
    if (! isRestart)
    {
        QDateTime myDateTime = getCurrentTime();
        currentSeconds = 0;

        hourlyMeteoMaps->setComputed(false);
        radiationMaps->setComputed(false);

        if (processes.computeMeteo)
        {
            if (! interpolateAndSaveHourlyMeteo(airTemperature, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(precipitation, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(airRelHumidity, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(windScalarIntensity, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            hourlyMeteoMaps->setComputed(true);
            qApp->processEvents();
        }

        if (processes.computeRadiation)
        {
            if (! interpolateAndSaveHourlyMeteo(globalIrradiance, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            qApp->processEvents();
        }

        if (processes.computeSnow)
        {
            // TODO: link evaporation to water flow
            // TODO: link snowmelt to surface water content
            if (! computeSnowModel())
            {
                return false;
            }
            qApp->processEvents();
        }

        if (processes.computeWater)
        {
            // initalize sink / source
            for (unsigned long i = 0; i < nrNodes; i++)
            {
                waterSinkSource.at(size_t(i)) = 0.;
            }
        }

        if (processes.computeCrop || processes.computeWater)
        {
            if (! hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps))
            {
                errorString = "Missing ET0 values.";
                return false;
            }

            if (isSaveOutputRaster())
            {
                saveHourlyMeteoOutput(referenceEvapotranspiration, hourlyOutputPath, myDateTime);
            }

            if (processes.computeCrop)
            {
                updateDailyTemperatures();
            }
            if (processes.computeWater)
            {
                assignETreal();
            }

            qApp->processEvents();
        }

        if (processes.computeWater)
        {
            assignPrecipitation();

            if (! setSinkSource())
                return false;
        }

        emit updateOutputSignal();
    }

    // soil fluxes
    if (processes.computeWater)
    {
        if (! isRestart)
        {
            logInfo("\nCompute soil fluxes: " + getCurrentTime().toString());
        }

        runWaterFluxes3DModel(3600, isRestart);

        qApp->processEvents();
    }

    // soil heat
    if (processes.computeHeat)
    {
        //to do;
    }

    return true;
}


bool Crit3DProject::saveModelsState()
{
    QString statePath = getProjectPath() + PATH_STATES;
    if (! QDir(statePath).exists())
    {
        QDir().mkdir(statePath);
    }

    char hourStr[3];
    sprintf(hourStr, "%02d", currentHour);
    QString dateFolder = currentDate.toString("yyyyMMdd") + "_H" + hourStr;
    QString currentStatePath = statePath + "/" + dateFolder;
    if (! QDir(currentStatePath).exists())
    {
        QDir().mkdir(currentStatePath);
    }

    if (processes.computeSnow)
    {
        if (! saveSnowModelState(currentStatePath))
            return false;
    }

    if (processes.computeCrop)
    {
        // create crop path
        QString cropPath = currentStatePath + "/crop";
        if (QDir(cropPath).exists())
        {
            QDir(cropPath).removeRecursively();
        }
        QDir().mkdir(cropPath);

        // save degree days (state variable)
        std::string errorStr;
        if (! gis::writeEsriGrid((cropPath + "/degreeDays").toStdString(), &degreeDaysMap, errorStr))
        {
            logError("Error saving degree days map: " + QString::fromStdString(errorStr));
            return false;
        }
    }

    if (processes.computeWater)
    {
        if (! saveSoilWaterState(currentStatePath))
            return false;
    }

    return true;
}


bool Crit3DProject::saveSoilWaterState(const QString &currentStatePath)
{
    if (! isCriteria3DInitialized)
    {
        logError("Initialize water fluxes model before.");
        return false;
    }

    // check soil layers
    if (layerDepth.size() != nrLayers)
    {
        logError("Wrong number of layers:" + QString::number(nrLayers));
        return false;
    }

    // create water path
    QString waterPath = currentStatePath + "/water";
    if (QDir(waterPath).exists())
    {
        if (! QDir(waterPath).removeRecursively())
        {
            logError("Error deleting water directory.");
        }
    }
    QDir().mkdir(waterPath);

    // save water potential
    gis::Crit3DRasterGrid rasterGrid;
    for (unsigned int i = 0; i < nrLayers; i++)
    {
        if (! computeCriteria3DMap(rasterGrid, waterMatricPotential, i))
        {
            logError();
            return false;
        }

        int depthCm = int(round(layerDepth[i] * 100));
        QString fileName = "WP_" + QString::number(depthCm);
        std::string errorStr;
        if (! gis::writeEsriGrid((waterPath + "/" + fileName).toStdString(), &rasterGrid, errorStr))
        {
            logError("Error saving water potential: " + QString::fromStdString(errorStr));
            return false;
        }
    }

    return true;
}


bool Crit3DProject::saveSnowModelState(const QString &currentStatePath)
{
    if (! snowMaps.isInitialized)
    {
        logError("Initialize snow model before.");
        return false;
    }

    // create snow path
    QString snowPath = currentStatePath + "/snow";
    QDir().mkdir(snowPath);
    QString imgPath = snowPath + "/img";
    QDir().mkdir(imgPath);

    logInfo("Saving snow state: " + currentStatePath);
    std::string errorStr;
    if (!gis::writeEsriGrid((snowPath+"/SWE").toStdString(), snowMaps.getSnowWaterEquivalentMap(), errorStr))
    {
        logError("Error saving water equivalent map: " + QString::fromStdString(errorStr));
        return false;
    }
    // ENVI file
    if (!gis::writeEnviGrid((imgPath+"/SWE").toStdString(), gisSettings.utmZone, snowMaps.getSnowWaterEquivalentMap(), errorStr))
    {
        logError("Error saving water equivalent map (ENVI file): " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((snowPath+"/AgeOfSnow").toStdString(), snowMaps.getAgeOfSnowMap(), errorStr))
    {
        logError("Error saving age of snow map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SnowSurfaceTemp").toStdString(), snowMaps.getSnowSurfaceTempMap(), errorStr))
    {
        logError("Error saving snow surface temp map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/IceContent").toStdString(), snowMaps.getIceContentMap(), errorStr))
    {
        logError("Error saving ice content map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/LWContent").toStdString(), snowMaps.getLWContentMap(), errorStr))
    {
        logError("Error saving LW content map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/InternalEnergy").toStdString(), snowMaps.getInternalEnergyMap(), errorStr))
    {
        logError("Error saving internal energy map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SurfaceInternalEnergy").toStdString(), snowMaps.getSurfaceEnergyMap(), errorStr))
    {
        logError("Error saving surface energy map: " + QString::fromStdString(errorStr));
        return false;
    }

    return true;
}


QList<QString> Crit3DProject::getAllSavedState()
{
    QList<QString> states;
    QString statePath = getProjectPath() + PATH_STATES;
    QDir dir(statePath);
    if (! dir.exists())
    {
        errorString = "STATES directory is missing.";
        return states;
    }
    QFileInfoList list = dir.entryInfoList(QDir::AllDirs | QDir::NoDot | QDir::NoDotDot | QDir::NoSymLinks);

    if (list.size() == 0)
    {
        errorString = "STATES directory is empty.";
        return states;
    }

    for (int i=0; i < list.size(); i++)
    {
        if (list[i].baseName().size() == 12)
        {
            states << list[i].baseName();
        }
    }

    return states;
}


bool Crit3DProject::loadModelState(QString statePath)
{
    QDir stateDir(statePath);
    if (!stateDir.exists())
    {
        errorString = "State directory is missing.";
        return false;
    }

    // set current date/hour
    QString stateStr = getFileName(statePath);
    int year = stateStr.mid(0,4).toInt();
    int month = stateStr.mid(4,2).toInt();
    int day = stateStr.mid(6,2).toInt();
    int hour = stateStr.mid(10,2).toInt();
    if (hour == 24)
    {
        setCurrentDate(QDate(year, month, day).addDays(1));
        setCurrentHour(0);
    }
    else
    {
        setCurrentDate(QDate(year, month, day));
        setCurrentHour(hour);
    }

    std::string errorStr, fileName;

    // snow model
    QString snowPath = statePath + "/snow";
    QDir snowDir(snowPath);
    if (snowDir.exists())
    {
        if (! initializeSnowModel())
            return false;

        gis::Crit3DRasterGrid *tmpRaster = new gis::Crit3DRasterGrid();

        fileName = snowPath.toStdString() + "/SWE";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SWE map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSnowWaterEquivalentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/AgeOfSnow";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow AgeOfSnow map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getAgeOfSnowMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/IceContent";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow IceContent map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getIceContentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/InternalEnergy";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow InternalEnergy map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getInternalEnergyMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/LWContent";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow LWContent map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getLWContentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/SnowSurfaceTemp";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SurfaceTemp map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSnowSurfaceTempMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/SurfaceInternalEnergy";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SurfaceInternalEnergy map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSurfaceEnergyMap(), DEM.header, aggrAverage, 0.1f);

        processes.setComputeSnow(true);
    }

    // crop model
    QString cropPath = statePath + "/crop";
    QDir cropDir(cropPath);
    if (cropDir.exists())
    {
        gis::Crit3DRasterGrid myDegreeDaysMap;
        fileName = cropPath.toStdString() + "/degreeDays";
        if (! gis::readEsriGrid(fileName, &myDegreeDaysMap, errorStr))
        {
            errorString = "Wrong degree days map:\n" + QString::fromStdString(errorStr);
            return false;
        }

        if (! initializeCropFromDegreeDays(myDegreeDaysMap))
            return false;

        processes.setComputeCrop(true);
    }

    // water fluxes
    QString waterPath = statePath + "/water";
    QDir waterDir(waterPath);
    if (waterDir.exists())
    {
        if (! loadWaterPotentialState(waterPath))
        {
            isCriteria3DInitialized = false;
            processes.setComputeWater(false);
            return false;
        }

        processes.setComputeWater(true);
    }

    return true;
}


bool Crit3DProject::loadWaterPotentialState(QString waterPath)
{
    QDir waterDir(waterPath);

    QStringList filters ("*.flt");
    QFileInfoList fileList = waterDir.entryInfoList (filters);
    if (fileList.isEmpty())
    {
        errorString = "Water directory is empty.";
        return false;
    }

    if (! isCriteria3DInitialized)
    {
        logWarning("The water flow model will be initialized with the current settings.");
        if (! initializeCriteria3DModel())
        {
            logError();
            return false;
        }
    }

    std::vector<int> depthList;
    for (unsigned i = 0; i < fileList.size(); i++)
    {
        QString fileName = fileList.at(i).fileName();
        QString leftFileName = fileName.left(fileName.size() - 4);
        QString depthStr = leftFileName.right(leftFileName.size() - 3);
        bool isOk;
        int currentDepth = depthStr.toInt(&isOk);
        if (isOk)
        {
            depthList.push_back(currentDepth);
        }
    }

    if (depthList.empty())
    {
        errorString = "Missing depth in water potential fileName.";
        return false;
    }

    std::sort(depthList.begin(), depthList.end());
    double maxReadingDepth = *std::max_element(depthList.begin(), depthList.end()) / 100.;      // [m]
    double maxDepth = layerDepth[nrLayers-1];                                                   // [m]

    // check on data presence
    if (computationSoilDepth > 0)
    {
        double deltaDepth = std::max(0., maxDepth - maxReadingDepth);
        if ( (1. - deltaDepth/maxDepth) * 100 < meteoSettings->getMinimumPercentage() )
        {
            errorString = "Water potential data is not enough to cover the computation depth: "
                          + QString::number(computationSoilDepth) + " m";
            return false;
        }
    }

    std::vector<gis::Crit3DRasterGrid*> waterPotentialMapList;
    for (unsigned i = 0; i < depthList.size(); i++)
    {
        std::string fileName = waterPath.toStdString() + "/WP_" + std::to_string(depthList[i]);
        std::string errorStr;
        gis::Crit3DRasterGrid *currentWaterPotentialMap = new gis::Crit3DRasterGrid();
        if (! gis::readEsriGrid(fileName, currentWaterPotentialMap, errorStr))
        {
            errorString = "Wrong water potential map:\n" + QString::fromStdString(errorStr);
            return false;
        }
        waterPotentialMapList.push_back(currentWaterPotentialMap);
    }

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        int currentDepthCm = int(round(layerDepth[layer] * 100));
        int lastDepthIndex = int(depthList.size()) - 1;
        int layer0, layer1;
        double w0, w1;
        int i = 0;
        while (currentDepthCm > depthList[i] && i < lastDepthIndex)
        {
            i++;
        }
        if (currentDepthCm == depthList[i])
        {
            layer0 = i;
            layer1 = i;
        }
        else
        {
            if (currentDepthCm > depthList[i])
            {
                layer0 = i;
                layer1 = std::min(i+1, lastDepthIndex);
            }
            else
            {
                layer0 = std::max(0, i-1);
                layer1 = i;
            }
        }
        int delta = depthList[layer1] - depthList[layer0];
        if (delta == 0)
        {
            w0 = 1;
            w1 = 0;
        }
        else
        {
            w0 = (currentDepthCm - depthList[layer0]) / delta;
            w1 = 1 - w0;
        }

        float flag = waterPotentialMapList.at(layer0)->header->flag;
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = long(indexMap.at(layer).value[row][col]);
                if (index != long(indexMap.at(layer).header->flag))
                {
                    double x, y;
                    float waterPotential = NODATA;

                    gis::getUtmXYFromRowCol(*(indexMap.at(layer).header), row, col, &x, &y);
                    float wp0 = gis::getValueFromXY(*(waterPotentialMapList.at(layer0)), x, y);

                    if (! isEqual(wp0, flag))
                    {
                        // valid value
                        waterPotential = wp0;
                        if (w1 > 0)
                        {
                            float wp1 = gis::getValueFromXY(*(waterPotentialMapList.at(layer1)), x, y);
                            if (! isEqual(wp1, flag))
                            {
                                waterPotential = (w0 * wp0) + (w1 * wp1);
                            }
                        }
                    }
                    else
                    {
                        // search first valid value
                        int currentLayer = layer0 - 1;
                        while (isEqual(wp0, flag) && currentLayer > 0)
                        {
                            wp0 = gis::getValueFromXY(*(waterPotentialMapList.at(currentLayer)), x, y);
                            if (isEqual(wp0, flag))
                            {
                                currentLayer--;
                            }
                        }

                        if (currentLayer == 0)
                        {
                            errorString = "Missing water potential data in row, col: "
                                          + QString::number(row) + ", " +  QString::number(col);
                            return false;
                        }

                        double deltaDepth = (currentDepthCm - depthList[currentLayer]) / 100.;
                        if ( (1. - deltaDepth/maxDepth) * 100 < meteoSettings->getMinimumPercentage())
                        {
                            errorString = "The water potential data is not enough to cover the data in row, col: "
                                            + QString::number(row) + ", " +  QString::number(col);
                            return false;
                        }

                        waterPotential = wp0;
                    }

                    if (! isEqual(waterPotential, NODATA))
                    {
                        int myResult = soilFluxes3D::setMatricPotential(index, waterPotential);

                        if (isCrit3dError(myResult, errorString))
                        {
                            errorString = "Error in setMatricPotential: " + errorString + " in row:"
                                          + QString::number(row) + " col:" + QString::number(col);
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::writeOutputPointsTables()
{
    if (outputPointsDbHandler == nullptr)
    {
        errorString = "Open output DB before.";
        return false;
    }

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            QString tableName = QString::fromStdString(outputPoints[i].id);
            if (! outputPointsDbHandler->createTable(tableName, errorString))
                return false;

            if (processes.computeMeteo)
            {
                if (! outputPointsDbHandler->addColumn(tableName, airTemperature, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, precipitation, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, airRelHumidity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, windScalarIntensity, errorString)) return false;
            }
            if (processes.computeRadiation)
            {
                if (! outputPointsDbHandler->addColumn(tableName, atmTransmissivity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, globalIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, directIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, diffuseIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, reflectedIrradiance, errorString)) return false;
            }
            if (processes.computeSnow)
            {
                if (! outputPointsDbHandler->addColumn(tableName, snowWaterEquivalent, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowFall, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowMelt, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowSurfaceTemperature, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowSurfaceEnergy, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowInternalEnergy, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, sensibleHeat, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, latentHeat, errorString)) return false;
            }
            if (processes.computeWater)
            {
                for (int l = 0; l < waterContentDepth.size(); l++)
                {
                    int depth_cm = waterContentDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, volumetricWaterContent, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < waterPotentialDepth.size(); l++)
                {
                    int depth_cm = waterPotentialDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, waterMatricPotential, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < degreeOfSaturationDepth.size(); l++)
                {
                    int depth_cm = degreeOfSaturationDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, degreeOfSaturation, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < factorOfSafetyDepth.size(); l++)
                {
                    int depth_cm = factorOfSafetyDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, factorOfSafety, depth_cm, errorString))
                        return false;
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::writeOutputPointsData()
{
    QString tableName;
    std::vector<meteoVariable> meteoVarList;
    std::vector<float> meteoValuesList;
    std::vector<float> criteria3dValuesList;

    if (processes.computeMeteo)
    {
        meteoVarList.push_back(airTemperature);
        meteoVarList.push_back(precipitation);
        meteoVarList.push_back(airRelHumidity);
        meteoVarList.push_back(windScalarIntensity);
    }
    if (processes.computeRadiation)
    {
        meteoVarList.push_back(atmTransmissivity);
        meteoVarList.push_back(globalIrradiance);
        meteoVarList.push_back(directIrradiance);
        meteoVarList.push_back(diffuseIrradiance);
        meteoVarList.push_back(reflectedIrradiance);
    }
    if (processes.computeSnow)
    {
        meteoVarList.push_back(snowWaterEquivalent);
        meteoVarList.push_back(snowFall);
        meteoVarList.push_back(snowMelt);
        meteoVarList.push_back(snowSurfaceTemperature);
        meteoVarList.push_back(snowSurfaceEnergy);
        meteoVarList.push_back(snowInternalEnergy);
        meteoVarList.push_back(sensibleHeat);
        meteoVarList.push_back(latentHeat);
    }

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            double x = outputPoints[i].utm.x;
            double y = outputPoints[i].utm.y;
            tableName = QString::fromStdString(outputPoints[i].id);

            if (processes.computeMeteo)
            {
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyTair->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyPrec->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyRelHum->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyWindScalarInt->getValueFromXY(x, y));
            }
            if (processes.computeRadiation)
            {
                meteoValuesList.push_back(radiationMaps->transmissivityMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->globalRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->beamRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->diffuseRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->reflectedRadiationMap->getValueFromXY(x, y));
            }
            if (processes.computeSnow)
            {
                meteoValuesList.push_back(snowMaps.getSnowWaterEquivalentMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowFallMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowMeltMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowSurfaceTempMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSurfaceEnergyMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getInternalEnergyMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSensibleHeatMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getLatentHeatMap()->getValueFromXY(x, y));
            }
            if (processes.computeWater)
            {
                int row, col;
                gis::getRowColFromXY((*DEM.header), x, y, &row, &col);

                appendCriteria3DOutputValue(volumetricWaterContent, row, col, waterContentDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(waterMatricPotential, row, col, waterPotentialDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(degreeOfSaturation, row, col, degreeOfSaturationDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(factorOfSafety, row, col, factorOfSafetyDepth, criteria3dValuesList);
            }

            if (! outputPointsDbHandler->saveHourlyMeteoData(tableName, getCurrentTime(), meteoVarList, meteoValuesList, errorString))
            {
                return false;
            }
            if (! outputPointsDbHandler->saveHourlyCriteria3D_Data(tableName, getCurrentTime(), criteria3dValuesList,
                                                                  waterContentDepth, waterPotentialDepth,
                                                                  degreeOfSaturationDepth, factorOfSafetyDepth, errorString))
            {
                return false;
            }

            meteoValuesList.clear();
            criteria3dValuesList.clear();
        }
    }

    meteoVarList.clear();

    return true;
}


void Crit3DProject::appendCriteria3DOutputValue(criteria3DVariable myVar, int row, int col,
                                                const std::vector<int> &depthList, std::vector<float> &outputList)
{
    for (int l = 0; l < depthList.size(); l++)
    {
        float depth = depthList[l] * 0.01;                          // [cm] -> [m]
        int layerIndex = getSoilLayerIndex(depth);
        long nodeIndex = indexMap.at(layerIndex).value[row][col];
        float value = NODATA;

        if (nodeIndex != indexMap.at(layerIndex).header->flag)
        {
            if (myVar == factorOfSafety)
            {
                value = computeFactorOfSafety(row, col, layerIndex);
            }
            else
            {
                value = getCriteria3DVar(myVar, nodeIndex);
            }
        }

        outputList.push_back(value);
    }
}


bool Crit3DProject::writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList, const QList<QString> &csvFields,
                                              const QList<QList<QString>> &csvData)
{
    QList<QString> propertiesList;
    QList<int> posValues;

    for (int i = 0; i < joinedPropertiesList.size(); i++)
    {
        QList<QString> couple = joinedPropertiesList[i].split("-->");
        QString pragaProperty = couple[0];
        QString csvProperty = couple[1];
        int pos = csvFields.indexOf(csvProperty);
        if (pos != -1)
        {
            propertiesList << pragaProperty;
            posValues << pos;
        }
    }

    for (int row = 0; row < csvData.size(); row++)
    {
        QList<QString> csvDataList;

        for (int j = 0; j < posValues.size(); j++)
        {
            csvDataList << csvData[row][posValues[j]];
        }

        if (! meteoPointsDbHandler->updatePointProperties(propertiesList, csvDataList))
        {
            errorString = meteoPointsDbHandler->getErrorString();
            return false;
        }
    }

    return true;
}


//------------------------------------- 3D geometry and color --------------------------------------

void Crit3DProject::clearGeometry()
{
    if (openGlGeometry != nullptr)
    {
        openGlGeometry->clear();
        delete openGlGeometry;
        openGlGeometry = nullptr;
    }
}


bool Crit3DProject::initializeGeometry()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    this->clearGeometry();
    openGlGeometry = new Crit3DGeometry();

    // set center
    gis::Crit3DPoint center = DEM.getCenter();
    gis::updateMinMaxRasterGrid(&DEM);
    float zCenter = (DEM.maximum + DEM.minimum) * 0.5f;
    openGlGeometry->setCenter(float(center.utm.x), float(center.utm.y), zCenter);

    // set dimension
    float dx = float(DEM.header->nrCols * DEM.header->cellSize);
    float dy = float(DEM.header->nrRows * DEM.header->cellSize);
    float dz = DEM.maximum + DEM.minimum;
    openGlGeometry->setDimension(dx, dy);
    float magnify = ((dx + dy) * 0.5f) / (dz * 10.f);
    openGlGeometry->setMagnify(std::min(5.f, std::max(1.f, magnify)));

    // set triangles
    double x, y;
    float z1, z2, z3;
    gis::Crit3DPoint p1, p2, p3;
    Crit3DColor *c1, *c2, *c3;
    Crit3DColor sc1, sc2, sc3;
    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            z1 = DEM.getValueFromRowCol(row, col);
            if (! isEqual(z1, DEM.header->flag))
            {
                DEM.getXY(row, col, x, y);
                p1 = gis::Crit3DPoint(x, y, z1);
                c1 = DEM.colorScale->getColor(z1);
                shadowColor(*c1, sc1, row, col);

                z3 = DEM.getValueFromRowCol(row+1, col+1);
                if (! isEqual(z3, DEM.header->flag))
                {
                    DEM.getXY(row+1, col+1, x, y);
                    p3 = gis::Crit3DPoint(x, y, z3);
                    c3 = DEM.colorScale->getColor(z3);
                    shadowColor(*c3, sc3, row+1, col+1);

                    z2 = DEM.getValueFromRowCol(row+1, col);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        DEM.getXY(row+1, col, x, y);
                        p2 = gis::Crit3DPoint(x, y, z2);
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row+1, col);
                        openGlGeometry->addTriangle(p1, p2, p3, sc1, sc2, sc3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        DEM.getXY(row, col+1, x, y);
                        p2 = gis::Crit3DPoint(x, y, z2);
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row, col+1);
                        openGlGeometry->addTriangle(p3, p2, p1, sc3, sc2, sc1);
                    }
                }
            }
        }
    }

    return true;
}


void Crit3DProject::shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col)
{
    colorOut.red = colorIn.red;
    colorOut.green = colorIn.green;
    colorOut.blue = colorIn.blue;

    float aspect = radiationMaps->aspectMap->getValueFromRowCol(row, col);
    if (! isEqual(aspect, radiationMaps->aspectMap->header->flag))
    {
        float slopeDegree = radiationMaps->slopeMap->getValueFromRowCol(row, col);
        if (! isEqual(slopeDegree, radiationMaps->slopeMap->header->flag))
        {
            float slopeAmplification = 120.f / std::max(radiationMaps->slopeMap->maximum, 1.f);
            float shadow = -cos(aspect * DEG_TO_RAD) * std::max(5.f, slopeDegree * slopeAmplification);

            colorOut.red = std::min(255, std::max(0, int(colorOut.red + shadow)));
            colorOut.green = std::min(255, std::max(0, int(colorOut.green + shadow)));
            colorOut.blue = std::min(255, std::max(0, int(colorOut.blue + shadow)));
            if (slopeDegree > openGlGeometry->artifactSlope())
            {
                colorOut.red = std::min(255, std::max(0, int((colorOut.red + 256) / 2)));
                colorOut.green = std::min(255, std::max(0, int((colorOut.green + 256) / 2)));
                colorOut.blue = std::min(255, std::max(0, int((colorOut.blue + 256) / 2)));
            }
        }
    }
}


bool Crit3DProject::update3DColors(gis::Crit3DRasterGrid *rasterPointer)
{
    if (openGlGeometry == nullptr)
    {
        errorString = "Initialize 3D openGlGeometry before.";
        return false;
    }

    bool isShowVariable = false;
    if (rasterPointer != nullptr)
    {
        if (rasterPointer->header->isEqualTo(*(DEM.header)))
        {
            isShowVariable = true;
        }
    }

    float z1, z2, z3, value;
    Crit3DColor dtmColor1, dtmColor2, dtmColor3;
    Crit3DColor color1, color2, color3;             // final colors

    double variableRange = 0;
    if (isShowVariable)
    {
        variableRange = std::max(EPSILON, rasterPointer->colorScale->maximum() - rasterPointer->colorScale->minimum());
    }

    long i = 0;
    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            z1 = DEM.getValueFromRowCol(row, col);
            if (! isEqual(z1, DEM.header->flag))  
            {
                z3 = DEM.getValueFromRowCol(row+1, col+1);
                if (! isEqual(z3, DEM.header->flag))
                {
                    Crit3DColor* c1 = DEM.colorScale->getColor(z1);
                    shadowColor(*c1, dtmColor1, row, col);
                    color1 = dtmColor1;

                    Crit3DColor* c3 = DEM.colorScale->getColor(z3);
                    shadowColor(*c3, dtmColor3, row+1, col+1);
                    color3 = dtmColor3;

                    if (isShowVariable)
                    {
                        value = rasterPointer->getValueFromRowCol(row, col);
                        if (! isEqual(value, rasterPointer->header->flag))
                        {
                            Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                            double alpha = 0.66;

                            // check outliers
                            if (rasterPointer->colorScale->isHideOutliers())
                            {
                                if (value <= rasterPointer->colorScale->minimum()
                                    || value > rasterPointer->colorScale->maximum())
                                    alpha = 0;
                            }
                            if (rasterPointer->colorScale->isTransparent())
                            {
                                double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                alpha = sqrt(std::min(1., step/variableRange));
                            }
                            mixColor(dtmColor1, *variableColor, color1, alpha);
                        }

                        value = rasterPointer->getValueFromRowCol(row+1, col+1);
                        if (! isEqual(value, rasterPointer->header->flag))
                        {
                            Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                            double alpha = 0.66;

                            // check outliers
                            if (rasterPointer->colorScale->isHideOutliers())
                            {
                                if (value <= rasterPointer->colorScale->minimum()
                                    || value > rasterPointer->colorScale->maximum())
                                    alpha = 0;
                            }
                            if (rasterPointer->colorScale->isTransparent())
                            {
                                double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                alpha = sqrt(std::min(1., step/variableRange));
                            }
                            mixColor(dtmColor3, *variableColor, color3, alpha);
                        }
                    }

                    z2 = DEM.getValueFromRowCol(row+1, col);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        Crit3DColor* c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, dtmColor2, row+1, col);
                        color2 = dtmColor2;

                        if (isShowVariable)
                        {
                            value = rasterPointer->getValueFromRowCol(row+1, col);
                            if (! isEqual(value, rasterPointer->header->flag))
                            {
                                Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                                double alpha = 0.66;

                                // check outliers
                                if (rasterPointer->colorScale->isHideOutliers())
                                {
                                    if (value <= rasterPointer->colorScale->minimum()
                                        || value > rasterPointer->colorScale->maximum())
                                        alpha = 0;
                                }
                                if (rasterPointer->colorScale->isTransparent())
                                {
                                    double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                    alpha = sqrt(std::min(1., step/variableRange));
                                }
                                mixColor(dtmColor2, *variableColor, color2, alpha);
                            }
                        }

                        openGlGeometry->setVertexColor(i++, color1);
                        openGlGeometry->setVertexColor(i++, color2);
                        openGlGeometry->setVertexColor(i++, color3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        Crit3DColor* c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, dtmColor2, row, col+1);
                        color2 = dtmColor2;

                        if (isShowVariable)
                        {
                            value = rasterPointer->getValueFromRowCol(row, col+1);
                            if (! isEqual(value, rasterPointer->header->flag))
                            {
                                Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                                double alpha = 0.66;

                                // check outliers
                                if (rasterPointer->colorScale->isHideOutliers())
                                {
                                    if (value <= rasterPointer->colorScale->minimum()
                                        || value > rasterPointer->colorScale->maximum())
                                        alpha = 0;
                                }
                                if (rasterPointer->colorScale->isTransparent())
                                {
                                    double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                    alpha = sqrt(std::min(1., step/variableRange));
                                }
                                mixColor(dtmColor2, *variableColor, color2, alpha);
                            }
                        }

                        openGlGeometry->setVertexColor(i++, color3);
                        openGlGeometry->setVertexColor(i++, color2);
                        openGlGeometry->setVertexColor(i++, color1);
                    }
                }
            }
        }
    }

    return true;
}

