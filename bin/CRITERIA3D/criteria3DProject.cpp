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

#include <QtSql>
#include <QPaintEvent>
#include <QVector3D>


Crit3DProject::Crit3DProject() : Project3D()
{
    _saveOutputRaster = false;
    _saveOutputPoints = false;
    _saveDailyState = false;

    modelPause = false;
    modelStop = false;

    modelFirstTime.setTimeSpec(Qt::UTC);
    modelLastTime.setTimeSpec(Qt::UTC);
}


bool Crit3DProject::initializeCriteria3DModel()
{
    if (! check3DProject())
    {
        logError();
        return false;
    }

    clearWaterBalance3D();

    if (! setSoilIndexMap())
        return false;

    if (! initializeWaterBalance3D())
    {
        clearWaterBalance3D();
        logError("Criteria3D model is NOT initialized.");
        return false;
    }

    initializeCrop();

    isCriteria3DInitialized = true;
    logInfoGUI("Criteria3D model initialized");

    return true;
}


void Crit3DProject::initializeCrop()
{
    // initialize LAI and degree days map to NODATA
    laiMap.initializeGrid(*(DEM.header));
    degreeDaysMap.initializeGrid(*(DEM.header));

    if (! processes.computeCrop)
    {
        // nothing to do
        return;
    }

    dailyTminMap.initializeGrid(*(DEM.header));
    dailyTmaxMap.initializeGrid(*(DEM.header));

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // is land unit
                int index = getLandUnitIndexRowCol(row, col);
                if (index != NODATA)
                {
                    // is crop
                    if (landUnitList[index].idCrop != "")
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
    }
}


void Crit3DProject::dailyUpdateCrop()
{
    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // is crop
                int index = getLandUnitIndexRowCol(row, col);
                if (index != NODATA && landUnitList[index].idCrop != "")
                {
                    float tmin = dailyTminMap.value[row][col];
                    float tmax = dailyTminMap.value[row][col];
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

    // clean daily temp maps
    dailyTminMap.emptyGrid();
    dailyTmaxMap.emptyGrid();
}


/*!
 * \brief assignETreal
 *  assign soil evaporation and crop transpiration for the whole domain
 */
void Crit3DProject::assignETreal()
{
    totalEvaporation = 0;
    totalTranspiration = 0;

    double area = DEM.header->cellSize * DEM.header->cellSize;

    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                float lai = laiMap.value[row][col];
                if (isEqual(lai, NODATA))
                {
                    lai = 0;
                }

                // assign real evaporation
                double realEvap = assignEvaporation(row, col, lai);             // [mm]
                double evapFlow = area * (realEvap / 1000.);                    // [m3 h-1]
                totalEvaporation += evapFlow;

                // assign real transpiration
                if (lai > 0)
                {
                    float degreeDays = degreeDaysMap.value[row][col];
                    double realTransp = assignTranspiration(row, col, lai, degreeDays);     // [mm]
                    double traspFlow = area * (realTransp / 1000.);                         // [m3 h-1]
                    totalTranspiration += traspFlow;
                }
            }
        }
    }
}


void Crit3DProject::assignPrecipitation()
{
    // initialize
    totalPrecipitation = 0;

    double area = DEM.header->cellSize * DEM.header->cellSize;

    // precipitation
    for (long row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (long col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                double waterSource = 0;
                float prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                if (! isEqual(prec, hourlyMeteoMaps->mapHourlyPrec->header->flag))
                    waterSource += prec;

                if (waterSource > 0)
                {
                    double flow = area * (waterSource / 1000.);                     // [m3 h-1]
                    waterSinkSource[unsigned(surfaceIndex)] += flow / 3600.;        // [m3 s-1]
                    totalPrecipitation += flow;
                }
            }
        }
    }
}


bool Crit3DProject::runModels(QDateTime firstTime, QDateTime lastTime)
{
    // initialize meteo
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

    // initialize radiation
    if (processes.computeRadiation)
    {
        radiationMaps->initialize();
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
            dailyUpdateCrop();
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

        // TODO update crop roots

        // cycle on hours
        int firstHour = (myDate == firstDate) ? hour1 : 0;
        int lastHour = (myDate == lastDate) ? hour2 : 23;

        for (currentHour = firstHour; currentHour <= lastHour; currentHour++)
        {
            QDateTime myTime = QDateTime(myDate, QTime(currentHour, 0, 0), Qt::UTC);

            if (! modelHourlyCycle(myTime, currentOutputPath))
            {
                logError();
                return false;
            }

            emit updateOutputSignal();

            // output points
            if (isSaveOutputPoints())
            {
                if (! writeOutputPointsData())
                {
                    logError();
                    return false;
                }
            }

            if (modelPause || modelStop)
            {
                return true;
            }
        }

        if (isSaveDailyState())
        {
            saveModelState();
        }
    }

    closeLogInfo();
    return true;
}


void Crit3DProject::setSaveDailyState(bool isSave)
{
    _saveDailyState = isSave;
}

bool Crit3DProject::isSaveDailyState()
{
    return _saveDailyState;
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

    snowModel.initialize();

    if (! loadProjectSettings(myFileName))
        return false;

    if (! loadProject3DSettings())
        return false;

    if (! loadProject())
    {
        if (errorType != ERROR_DBGRID)
            return false;
    }

    // only for 3d model
    if (meteoPointsLoaded)
    {
        meteoPointsDbFirstTime = findDbPointFirstTime();
    }

    if (! loadCriteria3DParameters())
        return false;

    // soil map and data
    if (soilMapFileName != "") loadSoilMap(soilMapFileName);
    if (soilDbFileName != "") loadSoilDatabase(soilDbFileName);

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
    if (parameters == nullptr)
    {
        logError("parameters is null");
        return false;
    }
    Q_FOREACH (QString group, parameters->childGroups())
    {
        if (group == "snow")
        {
            parameters->beginGroup(group);
            if (parameters->contains("tempMaxWithSnow") && !parameters->value("tempMaxWithSnow").toString().isEmpty())
            {
                snowModel.snowParameters.tempMaxWithSnow = parameters->value("tempMaxWithSnow").toDouble();
            }
            if (parameters->contains("tempMinWithRain") && !parameters->value("tempMinWithRain").toString().isEmpty())
            {
                snowModel.snowParameters.tempMinWithRain = parameters->value("tempMinWithRain").toDouble();
            }
            if (parameters->contains("snowWaterHoldingCapacity") && !parameters->value("snowWaterHoldingCapacity").toString().isEmpty())
            {
                snowModel.snowParameters.snowWaterHoldingCapacity = parameters->value("snowWaterHoldingCapacity").toDouble();
            }
            if (parameters->contains("snowSkinThickness") && !parameters->value("snowSkinThickness").toString().isEmpty())
            {
                snowModel.snowParameters.snowSkinThickness = parameters->value("snowSkinThickness").toDouble();
            }
            if (parameters->contains("snowVegetationHeight") && !parameters->value("snowVegetationHeight").toString().isEmpty())
            {
                snowModel.snowParameters.snowVegetationHeight = parameters->value("snowVegetationHeight").toDouble();
            }
            if (parameters->contains("soilAlbedo") && !parameters->value("soilAlbedo").toString().isEmpty())
            {
                snowModel.snowParameters.soilAlbedo = parameters->value("soilAlbedo").toDouble();
            }
            parameters->endGroup();
        }
    }
    return true;
}

bool Crit3DProject::writeCriteria3DParameters()
{
    QString fileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing parameters file: " + fileName);
        return false;
    }
    if (parameters == nullptr)
    {
        logError("parameters is null");
        return false;
    }

    parameters->setValue("snow/tempMaxWithSnow", snowModel.snowParameters.tempMaxWithSnow);
    parameters->setValue("snow/tempMinWithRain", snowModel.snowParameters.tempMinWithRain);
    parameters->setValue("snow/snowWaterHoldingCapacity", snowModel.snowParameters.snowWaterHoldingCapacity);
    parameters->setValue("snow/snowSkinThickness", snowModel.snowParameters.snowSkinThickness);
    parameters->setValue("snow/snowVegetationHeight", snowModel.snowParameters.snowVegetationHeight);
    parameters->setValue("snow/soilAlbedo", snowModel.snowParameters.soilAlbedo);

    parameters->sync();
    return true;
}


bool Crit3DProject::loadSoilMap(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing soil map filename");
        return false;
    }

    soilMapFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_GEO);

    std::string errorStr;
    if (! gis::openRaster(fileName.toStdString(), &soilMap, gisSettings.utmZone, errorStr))
    {
        logError("Load soil map failed: " + fileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    logInfo("Soil map = " + fileName);
    return true;
}


bool Crit3DProject::loadLandUseMap(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing land use map filename");
        return false;
    }

    landUseMapFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_GEO);

    std::string errorStr;
    if (! gis::openRaster(fileName.toStdString(), &landUseMap, gisSettings.utmZone, errorStr))
    {
        logError("Load land use map failed: " + fileName + "\n" + QString::fromStdString(errorStr));
        return false;
    }

    logInfo("Land use map = " + fileName);
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


bool Crit3DProject::setSoilIndexMap()
{
    double x, y;
    soilIndexMap.initializeGrid(*(DEM.header));

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                DEM.getXY(row, col, x, y);
                int soilIndex = getCrit3DSoilIndex(x, y);
                if (soilIndex != NODATA)
                    soilIndexMap.value[row][col] = float(soilIndex);
            }
        }
    }

    soilIndexMap.isLoaded = true;
    return true;
}


int Crit3DProject::getCrit3DSoilId(double x, double y)
{
    if (! soilMap.isLoaded)
        return NODATA;

    int idSoil = int(gis::getValueFromXY(soilMap, x, y));

    if (idSoil == int(soilMap.header->flag))
    {
        return NODATA;
    }
    else
    {
        return idSoil;
    }
}


int Crit3DProject::getCrit3DSoilIndex(double x, double y)
{
    int idSoil = getCrit3DSoilId(x, y);
    if (idSoil == NODATA) return NODATA;

    for (unsigned int index = 0; index < soilList.size(); index++)
    {
        if (soilList[index].id == idSoil)
        {
            return signed(index);
        }
    }

    return NODATA;
}


QString Crit3DProject::getCrit3DSoilCode(double x, double y)
{
    int idSoil = getCrit3DSoilId(x, y);
    if (idSoil == NODATA) return "";

    for (unsigned int i = 0; i < soilList.size(); i++)
    {
        if (soilList[i].id == idSoil)
        {
            return QString::fromStdString(soilList[i].code);
        }
    }

    return "NOT FOUND";
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
    laiMap.clear();

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
        logError("Load a DTM before.");
        return false;
    }

    snowMaps.initialize(DEM, snowModel.snowParameters.snowSkinThickness);
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
    double myWaterContent = 0;

    snowModel.setInputData(airT, prec, relHum, windInt, globalRad, beamRad, transmissivity, clearSkyTrans, myWaterContent);

    snowModel.computeSnowBrooksModel();

    snowMaps.updateMap(snowModel, row, col);
}


// it assumes that header of meteo and snow maps = header of DEM
bool Crit3DProject::computeSnowModel()
{
    // check
    if (! hourlyMeteoMaps->getComputed())
    {
        logError("Missing meteo map!");
        return false;
    }

    if (! radiationMaps->getComputed())
    {
        logError("Missing radiation map!");
        return false;
    }

    if (! snowMaps.isInitialized)
    {
        if (! initializeSnowModel())
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


bool Crit3DProject::modelHourlyCycle(QDateTime myTime, const QString& hourlyOutputPath)
{
    hourlyMeteoMaps->setComputed(false);
    radiationMaps->setComputed(false);

    if (processes.computeMeteo)
    {
        if (! interpolateAndSaveHourlyMeteo(airTemperature, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(precipitation, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(airRelHumidity, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(windScalarIntensity, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();

        hourlyMeteoMaps->setComputed(true);
    }

    if (processes.computeRadiation)
    {
        if (! interpolateAndSaveHourlyMeteo(globalIrradiance, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();
    }

    if (processes.computeSnow)
    {
        // check evaporation
        // check snowmelt -> surface H0
        if (! computeSnowModel())
        {
            return false;
        }
        qApp->processEvents();
    }

    // initalize sink / source
    for (unsigned long i = 0; i < nrNodes; i++)
    {
        waterSinkSource.at(size_t(i)) = 0.;
    }

    if (processes.computeEvaporation || processes.computeCrop)
    {
        if (! hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps))
            return false;

        if (isSaveOutputRaster())
        {
            saveHourlyMeteoOutput(referenceEvapotranspiration, hourlyOutputPath, myTime);
        }

        assignETreal();

        qApp->processEvents();
    }

    if (processes.computeCrop)
    {
        updateDailyTemperatures();
        qApp->processEvents();
    }

    // soil water balance
    if (processes.computeWater)
    {
        assignPrecipitation();

        if (! setSinkSource())
        {
            logError();
            return false;
        }

        logInfo("\nWater balance: " + myTime.toString());
        computeWaterBalance3D(3600);
    }

    // soil heat
    if (processes.computeHeat)
    {
        //to do;
    }

    return true;
}


bool Crit3DProject::saveModelState()
{
    if (! snowMaps.isInitialized)
    {
        logError("Initialize snow model before.");
        return false;
    }

    QString statePath = getProjectPath() + PATH_STATES;
    if (!QDir(statePath).exists())
    {
        QDir().mkdir(statePath);
    }

    char hourStr[3];
    sprintf(hourStr, "%02d", currentHour);
    QString dateFolder = currentDate.toString("yyyyMMdd") + "_H" + hourStr;
    if (!QDir(statePath+"/"+dateFolder).exists())
    {
        QDir().mkdir(statePath + "/" + dateFolder);
    }

    // create snow path
    QString snowPath = statePath + "/" + dateFolder + "/snow";
    QDir().mkdir(snowPath);
    QString imgPath = snowPath + "/img";
    QDir().mkdir(imgPath);

    logInfo("Saving snow state: " + dateFolder);
    std::string error;
    if (!gis::writeEsriGrid((snowPath+"/SWE").toStdString(), snowMaps.getSnowWaterEquivalentMap(), error))
    {
        logError("Error saving water equivalent map: " + QString::fromStdString(error));
        return false;
    }
    // ENVI file
    if (!gis::writeEnviGrid((imgPath+"/SWE").toStdString(), gisSettings.utmZone, snowMaps.getSnowWaterEquivalentMap(), error))
    {
        logError("Error saving water equivalent map (ENVI file): " + QString::fromStdString(error));
        return false;
    }

    if (!gis::writeEsriGrid((snowPath+"/AgeOfSnow").toStdString(), snowMaps.getAgeOfSnowMap(), error))
    {
        logError("Error saving age of snow map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SnowSurfaceTemp").toStdString(), snowMaps.getSnowSurfaceTempMap(), error))
    {
        logError("Error saving snow surface temp map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/IceContent").toStdString(), snowMaps.getIceContentMap(), error))
    {
        logError("Error saving ice content map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/LWContent").toStdString(), snowMaps.getLWContentMap(), error))
    {
        logError("Error saving LW content map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/InternalEnergy").toStdString(), snowMaps.getInternalEnergyMap(), error))
    {
        logError("Error saving internal energy map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SurfaceInternalEnergy").toStdString(), snowMaps.getSurfaceEnergyMap(), error))
    {
        logError("Error saving surface energy map: " + QString::fromStdString(error));
        return false;
    }

    return true;
}


QList<QString> Crit3DProject::getAllSavedState()
{
    QList<QString> states;
    QString statePath = getProjectPath() + PATH_STATES;
    QDir dir(statePath);
    if (!dir.exists())
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

    for (int i=0; i<list.size(); i++)
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
    setCurrentDate(QDate(year, month, day));
    setCurrentHour(hour);

    // snow model
    QString snowPath = statePath + "/snow";
    QDir snowDir(snowPath);
    if (snowDir.exists())
    {
        if (! initializeSnowModel())
            return false;

        std::string error;
        std::string fileName;

        fileName = snowPath.toStdString() + "/SWE";
        if (! gis::readEsriGrid(fileName, snowMaps.getSnowWaterEquivalentMap(), error))
        {
            errorString = "Wrong Snow SWE map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/AgeOfSnow";
        if (! gis::readEsriGrid(fileName, snowMaps.getAgeOfSnowMap(), error))
        {
            errorString = "Wrong Snow AgeOfSnow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/IceContent";
        if (! gis::readEsriGrid(fileName, snowMaps.getIceContentMap(), error))
        {
            errorString = "Wrong Snow IceContent map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/InternalEnergy";
        if (! gis::readEsriGrid(fileName, snowMaps.getInternalEnergyMap(), error))
        {
            errorString = "Wrong Snow InternalEnergy map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/LWContent";
        if (! gis::readEsriGrid(fileName, snowMaps.getLWContentMap(), error))
        {
            errorString = "Wrong Snow LWContent map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/SnowSurfaceTemp";
        if (! gis::readEsriGrid(fileName, snowMaps.getSnowSurfaceTempMap(), error))
        {
            errorString = "Wrong Snow SurfaceTemp map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/SurfaceInternalEnergy";
        if (! gis::readEsriGrid(fileName, snowMaps.getSurfaceEnergyMap(), error))
        {
            errorString = "Wrong Snow SurfaceInternalEnergy map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
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
        }
    }

    return true;
}


bool Crit3DProject::writeOutputPointsData()
{
    QString tableName;
    std::vector<meteoVariable> varList;
    std::vector<float> valuesList;

    if (processes.computeMeteo)
    {
        varList.push_back(airTemperature);
        varList.push_back(precipitation);
        varList.push_back(airRelHumidity);
        varList.push_back(windScalarIntensity);
    }
    if (processes.computeRadiation)
    {
        varList.push_back(atmTransmissivity);
        varList.push_back(globalIrradiance);
        varList.push_back(directIrradiance);
        varList.push_back(diffuseIrradiance);
        varList.push_back(reflectedIrradiance);
    }
    if (processes.computeSnow)
    {
        varList.push_back(snowWaterEquivalent);
        varList.push_back(snowFall);
        varList.push_back(snowMelt);
        varList.push_back(snowSurfaceTemperature);
        varList.push_back(snowSurfaceEnergy);
        varList.push_back(snowInternalEnergy);
        varList.push_back(sensibleHeat);
        varList.push_back(latentHeat);
    }

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            float x = float(outputPoints[i].utm.x);
            float y = float(outputPoints[i].utm.y);
            tableName = QString::fromStdString(outputPoints[i].id);
            if (processes.computeMeteo)
            {
                valuesList.push_back(hourlyMeteoMaps->mapHourlyTair->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyPrec->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyRelHum->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyWindScalarInt->getValueFromXY(x, y));
            }
            if (processes.computeRadiation)
            {
                valuesList.push_back(radiationMaps->transmissivityMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->globalRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->beamRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->diffuseRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->reflectedRadiationMap->getValueFromXY(x, y));
            }
            if (processes.computeSnow)
            {
                valuesList.push_back(snowMaps.getSnowWaterEquivalentMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getSnowFallMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getSnowMeltMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getSnowSurfaceTempMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getSurfaceEnergyMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getInternalEnergyMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getSensibleHeatMap()->getValueFromXY(x, y));
                valuesList.push_back(snowMaps.getLatentHeatMap()->getValueFromXY(x, y));
            }

            if (! outputPointsDbHandler->saveHourlyData(tableName, getCurrentTime(), varList, valuesList, errorString))
            {
                return false;
            }
            valuesList.clear();
        }
    }
    varList.clear();

    return true;
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
            if (slope > openGlGeometry->artifactSlope())
            {
                colorOut.red = std::min(255, std::max(0, int((colorOut.red + 256) / 2)));
                colorOut.green = std::min(255, std::max(0, int((colorOut.green + 256) / 2)));
                colorOut.blue = std::min(255, std::max(0, int((colorOut.blue + 256) / 2)));
            }
        }
    }
}


bool Crit3DProject::update3DColors()
{
    if (openGlGeometry == nullptr)
    {
        errorString = "Initialize 3D openGlGeometry before.";
        return false;
    }

    float z1, z2, z3;
    Crit3DColor *c1, *c2, *c3;
    Crit3DColor sc1, sc2, sc3;
    long i = 0;
    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            z1 = DEM.getValueFromRowCol(row, col);
            if (! isEqual(z1, DEM.header->flag))
            {
                c1 = DEM.colorScale->getColor(z1);
                shadowColor(*c1, sc1, row, col);
                z3 = DEM.getValueFromRowCol(row+1, col+1);
                if (! isEqual(z3, DEM.header->flag))
                {
                    c3 = DEM.colorScale->getColor(z3);
                    shadowColor(*c3, sc3, row+1, col+1);
                    z2 = DEM.getValueFromRowCol(row+1, col);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row+1, col);
                        openGlGeometry->setVertexColor(i++, sc1);
                        openGlGeometry->setVertexColor(i++, sc2);
                        openGlGeometry->setVertexColor(i++, sc3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row, col+1);
                        openGlGeometry->setVertexColor(i++, sc3);
                        openGlGeometry->setVertexColor(i++, sc2);
                        openGlGeometry->setVertexColor(i++, sc1);
                    }
                }
            }
        }
    }

    return true;
}

