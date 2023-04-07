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
#include "soilDbTools.h"
#include "gis.h"
#include "color.h"
#include "statistics.h"

#include <QtSql>
#include <QPaintEvent>
#include <QVector3D>


Crit3DProject::Crit3DProject() : Project3D()
{
    _saveOutputRaster = false;
    _saveOutputPoints = false;
    _saveDailyState = false;

    computeMeteo = false;
    computeRadiation = false;
    computeCrop = false;
    computeWater = false;
    computeSnow = false;
    computeHeat = false;
    computeSolutes = false;
    computeAdvectiveHeat = false;
    computeLatentHeat = false;

    modelPause = false;
    modelStop = false;

    modelFirstTime.setTimeSpec(Qt::UTC);
    modelLastTime.setTimeSpec(Qt::UTC);
}


bool Crit3DProject::initializeCriteria3DModel()
{
    /*if (! check3DProject())
    {
        logError();
        return false;
    }*/

    clearWaterBalance3D();

    if (! setSoilIndexMap())
        return false;

    // TODO set soilUseMap()

    /* TODO initialize root density
    // andrebbe rifatto per ogni tipo di suolo (ora considera solo suolo 0)
    int nrSoilLayersWithoutRoots = 2;
    int soilLayerWithRoot = this->nrSoilLayers - nrSoilLayersWithoutRoots;
    double depthModeRootDensity = 0.35*this->soilDepth;     //[m] depth of mode of root density
    double depthMeanRootDensity = 0.5*this->soilDepth;      //[m] depth of mean of root density
    initializeRootProperties(&(this->soilList[0]), this->nrSoilLayers, this->computationSoilDepth,
                         this->layerDepth.data(), this->layerThickness.data(),
                         nrSoilLayersWithoutRoots, soilLayerWithRoot,
                         GAMMA_DISTRIBUTION, depthModeRootDensity, depthMeanRootDensity);
    */

    if (! initializeWaterBalance3D())
    {
        clearWaterBalance3D();
        logError("Criteria3D model not initialized.");
        return false;
    }

    isCriteria3DInitialized = true;
    logInfoGUI("Criteria3D model initialized");

    return true;
}


bool Crit3DProject::runModels(QDateTime firstTime, QDateTime lastTime)
{
    // initialize meteo
    if (computeMeteo)
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
    if (computeRadiation)
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

    if (! loadCriteria3DSettings())
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

    // soiluse map and data
    // TODO soilUse map

    if (cropDbFileName != "") loadCropDatabase(cropDbFileName);

    if (projectName != "")
    {
        logInfo("Project " + projectName + " loaded");
    }

    isProjectLoaded = true;
    return isProjectLoaded;
}


bool Crit3DProject::loadCriteria3DSettings()
{
    projectSettings->beginGroup("project");
        cropDbFileName = projectSettings->value("crop_db").toString();
        if (cropDbFileName == "")
            cropDbFileName = projectSettings->value("db_crop").toString();
        soilMapFileName = projectSettings->value("soil_map").toString();
    projectSettings->endGroup();

    projectSettings->beginGroup("simulation");
        computeHeat = projectSettings->value("compute_heat").toBool();
        computeLatentHeat = projectSettings->value("compute_latent_heat").toBool();
        computeAdvectiveHeat = projectSettings->value("compute_advective_heat").toBool();
    projectSettings->endGroup();

    return true;
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

    std::string myError;
    std::string myFileName = fileName.left(fileName.length()-4).toStdString();

    if (! gis::readEsriGrid(myFileName, &soilMap, myError))
    {
        logError("Load soil map failed: " + fileName);
        return false;
    }

    logInfo("Soil map = " + fileName);
    return true;
}


bool Crit3DProject::check3DProject()
{
    if (!DEM.isLoaded || !soilMap.isLoaded || soilList.size() == 0)
    {
        if (! DEM.isLoaded)
            errorString = ERROR_STR_MISSING_DEM;
        else if (! soilMap.isLoaded)
            errorString =  "Missing soil map.";
        else if (soilList.size() == 0)
            errorString = "Missing soil properties.";
        return false;
    }

    return true;
}


bool Crit3DProject::setSoilIndexMap()
{
    if (! check3DProject())
    {
        logError();
        return false;
    }

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
    unsigned int horizonIndex = unsigned(soil::getHorizonIndex(&(soilList[unsigned(soilIndex)]),
                                                               layerDepth[unsigned(layerIndex)]));
    if (myVar == soil::soilWaterPotentialWP)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].wiltingPoint;
    else if (myVar == soil::soilWaterPotentialFC)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].fieldCapacity;
    else if (myVar == soil::soilWaterContentFC)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].waterContentFC;
    else if (myVar == soil::soilWaterContentSat)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].vanGenuchten.thetaS;
    else if (myVar == soil::soilWaterContentWP)
    {
        double signPsiLeaf = -160;      //[m]
        return soil::thetaFromSignPsi(signPsiLeaf, &(soilList[unsigned(soilIndex)].horizon[horizonIndex]));
    }
    else
        return NODATA;
}


void Crit3DProject::clear3DProject()
{
    soilUseMap.clear();
    soilMap.clear();
    snowMaps.clear();

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
        if (! this->initializeSnowModel())
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
                    this->computeSnowPoint(row, col);
                }
            }
        }
    }
    else
    {
        for (long row = 0; row < DEM.header->nrRows; row++)
        {
            for (long col = 0; col < DEM.header->nrCols; col++)
            {
                if (! isEqual(DEM.value[row][col], DEM.header->flag))
                {
                    this->computeSnowPoint(row, col);
                }
            }
        }
        snowMaps.updateRangeMaps();
    }

    return true;
}


bool Crit3DProject::modelHourlyCycle(QDateTime myTime, const QString& hourlyOutputPath)
{
    hourlyMeteoMaps->setComputed(false);
    radiationMaps->setComputed(false);

    if (computeMeteo)
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

    if (computeRadiation)
    {
        if (! interpolateAndSaveHourlyMeteo(globalIrradiance, myTime, hourlyOutputPath, isSaveOutputRaster()))
            return false;
        qApp->processEvents();
    }

    if (computeCrop)
    {
        if (! hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps))
            return false;

        if (isSaveOutputRaster())
        {
            saveHourlyMeteoOutput(referenceEvapotranspiration, hourlyOutputPath, myTime);
        }
        qApp->processEvents();

        // computeCrop(myTime);

        // TODO compute evap/transp
    }

    if (computeSnow)
    {
        if (! computeSnowModel())
            return false;
        qApp->processEvents();
    }

    // soil water balance
    if (computeWater)
    {
        if (! computeWaterSinkSource())
        {
            logError();
            return false;
        }

        logInfo("\nWater balance: " + myTime.toString());
        computeWaterBalance3D(3600);
        qApp->processEvents();

        //updateWaterBalanceMaps();
    }

    // soil heat
    if (computeHeat)
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

            if (computeMeteo)
            {
                if (! outputPointsDbHandler->addColumn(tableName, airTemperature, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, precipitation, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, airRelHumidity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, windScalarIntensity, errorString)) return false;
            }
            if (computeRadiation)
            {
                if (! outputPointsDbHandler->addColumn(tableName, atmTransmissivity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, globalIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, directIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, diffuseIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, reflectedIrradiance, errorString)) return false;
            }
            if (computeSnow)
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

    if (computeMeteo)
    {
        varList.push_back(airTemperature);
        varList.push_back(precipitation);
        varList.push_back(airRelHumidity);
        varList.push_back(windScalarIntensity);
    }
    if (computeRadiation)
    {
        varList.push_back(atmTransmissivity);
        varList.push_back(globalIrradiance);
        varList.push_back(directIrradiance);
        varList.push_back(diffuseIrradiance);
        varList.push_back(reflectedIrradiance);
    }
    if (computeSnow)
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
            if (computeMeteo)
            {
                valuesList.push_back(hourlyMeteoMaps->mapHourlyTair->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyPrec->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyRelHum->getValueFromXY(x, y));
                valuesList.push_back(hourlyMeteoMaps->mapHourlyWindScalarInt->getValueFromXY(x, y));
            }
            if (computeRadiation)
            {
                valuesList.push_back(radiationMaps->transmissivityMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->globalRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->beamRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->diffuseRadiationMap->getValueFromXY(x, y));
                valuesList.push_back(radiationMaps->reflectedRadiationMap->getValueFromXY(x, y));
            }
            if (computeSnow)
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


void Crit3DProject::shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col)
{
    colorOut.red = colorIn.red;
    colorOut.green = colorIn.green;
    colorOut.blue = colorIn.blue;

    float aspect = radiationMaps->aspectMap->getValueFromRowCol(row, col);
    if (! isEqual(aspect, radiationMaps->aspectMap->header->flag))
    {
        float slope = radiationMaps->slopeMap->getValueFromRowCol(row, col);
        if (! isEqual(slope, radiationMaps->slopeMap->header->flag))
        {
            float slopeAmplification = 120.f / std::max(radiationMaps->slopeMap->maximum, 1.f);
            float shadow = -cos(aspect * float(DEG_TO_RAD)) * std::max(5.f, slope * slopeAmplification);
            colorOut.red = std::min(255, std::max(0, int(colorOut.red + shadow)));
            colorOut.green = std::min(255, std::max(0, int(colorOut.green + shadow)));
            colorOut.blue = std::min(255, std::max(0, int(colorOut.blue + shadow)));
            if (slope > geometry->artifactSlope())
            {
                colorOut.red = std::min(255, std::max(0, int((colorOut.red + 256) / 2)));
                colorOut.green = std::min(255, std::max(0, int((colorOut.green + 256) / 2)));
                colorOut.blue = std::min(255, std::max(0, int((colorOut.blue + 256) / 2)));
            }
        }
    }
}


void Crit3DProject::clearGeometry()
{
    if (geometry != nullptr)
    {
        geometry->clear();
        delete geometry;
        geometry = nullptr;
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
    geometry = new Crit3DGeometry();

    // set center
    gis::Crit3DPoint center = DEM.getCenter();
    gis::updateMinMaxRasterGrid(&DEM);
    float zCenter = (DEM.maximum + DEM.minimum) * 0.5f;
    geometry->setCenter(float(center.utm.x), float(center.utm.y), zCenter);

    // set dimension
    float dx = float(DEM.header->nrCols * DEM.header->cellSize);
    float dy = float(DEM.header->nrRows * DEM.header->cellSize);
    float dz = DEM.maximum + DEM.minimum;
    geometry->setDimension(dx, dy);
    float magnify = ((dx + dy) * 0.5f) / (dz * 10.f);
    geometry->setMagnify(std::min(5.f, std::max(1.f, magnify)));

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
                        geometry->addTriangle(p1, p2, p3, sc1, sc2, sc3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        DEM.getXY(row, col+1, x, y);
                        p2 = gis::Crit3DPoint(x, y, z2);
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row, col+1);
                        geometry->addTriangle(p3, p2, p1, sc3, sc2, sc1);
                    }
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::update3DColors()
{
    if (geometry == nullptr)
    {
        errorString = "Initialize 3D geometry before.";
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
                        geometry->setVertexColor(i++, sc1);
                        geometry->setVertexColor(i++, sc2);
                        geometry->setVertexColor(i++, sc3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row, col+1);
                        geometry->setVertexColor(i++, sc3);
                        geometry->setVertexColor(i++, sc2);
                        geometry->setVertexColor(i++, sc1);
                    }
                }
            }
        }
    }

    return true;
}

