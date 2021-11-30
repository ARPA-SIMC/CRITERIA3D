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


#include "commonConstants.h"
#include "utilities.h"
#include "criteria3DProject.h"
#include "soilDbTools.h"
#include "gis.h"
#include "statistics.h"

#include <QtSql>
#include <QPaintEvent>


Crit3DProject::Crit3DProject() : Project3D()
{
    saveOutputRaster = false;
    saveOutputPoints = false;
    saveDailyState = false;
    computeOnlyPoints = false;

    isMeteo = false;
    isRadiation = false;
    isCrop = false;
    isWater = false;
    isSnow = false;

    modelPause = false;
    modelStop = false;

    modelFirstTime.setTimeSpec(Qt::UTC);
    modelLastTime.setTimeSpec(Qt::UTC);
}


void Crit3DProject::setSaveDailyState(bool isSave)
{
    saveDailyState = isSave;
}

bool Crit3DProject::isSaveDailyState()
{
    return saveDailyState;
}

void Crit3DProject::setSaveOutputRaster(bool isSave)
{
    saveOutputRaster = isSave;
}

bool Crit3DProject::isSaveOutputRaster()
{
    return saveOutputRaster;
}

void Crit3DProject::setSaveOutputPoints(bool isSave)
{
    saveOutputPoints = isSave;
}

void Crit3DProject::setComputeOnlyPoints(bool isUse)
{
    computeOnlyPoints = isUse;
}

bool Crit3DProject::isComputeOnlyPoints()
{
    return computeOnlyPoints;
}

// true if at least one point is active
bool Crit3DProject::isSaveOutputPoints()
{
    if (! saveOutputPoints || outputPoints.empty())
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

    clearCriteria3DProject();

    initializeProject();
    initializeProject3D();

    snowModel.initialize();

    if (! loadProjectSettings(myFileName))
        return false;

    if (! loadCriteria3DSettings())
        return false;

    if (! loadProject())
    {
        if (errorType != ERROR_DBGRID)
            return false;
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

    soilDbFileName = projectSettings->value("soil_db").toString();
    if (soilDbFileName == "")
        soilDbFileName = projectSettings->value("db_soil").toString();

    cropDbFileName = projectSettings->value("crop_db").toString();
    if (cropDbFileName == "")
        cropDbFileName = projectSettings->value("db_crop").toString();

    soilMapFileName = projectSettings->value("soil_map").toString();

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

    if (! gis::readEsriGrid(myFileName, &soilMap, &myError))
    {
        logError("Load soil map failed: " + fileName);
        return false;
    }

    logInfo("Soil map = " + fileName);
    return true;
}


bool Crit3DProject::setSoilIndexMap()
{
    // check
    if (!DEM.isLoaded || !soilMap.isLoaded || soilList.size() == 0)
    {
        if (!DEM.isLoaded)
            logError("Missing Digital Elevation Model.");
        else if (!soilMap.isLoaded)
            logError("Missing soil map.");
        else if (soilList.size() == 0)
            logError("Missing soil properties.");
        return false;
    }

    int soilIndex;
    double x, y;
    soilIndexMap.initializeGrid(*(DEM.header));
    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                gis::getUtmXYFromRowCol(DEM, row, col, &x, &y);
                soilIndex = getCrit3DSoilIndex(x, y);
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


void Crit3DProject::clearCriteria3DProject()
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


bool Crit3DProject::initializeCriteria3DModel()
{
    // check data
    if (! this->DEM.isLoaded)
    {
        logError("Missing Digital Elevation Model.");
        return false;
    }
    else if (! this->soilMap.isLoaded)
    {
        logError("Missing soil map.");
        return false;
    }
    else if (this->soilList.size() == 0)
    {
        logError("Missing soil properties.");
        return false;
    }

    clearWaterBalance3D();

    if (!setSoilIndexMap()) return false;

    // TODO set soilUseMap()

    /* TODO initialize root density
    // andrebbe rifatto per ogni tipo di suolo (ora considera solo suolo 0)
    int nrSoilLayersWithoutRoots = 2;
    int soilLayerWithRoot = this->nrSoilLayers - nrSoilLayersWithoutRoots;
    double depthModeRootDensity = 0.35*this->soilDepth;     //[m] depth of mode of root density
    double depthMeanRootDensity = 0.5*this->soilDepth;      //[m] depth of mean of root density
    initializeRootProperties(&(this->soilList[0]), this->nrSoilLayers, this->soilDepth,
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
    logInfo("Criteria3D model initialized");

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

        aggregateAndSaveDailyMap(airTemperature, aggrMin, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airTemperature, aggrMax, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airTemperature, aggrAverage, crit3DDate, dailyPath,hourlyPath, "");
        aggregateAndSaveDailyMap(precipitation, aggrSum, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(referenceEvapotranspiration, aggrSum, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrMin, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrMax, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrAverage, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(globalIrradiance, aggrSum, crit3DDate, dailyPath, hourlyPath, "");

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


// assume that header of all meteo and snow maps = header of DEM
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

    double airT, prec, relHum, windInt, globalRad, beamRad, transmissivity, clearSkyTrans, myWaterContent;

    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                snowMaps.setPoint(snowModel, row, col);

                airT = hourlyMeteoMaps->mapHourlyTair->value[row][col];
                prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                relHum = hourlyMeteoMaps->mapHourlyRelHum->value[row][col];
                windInt = hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col];
                globalRad = radiationMaps->globalRadiationMap->value[row][col];
                beamRad = radiationMaps->beamRadiationMap->value[row][col];
                transmissivity = radiationMaps->transmissivityMap->value[row][col];
                clearSkyTrans = radSettings.getClearSky();
                myWaterContent = 0;

                snowModel.setInputData(airT, prec, relHum, windInt, globalRad, beamRad, transmissivity, clearSkyTrans, myWaterContent);

                snowModel.computeSnowBrooksModel();

                snowMaps.updateMap(snowModel, row, col);
            }
        }
    }

    snowMaps.updateRangeMaps();

    return true;
}


bool Crit3DProject::modelHourlyCycle(QDateTime myTime, const QString& hourlyOutputPath)
{
    hourlyMeteoMaps->setComputed(false);
    radiationMaps->setComputed(false);

    if (isMeteo)
    {
        if (! interpolateAndSaveHourlyMeteo(airTemperature, myTime, hourlyOutputPath, this->saveOutputRaster))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(precipitation, myTime, hourlyOutputPath, this->saveOutputRaster))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(airRelHumidity, myTime, hourlyOutputPath, this->saveOutputRaster))
            return false;
        qApp->processEvents();

        if (! interpolateAndSaveHourlyMeteo(windScalarIntensity, myTime, hourlyOutputPath, this->saveOutputRaster))
            return false;
        qApp->processEvents();

        hourlyMeteoMaps->setComputed(true);
    }

    // radiation model
    if (isRadiation)
    {
        if (! interpolateAndSaveHourlyMeteo(globalIrradiance, myTime, hourlyOutputPath, this->saveOutputRaster))
            return false;
        qApp->processEvents();
    }

    if (isCrop)
    {
        if (! hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps))
            return false;
        if (this->saveOutputRaster)
        {
            saveHourlyMeteoOutput(referenceEvapotranspiration, hourlyOutputPath, myTime, "");
        }
        qApp->processEvents();

        computeCrop(myTime);

        // TODO compute evap/transp

        qApp->processEvents();
    }

    if (isSnow)
    {
        if (! computeSnowModel())
            return false;
        qApp->processEvents();
    }

    if (isWater)
    {
        // soil water balance
        if (! computeWaterSinkSource()) return false;
        qApp->processEvents();

        computeWaterBalance3D(3600);
        qApp->processEvents();

        //updateWaterBalanceMaps();
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

    QString statePath = getProjectPath() + "/STATES";
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

    QString snowPath = statePath + "/" + dateFolder + "/snow";
    QDir().mkdir(snowPath);

    logInfo("Saving snow state: " + dateFolder);
    std::string error;
    if (!gis::writeEsriGrid((snowPath+"/SWE").toStdString(), snowMaps.getSnowWaterEquivalentMap(), &error))
    {
        logError("Error saving water equivalent map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/AgeOfSnow").toStdString(), snowMaps.getAgeOfSnowMap(), &error))
    {
        logError("Error saving age of snow map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SnowSurfaceTemp").toStdString(), snowMaps.getSnowSurfaceTempMap(), &error))
    {
        logError("Error saving snow surface temp map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/IceContent").toStdString(), snowMaps.getIceContentMap(), &error))
    {
        logError("Error saving ice content map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/LWContent").toStdString(), snowMaps.getLWContentMap(), &error))
    {
        logError("Error saving LW content map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/InternalEnergy").toStdString(), snowMaps.getInternalEnergyMap(), &error))
    {
        logError("Error saving internal energy map: " + QString::fromStdString(error));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SurfaceInternalEnergy").toStdString(), snowMaps.getSurfaceInternalEnergyMap(), &error))
    {
        logError("Error saving surface internal energy map: " + QString::fromStdString(error));
        return false;
    }

    return true;
}


QList<QString> Crit3DProject::getAllSavedState()
{
    QList<QString> states;
    QString statePath = getProjectPath() + "/STATES";
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


bool Crit3DProject::loadModelState(QString stateStr)
{
    // state folder
    QString statePath = getProjectPath() + "/STATES/" + stateStr;
    QDir stateDir(statePath);
    if (!stateDir.exists())
    {
        errorString = "STATES directory is missing.";
        return false;
    }

    // set current date/hour
    int year = stateStr.midRef(0,4).toInt();
    int month = stateStr.midRef(4,2).toInt();
    int day = stateStr.midRef(6,2).toInt();
    int hour = stateStr.midRef(10,2).toInt();
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
        if (! gis::readEsriGrid(fileName, snowMaps.getSnowWaterEquivalentMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/AgeOfSnow";
        if (! gis::readEsriGrid(fileName, snowMaps.getAgeOfSnowMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/IceContent";
        if (! gis::readEsriGrid(fileName, snowMaps.getIceContentMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/InternalEnergy";
        if (! gis::readEsriGrid(fileName, snowMaps.getInternalEnergyMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/LWContent";
        if (! gis::readEsriGrid(fileName, snowMaps.getLWContentMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/SnowSurfaceTemp";
        if (! gis::readEsriGrid(fileName, snowMaps.getSnowSurfaceTempMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }

        fileName = snowPath.toStdString() + "/SurfaceInternalEnergy";
        if (! gis::readEsriGrid(fileName, snowMaps.getSurfaceInternalEnergyMap(), &error))
        {
            errorString = "Wrong snow map:\n" + QString::fromStdString(error);
            snowMaps.isInitialized = false;
            return false;
        }
    }

    return true;
}



