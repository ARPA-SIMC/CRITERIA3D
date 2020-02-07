#include "project.h"
#include "formInfo.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "spatialControl.h"
#include "radiationSettings.h"
#include "solarRadiation.h"
#include "interpolationCmd.h"
#include "interpolation.h"
#include "transmissivity.h"
#include "utilities.h"
#include "aggregation.h"

#include <iostream>
#include <QDir>
#include <QFile>
#include <QSqlQuery>
#include <QMessageBox>


Project::Project()
{
    // TODO remove pointers
    meteoSettings = new Crit3DMeteoSettings();
    quality = new Crit3DQuality();
    meteoPointsColorScale = new Crit3DColorScale();

    // They not change after loading default settings
    appPath = "";
    defaultPath = "";

    initializeProject();

    modality = MODE_GUI;
}

void Project::initializeProject()
{
    projectPath = "";
    projectName = "";
    isProjectLoaded = false;
    requestedExit = false;
    logFileName = "";
    errorString = "";
    currentTileMap = "";

    nrMeteoPoints = 0;
    meteoPoints = nullptr;
    meteoPointsDbHandler = nullptr;
    meteoGridDbHandler = nullptr;
    aggregationDbHandler = nullptr;

    meteoSettings->initialize();
    quality->initialize();

    checkSpatialQuality = true;
    currentVariable = noMeteoVar;
    currentFrequency = noFrequency;
    currentDate.setDate(1800,1,1);
    previousDate = currentDate;
    currentHour = 12;

    parameters = nullptr;
    projectSettings = nullptr;
    radiationMaps = nullptr;
    hourlyMeteoMaps = nullptr;

    gisSettings.initialize();
    radSettings.initialize();
    interpolationSettings.initialize();
    qualityInterpolationSettings.initialize();

    parametersFileName = "";
    demFileName = "";
    dbPointsFileName = "";
    dbGridXMLFileName = "";

    meteoPointsLoaded = false;
    meteoGridLoaded = false;
    loadGridDataAtStart = false;

    proxyGridSeries.clear();
}

void Project::clearProject()
{
    if (logFile.is_open()) logFile.close();

    meteoPointsColorScale->setRange(NODATA, NODATA);
    meteoPointsSelected.clear();

    delete parameters;
    delete projectSettings;
    delete aggregationDbHandler;

    clearProxyDEM();
    DEM.clear();

    delete radiationMaps;
    delete hourlyMeteoMaps;
    radiationMaps = nullptr;
    hourlyMeteoMaps = nullptr;

    closeMeteoPointsDB();
    closeMeteoGridDB();

    isProjectLoaded = false;
}

void Project::clearProxyDEM()
{
    int index = interpolationSettings.getIndexHeight();
    int indexQuality = qualityInterpolationSettings.getIndexHeight();

    // if no elevation proxy defined nothing to do
    if (index == NODATA && indexQuality == NODATA) return;

    Crit3DProxy* proxyHeight;

    if (index != NODATA)
    {
        proxyHeight = interpolationSettings.getProxy(unsigned(index));
        proxyHeight->setGrid(nullptr);
    }

    if (indexQuality != NODATA)
    {
        proxyHeight = qualityInterpolationSettings.getProxy(unsigned(indexQuality));
        proxyHeight->setGrid(nullptr);
    }
}


void Project::setProxyDEM()
{
    int index = interpolationSettings.getIndexHeight();
    int indexQuality = qualityInterpolationSettings.getIndexHeight();

    // if no elevation proxy defined nothing to do
    if (index == NODATA && indexQuality == NODATA) return;

    Crit3DProxy* proxyHeight;

    if (index != NODATA)
    {
        proxyHeight = interpolationSettings.getProxy(unsigned(index));

        // if no alternative DEM defined and project DEM loaded, use it for elevation proxy
        if (proxyHeight->getGridName() == "" && DEM.isLoaded)
            proxyHeight->setGrid(&DEM);
    }

    if (indexQuality != NODATA)
    {
        proxyHeight = qualityInterpolationSettings.getProxy(unsigned(indexQuality));
        if (proxyHeight->getGridName() == "" && DEM.isLoaded)
            proxyHeight->setGrid(&DEM);
    }
}

bool Project::checkProxy(QString name_, QString gridName_, QString table_, QString field_, QString *error)
{
    if (name_ == "")
    {
        *error = "no name";
        return false;
    }

    bool isHeight = (getProxyPragaName(name_.toStdString()) == height);

    if (!isHeight & (gridName_ == "") & (table_ == "" && field_ == ""))
    {
        *error = "error reading grid, table or field for proxy " + name_;
        return false;
    }

    return true;
}

void Project::addProxyToProject(QString name_, QString gridName_, QString table_, QString field_, bool isForQuality_, bool isActive_)
{
    Crit3DProxy myProxy;

    myProxy.setName(name_.toStdString());
    myProxy.setGridName(gridName_.toStdString());
    myProxy.setProxyTable(table_.toStdString());
    myProxy.setProxyField(field_.toStdString());
    myProxy.setForQualityControl(isForQuality_);

    interpolationSettings.addProxy(myProxy, isActive_);
    if (isForQuality_)
        qualityInterpolationSettings.addProxy(myProxy, isActive_);

    if (getProxyPragaName(name_.toStdString()) == height) setProxyDEM();
}


void Project::addProxyGridSeries(QString name_, std::vector <QString> gridNames, std::vector <unsigned> gridYears)
{
    // no check on grids
    std::string myError;

    Crit3DProxyGridSeries mySeries(name_);

    for (unsigned i=0; i < gridNames.size(); i++)
        mySeries.addGridToSeries(gridNames[i], signed(gridYears[i]));

    proxyGridSeries.push_back(mySeries);
}

bool Project::loadParameters(QString parametersFileName)
{
    parametersFileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);

    if (! QFile(parametersFileName).exists() || ! QFileInfo(parametersFileName).isFile())
    {
        logError("Missing parameters file: " + parametersFileName);
        return false;
    }

    delete parameters;
    parameters = new QSettings(parametersFileName, QSettings::IniFormat);

    //interpolation settings
    interpolationSettings.initialize();
    qualityInterpolationSettings.initialize();

    QString gridName = "";
    QString proxyName = "", proxyGridName = "", proxyTable = "", proxyField = "";
    bool isActive = false, forQuality = false;
    QStringList myList;
    std::vector <QString> proxyGridSeriesNames;
    std::vector <unsigned> proxyGridSeriesYears;

    Q_FOREACH (QString group, parameters->childGroups())
    {
        //meteo settings
        if (group == "meteo")
        {
            parameters->beginGroup(group);

            if (parameters->contains("min_percentage") && !parameters->value("min_percentage").toString().isEmpty())
            {
                meteoSettings->setMinimumPercentage(parameters->value("min_percentage").toFloat());
            }
            if (parameters->contains("prec_threshold") && !parameters->value("prec_threshold").toString().isEmpty())
            {
                meteoSettings->setRainfallThreshold(parameters->value("prec_threshold").toFloat());
            }
            if (parameters->contains("thom_threshold") && !parameters->value("thom_threshold").toString().isEmpty())
            {
                meteoSettings->setThomThreshold(parameters->value("thom_threshold").toFloat());
            }
            if (parameters->contains("samani_coefficient") && !parameters->value("samani_coefficient").toString().isEmpty())
            {
                meteoSettings->setTransSamaniCoefficient(parameters->value("samani_coefficient").toFloat());
            }
            if (parameters->contains("hourly_intervals") && !parameters->value("hourly_intervals").toString().isEmpty())
            {
                meteoSettings->setHourlyIntervals(parameters->value("hourly_intervals").toInt());
            }
            if (parameters->contains("wind_intensity_default") && !parameters->value("wind_intensity_default").toString().isEmpty())
            {
                meteoSettings->setWindIntensityDefault(parameters->value("wind_intensity_default").toInt());
            }

            parameters->endGroup();
        }

        if (group == "climate")
        {
            parameters->beginGroup(group);

            if (parameters->contains("tmin"))
            {
                myList = parameters->value("tmin").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmin values";
                    return  false;
                }

                climateParameters.tmin = StringListToFloat(myList);
            }

            if (parameters->contains("tmax"))
            {
                myList = parameters->value("tmax").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmax values";
                    return  false;
                }

                climateParameters.tmax = StringListToFloat(myList);
            }

            if (parameters->contains("tmin_lapserate"))
            {
                myList = parameters->value("tmin_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmin lapse rate values";
                    return  false;
                }

                climateParameters.tminLapseRate = StringListToFloat(myList);
            }

            if (parameters->contains("tmax_lapserate"))
            {
                myList = parameters->value("tmax_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmax lapse rate values";
                    return  false;
                }

                climateParameters.tmaxLapseRate = StringListToFloat(myList);
            }

            if (parameters->contains("tdmin"))
            {
                myList = parameters->value("tdmin").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmin values";
                    return  false;
                }

                climateParameters.tdmin = StringListToFloat(myList);
            }

            if (parameters->contains("tdmax"))
            {
                myList = parameters->value("tdmax").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmax values";
                    return  false;
                }

                climateParameters.tdmax = StringListToFloat(myList);
            }

            if (parameters->contains("tdmin_lapserate"))
            {
                myList = parameters->value("tdmin_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmin lapse rate values";
                    return  false;
                }

                climateParameters.tdMinLapseRate = StringListToFloat(myList);
            }

            if (parameters->contains("tdmax_lapserate"))
            {
                myList = parameters->value("tdmax_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmax lapse rate values";
                    return  false;
                }

                climateParameters.tdMaxLapseRate = StringListToFloat(myList);
            }

            parameters->endGroup();
        }

        if (group == "radiation")
        {
            parameters->beginGroup(group);

            if (parameters->contains("algorithm"))
            {
                std::string algorithm = parameters->value("algorithm").toString().toStdString();
                if (radAlgorithmToString.find(algorithm) == radAlgorithmToString.end())
                {
                    errorString = "Unknown radiation algorithm: " + QString::fromStdString(algorithm);
                    return false;
                }
                else
                    radSettings.setAlgorithm(radAlgorithmToString.at(algorithm));
            }

            if (parameters->contains("real_sky_algorithm"))
            {
                std::string realSkyAlgorithm = parameters->value("real_sky_algorithm").toString().toStdString();
                if (realSkyAlgorithmToString.find(realSkyAlgorithm) == realSkyAlgorithmToString.end())
                {
                    errorString = "Unknown radiation real sky algorithm: " + QString::fromStdString(realSkyAlgorithm);
                    return false;
                }
                else
                    radSettings.setRealSkyAlgorithm(realSkyAlgorithmToString.at(realSkyAlgorithm));
            }

            if (parameters->contains("linke_mode"))
            {
                std::string linkeMode = parameters->value("linke_mode").toString().toStdString();
                if (paramModeToString.find(linkeMode) == paramModeToString.end())
                {
                    errorString = "Unknown Linke mode: " + QString::fromStdString(linkeMode);
                    return false;
                }
                else
                    radSettings.setLinkeMode(paramModeToString.at(linkeMode));
            }

            if (parameters->contains("albedo_mode"))
            {
                std::string albedoMode = parameters->value("albedo_mode").toString().toStdString();
                if (paramModeToString.find(albedoMode) == paramModeToString.end())
                {
                    errorString = "Unknown albedo mode: " + QString::fromStdString(albedoMode);
                    return false;
                }
                else
                    radSettings.setAlbedoMode(paramModeToString.at(albedoMode));
            }

            if (parameters->contains("tilt_mode"))
            {
                std::string tiltMode = parameters->value("tilt_mode").toString().toStdString();
                if (tiltModeToString.find(tiltMode) == tiltModeToString.end())
                {
                    errorString = "Unknown albedo mode: " + QString::fromStdString(tiltMode);
                    return false;
                }
                else
                    radSettings.setTiltMode(tiltModeToString.at(tiltMode));
            }

            if (parameters->contains("real_sky"))
                radSettings.setRealSky(parameters->value("real_sky").toBool());

            if (parameters->contains("shadowing"))
                radSettings.setShadowing(parameters->value("shadowing").toBool());

            if (parameters->contains("linke"))
                radSettings.setLinke(parameters->value("linke").toFloat());

            if (parameters->contains("albedo"))
                radSettings.setAlbedo(parameters->value("albedo").toFloat());

            if (parameters->contains("tilt"))
                radSettings.setTilt(parameters->value("tilt").toFloat());

            if (parameters->contains("aspect"))
                radSettings.setAspect(parameters->value("aspect").toFloat());

            if (parameters->contains("clear_sky"))
                radSettings.setClearSky(parameters->value("clear_sky").toFloat());

            if (parameters->contains("linke_map"))
                radSettings.setLinkeMapName(parameters->value("linke_map").toString().toStdString());

            if (parameters->contains("albedo_map"))
                radSettings.setAlbedoMapName(parameters->value("albedo_map").toString().toStdString());

            if (parameters->contains("linke_monthly"))
            {
                QStringList myLinkeStr = parameters->value("linke_monthly").toStringList();
                if (myLinkeStr.size() < 12)
                {
                    errorString = "Incomplete monthly Linke values";
                    return  false;
                }

                radSettings.setLinkeMonthly(StringListToFloat(myLinkeStr));
            }

            if (parameters->contains("albedo_monthly"))
            {
                QStringList myAlbedoStr = parameters->value("albedo_monthly").toStringList();
                if (myAlbedoStr.size() < 12)
                {
                    errorString = "Incomplete monthly albedo values";
                    return  false;
                }

                radSettings.setAlbedoMonthly(StringListToFloat(myAlbedoStr));
            }

            parameters->endGroup();
        }

        //interpolation
        if (group == "interpolation")
        {
            parameters->beginGroup(group);

            if (parameters->contains("algorithm"))
            {
                std::string algorithm = parameters->value("algorithm").toString().toStdString();
                if (interpolationMethodNames.find(algorithm) == interpolationMethodNames.end())
                {
                    errorString = "Unknown interpolation method";
                    return false;
                }
                else
                    interpolationSettings.setInterpolationMethod(interpolationMethodNames.at(algorithm));
            }

            if (parameters->contains("aggregationMethod"))
            {
                std::string aggrMethod = parameters->value("aggregationMethod").toString().toStdString();
                if (aggregationMethodToString.find(aggrMethod) == aggregationMethodToString.end())
                {
                    errorString = "Unknown aggregation method";
                    return false;
                }
                else
                    interpolationSettings.setMeteoGridAggrMethod(aggregationMethodToString.at(aggrMethod));
            }

            if (parameters->contains("thermalInversion"))
            {
                interpolationSettings.setUseThermalInversion(parameters->value("thermalInversion").toBool());
                qualityInterpolationSettings.setUseThermalInversion(parameters->value("thermalInversion").toBool());
            }

            if (parameters->contains("topographicDistance"))
                interpolationSettings.setUseTAD(parameters->value("topographicDistance").toBool());

            if (parameters->contains("lapseRateCode"))
            {
                interpolationSettings.setUseLapseRateCode(parameters->value("lapseRateCode").toBool());
                qualityInterpolationSettings.setUseLapseRateCode(parameters->value("lapseRateCode").toBool());
            }

            if (parameters->contains("optimalDetrending"))
                interpolationSettings.setUseBestDetrending(parameters->value("optimalDetrending").toBool());

            if (parameters->contains("minRegressionR2"))
            {
                interpolationSettings.setMinRegressionR2(parameters->value("minRegressionR2").toFloat());
                qualityInterpolationSettings.setMinRegressionR2(parameters->value("minRegressionR2").toFloat());
            }

            if (parameters->contains("useDewPoint"))
                interpolationSettings.setUseDewPoint(parameters->value("useDewPoint").toBool());

            if (parameters->contains("useInterpolationTemperatureForRH"))
                interpolationSettings.setUseInterpolatedTForRH(parameters->value("useInterpolationTemperatureForRH").toBool());

            parameters->endGroup();

        }

        if (group == "quality")
        {
            parameters->beginGroup(group);
            if (parameters->contains("reference_height") && !parameters->value("reference_height").toString().isEmpty())
            {
                quality->setReferenceHeight(parameters->value("reference_height").toFloat());
            }
            if (parameters->contains("delta_temperature_suspect") && !parameters->value("delta_temperature_suspect").toString().isEmpty())
            {
                quality->setDeltaTSuspect(parameters->value("delta_temperature_suspect").toFloat());
            }
            if (parameters->contains("delta_temperature_wrong") && !parameters->value("delta_temperature_wrong").toString().isEmpty())
            {
                quality->setDeltaTWrong(parameters->value("delta_temperature_wrong").toFloat());
            }
            if (parameters->contains("relhum_tolerance") && !parameters->value("relhum_tolerance").toString().isEmpty())
            {
                quality->setRelHumTolerance(parameters->value("relhum_tolerance").toFloat());
            }

            parameters->endGroup();
        }

        //proxy variables (for interpolation)
        if (group.startsWith("proxy_"))
        {
            proxyName = group.right(group.size()-6);

            parameters->beginGroup(group);

            proxyTable = parameters->value("table").toString();
            proxyField = parameters->value("field").toString();
            isActive = parameters->value("active").toBool();
            forQuality = parameters->value("use_for_spatial_quality_control").toBool();
            proxyGridName = parameters->value("raster").toString();

            parameters->endGroup();

            if (checkProxy(proxyName, proxyGridName, proxyTable, proxyField, &errorString))
                addProxyToProject(proxyName, proxyGridName, proxyTable, proxyField, forQuality, isActive);
            else
                logError();
        }

        //proxy grid annual series
        if (group.startsWith("proxygrid"))
        {
            proxyName = group.right(group.length()-10);

            parameters->beginGroup(group);
            int nrGrids = parameters->beginReadArray("grids");
            for (int i = 0; i < nrGrids; ++i) {
                parameters->setArrayIndex(i);
                proxyGridSeriesNames.push_back(parameters->value("name").toString());
                proxyGridSeriesYears.push_back(parameters->value("year").toUInt());
            }
            parameters->endArray();
            parameters->endGroup();

            addProxyGridSeries(proxyName, proxyGridSeriesNames, proxyGridSeriesYears);
        }
    }

    // check proxy grids for detrending
    if (!loadProxyGrids())
        return false;

    if (!loadRadiationGrids())
        return false;

    return true;
}


bool Project::getMeteoPointSelected(int i)
{
    if (meteoPointsSelected.isEmpty()) return true;

    for (int j = 0; j < meteoPointsSelected.size(); j++)
    {
        if (isEqual(meteoPoints[i].latitude, meteoPointsSelected[j].latitude)
            && isEqual(meteoPoints[i].longitude, meteoPointsSelected[j].longitude))
            return true;
    }

    return false;
}


void Project::setApplicationPath(QString myPath)
{
    this->appPath = myPath;
}

QString Project::getApplicationPath()
{
    char* appImagePath;
    appImagePath = getenv ("APPIMAGE");
    if (appImagePath!=nullptr)
    {
        QDir d = QFileInfo(appImagePath).absoluteDir();
        QString absolute = d.absolutePath()+"/";
        return absolute;
    }
    else
    {
        return this->appPath;
    }
}

void Project::setDefaultPath(QString myPath)
{
    this->defaultPath = myPath;
}

QString Project::getDefaultPath()
{
    return this->defaultPath;
}

void Project::setProjectPath(QString myPath)
{
    this->projectPath = myPath;
}

QString Project::getProjectPath()
{
    return this->projectPath;
}

void Project::setCurrentVariable(meteoVariable variable)
{
    this->currentVariable = variable;
}

meteoVariable Project::getCurrentVariable()
{
    return this->currentVariable;
}

void Project::setCurrentDate(QDate myDate)
{
    if (myDate != this->currentDate)
    {
        this->previousDate = this->currentDate;
        this->currentDate = myDate;
    }
}

QDate Project::getCurrentDate()
{
    return this->currentDate;
}

void Project::setCurrentHour(int myHour)
{
    this->currentHour = myHour;
}

int Project::getCurrentHour()
{
    return this->currentHour;
}

Crit3DTime Project::getCrit3DCurrentTime()
{
    return getCrit3DTime(this->currentDate, this->currentHour);
}

QDateTime Project::getCurrentTime()
{
    return QDateTime(this->currentDate, QTime(this->currentHour, 0, 0));
}

void Project::getMeteoPointsRange(float *minimum, float *maximum)
{
    *minimum = NODATA;
    *maximum = NODATA;

    if (currentFrequency == noFrequency || currentVariable == noMeteoVar)
        return;

    float v;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        v = meteoPoints[i].currentValue;

        if (int(v) != int(NODATA) && meteoPoints[i].quality == quality::accepted)
        {
            if (int(*minimum) == int(NODATA))
            {
                *minimum = v;
                *maximum = v;
            }
            else if (v < *minimum) *minimum = v;
            else if (v > *maximum) *maximum = v;
        }
    }
}


void Project::clearMeteoPoints()
{
    if (nrMeteoPoints > 0 && meteoPoints != nullptr)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            meteoPoints[i].cleanObsDataH();
            meteoPoints[i].cleanObsDataD();
            meteoPoints[i].cleanObsDataM();
            meteoPoints[i].proxyValues.clear();
            if (meteoPoints[i].topographicDistance != nullptr)
            {
                meteoPoints[i].topographicDistance->clear();
                delete meteoPoints[i].topographicDistance;
            }
        }
        delete [] meteoPoints;
    }

    nrMeteoPoints = 0;
    meteoPoints = nullptr;
}


void Project::closeMeteoPointsDB()
{
    if (meteoPointsDbHandler != nullptr)
    {
        delete meteoPointsDbHandler;
        meteoPointsDbHandler = nullptr;
    }

    clearMeteoPoints();
    meteoPointsSelected.clear();

    dbPointsFileName = "";
    meteoPointsLoaded = false;
}


void Project::closeMeteoGridDB()
{
    //TODO check clean data

    if (meteoGridDbHandler != nullptr)
    {
        delete meteoGridDbHandler;
    }

    dbGridXMLFileName = "";
    meteoGridDbHandler = nullptr;
    meteoGridLoaded = false;
}


/*!
 * \brief loadDEM
 * \param myFileName the name of the Digital Elevation Model file
 * \return true if file is ok, false otherwise
 */
bool Project::loadDEM(QString myFileName)
{
    if (myFileName == "")
    {
        logError("Missing DEM filename");
        return false;
    }

    this->demFileName = myFileName;
    myFileName = getCompleteFileName(myFileName, PATH_DEM);

    std::string error, fileName;
    if (myFileName.right(4).left(1) == ".")
    {
        myFileName = myFileName.left(myFileName.length()-4);
    }
    fileName = myFileName.toStdString();

    if (! gis::readEsriGrid(fileName, &DEM, &error))
    {
        this->logError("Wrong Digital Elevation Model file.\n" + QString::fromStdString(error));
        return false;
    }

    setColorScale(noMeteoTerrain, DEM.colorScale);

    // initialize radiation maps (slope, aspect, lat/lon, transmissivity, etc.)
    if (radiationMaps != nullptr) radiationMaps->clear();
    radiationMaps = new Crit3DRadiationMaps(DEM, gisSettings);

    // initialize hourly meteo maps
    if (hourlyMeteoMaps != nullptr) hourlyMeteoMaps->clear();
    hourlyMeteoMaps = new Crit3DHourlyMeteoMaps(DEM);

    //reset aggregationPoints meteoGrid
    if (meteoGridDbHandler != nullptr)
    {
        meteoGridDbHandler->meteoGrid()->setIsAggregationDefined(false);
        // TODO
        // ciclo sulle celle della meteo grid -> clean vettore aggregation points
    }

    setProxyDEM();

    //set interpolation settings DEM
    interpolationSettings.setCurrentDEM(&DEM);
    qualityInterpolationSettings.setCurrentDEM(&DEM);

    //check points position with respect to DEM
    checkMeteoPointsDEM();

    logInfo("DEM = " + myFileName);
    return true;
}


bool Project::loadMeteoPointsDB(QString dbName)
{
    if (dbName == "") return false;

    closeMeteoPointsDB();

    dbPointsFileName = dbName;
    dbName = getCompleteFileName(dbName, PATH_METEOPOINT);

    meteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(dbName);
    if (meteoPointsDbHandler->error != "")
    {
        logError("Function loadMeteoPointsDB:\n" + dbName + "\n" + meteoPointsDbHandler->error);
        closeMeteoPointsDB();
        return false;
    }

    if (! meteoPointsDbHandler->loadVariableProperties())
    {
        logError(meteoPointsDbHandler->error);
        closeMeteoPointsDB();
        return false;
    }
    QList<Crit3DMeteoPoint> listMeteoPoints = meteoPointsDbHandler->getPropertiesFromDb(gisSettings, &errorString);

    nrMeteoPoints = listMeteoPoints.size();
    if (nrMeteoPoints == 0)
    {
        errorString = "Error in reading the point properties:\n" + errorString;
        logError();
        closeMeteoPointsDB();
        return false;
    }

    meteoPoints = new Crit3DMeteoPoint[unsigned(nrMeteoPoints)];

    for (int i=0; i < nrMeteoPoints; i++)
        meteoPoints[i] = listMeteoPoints[i];

    listMeteoPoints.clear();

    // find last date
    QDateTime dbLastTime = findDbPointLastTime();
    if (! dbLastTime.isNull())
    {
        if (dbLastTime.time().hour() == 00)
        {
            setCurrentDate(dbLastTime.date().addDays(-1));
            setCurrentHour(24);
        }
        else
        {
            setCurrentDate(dbLastTime.date());
            setCurrentHour(dbLastTime.time().hour());
        }
    }

    // load proxy values for detrending
    if (! readProxyValues())
    {
        logInfo("Error reading proxy values");
    }

    //position with respect to DEM
    if (DEM.isLoaded)
        checkMeteoPointsDEM();

    meteoPointsLoaded = true;
    logInfo("Meteo points DB = " + dbName);

    return true;
}


bool Project::loadMeteoGridDB(QString xmlName)
{
    if (xmlName == "") return false;

    dbGridXMLFileName = xmlName;
    xmlName = getCompleteFileName(xmlName, PATH_METEOGRID);

    meteoGridDbHandler = new Crit3DMeteoGridDbHandler();
    meteoGridDbHandler->meteoGrid()->setGisSettings(this->gisSettings);

    if (! meteoGridDbHandler->parseXMLGrid(xmlName, &errorString)) return false;

    if (! this->meteoGridDbHandler->openDatabase(&errorString)) return false;

    if (! this->meteoGridDbHandler->loadCellProperties(&errorString)) return false;

    this->meteoGridDbHandler->updateGridDate(&errorString);

    if (loadGridDataAtStart)
        setCurrentDate(meteoGridDbHandler->lastDate());

    meteoGridLoaded = true;
    logInfo("Meteo Grid = " + xmlName);

    return true;
}


bool Project::loadAggregationdDB(QString dbName)
{
    if (dbName == "") return false;

    aggregationDbHandler = new Crit3DAggregationsDbHandler(dbName);
    if (aggregationDbHandler->error() != "")
    {
        logError(aggregationDbHandler->error());
        return false;
    }
    if (aggregationDbHandler->loadVariableProperties())
    {
        return false;
    }
    return true;
}

bool Project::loadMeteoPointsData(QDate firstDate, QDate lastDate, bool loadHourly, bool loadDaily, bool showInfo)
{
    //check
    if (firstDate == QDate(1800,1,1) || lastDate == QDate(1800,1,1)) return false;

    bool isData = false;
    FormInfo myInfo;
    int step = 0;

    QString infoStr = "Load data: " + firstDate.toString();

    if (firstDate != lastDate)
        infoStr += " - " + lastDate.toString();

    if (showInfo)
        step = myInfo.start(infoStr, nrMeteoPoints);

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % step) == 0) myInfo.setValue(i);
        }

        if (loadHourly)
            if (meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;

        if (loadDaily)
            if (meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;
    }

    if (showInfo) myInfo.close();

    return isData;
}


void Project::loadMeteoGridData(QDate firstDate, QDate lastDate, bool showInfo)
{
    if (this->meteoGridDbHandler != nullptr)
    {
        this->loadMeteoGridDailyData(firstDate, lastDate, showInfo);
        this->loadMeteoGridHourlyData(QDateTime(firstDate, QTime(1,0)), QDateTime(lastDate.addDays(1), QTime(0,0)), showInfo);
    }
}


bool Project::loadMeteoGridDailyData(QDate firstDate, QDate lastDate, bool showInfo)
{
    if (! meteoGridDbHandler->tableDaily().exists) return false;

    std::string id;
    int count = 0;

    FormInfo myInfo;
    int infoStep = 1;

    if (showInfo)
    {
        QString infoStr = "Load meteo grid daily data: " + firstDate.toString();
        if (firstDate != lastDate) infoStr += " - " + lastDate.toString();
        infoStep = myInfo.start(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
            myInfo.setValue(row);

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                {
                    if (this->meteoGridDbHandler->loadGridDailyData(&errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
                else
                {
                    if (this->meteoGridDbHandler->loadGridDailyDataFixedFields(&errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
            }
        }
    }

    if (showInfo) myInfo.close();

    if (count == 0)
    {
        errorString = "No Data Available";
        return false;
    }
    else
        return true;
}


bool Project::loadMeteoGridHourlyData(QDateTime firstDate, QDateTime lastDate, bool showInfo)
{
    std::string id;
    int count = 0;
    FormInfo myInfo;
    int infoStep = 1;

    if (! meteoGridDbHandler->tableHourly().exists) return false;

    if (showInfo)
    {
        QString infoStr = "Load meteo grid hourly data: " + firstDate.toString("yyyy-MM-dd:hh") + " - " + lastDate.toString("yyyy-MM-dd:hh");
        infoStep = myInfo.start(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
            myInfo.setValue(row);

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                {
                    if (this->meteoGridDbHandler->loadGridHourlyData(&errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
                else
                {
                    if (this->meteoGridDbHandler->loadGridHourlyDataFixedFields(&errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
            }
        }
    }

    if (showInfo) myInfo.close();

    if (count == 0)
    {
        errorString = "No Data Available";
        return false;
    }
    else
        return true;
}


QDateTime Project::findDbPointLastTime()
{
    QDateTime lastTime;

    QDateTime lastDateD = meteoPointsDbHandler->getLastDate(daily);
    if (! lastDateD.isNull()) lastTime = lastDateD;

    QDateTime lastDateH = meteoPointsDbHandler->getLastDate(hourly);
    if (! lastDateH.isNull())
    {
        if (! lastTime.isNull())
            lastTime = (lastDateD > lastDateH) ? lastDateD : lastDateH;
        else
            lastTime = lastDateH;
    }

    return lastTime;
}

QDateTime Project::findDbPointFirstTime()
{
    QDateTime firstTime;

    QDateTime firstDateD = meteoPointsDbHandler->getFirstDate(daily);
    if (! firstDateD.isNull()) firstTime = firstDateD;

    QDateTime firstDateH = meteoPointsDbHandler->getFirstDate(hourly);
    if (! firstDateH.isNull())
    {
        if (! firstTime.isNull())
            firstTime = (firstDateD > firstDateH) ? firstDateD : firstDateH;
        else
            firstTime = firstDateH;
    }

    return firstTime;
}

void Project::checkMeteoPointsDEM()
{
    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (! gis::isOutOfGridXY(meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, DEM.header)
                && (! isEqual(gis::getValueFromXY(DEM, meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y), DEM.header->flag)))
             meteoPoints[i].isInsideDem = true;
        else
            meteoPoints[i].isInsideDem = false;
    }
}


bool Project::readPointProxyValues(Crit3DMeteoPoint* myPoint, QSqlDatabase* myDb)
{
    if (myPoint == nullptr) return false;

    QSqlQuery qry(*myDb);

    QString proxyField;
    QString proxyTable;
    QString statement;
    int nrProxy;
    Crit3DProxy* myProxy;

    nrProxy = int(interpolationSettings.getProxyNr());
    myPoint->proxyValues.resize(unsigned(nrProxy));

    for (unsigned int i=0; i < unsigned(nrProxy); i++)
    {
        myPoint->proxyValues[i] = NODATA;

        // read only for active proxies
        if (interpolationSettings.getSelectedCombination().getValue(i))
        {
            myProxy = interpolationSettings.getProxy(i);
            proxyField = QString::fromStdString(myProxy->getProxyField());
            proxyTable = QString::fromStdString(myProxy->getProxyTable());
            if (proxyField != "" && proxyTable != "")
            {
                statement = QString("SELECT %1 FROM %2 WHERE id_point = '%3'").arg(proxyField).arg(proxyTable).arg(QString::fromStdString((*myPoint).id));
                if(qry.exec(statement))
                {
                    qry.last();
                    if (qry.value(proxyField) != "")
                        myPoint->proxyValues[i] = qry.value(proxyField).toFloat();
                }
            }

            if (int(myPoint->proxyValues[i]) == int(NODATA))
            {
                gis::Crit3DRasterGrid* proxyGrid = myProxy->getGrid();
                if (proxyGrid == nullptr || ! proxyGrid->isLoaded)
                    return false;
                else
                {
                    float myValue = gis::getValueFromXY(*proxyGrid, myPoint->point.utm.x, myPoint->point.utm.y);
                    if (int(myValue) != int(proxyGrid->header->flag))
                        myPoint->proxyValues[i] = myValue;
                }
            }
        }
    }

    return true;
}


bool Project::loadProxyGrids()
{
    for (unsigned int i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        Crit3DProxy* myProxy = interpolationSettings.getProxy(i);

        logInfo("Loading grid for proxy: " + QString::fromStdString(myProxy->getName()));

        if (interpolationSettings.getSelectedCombination().getValue(i) || myProxy->getForQualityControl())
        {
            gis::Crit3DRasterGrid* myGrid = myProxy->getGrid();

            QString fileName = QString::fromStdString(myProxy->getGridName());
            fileName = getCompleteFileName(fileName, PATH_GEO);

            if (! myGrid->isLoaded && fileName != "")
            {
                gis::Crit3DRasterGrid proxyGrid;
                std::string myError;
                if (gis::readEsriGrid(fileName.toStdString(), &proxyGrid, & myError))
                {
                    gis::Crit3DRasterGrid* resGrid = new gis::Crit3DRasterGrid();
                    gis::resampleGrid(proxyGrid, resGrid, *(DEM.header), aggrAverage, 0);
                    myProxy->setGrid(resGrid);
                }
                else
                {
                    logError("Error loading proxy grid " + fileName);
                    interpolationSettings.getSelectedCombination().setValue(i, false);
                }
                proxyGrid.clear();
            }
        }
    }

    return true;
}


bool Project::loadRadiationGrids()
{
    std::string* myError = new std::string();
    gis::Crit3DRasterGrid *grdLinke, *grdAlbedo;
    std::string gridName;

    if (radSettings.getLinkeMode() == PARAM_MODE_MAP)
    {
        gridName = getCompleteFileName(QString::fromStdString(radSettings.getLinkeMapName()), PATH_GEO).toStdString();
        if (gridName != "")
        {
            grdLinke = new gis::Crit3DRasterGrid();
            if (!gis::readEsriGrid(gridName, grdLinke, myError))
            {
                logError("Error loading Linke grid " + QString::fromStdString(gridName));
                return false;
            }
            radSettings.setLinkeMap(grdLinke);
        }
    }

    if (radSettings.getAlbedoMode() == PARAM_MODE_MAP)
    {
        gridName = getCompleteFileName(QString::fromStdString(radSettings.getAlbedoMapName()), PATH_GEO).toStdString();
        if (gridName != "")
        {
            grdAlbedo = new gis::Crit3DRasterGrid();
            if (!gis::readEsriGrid(gridName, grdAlbedo, myError))
            {
                logError("Error loading albedo grid " + QString::fromStdString(gridName));
                return false;
            }
            radSettings.setAlbedoMap(grdAlbedo);
        }
    }

    return true;
}

bool Project::readProxyValues()
{
    if (meteoPointsDbHandler == nullptr) return false;

    QSqlDatabase myDb = this->meteoPointsDbHandler->getDb();

    for (int i = 0; i < this->nrMeteoPoints; i++)
    {
        if (! readPointProxyValues(&(this->meteoPoints[i]), &myDb)) return false;
    }

    return true;
}


bool Project::updateProxy()
{
    if (! loadProxyGrids()) return false;
    if (! readProxyValues()) return false;
    return true;
}

bool Project::writeTopographicDistanceMaps(bool onlyWithData)
{
    if (nrMeteoPoints == 0)
    {
        logError("Open a meteo points DB before.");
        return false;
    }

    if (! DEM.isLoaded)
    {
        logError("Load a Digital Elevation Map before.");
        return false;
    }

    QString mapsFolder = projectPath + PATH_TAD;
    if (! QDir(mapsFolder).exists())
        QDir().mkdir(mapsFolder);

    FormInfo myInfo;
    QString infoStr = "Computing topographic distance maps...";
    int infoStep = myInfo.start(infoStr, nrMeteoPoints);

    std::string myError;
    std::string fileName;
    gis::Crit3DRasterGrid myMap;
    bool isSelected;

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if ((i % infoStep) == 0) myInfo.setValue(i);

        if (meteoPoints[i].active)
        {
            if (! onlyWithData)
                isSelected = true;
            else {
                isSelected = meteoPointsDbHandler->existData(&meteoPoints[i], daily) || meteoPointsDbHandler->existData(&meteoPoints[i], hourly);
            }

            if (isSelected)
            {
                if (gis::topographicDistanceMap(meteoPoints[i].point, DEM, &myMap))
                {
                    fileName = mapsFolder.toStdString() + "TAD_" + meteoPoints[i].id;
                    if (! gis::writeEsriGrid(fileName, &myMap, &myError))
                    {
                        logError(QString::fromStdString(myError));
                        return false;
                    }
                }
            }
        }
    }

    myInfo.close();

    return true;
}


bool Project::loadTopographicDistanceMaps(bool showInfo)
{
    if (nrMeteoPoints == 0)
    {
        logError("Open a meteo points DB before.");
        return false;
    }

    QString mapsFolder = projectPath + PATH_TAD;
    if (! QDir(mapsFolder).exists())
    {
        logError("TAD folder not found. Please create TAD Maps.");
        return false;
    }

    FormInfo myInfo;
    int infoStep;
    if (showInfo)
    {
        QString infoStr = "Loading topographic distance maps...";
        infoStep = myInfo.start(infoStr, nrMeteoPoints);
    }

    std::string myError;
    std::string fileName;


    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
            if ((i % infoStep) == 0)
                myInfo.setValue(i);

        if (meteoPoints[i].active)
        {
            fileName = mapsFolder.toStdString() + "TAD_" + meteoPoints[i].id;
            meteoPoints[i].topographicDistance = new gis::Crit3DRasterGrid();
            if (! gis::readEsriGrid(fileName, meteoPoints[i].topographicDistance, &myError))
            {
                logError(QString::fromStdString(myError));
                return false;
            }
        }
    }

    if (showInfo) myInfo.close();

    return true;
}

void Project::passInterpolatedTemperatureToHumidityPoints(Crit3DTime myTime)
{
    if (! hourlyMeteoMaps->mapHourlyTair->isLoaded) return;

    float airRelHum, airT;
    int row, col;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        airRelHum = meteoPoints[i].getMeteoPointValue(myTime, airRelHumidity);
        airT = meteoPoints[i].getMeteoPointValue(myTime, airTemperature);

        if (! isEqual(airRelHum, NODATA) && isEqual(airT, NODATA))
        {
            gis::getRowColFromXY(*(hourlyMeteoMaps->mapHourlyTair), meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, &row, &col);
            if (! gis::isOutOfGridRowCol(row, col, *(hourlyMeteoMaps->mapHourlyTair)))
            {
                meteoPoints[i].setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(),
                                          airTemperature, hourlyMeteoMaps->mapHourlyRelHum->value[row][col]);
            }
        }
    }
}


bool Project::interpolationDem(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo)
{
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    // check quality and pass data to interpolation
    if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         &qualityInterpolationSettings, &interpolationSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    // Proxy vars regression and detrend
    if (showInfo && modality == MODE_GUI)
    {
        FormInfo myInfo;
        myInfo.start("Preparing interpolation...", 0);
    }

    //detrending and checking precipitation
    if (! preInterpolation(interpolationPoints, &interpolationSettings, &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime))
    {
        logError("Interpolation: error in function preInterpolation");
        return false;
    }

    // Interpolate
    if (! interpolationRaster(interpolationPoints, &interpolationSettings, myRaster, DEM, myVar, showInfo && modality == MODE_GUI))
    {
        logError("Interpolation: error in function interpolationRaster");
        return false;
    }

    myRaster->setMapTime(myTime);

    return true;
}


bool Project::interpolateDemRadiation(const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo)
{
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    radSettings.setGisSettings(&gisSettings);

    gis::Crit3DPoint mapCenter = DEM.mapCenter();

    int intervalWidth = radiation::estimateTransmissivityWindow(&radSettings, DEM, mapCenter, myTime, int(HOUR_SECONDS));

    // at least a meteoPoint with transmissivity data
    if (! computeTransmissivity(&radSettings, meteoPoints, nrMeteoPoints, intervalWidth, myTime, DEM))
    {
        // TODO: add flag to parameters. Could be NOT wanted
        if (! computeTransmissivityFromTRange(meteoPoints, nrMeteoPoints, myTime))
        {
            logError("Function interpolateDemRadiation: cannot compute transmissivity.");
            return false;
        }
    }

    if (! checkAndPassDataToInterpolation(quality, atmTransmissivity, meteoPoints, nrMeteoPoints,
                                        myTime, &qualityInterpolationSettings,
                                        &interpolationSettings, &climateParameters, interpolationPoints, checkSpatialQuality))
    {
        logError("Function interpolateRasterRadiation: not enough transmissivity data.");
        return false;
    }

    preInterpolation(interpolationPoints, &interpolationSettings, &climateParameters, meteoPoints, nrMeteoPoints, atmTransmissivity, myTime);

    if (! interpolationRaster(interpolationPoints, &interpolationSettings, this->radiationMaps->transmissivityMap, DEM, atmTransmissivity, showInfo))
    {
        logError("Function interpolateRasterRadiation: error interpolating transmissivity.");
        return false;
    }

    if (! radiation::computeRadiationGridPresentTime(&radSettings, this->DEM, this->radiationMaps, myTime))
    {
        logError("Function interpolateRasterRadiation: error computing solar radiation");
        return false;
    }

    if (myRaster != this->radiationMaps->globalRadiationMap)
    {
        myRaster->copyGrid(*(this->radiationMaps->globalRadiationMap));
    }

    return true;
}


bool Project::interpolationDemMain(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo)
{
    if (! DEM.isLoaded)
    {
        logError("Digital Elevation Model not loaded");
        return false;
    }

    if (nrMeteoPoints == 0)
    {
        logError("Open a meteo points DB before.");
        return false;
    }

    if (myVar == noMeteoVar)
    {
        logError("Select a variable before.");
        return false;
    }

    if (myVar == globalIrradiance)
    {
        Crit3DTime halfHour = myTime.addSeconds(-1800);
        return interpolateDemRadiation(halfHour, myRaster, showInfo);
    }
    else
    {
        return interpolationDem(myVar, myTime, myRaster, showInfo);
    }
}



float Project::meteoDataConsistency(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin)
{
    float dataConsistency = 0.0;
    for (int i = 0; i < nrMeteoPoints; i++)
        dataConsistency = MAXVALUE(dataConsistency, meteoPoints[i].obsDataConsistencyH(myVar, timeIni, timeFin));

    return dataConsistency;
}


QString Project::getCompleteFileName(QString fileName, QString secondaryPath)
{
    if (fileName == "") return fileName;

    if (getFilePath(fileName) == "")
    {
        QString completeFileName = this->getDefaultPath() + secondaryPath + fileName;
        return QDir().cleanPath(completeFileName);
    }
    else if (fileName.left(1) == ".")
    {
        QString completeFileName = this->getProjectPath() + fileName;
        return QDir().cleanPath(completeFileName);
    }
    else
    {
        return fileName;
    }
}


QString Project::getRelativePath(QString fileName)
{
    if (fileName != "" && fileName.left(1) != "." && getFilePath(fileName) != "")
    {
        QDir projectDir(getProjectPath());
        QString relativePath = projectDir.relativeFilePath(fileName);
        if (relativePath != fileName)
        {
            fileName = relativePath;
            if (fileName.left(1) != ".")
            {
                fileName = "./" + relativePath;
            }
        }
    }
    return fileName;
}


bool Project::loadProjectSettings(QString settingsFileName)
{
    if (! QFile(settingsFileName).exists() || ! QFileInfo(settingsFileName).isFile())
    {
        logError("Missing settings file: " + settingsFileName);
        return false;
    }

    // project path
    QString filePath = getFilePath(settingsFileName);
    setProjectPath(filePath);

    delete projectSettings;
    projectSettings = new QSettings(settingsFileName, QSettings::IniFormat);

    projectSettings->beginGroup("location");
        double latitude = projectSettings->value("lat").toDouble();
        double longitude = projectSettings->value("lon").toDouble();
        int utmZone = projectSettings->value("utm_zone").toInt();
        int isUtc = projectSettings->value("is_utc").toBool();
        int timeZone = projectSettings->value("time_zone").toInt();
    projectSettings->endGroup();

    if (! gis::isValidUtmTimeZone(utmZone, timeZone))
    {
        logError("Wrong time_zone or utm_zone in file:\n" + settingsFileName);
        return false;
    }

    gisSettings.startLocation.latitude = latitude;
    gisSettings.startLocation.longitude = longitude;
    gisSettings.utmZone = utmZone;
    gisSettings.isUTC = isUtc;
    gisSettings.timeZone = timeZone;

    projectSettings->beginGroup("project");
        QString myPath = projectSettings->value("path").toString();
        if (! myPath.isEmpty())
        {
            // modify project path
            QString newProjectPath;
            if(myPath.left(1) == ".")
            {
                newProjectPath = getProjectPath() + myPath;
                newProjectPath = QDir::cleanPath(newProjectPath);
            }
            else newProjectPath = myPath;

            if (newProjectPath.right(1) != "/") newProjectPath += "/";
            setProjectPath(newProjectPath);
        }

        projectName = projectSettings->value("name").toString();
        demFileName = projectSettings->value("dem").toString();
        dbPointsFileName = projectSettings->value("meteo_points").toString();
        dbGridXMLFileName = projectSettings->value("meteo_grid").toString();
        loadGridDataAtStart = projectSettings->value("load_grid_data_at_start").toBool();
    projectSettings->endGroup();

    projectSettings->beginGroup("settings");
        parametersFileName = projectSettings->value("parameters_file").toString();
        logFileName = projectSettings->value("log_file").toString();
        currentTileMap = projectSettings->value("tile_map").toString();
    projectSettings->endGroup();
    return true;
}


bool Project::searchDefaultPath(QString* path)
{
    QString myPath = getApplicationPath();
    QString myVolumeDOS = myPath.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(myPath + "DATA").exists())
        {
            isFound = true;
            break;
        }

        if (QDir::cleanPath(myPath) == "/" || QDir::cleanPath(myPath) == myVolumeDOS)
            break;

        myPath += "../";
    }

    if (! isFound)
    {
        logError("DATA directory is missing");
        return false;
    }

    *path = QDir::cleanPath(myPath) + "/DATA/";
    return true;
}


frequencyType Project::getCurrentFrequency() const
{
    return currentFrequency;
}

void Project::setCurrentFrequency(const frequencyType &value)
{
    currentFrequency = value;
}

void Project::saveProjectSettings()
{
    projectSettings->beginGroup("location");
    projectSettings->setValue("lat", gisSettings.startLocation.latitude);
    projectSettings->setValue("lon", gisSettings.startLocation.longitude);
        projectSettings->setValue("utm_zone", gisSettings.utmZone);
        projectSettings->setValue("time_zone", gisSettings.timeZone);
        projectSettings->setValue("is_utc", gisSettings.isUTC);
    projectSettings->endGroup();

    projectSettings->beginGroup("project");
        projectSettings->setValue("name", projectName);
        projectSettings->setValue("dem", getRelativePath(demFileName));
        projectSettings->setValue("meteo_points", getRelativePath(dbPointsFileName));
        projectSettings->setValue("meteo_grid", getRelativePath(dbGridXMLFileName));
        projectSettings->setValue("load_grid_data_at_start", loadGridDataAtStart);
    projectSettings->endGroup();

    projectSettings->beginGroup("settings");
        projectSettings->setValue("parameters_file", getRelativePath(parametersFileName));
    projectSettings->endGroup();

    projectSettings->sync();
}

void Project::saveRadiationParameters()
{
    parameters->beginGroup("radiation");
        parameters->setValue("algorithm", QString::fromStdString(getKeyStringRadAlgorithm(radSettings.getAlgorithm())));
        parameters->setValue("real_sky_algorithm", QString::fromStdString(getKeyStringRealSky(radSettings.getRealSkyAlgorithm())));
        parameters->setValue("linke_mode", QString::fromStdString(getKeyStringParamMode(radSettings.getLinkeMode())));
        parameters->setValue("albedo_mode", QString::fromStdString(getKeyStringParamMode(radSettings.getAlbedoMode())));
        parameters->setValue("tilt_mode", QString::fromStdString(getKeyStringTiltMode(radSettings.getTiltMode())));
        parameters->setValue("real_sky", radSettings.getRealSky());
        parameters->setValue("shadowing", radSettings.getShadowing());
        parameters->setValue("linke", QString::number(double(radSettings.getLinke())));
        parameters->setValue("albedo", QString::number(double(radSettings.getAlbedo())));
        parameters->setValue("tilt", QString::number(double(radSettings.getTilt())));
        parameters->setValue("aspect", QString::number(double(radSettings.getAspect())));
        parameters->setValue("clear_sky", QString::number(double(radSettings.getClearSky())));
        parameters->setValue("linke_map", getRelativePath(QString::fromStdString(radSettings.getLinkeMapName())));
        parameters->setValue("albedo_map", getRelativePath(QString::fromStdString(radSettings.getAlbedoMapName())));
        parameters->setValue("linke_monthly", FloatVectorToStringList(radSettings.getLinkeMonthly()));
    parameters->endGroup();
}

void Project::saveInterpolationParameters()
{
    parameters->beginGroup("interpolation");
        parameters->setValue("aggregationMethod", QString::fromStdString(getKeyStringAggregationMethod(interpolationSettings.getMeteoGridAggrMethod())));
        parameters->setValue("algorithm", QString::fromStdString(getKeyStringInterpolationMethod(interpolationSettings.getInterpolationMethod())));
        parameters->setValue("lapseRateCode", interpolationSettings.getUseLapseRateCode());
        parameters->setValue("thermalInversion", interpolationSettings.getUseThermalInversion());
        parameters->setValue("topographicDistance", interpolationSettings.getUseTAD());
        parameters->setValue("optimalDetrending", interpolationSettings.getUseBestDetrending());
        parameters->setValue("useDewPoint", interpolationSettings.getUseDewPoint());
        parameters->setValue("useInterpolationTemperatureForRH", interpolationSettings.getUseInterpolatedTForRH());
        parameters->setValue("thermalInversion", interpolationSettings.getUseThermalInversion());
        parameters->setValue("minRegressionR2", QString::number(interpolationSettings.getMinRegressionR2()));
    parameters->endGroup();

    saveProxies();

    parameters->sync();
}

void Project::saveProxies()
{
    Q_FOREACH (QString group, parameters->childGroups())
    {
        if (group.left(6) == "proxy_")
            parameters->remove(group);
    }

    Crit3DProxy* myProxy;
    for (unsigned int i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        myProxy = interpolationSettings.getProxy(i);
        parameters->beginGroup("proxy_" + QString::fromStdString(myProxy->getName()));
            parameters->setValue("active", interpolationSettings.getSelectedCombination().getValue(i));
            parameters->setValue("table", QString::fromStdString(myProxy->getProxyTable()));
            parameters->setValue("field", QString::fromStdString(myProxy->getProxyField()));
            parameters->setValue("use_for_spatial_quality_control", myProxy->getForQualityControl());
            parameters->setValue("raster", getRelativePath(QString::fromStdString(myProxy->getGridName())));
        parameters->endGroup();
    }
}

void Project::saveGenericParameters()
{
    parameters->beginGroup("meteo");
        parameters->setValue("min_percentage", QString::number(meteoSettings->getMinimumPercentage()));
        parameters->setValue("prec_threshold", QString::number(meteoSettings->getRainfallThreshold()));
        parameters->setValue("samani_coefficient", QString::number(meteoSettings->getTransSamaniCoefficient()));
        parameters->setValue("thom_threshold", QString::number(meteoSettings->getThomThreshold()));
        parameters->setValue("wind_intensity_default", QString::number(meteoSettings->getWindIntensityDefault()));
        parameters->setValue("hourly_intervals", QString::number(meteoSettings->getHourlyIntervals()));
    parameters->endGroup();

    parameters->beginGroup("quality");
        parameters->setValue("reference_height", QString::number(quality->getReferenceHeight()));
        parameters->setValue("delta_temperature_suspect", QString::number(quality->getDeltaTSuspect()));
        parameters->setValue("delta_temperature_wrong", QString::number(quality->getDeltaTWrong()));
        parameters->setValue("relhum_tolerance", QString::number(quality->getRelHumTolerance()));
    parameters->endGroup();

    parameters->beginGroup("climate");
        parameters->setValue("tmin", FloatVectorToStringList(climateParameters.tmin));
        parameters->setValue("tmax", FloatVectorToStringList(climateParameters.tmax));
        parameters->setValue("tdmin", FloatVectorToStringList(climateParameters.tdmin));
        parameters->setValue("tdmax", FloatVectorToStringList(climateParameters.tdmax));
        parameters->setValue("tmin_lapserate", FloatVectorToStringList(climateParameters.tminLapseRate));
        parameters->setValue("tmax_lapserate", FloatVectorToStringList(climateParameters.tmaxLapseRate));
        parameters->setValue("tdmin_lapserate", FloatVectorToStringList(climateParameters.tdMinLapseRate));
        parameters->setValue("tdmax_lapserate", FloatVectorToStringList(climateParameters.tdMaxLapseRate));
    parameters->endGroup();

    parameters->sync();
}


void Project::saveAllParameters()
{
    saveGenericParameters();
    saveInterpolationParameters();
    saveRadiationParameters();
}


void Project::saveProject()
{
    saveProjectSettings();
    saveAllParameters();
}


void Project::createProject(QString path_, QString name_, QString description)
{
    // name
    if (description != "")
        projectName = description;
    else
        projectName = name_;

    // folder
    QString myPath = path_ + "/" + name_ + "/";
    if (! QDir(myPath).exists())
        QDir().mkdir(myPath);

    projectPath = myPath;

    // settings
    delete projectSettings;
    projectSettings = new QSettings(projectPath + name_ + ".ini", QSettings::IniFormat);

    // parameters
    delete parameters;
    parametersFileName = projectPath + PATH_SETTINGS + "parameters.ini";
    parameters = new QSettings(parametersFileName, QSettings::IniFormat);

    saveProject();
}


bool Project::createDefaultProject(QString fileName)
{
    QString path;
    if (! searchDefaultPath(&path)) return false;

    QSettings* defaultSettings = new QSettings(fileName, QSettings::IniFormat);

    defaultSettings->beginGroup("project");
        defaultSettings->setValue("path", path);
    defaultSettings->endGroup();

    defaultSettings->beginGroup("location");
        defaultSettings->setValue("lat", gisSettings.startLocation.latitude);
        defaultSettings->setValue("lon", gisSettings.startLocation.longitude);
        defaultSettings->setValue("utm_zone", gisSettings.utmZone);
        defaultSettings->setValue("time_zone", gisSettings.timeZone);
        defaultSettings->setValue("is_utc", gisSettings.isUTC);
    defaultSettings->endGroup();

    defaultSettings->beginGroup("settings");
        defaultSettings->setValue("parameters_file", "parameters.ini");
    defaultSettings->endGroup();

    defaultSettings->sync();

    return true;
}


bool Project::start(QString appPath)
{
    if (appPath.right(1) != "/") appPath += "/";
    setApplicationPath(appPath);
    QString defaultProject = getApplicationPath() + "default.ini";

    if (! QFile(defaultProject).exists())
    {
        if (! createDefaultProject(defaultProject))
            return false;
    }

    if (! loadProjectSettings(defaultProject))
        return false;

    setDefaultPath(getProjectPath());
    return true;
}


bool Project::loadProject()
{
    if (logFileName != "") setLogFile(logFileName);

    if (! loadParameters(parametersFileName))
    {
        logError();
        return false;
    }

    if (demFileName != "") loadDEM(demFileName);

    if (dbPointsFileName != "") loadMeteoPointsDB(dbPointsFileName);

    if (dbGridXMLFileName != "") loadMeteoGridDB(dbGridXMLFileName);

    return true;
}


bool Project::checkMeteoGridForExport()
{
    if (! meteoGridLoaded || meteoGridDbHandler == nullptr)
    {
        logError("Open meteo grid before.");
        return false;
    }

    if (meteoGridDbHandler->meteoGrid()->gridStructure().isUTM() ||
        meteoGridDbHandler->meteoGrid()->gridStructure().isTIN())
    {
        logError("latlon grid requested.");
        return false;
    }

    return true;
}


gis::Crit3DRasterGrid* Project::getHourlyMeteoRaster(meteoVariable myVar)
{
    if (myVar == globalIrradiance)
    {
        return radiationMaps->globalRadiationMap;
    }
    else
    {
        return hourlyMeteoMaps->getMapFromVar(myVar);
    }
}


/*!
    \name importHourlyMeteoData
    \brief import hourly meteo data from .csv files
    \details format:
    DATE(yyyy-mm-dd), HOUR, TAVG, PREC, RHAVG, RAD, W_SCAL_INT
*/
void Project::importHourlyMeteoData(const QString& csvFileName, bool importAllFiles, bool deletePreviousData)
{
    QString filePath = getFilePath(csvFileName);
    QStringList fileList;

    if (importAllFiles)
    {
        QDir myDir = QDir(filePath);
        myDir.setNameFilters(QStringList("*.csv"));
        fileList = myDir.entryList();
    }
    else
    {
        // single file
        fileList << getFileName(csvFileName);
    }

    // cycle on files
    for (int i=0; i < fileList.count(); i++)
    {
        QString myLog = "";
        QString fileNameComplete = filePath + fileList[i];

        if (meteoPointsDbHandler->importHourlyMeteoData(fileNameComplete, deletePreviousData, &myLog))
            logInfo(myLog);
        else
            logError(myLog);
    }
}



/* ---------------------------------------------
 * LOG functions
 * --------------------------------------------*/

bool Project::setLogFile(QString myFileName)
{
    this->logFileName = myFileName;
    myFileName = getCompleteFileName(myFileName, "LOG/");

    QString filePath = getFilePath(myFileName);
    QString fileName = getFileName(myFileName);

    if (!QDir(filePath).exists())
    {
         QDir().mkdir(filePath);
    }

    QString myDate = QDateTime().currentDateTime().toString("yyyyMMdd_HHmm");
    fileName = myDate + "_" + fileName;

    QString currentFileName = filePath + fileName;

    logFile.open(currentFileName.toStdString().c_str());
    if (logFile.is_open())
    {
        logInfo("LogFile: " + currentFileName);
        return true;
    }
    else
    {
        logError("Unable to open file: " + currentFileName);
        return false;
    }
}


void Project::logInfo(QString myStr)
{
    // standard output in all modalities
    std::cout << myStr.toStdString() << std::endl;

    if (logFile.is_open())
    {
        logFile << myStr.toStdString() << std::endl;
    }
}


void Project::logInfoGUI(QString myStr)
{
    if (modality == MODE_GUI)
    {
        QMessageBox::information(nullptr, "Information", myStr);
    }
    else
    {
        std::cout << myStr.toStdString() << std::endl;
    }

    if (logFile.is_open())
    {
        logFile << myStr.toStdString() << std::endl;
    }
}


void Project::logError(QString myStr)
{
    errorString = myStr;
    logError();
}


void Project::logError()
{
    if (logFile.is_open())
    {
        logFile << "ERROR! " << errorString.toStdString() << std::endl;
    }

    if (modality == MODE_GUI)
    {
        QMessageBox::critical(nullptr, "ERROR!", errorString);
    }
    else
    {
        std::cout << "ERROR! " << errorString.toStdString() << std::endl;
    }
}




