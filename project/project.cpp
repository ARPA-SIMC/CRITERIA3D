#include "project.h"
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
#include "meteoWidget.h"
#include "dialogSelectionMeteoPoint.h"
#include "dialogPointDeleteData.h"
#include "formInfo.h"


#include <iostream>
#include <QDir>
#include <QFile>
#include <QSqlQuery>
#include <QMessageBox>
#include <string>


Project::Project()
{
    // TODO remove pointers
    meteoSettings = new Crit3DMeteoSettings();
    quality = new Crit3DQuality();
    meteoPointsColorScale = new Crit3DColorScale();
    meteoGridDbHandler = nullptr;
    formLog = nullptr;

    // They not change after loading default settings
    appPath = "";
    defaultPath = "";
    computeOnlyPoints = false;

    initializeProject();

    modality = MODE_GUI;

    verboseStdoutLogging = true;
}


void Project::initializeProject()
{
    projectPath = "";
    projectName = "";
    isProjectLoaded = false;
    requestedExit = false;
    logFileName = "";
    errorString = "";
    errorType = ERROR_NONE;
    currentTileMap = "";

    nrMeteoPoints = 0;
    meteoPoints = nullptr;
    meteoPointsDbHandler = nullptr;
    outputPointsDbHandler = nullptr;
    meteoGridDbHandler = nullptr;
    aggregationDbHandler = nullptr;

    meteoSettings->initialize();
    quality->initialize();

    checkSpatialQuality = true;
    currentVariable = noMeteoVar;
    currentFrequency = noFrequency;
    currentDate.setDate(1800,1,1);
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
    dbAggregationFileName = "";
    aggregationPath = "";
    dbGridXMLFileName = "";
    outputPointsFileName = "";
    currentDbOutputFileName = "";

    meteoPointsLoaded = false;
    meteoGridLoaded = false;
    loadGridDataAtStart = false;

    proxyGridSeries.clear();
    proxyWidget = nullptr;
}

void Project::clearProject()
{
    if (logFile.is_open()) logFile.close();

    meteoPointsColorScale->setRange(NODATA, NODATA);

    delete parameters;
    delete projectSettings;
    delete aggregationDbHandler;
    meteoWidgetPointList.clear();
    meteoWidgetGridList.clear();

    clearProxyDEM();
    DEM.clear();

    delete radiationMaps;
    delete hourlyMeteoMaps;
    radiationMaps = nullptr;
    hourlyMeteoMaps = nullptr;

    closeMeteoPointsDB();
    closeMeteoGridDB();

    closeOutputPointsDB();
    outputPoints.clear();
    outputPointsFileName = "";

    if (proxyWidget != nullptr)
    {
        delete proxyWidget;
    }

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

bool Project::checkProxy(const Crit3DProxy &myProxy, QString* error)
{
    std::string name_ = myProxy.getName();

    if (name_ == "")
    {
        *error = "no name";
        return false;
    }

    bool isHeight = (getProxyPragaName(name_) == height);

    if (!isHeight && (myProxy.getGridName() == "") && (myProxy.getProxyTable() == "" && myProxy.getProxyField() == ""))
    {
        *error = "error reading grid, table or field for proxy " + QString::fromStdString(name_);
        return false;
    }

    return true;
}

bool Project::addProxyToProject(std::vector <Crit3DProxy> proxyList, std::deque <bool> proxyActive, std::vector <int> proxyOrder)
{
    unsigned i, order;
    bool orderFound;

    for (order=1; order <= proxyList.size(); order++)
    {
        orderFound = false;

        for (i=0; i < proxyList.size(); i++)
            if (unsigned(proxyOrder[i]) == order)
            {
                interpolationSettings.addProxy(proxyList[i], proxyActive[i]);
                if (proxyList[i].getForQualityControl())
                    qualityInterpolationSettings.addProxy(proxyList[i], proxyActive[i]);

                orderFound = true;

                break;
            }

        if (! orderFound) return false;
    }

    for (i=0; i < interpolationSettings.getProxyNr(); i++)
        if (getProxyPragaName(interpolationSettings.getProxy(i)->getName()) == height) setProxyDEM();

    return true;

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
        errorString = "Missing parameters file: " + parametersFileName;
        errorString += "\nCheck project.path in " + projectSettings->fileName();
        return false;
    }

    delete parameters;
    parameters = new QSettings(parametersFileName, QSettings::IniFormat);

    //interpolation settings
    interpolationSettings.initialize();
    qualityInterpolationSettings.initialize();

    std::vector <Crit3DProxy> proxyListTmp;
    std::deque <bool> proxyActiveTmp;
    std::vector <int> proxyOrder;

    QList<QString> myList;
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
            if (parameters->contains("temperature_threshold") && !parameters->value("temperature_threshold").toString().isEmpty())
            {
                meteoSettings->setTemperatureThreshold(parameters->value("temperature_threshold").toFloat());
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
            if (parameters->contains("compute_tavg") && !parameters->value("compute_tavg").toString().isEmpty())
            {
                meteoSettings->setAutomaticTavg(parameters->value("compute_tavg").toBool());
            }
            if (parameters->contains("compute_et0hs") && !parameters->value("compute_et0hs").toString().isEmpty())
            {
                meteoSettings->setAutomaticET0HS(parameters->value("compute_et0hs").toBool());
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
                QList<QString> myLinkeStr = parameters->value("linke_monthly").toStringList();
                if (myLinkeStr.size() < 12)
                {
                    errorString = "Incomplete monthly Linke values";
                    return  false;
                }

                radSettings.setLinkeMonthly(StringListToFloat(myLinkeStr));
            }

            if (parameters->contains("albedo_monthly"))
            {
                QList<QString> myAlbedoStr = parameters->value("albedo_monthly").toStringList();
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
                interpolationSettings.setUseTD(parameters->value("topographicDistance").toBool());

            if (parameters->contains("dynamicLapserate"))
                interpolationSettings.setUseDynamicLapserate(parameters->value("dynamicLapserate").toBool());

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

            if (parameters->contains("topographicDistanceMaxMultiplier"))
            {
                interpolationSettings.setTopoDist_maxKh(parameters->value("topographicDistanceMaxMultiplier").toInt());
                qualityInterpolationSettings.setTopoDist_maxKh(parameters->value("topographicDistanceMaxMultiplier").toInt());
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
            Crit3DProxy* myProxy = new Crit3DProxy();

            myProxy->setName(group.right(group.size()-6).toStdString());

            parameters->beginGroup(group);

            myProxy->setProxyTable(parameters->value("table").toString().toStdString());
            myProxy->setProxyField(parameters->value("field").toString().toStdString());
            myProxy->setGridName(getCompleteFileName(parameters->value("raster").toString(), PATH_GEO).toStdString());
            myProxy->setForQualityControl(parameters->value("use_for_spatial_quality_control").toBool());

            if (! parameters->contains("active"))
            {
                errorString = "active not specified for proxy " + QString::fromStdString(myProxy->getName());
                return false;
            }

            if (! parameters->contains("order"))
            {
                errorString = "order not specified for proxy " + QString::fromStdString(myProxy->getName());
                return false;
            }

            if (checkProxy(*myProxy, &errorString))
            {
                proxyListTmp.push_back(*myProxy);
                proxyActiveTmp.push_back(parameters->value("active").toBool());
                proxyOrder.push_back(parameters->value("order").toInt());
            }
            else
                logError();

            parameters->endGroup();
        }

        //proxy grid annual series
        if (group.startsWith("proxygrid"))
        {
            QString proxyName = group.right(group.length()-10);

            parameters->beginGroup(group);
            int nrGrids = parameters->beginReadArray("grids");
            for (int i = 0; i < nrGrids; ++i) {
                parameters->setArrayIndex(i);
                proxyGridSeriesNames.push_back(getCompleteFileName(parameters->value("name").toString(), PATH_GEO));
                proxyGridSeriesYears.push_back(parameters->value("year").toUInt());
            }
            parameters->endArray();
            parameters->endGroup();

            addProxyGridSeries(proxyName, proxyGridSeriesNames, proxyGridSeriesYears);
        }
    }

    if (proxyListTmp.size() > 0)
        addProxyToProject(proxyListTmp, proxyActiveTmp, proxyOrder);

    if (! updateProxy())
        return false;

    if (!loadRadiationGrids())
        return false;

    return true;
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

meteoVariable Project::getCurrentVariable() const
{
    return this->currentVariable;
}

void Project::setCurrentDate(QDate myDate)
{
    if (myDate != this->currentDate)
    {
        this->currentDate = myDate;
        if (proxyWidget != nullptr)
        {
            proxyWidget->updateDateTime(currentDate, currentHour);
        }
    }
}

QDate Project::getCurrentDate()
{
    return this->currentDate;
}

void Project::setCurrentHour(int myHour)
{
    this->currentHour = myHour;
    if (proxyWidget != nullptr)
    {
        proxyWidget->updateDateTime(currentDate, currentHour);
    }
}

int Project::getCurrentHour()
{
    return this->currentHour;
}

Crit3DTime Project::getCrit3DCurrentTime()
{
    if (currentFrequency == hourly)
    {
        return getCrit3DTime(this->currentDate, this->currentHour);
    }
    else
    {
        return getCrit3DTime(this->currentDate, 0);
    }
}


QDateTime Project::getCurrentTime()
{
    QDateTime myTime;
    myTime.setDate(this->currentDate);
    return myTime.addSecs(this->currentHour * HOUR_SECONDS);
}

void Project::getMeteoPointsRange(float& minimum, float& maximum, bool useNotActivePoints)
{
    minimum = NODATA;
    maximum = NODATA;

    if (currentFrequency == noFrequency || currentVariable == noMeteoVar)
        return;

    float v;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active || useNotActivePoints)
        {
            v = meteoPoints[i].currentValue;
            if (! isEqual(v, NODATA) && meteoPoints[i].quality == quality::accepted)
            {
                if (isEqual(minimum, NODATA))
                {
                    minimum = v;
                    maximum = v;
                }
                else if (v < minimum) minimum = v;
                else if (v > maximum) maximum = v;
            }
        }
    }
}

void Project::cleanMeteoPointsData()
{
    if (nrMeteoPoints > 0 && meteoPoints != nullptr)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            meteoPoints[i].cleanObsDataH();
            meteoPoints[i].cleanObsDataD();
            meteoPoints[i].cleanObsDataM();
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

    dbPointsFileName = "";
    meteoPointsLoaded = false;
}


void Project::closeOutputPointsDB()
{
    if (outputPointsDbHandler != nullptr)
    {
        delete outputPointsDbHandler;
        outputPointsDbHandler = nullptr;
    }

    currentDbOutputFileName = "";
}


void Project::closeMeteoGridDB()
{
    if (meteoGridDbHandler != nullptr)
    {
        delete meteoGridDbHandler;  //this also close db
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

    logInfoGUI("Load DEM = " + myFileName);

    demFileName = myFileName;
    myFileName = getCompleteFileName(myFileName, PATH_DEM);

    std::string error;
    if (! gis::openRaster(myFileName.toStdString(), &DEM, gisSettings.utmZone, error))
    {
        closeLogInfo();
        logError("Wrong Digital Elevation Model:\n" + QString::fromStdString(error));
        errorType = ERROR_DEM;
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
    interpolationSettings.setProxyLoaded(false);
    if (! updateProxy()) return false;

    //set interpolation settings DEM
    interpolationSettings.setCurrentDEM(&DEM);
    qualityInterpolationSettings.setCurrentDEM(&DEM);

    //check points position with respect to DEM
    checkMeteoPointsDEM();

    return true;
}


bool Project::loadMeteoPointsDB(QString fileName)
{
    if (fileName == "") return false;

    logInfoGUI("Load meteo points DB = " + fileName);

    closeMeteoPointsDB();

    dbPointsFileName = fileName;
    QString dbName = getCompleteFileName(fileName, PATH_METEOPOINT);
    if (! QFile(dbName).exists())
    {
        errorString = "Meteo points DB does not exists:\n" + dbName;
        return false;
    }

    meteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(dbName);
    if (meteoPointsDbHandler->error != "")
    {
        errorString = "Function loadMeteoPointsDB:\n" + dbName + "\n" + meteoPointsDbHandler->error;
        closeMeteoPointsDB();
        return false;
    }

    if (! meteoPointsDbHandler->loadVariableProperties())
    {
        errorString = meteoPointsDbHandler->error;
        closeMeteoPointsDB();
        return false;
    }

    QList<Crit3DMeteoPoint> listMeteoPoints;
    errorString = "";
    if (! meteoPointsDbHandler->getPropertiesFromDb(listMeteoPoints, gisSettings, errorString))
    {
        errorString = "Error in reading table 'point_properties'\n" + errorString;
        closeMeteoPointsDB();
        return false;
    }

    nrMeteoPoints = listMeteoPoints.size();
    if (nrMeteoPoints == 0)
    {
        errorString = "Missing data in the table 'point_properties'\n" + errorString;
        closeMeteoPointsDB();
        return false;
    }

    // warning
    if (errorString != "")
    {
        logError();
        errorString = "";
    }

    meteoPoints = new Crit3DMeteoPoint[unsigned(nrMeteoPoints)];

    for (int i=0; i < nrMeteoPoints; i++)
        meteoPoints[i] = listMeteoPoints[i];

    listMeteoPoints.clear();

    // find dates
    meteoPointsDbLastTime = findDbPointLastTime();

    if (! meteoPointsDbLastTime.isNull())
    {
        if (meteoPointsDbLastTime.time().hour() == 00)
        {
            setCurrentDate(meteoPointsDbLastTime.date().addDays(-1));
            setCurrentHour(24);
        }
        else
        {
            setCurrentDate(meteoPointsDbLastTime.date());
            setCurrentHour(meteoPointsDbLastTime.time().hour());
        }
    }

    // load proxy values for detrending
    logInfoGUI("Read proxy values: " + fileName);
    if (! readProxyValues())
    {
        logError("Error reading proxy values");
    }

    //position with respect to DEM
    if (DEM.isLoaded)
        checkMeteoPointsDEM();

    meteoPointsLoaded = true;
    logInfo("Meteo points DB = " + dbName);
    closeLogInfo();

    return true;
}

bool Project::loadAggregationDBAsMeteoPoints(QString fileName)
{
    if (fileName == "") return false;

    logInfoGUI("Load meteo points DB = " + fileName);

    dbPointsFileName = fileName;
    QString dbName = getCompleteFileName(fileName, PATH_PROJECT);
    if (! QFile(dbName).exists())
    {
        errorString = "Aggregation points DB does not exists:\n" + dbName;
        return false;
    }

    meteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(dbName);
    if (meteoPointsDbHandler->error != "")
    {
        errorString = "Function loadAggregationPointsDB:\n" + dbName + "\n" + meteoPointsDbHandler->error;
        closeMeteoPointsDB();
        return false;
    }

    if (! meteoPointsDbHandler->loadVariableProperties())
    {
        errorString = meteoPointsDbHandler->error;
        closeMeteoPointsDB();
        return false;
    }

    QList<Crit3DMeteoPoint> listMeteoPoints;
    errorString = "";
    if (! meteoPointsDbHandler->getPropertiesFromDb(listMeteoPoints, gisSettings, errorString))
    {
        errorString = "Error in reading table 'point_properties'\n" + errorString;
        closeMeteoPointsDB();
        return false;
    }

    nrMeteoPoints = listMeteoPoints.size();
    if (nrMeteoPoints == 0)
    {
        errorString = "Missing data in the table 'point_properties'\n" + errorString;
        closeMeteoPointsDB();
        return false;
    }

    // warning
    if (errorString != "")
    {
        logError();
        errorString = "";
    }

    meteoPoints = new Crit3DMeteoPoint[unsigned(nrMeteoPoints)];

    for (int i=0; i < nrMeteoPoints; i++)
        meteoPoints[i] = listMeteoPoints[i];

    listMeteoPoints.clear();

    // find dates
    meteoPointsDbLastTime = findDbPointLastTime();

    if (! meteoPointsDbLastTime.isNull())
    {
        if (meteoPointsDbLastTime.time().hour() == 00)
        {
            setCurrentDate(meteoPointsDbLastTime.date().addDays(-1));
            setCurrentHour(24);
        }
        else
        {
            setCurrentDate(meteoPointsDbLastTime.date());
            setCurrentHour(meteoPointsDbLastTime.time().hour());
        }
    }

    // load proxy values for detrending
    logInfoGUI("Read proxy values: " + fileName);
    if (! readProxyValues())
    {
        logError("Error reading proxy values");
    }

    //position with respect to DEM
    if (DEM.isLoaded)
        checkMeteoPointsDEM();

    meteoPointsLoaded = true;
    logInfo("Meteo points DB = " + dbName);
    closeLogInfo();

    return true;
}


bool Project::newOutputPointsDB(QString dbName)
{
    if (dbName == "") return false;

    closeOutputPointsDB();
    currentDbOutputFileName = dbName;

    dbName = getCompleteFileName(dbName, PATH_METEOPOINT);
    QFile outputDb(dbName);
    if (outputDb.exists())
    {
        if (!outputDb.remove())
        {
            logError("Failed to remove existing output db.");
            currentDbOutputFileName = "";
            return false;
        }
    }

    outputPointsDbHandler = new Crit3DOutputPointsDbHandler(dbName);
    if (outputPointsDbHandler->getErrorString() != "")
    {
        logError("Function newOutputPointsDB:\n" + dbName + "\n" + outputPointsDbHandler->getErrorString());
        closeOutputPointsDB();
        return false;
    }

    return true;
}


bool Project::loadOutputPointsDB(QString dbName)
{
    if (dbName == "") return false;

    closeOutputPointsDB();

    currentDbOutputFileName = dbName;
    dbName = getCompleteFileName(dbName, PATH_METEOPOINT);
    if (! QFile(dbName).exists())
    {
        logError("Output points db does not exists:\n" + dbName);
        return false;
    }

    outputPointsDbHandler = new Crit3DOutputPointsDbHandler(dbName);
    if (outputPointsDbHandler->getErrorString() != "")
    {
        logError("Function loadOutputPointsDB:\n" + dbName + "\n" + outputPointsDbHandler->getErrorString());
        closeOutputPointsDB();
        return false;
    }

    return true;
}


bool Project::loadMeteoGridDB(QString xmlName)
{
    if (xmlName == "") return false;

    closeMeteoGridDB();

    dbGridXMLFileName = xmlName;
    xmlName = getCompleteFileName(xmlName, PATH_METEOGRID);

    meteoGridDbHandler = new Crit3DMeteoGridDbHandler();
    meteoGridDbHandler->meteoGrid()->setGisSettings(this->gisSettings);

    if (! meteoGridDbHandler->parseXMLGrid(xmlName, &errorString)) return false;

    if (! this->meteoGridDbHandler->openDatabase(&errorString)) return false;

    if (! this->meteoGridDbHandler->loadCellProperties(&errorString)) return false;

    if (! this->meteoGridDbHandler->meteoGrid()->createRasterGrid()) return false;

    if (!meteoGridDbHandler->updateGridDate(&errorString))
    {
        logInfoGUI("updateGridDate: " + errorString);
    }

    if (loadGridDataAtStart || ! meteoPointsLoaded)
        setCurrentDate(meteoGridDbHandler->lastDate());

    meteoGridLoaded = true;
    logInfo("Meteo Grid = " + xmlName);

    return true;
}


bool Project::newMeteoGridDB(QString xmlName)
{
    if (xmlName == "") return false;

    closeMeteoGridDB();

    dbGridXMLFileName = xmlName;
    xmlName = getCompleteFileName(xmlName, PATH_METEOGRID);

    meteoGridDbHandler = new Crit3DMeteoGridDbHandler();
    meteoGridDbHandler->meteoGrid()->setGisSettings(this->gisSettings);

    if (! meteoGridDbHandler->parseXMLGrid(xmlName, &errorString)) return false;

    if (! this->meteoGridDbHandler->newDatabase(&errorString)) return false;

    if (! this->meteoGridDbHandler->newCellProperties(&errorString)) return false;

    Crit3DMeteoGridStructure structure = this->meteoGridDbHandler->meteoGrid()->gridStructure();

    if (! this->meteoGridDbHandler->writeCellProperties(&errorString, structure.nrRow(), structure.nrCol())) return false;

    if (! this->meteoGridDbHandler->meteoGrid()->createRasterGrid()) return false;

    if (!meteoGridDbHandler->updateGridDate(&errorString))
    {
        logInfoGUI("updateGridDate: " + errorString);
    }

    if (loadGridDataAtStart || ! meteoPointsLoaded)
        setCurrentDate(meteoGridDbHandler->lastDate());

    meteoGridLoaded = true;
    logInfo("Meteo Grid = " + xmlName);

    return true;
}

bool Project::deleteMeteoGridDB()
{
    if (!meteoGridDbHandler->deleteDatabase(&errorString))
    {
        logInfoGUI("delete meteo grid error: " + errorString);
        return false;
    }
    return true;
}


bool Project::loadAggregationdDB(QString dbName)
{
    if (dbName == "") return false;

    dbAggregationFileName = dbName;
    dbName = getCompleteFileName(dbName, PATH_PROJECT);

    aggregationDbHandler = new Crit3DAggregationsDbHandler(dbName);
    if (aggregationDbHandler->error() != "")
    {
        logError(aggregationDbHandler->error());
        return false;
    }
    if (!aggregationDbHandler->loadVariableProperties())
    {
        return false;
    }
    aggregationPath = QFileInfo(dbAggregationFileName).absolutePath();
    return true;
}


bool Project::loadMeteoPointsData(const QDate& firstDate, const QDate& lastDate, bool loadHourly, bool loadDaily, bool showInfo)
{
    //check
    if (firstDate == QDate(1800,1,1) || lastDate == QDate(1800,1,1)) return false;

    bool isData = false;
    int step = 0;

    QString infoStr = "Load meteo points data: " + firstDate.toString();

    if (firstDate != lastDate)
    {
        infoStr += " - " + lastDate.toString();
    }

    if (showInfo)
    {
        step = setProgressBar(infoStr, nrMeteoPoints);
    }

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % step) == 0) updateProgressBar(i);
        }

        if (loadHourly)
            if (meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;

        if (loadDaily)
            if (meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;
    }

    if (showInfo) closeProgressBar();

    return isData;
}


bool Project::loadMeteoPointsData(const QDate &firstDate, const QDate &lastDate, bool loadHourly, bool loadDaily, const QString &dataset, bool showInfo)
{
    //check
    if (firstDate == QDate(1800,1,1) || lastDate == QDate(1800,1,1)) return false;

    bool isData = false;
    int step = 0;

    QString infoStr = "Load meteo points data: " + firstDate.toString();

    if (firstDate != lastDate)
        infoStr += " - " + lastDate.toString();

    if (showInfo)
    {
        step = setProgressBar(infoStr, nrMeteoPoints);
    }

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % step) == 0) updateProgressBar(i);
        }

        if (meteoPoints[i].dataset == dataset.toStdString())
        {
            if (loadHourly)
                if (meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;

            if (loadDaily)
                if (meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &(meteoPoints[i]))) isData = true;
        }
    }

    if (showInfo) closeProgressBar();
    return isData;
}


void Project::loadMeteoGridData(QDate firstDate, QDate lastDate, bool showInfo)
{
    if (this->meteoGridDbHandler != nullptr)
    {
        this->loadMeteoGridDailyData(firstDate, lastDate, showInfo);
        this->loadMeteoGridHourlyData(QDateTime(firstDate, QTime(1,0), Qt::UTC), QDateTime(lastDate.addDays(1), QTime(0,0), Qt::UTC), showInfo);
        this->loadMeteoGridMonthlyData(firstDate, lastDate, showInfo);
    }
}


bool Project::loadMeteoGridDailyData(QDate firstDate, QDate lastDate, bool showInfo)
{
    if (! meteoGridDbHandler->tableDaily().exists) return false;

    std::string id;
    int count = 0;

    int infoStep = 1;

    if (showInfo)
    {
        QString infoStr = "Load meteo grid daily data: " + firstDate.toString();
        if (firstDate != lastDate) infoStr += " - " + lastDate.toString();
        infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
        {
            updateProgressBar(row);
        }

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                {
                    if (this->meteoGridDbHandler->gridStructure().isEnsemble())
                    {
                        int memberNr = 1;
                        if (this->meteoGridDbHandler->loadGridDailyDataEnsemble(errorString, QString::fromStdString(id), memberNr, firstDate, lastDate))
                        {
                            count = count + 1;
                        }
                    }
                    else
                    {
                        if (this->meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(id), firstDate, lastDate))
                        {
                            count = count + 1;
                        }
                    }
                }
                else
                {
                    if (this->meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

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
    int infoStep = 1;

    if (! meteoGridDbHandler->tableHourly().exists) return false;

    if (showInfo)
    {
        QString infoStr = "Load meteo grid hourly data: " + firstDate.toString("yyyy-MM-dd:hh") + " - " + lastDate.toString("yyyy-MM-dd:hh");
        infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
        {
            updateProgressBar(row);
        }

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                {
                    if (this->meteoGridDbHandler->loadGridHourlyData(errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
                else
                {
                    if (this->meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(id), firstDate, lastDate))
                    {
                        count = count + 1;
                    }
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

    if (count == 0)
    {
        errorString = "No Data Available";
        return false;
    }
    else
        return true;
}

bool Project::loadMeteoGridMonthlyData(QDate firstDate, QDate lastDate, bool showInfo)
{
    if (! meteoGridDbHandler->tableMonthly().exists) return false;

    std::string id;
    int count = 0;

    int infoStep = 1;

    if (showInfo)
    {
        QString infoStr = "Load meteo grid monthly data: " + firstDate.toString();
        if (firstDate != lastDate) infoStr += " - " + lastDate.toString();
        infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
        {
            updateProgressBar(row);
        }

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {

                if (this->meteoGridDbHandler->loadGridMonthlyData(errorString, QString::fromStdString(id), firstDate, lastDate))
                {
                    count = count + 1;
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

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
    lastTime.setTimeSpec(Qt::UTC);

    QDateTime lastDateD;
    lastDateD.setTimeSpec(Qt::UTC);
    lastDateD = meteoPointsDbHandler->getLastDate(daily);
    if (! lastDateD.isNull()) lastTime = lastDateD;

    QDateTime lastDateH;
    lastDateH.setTimeSpec(Qt::UTC);
    lastDateH = meteoPointsDbHandler->getLastDate(hourly);

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
    firstTime.setTimeSpec(Qt::UTC);

    QDateTime firstDateD;
    firstDateD.setTimeSpec(Qt::UTC);
    firstDateD = meteoPointsDbHandler->getFirstDate(daily);
    if (! firstDateD.isNull()) firstTime = firstDateD;

    QDateTime firstDateH;
    firstDateH.setTimeSpec(Qt::UTC);
    firstDateH = meteoPointsDbHandler->getFirstDate(hourly);

    if (! firstDateH.isNull())
    {
        if (! firstTime.isNull())
            firstTime = (firstDateD < firstDateH) ? firstDateD : firstDateH;
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
    float myValue;

    nrProxy = int(interpolationSettings.getProxyNr());
    myPoint->proxyValues.resize(unsigned(nrProxy));

    for (unsigned int i=0; i < unsigned(nrProxy); i++)
    {
        myPoint->proxyValues[i] = NODATA;

        // read for all proxies (also not active) for proxy graphs

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
                {
                    if (! qry.value(proxyField).isNull())
                        myPoint->proxyValues[i] = qry.value(proxyField).toFloat();
                }
            }
        }

        if (int(myPoint->proxyValues[i]) == int(NODATA))
        {
            gis::Crit3DRasterGrid* proxyGrid = myProxy->getGrid();
            if (proxyGrid == nullptr || ! proxyGrid->isLoaded)
                return false;
            else
            {
                myValue = gis::getValueFromXY(*proxyGrid, myPoint->point.utm.x, myPoint->point.utm.y);
                if (! isEqual(myValue, proxyGrid->header->flag))
                    myPoint->proxyValues[i] = myValue;
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

        logInfoGUI("Loading grid for proxy: " + QString::fromStdString(myProxy->getName()));

        gis::Crit3DRasterGrid* myGrid = myProxy->getGrid();

        QString fileName = QString::fromStdString(myProxy->getGridName());
        fileName = getCompleteFileName(fileName, PATH_GEO);

        if (! myGrid->isLoaded && fileName != "")
        {
            gis::Crit3DRasterGrid proxyGrid;
            std::string myError;
            if (DEM.isLoaded && gis::readEsriGrid(fileName.toStdString(), &proxyGrid, myError))
            {
                gis::Crit3DRasterGrid* resGrid = new gis::Crit3DRasterGrid();
                gis::resampleGrid(proxyGrid, resGrid, DEM.header, aggrAverage, 0);
                myProxy->setGrid(resGrid);
            }
            else
            {
                errorString = "Error loading proxy grid " + fileName;
                return false;
            }

            proxyGrid.clear();
        }

        closeLogInfo();
    }

    return true;
}


bool Project::loadRadiationGrids()
{
    std::string myError = "";
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
    QSqlDatabase myDb = this->meteoPointsDbHandler->getDb();

    for (int i = 0; i < this->nrMeteoPoints; i++)
    {
        if (! readPointProxyValues(&(this->meteoPoints[i]), &myDb)) return false;
    }

    return true;
}


bool Project::updateProxy()
{
    if (DEM.isLoaded)
        if (! interpolationSettings.getProxyLoaded())
        {
            if (loadProxyGrids()) interpolationSettings.setProxyLoaded(true);
            else return false;

            if (meteoPointsDbHandler != nullptr)
                if (! readProxyValues()) return false;
        }

    return true;
}

bool Project::writeTopographicDistanceMaps(bool onlyWithData, bool showInfo)
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

    QString mapsFolder = projectPath + PATH_TD;
    if (! QDir(mapsFolder).exists())
        QDir().mkdir(mapsFolder);

    QString infoStr = "Computing topographic distance maps...";
    int infoStep = 0;
    if (showInfo)
    {
        infoStep = setProgressBar(infoStr, nrMeteoPoints);
    }

    std::string myError;
    std::string fileName;
    gis::Crit3DRasterGrid myMap;
    bool isSelected;

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % infoStep) == 0) updateProgressBar(i);
        }

        if (meteoPoints[i].active)
        {
            if (! onlyWithData)
                isSelected = true;
            else {
                isSelected = meteoPointsDbHandler->existData(&meteoPoints[i], daily) || meteoPointsDbHandler->existData(&meteoPoints[i], hourly);
            }

            if (isSelected)
            {
                if (!writeTopographicDistanceMap(i, DEM, mapsFolder))
                {
                    return false;
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

    return true;
}

bool Project::writeTopographicDistanceMap(int pointIndex, const gis::Crit3DRasterGrid& demMap, QString pathTd)
{
    if (nrMeteoPoints == 0)
    {
        logError("Open a meteo points DB before.");
        return false;
    }

    if (! demMap.isLoaded)
    {
        logError("Load a Digital Elevation Map before.");
        return false;
    }

    if (! QDir(pathTd).exists())
        QDir().mkdir(pathTd);

    std::string myError;
    std::string fileName;
    gis::Crit3DRasterGrid myMap;

    if (gis::topographicDistanceMap(meteoPoints[pointIndex].point, demMap, &myMap))
    {
        fileName = pathTd.toStdString() + "TD_" + QFileInfo(demFileName).baseName().toStdString() + "_" + meteoPoints[pointIndex].id;
        if (! gis::writeEsriGrid(fileName, &myMap, myError))
        {
            logError(QString::fromStdString(myError));
            return false;
        }
    }
    return true;
}

bool Project::loadTopographicDistanceMaps(bool onlyWithData, bool showInfo)
{
    if (nrMeteoPoints == 0)
    {
        logError("Open a meteo points DB before.");
        return false;
    }

    QString mapsFolder = projectPath + PATH_TD;
    if (! QDir(mapsFolder).exists())
    {
        QDir().mkdir(mapsFolder);
    }

    int infoStep = 0;
    if (showInfo)
    {
        QString infoStr = "Loading topographic distance maps...";
        infoStep = setProgressBar(infoStr, nrMeteoPoints);
    }

    std::string myError;
    std::string fileName;
    bool isSelected;

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % infoStep) == 0)
                updateProgressBar(i);
        }

        if (meteoPoints[i].active)
        {
            if (! onlyWithData)
                isSelected = true;
            else {
                isSelected = meteoPointsDbHandler->existData(&meteoPoints[i], daily) || meteoPointsDbHandler->existData(&meteoPoints[i], hourly);
            }

            if (isSelected)
            {
                fileName = mapsFolder.toStdString() + "TD_" + QFileInfo(demFileName).baseName().toStdString() + "_" + meteoPoints[i].id;
                if (!QFile::exists(QString::fromStdString(fileName + ".flt")))
                {
                    if (!writeTopographicDistanceMap(i, DEM, mapsFolder))
                    {
                        return false;
                    }
                    if (showInfo) logInfo(QString::fromStdString(fileName) + " successfully created!");
                }
                meteoPoints[i].topographicDistance = new gis::Crit3DRasterGrid();
                if (! gis::readEsriGrid(fileName, meteoPoints[i].topographicDistance, myError))
                {
                    logError(QString::fromStdString(myError));
                    return false;
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

    return true;
}

void Project::passInterpolatedTemperatureToHumidityPoints(Crit3DTime myTime, Crit3DMeteoSettings* meteoSettings)
{
    if (! hourlyMeteoMaps->mapHourlyTair->isLoaded) return;

    float airRelHum, airT;
    int row, col;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        airRelHum = meteoPoints[i].getMeteoPointValue(myTime, airRelHumidity, meteoSettings);
        airT = meteoPoints[i].getMeteoPointValue(myTime, airTemperature, meteoSettings);

        if (! isEqual(airRelHum, NODATA) && isEqual(airT, NODATA))
        {
            hourlyMeteoMaps->mapHourlyTair->getRowCol(meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, row, col);
            if (! gis::isOutOfGridRowCol(row, col, *(hourlyMeteoMaps->mapHourlyTair)))
            {
                meteoPoints[i].setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(),
                                          airTemperature, hourlyMeteoMaps->mapHourlyTair->value[row][col]);
            }
        }
    }
}


bool Project::interpolationOutputPoints(std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                        gis::Crit3DRasterGrid *outputGrid, meteoVariable myVar)
{
    if (! getComputeOnlyPoints()) return false;

    std::vector <float> proxyValues;
    proxyValues.resize(unsigned(interpolationSettings.getProxyNr()));

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            double x = outputPoints[i].utm.x;
            double y = outputPoints[i].utm.y;
            double z = outputPoints[i].z;

            int row, col;
            outputGrid->getRowCol(x, y, row, col);
            if (! gis::isOutOfGridRowCol(row, col, *outputGrid))
            {
                if (getUseDetrendingVar(myVar)) getProxyValuesXY(x, y, &interpolationSettings, proxyValues);
                outputPoints[i].currentValue = interpolate(interpolationPoints, &interpolationSettings,
                                                       meteoSettings, myVar, x, y, z, proxyValues, true);
                outputGrid->value[row][col] = outputPoints[i].currentValue;
            }
        }
    }

    return true;
}

bool Project::computeStatisticsCrossValidation(Crit3DTime myTime, meteoVariable myVar, crossValidationStatistics* myStats)
{
    myStats->initialize();

    std::vector <float> obs;
    std::vector <float> pre;

    float value;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active)
        {
            value = meteoPoints[i].getMeteoPointValue(myTime, myVar, meteoSettings);

            if (! isEqual(value, NODATA) && ! isEqual(meteoPoints[i].residual, NODATA))
            {
                obs.push_back(value);
                pre.push_back(value + meteoPoints[i].residual);
            }
        }
    }

    if (obs.size() > 0)
    {
        myStats->setMeanAbsoluteError(statistics::meanAbsoluteError(obs, pre));
        myStats->setMeanBiasError(statistics::meanError(obs, pre));
        myStats->setRootMeanSquareError(statistics::rootMeanSquareError(obs, pre));
        myStats->setCompoundRelativeError(statistics::compoundRelativeError(obs, pre));

        float intercept, slope, r2;
        statistics::linearRegression(obs, pre, int(obs.size()), false, &intercept, &slope, &r2);
        myStats->setR2(r2);
    }

    return true;
}

bool Project::interpolationCv(meteoVariable myVar, const Crit3DTime& myTime, crossValidationStatistics *myStats)
{
    if (! checkInterpolationMain(myVar)) return false;

    // check variables
    if ( interpolationSettings.getUseDewPoint() &&
        (myVar == dailyAirRelHumidityAvg ||
        myVar == dailyAirRelHumidityMin ||
        myVar == dailyAirRelHumidityMax ||
        myVar == airRelHumidity))
    {
        logError("Cross validation is not available for " + QString::fromStdString(getVariableString(myVar))
                 + "\n Deactive 'Interpolate relative humidity using dew point' option.");
        return false;
    }

    if (myVar == dailyGlobalRadiation ||
        myVar == dailyLeafWetness ||
        myVar == dailyWindVectorDirectionPrevailing ||
        myVar == dailyWindVectorIntensityAvg ||
        myVar == dailyWindVectorIntensityMax ||
        myVar == globalIrradiance)
    {
        logError("Cross validation is not available for " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    // check quality and pass data to interpolation
    if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         &qualityInterpolationSettings, &interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    if (! preInterpolation(interpolationPoints, &interpolationSettings, meteoSettings, &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime))
    {
        logError("Interpolation: error in function preInterpolation");
        return false;
    }

    if (! computeResiduals(myVar, meteoPoints, nrMeteoPoints, interpolationPoints, &interpolationSettings, meteoSettings, true, true))
        return false;

    if (! computeStatisticsCrossValidation(myTime, myVar, myStats))
        return false;

    return true;
}


bool Project::interpolationDem(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    // check quality and pass data to interpolation
    if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         &qualityInterpolationSettings, &interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    // detrending and checking precipitation
    bool interpolationReady = preInterpolation(interpolationPoints, &interpolationSettings, meteoSettings,
                                               &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime);

    if (! interpolationReady)
    {
        logError("Interpolation: error in function preInterpolation");
        return false;
    }

    // interpolate
    bool result;
    if (getComputeOnlyPoints())
    {
        result = interpolationOutputPoints(interpolationPoints, myRaster, myVar);
    }
    else
    {
        result = interpolationRaster(interpolationPoints, &interpolationSettings, meteoSettings, myRaster, DEM, myVar);
    }

    if (!result)
    {
        logError("Interpolation: error in function interpolationRaster");
        return false;
    }

    myRaster->setMapTime(myTime);

    return true;
}


bool Project::interpolationDemDynamicLapserate(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (!getUseDetrendingVar(myVar) || !interpolationSettings.getUseDynamicLapserate())
        return false;

    // pass data to interpolation
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         &qualityInterpolationSettings, &interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }


    // optimal detrending combination
    if (interpolationSettings.getUseBestDetrending())
    {
        std::vector <Crit3DInterpolationDataPoint> interpolationPointsTmp = interpolationPoints;
        optimalDetrending(myVar, meteoPoints, nrMeteoPoints, interpolationPointsTmp, &interpolationSettings, meteoSettings, &climateParameters, myTime);
        interpolationSettings.setCurrentCombination(interpolationSettings.getOptimalCombination());
    }
    else
    {
        interpolationSettings.setCurrentCombination(interpolationSettings.getSelectedCombination());
    }

    std::vector <float> proxyValues;
    proxyValues.resize(unsigned(interpolationSettings.getProxyNr()));
    double x, y;

    if (getComputeOnlyPoints())
    {
        for (unsigned int i = 0; i < outputPoints.size(); i++)
        {
            if (outputPoints[i].active)
            {
                x = outputPoints[i].utm.x;
                y = outputPoints[i].utm.y;
                int row, col;
                myRaster->getRowCol(x, y, row, col);
                if (! myRaster->isOutOfGrid(row, col))
                {
                    std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                    dynamicSelection(interpolationPoints, subsetInterpolationPoints, x, y, interpolationSettings, true);
                    detrending(subsetInterpolationPoints, interpolationSettings.getCurrentCombination(), &interpolationSettings, &climateParameters, myVar, myTime);

                    getProxyValuesXY(x, y, &interpolationSettings, proxyValues);
                    outputPoints[i].currentValue = interpolate(subsetInterpolationPoints, &interpolationSettings, meteoSettings,
                                                               myVar, x, y, outputPoints[i].z, proxyValues, true);

                    myRaster->value[row][col] = outputPoints[i].currentValue;
                }
            }
        }
    }
    else
    {
        gis::Crit3DRasterHeader myHeader = *(DEM.header);
        myRaster->initializeGrid(myHeader);

        for (long row = 0; row < myHeader.nrRows ; row++)
        {
            for (long col = 0; col < myHeader.nrCols; col++)
            {
                float z = DEM.value[row][col];
                if (! isEqual(z, myHeader.flag))
                {
                    gis::getUtmXYFromRowCol(myHeader, row, col, &x, &y);

                    std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                    dynamicSelection(interpolationPoints, subsetInterpolationPoints, x, y, interpolationSettings, true);
                    detrending(subsetInterpolationPoints, interpolationSettings.getCurrentCombination(), &interpolationSettings, &climateParameters, myVar, myTime);

                    getProxyValuesXY(x, y, &interpolationSettings, proxyValues);
                    myRaster->value[row][col] = interpolate(subsetInterpolationPoints, &interpolationSettings, meteoSettings,
                                                            myVar, x, y, z, proxyValues, true);
                }
            }
        }

        if (! gis::updateMinMaxRasterGrid(myRaster))
            return false;
    }

    myRaster->setMapTime(myTime);

    return true;
}


bool Project::interpolateDemRadiation(const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    this->radiationMaps->initialize();

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    radSettings.setGisSettings(&gisSettings);

    gis::Crit3DPoint mapCenter = DEM.getCenter();

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

    bool result;
    result = checkAndPassDataToInterpolation(quality, atmTransmissivity, meteoPoints, nrMeteoPoints,
                                          myTime, &qualityInterpolationSettings, &interpolationSettings,
                                          meteoSettings, &climateParameters, interpolationPoints, checkSpatialQuality);
    if (! result)
    {
        logError("Function interpolateDemRadiation: not enough transmissivity data.");
        return false;
    }

    preInterpolation(interpolationPoints, &interpolationSettings, meteoSettings, &climateParameters,
                     meteoPoints, nrMeteoPoints, atmTransmissivity, myTime);

    // interpolate transmissivity
    if (getComputeOnlyPoints())
    {
        result = interpolationOutputPoints(interpolationPoints, this->radiationMaps->transmissivityMap, atmTransmissivity);
    }
    else
    {
        result = interpolationRaster(interpolationPoints, &interpolationSettings, meteoSettings,
                                     this->radiationMaps->transmissivityMap, DEM, atmTransmissivity);
    }
    if (! result)
    {
        logError("Function interpolateDemRadiation: error interpolating transmissivity.");
        return false;
    }

    // compute radiation
    if (getComputeOnlyPoints())
    {
        result = radiation::computeRadiationOutputPoints(&radSettings, this->DEM, this->radiationMaps, outputPoints, myTime);
    }
    else
    {
        result = radiation::computeRadiationGrid(&radSettings, this->DEM, this->radiationMaps, myTime);
    }
    if (! result)
    {
        logError("Function interpolateDemRadiation: error computing solar radiation");
        return false;
    }

    if (myRaster != this->radiationMaps->globalRadiationMap)
    {
        myRaster->copyGrid(*(this->radiationMaps->globalRadiationMap));
    }

    return true;
}

bool Project::checkInterpolationMain(meteoVariable myVar)
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

    return true;

}

bool Project::checkInterpolationMainSimple(meteoVariable myVar)
{
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

    return true;

}

bool Project::interpolationDemMain(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (! checkInterpolationMain(myVar))
        return false;

    // solar radiation model
    if (myVar == globalIrradiance)
    {
        Crit3DTime halfHour = myTime.addSeconds(-1800);
        return interpolateDemRadiation(halfHour, myRaster);
    }

    // dynamic lapserate
    if (getUseDetrendingVar(myVar) && interpolationSettings.getUseDynamicLapserate())
    {
        return interpolationDemDynamicLapserate(myVar, myTime, myRaster);
    }
    else
    {
        return interpolationDem(myVar, myTime, myRaster);
    }
}


bool Project::meteoGridAggregateProxy(std::vector <gis::Crit3DRasterGrid> &myGrids)
{
    gis::Crit3DRasterGrid* myGrid;
    gis::Crit3DRasterGrid* proxyGrid;

    float cellSize = computeDefaultCellSizeFromMeteoGrid(1);
    gis::Crit3DRasterGrid meteoGridRaster;
    if (! meteoGridDbHandler->MeteoGridToRasterFlt(cellSize, gisSettings, meteoGridRaster))
        return false;

    for (unsigned int i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        myGrid = new gis::Crit3DRasterGrid();

        if (interpolationSettings.getCurrentCombination().getValue(i))
        {
            proxyGrid = interpolationSettings.getProxy(i)->getGrid();
            if (proxyGrid != nullptr && proxyGrid->isLoaded)
                gis::resampleGrid(*proxyGrid, myGrid, meteoGridRaster.header, aggrAverage, 0);

            myGrids.push_back(*myGrid);
        }
    }

    return true;
}

bool Project::interpolationGrid(meteoVariable myVar, const Crit3DTime& myTime)
{
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    // check quality and pass data to interpolation
    if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         &qualityInterpolationSettings, &interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    // detrending and checking precipitation
    bool interpolationReady = preInterpolation(interpolationPoints, &interpolationSettings, meteoSettings,
                                               &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime);

    if (! interpolationReady)
    {
        logError("Interpolation: error in function preInterpolation");
        return false;
    }

    // proxy aggregation
    std::vector <gis::Crit3DRasterGrid> meteoGridProxies;
    if (getUseDetrendingVar(myVar))
        if (! meteoGridAggregateProxy(meteoGridProxies)) return false;

    std::string errString;
    gis::writeEsriGrid("C:\\Users\\gantolini\\Desktop\\tmp\\testDemGrid", &meteoGridProxies[0], errString);

    frequencyType freq = getVarFrequency(myVar);

    float myX, myY, myZ;
    std::vector <float> proxyValues;
    float interpolatedValue = NODATA;
    Crit3DProxyCombination myCombination = interpolationSettings.getCurrentCombination();
    unsigned int i, proxyIndex;
    float proxyValue;

    for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
    {
        for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
        {
            if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active)
            {
                myX = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.x;
                myY = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.y;
                myZ = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.z;

                proxyValues.clear();

                if (getUseDetrendingVar(myVar))
                {
                    proxyIndex = 0;
                    proxyValue = NODATA;
                    for (i=0; i < interpolationSettings.getProxyNr(); i++)
                    {
                        if (myCombination.getValue(i))
                        {
                            if (proxyIndex < meteoGridProxies.size())
                            {
                                proxyValue = gis::getValueFromXY(meteoGridProxies[proxyIndex], myX, myY);
                                proxyValues.push_back(proxyValue);
                            }

                        }
                    }
                }

                interpolatedValue = interpolate(interpolationPoints, &interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true);

                if (freq == hourly)
                {
                    if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->nrObsDataDaysH == 0)
                        meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->initializeObsDataH(1, 1, myTime.date);

                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar, float(interpolatedValue));
                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = float(interpolatedValue);
                }
                else if (freq == daily)
                {
                    if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->nrObsDataDaysD == 0)
                        meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->initializeObsDataD(1, myTime.date);

                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->setMeteoPointValueD(myTime.date, myVar, float(interpolatedValue));
                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = float(interpolatedValue);

                }
            }
        }
    }

    return true;
}

bool Project::interpolationGridMain(meteoVariable myVar, const Crit3DTime& myTime)
{
    if (! checkInterpolationMainSimple(myVar))
        return false;

    /* solar radiation model
    if (myVar == globalIrradiance)
    {
        Crit3DTime halfHour = myTime.addSeconds(-1800);
        return interpolateDemRadiation(halfHour, myRaster);
    }
    */

    // dynamic lapserate
    /*if (getUseDetrendingVar(myVar) && interpolationSettings.getUseDynamicLapserate())
    {
        return interpolationDemDynamicLapserate(myVar, myTime, myRaster);
    }
    else
    {
        return interpolationDem(myVar, myTime, myRaster);
    }
    */


    return interpolationGrid(myVar, myTime);
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
            }
           else if(myPath.left(5) == "$HOME")
            {
                newProjectPath = QDir::homePath() + myPath.right(myPath.length()-5);
            }
            else newProjectPath = myPath;

            newProjectPath = QDir::cleanPath(newProjectPath);

            if (newProjectPath.right(1) != "/") newProjectPath += "/";
            setProjectPath(newProjectPath);
        }

        projectName = projectSettings->value("name").toString();
        demFileName = projectSettings->value("dem").toString();

        dbPointsFileName = projectSettings->value("meteo_points").toString();

        outputPointsFileName = projectSettings->value("output_points").toString();
        dbAggregationFileName = projectSettings->value("aggregation_points").toString();

        dbGridXMLFileName = projectSettings->value("meteo_grid").toString();
        loadGridDataAtStart = projectSettings->value("load_grid_data_at_start").toBool();

    projectSettings->endGroup();

    projectSettings->beginGroup("settings");
        parametersFileName = projectSettings->value("parameters_file").toString();
        logFileName = projectSettings->value("log_file").toString();
        verboseStdoutLogging = projectSettings->value("verbose_stdout_log", "true").toBool();
        currentTileMap = projectSettings->value("tile_map").toString();
    projectSettings->endGroup();
    return true;
}


bool Project::searchDefaultPath(QString* defaultPath)
{
    QString myPath = getApplicationPath();
    QString myRoot = QDir::rootPath();
    // windows: installation on other volume (for example D:)
    QString myVolume = myPath.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(myPath + "/DATA").exists())
        {
            isFound = true;
            break;
        }
        if (QDir::cleanPath(myPath) == myRoot || QDir::cleanPath(myPath) == myVolume)
            break;

        myPath = QFileInfo(myPath).dir().absolutePath();
    }

    if (! isFound)
    {
        logError("DATA directory is missing");
        return false;
    }

    *defaultPath = QDir::cleanPath(myPath) + "/DATA/";
    return true;
}


frequencyType Project::getCurrentFrequency() const
{
    return currentFrequency;
}

void Project::setCurrentFrequency(const frequencyType &value)
{
    currentFrequency = value;
    if (proxyWidget != nullptr)
    {
        proxyWidget->updateFrequency(currentFrequency);
    }
}


void Project::saveProjectLocation()
{
    projectSettings->beginGroup("location");
    projectSettings->setValue("lat", gisSettings.startLocation.latitude);
    projectSettings->setValue("lon", gisSettings.startLocation.longitude);
    projectSettings->setValue("utm_zone", gisSettings.utmZone);
    projectSettings->setValue("time_zone", gisSettings.timeZone);
    projectSettings->setValue("is_utc", gisSettings.isUTC);
    projectSettings->endGroup();

    projectSettings->sync();
}


void Project::saveProjectSettings()
{
    saveProjectLocation();

    projectSettings->beginGroup("project");
        projectSettings->setValue("name", projectName);
        projectSettings->setValue("dem", getRelativePath(demFileName));
        projectSettings->setValue("meteo_points", getRelativePath(dbPointsFileName));
        projectSettings->setValue("output_points", getRelativePath(outputPointsFileName));
        projectSettings->setValue("aggregation_points", getRelativePath(dbAggregationFileName));
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
        parameters->setValue("topographicDistance", interpolationSettings.getUseTD());
        parameters->setValue("dynamicLapserate", interpolationSettings.getUseDynamicLapserate());
        parameters->setValue("topographicDistanceMaxMultiplier", QString::number(interpolationSettings.getTopoDist_maxKh()));
        parameters->setValue("optimalDetrending", interpolationSettings.getUseBestDetrending());
        parameters->setValue("useDewPoint", interpolationSettings.getUseDewPoint());
        parameters->setValue("useInterpolationTemperatureForRH", interpolationSettings.getUseInterpolatedTForRH());
        parameters->setValue("thermalInversion", interpolationSettings.getUseThermalInversion());
        parameters->setValue("minRegressionR2", QString::number(double(interpolationSettings.getMinRegressionR2())));
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
            parameters->setValue("order", i+1);
            parameters->setValue("active", interpolationSettings.getSelectedCombination().getValue(i));
            parameters->setValue("use_for_spatial_quality_control", myProxy->getForQualityControl());
            if (myProxy->getProxyTable() != "") parameters->setValue("table", QString::fromStdString(myProxy->getProxyTable()));
            if (myProxy->getProxyField() != "") parameters->setValue("field", QString::fromStdString(myProxy->getProxyField()));
            if (myProxy->getGridName() != "") parameters->setValue("raster", getRelativePath(QString::fromStdString(myProxy->getGridName())));
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
        parameters->setValue("temperature_threshold", QString::number(meteoSettings->getTemperatureThreshold()));
        parameters->setValue("wind_intensity_default", QString::number(meteoSettings->getWindIntensityDefault()));
        parameters->setValue("hourly_intervals", QString::number(meteoSettings->getHourlyIntervals()));
        parameters->setValue("compute_tavg", meteoSettings->getAutomaticTavg());
        parameters->setValue("compute_et0hs", meteoSettings->getAutomaticET0HS());
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
        errorType = ERROR_SETTINGS;
        errorString = "load parameters failed:\n" + errorString;
        logError();
        return false;
    }

    if (demFileName != "")
        if (! loadDEM(demFileName)) return false;

    if (dbPointsFileName != "")
        if (! loadMeteoPointsDB(dbPointsFileName))
        {
            errorString = "load Meteo Points DB failed";
            errorType = ERROR_DBPOINT;
            logError();
            return false;
        }

    if (dbAggregationFileName != "")
    {
        if (! loadAggregationdDB(projectPath+"/"+dbAggregationFileName))
        {
            errorString = "load Aggregation DB failed";
            errorType = ERROR_DBPOINT;
            logError();
            return false;
        }
        // LC nel caso ci sia solo il dbAggregation ma si vogliano utilizzare le funzioni "nate" per i db point (es. calcolo clima)
        // copio il dbAggregation in dbPointsFileName così può essere trattato allo stesso modo.
        // Utile soprattutto nel caso di chiamate da shell in quanto da GUI è possibile direttamente aprire un db aggregation come db points.
        if (dbPointsFileName.isEmpty())
        {
            if (! loadAggregationDBAsMeteoPoints(dbAggregationFileName))
            {
                logInfo(errorString);
            }
        }
    }

    if (dbGridXMLFileName != "")
        if (! loadMeteoGridDB(dbGridXMLFileName))
        {
            errorString = "load Meteo Grid DB failed";
            errorType = ERROR_DBGRID;
            logError();
            return false;
        }

    if (outputPointsFileName != "")
        if (! loadOutputPointList(outputPointsFileName))
        {
            errorString = "load Output Point List failed";
            errorType = ERROR_OUTPUTPOINTLIST;
            logError();
            return false;
        }

    closeLogInfo();
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
    QList<QString> fileList;

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

void Project::showMeteoWidgetPoint(std::string idMeteoPoint, std::string namePoint, bool isAppend)
{
    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    QDateTime firstHourly = meteoPointsDbHandler->getFirstDate(hourly, idMeteoPoint);
    QDateTime lastHourly = meteoPointsDbHandler->getLastDate(hourly, idMeteoPoint);
    bool hasHourlyData = !(firstHourly.isNull() || lastHourly.isNull());

    if (!hasDailyData && !hasHourlyData)
    {
        logInfoGUI("No data.");
        return;
    }

    // set minimum and maximum dates
    QDate firstDate, lastDate;
    if (hasDailyData)
    {
        firstDate = firstDaily;
        lastDate = lastDaily;
        if (hasHourlyData)
        {
            firstDate = std::min(firstDate, firstHourly.date());
            lastDate = std::max(lastDaily, lastHourly.date());
        }
    }
    else if (hasHourlyData)
    {
        firstDate = firstHourly.date();
        lastDate = lastHourly.date();
    }

    int meteoWidgetId = 0;
    if (meteoWidgetPointList.isEmpty())
    {
        isAppend = false;
    }

    Crit3DMeteoPoint mp;
    mp.setId(idMeteoPoint);
    mp.setName(namePoint);

    if (isAppend)
    {
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), &mp);
        meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), &mp);
        meteoWidgetPointList[meteoWidgetPointList.size()-1]->draw(mp, isAppend);
    }
    else if (!isAppend)
    {
        bool isGrid = false;
        Crit3DMeteoWidget* meteoWidgetPoint = new Crit3DMeteoWidget(isGrid, projectPath, meteoSettings);
        if (!meteoWidgetPointList.isEmpty())
        {
            meteoWidgetId = meteoWidgetPointList[meteoWidgetPointList.size()-1]->getMeteoWidgetID()+1;
        }
        else
        {
            meteoWidgetId = 0;
        }
        meteoWidgetPoint->setMeteoWidgetID(meteoWidgetId);
        meteoWidgetPointList.append(meteoWidgetPoint);
        QObject::connect(meteoWidgetPoint, SIGNAL(closeWidgetPoint(int)), this, SLOT(deleteMeteoWidgetPoint(int)));
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), &mp);
        meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), &mp);

        meteoWidgetPoint->setDateInterval(firstDate, lastDate);
        meteoWidgetPoint->setCurrentDate(this->currentDate);
        meteoWidgetPoint->draw(mp, isAppend);
    }

    closeLogInfo();
}


void Project::showMeteoWidgetGrid(std::string idCell, bool isAppend)
{
    QDate firstDate = meteoGridDbHandler->firstDate();
    QDate lastDate = meteoGridDbHandler->lastDate();

    QDateTime firstDateTime = QDateTime(firstDate, QTime(1,0), Qt::UTC);
    QDateTime lastDateTime = QDateTime(lastDate.addDays(1), QTime(0,0), Qt::UTC);

    int meteoWidgetId = 0;
    if (meteoWidgetGridList.isEmpty() || meteoGridDbHandler->gridStructure().isEnsemble())
    {
        isAppend = false;
    }

    if (meteoGridDbHandler->gridStructure().isEnsemble())
    {
        isAppend = false;
        logInfoGUI("meteo grid is ensemble: append mode is not possible, a new widget is opening");
    }

    if (isAppend)
    {
        logInfoGUI("Loading data...");
        if (!meteoGridDbHandler->gridStructure().isFixedFields())
        {
            meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(idCell), firstDate, lastDate);
            meteoGridDbHandler->loadGridHourlyData(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
        }
        else
        {
            meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(idCell), firstDate, lastDate);
            meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
        }
        closeLogInfo();
        if(meteoWidgetGridList[meteoWidgetGridList.size()-1]->getIsEnsemble())
        {
            // an ensemble grid is already open, append on that
            // The new one is not ensemble (otherwise append mode is not possible)
            meteoWidgetGridList[meteoWidgetGridList.size()-1]->setIsEnsemble(false);
        }
        unsigned row;
        unsigned col;
        if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row,&col,idCell))
        {
            meteoWidgetGridList[meteoWidgetGridList.size()-1]->draw(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col), isAppend);
        }
        return;
    }
    else if (!isAppend)
    {
        bool isGrid = true;
        Crit3DMeteoWidget* meteoWidgetGrid = new Crit3DMeteoWidget(isGrid, projectPath, meteoSettings);
        if (!meteoWidgetGridList.isEmpty())
        {
             meteoWidgetId = meteoWidgetGridList[meteoWidgetGridList.size()-1]->getMeteoWidgetID()+1;
        }
        else
        {
            meteoWidgetId = 0;
        }
        meteoWidgetGrid->setMeteoWidgetID(meteoWidgetId);
        meteoWidgetGrid->setCurrentDate(this->currentDate);
        meteoWidgetGridList.append(meteoWidgetGrid);

        QObject::connect(meteoWidgetGrid, SIGNAL(closeWidgetGrid(int)), this, SLOT(deleteMeteoWidgetGrid(int)));
        logInfoGUI("Loading data...");
        if (meteoGridDbHandler->gridStructure().isEnsemble())
        {
            meteoWidgetGrid->setIsEnsemble(true);
            meteoWidgetGrid->setNrMembers(meteoGridDbHandler->gridStructure().nrMembers());
            unsigned row;
            unsigned col;
            int nMembers = meteoGridDbHandler->gridStructure().nrMembers();
            if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row,&col,idCell))
            {
                meteoWidgetGrid->setDateInterval(firstDate, lastDate);
            }
            else
            {
                closeLogInfo();
                return;
            }
            for (int i = 1; i<=nMembers; i++)
            {
                meteoGridDbHandler->loadGridDailyDataEnsemble(errorString, QString::fromStdString(idCell), i, firstDate, lastDate);
                meteoWidgetGrid->addMeteoPointsEnsemble(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col));
            }
            meteoWidgetGrid->drawEnsemble();
            closeLogInfo();
        }
        else
        {
            if (!meteoGridDbHandler->gridStructure().isFixedFields())
            {
                meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(idCell), firstDate, lastDate);
                meteoGridDbHandler->loadGridHourlyData(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
            }
            else
            {
                meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(idCell), firstDate, lastDate);
                meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
            }
            closeLogInfo();
            unsigned row;
            unsigned col;
            if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row,&col,idCell))
            {
                meteoWidgetGrid->setDateInterval(firstDate, lastDate);
                meteoWidgetGrid->draw(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col), isAppend);
            }
        }
        return;
    }
}

void Project::deleteMeteoWidgetPoint(int id)
{

    for (int i = 0; i<meteoWidgetPointList.size(); i++)
    {
        if (meteoWidgetPointList[i]->getMeteoWidgetID() == id)
        {
            meteoWidgetPointList.removeAt(i);
            break;
        }
    }
}

void Project::deleteMeteoWidgetGrid(int id)
{
    for (int i = 0; i<meteoWidgetGridList.size(); i++)
    {
        if (meteoWidgetGridList[i]->getMeteoWidgetID() == id)
        {
            meteoWidgetGridList.removeAt(i);
            break;
        }
    }
}

void Project::deleteProxyWidget()
{
    proxyWidget = nullptr;
}

bool Project::parseMeteoPointsPropertiesCSV(QString csvFileName, QList<QString>* csvFields)
{
    if (! QFile(csvFileName).exists() || ! QFileInfo(csvFileName).isFile())
    {
        logError("Missing file: " + csvFileName);
        return false;
    }
    importProperties = new ImportPropertiesCSV(csvFileName);

    errorString = "";
    if (!importProperties->parserCSV(&errorString))
    {
        logError(errorString);
        delete importProperties;
        return false;
    }
    *csvFields = importProperties->getHeader();
    return true;
}


bool Project::writeMeteoPointsProperties(QList<QString> joinedList)
{
    QList<QString> header = importProperties->getHeader();
    QList<QList<QString>> dataFields = importProperties->getData();

    QList<QString> column;
    QList<int> posValues;

    for (int i = 0; i<joinedList.size(); i++)
    {
        QList<QString> couple = joinedList[i].split("-->");
        QString pragaProperties = couple[0];
        QString fileProperties = couple[1];
        int pos = header.indexOf(fileProperties);
        if (pos != -1)
        {
            column << pragaProperties;
            posValues << pos;
        }
    }

    QList<QString> values;

    for (int row = 0; row<dataFields.size(); row++)
    {
        values.clear();
        for (int j = 0; j<posValues.size(); j++)
        {
            values << dataFields[row][posValues[j]];
        }
        meteoPointsDbHandler->updatePointProperties(column, values);
    }

    return true;
}

void Project::showProxyGraph()
{
    Crit3DMeteoPoint* meteoPointsSelected;
    int nSelected = 0;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].selected)
        {
            nSelected = nSelected + 1;
        }
    }
    if (nSelected == 0)
    {
        proxyWidget = new Crit3DProxyWidget(&interpolationSettings, meteoPoints, nrMeteoPoints, currentFrequency, currentDate, currentHour, quality, &qualityInterpolationSettings, meteoSettings, &climateParameters, checkSpatialQuality);
    }
    else
    {
        meteoPointsSelected = new Crit3DMeteoPoint[unsigned(nSelected)];
        int posMpSelected = 0;
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            if (meteoPoints[i].selected)
            {
                meteoPointsSelected[posMpSelected] = meteoPoints[i];
                posMpSelected = posMpSelected + 1;
            }
        }
        proxyWidget = new Crit3DProxyWidget(&interpolationSettings, meteoPointsSelected, nSelected, currentFrequency, currentDate, currentHour, quality, &qualityInterpolationSettings, meteoSettings, &climateParameters, checkSpatialQuality);
    }
    QObject::connect(proxyWidget, SIGNAL(closeProxyWidget()), this, SLOT(deleteProxyWidget()));
    return;
}


void Project::clearSelectedPoints()
{
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        meteoPoints[i].selected = false;
    }
}


void Project::clearSelectedOutputPoints()
{
    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        outputPoints[i].selected = false;
    }
}


bool Project::setActiveStateSelectedPoints(bool isActive)
{
    if (meteoPointsDbHandler == nullptr)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    QList<QString> selectedPointList;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].selected)
        {
            meteoPoints[i].active = isActive;
            selectedPointList << QString::fromStdString(meteoPoints[i].id);
        }
    }

    if (selectedPointList.isEmpty())
    {
        logError("No meteo points selected.");
        return false;
    }

    if (!meteoPointsDbHandler->setActiveStatePointList(selectedPointList, isActive))
    {
        logError("Failed to activate/deactivate selected points:\n" + meteoPointsDbHandler->error);
        return false;
    }

    clearSelectedPoints();
    return true;
}


bool Project::setActiveStatePointList(QString fileName, bool isActive)
{
    QList<QString> pointList = readListSingleColumn(fileName, errorString);
    if (pointList.size() == 0)
    {
        logError();
        return false;
    }

    if (!meteoPointsDbHandler->setActiveStatePointList(pointList, isActive))
    {
        logError("Failed to activate/deactivate point list:\n" + meteoPointsDbHandler->error);
        return false;
    }

    for (int j = 0; j < pointList.size(); j++)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            if (meteoPoints[i].id == pointList[j].toStdString())
            {
                meteoPoints[i].active = isActive;
            }
        }
    }

    return true;
}


bool Project::setActiveStateWithCriteria(bool isActive)
{
    if (meteoPointsDbHandler == nullptr)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    DialogSelectionMeteoPoint dialogPointSelection(isActive, meteoPointsDbHandler);
    if (dialogPointSelection.result() != QDialog::Accepted)
        return false;

    QString selection = dialogPointSelection.getSelection();
    QString operation = dialogPointSelection.getOperation();
    QString item = dialogPointSelection.getItem();
    QString condition;
    if (operation != "Like")
    {
        condition = selection + " " + operation + " '" +item +"'";
    }
    else
    {
        condition = selection + " " + operation + " '%" +item +"%'";
    }
    if (selection != "DEM distance [m]")
    {
        meteoPointsDbHandler->setActiveStateIfCondition(isActive, condition);
    }
    else
    {
        if (!DEM.isLoaded)
        {
            logError("No DEM open");
            return false;
        }
        QList<QString> points;
        setProgressBar("Checking distance...", nrMeteoPoints);
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            updateProgressBar(i);
            if (!meteoPoints[i].active)
            {
                float distance = gis::closestDistanceFromGrid(meteoPoints[i].point, DEM);
                if (operation == "=")
                {
                    if (isEqual(distance, item.toFloat()))
                    {
                        points.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == "!=")
                {
                    if (! isEqual(distance, item.toFloat()))
                    {
                        points.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == ">")
                {
                    if (distance > item.toFloat())
                    {
                        points.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == "<")
                {
                    if (distance < item.toFloat())
                    {
                        points.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
            }
        }
        closeProgressBar();

        if (points.isEmpty())
        {
            logError("No points fit your requirements.");
            return false;
        }
        if (!meteoPointsDbHandler->setActiveStatePointList(points, isActive))
        {
            logError("Failed to activate/deactivate points selected:\n" + meteoPointsDbHandler->error);
            return false;
        }
    }

    return true;
}


bool Project::setMarkedFromPointList(QString fileName)
{
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        meteoPoints[i].marked = false;
    }

    QList<QString> pointList = readListSingleColumn(fileName, errorString);
    if (pointList.size() == 0)
    {
        logError();
        return false;
    }

    for (int j = 0; j < pointList.size(); j++)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            if (meteoPoints[i].id == pointList[j].toStdString())
            {
                meteoPoints[i].marked = true;
            }
        }
    }

    return true;
}


bool Project::deleteMeteoPoints(const QList<QString>& pointList)
{
    logInfoGUI("Deleting points...");
        bool isOk = meteoPointsDbHandler->deleteAllPointsFromIdList(pointList);
    closeLogInfo();

    if (!isOk)
    {
        logError("Failed to delete points:" + meteoPointsDbHandler->error);
        return false;
    }

    if (meteoPointsDbHandler->error != "")
    {
        logError("WARNING: " + meteoPointsDbHandler->error);
    }

    // reload meteoPoint, point properties table is changed
    QString dbName = dbPointsFileName;
    closeMeteoPointsDB();

    return loadMeteoPointsDB(dbName);
}


bool Project::deleteMeteoPointsData(const QList<QString>& pointList)
{
    if (pointList.isEmpty())
    {
        logError("No data to delete.");
        return true;
    }

    DialogPointDeleteData dialogPointDelete(currentDate);
    if (dialogPointDelete.result() != QDialog::Accepted)
        return true;

    QList<meteoVariable> dailyVarList = dialogPointDelete.getVarD();
    QList<meteoVariable> hourlyVarList = dialogPointDelete.getVarH();
    QDate startDate = dialogPointDelete.getFirstDate();
    QDate endDate = dialogPointDelete.getLastDate();
    bool allDaily = dialogPointDelete.getAllDailyVar();
    bool allHourly = dialogPointDelete.getAllHourlyVar();

    QString question = "Data of " + QString::number(pointList.size()) + " points\n";
    question += QString::number(dailyVarList.size()) + " daily vars\n";
    question += QString::number(hourlyVarList.size()) + " hourly vars\n";
    question += "from " + startDate.toString() + " to " + endDate.toString() + "\n";
    question += "will be deleted. Are you sure?";
    if (QMessageBox::question(nullptr, "Question", question) != QMessageBox::Yes)
        return true;

    setProgressBar("Deleting data...", pointList.size());
    for (int i = 0; i < pointList.size(); i++)
    {
        updateProgressBar(i);

        if (allDaily)
        {
            if (!meteoPointsDbHandler->deleteData(pointList[i], daily, startDate, endDate))
            {
                closeProgressBar();
                return false;
            }
        }
        else
        {
            if (!dailyVarList.isEmpty())
            {
                if (!meteoPointsDbHandler->deleteData(pointList[i], daily, dailyVarList, startDate, endDate))
                {
                    closeProgressBar();
                    return false;
                }
            }
        }
        if (allHourly)
        {
            if (!meteoPointsDbHandler->deleteData(pointList[i], hourly, startDate, endDate))
            {
                closeProgressBar();
                return false;
            }
        }
        else
        {
            if (!hourlyVarList.isEmpty())
            {
                if (!meteoPointsDbHandler->deleteData(pointList[i], hourly, hourlyVarList, startDate, endDate))
                {
                    closeProgressBar();
                    return false;
                }
            }
        }
    }
    closeProgressBar();

    return true;
}

bool Project::loadOutputPointList(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing csv filename");
        return false;
    }
    outputPoints.clear();
    outputPointsFileName = fileName;

    QString csvFileName = getCompleteFileName(fileName, PATH_OUTPUT);
    if (! QFile(csvFileName).exists() || ! QFileInfo(csvFileName).isFile())
    {
        logError("Missing csv file: " + csvFileName);
        return false;
    }

    if (!loadOutputPointListCsv(csvFileName, outputPoints, gisSettings.utmZone, errorString))
    {
        logError("Error importing output list: " + errorString);
        errorString.clear();
        return false;
    }

    return true;
}

bool Project::writeOutputPointList(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing csv filename");
        return false;
    }

    QString csvFileName = getCompleteFileName(fileName, PATH_OUTPUT);
    if (! QFile(csvFileName).exists() || ! QFileInfo(csvFileName).isFile())
    {
        logError("Missing csv file: " + csvFileName);
        return false;
    }

    if (!writeOutputPointListCsv(csvFileName, outputPoints, errorString))
    {
        logError("Error writing output list to csv: " + errorString);
        errorString.clear();
        return false;
    }

    return true;
}

void Project::setComputeOnlyPoints(bool isOnlyPoints)
{
    computeOnlyPoints = isOnlyPoints;
}

bool Project::getComputeOnlyPoints()
{
    return computeOnlyPoints;
}

bool Project::exportMeteoGridToRasterFlt(QString fileName, double cellSize)
{
    if (fileName != "")
    {
        gis::Crit3DRasterGrid myGrid;
        if (!meteoGridDbHandler->MeteoGridToRasterFlt(cellSize, gisSettings, myGrid))
        {
            errorString = "initializeGrid failed";
            return false;
        }

        std::string myError = errorString.toStdString();
        QString fileWithoutExtension = QFileInfo(fileName).absolutePath() + QDir::separator() + QFileInfo(fileName).baseName();
        if (!gis::writeEsriGrid(fileWithoutExtension.toStdString(), &myGrid, myError))
        {
            errorString = QString::fromStdString(myError);
            return false;
        }
        return true;

    }
    return false;
}


bool Project::exportMeteoGridToCsv(QString fileName)
{
    if (fileName == "")
        return false;

    QFile myFile(fileName);
    if (!myFile.open(QIODevice::WriteOnly | QFile::Truncate))
    {
        logError("Open CSV failed: " + fileName);
        return false;
    }

    QTextStream out(&myFile);

    for (int row = 0; row < meteoGridDbHandler->meteoGrid()->dataMeteoGrid.header->nrRows; row++)
    {
        for (int col = 0; col < meteoGridDbHandler->meteoGrid()->dataMeteoGrid.header->nrCols; col++)
        {
            float value = meteoGridDbHandler->meteoGrid()->dataMeteoGrid.value[row][col];
            std::string id = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->id;
            std::string name = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->name;

            if (value != NO_ACTIVE && value != NODATA)
            {
                out << QString::fromStdString(id + ',' + name + ',') + QString::number(value) + "\n";
            }
        }
    }
    myFile.close();

    return true;
}


int Project::computeDefaultCellSizeFromMeteoGrid(float resolutionRatio)
{
    if (meteoGridDbHandler->gridStructure().isUTM())
    {
        return meteoGridDbHandler->meteoGrid()->dataMeteoGrid.header->cellSize;
    }

    // lat lon grid
    gis::Crit3DLatLonHeader latlonHeader = meteoGridDbHandler->gridStructure().header();
    int cellSize = gis::getGeoCellSizeFromLatLonHeader(gisSettings, &latlonHeader);

    cellSize *= resolutionRatio;
    // round cellSize
    int nTimes = int(floor(log10(cellSize)));
    int roundValue = int(round(cellSize / pow(10, nTimes)));
    cellSize = roundValue * int(pow(10, nTimes));

    return cellSize;
}


/* ---------------------------------------------
 * LOG functions
 * --------------------------------------------*/

bool Project::setLogFile(QString myFileName)
{
    QString fileNameWithPath = getCompleteFileName(myFileName, PATH_LOG);

    QString logFilePath = getFilePath(fileNameWithPath);
    QString endLogFileName = getFileName(fileNameWithPath);

    if (!QDir(logFilePath).exists())
    {
         QDir().mkdir(logFilePath);
    }

    // remove previous log files (older than 7 days)
    removeOldFiles(logFilePath, endLogFileName, 7);

    QDate myQDate = QDateTime().currentDateTime().date();
    QTime myQTime = QDateTime().currentDateTime().time();
    QString myDate = QDateTime(myQDate, myQTime, Qt::UTC).currentDateTime().toString("yyyyMMdd_HHmm");

    logFileName = logFilePath + myDate + "_" + endLogFileName;

    logFile.open(logFileName.toStdString().c_str());
    if (logFile.is_open())
    {
        logInfo("LogFile = " + logFileName);
        return true;
    }
    else
    {
        logError("Unable to open log file: " + logFileName);
        return false;
    }
}


void Project::logInfo(QString myStr)
{
    if (verboseStdoutLogging) {
        // standard output in all modalities
        std::cout << myStr.toStdString() << std::endl;
    }

    if (logFile.is_open())
    {
        logFile << myStr.toStdString() << std::endl;
    }
}


void Project::logInfoGUI(QString myStr)
{
    if (modality == MODE_GUI)
    {
        if (formLog == nullptr)
        {
            formLog = new FormInfo();
        }
        formLog->showInfo(myStr);
    }
    else
    {
        logInfo(myStr);
    }
}


void Project::closeLogInfo()
{
    if ((modality == MODE_GUI) && (formLog != nullptr))
    {
        formLog->close();
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


/* ---------------------------------------------
 * Progress bar
 * --------------------------------------------*/

int Project::setProgressBar(QString myStr, int nrValues)
{
    if (modality == MODE_GUI)
    {
        if (formLog == nullptr)
        {
            formLog = new FormInfo();
        }
        return formLog->start(myStr, nrValues);
    }
    else
    {
        std::cout << myStr.toStdString() << std::endl;
        return std::max(1, int(nrValues / 50));
    }
}


void Project::updateProgressBar(int value)
{
    if (modality == MODE_GUI)
    {
        if (formLog != nullptr)
        {
            formLog->setValue(value);
        }
    }
    else
    {
        std::cout << "*";
    }
}


void Project::updateProgressBarText(QString myStr)
{
    if (modality == MODE_GUI)
    {
        if (formLog != nullptr)
        {
            formLog->setText(myStr);
        }
    }
    else
    {
        std::cout << myStr.toStdString();
    }
}


void Project::closeProgressBar()
{
    if (modality == MODE_GUI)
    {
        if (formLog != nullptr)
        {
            formLog->close();
        }
    }
    else
    {
        std::cout << std::endl;
    }
}

