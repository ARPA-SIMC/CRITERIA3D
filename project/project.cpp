#include "project.h"
#include "dbArkimet.h"
#include "dbMeteoGrid.h"
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
#include "importData.h"
#include "dialogSummary.h"
#include "waterTableWidget.h"
#include "utilities.h"


#include <iostream>
#include <QDir>
#include <QFile>
#include <QSqlQuery>
#include <QMessageBox>
#include <string>
#include <omp.h>


Project::Project()
{
    // TODO remove pointers
    meteoSettings = new Crit3DMeteoSettings();
    quality = new Crit3DQuality();
    meteoPointsColorScale = new Crit3DColorScale();
    meteoGridDbHandler = nullptr;

    _appPath = "";
    _defaultPath = "";
    _computeOnlyPoints = false;
    _verboseStdoutLogging = true;

    _formLog = nullptr;

    initializeProject();

    modality = MODE_GUI;
}


void Project::initializeProject()
{
    _projectPath = "";
    _projectName = "";
    _isProjectLoaded = false;
    _isRequestedExit = false;
    _isParallelComputing = true;
    logFileName = "";
    errorString = "";
    errorType = ERROR_NONE;
    _currentTileMap = "";

    _currentVariable = noMeteoVar;
    _currentFrequency = noFrequency;
    _currentDate.setDate(1800,1,1);
    _currentHour = 12;

    nrMeteoPoints = 0;
    meteoPoints = nullptr;
    meteoPointsDbHandler = nullptr;
    outputPointsDbHandler = nullptr;
    meteoGridDbHandler = nullptr;
    aggregationDbHandler = nullptr;

    meteoPointsDbFirstTime.setTimeZone(QTimeZone::utc());
    meteoPointsDbLastTime.setTimeZone(QTimeZone::utc());
    meteoPointsDbFirstTime.setSecsSinceEpoch(0);
    meteoPointsDbLastTime.setSecsSinceEpoch(0);

    meteoSettings->initialize();
    quality->initialize();
    checkSpatialQuality = true;

    parametersSettings = nullptr;
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
	glocalMapName = "";
    glocalPointsName = "";
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

    delete parametersSettings;
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

    _isProjectLoaded = false;
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
        {
            proxyHeight->setGrid(&DEM);
        }
    }

    if (indexQuality != NODATA)
    {
        proxyHeight = qualityInterpolationSettings.getProxy(unsigned(indexQuality));
        if (proxyHeight->getGridName() == "" && DEM.isLoaded)
            proxyHeight->setGrid(&DEM);
    }
}


bool Project::checkProxy(Crit3DProxy &myProxy, QString* error, bool isActive)
{
    std::string name_ = myProxy.getName();

    if (name_ == "")
    {
        *error = "no name";
        return false;
    }

    bool isHeight = (getProxyPragaName(name_) == proxyHeight);

    if (!isHeight && (myProxy.getGridName() == "") && (myProxy.getProxyTable() == "" && myProxy.getProxyField() == ""))
    {
        *error = "error reading grid, table or field for proxy " + QString::fromStdString(name_);
        return false;
    }

    if (interpolationSettings.getUseMultipleDetrending() && isActive)
        checkProxyForMultipleDetrending(myProxy, isHeight);

    return true;
}

void Project::checkProxyForMultipleDetrending(Crit3DProxy &myProxy, bool isHeight)
{
    const double H0_MIN = -350; //height of single inversion point (double piecewise) or first inversion point (triple piecewise)
    const double H0_MAX = 2500;
    const double DELTA_MIN = 300; //height difference between inversion points (for triple piecewise only)
    const double DELTA_MAX = 1000;
    const double SLOPE_MIN = 0.002; //ascending slope
    const double SLOPE_MAX = 0.007;
    const double INVSLOPE_MIN = -0.01; //inversion slope
    const double INVSLOPE_MAX = -0.0015;

    int nrParameters = 0;
    if (isHeight)
    {
        if (myProxy.getFittingFunctionName() == piecewiseTwo)
            nrParameters = 4;
        else if (myProxy.getFittingFunctionName() == piecewiseThree)
            nrParameters = 5;
        else if (myProxy.getFittingFunctionName() == piecewiseThreeFree)
            nrParameters = 6;
    }
    else
        nrParameters = 2;

    if (int(myProxy.getFittingParametersRange().size()) != nrParameters*2)
    {
        logError("wrong number of parameters for proxy: " + QString::fromStdString(myProxy.getName()) + ". You can change them in the parameters.ini file, but default parameters will be used for now.");
        if (nrParameters == 2)
            myProxy.setFittingParametersRange({-10, -50, 11, 50});
        else if (nrParameters == 4)
            myProxy.setFittingParametersRange({0, -30, SLOPE_MIN, INVSLOPE_MIN,
                                               H0_MAX, 50, SLOPE_MAX, INVSLOPE_MAX});
        else if (nrParameters == 5)
            myProxy.setFittingParametersRange({H0_MIN, -30, DELTA_MIN, SLOPE_MIN, INVSLOPE_MIN,
                                               H0_MAX, 50, DELTA_MAX, SLOPE_MAX, INVSLOPE_MAX});
        else if (nrParameters == 6)
            myProxy.setFittingParametersRange({H0_MIN, -30, DELTA_MIN, SLOPE_MIN, INVSLOPE_MIN, INVSLOPE_MIN,
                                               H0_MAX, 50, DELTA_MAX, SLOPE_MAX, INVSLOPE_MAX, INVSLOPE_MAX});
    }

    if (isHeight && int(myProxy.getFittingFirstGuess().size()) != nrParameters)
    {
        logError("wrong number of first guess settings for proxy: " + QString::fromStdString(myProxy.getName()) + ". You can change them in the parameters.ini file, but default settings will be used for now.");
        if (nrParameters == 4)
            myProxy.setFittingFirstGuess({0, 1, 1, 1});
        else if (nrParameters == 5)
            myProxy.setFittingFirstGuess({0, 1, 1, 1, 1});
        else if (nrParameters == 6)
            myProxy.setFittingFirstGuess({0, 1, 1, 1, 1, 1});
    }

    return;
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
        if (getProxyPragaName(interpolationSettings.getProxy(i)->getName()) == proxyHeight) setProxyDEM();

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
        errorString = "Missing file: " + parametersFileName;
        return false;
    }

    delete parametersSettings;
    parametersSettings = new QSettings(parametersFileName, QSettings::IniFormat);

    //interpolation settings
    interpolationSettings.initialize();
    qualityInterpolationSettings.initialize();

    std::vector <Crit3DProxy> proxyListTmp;
    std::deque <bool> proxyActiveTmp;
    std::vector <int> proxyOrder;

    QList<QString> myList;
    std::vector <QString> proxyGridSeriesNames;
    std::vector <unsigned> proxyGridSeriesYears;

    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        //meteo settings
        if (group == "meteo")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("min_percentage") && !parametersSettings->value("min_percentage").toString().isEmpty())
            {
                meteoSettings->setMinimumPercentage(parametersSettings->value("min_percentage").toFloat());
            }
            if (parametersSettings->contains("prec_threshold") && !parametersSettings->value("prec_threshold").toString().isEmpty())
            {
                meteoSettings->setRainfallThreshold(parametersSettings->value("prec_threshold").toFloat());
            }
            if (parametersSettings->contains("thom_threshold") && !parametersSettings->value("thom_threshold").toString().isEmpty())
            {
                meteoSettings->setThomThreshold(parametersSettings->value("thom_threshold").toFloat());
            }
            if (parametersSettings->contains("temperature_threshold") && !parametersSettings->value("temperature_threshold").toString().isEmpty())
            {
                meteoSettings->setTemperatureThreshold(parametersSettings->value("temperature_threshold").toFloat());
            }
            if (parametersSettings->contains("samani_coefficient") && !parametersSettings->value("samani_coefficient").toString().isEmpty())
            {
                meteoSettings->setTransSamaniCoefficient(parametersSettings->value("samani_coefficient").toFloat());
            }
            if (parametersSettings->contains("hourly_intervals") && !parametersSettings->value("hourly_intervals").toString().isEmpty())
            {
                meteoSettings->setHourlyIntervals(parametersSettings->value("hourly_intervals").toInt());
            }
            if (parametersSettings->contains("wind_intensity_default") && !parametersSettings->value("wind_intensity_default").toString().isEmpty())
            {
                meteoSettings->setWindIntensityDefault(parametersSettings->value("wind_intensity_default").toInt());
            }
            if (parametersSettings->contains("compute_tavg") && !parametersSettings->value("compute_tavg").toString().isEmpty())
            {
                meteoSettings->setAutomaticTavg(parametersSettings->value("compute_tavg").toBool());
            }
            if (parametersSettings->contains("compute_et0hs") && !parametersSettings->value("compute_et0hs").toString().isEmpty())
            {
                meteoSettings->setAutomaticET0HS(parametersSettings->value("compute_et0hs").toBool());
            }

            parametersSettings->endGroup();
        }

        if (group == "climate")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("tmin"))
            {
                myList = parametersSettings->value("tmin").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmin values";
                    return  false;
                }

                climateParameters.tmin = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tmax"))
            {
                myList = parametersSettings->value("tmax").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmax values";
                    return  false;
                }

                climateParameters.tmax = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tmin_lapserate"))
            {
                myList = parametersSettings->value("tmin_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmin lapse rate values";
                    return  false;
                }

                climateParameters.tminLapseRate = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tmax_lapserate"))
            {
                myList = parametersSettings->value("tmax_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tmax lapse rate values";
                    return  false;
                }

                climateParameters.tmaxLapseRate = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tdmin"))
            {
                myList = parametersSettings->value("tdmin").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmin values";
                    return  false;
                }

                climateParameters.tdmin = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tdmax"))
            {
                myList = parametersSettings->value("tdmax").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmax values";
                    return  false;
                }

                climateParameters.tdmax = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tdmin_lapserate"))
            {
                myList = parametersSettings->value("tdmin_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmin lapse rate values";
                    return  false;
                }

                climateParameters.tdMinLapseRate = StringListToFloat(myList);
            }

            if (parametersSettings->contains("tdmax_lapserate"))
            {
                myList = parametersSettings->value("tdmax_lapserate").toStringList();
                if (myList.size() < 12)
                {
                    errorString = "Incomplete climate monthly tdmax lapse rate values";
                    return  false;
                }

                climateParameters.tdMaxLapseRate = StringListToFloat(myList);
            }

            parametersSettings->endGroup();
        }

        if (group == "radiation")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("algorithm"))
            {
                std::string algorithm = parametersSettings->value("algorithm").toString().toStdString();
                if (radAlgorithmToString.find(algorithm) == radAlgorithmToString.end())
                {
                    errorString = "Unknown radiation algorithm: " + QString::fromStdString(algorithm);
                    return false;
                }
                else
                    radSettings.setAlgorithm(radAlgorithmToString.at(algorithm));
            }

            if (parametersSettings->contains("real_sky_algorithm"))
            {
                std::string realSkyAlgorithm = parametersSettings->value("real_sky_algorithm").toString().toStdString();
                if (realSkyAlgorithmToString.find(realSkyAlgorithm) == realSkyAlgorithmToString.end())
                {
                    errorString = "Unknown radiation real sky algorithm: " + QString::fromStdString(realSkyAlgorithm);
                    return false;
                }
                else
                    radSettings.setRealSkyAlgorithm(realSkyAlgorithmToString.at(realSkyAlgorithm));
            }

            if (parametersSettings->contains("linke_mode"))
            {
                std::string linkeMode = parametersSettings->value("linke_mode").toString().toStdString();
                if (paramModeToString.find(linkeMode) == paramModeToString.end())
                {
                    errorString = "Unknown Linke mode: " + QString::fromStdString(linkeMode);
                    return false;
                }
                else
                    radSettings.setLinkeMode(paramModeToString.at(linkeMode));
            }

            if (parametersSettings->contains("albedo_mode"))
            {
                std::string albedoMode = parametersSettings->value("albedo_mode").toString().toStdString();
                if (paramModeToString.find(albedoMode) == paramModeToString.end())
                {
                    errorString = "Unknown albedo mode: " + QString::fromStdString(albedoMode);
                    return false;
                }
                else
                    radSettings.setAlbedoMode(paramModeToString.at(albedoMode));
            }

            if (parametersSettings->contains("tilt_mode"))
            {
                std::string tiltMode = parametersSettings->value("tilt_mode").toString().toStdString();
                if (tiltModeToString.find(tiltMode) == tiltModeToString.end())
                {
                    errorString = "Unknown albedo mode: " + QString::fromStdString(tiltMode);
                    return false;
                }
                else
                    radSettings.setTiltMode(tiltModeToString.at(tiltMode));
            }

            if (parametersSettings->contains("real_sky"))
                radSettings.setRealSky(parametersSettings->value("real_sky").toBool());

            if (parametersSettings->contains("shadowing"))
                radSettings.setShadowing(parametersSettings->value("shadowing").toBool());

            if (parametersSettings->contains("linke"))
                radSettings.setLinke(parametersSettings->value("linke").toFloat());

            if (parametersSettings->contains("albedo"))
                radSettings.setAlbedo(parametersSettings->value("albedo").toFloat());

            if (parametersSettings->contains("tilt"))
                radSettings.setTilt(parametersSettings->value("tilt").toFloat());

            if (parametersSettings->contains("aspect"))
                radSettings.setAspect(parametersSettings->value("aspect").toFloat());

            if (parametersSettings->contains("clear_sky"))
                radSettings.setClearSky(parametersSettings->value("clear_sky").toFloat());

            if (parametersSettings->contains("linke_map"))
                radSettings.setLinkeMapName(parametersSettings->value("linke_map").toString().toStdString());

            if (parametersSettings->contains("albedo_map"))
                radSettings.setAlbedoMapName(parametersSettings->value("albedo_map").toString().toStdString());

            if (parametersSettings->contains("land_use") && !parametersSettings->contains("linke_monthly"))
            {
                std::string landUse = parametersSettings->value("land_use").toString().toStdString();
                if (landUseToString.find(landUse) == landUseToString.end())
                {
                    errorString = "Unknown land use: " + QString::fromStdString(landUse);
                    return false;
                }
                else
                    radSettings.setLandUse(landUseToString.at(landUse));
            }
            else if (parametersSettings->contains("linke_monthly"))
            {
                QList<QString> myLinkeStr = parametersSettings->value("linke_monthly").toStringList();
                if (myLinkeStr.size() < 12)
                {
                    errorString = "Incomplete monthly Linke values";
                    return  false;
                }

                radSettings.setLinkeMonthly(StringListToFloat(myLinkeStr));
            }            

            if (parametersSettings->contains("albedo_monthly"))
            {
                QList<QString> myAlbedoStr = parametersSettings->value("albedo_monthly").toStringList();
                if (myAlbedoStr.size() < 12)
                {
                    errorString = "Incomplete monthly albedo values";
                    return  false;
                }

                radSettings.setAlbedoMonthly(StringListToFloat(myAlbedoStr));
            }

            parametersSettings->endGroup();
        }

        //interpolation
        if (group == "interpolation")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("algorithm"))
            {
                std::string algorithm = parametersSettings->value("algorithm").toString().toStdString();
                if (interpolationMethodNames.find(algorithm) == interpolationMethodNames.end())
                {
                    errorString = "Unknown interpolation method";
                    return false;
                }
                else
                    interpolationSettings.setInterpolationMethod(interpolationMethodNames.at(algorithm));
            }

            if (parametersSettings->contains("aggregationMethod"))
            {
                std::string aggrMethod = parametersSettings->value("aggregationMethod").toString().toStdString();
                if (aggregationMethodToString.find(aggrMethod) == aggregationMethodToString.end())
                {
                    errorString = "Unknown aggregation method";
                    return false;
                }
                else
                    interpolationSettings.setMeteoGridAggrMethod(aggregationMethodToString.at(aggrMethod));
            }

            if (parametersSettings->contains("thermalInversion"))
            {
                interpolationSettings.setUseThermalInversion(parametersSettings->value("thermalInversion").toBool());
                qualityInterpolationSettings.setUseThermalInversion(parametersSettings->value("thermalInversion").toBool());
            }

            if (parametersSettings->contains("excludeStationsOutsideDEM"))
                interpolationSettings.setUseExcludeStationsOutsideDEM(parametersSettings->value("excludeStationsOutsideDEM").toBool());

            if (parametersSettings->contains("topographicDistance"))
                interpolationSettings.setUseTD(parametersSettings->value("topographicDistance").toBool());

            if (parametersSettings->contains("optimalDetrending"))
                interpolationSettings.setUseBestDetrending(parametersSettings->value("optimalDetrending").toBool());

            if (parametersSettings->contains("multipleDetrending"))
                interpolationSettings.setUseMultipleDetrending(parametersSettings->value("multipleDetrending").toBool());

            if (parametersSettings->contains("localDetrending"))
                interpolationSettings.setUseLocalDetrending(parametersSettings->value("localDetrending").toBool());
			
			if (parametersSettings->contains("glocalDetrending"))
                interpolationSettings.setUseGlocalDetrending(parametersSettings->value("glocalDetrending").toBool());

            if (interpolationSettings.getUseMultipleDetrending() && interpolationSettings.getUseBestDetrending())
            {
                interpolationSettings.setUseMultipleDetrending(false);
                interpolationSettings.setUseLocalDetrending(false);
                interpolationSettings.setUseGlocalDetrending(false);
            }

            if (parametersSettings->contains("glocalMapName"))
                glocalMapName = parametersSettings->value("glocalMapName").toString();

            if (parametersSettings->contains("glocalPointsName"))
                glocalPointsName = parametersSettings->value("glocalPointsName").toString();

            if (parametersSettings->contains("meteogrid_upscalefromdem"))
                interpolationSettings.setMeteoGridUpscaleFromDem(parametersSettings->value("meteogrid_upscalefromdem").toBool());

            if (parametersSettings->contains("lapseRateCode"))
            {
                interpolationSettings.setUseLapseRateCode(parametersSettings->value("lapseRateCode").toBool());
                qualityInterpolationSettings.setUseLapseRateCode(parametersSettings->value("lapseRateCode").toBool());
            }

            if (parametersSettings->contains("minRegressionR2"))
            {
                interpolationSettings.setMinRegressionR2(parametersSettings->value("minRegressionR2").toFloat());
                qualityInterpolationSettings.setMinRegressionR2(parametersSettings->value("minRegressionR2").toFloat());
            }

            if (parametersSettings->contains("min_points_local_detrending"))
                interpolationSettings.setMinPointsLocalDetrending(parametersSettings->value("min_points_local_detrending").toInt());

            if (parametersSettings->contains("topographicDistanceMaxMultiplier"))
            {
                interpolationSettings.setTopoDist_maxKh(parametersSettings->value("topographicDistanceMaxMultiplier").toInt());
                qualityInterpolationSettings.setTopoDist_maxKh(parametersSettings->value("topographicDistanceMaxMultiplier").toInt());
            }

            if (parametersSettings->contains("useDewPoint"))
                interpolationSettings.setUseDewPoint(parametersSettings->value("useDewPoint").toBool());

            if (parametersSettings->contains("useInterpolationTemperatureForRH"))
                interpolationSettings.setUseInterpolatedTForRH(parametersSettings->value("useInterpolationTemperatureForRH").toBool());

            if (parametersSettings->contains("doNotRetrend"))
                interpolationSettings.setUseDoNotRetrend(parametersSettings->value("doNotRetrend").toBool());

            if (parametersSettings->contains("retrendOnly"))
                interpolationSettings.setUseRetrendOnly(parametersSettings->value("retrendOnly").toBool());

            parametersSettings->endGroup();

        }

        if (group == "quality")
        {
            parametersSettings->beginGroup(group);
            if (parametersSettings->contains("reference_height") && !parametersSettings->value("reference_height").toString().isEmpty())
            {
                quality->setReferenceHeight(parametersSettings->value("reference_height").toFloat());
            }
            if (parametersSettings->contains("delta_temperature_suspect") && !parametersSettings->value("delta_temperature_suspect").toString().isEmpty())
            {
                quality->setDeltaTSuspect(parametersSettings->value("delta_temperature_suspect").toFloat());
            }
            if (parametersSettings->contains("delta_temperature_wrong") && !parametersSettings->value("delta_temperature_wrong").toString().isEmpty())
            {
                quality->setDeltaTWrong(parametersSettings->value("delta_temperature_wrong").toFloat());
            }
            if (parametersSettings->contains("relhum_tolerance") && !parametersSettings->value("relhum_tolerance").toString().isEmpty())
            {
                quality->setRelHumTolerance(parametersSettings->value("relhum_tolerance").toFloat());
            }
            if (parametersSettings->contains("water_table_maximum_depth") && !parametersSettings->value("water_table_maximum_depth").toString().isEmpty())
            {
                quality->setWaterTableMaximumDepth(parametersSettings->value("water_table_maximum_depth").toFloat());
            }

            parametersSettings->endGroup();
        }

        // proxy variables (for interpolation)
        if (group.startsWith("proxy_"))
        {
            QString name_;

            Crit3DProxy* myProxy = new Crit3DProxy();

            name_ = group.right(group.size()-6);
            myProxy->setName(name_.toStdString());
            myProxy->setFittingFunctionName(noFunction);

            parametersSettings->beginGroup(group);

            myProxy->setProxyTable(parametersSettings->value("table").toString().toStdString());
            myProxy->setProxyField(parametersSettings->value("field").toString().toStdString());
            myProxy->setGridName(getCompleteFileName(parametersSettings->value("raster").toString(), PATH_GEO).toStdString());
            myProxy->setForQualityControl(parametersSettings->value("use_for_spatial_quality_control").toBool());

            if (parametersSettings->contains("stddev_threshold"))
                myProxy->setStdDevThreshold(parametersSettings->value("stddev_threshold").toFloat());


            if (interpolationSettings.getUseMultipleDetrending())
            {
                if (getProxyPragaName(name_.toStdString()) == proxyHeight)
                {
                    if (parametersSettings->contains("fitting_function"))
                    {
                        std::string elevationFuction = parametersSettings->value("fitting_function").toString().toStdString();
                        if (fittingFunctionNames.find(elevationFuction) == fittingFunctionNames.end())
                        {
                            errorString = "Unknown function for elevation. Choose between: double_piecewise, triple_piecewise, free_triple_piecewise.";
                            return false;
                        }
                        else
                            myProxy->setFittingFunctionName(fittingFunctionNames.at(elevationFuction));
                    }

                    if (parametersSettings->contains("fitting_first_guess"))
                    {
                        myList = parametersSettings->value("fitting_first_guess").toStringList();
                        myProxy->setFittingFirstGuess(StringListToInt(myList));
                    }
                }

                if (parametersSettings->contains("fitting_parameters_max") && parametersSettings->contains("fitting_parameters_min"))
                {
                    myList = parametersSettings->value("fitting_parameters_min").toStringList();
                    QList<QString> myList2 = parametersSettings->value("fitting_parameters_max").toStringList();
                    for (int i = 0; i < myList2.size(); i++)
                        myList.append(myList2[i]);

                    myProxy->setFittingParametersRange(StringListToDouble(myList));
                }
            }

            if (! parametersSettings->contains("active"))
            {
                errorString = "active not specified for proxy " + QString::fromStdString(myProxy->getName());
                return false;
            }

            if (! parametersSettings->contains("order"))
            {
                errorString = "order not specified for proxy " + QString::fromStdString(myProxy->getName());
                return false;
            }

            if (checkProxy(*myProxy, &errorString, parametersSettings->value("active").toBool()))
            {
                proxyListTmp.push_back(*myProxy);
                proxyActiveTmp.push_back(parametersSettings->value("active").toBool());
                proxyOrder.push_back(parametersSettings->value("order").toInt());
            }
            else
                return false;

            parametersSettings->endGroup();
        }

        //proxy grid annual series
        if (group.startsWith("proxygrid"))
        {
            QString proxyName = group.right(group.length()-10);

            parametersSettings->beginGroup(group);
            int nrGrids = parametersSettings->beginReadArray("grids");
            for (int i = 0; i < nrGrids; ++i) {
                parametersSettings->setArrayIndex(i);
                proxyGridSeriesNames.push_back(getCompleteFileName(parametersSettings->value("name").toString(), PATH_GEO));
                proxyGridSeriesYears.push_back(parametersSettings->value("year").toUInt());
            }
            parametersSettings->endArray();
            parametersSettings->endGroup();

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


QString Project::getApplicationPath() const
{
    char* appImagePath = getenv("APPIMAGE");
    if (appImagePath != nullptr)
    {
        QDir d = QFileInfo(appImagePath).absoluteDir();
        QString absolute = d.absolutePath() + "/";
        return absolute;
    }
    else
    {
        return _appPath;
    }
}


void Project::setCurrentDate(QDate myDate)
{
    if (myDate != _currentDate)
    {
        _currentDate = myDate;
        if (proxyWidget != nullptr)
        {
            proxyWidget->updateDateTime(_currentDate, _currentHour);
        }
    }
}

void Project::setCurrentHour(int myHour)
{
    _currentHour = myHour;
    if (proxyWidget != nullptr)
    {
        proxyWidget->updateDateTime(_currentDate, _currentHour);
    }
}


Crit3DTime Project::getCrit3DCurrentTime() const
{
    if (_currentFrequency == hourly)
    {
        return getCrit3DTime(_currentDate, _currentHour);
    }
    else
    {
        return getCrit3DTime(_currentDate, 0);
    }
}


QDateTime Project::getCurrentTime() const
{
    QDateTime myDateTime;
    if (gisSettings.isUTC)
    {
        myDateTime.setTimeZone(QTimeZone::utc());
    }

    myDateTime.setDate(_currentDate);
    return myDateTime.addSecs(_currentHour * HOUR_SECONDS);
}


void Project::getMeteoPointsRange(float& minimum, float& maximum, bool useNotActivePoints)
{
    minimum = NODATA;
    maximum = NODATA;

    if (_currentFrequency == noFrequency || _currentVariable == noMeteoVar)
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
            meteoPoints[i].cleanAllData();
        }
    }
}

void Project::clearMeteoPoints()
{
    if (nrMeteoPoints > 0 && meteoPoints != nullptr)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            meteoPoints[i].cleanAllData();

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
 * \param fileName the name of the Digital Elevation Model file
 * \return true if file is ok, false otherwise
 */
bool Project::loadDEM(const QString &fileName)
{
    if (fileName == "")
    {
        logError("Missing DEM filename");
        return false;
    }

    logInfoGUI("Load Digital Elevation Model = " + fileName);
    demFileName = fileName;
    QString completeFileName = getCompleteFileName(fileName, PATH_DEM);

    std::string errorStr;
    bool isOk = gis::openRaster(completeFileName.toStdString(), &DEM, gisSettings.utmZone, errorStr);
    closeLogInfo();

    if (! isOk)
    {
        errorString = "Wrong Digital Elevation Model: " + completeFileName + "\n" + QString::fromStdString(errorStr);
        errorType = ERROR_DEM;
        return false;
    }
    logInfo("Digital Elevation Model = " + completeFileName);

    // check nodata
    if (! isEqual(DEM.header->flag, NODATA))
    {
        QString infoStr = "WARNING: " + QString::number(DEM.header->flag) + " is not a valid NODATA value for DEM!";
        infoStr += " It will be converted in: " + QString::number(NODATA);
        logInfo(infoStr);
        gis::convertFlagToNodata(DEM);
    }

    setColorScale(noMeteoTerrain, DEM.colorScale);

    // initialize radiation maps (slope, aspect, lat/lon, transmissivity, etc.)
    if (radiationMaps != nullptr)
        radiationMaps->clear();
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
    if (! updateProxy())
        return false;

    //set interpolation settings DEM
    interpolationSettings.setCurrentDEM(&DEM);
    qualityInterpolationSettings.setCurrentDEM(&DEM);

    //check points position with respect to DEM
    checkMeteoPointsDEM();

    return true;
}


bool Project::loadMeteoPointsDB(QString fileName)
{
    if (fileName == "")
        return false;

    closeMeteoPointsDB();

    dbPointsFileName = fileName;
    QString dbName = getCompleteFileName(fileName, PATH_METEOPOINT);
    if (! QFile(dbName).exists())
    {
        errorString = "Meteo points DB does not exists:\n" + dbName;
        return false;
    }

    meteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(dbName);
    if (! meteoPointsDbHandler->getErrorString().isEmpty())
    {
        errorString = meteoPointsDbHandler->getErrorString();
        closeMeteoPointsDB();
        return false;
    }

    if (! meteoPointsDbHandler->loadVariableProperties())
    {
        errorString = meteoPointsDbHandler->getErrorString();
        closeMeteoPointsDB();
        return false;
    }

    QList<Crit3DMeteoPoint> listMeteoPoints;
    errorString = "";
    if (! meteoPointsDbHandler->getPropertiesFromDb(listMeteoPoints, gisSettings, errorString))
    {
        errorString = "Error in reading the table point_properties\n" + errorString;
        closeMeteoPointsDB();
        return false;
    }

    nrMeteoPoints = listMeteoPoints.size();
    if (nrMeteoPoints == 0)
    {
        errorString = "Missing data in the table point_properties\n" + errorString;
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
    logInfoGUI("Check meteopoints last date...");
    meteoPointsDbLastTime = findDbPointLastTime();
    meteoPointsDbFirstTime.setSecsSinceEpoch(0);

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
    readProxyValues();

    closeLogInfo();

    // position with respect to DEM
    if (DEM.isLoaded)
    {
        checkMeteoPointsDEM();
    }

    meteoPointsLoaded = true;
    logInfo("Meteo points DB = " + dbName);

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
    if (meteoPointsDbHandler->getErrorString() != "")
    {
        errorString = "Function loadAggregationPointsDB:\n" + dbName + "\n" + meteoPointsDbHandler->getErrorString();
        closeMeteoPointsDB();
        return false;
    }

    if (! meteoPointsDbHandler->loadVariableProperties())
    {
        errorString = meteoPointsDbHandler->getErrorString();
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
    meteoPointsDbFirstTime.setSecsSinceEpoch(0);

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
    readProxyValues();

    //position with respect to DEM
    if (DEM.isLoaded)
    {
        checkMeteoPointsDEM();
    }

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

    if (! meteoGridDbHandler->parseXMLGrid(xmlName, errorString)) return false;

    if (! this->meteoGridDbHandler->openDatabase(errorString)) return false;

    if (! this->meteoGridDbHandler->loadCellProperties(errorString)) return false;

    if (! this->meteoGridDbHandler->meteoGrid()->createRasterGrid()) return false;

    if (! meteoGridDbHandler->updateMeteoGridDate(errorString))
    {
        logWarning("There was a problem reading the time range of MeteoGrid data: " + errorString);
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

    if (! meteoGridDbHandler->parseXMLGrid(xmlName, errorString))
        return false;

    if (! this->meteoGridDbHandler->newDatabase(errorString))
        return false;

    if (! this->meteoGridDbHandler->newCellProperties(errorString))
        return false;

    Crit3DMeteoGridStructure structure = this->meteoGridDbHandler->meteoGrid()->gridStructure();

    if (! this->meteoGridDbHandler->writeCellProperties(structure.nrRow(), structure.nrCol(), errorString))
        return false;

    if (! this->meteoGridDbHandler->meteoGrid()->createRasterGrid())
        return false;

    if (! meteoGridDbHandler->updateMeteoGridDate(errorString))
    {
        logInfoGUI("Error in updateMeteoGridDate: " + errorString);
    }

    if (loadGridDataAtStart || ! meteoPointsLoaded)
        setCurrentDate(meteoGridDbHandler->lastDate());

    meteoGridLoaded = true;
    logInfo("Meteo Grid = " + xmlName);

    return true;
}


bool Project::deleteMeteoGridDB()
{
    if (!meteoGridDbHandler->deleteDatabase(errorString))
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
    if (! aggregationDbHandler->loadVariableProperties())
    {
        return false;
    }
    aggregationPath = QFileInfo(dbAggregationFileName).absolutePath();
    return true;
}


bool Project::loadMeteoPointsData(const QDate& firstDate, const QDate& lastDate, bool loadHourly, bool loadDaily, bool showInfo)
{
    //check
    if (firstDate == QDate(1800,1,1) || lastDate == QDate(1800,1,1))
        return false;

    QString infoStr = "Load meteo points data: " + firstDate.toString();

    if (firstDate != lastDate)
    {
        infoStr += " - " + lastDate.toString();
    }

    int step = 0;
    if (showInfo)
    {
        step = setProgressBar(infoStr, nrMeteoPoints);
    }

    Crit3DDate myFirstDate = getCrit3DDate(firstDate);
    Crit3DDate myLastDate = getCrit3DDate(lastDate);
    bool isDataOk = false;

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (showInfo)
        {
            if ((i % step) == 0)
                updateProgressBar(i);
        }

        if (loadHourly)
        {
            if (meteoPointsDbHandler->loadHourlyData(myFirstDate, myLastDate, meteoPoints[i]))
                isDataOk = true;
        }

        if (loadDaily)
        {
            if (meteoPointsDbHandler->loadDailyData(myFirstDate, myLastDate, meteoPoints[i]))
                isDataOk = true;
        }
    }

    if (showInfo) closeProgressBar();

    return isDataOk;
}


bool Project::loadMeteoPointsData(const QDate &firstDate, const QDate &lastDate,
                                  bool loadHourly, bool loadDaily, const QString &dataset, bool showInfo)
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
                if (meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), meteoPoints[i])) isData = true;

            if (loadDaily)
                if (meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), meteoPoints[i])) isData = true;
        }
    }

    if (showInfo) closeProgressBar();
    return isData;
}


void Project::loadMeteoGridData(const QDate &firstDate, const QDate &lastDate, bool showInfo)
{
    if (this->meteoGridDbHandler != nullptr)
    {
        this->loadMeteoGridDailyData(firstDate, lastDate, showInfo);
        this->loadMeteoGridHourlyData(QDateTime(firstDate, QTime(1,0), Qt::UTC), QDateTime(lastDate.addDays(1), QTime(0,0), Qt::UTC), showInfo);
        this->loadMeteoGridMonthlyData(firstDate, lastDate, showInfo);
    }
}


bool Project::loadMeteoGridDailyData(const QDate &firstDate, const QDate &lastDate, bool showInfo)
{
    if (! meteoGridDbHandler->tableDaily().exists)
        return false;

    int nrActiveCells = 0;
    if (showInfo)
    {
        QString infoStr = "Load meteo grid daily data: " + firstDate.toString();
        if (firstDate != lastDate) infoStr += " - " + lastDate.toString();
        nrActiveCells = meteoGridDbHandler->getActiveCellsNr();
        setProgressBar(infoStr, nrActiveCells);
    }

    const auto &gridStructure = meteoGridDbHandler->gridStructure();

    unsigned int maxThreads = 1;
    if (_isParallelComputing)
    {
        maxThreads = omp_get_max_threads();
    }
    omp_set_num_threads(static_cast<int>(maxThreads));

    int count = 0;
    #pragma omp parallel
    {
        std::string id;
        QString errorStr;
        QSqlDatabase myDb;
        meteoGridDbHandler->openNewConnection(myDb, QString::number(omp_get_thread_num()), errorStr);

        #pragma omp for schedule(dynamic) reduction(+:count)
        for (int row = 0; row < gridStructure.header().nrRows; row++)
        {
            if (showInfo && omp_get_thread_num() == 0)
            {
                updateProgressBar(count * maxThreads);
            }

            for (int col = 0; col < gridStructure.header().nrCols; col++)
            {
                if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, id))
                {
                    if (! gridStructure.isFixedFields())
                    {
                        if (gridStructure.isEnsemble())
                        {
                            int memberNr = 1;
                            if (meteoGridDbHandler->loadGridDailyDataEnsemble(errorStr, QString::fromStdString(id),
                                                                                    memberNr, firstDate, lastDate))
                                ++count;
                        }
                        else
                        {
                            if (meteoGridDbHandler->loadGridDailyDataRowCol(row, col, myDb, QString::fromStdString(id),
                                                                                  firstDate, lastDate, errorStr))
                                ++count;
                        }
                    }
                    else
                    {
                        if (meteoGridDbHandler->loadGridDailyDataFixedFields(errorStr, QString::fromStdString(id),
                                                                                   firstDate, lastDate))
                            ++count;
                    }
                }
            }
        }

        myDb.close();
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


bool Project::loadMeteoGridHourlyData(QDateTime firstDateTime, QDateTime lastDateTime, bool showInfo)
{
    if (! meteoGridDbHandler->tableHourly().exists)
        return false;

    int nrActiveCells = 0;
    if (showInfo)
    {
        QString infoStr = "Load meteo grid hourly data: "
                + firstDateTime.toString("yyyy-MM-dd") + " 01:00 - "
                + lastDateTime.toString("yyyy-MM-dd") + " 00:00";
        nrActiveCells = meteoGridDbHandler->getActiveCellsNr();
        setProgressBar(infoStr, nrActiveCells);
    }

    const auto &gridStructure = meteoGridDbHandler->gridStructure();

    unsigned int maxThreads = 1;
    if (_isParallelComputing)
    {
        maxThreads = omp_get_max_threads();
    }
    omp_set_num_threads(static_cast<int>(maxThreads));

    int count = 0;
    #pragma omp parallel
    {
        std::string id;
        QString myErrorStr;
        QSqlDatabase myDb;
        meteoGridDbHandler->openNewConnection(myDb, QString::number(omp_get_thread_num()), myErrorStr);

        #pragma omp for schedule(dynamic) reduction(+:count)
        for (int row = 0; row < gridStructure.header().nrRows; row++)
        {
            if (showInfo && omp_get_thread_num() == 0)
            {
                updateProgressBar(count * maxThreads);
            }

            for (int col = 0; col < gridStructure.header().nrCols; col++)
            {
                if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, id))
                {
                    if (! gridStructure.isFixedFields())
                    {
                        if (meteoGridDbHandler->loadGridHourlyData(myDb, QString::fromStdString(id), firstDateTime, lastDateTime, myErrorStr))
                            ++count;
                    }
                    else
                    {
                        if (meteoGridDbHandler->loadGridHourlyDataFixedFields(myErrorStr, QString::fromStdString(id), firstDateTime, lastDateTime))
                            ++count;
                    }
                }
            }
        }

        myDb.close();
    }

    if (showInfo) closeProgressBar();

    if (count == 0)
    {
        errorString = "No Data Available";
        return false;
    }

    return true;
}


bool Project::loadMeteoGridMonthlyData(const QDate &firstDate, const QDate &lastDate, bool showInfo)
{
    if (! meteoGridDbHandler->tableMonthly().exists)
        return false;

    int nrActiveCells = 0;
    if (showInfo)
    {
        QString infoStr = "Load meteo grid monthly data: " + firstDate.toString();
        if (firstDate != lastDate)
            infoStr += " - " + lastDate.toString();
        nrActiveCells = meteoGridDbHandler->getActiveCellsNr();
        setProgressBar(infoStr, nrActiveCells);
    }

    const auto &gridStructure = meteoGridDbHandler->gridStructure();

    unsigned int maxThreads = 1;
    if (_isParallelComputing)
    {
        maxThreads = omp_get_max_threads();
    }
    omp_set_num_threads(static_cast<int>(maxThreads));

    int count = 0;
    #pragma omp parallel
    {
        std::string id;
        QString myErrorStr;
        QSqlDatabase myDb;
        meteoGridDbHandler->openNewConnection(myDb, QString::number(omp_get_thread_num()), myErrorStr);

        #pragma omp for schedule(dynamic) reduction(+:count)
        for (int row = 0; row < gridStructure.header().nrRows; row++)
        {
            if (showInfo && omp_get_thread_num() == 0)
            {
                updateProgressBar(count * maxThreads);
            }

            for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
            {
                if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, id))
                {
                    bool isOk;
                    if (firstDate == lastDate)
                    {
                       isOk = meteoGridDbHandler->loadGridMonthlySingleDate(myDb, QString::fromStdString(id), firstDate, myErrorStr);
                    }
                    else
                    {
                        isOk = meteoGridDbHandler->loadGridMonthlyData(myDb, QString::fromStdString(id), firstDate, lastDate, myErrorStr);
                    }

                    if (isOk)
                        ++count;
                }
            }
        }

        myDb.close();
    }

    if (showInfo) closeProgressBar();

    if (count == 0 && errorString != "")
    {
        logError("No Data Available: " + errorString);
        return false;
    }

    return true;
}


QDateTime Project::findDbPointLastTime()
{
    QDateTime lastTime;
    lastTime.setTimeZone(QTimeZone::utc());

    QDateTime lastDateD;
    lastDateD.setTimeZone(QTimeZone::utc());
    lastDateD = meteoPointsDbHandler->getLastDate(daily);
    if (! lastDateD.isNull())
    {
        lastTime = lastDateD;
    }

    QDateTime lastDateH;
    lastDateH.setTimeZone(QTimeZone::utc());
    lastDateH = meteoPointsDbHandler->getLastDate(hourly);

    if (! lastDateH.isNull())
    {
        if (! lastDateD.isNull())
        {
            lastTime = (lastDateD > lastDateH) ? lastDateD : lastDateH;
        }
        else
        {
            lastTime = lastDateH;
        }
    }

    return lastTime;
}


QDateTime Project::findDbPointFirstTime()
{
    QDateTime firstTime;
    firstTime.setTimeZone(QTimeZone::utc());

    QDateTime firstDateD;
    firstDateD.setTimeZone(QTimeZone::utc());
    firstDateD = meteoPointsDbHandler->getFirstDate(daily);
    if (! firstDateD.isNull()) firstTime = firstDateD;

    QDateTime firstDateH;
    firstDateH.setTimeZone(QTimeZone::utc());
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
            statement = QString("SELECT %1 FROM %2 WHERE id_point = '%3'").arg(proxyField, proxyTable, QString::fromStdString((*myPoint).id));
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

        if (isEqual(myPoint->proxyValues[i], NODATA))
        {
            gis::Crit3DRasterGrid* proxyGrid = myProxy->getGrid();
            if (proxyGrid == nullptr || ! proxyGrid->isLoaded)
            {
                errorString = "Error in proxy grid: " + QString::fromStdString(myProxy->getGridName());
                return false;
            }
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
                errorString = "Error loading raster proxy:\n" + fileName
                        + "\nHow to fix it: check the proxy section in the parameters.ini";
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
        if (! readPointProxyValues(&(this->meteoPoints[i]), &myDb))
            return false;
    }

    return true;
}


bool Project::updateProxy()
{
    if (DEM.isLoaded)
    {
        if (! interpolationSettings.getProxyLoaded())
        {
            if (! loadProxyGrids())
                return false;

            interpolationSettings.setProxyLoaded(true);

            if (meteoPointsDbHandler != nullptr)
            {
                if (! readProxyValues())
                    return false;
            }
        }
    }

    return true;
}

bool Project::writeTopographicDistanceMaps(bool onlyWithData, bool showInfo)
{
    if (nrMeteoPoints == 0)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    if (! DEM.isLoaded)
    {
        logError(ERROR_STR_MISSING_DEM);
        return false;
    }

    QString mapsFolder = _projectPath + PATH_TD;
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
                isSelected = meteoPointsDbHandler->existData(meteoPoints[i], daily) || meteoPointsDbHandler->existData(meteoPoints[i], hourly);
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
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    if (! demMap.isLoaded)
    {
        logError(ERROR_STR_MISSING_DEM);
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
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    QString mapsFolder = _projectPath + PATH_TD;
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
            {
                isSelected = true;
            }
            else
            {
                isSelected = meteoPointsDbHandler->existData(meteoPoints[i], daily) || meteoPointsDbHandler->existData(meteoPoints[i], hourly);
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

bool Project::writeGlocalWeightsMaps(float windowWidth)
{
    //controlla se c'è mappa aree

    gis::Crit3DRasterGrid* zoneGrid = interpolationSettings.getMacroAreasMap();
    if (! zoneGrid->isLoaded)
    {
        errorString = "Load a zone grid before";
        return false;
    }

    QString mapsFolder = _projectPath + PATH_GLOCAL;
    if (! QDir(mapsFolder).exists())
    {
        QDir().mkdir(mapsFolder);
    }

    std::vector <int> zoneCount((unsigned int)(zoneGrid->maximum));

    int i, j;

    //create raster for each zone
    std::vector <gis::Crit3DRasterGrid> zoneWeightMaps((unsigned int)(zoneGrid->maximum));
    for (i=0; i < zoneGrid->maximum; i++)
        zoneWeightMaps[i].initializeGrid(*zoneGrid);

    int cellNr = round(windowWidth / zoneGrid->header->cellSize);
    float cellDelta;

    int r1, r2, c1, c2, windowRow, windowCol, cellCount;

    for (int zoneRow = 0; zoneRow < zoneGrid->header->nrRows; zoneRow++)
    {
        for (int zoneCol = 0; zoneCol < zoneGrid->header->nrCols; zoneCol++)
        {
            if (! isEqual(zoneGrid->value[zoneRow][zoneCol], zoneGrid->header->flag))
            {
                cellCount = 0;

                for (j=0; j < zoneGrid->maximum; j++)
                    zoneCount[j] = 0;

                r1 = zoneRow - cellNr;
                r2 = zoneRow + cellNr;
                c1 = zoneCol - cellNr;
                c2 = zoneCol + cellNr;

                for (windowRow = r1; windowRow <= r2; windowRow++)
                {
                    for (windowCol = c1; windowCol <= c2; windowCol++)
                    {
                        if (! gis::isOutOfGridRowCol(windowRow, windowCol, *zoneGrid))
                        {
                            i = zoneGrid->value[windowRow][windowCol];

                            if (! isEqual(i, zoneGrid->header->flag))
                            {
                                cellDelta = gis::computeDistance(windowRow, windowCol, zoneRow, zoneCol);

                                if (cellDelta <= cellNr)
                                {
                                    cellCount++;
                                    zoneCount[i-1]++;
                                }
                            }
                        }
                    }
                }

                for (i=0; i < zoneGrid->maximum; i++)
                    if (cellCount > 0)
                        zoneWeightMaps[i].value[zoneRow][zoneCol] = float(zoneCount[i]) / float(cellCount);

            }
        }
    }

    QString fileName;
    std::string myError;
    for (int i=0; i < zoneGrid->maximum; i++)
    {
        fileName = mapsFolder + "glocalWeight_" + QString::number(i+1);
        if (! gis::writeEsriGrid(fileName.toStdString(), &zoneWeightMaps[i], myError))
        {
            errorString = QString::fromStdString(myError);
            return false;
        }
    }

    return true;
}


bool Project::loadGlocalAreasMap()
{
    //TODO: add a check for the code values?
    if (glocalMapName.isEmpty())
    {
        errorString = "Missing Glocal map name.";
        return false;
    }
    QString fileNameMap = getCompleteFileName(glocalMapName, PATH_GEO);

    if (! QFile::exists(fileNameMap + ".flt"))
    {
        errorString = "Could not find file:\n" + fileNameMap;
        return false;
    }

    gis::Crit3DRasterGrid* macroAreasGrid = new gis::Crit3DRasterGrid();

    std::string myError;
    if (gis::readEsriGrid(fileNameMap.toStdString(), macroAreasGrid, myError))
        interpolationSettings.setMacroAreasMap(macroAreasGrid);
    else
    {
        errorString = "Error loading:\n" + fileNameMap + "\n" + QString::fromStdString(myError);
        return false;
    }

    return true;
}


bool Project::loadGlocalStationsAndCells(bool isGrid, QString fileNameStations)
{
    //leggi csv aree
    std::vector<std::vector<std::string>> areaPoints;

    if (! loadGlocalStationsCsv(fileNameStations, areaPoints)) return false;

    //salva in vettore gli indici dei meteopoints che appartengono a area n, infine salva quel vettore
    std::vector<int> temp;
    std::vector<Crit3DMacroArea> myAreas;
    Crit3DMacroArea myArea;

    // per ogni area, rappresentata da righe di areaPoints, si fa ciclo sui meteopoints
    for (unsigned j = 0; j < areaPoints.size(); j++)
    {
        for (unsigned k = 0; k < areaPoints[j].size(); k++)
        {
            // controlla se id si trova nel vettore areaPoints[j] e salva l'indice del meteopoint
            for (int i=0; i < nrMeteoPoints; i++)
                if (areaPoints[j][k] == meteoPoints[i].id)
                {
                    temp.push_back(i);
                    break;
                }
        }
        // salvataggio di temp dentro il corrispettivo crit3dmacroarea
        myArea.setMeteoPoints(temp);
        myAreas.push_back(myArea);
        temp.clear();
        myArea.clear();
    }

    // assegnazione pesi a ogni cella di ogni area
    if (! loadGlocalWeightMaps(myAreas, isGrid))
        return false;

    interpolationSettings.setMacroAreas(myAreas);

    return (areaPoints.size() > 0);
}


bool Project::loadGlocalWeightMaps(std::vector<Crit3DMacroArea> &myAreas, bool isGrid)
{
    QString mapsFolder = _projectPath + PATH_GLOCAL;
    if (! QDir(mapsFolder).exists())
    {
        errorString = "Glocal dir not found. Please create glocal weight maps";
        return false;
    }

    if (QDir(mapsFolder).isEmpty(QDir::Files))
    {
        errorString = "Glocal dir empty. Please create glocal weight maps";
        return false;
    }

    std::string myError;
    gis::Crit3DRasterGrid* macroAreasGrid = new gis::Crit3DRasterGrid();
    std::string fileName = mapsFolder.toStdString() + "glocalWeight_";

    std::vector<float> areaCells;
    int nrCols, nrRows;
    double myX, myY;
    float myValue = NODATA;

    if (!isGrid)
    {
        nrCols = DEM.header->nrCols;
        nrRows = DEM.header->nrRows;
    }
    else
    {
        nrCols = meteoGridDbHandler->gridStructure().header().nrCols;
        nrRows = meteoGridDbHandler->gridStructure().header().nrRows;
    }

    if (nrCols == 0 || nrRows == 0)
    {
        macroAreasGrid->clear();
        errorString = "DEM file missing. You can open it manually or add it in the .ini file.";
        return false;
    }

    unsigned nrAreasWithCells = 0;
    std::vector<Crit3DMacroArea> existingAreas = interpolationSettings.getMacroAreas();

    for (std::size_t i = 0; i < myAreas.size(); i++)
    {
        //se ci sono già celle caricate di DEM o grid, salvale
        if (i < existingAreas.size())
        {
            myAreas[i].setAreaCellsDEM(existingAreas[i].getAreaCellsDEM());
            myAreas[i].setAreaCellsGrid(existingAreas[i].getAreaCellsGrid());
        }

        if (QFile::exists(QString::fromStdString(fileName) + QString::number(i) + ".flt"))
        {
            if (readEsriGrid(fileName + std::to_string(i), macroAreasGrid, myError))
            {
                for (int row = 0; row < nrRows; row++)
                {
                    for (int col = 0; col < nrCols; col++)
                    {
                        if (isGrid)
                        {
                            myX = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.x;
                            myY = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.y;
                        }
                        else
                            DEM.getXY(row, col, myX, myY);

                        myValue = macroAreasGrid->getValueFromXY(myX, myY);

                        if (!isEqual(myValue, NODATA) && !isEqual(myValue, 0))
                        {
                            areaCells.push_back(row*nrCols+col);
                            areaCells.push_back(myValue);
                        }
                    }
                }
                if (areaCells.size() > 0) nrAreasWithCells++;

                if (isGrid)
                    myAreas[i].setAreaCellsGrid(areaCells);
                else
                    myAreas[i].setAreaCellsDEM(areaCells);

                areaCells.clear();
            }
            else
            {
                macroAreasGrid->clear();
                errorString = "macroareas not found";
                return false;
            }
        }
    }
    macroAreasGrid->clear();

    if (nrAreasWithCells == 0)
    {
        errorString = "No valid files found in glocal dir. Please create glocal weight maps";
        return false;
    }

    return true;
}

bool Project::loadGlocalStationsCsv(QString fileName, std::vector<std::vector<std::string>> &areaPoints)
{
    if (fileName == "")
    {
        errorString = "Missing csv filename";
        return false;
    }
    areaPoints.clear();

    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        errorString = "Missing csv file: " + fileName;
        return false;
    }

    QFile myFile(fileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        errorString = "Failed opening: " + fileName;
        return false;
    }

    QTextStream myStream (&myFile);
    QList<QString> line;

    if (myStream.atEnd())
    {
        errorString = "File is void";
        myFile.close();
        return false;
    }

    std::vector<std::string> temp;
    while(!myStream.atEnd())
    {
        line = myStream.readLine().split(',');
        if (line.size() > 1)
        {
            int areaNr = line[0].toInt();
            for (int i = 1; i < line.size(); i++)
            {
                temp.push_back(line[i].toStdString());
            }

            if (areaPoints.empty() || areaNr > ((int)areaPoints.size() - 1))
            {
                areaPoints.resize(areaNr + 1);
            }

            areaPoints[areaNr] = temp;
            temp.clear();
        }
    }
    myFile.close();

    return true;
}


bool Project::groupCellsInArea(std::vector<int> &areaPoints, int index, bool isGrid)
{
    unsigned int zoneNr;
    double myX, myY;
    gis::Crit3DRasterGrid* macroAreas = interpolationSettings.getMacroAreasMap();
    int nrCols;
    areaPoints.clear();

    if (!isGrid)
    {
        nrCols = DEM.header->nrCols;

        for (int i = 0; i < DEM.header->nrRows; i++)
        {
            for (int j = 0; j < DEM.header->nrCols; j++)
            {
                if (!isEqual(DEM.getValueFromRowCol(i,j),DEM.header->flag))
                {
                    gis::getUtmXYFromRowCol(DEM.header, i, j, &myX, &myY);

                    zoneNr = macroAreas->getValueFromXY(myX, myY);

                    if ((int)zoneNr == index)
                        areaPoints.push_back(i*nrCols+j);
                }
            }
        }
    }
    else
    {
        nrCols = meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols;

        for (int i = 0; i < meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows; i++)
        {
            for (int j = 0; j < meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols; j++)
            {
                if (meteoGridDbHandler->meteoGrid()->meteoPoints()[i][j]->active)
                {
                    myX = meteoGridDbHandler->meteoGrid()->meteoPoints()[i][j]->point.utm.x;
                    myY = meteoGridDbHandler->meteoGrid()->meteoPoints()[i][j]->point.utm.y;

                    zoneNr = macroAreas->getValueFromXY(myX, myY);

                    if ((int)zoneNr == index)
                        areaPoints.push_back(i*nrCols+j);
                }
            }
        }
    }

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

void Project::passGridTemperatureToHumidityPoints(Crit3DTime myTime, Crit3DMeteoSettings* meteoSettings)
{
    float airRelHum, airT;
    int row, col;
    double lat, lon;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        airRelHum = meteoPoints[i].getMeteoPointValue(myTime, airRelHumidity, meteoSettings);
        airT = meteoPoints[i].getMeteoPointValue(myTime, airTemperature, meteoSettings);

        if (! isEqual(airRelHum, NODATA) && isEqual(airT, NODATA))
        {
            if (meteoGridDbHandler->meteoGrid()->gridStructure().isUTM())
            {
                gis::getRowColFromLonLat(meteoGridDbHandler->meteoGrid()->gridStructure().header(),
                                     meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, &row, &col);
            }
            else
            {
                gis::getLatLonFromUtm(gisSettings, meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, &lat, &lon);
                gis::getRowColFromLonLat(meteoGridDbHandler->meteoGrid()->gridStructure().header(),
                                         lon, lat, &row, &col);
            }

            if (! gis::isOutOfGridRowCol(row, col, meteoGridDbHandler->meteoGrid()->gridStructure().header()))
            {
                if (meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).active)
                {
                    meteoPoints[i].setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), airTemperature,
                                                   meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), airTemperature));
                }
            }
        }
    }
}

bool Project::interpolationOutputPoints(std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                        gis::Crit3DRasterGrid *outputGrid, meteoVariable myVar)
{
    // GA 2024-03-29 mi sembra che qua manchi il local detrending

    if (! getComputeOnlyPoints()) return false;

    if (outputPoints.empty())
    {
        errorString = "Missing output points.";
        return false;
    }

    if (interpolationSettings.getUseMultipleDetrending())
        interpolationSettings.clearFitting();

    std::vector <double> proxyValues;
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
                if (getUseDetrendingVar(myVar))
                {
                    getProxyValuesXY(x, y, interpolationSettings, proxyValues);
                }

                outputPoints[i].currentValue = interpolate(interpolationPoints, interpolationSettings,
                                                            meteoSettings, myVar, x, y, z, proxyValues, true);

                outputGrid->value[row][col] = outputPoints[i].currentValue;
            }
        }
    }

    return true;
}


bool Project::computeStatisticsCrossValidation()
{
    crossValidationStatistics.initialize();

    std::vector <float> obs;
    std::vector <float> pre;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active)
        {
            float value = meteoPoints[i].currentValue;

            if (! isEqual(value, NODATA) && ! isEqual(meteoPoints[i].residual, NODATA))
            {
                obs.push_back(value);
                pre.push_back(value - meteoPoints[i].residual);
            }
        }
    }

    if (obs.size() > 0)
    {
        crossValidationStatistics.setMeanAbsoluteError(statistics::meanAbsoluteError(obs, pre));
        crossValidationStatistics.setMeanBiasError(statistics::meanError(obs, pre));
        crossValidationStatistics.setRootMeanSquareError(statistics::rootMeanSquareError(obs, pre));
        crossValidationStatistics.setNashSutcliffeEfficiency(statistics::NashSutcliffeEfficiency(obs, pre));

        float intercept, slope, r2;
        statistics::linearRegression(obs, pre, int(obs.size()), false, &intercept, &slope, &r2);
        crossValidationStatistics.setR2(r2);
    }

    return true;
}

bool Project::computeStatisticsGlocalCrossValidation(Crit3DMacroArea myArea)
{
    std::vector<int> meteoPointsList = myArea.getMeteoPoints();


    Crit3DCrossValidationStatistics cvStatistics;
    cvStatistics.initialize();

    std::vector <float> obs;
    std::vector <float> pre;

    for (int i = 0; i < (int)meteoPointsList.size(); i++)
    {
        if (meteoPoints[meteoPointsList[i]].active)
        {
            float value = meteoPoints[meteoPointsList[i]].currentValue;

            if (! isEqual(value, NODATA) && ! isEqual(meteoPoints[meteoPointsList[i]].residual, NODATA))
            {
                obs.push_back(value);
                pre.push_back(value - meteoPoints[meteoPointsList[i]].residual);
            }
        }
    }

    if (obs.size() > 0)
    {
        cvStatistics.setMeanAbsoluteError(statistics::meanAbsoluteError(obs, pre));
        cvStatistics.setMeanBiasError(statistics::meanError(obs, pre));
        cvStatistics.setRootMeanSquareError(statistics::rootMeanSquareError(obs, pre));
        cvStatistics.setNashSutcliffeEfficiency(statistics::NashSutcliffeEfficiency(obs, pre));

        float intercept, slope, r2;
        statistics::linearRegression(obs, pre, int(obs.size()), false, &intercept, &slope, &r2);
        cvStatistics.setR2(r2);
    }

    glocalCrossValidationStatistics.push_back(cvStatistics);

    return true;
}


bool Project::interpolationCv(meteoVariable myVar, const Crit3DTime& myTime)
{
    if (! checkInterpolation(myVar)) return false;

    // check variables
    if ( interpolationSettings.getUseDewPoint() &&
        (myVar == dailyAirRelHumidityAvg ||
        myVar == dailyAirRelHumidityMin ||
        myVar == dailyAirRelHumidityMax ||
        myVar == airRelHumidity))
    {
        logError("Cross validation is not available for " + QString::fromStdString(getVariableString(myVar))
                 + "\n Deactivate option 'Interpolate relative humidity using dew point'");
        return false;
    }

    if (myVar == dailyGlobalRadiation ||
        myVar == globalIrradiance ||
        myVar == dailyLeafWetness ||
        myVar == dailyWindVectorDirectionPrevailing ||
        myVar == dailyWindVectorIntensityAvg ||
        myVar == dailyWindVectorIntensityMax )
    {
        logError("Cross validation is not available for " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    std::string errorStdStr;

    // check quality and pass data to interpolation
    if (! checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         qualityInterpolationSettings, interpolationSettings, meteoSettings,
                                         &climateParameters, interpolationPoints,
                                         checkSpatialQuality, errorStdStr))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)) + "\n" + QString::fromStdString(errorStdStr));
        return false;
    }

    if (interpolationSettings.getUseMultipleDetrending())
        interpolationSettings.clearFitting();

    if (! interpolationSettings.getUseLocalDetrending() && ! preInterpolation(interpolationPoints, interpolationSettings, meteoSettings, &climateParameters,
                          meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
    {
        logError("Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr));
        return false;
    }

    if (! interpolationSettings.getUseLocalDetrending() && ! interpolationSettings.getUseGlocalDetrending())
    {
        if (! computeResiduals(myVar, meteoPoints, nrMeteoPoints, interpolationPoints, interpolationSettings, meteoSettings, interpolationSettings.getUseExcludeStationsOutsideDEM(), true))
            return false;
    }
    else if (interpolationSettings.getUseGlocalDetrending())
    {
        glocalCrossValidationStatistics.clear();
        interpolationSettings.clearMacroAreaNumber();

        if(!setMultipleDetrendingHeightTemperatureRange(interpolationSettings))
        {
            errorString = "Error in function preInterpolation: \n couldn't set temperature ranges for height proxy.";
            return false;
        }

        if (! computeResidualsAndStatisticsGlocalDetrending(myVar, interpolationPoints))
        {
            return false;
        }

    }
    else if (interpolationSettings.getUseLocalDetrending())
    {
        if(! setMultipleDetrendingHeightTemperatureRange(interpolationSettings))
        {
            errorString = "Error in function preInterpolation: \n couldn't set temperature ranges for height proxy.";
            return false;
        }

        if (! computeResidualsLocalDetrending(myVar, myTime, meteoPoints, nrMeteoPoints, interpolationPoints,
                                             interpolationSettings, meteoSettings, &climateParameters, true, true))
            return false;
    }

    if (! interpolationSettings.getUseGlocalDetrending())
        if (! computeStatisticsCrossValidation())
            return false;

    return true;
}



bool Project::interpolationDem(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    std::string errorStdStr;

    // check quality and pass data to interpolation
    if (! checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         qualityInterpolationSettings, interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality, errorStdStr))
    {
        errorString = "No data available: " + QString::fromStdString(getVariableString(myVar))
                      + "\n" + QString::fromStdString(errorStdStr);
        return false;
    }

    // detrending, checking precipitation and optimizing td parametersSettings
    if (! preInterpolation(interpolationPoints, interpolationSettings, meteoSettings,
                         &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
    {
        errorString = "Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr);
        return false;
    }

    // interpolate
    if (getComputeOnlyPoints())
    {
        myRaster->initializeGrid(DEM);
        if (! interpolationOutputPoints(interpolationPoints, myRaster, myVar))
            return false;
    }
    else
    {
        if (! interpolationRaster(interpolationPoints, interpolationSettings, meteoSettings, myRaster, DEM, myVar, _isParallelComputing))
        {
            errorString = "Error in function interpolationRaster.";
            return false;
        }
    }

    myRaster->setMapTime(myTime);

    return true;
}


bool Project::interpolationDemLocalDetrending(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (! getUseDetrendingVar(myVar) || ! interpolationSettings.getUseLocalDetrending())
        return false;

    // pass data to interpolation
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    std::string errorStdStr;

    if (! checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         qualityInterpolationSettings, interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality, errorStdStr))
    {
        errorString = "No data available: " + QString::fromStdString(getVariableString(myVar))
                      + "\n" + QString::fromStdString(errorStdStr);
        return false;
    }

    Crit3DProxyCombination myCombination = interpolationSettings.getSelectedCombination();
    interpolationSettings.setCurrentCombination(myCombination);

    if (getComputeOnlyPoints())
    {
        std::vector <double> proxyValues(interpolationSettings.getProxyNr());
        for (unsigned int i = 0; i < outputPoints.size(); i++)
        {
            if (outputPoints[i].active)
            {
                double x = outputPoints[i].utm.x;
                double y = outputPoints[i].utm.y;
                int row, col;
                myRaster->getRowCol(x, y, row, col);
                if (! myRaster->isOutOfGrid(row, col))
                {
                    getProxyValuesXY(x, y, interpolationSettings, proxyValues);

                    std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                    localSelection(interpolationPoints, subsetInterpolationPoints, x, y, interpolationSettings, false); //CT supplementari vengono usate anche in interpolate?
                    if (! preInterpolation(subsetInterpolationPoints, interpolationSettings, meteoSettings, &climateParameters,
                                          meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
                    {
                        errorString = "Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr);
                        return false;
                    }


                    outputPoints[i].currentValue = interpolate(subsetInterpolationPoints, interpolationSettings, meteoSettings,
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

        if(! setMultipleDetrendingHeightTemperatureRange(interpolationSettings))
        {
            errorString = "Error in function preInterpolation: \n couldn't set temperature ranges for height proxy.";
            return false;
        }

        unsigned int maxThreads = 1;
        if (_isParallelComputing)
        {
            maxThreads = omp_get_max_threads();
        }
        omp_set_num_threads(static_cast<int>(maxThreads));

        #pragma omp parallel
        {
            Crit3DInterpolationSettings myInterpolationSettings = interpolationSettings;
            std::vector<double> proxyValues(myInterpolationSettings.getProxyNr());

            #pragma omp for
            for (long row = 0; row < myHeader.nrRows ; row++)
            {
                for (long col = 0; col < myHeader.nrCols; col++)
                {
                    float z = DEM.value[row][col];
                    if (! isEqual(z, myHeader.flag))
                    {
                        double x, y;
                        gis::getUtmXYFromRowCol(myHeader, row, col, &x, &y);

                        if (getUseDetrendingVar(myVar))
                        {
                            getProxyValuesXY(x, y, myInterpolationSettings, proxyValues);
                        }

                        std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                        localSelection(interpolationPoints, subsetInterpolationPoints, x, y, myInterpolationSettings, false);

                        preInterpolation(subsetInterpolationPoints, myInterpolationSettings, meteoSettings, &climateParameters,
                                         meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr);

                        myRaster->value[row][col] = interpolate(subsetInterpolationPoints, myInterpolationSettings, meteoSettings,
                                                                myVar, x, y, z, proxyValues, true);
                        myInterpolationSettings.clearFitting();
                        myInterpolationSettings.setCurrentCombination(myInterpolationSettings.getSelectedCombination());

                        subsetInterpolationPoints.clear();
                    }
                }
            }
        }

        if (! gis::updateMinMaxRasterGrid(myRaster))
            return false;
    }

    myRaster->setMapTime(myTime);

    return true;
}


bool Project::interpolationDemGlocalDetrending(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (!getUseDetrendingVar(myVar) || !interpolationSettings.getUseGlocalDetrending())
        return false;

    // pass data to interpolation
    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    std::string errorStdStr;

    if (! checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         qualityInterpolationSettings, interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality, errorStdStr))
    {
        errorString = "No data available: " + QString::fromStdString(getVariableString(myVar))
                      + "\n" + QString::fromStdString(errorStdStr);
        return false;
    }

    Crit3DProxyCombination myCombination = interpolationSettings.getSelectedCombination();
    interpolationSettings.setCurrentCombination(myCombination);

    // obtain fitting parameters and combination for every macro area
    if (! preInterpolation(interpolationPoints, interpolationSettings, meteoSettings, &climateParameters,
                          meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
    {
        errorString = "Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr);
        return false;
    }

    bool isOk = true;

    if (getComputeOnlyPoints())
    {
        //TODO
    }
    else
    {
        gis::Crit3DRasterHeader myHeader = *(DEM.header);
        myRaster->initializeGrid(myHeader);

        if(! setMultipleDetrendingHeightTemperatureRange(interpolationSettings))
        {
            errorString = "Error in function preInterpolation: \n couldn't set temperature ranges for height proxy.";
            return false;
        }

        //preparing to start a "global" interpolation for every single macro area
        int elevationPos = NODATA;
        for (unsigned int pos=0; pos < interpolationSettings.getCurrentCombination().getProxySize(); pos++)
        {
            if (getProxyPragaName(interpolationSettings.getProxy(pos)->getName()) == proxyHeight)
                elevationPos = pos;
        }

        unsigned int maxThreads = 1;
        if (_isParallelComputing)
        {
            maxThreads = omp_get_max_threads();
        }
        omp_set_num_threads(static_cast<int>(maxThreads));

        #pragma omp parallel
        {
            Crit3DInterpolationSettings myInterpolationSettings = interpolationSettings;
            std::vector<double> proxyValues(myInterpolationSettings.getProxyNr());

            #pragma omp for
            for (int areaIndex = 0; areaIndex < myInterpolationSettings.getMacroAreasSize(); areaIndex++)
            {
                if (! isOk) continue;   // early skip if already failed

                // load macro area and its cells
                Crit3DMacroArea macroArea = myInterpolationSettings.getMacroArea(areaIndex);
                std::vector<float> areaCells = macroArea.getAreaCellsDEM();
                std::vector<Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                double interpolatedValue = NODATA;

                if (! areaCells.empty())
                {
                    macroAreaDetrending(macroArea, myVar, myInterpolationSettings, meteoSettings,
                                        meteoPoints, interpolationPoints, subsetInterpolationPoints, elevationPos);

                    // calculate value for every cell
                    for (std::size_t cellIndex = 0; cellIndex < areaCells.size(); cellIndex = cellIndex + 2)
                    {
                        if (! isOk) break;   // early skip if already failed

                        int row = int(areaCells[cellIndex] / DEM.header->nrCols);
                        int col = int(areaCells[cellIndex]) % DEM.header->nrCols;

                        float z = DEM.value[row][col];

                        if (! isEqual(z, myHeader.flag))
                        {
                            double x, y;
                            gis::getUtmXYFromRowCol(myHeader, row, col, &x, &y);

                            if (! getSignificantProxyValuesXY(x, y, myInterpolationSettings, proxyValues))
                            {
                                myRaster->value[row][col] = NODATA;
                                continue;
                            }

                            interpolatedValue = interpolate(subsetInterpolationPoints, myInterpolationSettings, meteoSettings,
                                                            myVar, x, y, z, proxyValues, true);

                            if (! isEqual(interpolatedValue, NODATA))
                            {
                                // protect writes if same cell could be touched by multiple threads
                                #pragma omp critical
                                {
                                    if (isEqual(myRaster->value[row][col], NODATA))
                                        myRaster->value[row][col] = interpolatedValue * areaCells[cellIndex + 1];
                                    else
                                        myRaster->value[row][col] += interpolatedValue * areaCells[cellIndex + 1];
                                }
                            }
                            else
                            {
                                // cannot return directly; set flag instead
                                isOk = false;
                            }
                        }
                    }
                }
            }
        }
    }

    return isOk;
}


bool Project::interpolateDemRadiation(const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (! radiationMaps->latMap->isLoaded || ! radiationMaps->lonMap->isLoaded)
        return false;

    if (! radiationMaps->slopeMap->isLoaded || ! radiationMaps->aspectMap->isLoaded)
        return false;

    radiationMaps->initialize();

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    radSettings.setGisSettings(&gisSettings);

    gis::Crit3DPoint mapCenter = DEM.getCenter();

    int intervalWidth = radiation::estimateTransmissivityWindow(&radSettings, DEM, mapCenter, myTime, int(HOUR_SECONDS));

    // at least a meteoPoint with transmissivity data
    if (! computeTransmissivity(&radSettings, meteoPoints, nrMeteoPoints, intervalWidth, myTime, DEM))
    {
        // TODO: add flag to parametersSettings. Could be NOT wanted
        if (! computeTransmissivityFromTRange(meteoPoints, nrMeteoPoints, myTime))
        {
            logError("Error in function interpolateDemRadiation: cannot compute transmissivity.");
            return false;
        }
    }

    bool result;
    std::string errorStdStr;

    result = checkAndPassDataToInterpolation(quality, atmTransmissivity, meteoPoints, nrMeteoPoints,
                                          myTime, qualityInterpolationSettings, interpolationSettings,
                                          meteoSettings, &climateParameters,
                                          interpolationPoints, checkSpatialQuality, errorStdStr);
    if (! result)
    {
        logError("Error in function interpolateDemRadiation: not enough transmissivity data.");
        return false;
    }
    if (! preInterpolation(interpolationPoints, interpolationSettings, meteoSettings, &climateParameters,
                            meteoPoints, nrMeteoPoints, atmTransmissivity, myTime, errorStdStr))
    {
        logError("Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr));
        return false;
    }

    // interpolate transmissivity
    if (getComputeOnlyPoints())
    {
        result = interpolationOutputPoints(interpolationPoints, radiationMaps->transmissivityMap, atmTransmissivity);
    }
    else
    {
        result = interpolationRaster(interpolationPoints, interpolationSettings, meteoSettings,
                                     radiationMaps->transmissivityMap, DEM, atmTransmissivity, _isParallelComputing);
    }
    if (! result)
    {
        logError("Function interpolateDemRadiation: error in interpolating transmissivity.");
        return false;
    }

    // compute radiation
    if (getComputeOnlyPoints())
    {
        result = radiation::computeRadiationOutputPoints(&radSettings, DEM, radiationMaps, outputPoints, myTime);
    }
    else
    {
        result = radiation::computeRadiationDEM(&radSettings, DEM, radiationMaps, myTime, _isParallelComputing);
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


bool Project::checkInterpolation(meteoVariable myVar)
{
    if (! DEM.isLoaded)
    {
        logError(ERROR_STR_MISSING_DEM);
        return false;
    }

    if (nrMeteoPoints == 0)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    if (myVar == noMeteoVar)
    {
        logError("Select a variable before.");
        return false;
    }

    return true;
}


bool Project::checkInterpolationGrid(meteoVariable myVar)
{
    if (nrMeteoPoints == 0)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    if (myVar == noMeteoVar)
    {
        logError("Select a variable before.");
        return false;
    }

    return true;
}


bool Project::checkGlocal(bool isGrid)
{
    if (interpolationSettings.getUseGlocalDetrending() && ! interpolationSettings.isGlocalReady(isGrid))
    {
        if (! loadGlocalAreasMap())
            return false;
        if (! loadGlocalStationsAndCells(isGrid, getCompleteFileName(glocalPointsName, PATH_GEO)))
            return false;
    }

    return true;
}


bool Project::interpolationDemMain(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster)
{
    if (! checkInterpolation(myVar))
        return false;

    // solar radiation model
    if (myVar == globalIrradiance)
    {
        Crit3DTime halfHour = myTime.addSeconds(-1800);
        return interpolateDemRadiation(halfHour, myRaster);
    }

    if (! checkGlocal(false)) return false;

    if (interpolationSettings.getUseMultipleDetrending())
        interpolationSettings.clearFitting();

    // dynamic lapserate
    if (getUseDetrendingVar(myVar) && interpolationSettings.getUseLocalDetrending())
    {
        return interpolationDemLocalDetrending(myVar, myTime, myRaster);
    }
	else if (getUseDetrendingVar(myVar) && interpolationSettings.getUseGlocalDetrending())
    {
        return interpolationDemGlocalDetrending(myVar, myTime, myRaster);
    }
    else
    {
        return interpolationDem(myVar, myTime, myRaster);
    }
}


bool Project::meteoGridAggregateProxy(std::vector <gis::Crit3DRasterGrid*> &myGrids)
{
    gis::Crit3DRasterGrid* proxyGrid;

    float cellSize = computeDefaultCellSizeFromMeteoGrid(1);
    gis::Crit3DRasterGrid meteoGridRaster;
    if (! meteoGridDbHandler->MeteoGridToRasterFlt(cellSize, gisSettings, meteoGridRaster))
        return false;

    for (unsigned int i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        if (interpolationSettings.getCurrentCombination().isProxyActive(i))
        {
            gis::Crit3DRasterGrid* myGrid = new gis::Crit3DRasterGrid();

            proxyGrid = interpolationSettings.getProxy(i)->getGrid();
            if (proxyGrid != nullptr && proxyGrid->isLoaded)
                gis::resampleGrid(*proxyGrid, myGrid, meteoGridRaster.header, aggrAverage, 0);

            myGrids.push_back(myGrid);
        }
    }

    return true;
}


bool Project::interpolationGrid(meteoVariable myVar, const Crit3DTime& myTime)
{
    if (! checkInterpolationGrid(myVar))
        return false;

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    std::string errorStdStr;

    if (interpolationSettings.getUseMultipleDetrending())
        interpolationSettings.clearFitting();

    // check quality and pass data to interpolation
    if (! checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, myTime,
                                         qualityInterpolationSettings, interpolationSettings, meteoSettings, &climateParameters, interpolationPoints,
                                         checkSpatialQuality, errorStdStr))
    {
        logError("No data available: " + QString::fromStdString(getVariableString(myVar)));
        return false;
    }

    Crit3DProxyCombination myCombination;

    if (! interpolationSettings.getUseLocalDetrending())
    {
        if (! preInterpolation(interpolationPoints, interpolationSettings, meteoSettings,
                              &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
        {
            logError("Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr));
            return false;
        }
        myCombination = interpolationSettings.getCurrentCombination();
    }
    else
    {
        myCombination = interpolationSettings.getSelectedCombination();
        interpolationSettings.setCurrentCombination(myCombination);
        if(! setMultipleDetrendingHeightTemperatureRange(interpolationSettings))
        {
            errorString = "Error in function preInterpolation: \n couldn't set temperature ranges for height proxy.";
            return false;
        }
    }

    // proxy aggregation
    std::vector <gis::Crit3DRasterGrid*> meteoGridProxies;
    if (getUseDetrendingVar(myVar))
        if (! meteoGridAggregateProxy(meteoGridProxies)) return false;

    frequencyType freq = getVarFrequency(myVar);

    float myX, myY, myZ;
    std::vector <double> proxyValues;
    proxyValues.resize(unsigned(interpolationSettings.getProxyNr()));

    float interpolatedValue = NODATA;
    unsigned int i, proxyIndex;

    if (! interpolationSettings.getUseGlocalDetrending())
    {
    for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
    {
        for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
        {
            if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active)
            {
                myX = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.x;
                myY = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.y;
                myZ = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.z;

                if (getUseDetrendingVar(myVar))
                {
                    proxyIndex = 0;

                    for (i=0; i < interpolationSettings.getProxyNr(); i++)
                    {
                        proxyValues[i] = NODATA;

                        if (myCombination.isProxyActive(i))
                        {
                            if (proxyIndex < meteoGridProxies.size())
                            {
                                float proxyValue = gis::getValueFromXY(*meteoGridProxies[proxyIndex], myX, myY);
                                if (proxyValue != meteoGridProxies[proxyIndex]->header->flag)
                                    proxyValues[i] = double(proxyValue);
                            }

                            proxyIndex++;
                        }
                    }
                    if (interpolationSettings.getUseLocalDetrending())
                    {
                        std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;
                        localSelection(interpolationPoints, subsetInterpolationPoints, myX, myY, interpolationSettings, false); //CT supplementari usate anche in interpolate?
                        if (! preInterpolation(subsetInterpolationPoints, interpolationSettings, meteoSettings,
                                              &climateParameters, meteoPoints, nrMeteoPoints, myVar, myTime, errorStdStr))
                        {
                            logError("Error in function preInterpolation:\n" + QString::fromStdString(errorStdStr));
                            return false;
                        }

                        interpolatedValue = interpolate(subsetInterpolationPoints, interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true);
                    }
                    else
                    {
                        interpolatedValue = interpolate(interpolationPoints, interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true);
                    }
                }
                else
                {
                    interpolatedValue = interpolate(interpolationPoints, interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true);
                }
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
	}
    else
    {
        //set all cells to NODATA
        for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
        {
            for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
            {
                if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active)
                {
                    if (freq == hourly)
                        meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = NODATA;
                    else if (freq == daily)
                        meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = NODATA;
                }
            }
        }
        unsigned row, col;
        std::vector<float> areaCells;
        float myValue = NODATA;

        int elevationPos = NODATA;
        for (unsigned int pos=0; pos < interpolationSettings.getCurrentCombination().getProxySize(); pos++)
        {
            if (getProxyPragaName(interpolationSettings.getProxy(pos)->getName()) == proxyHeight)
                elevationPos = pos;
        }

        for (int areaIndex = 0; areaIndex < interpolationSettings.getMacroAreasSize(); areaIndex++)
        {
            //load macro area and its cells
            Crit3DMacroArea macroArea = interpolationSettings.getMacroArea(areaIndex);
            areaCells = macroArea.getAreaCellsGrid();
			std::vector<Crit3DInterpolationDataPoint> subsetInterpolationPoints;

            if (! areaCells.empty())
            {
                macroAreaDetrending(macroArea, myVar, interpolationSettings, meteoSettings, meteoPoints,
                                    interpolationPoints, subsetInterpolationPoints, elevationPos);
                unsigned nrCols = meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols;
                //calculate value for every cell
                for (unsigned cellIndex = 0; cellIndex < areaCells.size(); cellIndex = cellIndex + 2)
                {
                    row = unsigned(areaCells[cellIndex]/nrCols);
                    col = int(areaCells[cellIndex])%nrCols;

                    if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active)
                    {
                        myX = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.x;
                        myY = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.utm.y;
                        myZ = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->point.z;

                        if (getUseDetrendingVar(myVar))
                        {
                            bool proxyFlag = true;
                            proxyIndex = 0;

                            for (i=0; i < interpolationSettings.getProxyNr(); i++)
                            {
                                proxyValues[i] = NODATA;

                                if (interpolationSettings.getCurrentCombination().isProxyActive(i))
                                {
                                    if (proxyIndex < meteoGridProxies.size())
                                    {
                                        float proxyValue = gis::getValueFromXY(*meteoGridProxies[proxyIndex], myX, myY);
                                        if (proxyValue != meteoGridProxies[proxyIndex]->header->flag)
                                            proxyValues[i] = double(proxyValue);
                                        else
                                        {
                                            proxyFlag = false;
                                            break;
                                        }
                                    }
                                    proxyIndex++;
                                }
                            }

                            if (! proxyFlag)
                                continue;

                            double temp = interpolate(subsetInterpolationPoints, interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true);
                            if (! isEqual(temp, NODATA))
                                interpolatedValue = interpolate(subsetInterpolationPoints, interpolationSettings, meteoSettings, myVar, myX, myY, myZ, proxyValues, true)
                                                * areaCells[cellIndex + 1];
                            else
                                return false;
                        }
                        if (freq == hourly)
                        {
                            if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->nrObsDataDaysH == 0)
                                meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->initializeObsDataH(1, 1, myTime.date);

                            myValue = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue;

                            if (isEqual(myValue, NODATA))
                                myValue = 0;

                            meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar, float(interpolatedValue+myValue));
                            meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = float(interpolatedValue+myValue);

                        }
                        else if (freq == daily)
                        {
                            if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->nrObsDataDaysD == 0)
                                meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->initializeObsDataD(1, myTime.date);

                            myValue = meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue;

                            if (isEqual(myValue, NODATA))
                                myValue = 0;

                            meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->setMeteoPointValueD(myTime.date, myVar, float(interpolatedValue+myValue));
                            meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->currentValue = float(interpolatedValue+myValue);

                        }
                    }
                }
            }
        }
    }

    return true;
}


float Project::meteoDataConsistency(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin)
{
    float dataConsistency = 0.0;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        dataConsistency = MAXVALUE(dataConsistency, meteoPoints[i].obsDataConsistencyH(myVar, timeIni, timeFin));
    }

    return dataConsistency;
}


QString Project::getCompleteFileName(QString fileName, QString secondaryPath)
{
    if (fileName.isEmpty()) return fileName;

    if (getFilePath(fileName) == "")
    {
        QString completeFileName = this->getDefaultPath() + secondaryPath + fileName;
        return QDir().cleanPath(completeFileName);
    }
    else if (fileName.at(0) == '.')
    {
        QString completeFileName = this->getProjectPath() + fileName;
        return QDir().cleanPath(completeFileName);
    }
    else
    {
        return fileName;
    }
}


QString Project::getRelativePath(QString fileName) const
{
    if (! fileName.isEmpty() && fileName.at(0) != '.' && getFilePath(fileName) != "")
    {
        QDir projectDir(getProjectPath());
        QString relativePath = projectDir.relativeFilePath(fileName);
        if (relativePath != fileName)
        {
            fileName = relativePath;
            if (fileName.at(0) != '.')
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
            if (myPath.at(0) == '.')
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

        _projectName = projectSettings->value("name").toString();
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
        _verboseStdoutLogging = projectSettings->value("verbose_stdout_log", "true").toBool();
        _currentTileMap = projectSettings->value("tile_map").toString();
    projectSettings->endGroup();
    return true;
}


bool Project::searchDefaultPath(QString& defaultPath)
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

    defaultPath = QDir::cleanPath(myPath) + "/DATA/";
    return true;
}


Crit3DCrossValidationStatistics Project::getCrossValidationStatistics() const
{
    return crossValidationStatistics;
}

void Project::setCrossValidationStatistics(const Crit3DCrossValidationStatistics &newCrossValidationStatistics)
{
    crossValidationStatistics = newCrossValidationStatistics;
}


frequencyType Project::getCurrentFrequency() const
{
    return _currentFrequency;
}


void Project::setCurrentFrequency(const frequencyType &value)
{
    _currentFrequency = value;
    if (proxyWidget != nullptr)
    {
        proxyWidget->updateFrequency(_currentFrequency);
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
        projectSettings->setValue("name", _projectName);
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
    parametersSettings->beginGroup("radiation");
        parametersSettings->setValue("algorithm", QString::fromStdString(getKeyStringRadAlgorithm(radSettings.getAlgorithm())));
        parametersSettings->setValue("real_sky_algorithm", QString::fromStdString(getKeyStringRealSky(radSettings.getRealSkyAlgorithm())));
        parametersSettings->setValue("linke_mode", QString::fromStdString(getKeyStringParamMode(radSettings.getLinkeMode())));
        parametersSettings->setValue("albedo_mode", QString::fromStdString(getKeyStringParamMode(radSettings.getAlbedoMode())));
        parametersSettings->setValue("tilt_mode", QString::fromStdString(getKeyStringTiltMode(radSettings.getTiltMode())));
        parametersSettings->setValue("real_sky", radSettings.getRealSky());
        parametersSettings->setValue("shadowing", radSettings.getShadowing());
        parametersSettings->setValue("linke", QString::number(double(radSettings.getLinke())));
        parametersSettings->setValue("albedo", QString::number(double(radSettings.getAlbedo())));
        parametersSettings->setValue("tilt", QString::number(double(radSettings.getTilt())));
        parametersSettings->setValue("aspect", QString::number(double(radSettings.getAspect())));
        parametersSettings->setValue("clear_sky", QString::number(double(radSettings.getClearSky())));
        parametersSettings->setValue("linke_map", getRelativePath(QString::fromStdString(radSettings.getLinkeMapName())));
        parametersSettings->setValue("albedo_map", getRelativePath(QString::fromStdString(radSettings.getAlbedoMapName())));
        parametersSettings->setValue("linke_monthly", FloatVectorToStringList(radSettings.getLinkeMonthly()));
        parametersSettings->setValue("land_use", QString::fromStdString(getKeyStringLandUse(radSettings.getLandUse())));
    parametersSettings->endGroup();
}

void Project::saveInterpolationParameters()
{
    parametersSettings->beginGroup("interpolation");
        parametersSettings->setValue("aggregationMethod", QString::fromStdString(getKeyStringAggregationMethod(interpolationSettings.getMeteoGridAggrMethod())));
        parametersSettings->setValue("algorithm", QString::fromStdString(getKeyStringInterpolationMethod(interpolationSettings.getInterpolationMethod())));
        parametersSettings->setValue("lapseRateCode", interpolationSettings.getUseLapseRateCode());
        parametersSettings->setValue("meteogrid_upscalefromdem", interpolationSettings.getMeteoGridUpscaleFromDem());
        parametersSettings->setValue("thermalInversion", interpolationSettings.getUseThermalInversion());
        parametersSettings->setValue("excludeStationsOutsideDEM", interpolationSettings.getUseExcludeStationsOutsideDEM());
        parametersSettings->setValue("topographicDistance", interpolationSettings.getUseTD());
        parametersSettings->setValue("localDetrending", interpolationSettings.getUseLocalDetrending());
        parametersSettings->setValue("topographicDistanceMaxMultiplier", QString::number(interpolationSettings.getTopoDist_maxKh()));
        parametersSettings->setValue("optimalDetrending", interpolationSettings.getUseBestDetrending());
        parametersSettings->setValue("multipleDetrending", interpolationSettings.getUseMultipleDetrending());
        parametersSettings->setValue("glocalDetrending", interpolationSettings.getUseGlocalDetrending());
        parametersSettings->setValue("doNotRetrend", interpolationSettings.getUseDoNotRetrend());
        parametersSettings->setValue("retrendOnly", interpolationSettings.getUseRetrendOnly());
        parametersSettings->setValue("useDewPoint", interpolationSettings.getUseDewPoint());
        parametersSettings->setValue("useInterpolationTemperatureForRH", interpolationSettings.getUseInterpolatedTForRH());
        parametersSettings->setValue("thermalInversion", interpolationSettings.getUseThermalInversion());
        parametersSettings->setValue("minRegressionR2", QString::number(double(interpolationSettings.getMinRegressionR2())));
        parametersSettings->setValue("min_points_local_detrending", QString::number(int(interpolationSettings.getMinPointsLocalDetrending())));
    parametersSettings->endGroup();


    saveProxies();

    parametersSettings->sync();
}

void Project::saveProxies()
{
    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        if (group.left(6) == "proxy_")
            parametersSettings->remove(group);
    }

    Crit3DProxy* myProxy;
    for (unsigned int i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        myProxy = interpolationSettings.getProxy(i);
        parametersSettings->beginGroup("proxy_" + QString::fromStdString(myProxy->getName()));
            parametersSettings->setValue("order", i+1);
            parametersSettings->setValue("active", interpolationSettings.getSelectedCombination().isProxyActive(i));
            parametersSettings->setValue("use_for_spatial_quality_control", myProxy->getForQualityControl());
            if (myProxy->getProxyTable() != "") parametersSettings->setValue("table", QString::fromStdString(myProxy->getProxyTable()));
            if (myProxy->getProxyField() != "") parametersSettings->setValue("field", QString::fromStdString(myProxy->getProxyField()));
            if (myProxy->getGridName() != "") parametersSettings->setValue("raster", getRelativePath(QString::fromStdString(myProxy->getGridName())));
            if (myProxy->getStdDevThreshold() != NODATA) parametersSettings->setValue("stddev_threshold", QString::number(myProxy->getStdDevThreshold()));
            if (!myProxy->getFittingParametersMin().empty()) parametersSettings->setValue("fitting_parameters_min", DoubleVectorToStringList(myProxy->getFittingParametersMin()));
            if (!myProxy->getFittingParametersMax().empty()) parametersSettings->setValue("fitting_parameters_max", DoubleVectorToStringList(myProxy->getFittingParametersMax()));
            if (myProxy->getFittingFirstGuess().size() > 0) parametersSettings->setValue("fitting_first_guess", IntVectorToStringList(myProxy->getFittingFirstGuess()));
            if (getProxyPragaName(myProxy->getName()) == proxyHeight && myProxy->getFittingFunctionName() != noFunction) parametersSettings->setValue("fitting_function", QString::fromStdString(getKeyStringElevationFunction(myProxy->getFittingFunctionName())));
        parametersSettings->endGroup();
    }
}

void Project::saveGenericParameters()
{
    parametersSettings->beginGroup("meteo");
        parametersSettings->setValue("min_percentage", QString::number(meteoSettings->getMinimumPercentage()));
        parametersSettings->setValue("prec_threshold", QString::number(meteoSettings->getRainfallThreshold()));
        parametersSettings->setValue("samani_coefficient", QString::number(meteoSettings->getTransSamaniCoefficient()));
        parametersSettings->setValue("thom_threshold", QString::number(meteoSettings->getThomThreshold()));
        parametersSettings->setValue("temperature_threshold", QString::number(meteoSettings->getTemperatureThreshold()));
        parametersSettings->setValue("wind_intensity_default", QString::number(meteoSettings->getWindIntensityDefault()));
        parametersSettings->setValue("hourly_intervals", QString::number(meteoSettings->getHourlyIntervals()));
        parametersSettings->setValue("compute_tavg", meteoSettings->getAutomaticTavg());
        parametersSettings->setValue("compute_et0hs", meteoSettings->getAutomaticET0HS());
    parametersSettings->endGroup();

    parametersSettings->beginGroup("quality");
        parametersSettings->setValue("reference_height", QString::number(quality->getReferenceHeight()));
        parametersSettings->setValue("delta_temperature_suspect", QString::number(quality->getDeltaTSuspect()));
        parametersSettings->setValue("delta_temperature_wrong", QString::number(quality->getDeltaTWrong()));
        parametersSettings->setValue("relhum_tolerance", QString::number(quality->getRelHumTolerance()));
        parametersSettings->setValue("water_table_maximum_depth", QString::number(quality->getWaterTableMaximumDepth()));
    parametersSettings->endGroup();

    parametersSettings->beginGroup("climate");

    parametersSettings->setValue("tmin", FloatVectorToStringList(climateParameters.tmin));
    parametersSettings->setValue("tmax", FloatVectorToStringList(climateParameters.tmax));
    parametersSettings->setValue("tdmin", FloatVectorToStringList(climateParameters.tdmin));
    parametersSettings->setValue("tdmax", FloatVectorToStringList(climateParameters.tdmax));
    parametersSettings->setValue("tmin_lapserate", FloatVectorToStringList(climateParameters.tminLapseRate));
    parametersSettings->setValue("tmax_lapserate", FloatVectorToStringList(climateParameters.tmaxLapseRate));
    parametersSettings->setValue("tdmin_lapserate", FloatVectorToStringList(climateParameters.tdMinLapseRate));
    parametersSettings->setValue("tdmax_lapserate", FloatVectorToStringList(climateParameters.tdMaxLapseRate));

    parametersSettings->endGroup();

    parametersSettings->sync();
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
        _projectName = description;
    else
        _projectName = name_;

    // folder
    QString myPath = path_ + "/" + name_ + "/";
    if (! QDir(myPath).exists())
        QDir().mkdir(myPath);

    _projectPath = myPath;

    // settings
    delete projectSettings;
    projectSettings = new QSettings(_projectPath + name_ + ".ini", QSettings::IniFormat);

    // parametersSettings
    delete parametersSettings;
    parametersFileName = _projectPath + PATH_SETTINGS + "parameters.ini";
    parametersSettings = new QSettings(parametersFileName, QSettings::IniFormat);

    saveProject();
}


bool Project::createDefaultProject(const QString &fileName)
{
    QString dataPath;
    if (! searchDefaultPath(dataPath)) return false;

    QSettings* defaultSettings = new QSettings(fileName, QSettings::IniFormat);

    defaultSettings->beginGroup("project");
        defaultSettings->setValue("path", dataPath);
        defaultSettings->setValue("name", "default");
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
        errorString = "Parameters loading failed.\n" + errorString;
        return false;
    }

    if (demFileName != "")
        if (! loadDEM(demFileName)) return false;

    if (dbPointsFileName != "")
        if (! loadMeteoPointsDB(dbPointsFileName))
        {
            errorString = "load Meteo Points DB failed:\n" + dbPointsFileName;
            errorType = ERROR_DBPOINT;
            logError();
            return false;
        }

    if (dbAggregationFileName != "")
    {
        if (! loadAggregationdDB(_projectPath + "/" + dbAggregationFileName))
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
        logError(ERROR_STR_MISSING_GRID);
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
    \details default format:
    DATE(yyyy-mm-dd), HOUR, TAVG, PREC, RHAVG, RAD, W_SCAL_INT
    - the filename must be equal to the pointcode
    - header is mandatory
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
    int step = setProgressBar("Load hourly data..", fileList.count());
    int nrLoaded = 0;
    for (int i=0; i < fileList.count(); i++)
    {
        if (i % step == 0)
            updateProgressBar(i);

        QString logStr = "";
        QString fileNameComplete = filePath + fileList[i];

        if (meteoPointsDbHandler->importHourlyMeteoData(fileNameComplete, deletePreviousData, logStr))
        {
            nrLoaded++;
            if (! logStr.isEmpty())
            {
                updateProgressBarText(logStr);
            }
        }
        else
        {
            logError(logStr);
        }
    }

    closeProgressBar();

    if (nrLoaded > 0)
    {
        // reload meteoPoint data are changed
        loadMeteoPointsDB(dbPointsFileName);
    }
}


void Project::showMeteoWidgetPoint(std::string idMeteoPoint, std::string namePoint, std::string dataset,
                                   double altitude, std::string lapseRateCode, bool isAppend)
{
    // check
    if (isAppend)
    {
        if (meteoWidgetPointList.isEmpty())
        {
            isAppend = false;
        }
        else
        {
            int lastIndex = meteoWidgetPointList.size() - 1;
            if (meteoWidgetPointList[lastIndex]->isAlreadyPresent(idMeteoPoint, dataset))
            {
                logWarning("This meteo point is already present.");
                return;
            }
        }
    }

    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    QDateTime firstHourly = meteoPointsDbHandler->getFirstDate(hourly, idMeteoPoint);
    QDateTime lastHourly = meteoPointsDbHandler->getLastDate(hourly, idMeteoPoint);
    bool hasHourlyData = !(firstHourly.isNull() || lastHourly.isNull());

    if (! hasDailyData && ! hasHourlyData)
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

    Crit3DMeteoPoint mp;
    mp.setId(idMeteoPoint);
    mp.setName(namePoint);
    mp.setLapseRateCode(lapseRateCode);
    mp.setDataset(dataset);
    mp.point.z = altitude;

    if (isAppend)
    {
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), mp);
        meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), mp);
        meteoWidgetPointList[meteoWidgetPointList.size()-1]->drawMeteoPoint(mp, isAppend);
    }
    else
    {
        bool isGrid = false;
        Crit3DMeteoWidget* meteoWidgetPoint = new Crit3DMeteoWidget(isGrid, _projectPath, meteoSettings);
        int meteoWidgetId;
        if (! meteoWidgetPointList.isEmpty())
        {
            meteoWidgetId = meteoWidgetPointList[meteoWidgetPointList.size()-1]->getMeteoWidgetID()+1;
        }
        else
        {
            meteoWidgetId = 0;
        }
        meteoWidgetPoint->setMeteoWidgetID(meteoWidgetId);
        meteoWidgetPoint->setAllMeteoPointsPointer(meteoPoints, nrMeteoPoints);
        meteoWidgetPointList.append(meteoWidgetPoint);
        QObject::connect(meteoWidgetPoint, SIGNAL(closeWidgetPoint(int)), this, SLOT(deleteMeteoWidgetPoint(int)));
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), mp);
        meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), mp);

        if (hasDailyData)
        {
            meteoWidgetPoint->setDailyRange(firstDaily, lastDaily);
        }
        if (hasHourlyData)
        {
            meteoWidgetPoint->setHourlyRange(firstHourly.date(), lastHourly.date());
        }

        meteoWidgetPoint->setCurrentDate(_currentDate);
        meteoWidgetPoint->drawMeteoPoint(mp, isAppend);
    }

    closeLogInfo();
}


void Project::showMeteoWidgetGrid(const std::string &idCell, const std::string &dataset, bool isAppend)
{
    // check
    if (isAppend)
    {
        if (meteoWidgetGridList.isEmpty())
        {
            isAppend = false;
        }
        else
        {
            int lastIndex = meteoWidgetGridList.size() - 1;
            if (meteoWidgetGridList[lastIndex]->isAlreadyPresent(idCell, dataset))
            {
                logWarning("This grid cell is already present.");
                return;
            }
        }
    }

    QDate firstDate = meteoGridDbHandler->firstDate();
    QDate lastDate = meteoGridDbHandler->lastDate();
    QDate firstMonthlyDate = meteoGridDbHandler->getFirstMonthlytDate();
    QDate lastMonthlyDate = meteoGridDbHandler->getLastMonthlyDate();

    QDateTime firstDateTime, lastDateTime;
    if (meteoGridDbHandler->getFirstHourlyDate().isValid())
    {
        firstDateTime = QDateTime(meteoGridDbHandler->getFirstHourlyDate(), QTime(1,0), Qt::UTC);
    }
    if (meteoGridDbHandler->getLastHourlyDate().isValid())
    {
        lastDateTime = QDateTime(meteoGridDbHandler->getLastHourlyDate().addDays(1), QTime(0,0), Qt::UTC);
    }

    if (meteoGridDbHandler->gridStructure().isEnsemble())
    {
        if (isAppend)
        {
            isAppend = false;
            logWarning("Meteo grid is ensemble: append mode is not possible.\nA new meteo widget will be open.");
        }
    }

    logInfoGUI("Loading data...\n");

    if (isAppend)
    {
        if (! meteoGridDbHandler->gridStructure().isFixedFields())
        {
            if (meteoGridDbHandler->isDaily())
            {
                meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(idCell), firstDate, lastDate);
            }
            if (meteoGridDbHandler->isHourly())
            {
                logInfoGUI("Loading hourly data...\n");
                meteoGridDbHandler->loadGridHourlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell),
                                                       firstDateTime, lastDateTime, errorString);
            }
            if (meteoGridDbHandler->isMonthly())
            {
                meteoGridDbHandler->loadGridMonthlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell), firstMonthlyDate, lastMonthlyDate, errorString);
            }
        }
        else
        {
            if (meteoGridDbHandler->isDaily())
            {
                meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(idCell), firstDate, lastDate);
            }
            if (meteoGridDbHandler->isHourly())
            {
                logInfoGUI("Loading hourly data...\n");
                meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
            }
            if (meteoGridDbHandler->isMonthly())
            {
                meteoGridDbHandler->loadGridMonthlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell), firstMonthlyDate, lastMonthlyDate, errorString);
            }
        }
        closeLogInfo();

        if(meteoWidgetGridList[meteoWidgetGridList.size()-1]->getIsEnsemble())
        {
            // an ensemble grid is already open, append on that
            // The new one is not ensemble (otherwise append mode is not possible)
            meteoWidgetGridList[meteoWidgetGridList.size()-1]->setIsEnsemble(false);
        }

        unsigned row, col;
        if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row, &col, idCell))
        {
            meteoWidgetGridList[meteoWidgetGridList.size()-1]->drawMeteoPoint(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col), isAppend);
        }
        return;
    }
    else
    {
        int meteoWidgetId;
        bool isGrid = true;
        Crit3DMeteoWidget* meteoWidgetGrid = new Crit3DMeteoWidget(isGrid, _projectPath, meteoSettings);
        if (! meteoWidgetGridList.isEmpty())
        {
             meteoWidgetId = meteoWidgetGridList[meteoWidgetGridList.size()-1]->getMeteoWidgetID()+1;
        }
        else
        {
            meteoWidgetId = 0;
        }

        meteoWidgetGrid->setMeteoWidgetID(meteoWidgetId);
        meteoWidgetGrid->setCurrentDate(_currentDate);
        meteoWidgetGridList.append(meteoWidgetGrid);

        QObject::connect(meteoWidgetGrid, SIGNAL(closeWidgetGrid(int)), this, SLOT(deleteMeteoWidgetGrid(int)));

        if (meteoGridDbHandler->gridStructure().isEnsemble())
        {
            meteoWidgetGrid->setIsEnsemble(true);
            meteoWidgetGrid->setNrMembers(meteoGridDbHandler->gridStructure().nrMembers());

            unsigned row, col;
            int nMembers = meteoGridDbHandler->gridStructure().nrMembers();
            if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row, &col, idCell))
            {
                if (meteoGridDbHandler->isDaily())
                {
                    meteoWidgetGrid->setDailyRange(firstDate, lastDate);
                }
                if (meteoGridDbHandler->isHourly())
                {
                    meteoWidgetGrid->setHourlyRange(firstDateTime.date(), lastDateTime.date());
                }
            }
            else
            {
                closeLogInfo();
                return;
            }

            for (int i = 1; i <= nMembers; i++)
            {
                meteoGridDbHandler->loadGridDailyDataEnsemble(errorString, QString::fromStdString(idCell), i, firstDate, lastDate);
                meteoWidgetGrid->addMeteoPointsEnsemble(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col));
            }
            meteoWidgetGrid->drawEnsemble();
            closeLogInfo();
        }
        else
        {
            if (! meteoGridDbHandler->gridStructure().isFixedFields())
            {
                if (meteoGridDbHandler->isDaily())
                {
                    meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(idCell), firstDate, lastDate);
                }
                if (meteoGridDbHandler->isHourly())
                {
                    logInfoGUI("Loading hourly data...\n");
                    meteoGridDbHandler->loadGridHourlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell),
                                                           firstDateTime, lastDateTime, errorString);
                }
                if (meteoGridDbHandler->isMonthly())
                {
                    meteoGridDbHandler->loadGridMonthlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell), firstMonthlyDate, lastMonthlyDate, errorString);
                }
            }
            else
            {
                if (meteoGridDbHandler->isDaily())
                {
                    meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(idCell), firstDate, lastDate);
                }
                if (meteoGridDbHandler->isHourly())
                {
                    logInfoGUI("Loading hourly data...\n");
                    meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(idCell), firstDateTime, lastDateTime);
                }
                if (meteoGridDbHandler->isMonthly())
                {
                    meteoGridDbHandler->loadGridMonthlyData(meteoGridDbHandler->db(), QString::fromStdString(idCell), firstMonthlyDate, lastMonthlyDate, errorString);
                }
            }
            closeLogInfo();

            unsigned row, col;
            if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row,&col,idCell))
            {
                if (meteoGridDbHandler->isDaily())
                {
                    meteoWidgetGrid->setDailyRange(firstDate, lastDate);
                }
                if (meteoGridDbHandler->isHourly())
                {
                    meteoWidgetGrid->setHourlyRange(firstDateTime.date(), lastDateTime.date());
                }

                meteoWidgetGrid->drawMeteoPoint(meteoGridDbHandler->meteoGrid()->meteoPoint(row,col), isAppend);
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


void Project::showProxyGraph(int macroAreaNumber)
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
        proxyWidget = new Crit3DProxyWidget(interpolationSettings, meteoPoints, nrMeteoPoints, _currentFrequency,
                                            _currentDate, _currentHour, quality, qualityInterpolationSettings,
                                            meteoSettings, &climateParameters, checkSpatialQuality, macroAreaNumber);
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
        proxyWidget = new Crit3DProxyWidget(interpolationSettings, meteoPointsSelected, nSelected, _currentFrequency,
                                            _currentDate, _currentHour, quality, qualityInterpolationSettings,
                                            meteoSettings, &climateParameters, checkSpatialQuality, macroAreaNumber);
    }
    QObject::connect(proxyWidget, SIGNAL(closeProxyWidget()), this, SLOT(deleteProxyWidget()));
    return;
}


void Project::showLocalProxyGraph(gis::Crit3DGeoPoint myPoint)
{
    gis::Crit3DUtmPoint myUtm;
    gis::getUtmFromLatLon(this->gisSettings.utmZone, myPoint, &myUtm);
    double myZGrid = NODATA;
    double myZDEM = NODATA;

    if (meteoGridLoaded)
    {
        //TODO
    }
    if (DEM.isLoaded)
    {
        int row, col;
        DEM.getRowCol(myUtm.x, myUtm.y, row, col);
        myZDEM = DEM.value[row][col];
    }

    if (! checkGlocal(false))
        return;

    localProxyWidget = new Crit3DLocalProxyWidget(myUtm.x, myUtm.y, myZDEM, myZGrid, this->gisSettings, interpolationSettings,
                                                  meteoPoints, nrMeteoPoints, _currentVariable, _currentFrequency,
                                                  _currentDate, _currentHour, quality, qualityInterpolationSettings,
                                                  meteoSettings, &climateParameters, checkSpatialQuality);
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
        logError("Failed to activate/deactivate selected points:\n" + meteoPointsDbHandler->getErrorString());
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
        logError("Failed to activate/deactivate point list:\n" + meteoPointsDbHandler->getErrorString());
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

    bool isSelect = false;
    DialogSelectionMeteoPoint dialogPointSelection(isActive, isSelect, meteoPointsDbHandler);
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
            logError("Failed to activate/deactivate points selected:\n" + meteoPointsDbHandler->getErrorString());
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

bool Project::setMarkedPointsOfMacroArea(int areaNumber, bool viewNotActivePoints)
{
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        meteoPoints[i].marked = false;
    }

    std::vector <int> pointList;
    if (areaNumber < 0 || areaNumber >= interpolationSettings.getMacroAreasSize())
    {
        logError("Invalid macro area number.");
        return false;
    }

    pointList = interpolationSettings.getMacroArea(areaNumber).getMeteoPoints();

    for (int j = 0; j < (int)pointList.size(); j++)
    {
        if (meteoPoints[pointList[j]].active || viewNotActivePoints)
            meteoPoints[pointList[j]].marked = true;
    }

    return true;
}


bool Project::deleteMeteoPoints(const QList<QString>& pointList)
{
    logInfoGUI("Deleting points...");
        bool isOk = meteoPointsDbHandler->deleteAllPointsFromIdList(pointList);
    closeLogInfo();

    if (! isOk)
    {
        logError("Failed to delete points:" + meteoPointsDbHandler->getErrorString());
        return false;
    }

    if (! meteoPointsDbHandler->getErrorString().isEmpty())
    {
        logError("WARNING: " + meteoPointsDbHandler->getErrorString());
    }

    // reload meteoPoint, point properties table is changed
    return loadMeteoPointsDB(dbPointsFileName);
}


bool Project::deleteMeteoPointsData(const QList<QString>& pointList)
{
    if (pointList.isEmpty())
    {
        logError("No data to delete.");
        return true;
    }

    DialogPointDeleteData dialogPointDelete(_currentDate);
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
            if (! meteoPointsDbHandler->deleteData(pointList[i], daily, startDate, endDate))
            {
                closeProgressBar();
                return false;
            }
        }
        else
        {
            if (! dailyVarList.isEmpty())
            {
                if (! meteoPointsDbHandler->deleteData(pointList[i], daily, dailyVarList, startDate, endDate))
                {
                    closeProgressBar();
                    return false;
                }
            }
        }
        if (allHourly)
        {
            if (! meteoPointsDbHandler->deleteData(pointList[i], hourly, startDate, endDate))
            {
                closeProgressBar();
                return false;
            }
        }
        else
        {
            if (! hourlyVarList.isEmpty())
            {
                if (! meteoPointsDbHandler->deleteData(pointList[i], hourly, hourlyVarList, startDate, endDate))
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

void Project::setComputeOnlyPoints(bool value)
{
    _computeOnlyPoints = value;
}

bool Project::getComputeOnlyPoints() const
{
    return _computeOnlyPoints;
}

bool Project::exportMeteoGridToRasterFlt(QString fileName, double cellSize)
{
    if (! meteoGridLoaded || meteoGridDbHandler == nullptr)
    {
        logInfoGUI("No meteogrid open");
        return false;
    }

    if (fileName == "")
    {
        logInfoGUI("No filename provided");
        return false;
    }

    if (cellSize < 0)
    {
        logInfoGUI("Incorrect cell size");
        return false;
    }

    gis::Crit3DRasterGrid myGrid;
    if (! meteoGridDbHandler->MeteoGridToRasterFlt(cellSize, gisSettings, myGrid))
    {
        errorString = "initializeGrid failed";
        return false;
    }

    std::string myError = errorString.toStdString();
    QString fileWithoutExtension = QFileInfo(fileName).absolutePath() + QDir::separator() + QFileInfo(fileName).baseName();
    if (! gis::writeEsriGrid(fileWithoutExtension.toStdString(), &myGrid, myError))
    {
        errorString = QString::fromStdString(myError);
        return false;
    }

    return true;
}


bool Project::loadAndExportMeteoGridToRasterFlt(QString outPath, double cellSize, meteoVariable myVar, QDate dateIni, QDate dateFin)
{
    if (! meteoGridLoaded || meteoGridDbHandler == nullptr)
    {
        logInfoGUI("No meteogrid open");
        return false;
    }

    if (outPath == "")
    {
        logInfoGUI("No filename provided");
        return false;
    }

    if (cellSize < 0)
    {
        logInfoGUI("Incorrect cell size");
        return false;
    }

    if (myVar == noMeteoVar)
    {
        logInfoGUI("No meteo variable");
        return false;
    }

    if (! dateIni.isValid() || ! dateFin.isValid())
    {
        logInfoGUI("Invalid dates");
        return false;
    }

    QDate date_ = dateIni;
    gis::Crit3DRasterGrid myGrid;
    QString fileName;

    if (! loadMeteoGridDailyData(dateIni, dateFin, false))
        return false;

   while (date_ <= dateFin)
    {

        meteoGridDbHandler->meteoGrid()->fillCurrentDailyValue(getCrit3DDate(date_), myVar, meteoSettings);
        meteoGridDbHandler->meteoGrid()->fillMeteoRaster();

        if (! meteoGridDbHandler->MeteoGridToRasterFlt(cellSize, gisSettings, myGrid))
            return false;

        fileName = outPath + QDir::separator() + QString::fromStdString(getVariableString(myVar)) + "_" + date_.toString("yyyyMMdd") + ".flt";
        if (! exportMeteoGridToRasterFlt(fileName, cellSize))
            return false;

        date_ = date_.addDays(1);
    }

    return true;
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

            if (value != NO_ACTIVE && value != NODATA)
            {
                int newRow = row;
                if (! meteoGridDbHandler->meteoGrid()->gridStructure().isUTM())
                    newRow = meteoGridDbHandler->meteoGrid()->gridStructure().nrRow() - 1 - row;

                std::string id = meteoGridDbHandler->meteoGrid()->meteoPoints()[newRow][col]->id;
                std::string name = meteoGridDbHandler->meteoGrid()->meteoPoints()[newRow][col]->name;

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


/*!
 * \brief ExportDailyDataCsv
 * export daily meteo data to csv files
 * \param isTPrec           save only variables: Tmin, Tmax, Tavg, Prec
 * \param idListFileName    filename of ID point list (list by columns)
 * if idListFile == ""      save ALL active meteo points
 * \param outputPath        path for output files
 * \return true on success, false otherwise
 */
bool Project::exportMeteoPointsDailyDataCsv(bool isTPrec, QDate firstDate, QDate lastDate, QString idListFileName, QString outputPath)
{
    errorString = "";
    if (! meteoPointsLoaded)
    {
        errorString = "No meteo points loaded.";
        return false;
    }

    // check output path
    QDir outDir(outputPath);
    if (! outDir.exists())
    {
        if (! outDir.mkpath(outputPath))
        {
            errorString = "Wrong outputPath, unable to create this directory: " + outputPath;
            return false;
        }
    }
    outputPath = outDir.absolutePath();

    // check ID list
    bool isList = (idListFileName != "");
    QList<QString> idList;
    if (isList)
    {
        if (! QFile::exists(idListFileName))
        {
            errorString = "The ID list file does not exist: " + idListFileName;
            return false;
        }

        idList = readListSingleColumn(idListFileName, errorString);
        if (errorString != "")
            return false;

        if (idList.size() == 0)
        {
            errorString = "The ID list file is empty: " + idListFileName;
            return false;
        }
    }

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        QString id = QString::fromStdString(meteoPoints[i].id);
        bool checkId = false;
        if (! isList && meteoPoints[i].active)
            checkId = true;
        if ( isList && idList.contains(id))
            checkId = true;

        if (checkId)
        {
            // read data
            if (! meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), meteoPoints[i]))
            {
                errorString = "Error in reading point id: " + id;
                return false;
            }

            // create csv file
            QString csvFileName = outputPath + "/" + id + ".csv";
            QFile outputFile(csvFileName);
            bool isOk = outputFile.open(QIODevice::WriteOnly | QFile::Truncate);
            if (! isOk)
            {
                errorString = "Open output CSV failed: " + csvFileName;
                return false;
            }

            // set header
            QTextStream out(&outputFile);
            out << "Date, Tmin (C), Tmax (C), Tavg (C), Prec (mm)";
            if (isTPrec)
                out << "\n";
            else
                out << "RHmin (%), RHmax (%), RHavg (%), Windspeed (m/s), Rad (MJ)\n";

            // save data
            QDate currentDate = firstDate;
            while (currentDate <= lastDate)
            {
                Crit3DDate myDate = getCrit3DDate(currentDate);

                float tmin = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirTemperatureMin);
                QString tminStr = "";
                if (tmin != NODATA)
                    tminStr = QString::number(tmin);

                float tmax = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirTemperatureMax);
                QString tmaxStr = "";
                if (tmax != NODATA)
                    tmaxStr = QString::number(tmax);

                float tavg = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
                QString tavgStr = "";
                if (tavg != NODATA)
                    tavgStr = QString::number(tavg);

                float prec = meteoPoints[i].getMeteoPointValueD(myDate, dailyPrecipitation);
                QString precStr = "";
                if (prec != NODATA)
                    precStr = QString::number(prec);

                out << currentDate.toString("yyyy-MM-dd") << "," << tminStr << "," << tmaxStr << "," << tavgStr << "," << precStr;
                if (isTPrec)
                    out << "\n";
                else
                {
                    float rhmin = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirRelHumidityMin);
                    QString rhminStr = "";
                    if (rhmin != NODATA)
                        rhminStr = QString::number(rhmin);

                    float rhmax = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirRelHumidityMax);
                    QString rhmaxStr = "";
                    if (rhmax != NODATA)
                        rhmaxStr = QString::number(rhmax);

                    float rhavg = meteoPoints[i].getMeteoPointValueD(myDate, dailyAirRelHumidityAvg);
                    QString rhavgStr = "";
                    if (rhavg != NODATA)
                        rhavgStr = QString::number(rhavg);

                    float wspeed = meteoPoints[i].getMeteoPointValueD(myDate, dailyWindScalarIntensityAvg);
                    QString wspeedStr = "";
                    if (wspeed != NODATA)
                        wspeedStr = QString::number(wspeed);

                    float rad = meteoPoints[i].getMeteoPointValueD(myDate, dailyGlobalRadiation);
                    QString radStr = "";
                    if (rad != NODATA)
                        radStr = QString::number(rad);

                    out << "," << rhminStr << "," << rhmaxStr << "," << rhavgStr << "," << wspeedStr << "," << radStr << "\n";
                }

                currentDate = currentDate.addDays(1);
            }

            outputFile.close();
        }
    }

    return true;
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
    if (_verboseStdoutLogging) {
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
        if (_formLog == nullptr)
        {
            _formLog = new FormInfo();
        }
        _formLog->showInfo(myStr);
        qApp->processEvents();
    }
    else
    {
        logInfo(myStr);
    }
}


void Project::closeLogInfo()
{
    if ((modality == MODE_GUI) && (_formLog != nullptr))
    {
        _formLog->close();
    }
}


void Project::logError(QString myStr)
{
    errorString = myStr;
    logError();
}


void Project::logWarning(QString myStr)
{
    errorString = myStr;
    logWarning();
}


void Project::logWarning()
{
    if (logFile.is_open())
    {
        logFile << "WARNING! " << errorString.toStdString() << std::endl;
    }

    if (modality == MODE_GUI)
    {
        QMessageBox::warning(nullptr, "WARNING", errorString);
    }
    else
    {
        std::cout << "WARNING! " << errorString.toStdString() << std::endl;
    }
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
        if (_formLog == nullptr)
        {
            _formLog = new FormInfo();
        }
        return _formLog->start(myStr, nrValues);
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
        if (_formLog != nullptr)
        {
            _formLog->setValue(value);
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
        if (_formLog != nullptr)
        {
            _formLog->setText(myStr);
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
        if (_formLog != nullptr)
        {
            _formLog->close();
        }
    }
    else
    {
        std::cout << std::endl;
    }
}


bool Project::waterTableImportLocation(const QString &csvFileName)
{
    if (logFileName == "")
    {
        setLogFile("waterTableLog.txt");
    }

    int wrongLines = 0;
    if (! loadWaterTableLocationCsv(csvFileName, wellPoints, gisSettings, errorString, wrongLines))
    {
        logError(errorString);
        return false;
    }

    if (wrongLines > 0)
    {
        logInfo(errorString);
        logWarning(QString::number(wrongLines) + " wrong lines of data were not loaded\nSee the log file for more information:\n" + logFileName);
    }

    errorString = "";
    return true;
}


bool Project::waterTableImportDepths(const QString &csvDepthsFileName)
{
    int wrongLines = 0;
    if (! loadWaterTableDepthCsv(csvDepthsFileName, wellPoints, quality->getWaterTableMaximumDepth(), errorString, wrongLines))
    {
        logError(errorString);
        return false;
    }

    if (wrongLines > 0)
    {
        logInfo(errorString);
        logWarning(QString::number(wrongLines) + " wrong lines of data were not loaded\nSee the log file for more information:\n" + logFileName);
    }

    errorString = "";
    return true;
}


bool Project::waterTableComputeSingleWell(int indexWell)
{
    if (indexWell == NODATA)
        return false;

    // sono necessari 24 mesi di dati meteo precedenti il primo dato osservato di falda
    wellPoints[indexWell].updateDates();
    QDate firstMeteoDate = wellPoints[indexWell].getFirstObsDate().addDays(-730);
    double wellUtmX = wellPoints[indexWell].getUtmX();
    double wellUtmY = wellPoints[indexWell].getUtmY();
    Crit3DMeteoPoint linkedMeteoPoint;

    bool isMeteoGridLoaded;
    if (this->meteoGridDbHandler != nullptr)
    {
        isMeteoGridLoaded = true;
    }
    else if (meteoPoints != nullptr)
    {
        isMeteoGridLoaded = false;
    }
    else
    {
        logError(ERROR_STR_MISSING_POINT_GRID);
        return false;
    }

    QString idStr = wellPoints[indexWell].getId();
    if (! waterTableAssignNearestMeteoPoint(isMeteoGridLoaded, wellUtmX, wellUtmY, firstMeteoDate, &linkedMeteoPoint))
    {
        logError("Missing weather data near well: " + idStr);
        return false;
    }
    if (linkedMeteoPoint.nrObsDataDaysD == 0)
    {
        logError("Missing weather data near well: " + idStr);
        return false;
    }

    std::vector<float> inputTMin;
    std::vector<float> inputTMax;
    std::vector<float> inputPrec;

    for (int i = 0; i < linkedMeteoPoint.nrObsDataDaysD; i++)
    {
        Crit3DDate myDate = linkedMeteoPoint.getFirstDailyData().addDays(i);
        float Tmin = linkedMeteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        float Tmax = linkedMeteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        float prec = linkedMeteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation);
        inputTMin.push_back(Tmin);
        inputTMax.push_back(Tmax);
        inputPrec.push_back(prec);
    }

    WaterTable waterTable(inputTMin, inputTMax, inputPrec, getQDate(linkedMeteoPoint.getFirstDailyData()),
                          getQDate(linkedMeteoPoint.getLastDailyData()), *meteoSettings);

    waterTable.computeWaterTableParameters(wellPoints[indexWell], 5);

    waterTable.computeWaterTableSeries();        // prepare series to show

    waterTableList.push_back(waterTable);
    return true;
}


void Project::waterTableShowSingleWell(const WaterTable &waterTable, const QString &idWell)
{
    DialogSummary* dialogResult = new DialogSummary(waterTable);   // show results
    dialogResult->show();
    WaterTableWidget* chartResult = new WaterTableWidget(idWell, waterTable, quality->getWaterTableMaximumDepth());
    chartResult->show();
    return;
}


bool Project::waterTableAssignNearestMeteoPoint(bool isMeteoGridLoaded, double wellUtmX, double wellUtmY, QDate firstMeteoDate, Crit3DMeteoPoint* linkedMeteoPoint)
{
    float minimumDistance = NODATA;
    bool isMeteoPointFound = false;

    if (isMeteoGridLoaded)
    {
        std::string assignNearestId;
        unsigned int assignNearestRow;
        unsigned int assignNearestCol;
        int zoneNumber;
        QDate lastDate = this->meteoGridDbHandler->getLastDailyDate();
        QDate firstDate = std::max(firstMeteoDate, this->meteoGridDbHandler->getFirstDailyDate());
        for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
        {
            for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
            {
                if (meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->active)
                {
                    double utmX = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->point.utm.x;
                    double utmY = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->point.utm.y;
                    if (utmX == NODATA || utmY == NODATA)
                    {
                        double lat = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->latitude;
                        double lon = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->longitude;
                        gis::latLonToUtm(lat, lon, &utmX, &utmY, &zoneNumber);
                    }
                    float myDistance = gis::computeDistance(wellUtmX, wellUtmY, utmX, utmY);
                    if (myDistance < MAXWELLDISTANCE )
                    {
                        if (myDistance < minimumDistance || minimumDistance == NODATA)
                        {
                            minimumDistance = myDistance;
                            assignNearestId = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->id;
                            assignNearestRow = row;
                            assignNearestCol = col;
                            isMeteoPointFound = true;
                        }
                    }
                }
            }
        }
        if (isMeteoPointFound)
        {
            meteoGridDbHandler->loadGridDailyMeteoPrec(errorString, QString::fromStdString(assignNearestId), firstDate, lastDate);
            if (! waterTableAssignMeteoData(meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol), firstDate))
            {
                return false;
            }
            else
            {
                linkedMeteoPoint->id = meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol)->id;
                linkedMeteoPoint->name = meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol)->name;
                linkedMeteoPoint->latitude = meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol)->latitude;
                linkedMeteoPoint->nrObsDataDaysD = meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol)->nrObsDataDaysD;
                linkedMeteoPoint->obsDataD = meteoGridDbHandler->meteoGrid()->meteoPointPointer(assignNearestRow, assignNearestCol)->obsDataD;
            }
        }
    }
    else
    {
        int assignNearestIndex;
        QDate lastDate = meteoPointsDbHandler->getLastDate(daily).date();
        for (int i = 0; i < nrMeteoPoints; i++)
        {

            double utmX = meteoPoints[i].point.utm.x;
            double utmY = meteoPoints[i].point.utm.y;
            float myDistance = gis::computeDistance(wellUtmX, wellUtmY, utmX, utmY);
            if (myDistance < MAXWELLDISTANCE )
            {
                if (myDistance < minimumDistance || minimumDistance == NODATA)
                {
                    meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstMeteoDate), getCrit3DDate(lastDate), meteoPoints[i]);
                    if (waterTableAssignMeteoData(&meteoPoints[i], firstMeteoDate))
                    {
                        minimumDistance = myDistance;
                        isMeteoPointFound = true;
                        assignNearestIndex = i;
                    }
                }
            }
        }
        if (isMeteoPointFound)
        {
            linkedMeteoPoint->id = meteoPoints[assignNearestIndex].id;
            linkedMeteoPoint->name = meteoPoints[assignNearestIndex].name;
            linkedMeteoPoint->latitude = meteoPoints[assignNearestIndex].latitude;
            linkedMeteoPoint->nrObsDataDaysD = meteoPoints[assignNearestIndex].nrObsDataDaysD;
            linkedMeteoPoint->obsDataD = meteoPoints[assignNearestIndex].obsDataD;
        }
    }

    return isMeteoPointFound;
}


bool Project::waterTableAssignMeteoData(Crit3DMeteoPoint* linkedMeteoPoint, QDate firstMeteoDate)
{
    QDate lastMeteoDate;
    lastMeteoDate.setDate(linkedMeteoPoint->getLastDailyData().year, linkedMeteoPoint->getLastDailyData().month, linkedMeteoPoint->getLastDailyData().day); // ultimo dato disponibile
    float precPerc = linkedMeteoPoint->getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyPrecipitation);
    float tMinPerc = linkedMeteoPoint->getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMin);
    float tMaxPerc = linkedMeteoPoint->getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMax);

    float minPercentage = meteoSettings->getMinimumPercentage();
    if (precPerc > minPercentage/100 && tMinPerc > minPercentage/100 && tMaxPerc > minPercentage/100)
    {
        return true;
    }
    else
    {
        errorString = "Not enough meteo data to analyze watertable period. Try to decrease the required percentage";
        return false;
    }
}


bool Project::assignAltitudeToAggregationPoints()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    if (aggregationDbHandler == nullptr)
    {
        errorString = "Open or create a new aggregation database before.";
        return false;
    }

    if (meteoPointsLoaded)
    {
        errorString = "Close Meteo Points db before execute this operation.";
        return false;
    }

    // check aggregation raster
    QString rasterName;
    if (! aggregationDbHandler->getRasterName(&rasterName))
    {
        errorString = "Missing raster name inside aggregation db.";
        return false;
    }

    QString rasterFileName = aggregationPath + "/" + rasterName;
    QFileInfo rasterFileFltInfo(rasterFileName + ".flt");
    QFileInfo rasterFileHdrInfo(rasterFileName + ".hdr");
    if (! rasterFileFltInfo.exists() || ! rasterFileHdrInfo.exists())
    {
        errorString = "Raster file does not exist: " + rasterFileName;
        return false;
    }

    // load aggregation db as meteo points db
    if (! loadMeteoPointsDB(aggregationDbHandler->name()) )
    {
        errorString = "Error in load aggregation points: " + errorString;
        return false;
    }

    // load aggregation raster
    std::string errorStr = "";
    std::string fileNameStdStr = rasterFileName.toStdString() + ".flt";
    gis::Crit3DRasterGrid *aggregationRaster;
    aggregationRaster = new(gis::Crit3DRasterGrid);
    if (! gis::openRaster(fileNameStdStr, aggregationRaster, gisSettings.utmZone, errorStr))
    {
        errorString = "Open raster failed: " + QString::fromStdString(errorStr);
        return false;
    }

    // resample aggregation DEM
    gis::Crit3DRasterGrid *aggregationDEM;
    aggregationDEM = new(gis::Crit3DRasterGrid);
    gis::resampleGrid(DEM, aggregationDEM, aggregationRaster->header, aggrAverage, 0.1f);

    setProgressBar("Compute altitude..", nrMeteoPoints);

    // compute average altitude from aggregation DEM
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        QString idStr = QString::fromStdString(meteoPoints[i].id);
        QList<QString> idList = idStr.split('_');
        float zoneNr = idList[0].toFloat();

        std::vector<float> values;
        for (int row = 0; row < aggregationRaster->header->nrRows; row++)
        {
            for (int col = 0; col < aggregationRaster->header->nrCols; col++)
            {
                if (isEqual(aggregationRaster->value[row][col], zoneNr))
                {
                    float z = aggregationDEM->value[row][col];
                    if (! isEqual(z, aggregationDEM->header->flag))
                    {
                        values.push_back(z);
                    }
                }
            }
        }

        // update point properties
        float altitude = statistics::mean(values);
        QString query = QString("UPDATE point_properties SET altitude = %1 WHERE id_point = '%2'").arg(altitude).arg(idStr);
        aggregationDbHandler->db().exec(query);

        updateProgressBar(i);
    }

    closeMeteoPointsDB();
    return true;
}


bool Project::assignAltitudeToMeteoPoints(double boundarySize)
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    if (! meteoPointsLoaded)
    {
        errorString = ERROR_STR_MISSING_DB;
        return false;
    }

    int stepCells = floor(boundarySize / DEM.header->cellSize * 0.5);
    int row, col;
    std::vector<float> values;
    QSqlDatabase meteoPointsDb = meteoPointsDbHandler->getDb();

    // compute average altitude from DEM
    setProgressBar("Compute altitude..", nrMeteoPoints);

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        QString idStr = QString::fromStdString(meteoPoints[i].id);
        DEM.getRowCol(meteoPoints[i].point.utm.x, meteoPoints[i].point.utm.y, row, col);
        values.clear();

        for (int r = row - stepCells; r <= row + stepCells; r++)
        {
            for (int c = col - stepCells; c <= col + stepCells; c++)
            {
                float z = DEM.getValueFromRowCol(r, c);
                if (! isEqual(z, DEM.header->flag))
                {
                    values.push_back(z);
                }
            }
        }

        // update point properties (NODATA if no value)
        float altitude = statistics::mean(values);
        QString query = QString("UPDATE point_properties SET altitude = %1 WHERE id_point = '%2'").arg(altitude).arg(idStr);
        meteoPointsDb.exec(query);

        updateProgressBar(i);
    }

    return true;
}


void Project::getMeteoPointsCurrentValues(std::vector<float> &validValues)
{
    // user has selected a set of points
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active && meteoPoints[i].selected)
        {
            if (meteoPoints[i].currentValue != NODATA)
            {
                validValues.push_back(meteoPoints[i].currentValue);
            }
        }
    }

    // no selection: all points
    if (validValues.size() == 0)
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            if (meteoPoints[i].active && meteoPoints[i].currentValue != NODATA)
            {
                validValues.push_back(meteoPoints[i].currentValue);
            }
        }
    }
}


bool Project::setSelectedStatePointList(QString fileName)
{
    QList<QString> pointList = readListSingleColumn(fileName, errorString);
    if (! selectPointList(pointList))
    {
        logWarning("No points to select.");
        return false;
    }

    return true;
}


bool Project::selectPointList(const QList<QString> &pointList)
{
    if (pointList.size() == 0)
        return false;

    int nrValidPoints = 0;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (pointList.contains(QString::fromStdString(meteoPoints[i].id)))
        {
            meteoPoints[i].selected = true;
            nrValidPoints++;
        }
    }

    return (nrValidPoints > 0);
}


bool Project::setSelectedStateWithCriteria()
{
    if (! meteoPointsLoaded)
    {
        logError(ERROR_STR_MISSING_DB);
        return false;
    }

    bool isActive = false;
    bool isSelect = true;
    DialogSelectionMeteoPoint dialogPointSelection(isActive, isSelect, meteoPointsDbHandler);
    if (dialogPointSelection.result() != QDialog::Accepted)
    {
        return false;
    }

    QString selection = dialogPointSelection.getSelection();
    QString operation = dialogPointSelection.getOperation();
    QString item = dialogPointSelection.getItem();

    QString condition;
    QList<QString> selectedPointsList;

    if (operation == "Like")
    {
        condition = selection + " " + operation + " '%" + item + "%'";
    }
    else
    {
        condition = selection + " " + operation + " '" + item + "'";
    }

    if (selection != "DEM distance [m]")
    {
        if (! meteoPointsDbHandler->getPointListWithCriteria(selectedPointsList, condition))
        {
            logError(meteoPointsDbHandler->getErrorString());
            return false;
        }
    }
    else
    {
        if (! DEM.isLoaded)
        {
            logError(ERROR_STR_MISSING_DEM);
            return false;
        }
        setProgressBar("Checking distance...", nrMeteoPoints);
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            updateProgressBar(i);
            if (!meteoPoints[i].selected)
            {
                float distance = gis::closestDistanceFromGrid(meteoPoints[i].point, DEM);
                if (operation == "=")
                {
                    if (isEqual(distance, item.toFloat()))
                    {
                        selectedPointsList.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == "!=")
                {
                    if (! isEqual(distance, item.toFloat()))
                    {
                        selectedPointsList.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == ">")
                {
                    if (distance > item.toFloat())
                    {
                        selectedPointsList.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
                else if (operation == "<")
                {
                    if (distance < item.toFloat())
                    {
                        selectedPointsList.append(QString::fromStdString(meteoPoints[i].id));
                    }
                }
            }
        }
        closeProgressBar();
    }

    if (selectedPointsList.isEmpty())
    {
        logWarning("No point matches your requirements.");
        return false;
    }

    if (! selectPointList(selectedPointsList))
    {
        logWarning("No points to select");
        return false;
    }

    return true;
}


bool Project::readVmArkimetData(const QList<QString> &vmFileList, frequencyType frequency, bool isPrec0024)
{
    if(nrMeteoPoints == 0)
    {
        errorString = ERROR_STR_MISSING_DB;
        return false;
    }

    if (vmFileList.isEmpty())
    {
        errorString = "No data.";
        return false;
    }

    // open db meteo
    QString dbName = meteoPointsDbHandler->getDbName();
    DbArkimet *dbMeteoArkimet = new DbArkimet(dbName);
    if (! dbMeteoArkimet->getErrorString().isEmpty())
    {
        errorString = dbMeteoArkimet->getErrorString();
        return false;
    }

    setProgressBar("Loading data...", vmFileList.size());
    for (int i=0; i < vmFileList.size(); i++)
    {
        updateProgressBar(i);
        if (frequency == daily)
        {
            if (! dbMeteoArkimet->readVmDataDaily(vmFileList[i], isPrec0024, errorString))
                return false;
        }
        else
        {
            // todo hourly
        }
    }
    closeProgressBar();

    delete dbMeteoArkimet;

    return true;
}


bool Project::computeResidualsAndStatisticsGlocalDetrending(meteoVariable myVar, std::vector<Crit3DInterpolationDataPoint> &interpolationPoints)
{
    //TODO: glocal cv with grid ONLY (no DEM)

    if (myVar == noMeteoVar)
        return false;

    int elevationPos = NODATA;
    for (unsigned int pos=0; pos < interpolationSettings.getCurrentCombination().getProxySize(); pos++)
    {
        if (getProxyPragaName(interpolationSettings.getProxy(pos)->getName()) == proxyHeight)
            elevationPos = pos;
    }

    for (int j = 0; j < nrMeteoPoints; j++)
    {
        meteoPoints[j].residual = NODATA;
    }

    //ciclo sulle aree
    for (int k = 0; k < interpolationSettings.getMacroAreasSize(); k++)
    {
        Crit3DMacroArea myArea = interpolationSettings.getMacroArea(k);
        std::vector<int> meteoPointsList = myArea.getMeteoPoints();

        //if (! myArea.getAreaCellsGrid().empty() || ! myArea.getAreaCellsDEM().empty() )
        if (! myArea.getAreaCellsDEM().empty() && ! meteoPointsList.empty())
        {
            interpolationSettings.pushMacroAreaNumber(k);

            if (! computeResidualsGlocalDetrending(myVar, myArea, elevationPos, meteoPoints, interpolationPoints,
                                                     interpolationSettings, meteoSettings, true, true))
                return false;

            if (! computeStatisticsGlocalCrossValidation(myArea))
                return false;
        }
    }

    return true;
}

bool Project::showMeteoWidgetMultiplePoints()
{
    std::vector<std::string> idMeteoPoint;
    std::vector<std::string> namePoint;
    std::vector<std::string> dataset;
    std::vector<double> altitude;
    std::vector<std::string> lapseRateCode;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].selected)
        {
            idMeteoPoint.push_back(meteoPoints[i].id);
            namePoint.push_back(meteoPoints[i].name);
            dataset.push_back(meteoPoints[i].dataset);
            altitude.push_back(meteoPoints[i].point.z);
            lapseRateCode.push_back(std::to_string(meteoPoints[i].lapseRateCode));
        }
    }

    bool isAppend = false;

    for (unsigned int i = 0; i < idMeteoPoint.size(); i++)
    {
        showMeteoWidgetPoint(idMeteoPoint[i], namePoint[i], dataset[i], altitude[i], lapseRateCode[i], isAppend);
        isAppend = true;
    }

    return true;
}

bool Project::getProjectList(QList<QString> &projectList)
{
    QString myProjectsPath = getDefaultPath() + PATH_PROJECT;
    QDir dir(myProjectsPath);
    if (! dir.exists())
    {
        errorString = "PROJECT directory is missing: " + myProjectsPath;
        return false;
    }
    QFileInfoList list = dir.entryInfoList(QDir::AllDirs | QDir::NoDot | QDir::NoDotDot | QDir::NoSymLinks);

    if (list.size() == 0)
    {
        errorString = "PROJECT directory is empty: " + myProjectsPath;
        return false;
    }

    for (int i=0; i < list.size(); i++)
    {
        projectList << list[i].baseName();
    }

    return true;
}
