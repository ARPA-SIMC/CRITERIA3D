#include <math.h>
#include "commonConstants.h"
#include "criteria1DError.h"
#include "criteria1DProject.h"
#include "basicMath.h"
#include "soilDbTools.h"
#include "cropDbTools.h"
#include "cropDbQuery.h"
#include "criteria1DMeteo.h"
#include "utilities.h"

#include <QSqlError>
#include <QDate>
#include <QSqlQuery>
#include <QDir>
#include <QSettings>


Crit1DProject::Crit1DProject()
{
    this->initialize();
}


void Crit1DProject::initialize()
{
    isProjectLoaded = false;

    path = "";
    projectName = "";
    logFileName = "";
    outputCsvFileName = "";
    addDateTimeLogFile = false;

    dbCropName = "";
    dbSoilName = "";
    dbMeteoName = "";
    dbForecastName = "";
    dbOutputName = "";
    dbComputationUnitsName = "";

    projectError = "";

    compUnitList.clear();

    isXmlMeteoGrid = false;
    isSaveState = false;
    isRestart = false;

    isYearlyStatistics = false;
    isMonthlyStatistics = false;
    isSeasonalForecast = false;
    isEnsembleForecast = false;
    isShortTermForecast = false;

    firstMonth = NODATA;
    daysOfForecast = NODATA;
    nrYears = NODATA;
    irriSeries.clear();
    precSeries.clear();

    firstSimulationDate = QDate(1800,1,1);
    lastSimulationDate = QDate(1800,1,1);

    outputString = "";

    // specific outputs
    waterDeficitDepth.clear();
    waterContentDepth.clear();
    degreeOfSaturationDepth.clear();
    waterPotentialDepth.clear();
    availableWaterDepth.clear();
    fractionAvailableWaterDepth.clear();
    factorOfSafetyDepth.clear();
    awcDepth.clear();

    texturalClassList.resize(13);
    geotechnicsClassList.resize(19);
}


void Crit1DProject::closeProject()
{
    if (isProjectLoaded)
    {
        logger.writeInfo("Close Project...");
        closeAllDatabase();

        isProjectLoaded = false;
    }
}


bool Crit1DProject::readSettings()
{
    QSettings* projectSettings;
    projectSettings = new QSettings(configFileName, QSettings::IniFormat);

    // PROJECT
    projectSettings->beginGroup("project");

    path += projectSettings->value("path","").toString();
    projectName = projectSettings->value("name", "CRITERIA1D").toString();

    dbCropName = projectSettings->value("db_crop","").toString();
    if (dbCropName.left(1) == ".")
        dbCropName = path + dbCropName;

    dbSoilName = projectSettings->value("db_soil","").toString();
    if (dbSoilName.left(1) == ".")
        dbSoilName = path + dbSoilName;

    dbMeteoName = projectSettings->value("db_meteo","").toString();
    if (dbMeteoName.left(1) == ".")
        dbMeteoName = path + dbMeteoName;
    if (dbMeteoName.right(3) == "xml")
        isXmlMeteoGrid = true;

    dbForecastName = projectSettings->value("db_forecast","").toString();
    if (dbForecastName.left(1) == ".")
        dbForecastName = path + dbForecastName;

    // computational units db
    dbComputationUnitsName = projectSettings->value("db_comp_units","").toString();
    if (dbComputationUnitsName == "")
    {
        // check old name
        dbComputationUnitsName = projectSettings->value("db_units","").toString();
    }
    if (dbComputationUnitsName == "")
    {
        projectError = "Missing information on computational units";
        return false;
    }
    if (dbComputationUnitsName.left(1) == ".")
        dbComputationUnitsName = path + dbComputationUnitsName;

   // output db
    dbOutputName = projectSettings->value("db_output","").toString();
    if (dbOutputName.left(1) == ".")
        dbOutputName = path + dbOutputName;

    // date
    if (firstSimulationDate == QDate(1800,1,1))
    {
        firstSimulationDate = projectSettings->value("firstDate",0).toDate();
        if (! firstSimulationDate.isValid())
        {
            firstSimulationDate = QDate(1800,1,1);
        }
    }

    if (lastSimulationDate == QDate(1800,1,1))
    {
        lastSimulationDate = projectSettings->value("lastDate",0).toDate();
        if (! lastSimulationDate.isValid())
        {
            lastSimulationDate = QDate(1800,1,1);
        }
    }

    addDateTimeLogFile = projectSettings->value("add_date_to_log","").toBool();
    isSaveState = projectSettings->value("save_state","").toBool();
    isRestart = projectSettings->value("restart","").toBool();

    projectSettings->endGroup();

    // FORECAST
    projectSettings->beginGroup("forecast");
        isYearlyStatistics = projectSettings->value("isYearlyStatistics", 0).toBool();
        isMonthlyStatistics = projectSettings->value("isMonthlyStatistics", 0).toBool();
        isSeasonalForecast = projectSettings->value("isSeasonalForecast", 0).toBool();
        isShortTermForecast = projectSettings->value("isShortTermForecast", 0).toBool();

        // ensemble forecast (typically they are monthly)
        isEnsembleForecast = projectSettings->value("isEnsembleForecast", 0).toBool();
        if (! isEnsembleForecast)
            // check also monthly
            isEnsembleForecast = projectSettings->value("isMonthlyForecast", 0).toBool();

        if (isShortTermForecast || isEnsembleForecast)
        {
            daysOfForecast = projectSettings->value("daysOfForecast", 0).toInt();
            if (daysOfForecast == 0)
            {
                projectError = "Missing daysOfForecast";
                return false;
            }
        }

        if (isYearlyStatistics)
        {
            firstMonth = 1;
        }
        else if (isMonthlyStatistics || isSeasonalForecast)
        {
            firstMonth = projectSettings->value("firstMonth", 0).toInt();
            if (firstMonth == 0)
            {
                projectError = "Missing firstMonth.";
                return false;
            }
        }

        int nrOfComputationType = 0;
        if (isShortTermForecast) nrOfComputationType++;
        if (isEnsembleForecast) nrOfComputationType++;
        if (isSeasonalForecast) nrOfComputationType++;
        if (isMonthlyStatistics) nrOfComputationType++;
        if (isYearlyStatistics) nrOfComputationType++;

        if (nrOfComputationType > 1)
        {
            projectError = "Too many forecast/computation types.";
            return false;
        }

    projectSettings->endGroup();

    projectSettings->beginGroup("csv");
        outputCsvFileName = projectSettings->value("csv_output","").toString();
        if (outputCsvFileName != "")
        {
            if (outputCsvFileName.right(4) == ".csv")
            {
                outputCsvFileName = outputCsvFileName.left(outputCsvFileName.length()-4);
            }

            bool addDate = projectSettings->value("add_date_to_filename","").toBool();
            if (addDate)
            {
                QString dateStr;
                if (lastSimulationDate == QDate(1800,1,1))
                {
                    dateStr = QDate::currentDate().toString("yyyy-MM-dd");
                }
                else
                {
                    dateStr = lastSimulationDate.addDays(1).toString("yyyy-MM-dd");
                }
                outputCsvFileName += "_" + dateStr;
            }
            outputCsvFileName += ".csv";

            if (outputCsvFileName.at(0) == '.')
            {
                outputCsvFileName = path + QDir::cleanPath(outputCsvFileName);
            }
        }

    projectSettings->endGroup();

    // OUTPUT variables (optional)
    QList<QString> depthList;
    projectSettings->beginGroup("output");
        depthList = projectSettings->value("waterContent").toStringList();
        if (! setVariableDepth(depthList, waterContentDepth))
        {
            projectError = "Wrong water content depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("degreeOfSaturation").toStringList();
        if (! setVariableDepth(depthList, degreeOfSaturationDepth))
        {
            projectError = "Wrong degree of saturation depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("waterPotential").toStringList();
        if (! setVariableDepth(depthList, waterPotentialDepth))
        {
            projectError = "Wrong water potential depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("waterDeficit").toStringList();
        if (! setVariableDepth(depthList, waterDeficitDepth))
        {
            projectError = "Wrong water deficit depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("awc").toStringList();
        if (! setVariableDepth(depthList, awcDepth))
        {
            projectError = "Wrong available water capacity depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("availableWater").toStringList();
        if (depthList.size() == 0)
            depthList = projectSettings->value("aw").toStringList();
        if (! setVariableDepth(depthList, availableWaterDepth))
        {
            projectError = "Wrong available water depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("fractionAvailableWater").toStringList();
        if (depthList.size() == 0)
            depthList = projectSettings->value("faw").toStringList();
        if (! setVariableDepth(depthList, fractionAvailableWaterDepth))
        {
            projectError = "Wrong fraction available water depth in " + configFileName;
            return false;
        }
        depthList = projectSettings->value("factorOfSafety").toStringList();
        if (depthList.size() == 0)
            depthList = projectSettings->value("factorOfSafety").toStringList();
        if (! setVariableDepth(depthList, factorOfSafetyDepth))
        {
            projectError = "Wrong slope stability depth in " + configFileName;
            return false;
        }
    projectSettings->endGroup();

    return true;
}


int Crit1DProject::initializeProject(QString settingsFileName)
{
    if (settingsFileName == "")
    {
        logger.writeError("Missing settings File.");
        return ERROR_SETTINGS_MISSING;
    }

    // Settings file
    QFile myFile(settingsFileName);
    if (myFile.exists())
    {
        configFileName = QDir(myFile.fileName()).canonicalPath();
        configFileName = QDir().cleanPath(configFileName);

        QFileInfo fileInfo(configFileName);
        path = fileInfo.path() + "/";
    }
    else
    {
        projectError = "Cannot find settings file: " + settingsFileName;
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (!readSettings())
        return ERROR_SETTINGS_MISSINGDATA;

    logger.setLog(path, projectName, addDateTimeLogFile);

    checkSimulationDates();

    int myError = openAllDatabase();
    if (myError != CRIT1D_OK)
        return myError;

    if (! loadVanGenuchtenParameters(dbSoil, texturalClassList, projectError))
        return ERROR_SOIL_PARAMETERS;

    if (! loadDriessenParameters(dbSoil, texturalClassList, projectError))
        return ERROR_SOIL_PARAMETERS;

    // missing table is not critical
    loadGeotechnicsParameters(dbSoil, geotechnicsClassList, projectError);
    projectError = "";

    // Computational unit list
    if (! readComputationUnitList(dbComputationUnitsName, compUnitList, projectError))
    {
        logger.writeError(projectError);
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(compUnitList.size()) + " distinct computational units.");

    isProjectLoaded = true;

    return CRIT1D_OK;
}


void Crit1DProject::checkSimulationDates()
{
    // first date
    QString dateStr = firstSimulationDate.toString("yyyy-MM-dd");
    if (dateStr == "1800-01-01")
    {
        isRestart = false;
        dateStr = "UNDEFINED";
    }
    logger.writeInfo("First simulation date: " + dateStr);
    QString boolStr = isRestart ? "TRUE" : "FALSE";
    logger.writeInfo("Restart: " + boolStr);

    // last date
    dateStr = lastSimulationDate.toString("yyyy-MM-dd");
    if (dateStr == "1800-01-01")
    {
        if (isXmlMeteoGrid)
        {
            lastSimulationDate = QDate::currentDate().addDays(-1);
            dateStr = lastSimulationDate.toString("yyyy-MM-dd");
        }
        else
        {
            isSaveState = false;
            dateStr = "UNDEFINED";
        }
    }

    logger.writeInfo("Last simulation date: " + dateStr);
    boolStr = isSaveState? "TRUE" : "FALSE";
    logger.writeInfo("Save state: " + boolStr);

    if (isSeasonalForecast)
    {
        logger.writeInfo("First forecast month: " + QString::number(firstMonth));
    }

    if (isMonthlyStatistics)
    {
        logger.writeInfo("Computation month: " + QString::number(firstMonth));
    }

    if (isShortTermForecast || isEnsembleForecast)
    {
        logger.writeInfo("Nr of forecast days: " + QString::number(daysOfForecast));
    }
}


bool Crit1DProject::setSoil(QString soilCode, QString &errorStr)
{
    if (! loadSoil(dbSoil, soilCode, myCase.mySoil, texturalClassList, geotechnicsClassList, myCase.fittingOptions, errorStr))
        return false;

    // warning: some soil data are wrong
    if (errorStr != "")
    {
        //logger.writeInfo("WARNING: " + errorStr);
        errorStr = "";
    }

    std::string errorStdString;
    if (! myCase.initializeSoil(errorStdString))
    {
        errorStr = QString::fromStdString(errorStdString);
        return false;
    }

    return true;
}


bool Crit1DProject::setMeteoXmlGrid(QString idMeteo, QString idForecast, unsigned int memberNr)
{
    unsigned row, col;
    unsigned nrDays = unsigned(firstSimulationDate.daysTo(lastSimulationDate)) + 1;

    if (!observedMeteoGrid->meteoGrid()->findMeteoPointFromId(&row, &col, idMeteo.toStdString()) )
    {
        projectError = "Missing observed meteo cell: " + idMeteo;
        return false;
    }

    if (!observedMeteoGrid->gridStructure().isFixedFields())
    {
        if (!observedMeteoGrid->loadGridDailyData(projectError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            projectError = "Missing observed data: " + idMeteo;
            return false;
        }
    }
    else
    {
        if (!observedMeteoGrid->loadGridDailyDataFixedFields(projectError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            if (projectError == "Missing MeteoPoint id")
            {
                projectError = "Missing observed meteo cell: " + idMeteo;
            }
            else
            {
                projectError = "Missing observed data: " + idMeteo;
            }
            return false;
        }
    }

    if (this->isShortTermForecast)
    {
        if (!this->forecastMeteoGrid->gridStructure().isFixedFields())
        {
            if (!this->forecastMeteoGrid->loadGridDailyData(projectError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell:" + idForecast;
                }
                else
                {
                    projectError = "Missing forecast data:" + idForecast;
                }
                return false;
            }
        }
        else
        {
            if (!this->forecastMeteoGrid->loadGridDailyDataFixedFields(projectError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell:" + idForecast;
                }
                else
                {
                    projectError = "Missing forecast data:" + idForecast;
                }
                return false;
            }
        }
        nrDays += unsigned(daysOfForecast);
    }

    if (this->isEnsembleForecast)
    {
        if (this->forecastMeteoGrid->gridStructure().isFixedFields())
        {
            projectError = "DB grid fixed fields: not available";
            return false;
        }
        else
        {
            if (!this->forecastMeteoGrid->loadGridDailyDataEnsemble(projectError, idForecast, int(memberNr), lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell:" + idForecast;
                }
                else
                {
                    projectError = "Missing forecast data:" + idForecast;
                }
                return false;
            }
        }
        nrDays += unsigned(daysOfForecast);
    }

    myCase.meteoPoint.latitude = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->latitude;
    myCase.meteoPoint.longitude = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->longitude;
    myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstSimulationDate));

    float tmin, tmax, tavg, prec;
    long lastIndex = long(firstSimulationDate.daysTo(lastSimulationDate)) + 1;
    for (int i = 0; i < lastIndex; i++)
    {
        Crit3DDate myDate = getCrit3DDate(firstSimulationDate.addDays(i));
        tmin = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);

        tmax = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);

        tavg = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
        if (int(tavg) == int(NODATA))
        {
            tavg = (tmax + tmin)/2;
        }
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);

        prec = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyPrecipitation);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyPrecipitation, prec);
    }
    if (isShortTermForecast || isEnsembleForecast)
    {
        QDate start = lastSimulationDate.addDays(1);
        QDate end = lastSimulationDate.addDays(daysOfForecast);
        for (int i = 0; i< start.daysTo(end)+1; i++)
        {
            Crit3DDate myDate = getCrit3DDate(start.addDays(i));
            tmin = this->forecastMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);

            tmax = this->forecastMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);

            tavg = this->forecastMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
            if (int(tavg) == int(NODATA))
            {
                tavg = (tmax + tmin)/2;
            }
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);

            prec = this->forecastMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyPrecipitation);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyPrecipitation, prec);
        }
    }
    return true;
}


bool Crit1DProject::setMeteoSqlite(QString idMeteo, QString idForecast)
{
    QString queryString = "SELECT * FROM point_properties WHERE id_meteo='" + idMeteo + "'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        // previous code version
        queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo + "'";
        query = dbMeteo.exec(queryString);
        query.last();
    }

    if (! query.isValid())
    {
        projectError = "Missing point_properties for ID meteo: " + idMeteo;
        return false;
    }

    QString tableName = query.value("table_name").toString();

    double myLat, myLon;
    if (getValue(query.value(("latitude")), &myLat))
    {
        myCase.meteoPoint.latitude = myLat;
    }
    else
    {
        projectError = "Missing latitude in idMeteo: " + idMeteo;
        return false;
    }

    if (getValue(query.value(("longitude")), &myLon))
    {
        myCase.meteoPoint.longitude = myLon;
    }
    else
    {
        projectError = "Missing longitude in idMeteo: " + idMeteo;
        return false;
    }

    queryString = "SELECT * FROM '" + tableName + "' ORDER BY [date]";
    query = this->dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().text() != "")
            projectError = "dbMeteo error: " + query.lastError().text();
        else
            projectError = "Missing meteo table:" + tableName;
        return false;
    }

    query.first();
    QDate firstDate = query.value("date").toDate();
    query.last();
    QDate lastDate = query.value("date").toDate();
    unsigned nrDays;
    bool subQuery = false;

    // check dates
    if (firstSimulationDate.toString("yyyy-MM-dd") != "1800-01-01")
    {
        if (firstSimulationDate < firstDate)
        {
            projectError = "Missing meteo data: required first date " + firstSimulationDate.toString("yyyy-MM-dd");
            return false;
        }
        else
        {
            firstDate = firstSimulationDate;
            subQuery = true;
        }
    }
    if (lastSimulationDate.toString("yyyy-MM-dd") != "1800-01-01")
    {
        if (lastSimulationDate > lastDate)
        {
            projectError = "Missing meteo data: required last date " + lastSimulationDate.toString("yyyy-MM-dd");
            return false;
        }
        else
        {
            lastDate = lastSimulationDate;
            subQuery = true;
        }
    }

    nrDays = unsigned(firstDate.daysTo(lastDate)) + 1;
    if (subQuery)
    {
        query.clear();
        queryString = "SELECT * FROM '" + tableName + "' WHERE date BETWEEN '"
                    + firstDate.toString("yyyy-MM-dd") + "' AND '" + lastDate.toString("yyyy-MM-dd") + "'"
                    + " ORDER BY date";
        query = this->dbMeteo.exec(queryString);
    }

    // Forecast: increase nr of days
    if (isShortTermForecast)
        nrDays += unsigned(daysOfForecast);

    // Initialize data
    myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstDate));

    // Read observed data
    int maxNrDays = NODATA; // all data
    if (! readDailyDataCriteria1D(query, myCase.meteoPoint, maxNrDays, projectError))
        return false;

    // write missing data on log
    if (projectError != "")
    {
        this->logger.writeInfo(projectError);
        projectError = "";
    }

    // Add Short-Term forecast
    if (this->isShortTermForecast)
    {
        queryString = "SELECT * FROM point_properties WHERE id_meteo='" + idForecast + "'";
        query = dbForecast.exec(queryString);
        query.last();

        if (! query.isValid())
        {
            // previous code version
            queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idForecast + "'";
            query = dbMeteo.exec(queryString);
            query.last();
        }

        if (! query.isValid())
        {
            if (query.lastError().text().isEmpty())
            {
                projectError = "DB: " + dbForecast.databaseName() + "\nMissing point_properties for id meteo:" + idForecast;
            }
            else
            {
                projectError = "dbForecast error: " + query.lastError().text();
            }
            return false;
        }

        QString tableNameForecast = query.value("table_name").toString();

        query.clear();
        queryString = "SELECT * FROM '" + tableNameForecast + "' ORDER BY [date]";
        query = this->dbForecast.exec(queryString);
        query.last();

        // check query
        if (! query.isValid())
        {
            if (query.lastError().text() != "")
                projectError = "dbForecast error: " + query.lastError().text();
            else
                projectError = "Missing forecast table:" + tableName;
            return false;
        }

        // check date
        query.first();
        QDate firstForecastDate = query.value("date").toDate();
        if (firstForecastDate != lastDate.addDays(1))
        {
            // previsioni indietro di un giorno: accettato ma tolgo un giorno
            if (firstForecastDate == lastDate)
            {
                myCase.meteoPoint.nrObsDataDaysD--;
            }
            else
            {
                projectError = "The forecast date doesn't match with observed data.";
                return false;
            }
        }

        // Read forecast data
        maxNrDays = daysOfForecast;
        if (! readDailyDataCriteria1D(query, myCase.meteoPoint, maxNrDays, projectError))
                return false;

        if (projectError != "")
        {
            this->logger.writeInfo(projectError);
        }

        // fill temperature (only forecast)
        // estende il dato precedente se mancante
        float previousTmin = NODATA;
        float previousTmax = NODATA;
        long lastIndex = long(firstDate.daysTo(lastDate));
        for (unsigned long i = unsigned(lastIndex); i < unsigned(myCase.meteoPoint.nrObsDataDaysD); i++)
        {
            // tmin
            if (int(myCase.meteoPoint.obsDataD[i].tMin) != int(NODATA))
                previousTmin = myCase.meteoPoint.obsDataD[i].tMin;
            else if (int(previousTmin) != int(NODATA))
                myCase.meteoPoint.obsDataD[i].tMin = previousTmin;

            // tmax
            if (int(myCase.meteoPoint.obsDataD[i].tMax) != int(NODATA))
                previousTmax = myCase.meteoPoint.obsDataD[i].tMax;
            else if (int(previousTmax) != int(NODATA))
                myCase.meteoPoint.obsDataD[i].tMax = previousTmax;
        }
    }

    // fill watertable (all data)
    // estende il dato precedente se mancante
    float previousWatertable = NODATA;
    for (unsigned long i = 0; i < unsigned(myCase.meteoPoint.nrObsDataDaysD); i++)
    {
        // watertable
        if (! isEqual(myCase.meteoPoint.obsDataD[i].waterTable, NODATA))
        {
            previousWatertable = myCase.meteoPoint.obsDataD[i].waterTable;
        }
        else if (isEqual(previousWatertable, NODATA))
        {
            myCase.meteoPoint.obsDataD[i].waterTable = previousWatertable;
        }
    }

    return true;
}


bool Crit1DProject::computeUnit(const Crit1DCompUnit& myUnit)
{
    myCase.unit = myUnit;
    return computeCase(0);
}


bool Crit1DProject::computeUnit(unsigned int unitIndex, unsigned int memberNr)
{
    myCase.unit = compUnitList[unitIndex];
    return computeCase(memberNr);
}


// use memberNr = 0 for deterministic run
bool Crit1DProject::computeCase(unsigned int memberNr)
{
    myCase.fittingOptions.useWaterRetentionData = myCase.unit.useWaterRetentionData;
    // user wants to compute factor of safety
    myCase.computeFactorOfSafety = (factorOfSafetyDepth.size() > 0);

    if (! loadCropParameters(dbCrop, myCase.unit.idCrop, myCase.crop, projectError))
        return false;

    if (! setSoil(myCase.unit.idSoil, projectError))
        return false;

    if (isXmlMeteoGrid)
    {
        if (! setMeteoXmlGrid(myCase.unit.idMeteo, myCase.unit.idForecast, memberNr))
            return false;
    }
    else
    {
        if (! setMeteoSqlite(myCase.unit.idMeteo, myCase.unit.idForecast))
            return false;
    }

    // check meteo data
    if (myCase.meteoPoint.nrObsDataDaysD == 0)
    {
        projectError = "Missing meteo data: " + QString::fromStdString(myCase.meteoPoint.name);
        return false;
    }

    if ( !isMonthlyStatistics && !isSeasonalForecast && !isEnsembleForecast )
    {
        if (! createOutputTable(projectError))
            return false;
    }
    // get irri ratio
    if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast)
    {
        float irriRatio = getIrriRatioFromCropClass(dbCrop, "crop_class", "id_class",
                                                myCase.unit.idCropClass, projectError);
        if (irriRatio < 0.001f)
        {
            // No irrigation: nothing to do
            return true;
        }
    }

    // set computation period (all meteo data)
    Crit3DDate myDate, firstDate, lastDate;
    unsigned long lastIndex = unsigned(myCase.meteoPoint.nrObsDataDaysD-1);
    firstDate = myCase.meteoPoint.obsDataD[0].date;
    lastDate = myCase.meteoPoint.obsDataD[lastIndex].date;

    if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast)
    {
        initializeIrrigationStatistics(firstDate, lastDate);
    }
    int indexIrrigationSeries = NODATA;

    // initialize crop
    unsigned nrLayers = unsigned(myCase.soilLayers.size());
    myCase.crop.initialize(myCase.meteoPoint.latitude, nrLayers,
                             myCase.mySoil.totalDepth, getDoyFromDate(firstDate));

    // initialize water content
    if (! myCase.initializeWaterContent(firstDate))
        return false;

    // restart
    bool isFirstDay = true;
    std::string errorString;
    if (isRestart)
    {
        QString outputDbPath = getFilePath(dbOutput.databaseName());
        QString stateDbName = outputDbPath + "state_" + firstSimulationDate.toString("yyyy_MM_dd")+".db";
        if (! restoreState(stateDbName, projectError))
        {
            return false;
        }

        float waterTable = myCase.meteoPoint.getMeteoPointValueD(firstDate, dailyWaterTableDepth);
        if (! myCase.crop.restore(myDate, myCase.meteoPoint.latitude, myCase.soilLayers, double(waterTable), errorString))
        {
            projectError = QString::fromStdString(errorString);
            return false;
        }
    }

    // daily cycle
    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! myCase.computeDailyModel(myDate, errorString))
        {
            projectError = QString::fromStdString(errorString);
            return false;
        }

        // output
        if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast)
        {
            updateIrrigationStatistics(myDate, indexIrrigationSeries);
        }
        if (isEnsembleForecast)
        {
            updateMediumTermForecastOutput(myDate, memberNr);
        }
        if ( !isEnsembleForecast && !isSeasonalForecast && !isMonthlyStatistics)
        {
            updateOutput(myDate, isFirstDay);
            isFirstDay = false;
        }
    }

    if (isSaveState)
    {
        if (! saveState(projectError))
            return false;
        logger.writeInfo("Save state:" + dbState.databaseName());
    }

    // SeasonalForecast, EnsembleForecast and MonthlyStatistics do not produce db output (too much useless data)
    if (isSeasonalForecast || isEnsembleForecast || isMonthlyStatistics)
        return true;
    else
        return saveOutput(projectError);
}


int Crit1DProject::computeAllUnits()
{
    bool isErrorModel = false;
    bool isErrorSoil = false;
    bool isErrorCrop = false;
    unsigned int nrUnitsComputed = 0;

    if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast || isEnsembleForecast)
    {
        if (!setPercentileOutputCsv())
            return ERROR_DBOUTPUT;
    }
    else
    {
        if (dbOutputName == "")
        {
            logger.writeError("Missing output db");
            return ERROR_DBOUTPUT;
        }

    }

    // create db state
    if (isSaveState)
    {
        if (! createDbState(projectError))
        {
            logger.writeError(projectError);
            return ERROR_DB_STATE;
        }
    }

    int infoStep = std::max(1, int(compUnitList.size() / 20));
    logger.writeInfo("COMPUTE...");

    try
    {
        for (unsigned int i = 0; i < compUnitList.size(); i++)
        {
            // is numerical
            if (compUnitList[i].isNumericalInfiltration)
            {
                logger.writeInfo(compUnitList[i].idCase + " - numerical computation...");
            }

            // CROP
            compUnitList[i].idCrop = getIdCropFromClass(dbCrop, "crop_class", "id_class",
                                                         compUnitList[i].idCropClass, projectError);
            if (compUnitList[i].idCrop == "")
            {
                logger.writeInfo("Unit " + compUnitList[i].idCase + " " + compUnitList[i].idCropClass + " ***** missing CROP *****");
                isErrorCrop = true;
                continue;
            }

            // IRRI_RATIO
            float irriRatio = getIrriRatioFromCropClass(dbCrop, "crop_class", "id_class",
                                                    compUnitList[i].idCropClass, projectError);

            if ((isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast || isEnsembleForecast || isShortTermForecast)
                && (int(irriRatio) == int(NODATA)))
            {
                logger.writeInfo("Unit " + compUnitList[i].idCase + " " + compUnitList[i].idCropClass + " ***** missing IRRIGATION RATIO *****");
                continue;
            }

            // SOIL
            if (compUnitList[i].idSoilNumber != NODATA)
                compUnitList[i].idSoil = getIdSoilString(dbSoil, compUnitList[i].idSoilNumber, projectError);

            if (compUnitList[i].idSoil == "")
            {
                logger.writeInfo("Unit " + compUnitList[i].idCase + " Soil nr." + QString::number(compUnitList[i].idSoilNumber) + " ***** missing SOIL *****");
                isErrorSoil = true;
                continue;
            }

            if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast)
            {
                if (computeIrrigationStatistics(i, irriRatio))
                    nrUnitsComputed++;
                else
                    isErrorModel = true;
            }
            else
            {
                if (isEnsembleForecast)
                {
                    if (computeMonthlyForecast(i, irriRatio))
                        nrUnitsComputed++;
                    else
                        isErrorModel = true;
                }
                else
                {
                    if (computeUnit(i, 0))
                    {
                        nrUnitsComputed++;
                    }
                    else
                    {
                        projectError = "Computational Unit: " + compUnitList[i].idCase + "\n" + projectError;
                        logger.writeError(projectError);
                        isErrorModel = true;
                    }
                }
            }

            if ((i+1) % infoStep == 0 && nrUnitsComputed > 0)
            {
                double percentage = (i+1) * 100.0 / compUnitList.size();
                logger.writeInfo("..." + QString::number(round(percentage)) + "%");
            }
        }

        if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast || isEnsembleForecast)
        {
            outputCsvFile.close();
        }

    } catch (std::exception &e)
    {
        qFatal("Error %s ", e.what());

    } catch (...)
    {
        qFatal("Error <unknown>");
        return ERROR_UNKNOWN;
    }

    // error check
    if (nrUnitsComputed == 0)
    {
        if (isErrorModel)
            return ERROR_METEO_OR_MODEL;
        else if (isErrorSoil)
            return ERROR_SOIL_MISSING;
        else if (isErrorCrop)
            return ERROR_CROP_MISSING;
        else
            return ERROR_UNKNOWN;
    }
    else if (nrUnitsComputed < compUnitList.size())
    {
        if (isErrorModel)
            return WARNING_METEO_OR_MODEL;
        else if (isErrorSoil)
            return WARNING_SOIL;
        else if (isErrorCrop)
            return WARNING_CROP;
    }

    return CRIT1D_OK;
}


// update values of medium term forecast
void Crit1DProject::updateMediumTermForecastOutput(Crit3DDate myDate, unsigned int memberNr)
{
    QDate myQdate = getQDate(myDate);

    if (myQdate == lastSimulationDate)
    {
        irriSeries[memberNr] = 0;
        precSeries[memberNr] = 0;
    }
    else if (myQdate > lastSimulationDate)
    {
        irriSeries[memberNr] += float(myCase.output.dailyIrrigation);
        precSeries[memberNr] += float(myCase.output.dailyPrec);
    }
}


// update values of annual irrigation
void Crit1DProject::updateIrrigationStatistics(Crit3DDate myDate, int &currentIndex)
{
    if ( !isYearlyStatistics && !isMonthlyStatistics && !isSeasonalForecast )
        return;

    bool isInsideSeason = false;

    if (isYearlyStatistics)
    {
        isInsideSeason = true;
    }

    if (isMonthlyStatistics)
    {
        isInsideSeason = (myDate.month == firstMonth);
    }

    if (isSeasonalForecast)
    {
        // interannual seasons
        if (firstMonth < 11)
        {
            if (myDate.month >= firstMonth && myDate.month <= firstMonth+2)
                isInsideSeason = true;
        }
        // NDJ or DJF
        else
        {
            int lastMonth = (firstMonth + 2) % 12;
            if (myDate.month >= firstMonth || myDate.month <= lastMonth)
                isInsideSeason = true;
        }
    }

    if (isInsideSeason)
    {
        // first date of season
        if (myDate.day == 1 && myDate.month == firstMonth)
        {
            if (currentIndex == NODATA)
                currentIndex = 0;
            else
                currentIndex++;
        }

        // sum of irrigations
        if (currentIndex != NODATA)
        {
            if (isEqual(irriSeries[unsigned(currentIndex)], NODATA))
                irriSeries[unsigned(currentIndex)] = float(myCase.output.dailyIrrigation);
            else
                irriSeries[unsigned(currentIndex)] += float(myCase.output.dailyIrrigation);
        }
    }
}


bool Crit1DProject::computeMonthlyForecast(unsigned int unitIndex, float irriRatio)
{
    logger.writeInfo(compUnitList[unitIndex].idCase);

    if (!forecastMeteoGrid->gridStructure().isEnsemble())
    {
        projectError = "Forecast grid is not Ensemble.";
        logger.writeError(projectError);
        return false;
    }

    nrYears = forecastMeteoGrid->gridStructure().nrMembers();
    if (nrYears < 1)
    {
        projectError = "Missing ensemble members.";
        logger.writeError(projectError);
        return false;
    }

    irriSeries.resize(unsigned(nrYears));
    precSeries.resize(unsigned(nrYears));
    for (unsigned int memberNr = 1; memberNr < unsigned(nrYears); memberNr++)
    {
        if (! computeUnit(unitIndex, memberNr))
        {
            logger.writeError(projectError);
            return false;
        }
    }

    // write output
    outputCsvFile << compUnitList[unitIndex].idCase.toStdString();
    outputCsvFile << "," << compUnitList[unitIndex].idCropClass.toStdString();

    // percentiles irrigation
    float percentile = sorting::percentile(irriSeries, nrYears, 5, true);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(irriSeries, nrYears, 25, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(irriSeries, nrYears, 50, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(irriSeries, nrYears, 75, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(irriSeries, nrYears, 95, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();

    // percentiles prec
    percentile = sorting::percentile(precSeries, nrYears, 5, true);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(precSeries, nrYears, 25, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(precSeries, nrYears, 50, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(precSeries, nrYears, 75, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(precSeries, nrYears, 95, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString() << "\n";

    outputCsvFile.flush();

    return true;
}


bool Crit1DProject::computeIrrigationStatistics(unsigned int index, float irriRatio)
{
    if (! computeUnit(index, 0))
    {
        projectError = "Computational Unit: " + compUnitList[index].idCase + " - " + projectError;
        logger.writeError(projectError);
        return false;
    }

    outputCsvFile << compUnitList[index].idCase.toStdString() << "," << compUnitList[index].idCrop.toStdString() << ",";
    outputCsvFile << compUnitList[index].idSoil.toStdString() << "," << compUnitList[index].idMeteo.toStdString();

    if (irriRatio < 0.001f)
    {
        // No irrigation
        outputCsvFile << ",0,0,0,0,0\n";
    }
    else
    {
        // irrigation percentiles
        float percentile = sorting::percentile(irriSeries, nrYears, 5, true);
        outputCsvFile << "," << percentile * irriRatio;
        percentile = sorting::percentile(irriSeries, nrYears, 25, false);
        outputCsvFile << "," << percentile * irriRatio;
        percentile = sorting::percentile(irriSeries, nrYears, 50, false);
        outputCsvFile << "," << percentile * irriRatio;
        percentile = sorting::percentile(irriSeries, nrYears, 75, false);
        outputCsvFile << "," << percentile * irriRatio;
        percentile = sorting::percentile(irriSeries, nrYears, 95, false);
        outputCsvFile << "," << percentile * irriRatio << "\n";
    }

    outputCsvFile.flush();
    return true;
}


bool Crit1DProject::setPercentileOutputCsv()
{
    QString outputCsvPath = getFilePath(outputCsvFileName);
    if (! QDir(outputCsvPath).exists())
    {
        QDir().mkdir(outputCsvPath);
    }

    outputCsvFile.open(outputCsvFileName.toStdString().c_str(), std::ios::out | std::ios::trunc);
    if ( outputCsvFile.fail())
    {
        logger.writeError("open failure: " + QString(strerror(errno)) + '\n');
        return false;
    }
    else
    {
        logger.writeInfo("Statistics output file (csv): " + outputCsvFileName + "\n");
    }

    if (isYearlyStatistics || isMonthlyStatistics || isSeasonalForecast)
    {
        outputCsvFile << "ID_CASE,CROP,SOIL,METEO,irri_05,irri_25,irri_50,irri_75,irri_95\n";
    }
    if (isEnsembleForecast)
    {
        outputCsvFile << "ID_CASE,CROP,irri_05,irri_25,irri_50,irri_75,irri_95,prec_05,prec_25,prec_50,prec_75,prec_95\n";
    }

    return true;
}


// initialize vector for annual values of irrigation
void Crit1DProject::initializeIrrigationStatistics(const Crit3DDate& firstDate, const Crit3DDate& lastDate)
{
    irriSeries.clear();

    nrYears = lastDate.year - firstDate.year +1;
    irriSeries.resize(unsigned(nrYears));

    for (int i = 0; i < nrYears; i++)
    {
        irriSeries[unsigned(i)] = NODATA;
    }
}


bool Crit1DProject::createDbState(QString &myError)
{
    // create db state
    QString dateStr = lastSimulationDate.addDays(1).toString("yyyy_MM_dd");
    QString outputDbPath = getFilePath(dbOutput.databaseName());
    QString dbStateName = outputDbPath + "state_" + dateStr + ".db";
    if (QFile::exists(dbStateName))
    {
        QFile::remove(dbStateName);
    }
    dbState = QSqlDatabase::addDatabase("QSQLITE", "state");
    dbState.setDatabaseName(dbStateName);

    if (! dbState.open())
    {
        myError = "Open state DB failed: " + dbState.lastError().text();
        return false;
    }

    // create tables
    QString queryString;
    QSqlQuery myQuery;
    queryString = "CREATE TABLE variables ( ID_CASE TEXT, DEGREE_DAYS REAL, DAYS_SINCE_IRR INTEGER )";
    myQuery = dbState.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating variables table \n" + myQuery.lastError().text();
        return false;
    }

    queryString = "CREATE TABLE waterContent ( ID_CASE TEXT, NR_LAYER INTEGER, WC REAL )";
    myQuery = dbState.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating waterContent table \n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit1DProject::restoreState(QString dbStateToRestoreName, QString &myError)
{
    if (!QFile::exists(dbStateToRestoreName))
    {
        myError = "DB state: " + dbStateToRestoreName + " does not exist";
        return false;
    }

    QSqlDatabase dbStateToRestore;
    if (QSqlDatabase::contains("stateToRestore"))
        dbStateToRestore = QSqlDatabase::database("stateToRestore");
    else
    {
        dbStateToRestore = QSqlDatabase::addDatabase("QSQLITE", "stateToRestore");
        dbStateToRestore.setDatabaseName(dbStateToRestoreName);
    }

    if (! dbStateToRestore.open())
    {
        myError = "Open state DB failed: " + dbStateToRestore.lastError().text();
        return false;
    }
    QSqlQuery qry(dbStateToRestore);
    qry.prepare( "SELECT * FROM variables WHERE ID_CASE = :id_case");
    qry.bindValue(":id_case", myCase.unit.idCase);

    if( !qry.exec() )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        double degreeDays;
        int daySinceIrr;
        if (qry.first())
        {
            if (!getValue(qry.value("DEGREE_DAYS"), &degreeDays))
            {
                myError = "DEGREE_DAYS not found";
                return false;
            }
            myCase.crop.degreeDays = degreeDays;
            if (!getValue(qry.value("DAYS_SINCE_IRR"), &daySinceIrr))
            {
                myCase.crop.daysSinceIrrigation = NODATA;
            }
            else
            {
                myCase.crop.daysSinceIrrigation = daySinceIrr;
            }
        }
        else
        {
            myError = "variables table: idCase not found";
            return false;
        }
    }
    qry.clear();
    qry.prepare( "SELECT * FROM waterContent WHERE ID_CASE = :id_case");
    qry.bindValue(":id_case", myCase.unit.idCase);

    if( !qry.exec() )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        int nrLayer = -1;
        double wc;
        while (qry.next())
        {
            if (!getValue(qry.value("NR_LAYER"), &nrLayer))
            {
                myError = "NR_LAYER not found";
                return false;
            }
            if (!getValue(qry.value("WC"), &wc))
            {
                myError = "WC not found";
                return false;
            }
            if (nrLayer < 0 || unsigned(nrLayer) >= myCase.soilLayers.size())
            {
                myError = "Invalid NR_LAYER";
                return false;
            }
            // check values
            if (nrLayer > 0)
            {
                wc = std::min(wc, myCase.soilLayers[unsigned(nrLayer)].SAT);
                wc = std::max(wc, myCase.soilLayers[unsigned(nrLayer)].HH);
            }
            myCase.soilLayers[unsigned(nrLayer)].waterContent = wc;
        }
        if (nrLayer == -1)
        {
            myError = "waterContent table: idCase not found";
            return false;
        }
    }

    dbStateToRestore.close();
    return true;
}


bool Crit1DProject::saveState(QString &myError)
{
    QString queryString;
    QSqlQuery qry(dbState);

    queryString = "INSERT INTO variables ( ID_CASE, DEGREE_DAYS, DAYS_SINCE_IRR ) VALUES ";
    queryString += "('" + myCase.unit.idCase + "'"
                + "," + QString::number(myCase.crop.degreeDays)
                + "," + QString::number(myCase.crop.daysSinceIrrigation) + ")";
    if( !qry.exec(queryString) )
    {
        myError = "Error in saving variables state:\n" + qry.lastError().text();
        return false;
    }
    qry.clear();

    queryString = "INSERT INTO waterContent ( ID_CASE, NR_LAYER, WC ) VALUES ";
    for (unsigned int i = 0; i<myCase.soilLayers.size(); i++)
    {
        queryString += "('" + myCase.unit.idCase + "'," + QString::number(i) + ","
                    + QString::number(myCase.soilLayers[i].waterContent) + ")";
        if (i < (myCase.soilLayers.size()-1))
            queryString += ",";
    }

    if( !qry.exec(queryString))
    {
        myError = "Error in saving waterContent state:\n" + qry.lastError().text();
        return false;
    }

    qry.clear();
    return true;
}


bool Crit1DProject::createOutputTable(QString &myError)
{
    QString queryString = "DROP TABLE '" + myCase.unit.idCase + "'";
    QSqlQuery myQuery = this->dbOutput.exec(queryString);

    queryString = "CREATE TABLE '" + myCase.unit.idCase + "'"
                  + " ( DATE TEXT, PREC REAL, IRRIGATION REAL, WATER_CONTENT REAL, SURFACE_WC REAL, "
                  + " AVAILABLE_WATER REAL, READILY_AW REAL, FRACTION_AW REAL, "
                  + " RUNOFF REAL, DRAINAGE REAL, LATERAL_DRAINAGE REAL, CAPILLARY_RISE REAL, "
                  + " ET0 REAL, TRANSP_MAX, TRANSP REAL, EVAP_MAX REAL, EVAP REAL, "
                  + " LAI REAL, ROOT_DEPTH REAL, BALANCE REAL";

    // specific depth variables
    for (unsigned int i = 0; i < waterContentDepth.size(); i++)
    {
        QString fieldName = "VWC_" + QString::number(waterContentDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < degreeOfSaturationDepth.size(); i++)
    {
        QString fieldName = "DEGSAT_" + QString::number(degreeOfSaturationDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < waterPotentialDepth.size(); i++)
    {
        QString fieldName = "WP_" + QString::number(waterPotentialDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
    {
        QString fieldName = "DEFICIT_" + QString::number(waterDeficitDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < awcDepth.size(); i++)
    {
        QString fieldName = "AWC_" + QString::number(awcDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < availableWaterDepth.size(); i++)
    {
        QString fieldName = "AW_" + QString::number(availableWaterDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < fractionAvailableWaterDepth.size(); i++)
    {
        QString fieldName = "FAW_" + QString::number(fractionAvailableWaterDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
    for (unsigned int i = 0; i < factorOfSafetyDepth.size(); i++)
    {
        QString fieldName = "FoS_" + QString::number(factorOfSafetyDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }

    queryString += ")";
    myQuery = this->dbOutput.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating table: " + myCase.unit.idCase + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


void Crit1DProject::updateOutput(Crit3DDate myDate, bool isFirst)
{
    if (isFirst)
    {
        outputString = "INSERT INTO '" + myCase.unit.idCase + "'"
                       + " (DATE, PREC, IRRIGATION, WATER_CONTENT, SURFACE_WC, "
                       + " AVAILABLE_WATER, READILY_AW, FRACTION_AW, "
                       + " RUNOFF, DRAINAGE, LATERAL_DRAINAGE, CAPILLARY_RISE, ET0, "
                       + " TRANSP_MAX, TRANSP, EVAP_MAX, EVAP, LAI, ROOT_DEPTH, BALANCE";

        // specific depth variables
        for (unsigned int i = 0; i < waterContentDepth.size(); i++)
        {
            QString fieldName = "VWC_" + QString::number(waterContentDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < degreeOfSaturationDepth.size(); i++)
        {
            QString fieldName = "DEGSAT_" + QString::number(degreeOfSaturationDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < waterPotentialDepth.size(); i++)
        {
            QString fieldName = "WP_" + QString::number(waterPotentialDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
        {
            QString fieldName = "DEFICIT_" + QString::number(waterDeficitDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < awcDepth.size(); i++)
        {
            QString fieldName = "AWC_" + QString::number(awcDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < availableWaterDepth.size(); i++)
        {
            QString fieldName = "AW_" + QString::number(availableWaterDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < fractionAvailableWaterDepth.size(); i++)
        {
            QString fieldName = "FAW_" + QString::number(fractionAvailableWaterDepth[i]);
            outputString += ", " + fieldName;
        }
        for (unsigned int i = 0; i < factorOfSafetyDepth.size(); i++)
        {
            QString fieldName = "FoS_" + QString::number(factorOfSafetyDepth[i]);
            outputString += ", " + fieldName;
        }

        outputString += ") VALUES ";
    }
    else
    {
        outputString += ",";
    }

    outputString += "('" + QString::fromStdString(myDate.toStdString()) + "'"
                    + "," + QString::number(myCase.output.dailyPrec)
                    + "," + QString::number(myCase.output.dailyIrrigation)
                    + "," + QString::number(myCase.output.dailySoilWaterContent, 'g', 4)
                    + "," + QString::number(myCase.output.dailySurfaceWaterContent, 'g', 3)
                    + "," + QString::number(myCase.output.dailyAvailableWater, 'g', 4)
                    + "," + QString::number(myCase.output.dailyReadilyAW, 'g', 4)
                    + "," + QString::number(myCase.output.dailyFractionAW, 'g', 3)
                    + "," + QString::number(myCase.output.dailySurfaceRunoff, 'g', 3)
                    + "," + QString::number(myCase.output.dailyDrainage, 'g', 3)
                    + "," + QString::number(myCase.output.dailyLateralDrainage, 'g', 3)
                    + "," + QString::number(myCase.output.dailyCapillaryRise, 'g', 3)
                    + "," + QString::number(myCase.output.dailyEt0, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxEvaporation, 'g', 3)
                    + "," + QString::number(myCase.output.dailyEvaporation, 'g', 3)
                    + "," + getOutputStringNullZero(myCase.crop.LAI)
                    + "," + getOutputStringNullZero(myCase.crop.roots.rootDepth)
                    + "," + QString::number(myCase.output.dailyBalance, 'g', 3);

    // specific depth variables
    for (unsigned int i = 0; i < waterContentDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getVolumetricWaterContent(waterContentDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < degreeOfSaturationDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getDegreeOfSaturation(degreeOfSaturationDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterPotentialDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterPotential(waterPotentialDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterDeficitSum(waterDeficitDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < awcDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterCapacitySum(awcDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < availableWaterDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getAvailableWaterSum(availableWaterDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < fractionAvailableWaterDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getFractionAW(fractionAvailableWaterDepth[i]), 'g', 3);
    }
    for (unsigned int i = 0; i < factorOfSafetyDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getSlopeStability(factorOfSafetyDepth[i]), 'g', 4);
    }

    outputString += ")";
}


bool Crit1DProject::saveOutput(QString &errorStr)
{
    QSqlQuery myQuery = dbOutput.exec(outputString);
    outputString.clear();

    if (myQuery.lastError().type() != QSqlError::NoError)
    {
        errorStr = "Error in saveOutput:\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


void Crit1DProject::closeAllDatabase()
{
    dbCrop.close();
    dbSoil.close();
    dbMeteo.close();
    dbForecast.close();
    dbOutput.close();
}


int Crit1DProject::openAllDatabase()
{
    closeAllDatabase();

    logger.writeInfo ("Crop DB: " + dbCropName);
    if (! QFile(dbCropName).exists())
    {
        projectError = "DB Crop file doesn't exist";
        closeAllDatabase();
        return ERROR_DBPARAMETERS;
    }

    dbCrop = QSqlDatabase::addDatabase("QSQLITE");
    dbCrop.setDatabaseName(dbCropName);
    if (! dbCrop.open())
    {
        projectError = "Open Crop DB failed: " + dbCrop.lastError().text();
        closeAllDatabase();
        return ERROR_DBPARAMETERS;
    }

    logger.writeInfo ("Soil DB: " + dbSoilName);
    if (! QFile(dbSoilName).exists())
    {
        projectError = "Soil DB file doesn't exist";
        closeAllDatabase();
        return ERROR_DBSOIL;
    }

    dbSoil = QSqlDatabase::addDatabase("QSQLITE", "soil");
    dbSoil.setDatabaseName(dbSoilName);
    if (! dbSoil.open())
    {
        projectError = "Open soil DB failed: " + dbSoil.lastError().text();
        closeAllDatabase();
        return ERROR_DBSOIL;
    }

    logger.writeInfo ("Meteo DB: " + dbMeteoName);
    if (! QFile(dbMeteoName).exists())
    {
        projectError = "Meteo points DB file doesn't exist";
        closeAllDatabase();
        return ERROR_DBMETEO_OBSERVED;
    }

    if (isXmlMeteoGrid)
    {
        observedMeteoGrid = new Crit3DMeteoGridDbHandler();
        if (! observedMeteoGrid->parseXMLGrid(dbMeteoName, &projectError))
        {
            return ERROR_XMLGRIDMETEO_OBSERVED;
        }
        if (! observedMeteoGrid->openDatabase(&projectError, "observed"))
        {
            return ERROR_DBMETEO_OBSERVED;
        }
        if (! observedMeteoGrid->loadCellProperties(&projectError))
        {
            return ERROR_PROPERTIES_DBMETEO_OBSERVED;
        }
    }
    else
    {
        dbMeteo = QSqlDatabase::addDatabase("QSQLITE", "meteo");
        dbMeteo.setDatabaseName(dbMeteoName);
        if (! dbMeteo.open())
        {
            projectError = "Open meteo DB failed: " + dbMeteo.lastError().text();
            closeAllDatabase();
            return ERROR_DBMETEO_OBSERVED;
        }
    }

    // meteo forecast
    if (isShortTermForecast || isEnsembleForecast)
    {
        logger.writeInfo ("Forecast DB: " + dbForecastName);
        if (! QFile(dbForecastName).exists())
        {
            projectError = "DBforecast file doesn't exist";
            closeAllDatabase();
            return ERROR_DBMETEO_FORECAST;
        }

        if (isXmlMeteoGrid)
        {
            forecastMeteoGrid = new Crit3DMeteoGridDbHandler();
            if (! forecastMeteoGrid->parseXMLGrid(dbForecastName, &projectError))
            {
                return ERROR_XMLGRIDMETEO_FORECAST;
            }
            if (! forecastMeteoGrid->openDatabase(&projectError, "forecast"))
            {
                return ERROR_DBMETEO_FORECAST;
            }
            if (! forecastMeteoGrid->loadCellProperties(&projectError))
            {
                return ERROR_PROPERTIES_DBMETEO_FORECAST;
            }
        }
        else
        {
            dbForecast = QSqlDatabase::addDatabase("QSQLITE", "forecast");
            dbForecast.setDatabaseName(dbForecastName);
            if (! dbForecast.open())
            {
                projectError = "Open forecast DB failed: " + dbForecast.lastError().text();
                closeAllDatabase();
                return ERROR_DBMETEO_FORECAST;
            }
        }
    }

    // output DB (not used in seasonal/monthly forecast)
    if ( !isMonthlyStatistics && !isSeasonalForecast && !isEnsembleForecast)
    {
        if (dbOutputName == "")
        {
            logger.writeError("Missing output DB");
                return ERROR_DBOUTPUT;
        }
        QFile::remove(dbOutputName);
        logger.writeInfo ("Output DB: " + dbOutputName);
        dbOutput = QSqlDatabase::addDatabase("QSQLITE", "output");
        dbOutput.setDatabaseName(dbOutputName);

        QString outputDbPath = getFilePath(dbOutputName);
        if (!QDir(outputDbPath).exists())
             QDir().mkdir(outputDbPath);

        if (! dbOutput.open())
        {
            projectError = "Open output DB failed: " + dbOutput.lastError().text();
            closeAllDatabase();
            return ERROR_DBOUTPUT;
        }
    }

    logger.writeInfo ("Computational units DB: " + dbComputationUnitsName);

    return CRIT1D_OK;
}



QString getOutputStringNullZero(double value)
{
    if (int(value) != int(NODATA))
        return QString::number(value, 'g', 3);
    else
        return QString::number(0);
}


bool setVariableDepth(const QList<QString>& depthList, std::vector<int>& variableDepth)
{
    int nrDepth = depthList.size();
    if (nrDepth > 0)
    {
        variableDepth.resize(unsigned(nrDepth));
        for (int i = 0; i < nrDepth; i++)
        {
            variableDepth[unsigned(i)] = depthList[i].toInt();
            if (variableDepth[unsigned(i)] <= 0)
                return false;
        }
    }

    return true;
}
