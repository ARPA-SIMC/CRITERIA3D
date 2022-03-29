#include "commonConstants.h"
#include "criteria1DError.h"
#include "criteria1DProject.h"
#include "basicMath.h"
#include "soilDbTools.h"
#include "cropDbTools.h"
#include "cropDbQuery.h"
#include "criteria1DMeteo.h"
#include "water1D.h"
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
    dbUnitsName = "";

    projectError = "";

    unitList.clear();

    isXmlMeteoGrid = false;
    isSaveState = false;
    isRestart = false;

    isSeasonalForecast = false;
    isMonthlyForecast = false;
    isShortTermForecast = false;

    firstSeasonMonth = NODATA;
    daysOfForecast = NODATA;
    nrForecasts = NODATA;
    forecastIrr.clear();
    forecastPrec.clear();

    firstSimulationDate = QDate(1800,1,1);
    lastSimulationDate = QDate(1800,1,1);

    outputString = "";
    // specific outputs
    waterDeficitDepth.clear();
    waterContentDepth.clear();
    waterPotentialDepth.clear();
    availableWaterDepth.clear();
    fractionAvailableWaterDepth.clear();
    awcDepth.clear();
}


void Crit1DProject::closeProject()
{
    if (isProjectLoaded)
    {
        logger.writeInfo("Close Project...");
        closeAllDatabase();
        logFile.close();

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

    // unitList list
    dbUnitsName = projectSettings->value("db_units","").toString();
    if (dbUnitsName.left(1) == ".")
        dbUnitsName = path + dbUnitsName;

    if (dbUnitsName == "")
    {
        projectError = "Missing information on computational units";
        return false;
    }

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
    if (isRestart && ! isSaveState)
    {
        logger.writeInfo("WARNING: it is not possible to restart without save state (check file ini).");
    }

    projectSettings->endGroup();

    // FORECAST
    projectSettings->beginGroup("forecast");

        isSeasonalForecast = projectSettings->value("isSeasonalForecast", 0).toBool();
        isShortTermForecast = projectSettings->value("isShortTermForecast", 0).toBool();
        isMonthlyForecast = projectSettings->value("isMonthlyForecast", 0).toBool();
        if (isShortTermForecast || isMonthlyForecast)
        {
            daysOfForecast = projectSettings->value("daysOfForecast", 0).toInt();
            if (daysOfForecast == 0)
            {
                projectError = "Missing daysOfForecast.";
                return false;
            }
        }
        if (isSeasonalForecast)
        {
            firstSeasonMonth = projectSettings->value("firstMonth", 0).toInt();
            if (firstSeasonMonth == 0)
            {
                projectError = "Missing firstSeasonMonth.";
                return false;
            }
        }
        if ((isShortTermForecast && isMonthlyForecast)
            || (isShortTermForecast && isSeasonalForecast)
            || (isMonthlyForecast && isSeasonalForecast))
        {
            projectError = "Too many forecast types.";
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
    QStringList depthList;
    projectSettings->beginGroup("output");
        depthList = projectSettings->value("waterContent").toStringList();
        if (! setVariableDepth(depthList, waterContentDepth))
        {
            projectError = "Wrong water content depth in " + configFileName;
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

    if (! loadVanGenuchtenParameters(&dbSoil, soilTexture, &projectError))
        return ERROR_SOIL_PARAMETERS;

    if (! loadDriessenParameters(&dbSoil, soilTexture, &projectError))
        return ERROR_SOIL_PARAMETERS;

    // Computation unit list
    if (! readUnitList(dbUnitsName, unitList, projectError))
    {
        logger.writeError(projectError);
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(unitList.size()) + " distinct computation units.");

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
    if (isSeasonalForecast)
    {
        logger.writeInfo("first forecast month: " + QString::number(firstSeasonMonth));
    }
    else
    {
        logger.writeInfo("Nr of forecast days: " + QString::number(daysOfForecast));
    }
}


bool Crit1DProject::setSoil(QString soilCode, QString &myError)
{
    if (! loadSoil(&dbSoil, soilCode, &(myCase.mySoil), soilTexture, &(myCase.fittingOptions), &myError))
        return false;

    std::string errorString;
    if (! myCase.initializeSoil(errorString))
    {
        myError = QString::fromStdString(errorString);
        return false;
    }


    return true;
}


bool Crit1DProject::setMeteoXmlGrid(QString idMeteo, QString idForecast, unsigned int memberNr)
{
    unsigned row;
    unsigned col;
    unsigned nrDays = unsigned(firstSimulationDate.daysTo(lastSimulationDate)) + 1;

    if (!observedMeteoGrid->meteoGrid()->findMeteoPointFromId(&row, &col, idMeteo.toStdString()) )
    {
        projectError = "Missing observed meteo cell";
        return false;
    }

    if (!observedMeteoGrid->gridStructure().isFixedFields())
    {
        if (!observedMeteoGrid->loadGridDailyData(&projectError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            projectError = "Missing observed data";
            return false;
        }
    }
    else
    {
        if (!observedMeteoGrid->loadGridDailyDataFixedFields(&projectError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            if (projectError == "Missing MeteoPoint id")
            {
                projectError = "Missing observed meteo cell";
            }
            else
            {
                projectError = "Missing observed data";
            }
            return false;
        }
    }

    if (this->isShortTermForecast)
    {
        if (!this->forecastMeteoGrid->gridStructure().isFixedFields())
        {
            if (!this->forecastMeteoGrid->loadGridDailyData(&projectError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell";
                }
                else
                {
                    projectError = "Missing forecast data";
                }
                return false;
            }
        }
        else
        {
            if (!this->forecastMeteoGrid->loadGridDailyDataFixedFields(&projectError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell";
                }
                else
                {
                    projectError = "Missing forecast data";
                }
                return false;
            }
        }
        nrDays += unsigned(daysOfForecast);
    }

    if (this->isMonthlyForecast)
    {
        if (this->forecastMeteoGrid->gridStructure().isFixedFields())
        {
            projectError = "DB grid fixed fields: not available";
            return false;
        }
        else
        {
            if (!this->forecastMeteoGrid->loadGridDailyDataEnsemble(&projectError, idForecast, int(memberNr), lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (projectError == "Missing MeteoPoint id")
                {
                    projectError = "Missing forecast meteo cell";
                }
                else
                {
                    projectError = "Missing forecast data";
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
    if (isShortTermForecast || isMonthlyForecast)
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
    QString queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo + "'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        projectError = "Missing meteo location: " + idMeteo;
        return false;
    }

    QString tableName = query.value("table_name").toString();

    double myLat, myLon;
    if (getValue(query.value(("latitude")), &myLat))
        myCase.meteoPoint.latitude = myLat;
    else
    {
        projectError = "Missing latitude in idMeteo: " + idMeteo;
        return false;
    }

    if (getValue(query.value(("longitude")), &myLon))
        myCase.meteoPoint.longitude = myLon;
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
                    + firstDate.toString("yyyy-MM-dd") + "' AND '" + lastDate.toString("yyyy-MM-dd") + "'";
        query = this->dbMeteo.exec(queryString);
    }

    // Forecast: increase nr of days
    if (this->isShortTermForecast)
        nrDays += unsigned(this->daysOfForecast);

    // Initialize data
    myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstDate));

    // Read observed data
    if (! readDailyDataCriteria1D(&query, &(myCase.meteoPoint), &projectError)) return false;

    // Add Short-Term forecast
    if (this->isShortTermForecast)
    {
        queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idForecast + "'";
        query = dbForecast.exec(queryString);
        query.last();

        if (! query.isValid())
        {
            if (query.lastError().text() != "")
                projectError = "dbForecast error: " + query.lastError().text();
            else
                projectError = "Missing forecast location:" + idForecast;
            return false;
        }
        QString tableNameForecast = query.value("table_name").toString();

        query.clear();
        queryString = "SELECT * FROM " + tableNameForecast + " ORDER BY [date]";
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
        if (! readDailyDataCriteria1D(&query, &(myCase.meteoPoint), &projectError)) return false;

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


bool Crit1DProject::computeUnit(const Crit1DUnit& myUnit)
{
    myCase.unit = myUnit;
    return computeCase(0);
}


bool Crit1DProject::computeUnit(unsigned int unitIndex, unsigned int memberNr)
{
    myCase.unit = unitList[unitIndex];
    return computeCase(memberNr);
}


// use memberNr = 0 for deterministic run
bool Crit1DProject::computeCase(unsigned int memberNr)
{
    myCase.fittingOptions.useWaterRetentionData = myCase.unit.useWaterRetentionData;

    if (! loadCropParameters(&dbCrop, myCase.unit.idCrop, &(myCase.crop), &projectError))
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
        projectError = "Missing meteo data.";
        return false;
    }

    if ((! isSeasonalForecast) && (! isMonthlyForecast))
    {
        if (! createOutputTable(projectError))
            return false;
    }

    // set computation period (all meteo data)
    Crit3DDate myDate, firstDate, lastDate;
    unsigned long lastIndex = unsigned(myCase.meteoPoint.nrObsDataDaysD-1);
    firstDate = myCase.meteoPoint.obsDataD[0].date;
    lastDate = myCase.meteoPoint.obsDataD[lastIndex].date;

    if (isSeasonalForecast)
    {
        initializeSeasonalForecast(firstDate, lastDate);
    }
    int indexSeasonalForecast = NODATA;

    // initialize crop
    unsigned nrLayers = unsigned(myCase.soilLayers.size());
    myCase.crop.initialize(myCase.meteoPoint.latitude, nrLayers,
                             myCase.mySoil.totalDepth, getDoyFromDate(firstDate));

    // initialize water content
    myCase.initializeWaterContent(firstDate);

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
        if (isSeasonalForecast)
        {
            updateSeasonalForecastOutput(myDate, indexSeasonalForecast);
        }
        else if (isMonthlyForecast)
        {
            updateMonthlyForecastOutput(myDate, memberNr);
        }
        else
        {
            prepareOutput(myDate, isFirstDay);
            isFirstDay = false;
        }
    }

    if (isSaveState)
    {
        if (! saveState(projectError))
            return false;
    }

    if (isSeasonalForecast || isMonthlyForecast)
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

    if (isSeasonalForecast || isMonthlyForecast)
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
        if (! createState(projectError))
        {
            logger.writeError(projectError);
            return ERROR_DB_STATE;
        }
    }

    try
    {
        for (unsigned int i = 0; i < unitList.size(); i++)
        {
            // is numerical
            //QString isNumerical = unitList[i].isNumericalInfiltration? "true" : "false";
            //logger.writeInfo("is numerical: " + isNumerical);

            // CROP
            unitList[i].idCrop = getCropFromClass(&dbCrop, "crop_class", "id_class",
                                                         unitList[i].idCropClass, &projectError).toUpper();
            if (unitList[i].idCrop == "")
            {
                logger.writeInfo("Unit " + unitList[i].idCase + " " + unitList[i].idCropClass + " ***** missing CROP *****");
                isErrorCrop = true;
                continue;
            }

            // IRRI_RATIO
            float irriRatio = getIrriRatioFromClass(&dbCrop, "crop_class", "id_class",
                                                            unitList[i].idCropClass, &projectError);
            if ((isSeasonalForecast || isMonthlyForecast || isShortTermForecast) && (int(irriRatio) == int(NODATA)))
            {
                logger.writeInfo("Unit " + unitList[i].idCase + " " + unitList[i].idCropClass + " ***** missing IRRIGATION RATIO *****");
                continue;
            }

            // SOIL
            unitList[i].idSoil = getIdSoilString(&dbSoil, unitList[i].idSoilNumber, &projectError);
            if (unitList[i].idSoil == "")
            {
                logger.writeInfo("Unit " + unitList[i].idCase + " Soil nr." + QString::number(unitList[i].idSoilNumber) + " ***** missing SOIL *****");
                isErrorSoil = true;
                continue;
            }

            if (isSeasonalForecast)
            {
                if (computeSeasonalForecast(i, irriRatio))
                    nrUnitsComputed++;
                else
                    isErrorModel = true;
            }
            else
            {
                if (isMonthlyForecast)
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
                        projectError = "Computational Unit: " + unitList[i].idCase + "\n" + projectError;
                        logger.writeError(projectError);
                        isErrorModel = true;
                    }
                }
            }
        }

        if (isSeasonalForecast || isMonthlyForecast)
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
    else if (nrUnitsComputed < unitList.size())
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


// update values of monthly forecast
void Crit1DProject::updateMonthlyForecastOutput(Crit3DDate myDate, unsigned int memberNr)
{
    QDate myQdate = getQDate(myDate);

    if (myQdate == lastSimulationDate)
    {
        forecastIrr[memberNr] = 0;
        forecastPrec[memberNr] = 0;
    }
    else if (myQdate > lastSimulationDate)
    {
        forecastIrr[memberNr] += float(myCase.output.dailyIrrigation);
        forecastPrec[memberNr] += float(myCase.output.dailyPrec);
    }
}


// update values of annual irrigation
void Crit1DProject::updateSeasonalForecastOutput(Crit3DDate myDate, int &indexForecast)
{
    bool isInsideSeason = false;

    // normal seasons
    if (firstSeasonMonth < 11)
    {
        if (myDate.month >= firstSeasonMonth && myDate.month <= firstSeasonMonth+2)
            isInsideSeason = true;
    }
    // NDJ or DJF
    else
    {
        int lastMonth = (firstSeasonMonth + 2) % 12;
        if (myDate.month >= firstSeasonMonth || myDate.month <= lastMonth)
            isInsideSeason = true;
    }

    if (isInsideSeason)
    {
        // first date of season
        if (myDate.day == 1 && myDate.month == firstSeasonMonth)
        {
            if (indexForecast == NODATA)
                indexForecast = 0;
            else
                indexForecast++;
        }

        // sum of irrigations
        if (indexForecast != NODATA)
        {
            unsigned int i = unsigned(indexForecast);
            if (int(forecastIrr[i]) == int(NODATA))
                forecastIrr[i] = float(myCase.output.dailyIrrigation);
            else
                forecastIrr[i] += float(myCase.output.dailyIrrigation);
        }
    }
}


bool Crit1DProject::computeMonthlyForecast(unsigned int unitIndex, float irriRatio)
{
    logger.writeInfo(unitList[unitIndex].idCase);

    if (!forecastMeteoGrid->gridStructure().isEnsemble())
    {
        projectError = "Forecast grid is not Ensemble.";
        logger.writeError(projectError);
        return false;
    }

    nrForecasts = forecastMeteoGrid->gridStructure().nrMembers();
    if (nrForecasts < 1)
    {
        projectError = "Missing ensemble members.";
        logger.writeError(projectError);
        return false;
    }

    forecastIrr.resize(unsigned(nrForecasts));
    forecastPrec.resize(unsigned(nrForecasts));
    for (unsigned int memberNr = 1; memberNr < unsigned(nrForecasts); memberNr++)
    {
        if (! computeUnit(unitIndex, memberNr))
        {
            logger.writeError(projectError);
            return false;
        }
    }

    // write output
    outputCsvFile << unitList[unitIndex].idCase.toStdString();
    outputCsvFile << "," << unitList[unitIndex].idCropClass.toStdString();

    // percentiles irrigation
    float percentile = sorting::percentile(forecastIrr, &(nrForecasts), 5, true);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 25, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 50, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 75, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 95, false);
    outputCsvFile << "," << QString::number(double(percentile * irriRatio), 'f', 1).toStdString();

    // percentiles prec
    percentile = sorting::percentile(forecastPrec, &(nrForecasts), 5, true);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastPrec, &(nrForecasts), 25, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastPrec, &(nrForecasts), 50, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastPrec, &(nrForecasts), 75, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString();
    percentile = sorting::percentile(forecastPrec, &(nrForecasts), 95, false);
    outputCsvFile << "," << QString::number(double(percentile), 'f', 1).toStdString() << "\n";

    outputCsvFile.flush();

    return true;
}


bool Crit1DProject::computeSeasonalForecast(unsigned int index, float irriRatio)
{
    if (irriRatio < 0.001f)
    {
        // No irrigation: nothing to do
        outputCsvFile << unitList[index].idCase.toStdString() << "," << unitList[index].idCrop.toStdString() << ",";
        outputCsvFile << unitList[index].idSoil.toStdString() << "," << unitList[index].idMeteo.toStdString();
        outputCsvFile << ",0,0,0,0,0\n";
        return true;
    }

    if (! computeUnit(index, 0))
    {
        logger.writeError(projectError);
        return false;
    }

    outputCsvFile << unitList[index].idCase.toStdString() << "," << unitList[index].idCrop.toStdString() << ",";
    outputCsvFile << unitList[index].idSoil.toStdString() << "," << unitList[index].idMeteo.toStdString();
    // percentiles
    float percentile = sorting::percentile(forecastIrr, &(nrForecasts), 5, true);
    outputCsvFile << "," << percentile * irriRatio;
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 25, false);
    outputCsvFile << "," << percentile * irriRatio;
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 50, false);
    outputCsvFile << "," << percentile * irriRatio;
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 75, false);
    outputCsvFile << "," << percentile * irriRatio;
    percentile = sorting::percentile(forecastIrr, &(nrForecasts), 95, false);
    outputCsvFile << "," << percentile * irriRatio << "\n";
    outputCsvFile.flush();

    return true;
}


bool Crit1DProject::setPercentileOutputCsv()
{
    if (isSeasonalForecast || isMonthlyForecast)
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
            logger.writeInfo("Output file: " + outputCsvFileName + "\n");
        }

        if (isSeasonalForecast)
        {
            outputCsvFile << "ID_CASE,CROP,SOIL,METEO,p5,p25,p50,p75,p95\n";
        }
        if (isMonthlyForecast)
        {
            outputCsvFile << "ID_CASE,CROP,irr5,irr25,irr50,irr75,irr95,prec5,prec25,prec50,prec75,prec95\n";
        }
    }

    return true;
}


// alloc memory for annual values of irrigation
void Crit1DProject::initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate)
{
    if (isSeasonalForecast)
    {
        forecastIrr.clear();

        nrForecasts = lastDate.year - firstDate.year +1;
        forecastIrr.resize(unsigned(nrForecasts));

        for (unsigned int i = 0; i < unsigned(nrForecasts); i++)
        {
            forecastIrr[i] = NODATA;
        }
    }
}


bool Crit1DProject::createState(QString &myError)
{
    // create db state
    QString date = lastSimulationDate.addDays(1).toString("yyyy_MM_dd");
    QString outputDbPath = getFilePath(dbOutput.databaseName());
    QString dbStateName = outputDbPath + "state_" + date + ".db";
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
        myError = "DB state: " +dbStateToRestoreName+" does not exist";
        return false;
    }
    QSqlDatabase dbStateToRestore = QSqlDatabase::addDatabase("QSQLITE", "stateToRestore");
    dbStateToRestore.setDatabaseName(dbStateToRestoreName);

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
        if (qry.next())
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
            myCase.soilLayers[unsigned(nrLayer)].waterContent = wc;
        }
        if (nrLayer == -1)
        {
            myError = "waterContent table: idCase not found";
            return false;
        }
    }

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
        QString fieldName = "SWC_" + QString::number(waterContentDepth[i]);
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

    queryString += ")";
    myQuery = this->dbOutput.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating table: " + myCase.unit.idCase + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


void Crit1DProject::prepareOutput(Crit3DDate myDate, bool isFirst)
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
            QString fieldName = "SWC_" + QString::number(waterContentDepth[i]);
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
        outputString += "," + QString::number(myCase.getWaterContent(waterContentDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterPotentialDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterPotential(waterPotentialDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterDeficit(waterDeficitDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < awcDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterCapacity(awcDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < availableWaterDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getAvailableWater(availableWaterDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < fractionAvailableWaterDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getFractionAW(fractionAvailableWaterDepth[i]), 'g', 3);
    }

    outputString += ")";
}


bool Crit1DProject::saveOutput(QString &myError)
{
    QSqlQuery myQuery = dbOutput.exec(outputString);
    outputString.clear();

    if (myQuery.lastError().type() != QSqlError::NoError)
    {
        myError = "Error in saving output:\n" + myQuery.lastError().text();
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
    if (isShortTermForecast || isMonthlyForecast)
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
    if ((! isSeasonalForecast) && (! isMonthlyForecast))
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

    // db units
    logger.writeInfo ("Computational units DB: " + dbUnitsName);

    return CRIT1D_OK;
}



QString getOutputStringNullZero(double value)
{
    if (int(value) != int(NODATA))
        return QString::number(value, 'g', 3);
    else
        return QString::number(0);
}


bool setVariableDepth(QList<QString>& depthList, std::vector<int>& variableDepth)
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

