#include "criteria1DSimulation.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "soilDbTools.h"
#include "cropDbTools.h"
#include "criteria1DMeteo.h"
#include "water1D.h"
#include "utilities.h"

#include <QSqlError>
#include <QDate>
#include <QVariant>
#include <QSqlQuery>
#include <QFile>
#include <QDebug>

Crit1DSimulation::Crit1DSimulation()
{
    isXmlGrid = false;
    isSeasonalForecast = false;
    isSaveState = false;
    isRestart = false;
    firstSeasonMonth = NODATA;
    nrSeasonalForecasts = 0;

    isShortTermForecast = false;
    daysOfForecast = NODATA;
    firstSimulationDate = QDate(1800,1,1);
    lastSimulationDate = QDate(1800,1,1);

    outputString = "";
    waterDeficitDepth.clear();
    waterContentDepth.clear();
    waterPotentialDepth.clear();
}


// update values of annual irrigation
void Crit1DSimulation::updateSeasonalForecast(Crit3DDate myDate, int* index)
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
            if (*index == NODATA)
                *index = 0;
            else
                (*index)++;
        }

        // sum of irrigations
        if (*index != NODATA)
        {
            unsigned int i = unsigned(*index);
            if (int(seasonalForecasts[i]) == int(NODATA))
                seasonalForecasts[i] = float(myCase.output.dailyIrrigation);
            else
                seasonalForecasts[i] += float(myCase.output.dailyIrrigation);
        }
    }
}


bool Crit1DSimulation::runModel(const Crit1DUnit& myUnit, QString &myError)
{
    myCase.unit = myUnit;
    myCase.fittingOptions.useWaterRetentionData = myUnit.useWaterRetentionData;

    if (! loadCropParameters(&dbCrop, myUnit.idCrop, &(myCase.myCrop), &myError))
        return false;

    if (! setSoil(myUnit.idSoil, myError))
        return false;

    if (isXmlGrid)
    {
        if (! setMeteoXmlGrid(myUnit.idMeteo, myUnit.idForecast, &myError))
            return false;
    }
    else
    {
        if (! setMeteoSqlite(myUnit.idMeteo, myUnit.idForecast, &myError))
            return false;
    }

    // check meteo data
    if (myCase.meteoPoint.nrObsDataDaysD == 0)
    {
        myError = "Missing meteo data.";
        return false;
    }

    if (! isSeasonalForecast)
    {
        if (! createOutputTable(myError))
            return false;
    }

    // set computation period (all meteo data)
    Crit3DDate myDate, firstDate, lastDate;
    unsigned long lastIndex = unsigned(myCase.meteoPoint.nrObsDataDaysD-1);
    firstDate = myCase.meteoPoint.obsDataD[0].date;
    lastDate = myCase.meteoPoint.obsDataD[lastIndex].date;

    if (isSeasonalForecast) initializeSeasonalForecast(firstDate, lastDate);
    int indexSeasonalForecast = NODATA;

    // initialize crop
    unsigned nrLayers = unsigned(myCase.soilLayers.size());
    myCase.myCrop.initialize(myCase.meteoPoint.latitude, nrLayers,
                             myCase.mySoil.totalDepth, getDoyFromDate(firstDate));

    std::string errorString;
    bool isFirstDay = true;

    // restart
    if (isRestart)
    {

        QString outputDbPath = getFilePath(dbOutput.databaseName());
        QString stateDbName = outputDbPath + "state_"+firstSimulationDate.toString("yyyy_MM_dd")+".db";
        if (! restoreState(stateDbName, myError))
        {
            return false;
        }

        double currentWaterTable = double(myCase.meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
        if (! myCase.myCrop.restore(myDate, myCase.meteoPoint.latitude, myCase.soilLayers, currentWaterTable, errorString))
        {
            myError = QString::fromStdString(errorString);
            return false;
        }
    }

    // daily cycle
    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! myCase.computeDailyModel(myDate, errorString))
        {
            myError = QString::fromStdString(errorString);
            return false;
        }

        // output
        if (isSeasonalForecast)
        {
            updateSeasonalForecast(myDate, &indexSeasonalForecast);
        }
        else
        {
            prepareOutput(myDate, isFirstDay);
            isFirstDay = false;
        }
    }

    if (isSaveState)
    {
        if (! saveState(myError))
            return false;
    }

    if (isSeasonalForecast)
        return true;
    else
        return saveOutput(myError);
}


bool Crit1DSimulation::setSoil(QString soilCode, QString &myError)
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


bool Crit1DSimulation::setMeteoXmlGrid(QString idMeteo, QString idForecast, QString *myError)
{

    unsigned row;
    unsigned col;
    unsigned nrDays = unsigned(firstSimulationDate.daysTo(lastSimulationDate)) + 1;

    if (!this->observedMeteoGrid->meteoGrid()->findMeteoPointFromId(&row, &col, idMeteo.toStdString()) )
    {
        *myError = "Missing observed meteo cell";
        return false;
    }

    if (!this->observedMeteoGrid->gridStructure().isFixedFields())
    {
        if (!this->observedMeteoGrid->loadGridDailyData(myError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            *myError = "Missing observed data";
            return false;
        }
    }
    else
    {
        if (!this->observedMeteoGrid->loadGridDailyDataFixedFields(myError, idMeteo, firstSimulationDate, lastSimulationDate))
        {
            if (*myError == "Missing MeteoPoint id")
            {
                *myError = "Missing observed meteo cell";
            }
            else
            {
                *myError = "Missing observed data";
            }
            return false;
        }
    }

    if (this->isShortTermForecast)
    {
        if (!this->forecastMeteoGrid->gridStructure().isFixedFields())
        {
            if (!this->forecastMeteoGrid->loadGridDailyData(myError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (*myError == "Missing MeteoPoint id")
                {
                    *myError = "Missing forecast meteo cell";
                }
                else
                {
                    *myError = "Missing forecast data";
                }
                return false;
            }
        }
        else
        {
            if (!this->forecastMeteoGrid->loadGridDailyDataFixedFields(myError, idForecast, lastSimulationDate.addDays(1), lastSimulationDate.addDays(daysOfForecast)))
            {
                if (*myError == "Missing MeteoPoint id")
                {
                    *myError = "Missing forecast meteo cell";
                }
                else
                {
                    *myError = "Missing forecast data";
                }
                return false;
            }
        }
        nrDays += this->daysOfForecast;
    }

    myCase.meteoPoint.latitude = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->latitude;
    myCase.meteoPoint.longitude = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->longitude;
    myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstSimulationDate));

    float tmin, tmax, tavg, prec;
    int lastIndex = firstSimulationDate.daysTo(lastSimulationDate)+1;
    for (int i = 0; i < lastIndex; i++)
    {
        Crit3DDate myDate = getCrit3DDate(firstSimulationDate.addDays(i));
        tmin = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);

        tmax = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);

        tavg = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
        if (tavg == NODATA)
        {
            tavg = (tmax + tmin)/2;
        }
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);

        prec = this->observedMeteoGrid->meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyPrecipitation);
        myCase.meteoPoint.setMeteoPointValueD(myDate, dailyPrecipitation, prec);
    }
    if (isShortTermForecast)
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
            if (tavg == NODATA)
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


bool Crit1DSimulation::setMeteoSqlite(QString idMeteo, QString idForecast, QString *myError)
{
    QString queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo + "'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        *myError = "Missing meteo location: " + idMeteo;
        return false;
    }

    QString tableName = query.value("table_name").toString();

    double myLat, myLon;
    if (getValue(query.value(("latitude")), &myLat))
        myCase.meteoPoint.latitude = myLat;
    else
    {
        *myError = "Missing latitude in idMeteo: " + idMeteo;
        return false;
    }

    if (getValue(query.value(("longitude")), &myLon))
        myCase.meteoPoint.longitude = myLon;

    queryString = "SELECT * FROM '" + tableName + "' ORDER BY [date]";
    query = this->dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().text() != "")
            *myError = "dbMeteo error: " + query.lastError().text();
        else
            *myError = "Missing meteo table:" + tableName;
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
            *myError = "Missing meteo data: required first date " + firstSimulationDate.toString("yyyy-MM-dd");
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
            *myError = "Missing meteo data: required last date " + lastSimulationDate.toString("yyyy-MM-dd");
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
    if (! readDailyDataCriteria1D(&query, &(myCase.meteoPoint), myError)) return false;

    // Add Short-Term forecast
    if (this->isShortTermForecast)
    {
        queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idForecast + "'";
        query = dbForecast.exec(queryString);
        query.last();

        if (! query.isValid())
        {
            QString idForecast5char = getId5Char(idForecast);
            queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idForecast5char + "'";
            query = dbForecast.exec(queryString);
            query.last();

            if (! query.isValid())
            {
                if (query.lastError().text() != "")
                    *myError = "dbForecast error: " + query.lastError().text();
                else
                    *myError = "Missing forecast location:" + idForecast;
                return false;
            }
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
                *myError = "dbForecast error: " + query.lastError().text();
            else
                *myError = "Missing forecast table:" + tableName;
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
                *myError = "The forecast date doesn't match with observed data.";
                return false;
            }
        }

        // Read forecast data
        if (! readDailyDataCriteria1D(&query, &(myCase.meteoPoint), myError)) return false;

        // fill temperature (only forecast)
        // estende il dato precedente se mancante
        float previousTmin = NODATA;
        float previousTmax = NODATA;
        long lastIndex = long(firstDate.daysTo(lastDate));
        for (unsigned long i = lastIndex; i < unsigned(myCase.meteoPoint.nrObsDataDaysD); i++)
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


// alloc memory for annual values of irrigation
void Crit1DSimulation::initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate)
{
    if (isSeasonalForecast)
    {
        seasonalForecasts.clear();

        nrSeasonalForecasts = lastDate.year - firstDate.year +1;
        seasonalForecasts.resize(unsigned(nrSeasonalForecasts));

        for (unsigned int i = 0; i < unsigned(nrSeasonalForecasts); i++)
        {
            seasonalForecasts[i] = NODATA;
        }
    }
}


bool Crit1DSimulation::createState(QString &myError)
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


bool Crit1DSimulation::restoreState(QString dbStateToRestoreName, QString &myError)
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
            myCase.myCrop.degreeDays = degreeDays;
            if (!getValue(qry.value("DAYS_SINCE_IRR"), &daySinceIrr))
            {
                myCase.myCrop.daysSinceIrrigation = NODATA;
            }
            else
            {
                myCase.myCrop.daysSinceIrrigation = daySinceIrr;
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
            if (nrLayer<0 || (unsigned)nrLayer>=myCase.soilLayers.size())
            {
                myError = "Invalid NR_LAYER";
                return false;
            }
            myCase.soilLayers[nrLayer].waterContent = wc;
        }
        if (nrLayer == -1)
        {
            myError = "waterContent table: idCase not found";
            return false;
        }
    }

    return true;
}


bool Crit1DSimulation::saveState(QString &myError)
{
    QString queryString;
    QSqlQuery qry(dbState);

    queryString = "INSERT INTO variables ( ID_CASE, DEGREE_DAYS, DAYS_SINCE_IRR ) VALUES ";
    queryString += "('" + myCase.unit.idCase + "'"
                + "," + QString::number(myCase.myCrop.degreeDays)
                + "," + QString::number(myCase.myCrop.daysSinceIrrigation) + ")";
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


bool Crit1DSimulation::createOutputTable(QString &myError)
{
    QString queryString = "DROP TABLE '" + myCase.unit.idCase + "'";
    QSqlQuery myQuery = this->dbOutput.exec(queryString);

    queryString = "CREATE TABLE '" + myCase.unit.idCase + "'"
                  + " ( DATE TEXT, PREC REAL, IRRIGATION REAL, WATER_CONTENT REAL, SURFACE_WC REAL, "
                  + " AVAILABLE_WATER REAL, READILY_AW REAL, FRACTION_AW REAL, "
                  + " RUNOFF REAL, DRAINAGE REAL, LATERAL_DRAINAGE REAL, CAPILLARY_RISE REAL, "
                  + " ET0 REAL, TRANSP_MAX, TRANSP REAL, EVAP_MAX REAL, EVAP REAL, LAI REAL, ROOT_DEPTH REAL";

    // specific depth variables
    for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
    {
        QString fieldName = "DEFICIT_" + QString::number(waterDeficitDepth[i]);
        queryString += ", " + fieldName + " REAL";
    }
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

    queryString += ")";
    myQuery = this->dbOutput.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating table: " + myCase.unit.idCase + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


void Crit1DSimulation::prepareOutput(Crit3DDate myDate, bool isFirst)
{
    if (isFirst)
    {
        outputString = "INSERT INTO '" + myCase.unit.idCase + "'"
                       + " (DATE, PREC, IRRIGATION, WATER_CONTENT, SURFACE_WC, "
                       + " AVAILABLE_WATER, READILY_AW, FRACTION_AW, "
                       + " RUNOFF, DRAINAGE, LATERAL_DRAINAGE, CAPILLARY_RISE, "
                       + " ET0, TRANSP_MAX, TRANSP, EVAP_MAX, EVAP, LAI, ROOT_DEPTH";

        // specific depth variables
        for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
        {
            QString fieldName = "DEFICIT_" + QString::number(waterDeficitDepth[i]);
            outputString += ", " + fieldName;
        }
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

        outputString += ") VALUES ";
    }
    else
    {
        outputString += ",";
    }

    outputString += "('" + QString::fromStdString(myDate.toStdString()) + "'"
                    + "," + QString::number(myCase.output.dailyPrec, 'g', 4)
                    + "," + QString::number(myCase.output.dailyIrrigation, 'g', 4)
                    + "," + QString::number(myCase.output.dailySoilWaterContent, 'g', 5)
                    + "," + QString::number(myCase.output.dailySurfaceWaterContent, 'g', 4)
                    + "," + QString::number(myCase.output.dailyAvailableWater, 'g', 4)
                    + "," + QString::number(myCase.output.dailyReadilyAW, 'g', 4)
                    + "," + QString::number(myCase.output.dailyFractionAW, 'g', 4)
                    + "," + QString::number(myCase.output.dailySurfaceRunoff, 'g', 4)
                    + "," + QString::number(myCase.output.dailyDrainage, 'g', 4)
                    + "," + QString::number(myCase.output.dailyLateralDrainage, 'g', 4)
                    + "," + QString::number(myCase.output.dailyCapillaryRise, 'g', 4)
                    + "," + QString::number(myCase.output.dailyEt0, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxEvaporation, 'g', 3)
                    + "," + QString::number(myCase.output.dailyEvaporation, 'g', 3)
                    + "," + getOutputStringNullZero(myCase.myCrop.LAI)
                    + "," + getOutputStringNullZero(myCase.myCrop.roots.rootDepth);

    // specific depth variables
    for (unsigned int i = 0; i < waterDeficitDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getSoilWaterDeficit(waterDeficitDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterContentDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterContent(waterContentDepth[i]), 'g', 4);
    }
    for (unsigned int i = 0; i < waterPotentialDepth.size(); i++)
    {
        outputString += "," + QString::number(myCase.getWaterPotential(waterPotentialDepth[i]), 'g', 4);
    }

    outputString += ")";
}


bool Crit1DSimulation::saveOutput(QString &myError)
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



QString getOutputStringNullZero(double value)
{
    if (int(value) != int(NODATA))
        return QString::number(value, 'g', 4);
    else
        return QString::number(0);
}


QString getId5Char(QString id)
{
    if (id.length() < 5)
        id = "0" + id;

    return id;
}


