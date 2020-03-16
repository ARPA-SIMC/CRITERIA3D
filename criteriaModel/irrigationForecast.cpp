#include "irrigationForecast.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "soilDbTools.h"
#include "cropDbTools.h"
#include "criteria1DdbMeteo.h"
#include "water1D.h"
#include "utilities.h"

#include <QSqlError>
#include <QDate>
#include <QVariant>
#include <QSqlQuery>


Crit1DUnit::Crit1DUnit()
{
    this->idCase = "";
    this->idCrop = "";
    this->idSoil = "";
    this->idMeteo = "";

    this->idCropClass = "";
    this->idForecast = "";
    this->idSoilNumber = NODATA;
    this->idCropNumber = NODATA;
}


Crit1DIrrigationForecast::Crit1DIrrigationForecast()
{
    isSeasonalForecast = false;
    firstSeasonMonth = NODATA;
    nrSeasonalForecasts = 0;

    isShortTermForecast = false;
    daysOfForecast = NODATA;

    outputString = "";
}


// update values of annual irrigation
void Crit1DIrrigationForecast::updateSeasonalForecast(Crit3DDate myDate, int index)
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
            if (index == NODATA)
                index = 0;
            else
                index++;
        }

        // sum of irrigations
        if (index != NODATA)
        {
            if (int(seasonalForecasts[index]) == int(NODATA))
                seasonalForecasts[index] = myCase.output.dailyIrrigation;
            else
                seasonalForecasts[index] += myCase.output.dailyIrrigation;
        }
    }
}


bool Crit1DIrrigationForecast::runModel(const Crit1DUnit& myUnit, QString &myError)
{
    myCase.idCase = myUnit.idCase;

    if (! setSoil(myUnit.idSoil, myError))
        return false;

    if (! setMeteo(myUnit.idMeteo, myUnit.idForecast, &myError))
        return false;

    if (! loadCropParameters(myUnit.idCrop, &(myCase.myCrop), &(dbCrop), &myError))
        return false;

    if (! isSeasonalForecast)
    {
        if (! createOutputTable(myError))
            return false;
    }

    // set computation period (all meteo data)
    Crit3DDate myDate, firstDate, lastDate;
    long lastIndex = myCase.meteoPoint.nrObsDataDaysD-1;
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
            updateSeasonalForecast(myDate, indexSeasonalForecast);
        }
        else
        {
            prepareOutput(myDate, isFirstDay);
            isFirstDay = false;
        }
    }

    if (isSeasonalForecast)
        return true;
    else
        return saveOutput(myError);
}


bool Crit1DIrrigationForecast::setSoil(QString soilCode, QString &myError)
{
    if (! loadSoil(&dbSoil, soilCode, &(myCase.mySoil), soilTexture, &fittingOptions, &myError))
        return false;

    std::string errorString;
    if (! myCase.initializeSoil(errorString))
    {
        myError = QString::fromStdString(errorString);
        return false;
    }

    return true;
}


bool Crit1DIrrigationForecast::setMeteo(QString idMeteo, QString idForecast, QString *myError)
{
    QString queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo + "'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        QString idMeteo5char = getId5Char(idMeteo);
        queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo5char + "'";
        query = dbMeteo.exec(queryString);
        query.last();

        if (! query.isValid())
        {
            if (query.lastError().text() != "")
                *myError = "dbMeteo error: " + query.lastError().text();
            else
                *myError = "Missing meteo location:" + idMeteo;
            return(false);
        }
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

    queryString = "SELECT * FROM " + tableName + " ORDER BY [date]";
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
    QDate firstObsDate = query.value("date").toDate();
    query.last();
    QDate lastObsDate = query.value("date").toDate();

    unsigned nrDays = unsigned(firstObsDate.daysTo(lastObsDate)) + 1;

    // Is Forecast: increase nr of days
    if (this->isShortTermForecast)
        nrDays += unsigned(this->daysOfForecast);

    // Initialize data
    myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstObsDate));

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
        if (firstForecastDate != lastObsDate.addDays(1))
        {
            // previsioni indietro di un giorno: accettato ma tolgo un giorno
            if (firstForecastDate == lastObsDate)
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
        long lastObservedIndex = long(firstObsDate.daysTo(lastObsDate));
        for (unsigned long i = lastObservedIndex; i < unsigned(myCase.meteoPoint.nrObsDataDaysD); i++)
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
void Crit1DIrrigationForecast::initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate)
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


bool Crit1DIrrigationForecast::createOutputTable(QString &myError)
{
    QString queryString = "DROP TABLE '" + myCase.idCase + "'";
    QSqlQuery myQuery = this->dbOutput.exec(queryString);

    queryString = "CREATE TABLE '" + myCase.idCase + "'"
                  + " ( DATE TEXT, PREC REAL, IRRIGATION REAL, WATER_CONTENT REAL, SURFACE_WC REAL, "
                  + " RAW REAL, DEFICIT REAL, DRAINAGE REAL, RUNOFF REAL, ET0 REAL, "
                  + " TRANSP_MAX, TRANSP REAL, EVAP_MAX REAL, EVAP REAL, LAI REAL, ROOTDEPTH REAL )";
    myQuery = this->dbOutput.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        myError = "Error in creating table: " + myCase.idCase + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


void Crit1DIrrigationForecast::prepareOutput(Crit3DDate myDate, bool isFirst)
{
    if (isFirst)
    {
        outputString = "INSERT INTO '" + myCase.idCase + "'"
                       + " (DATE, PREC, IRRIGATION, WATER_CONTENT, SURFACE_WC, RAW, DEFICIT, DRAINAGE, RUNOFF, ET0,"
                       + " TRANSP_MAX, TRANSP, EVAP_MAX, EVAP, LAI, ROOTDEPTH) "
                       + " VALUES ";
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
                    + "," + QString::number(myCase.output.dailyCropAvailableWater, 'g', 4)
                    + "," + QString::number(myCase.output.dailyWaterDeficit, 'g', 4)
                    + "," + QString::number(myCase.output.dailyDrainage, 'g', 4)
                    + "," + QString::number(myCase.output.dailySurfaceRunoff, 'g', 4)
                    + "," + QString::number(myCase.output.dailyEt0, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyTranspiration, 'g', 3)
                    + "," + QString::number(myCase.output.dailyMaxEvaporation, 'g', 3)
                    + "," + QString::number(myCase.output.dailyEvaporation, 'g', 3)
                    + "," + getOutputStringNullZero(myCase.myCrop.LAI)
                    + "," + getOutputStringNullZero(myCase.myCrop.roots.rootDepth)
                    + ")";
}


bool Crit1DIrrigationForecast::saveOutput(QString &myError)
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


