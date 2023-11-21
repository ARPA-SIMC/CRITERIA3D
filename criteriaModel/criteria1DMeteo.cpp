#include <QSqlQuery>
#include <QSqlError>
#include <QVariant>
#include <QDate>
#include <QUuid>

#include "commonConstants.h"
#include "crit3dDate.h"
#include "criteria1DMeteo.h"
#include "utilities.h"
#include "meteoPoint.h"
#include "basicMath.h"


bool openDbMeteo(QString dbName, QSqlDatabase &dbMeteo, QString &error)
{

    dbMeteo = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbMeteo.setDatabaseName(dbName);

    if (!dbMeteo.open())
    {
       error = "Connection with database fail: " + dbName;
       return false;
    }

    return true;
}


bool getMeteoPointList(const QSqlDatabase &dbMeteo, QList<QString> &idMeteoList, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT id_meteo FROM point_properties";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.first();

    if (! query.isValid())
    {
        // previous code version
        queryString = "SELECT id_meteo FROM meteo_locations";
        query = dbMeteo.exec(queryString);
        query.first();
    }

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return false;
    }

    do
    {
        QString idMeteo;
        getValue(query.value("id_meteo"), &idMeteo);
        if (idMeteo != "")
        {
            idMeteoList.append(idMeteo);
        }
    }
    while(query.next());

    return true;
}


bool getLatLonFromIdMeteo(QSqlDatabase &dbMeteo, QString idMeteo, QString &lat, QString &lon, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM point_properties WHERE id_meteo='" + idMeteo +"'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        // previous code version
        queryString = "SELECT * FROM meteo_locations WHERE id_meteo='" + idMeteo +"'";
        query = dbMeteo.exec(queryString);
        query.last();
    }

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return false;
    }

    getValue(query.value("latitude"), &lat);
    getValue(query.value("longitude"), &lon);

    return true;
}


bool updateLatFromIdMeteo(QSqlDatabase& dbMeteo, QString idMeteo, QString lat, QString& error)
{
    error = "";

    QSqlQuery qry(dbMeteo);
    if (idMeteo.isEmpty())
    {
        error = "id_meteo missing";
        return false;
    }

    qry.prepare( "UPDATE point_properties SET "
                 "latitude = :latitude WHERE id_meteo = :id_meteo");

    qry.bindValue(":latitude", lat);

    qry.bindValue(":id_meteo", idMeteo);

    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return false;
    }

    return true;
}


QString getTableNameFromIdMeteo(QSqlDatabase &dbMeteo, QString idMeteo, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT table_name FROM point_properties WHERE id_meteo='" + idMeteo + "'";
    QSqlQuery query = dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        // previous code version
        queryString = "SELECT table_name FROM meteo_locations WHERE id_meteo='" + idMeteo + "'";
        query = dbMeteo.exec(queryString);
        query.last();
    }

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return "";
    }

    QString table_name;
    getValue(query.value(0), &table_name);

    return table_name;
}


bool getYearList(QSqlDatabase* dbMeteo, QString table, QList<QString>* yearList, QString *error)
{
    *error = "";
    QString queryString = "SELECT date, strftime('%Y',date) as Year FROM '" + table +"'";

    QSqlQuery query = dbMeteo->exec(queryString);

    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }

    QString year;
    do
    {
        getValue(query.value("Year"), &year);
        if (year != "" && !yearList->contains(year))
        {
            yearList->append(year);
        }
    }
    while(query.next());

    return true;
}


bool checkYear(QSqlDatabase* dbMeteo, QString table, QString year, QString *error)
{
    *error = "";

    QString TMIN_MIN = "-50.0";
    QString TMIN_MAX = "40.0";

    QString TMAX_MIN = "-40.0";
    QString TMAX_MAX = "50.0";

    QString PREC_MIN = "0.0";

    // count valid temp and prec
    QString queryString = "SELECT COUNT(date) FROM '" + table +"'" + " WHERE strftime('%Y',date) = '" + year+"'";
    queryString = queryString + " AND tmin NOT LIKE '' AND tmax NOT LIKE '' AND prec NOT LIKE ''";
    queryString = queryString + " AND CAST(tmin AS float) >=" + TMIN_MIN + " AND CAST(tmin AS float) <=" + TMIN_MAX;
    queryString = queryString + " AND CAST(tmax AS float) >= " + TMAX_MIN + " AND CAST(tmax AS float) <= " + TMAX_MAX + " AND CAST(prec AS float) >= " + PREC_MIN;

    QSqlQuery query = dbMeteo->exec(queryString);
    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }
    int count;


    getValue(query.value(0), &count);
    QDate temp(year.toInt(), 1, 1);
    int daysInYear = temp.daysInYear();

    if (count < (daysInYear-MAX_MISSING_TOT_DAYS))
    {
        *error = "incomplete year, valid data missing more than MAX_MISSING_DAYS";
        return false;
    }

    // check consecutive missing days (1 missing day allowed for temperature)

    queryString = "SELECT * FROM '" + table +"'" + "WHERE strftime('%Y',date) = '" + year +"'";
    query = dbMeteo->exec(queryString);

    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }

    QDate date;
    QDate previousDate(year.toInt()-1, 12, 31);
    QDate lastDate(year.toInt(), 12, 31);
    float tmin = NODATA;
    float tmax = NODATA;
    float prec = NODATA;
    float tmin_min = TMIN_MIN.toFloat();
    float tmin_max = TMIN_MAX.toFloat();

    float tmax_min = TMAX_MIN.toFloat();
    float tmax_max = TMAX_MAX.toFloat();
    float prec_min = PREC_MIN.toFloat();

    int invalidTemp = 0;
    int invalidPrec = 0;

    do
    {
        getValue(query.value("date"), &date);
        getValue(query.value("tmin"), &tmin);
        getValue(query.value("tmax"), &tmax);
        getValue(query.value("prec"), &prec);
        // 2 days missing
        if (previousDate.daysTo(date) > (MAX_MISSING_CONSECUTIVE_DAYS_T+1))
        {
            *error = "incomplete year, missing more than 1 consecutive days";
            return false;
        }
        // 1 day missing, the next one invalid temp
        if ( (previousDate.daysTo(date) == (MAX_MISSING_CONSECUTIVE_DAYS_T+1)) && (tmin < tmin_min || tmin > tmin_max || tmax < tmax_min || tmax > tmax_max ) )
        {
            *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
            return false;
        }
        // no day missing, check valid temp
        if (tmin < tmin_min || tmin > tmin_max || tmax < tmax_min || tmax > tmax_max )
        {
            invalidTemp = invalidTemp + 1;
            if (invalidTemp > 1)
            {
                *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                return false;
            }
        }
        else
        {
            invalidTemp = 0;
        }

        // check valid prec
        if (prec < prec_min )
        {
            invalidPrec = invalidPrec + int(previousDate.daysTo(date));
            // 7 day missing, the next one invalid
            if ( invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC )
            {
                 *error = "incomplete year, missing valid data (prec) more than 7 consecutive days";
                 return false;
            }
        }
        else
        {
            invalidPrec = 0;
        }
        previousDate = date;

    }
    while(query.next());

    // check last day (temp)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_T || (date.daysTo(lastDate) == MAX_MISSING_CONSECUTIVE_DAYS_T && invalidTemp > 0) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (temp)";
        return false;
    }

    // check last day (prec)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_PREC || (date.daysTo(lastDate) + invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC ) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (prec)";
        return false;
    }

    return true;
}


bool getLastDate(QSqlDatabase* dbMeteo, QString table, QString year, QDate* date, QString *error)
{
    *error = "";

    QString queryString = "SELECT * FROM '" + table +"'" + "WHERE strftime('%Y',date) = '" + year +"' ORDER BY date DESC";
    QSqlQuery query = dbMeteo->exec(queryString);

    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }
    else
    {
        getValue(query.value("date"), date);
        return true;
    }
}


bool checkYearMeteoGridFixedFields(QSqlDatabase dbMeteo, QString tableD, QString fieldTime, QString fieldTmin, QString fieldTmax, QString fieldPrec, QString year, QString *error)
{

    QSqlQuery qry(dbMeteo);

    *error = "";

    QString TMIN_MIN = "-50.0";
    QString TMIN_MAX = "40.0";

    QString TMAX_MIN = "-40.0";
    QString TMAX_MAX = "50.0";

    QString PREC_MIN = "0.0";

    // count valid temp and prec
    QString statement = QString("SELECT COUNT(`%1`) FROM `%2` "
                                "WHERE DATE_FORMAT(`%1`,'%Y') = '%3' AND `%4` NOT LIKE '' "
                                "AND `%5` NOT LIKE '' AND `%6` NOT LIKE ''").arg(fieldTime, tableD, year, fieldTmin, fieldTmax, fieldPrec);
    statement = statement + QString(" AND `%1` >= '%2' AND `%1` <= '%3'").arg(fieldTmin, TMIN_MIN, TMIN_MAX);
    statement = statement + QString(" AND `%1` >= '%2' AND `%1` <= '%3' AND `%4` >= '%5'").arg(fieldTmax, TMAX_MIN, TMAX_MAX, fieldPrec, PREC_MIN);

    if( !qry.exec(statement) )
    {
        *error = qry.lastError().text();
        return false;
    }
    qry.first();
    if (! qry.isValid())
    {
        *error = qry.lastError().text();
        return false;
    }
    int count;


    getValue(qry.value(0), &count);
    QDate temp(year.toInt(), 1, 1);
    int daysInYear = temp.daysInYear();

    if (count < (daysInYear-MAX_MISSING_TOT_DAYS))
    {
        *error = "incomplete year, valid data missing more than MAX_MISSING_DAYS";
        return false;
    }

    // check consecutive missing days (1 missing day allowed for temperature)
    statement = QString("SELECT * FROM `%1` WHERE DATE_FORMAT(`%2`,'%Y') = '%3' ORDER BY `%2`").arg(tableD, fieldTime, year);
    if( !qry.exec(statement) )
    {
        *error = qry.lastError().text();
        return false;
    }

    qry.first();
    if (! qry.isValid())
    {
        *error = qry.lastError().text();
        return false;
    }

    QDate date;
    QDate previousDate(year.toInt()-1, 12, 31);
    QDate lastDate(year.toInt(), 12, 31);
    float tmin = NODATA;
    float tmax = NODATA;
    float prec = NODATA;
    float tmin_min = TMIN_MIN.toFloat();
    float tmin_max = TMIN_MAX.toFloat();

    float tmax_min = TMAX_MIN.toFloat();
    float tmax_max = TMAX_MAX.toFloat();
    float prec_min = PREC_MIN.toFloat();

    int invalidTemp = 0;
    int invalidPrec = 0;

    do
    {
        getValue(qry.value(fieldTime), &date);
        getValue(qry.value(fieldTmin), &tmin);
        getValue(qry.value(fieldTmax), &tmax);
        getValue(qry.value(fieldPrec), &prec);
        // 2 days missing
        if (previousDate.daysTo(date) > (MAX_MISSING_CONSECUTIVE_DAYS_T+1))
        {
            *error = "incomplete year, missing more than 1 consecutive days";
            return false;
        }
        // 1 day missing, the next one invalid temp
        if ( (previousDate.daysTo(date) == (MAX_MISSING_CONSECUTIVE_DAYS_T+1)) && (tmin < tmin_min || tmin > tmin_max || tmax < tmax_min || tmax > tmax_max ) )
        {
            *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
            return false;
        }
        // no day missing, check valid temp
        if (tmin < tmin_min || tmin > tmin_max || tmax < tmax_min || tmax > tmax_max )
        {
            invalidTemp = invalidTemp + 1;
            if (invalidTemp > 1)
            {
                *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                return false;
            }
        }
        else
        {
            invalidTemp = 0;
        }

        // check valid prec
        if (prec < prec_min )
        {
            invalidPrec = invalidPrec + int(previousDate.daysTo(date));
            // 7 day missing, the next one invalid temp
            if ( invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC )
            {
                 *error = "incomplete year, missing valid data (prec) more than 7 consecutive days";
                 return false;
            }
        }
        else
        {
            invalidPrec = 0;
        }
        previousDate = date;

    }
    while(qry.next());

    // check last day (temp)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_T || (date.daysTo(lastDate) == MAX_MISSING_CONSECUTIVE_DAYS_T && invalidTemp > 0) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (temp)";
        return false;
    }

    // check last day (prec)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_PREC || (date.daysTo(lastDate) + invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC ) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (prec)";
        return false;
    }

    return true;
}

bool getLastDateGrid(QSqlDatabase dbMeteo, QString table, QString fieldTime, QString year, QDate* date, QString *error)
{
    *error = "";
    QSqlQuery qry(dbMeteo);

    QString statement = QString("SELECT * FROM `%1` WHERE DATE_FORMAT(`%2`,'%Y') = '%3' ORDER BY `%2` DESC").arg(table, fieldTime, year);
    if( !qry.exec(statement) )
    {
        *error = qry.lastError().text();
        return false;
    }
    else
    {
        qry.first();
        getValue(qry.value(fieldTime), date);
        return true;
    }
}


bool checkYearMeteoGrid(QSqlDatabase dbMeteo, QString tableD, QString fieldTime, int varCodeTmin, int varCodeTmax, int varCodePrec, QString year, QString *error)
{
    QSqlQuery qry(dbMeteo);

    *error = "";

    QString TMIN_MIN = "-50.0";
    QString TMIN_MAX = "40.0";

    QString TMAX_MIN = "-40.0";
    QString TMAX_MAX = "50.0";

    QString PREC_MIN = "0.0";

    // count valid temp and prec
    QString statement = QString("SELECT COUNT(`%1`) FROM `%2` WHERE DATE_FORMAT(`%1`,'%Y') = '%3'").arg(fieldTime, tableD, year);
    statement = statement + QString(" AND ( (VariableCode = '%1' AND Value >= '%2' AND Value <= '%3')").arg(varCodeTmin).arg(TMIN_MIN, TMIN_MAX);
    statement = statement + QString(" OR ( VariableCode = '%1' AND Value >= '%2' AND Value <= '%3') ").arg(varCodeTmax).arg(TMAX_MIN, TMAX_MAX);
    statement = statement + QString(" OR ( VariableCode = '%1' AND Value >= '%2') )").arg(varCodePrec).arg(PREC_MIN);

    if( !qry.exec(statement) )
    {
        *error = qry.lastError().text();
        return false;
    }
    qry.first();
    if (! qry.isValid())
    {
        *error = qry.lastError().text();
        return false;
    }
    int count;

    getValue(qry.value(0), &count);
    QDate temp(year.toInt(), 1, 1);
    int daysInYear = temp.daysInYear();

    // 3 variables
    if (count/3 < (daysInYear-MAX_MISSING_TOT_DAYS))
    {
        *error = "incomplete year, valid data missing more than MAX_MISSING_DAYS";
        return false;
    }

    // check consecutive missing days (1 missing day allowed for temperature)
    statement = QString("SELECT * FROM `%1` WHERE DATE_FORMAT(`%2`,'%Y') = '%3' AND "
                        "( VariableCode = '%4' OR VariableCode = '%5' OR VariableCode = '%6') "
                        "ORDER BY `%2`").arg(tableD, fieldTime, year).arg(varCodeTmin, varCodeTmax, varCodePrec);
    if( !qry.exec(statement) )
    {
        *error = qry.lastError().text();
        return false;
    }

    qry.first();
    if (! qry.isValid())
    {
        *error = qry.lastError().text();
        return false;
    }

    QDate date;
    QDate previousDateTmin(year.toInt()-1, 12, 31);
    QDate previousDateTmax(year.toInt()-1, 12, 31);
    QDate previousDatePrec(year.toInt()-1, 12, 31);
    QDate lastDate(year.toInt(), 12, 31);
    float tmin = NODATA;
    float tmax = NODATA;
    float prec = NODATA;
    int variableCode = NODATA;
    float tmin_min = TMIN_MIN.toFloat();
    float tmin_max = TMIN_MAX.toFloat();

    float tmax_min = TMAX_MIN.toFloat();
    float tmax_max = TMAX_MAX.toFloat();
    float prec_min = PREC_MIN.toFloat();

    int invalidTempMin = 0;
    int invalidTempMax = 0;
    int invalidPrec = 0;

    do
    {
        getValue(qry.value(fieldTime), &date);
        getValue(qry.value("VariableCode"), &variableCode);

        if (variableCode == varCodeTmin)
        {
            getValue(qry.value("Value"), &tmin);
            // 2 days missing
            if (previousDateTmin.daysTo(date) > (MAX_MISSING_CONSECUTIVE_DAYS_T+1))
            {
                *error = "incomplete year, missing more than 1 consecutive days";
                return false;
            }
            // 1 day missing, the next one invalid temp
            if ( (previousDateTmin.daysTo(date) == (MAX_MISSING_CONSECUTIVE_DAYS_T+1)) && (tmin < tmin_min || tmin > tmin_max) )
            {
                *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                return false;
            }
            // no day missing, check valid temp
            if (tmin < tmin_min || tmin > tmin_max)
            {
                invalidTempMin = invalidTempMin + 1;
                if (invalidTempMin > 1)
                {
                    *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                    return false;
                }
            }
            else
            {
                invalidTempMin = 0;
            }
            previousDateTmin = date;
        }
        else if (variableCode == varCodeTmax)
        {
            getValue(qry.value("Value"), &tmax);
            // 2 days missing
            if (previousDateTmax.daysTo(date) > (MAX_MISSING_CONSECUTIVE_DAYS_T+1))
            {
                *error = "incomplete year, missing more than 1 consecutive days";
                return false;
            }
            // 1 day missing, the next one invalid temp
            if ( (previousDateTmax.daysTo(date) == (MAX_MISSING_CONSECUTIVE_DAYS_T+1)) && (tmax < tmax_min || tmax > tmax_max) )
            {
                *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                return false;
            }
            // no day missing, check valid temp
            if (tmax < tmax_min || tmax > tmax_max)
            {
                invalidTempMax = invalidTempMax + 1;
                if (invalidTempMax > 1)
                {
                    *error = "incomplete year, missing valid data (temp) more than 1 consecutive days";
                    return false;
                }
            }
            else
            {
                invalidTempMax = 0;
            }
            previousDateTmax = date;
        }
        else if (variableCode == varCodePrec)
        {
            getValue(qry.value("Value"), &prec);
            // check valid prec
            if (prec < prec_min )
            {
                invalidPrec = invalidPrec + int(previousDatePrec.daysTo(date));
                // 7 day missing, the next one invalid temp
                if ( invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC )
                {
                     *error = "incomplete year, missing valid data (prec) more than 7 consecutive days";
                     return false;
                }
            }
            else
            {
                invalidPrec = 0;
            }
            previousDatePrec = date;
        }

    }
    while(qry.next());

    // check last day (tempMin)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_T || (date.daysTo(lastDate) == MAX_MISSING_CONSECUTIVE_DAYS_T && invalidTempMin > 0) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (tempMin)";
        return false;
    }
    // check last day (tempMax)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_T || (date.daysTo(lastDate) == MAX_MISSING_CONSECUTIVE_DAYS_T && invalidTempMax > 0) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (tempMax)";
        return false;
    }

    // check last day (prec)
    if (date.daysTo(lastDate) > MAX_MISSING_CONSECUTIVE_DAYS_PREC || (date.daysTo(lastDate) + invalidPrec > MAX_MISSING_CONSECUTIVE_DAYS_PREC ) )
    {
        *error = "incomplete year, missing more than 1 consecutive days (prec)";
        return false;
    }

    return true;
}


bool fillDailyTempPrecCriteria1D(QSqlDatabase* dbMeteo, QString table, Crit3DMeteoPoint *meteoPoint, int validYear, QString *error)
{
    *error = "";

    QString queryString = "SELECT * FROM '" + table +"'" + " WHERE strftime('%Y',date) = '" + QString::number(validYear) +"'";
    QSqlQuery query = dbMeteo->exec(queryString);

    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }

    QDate date;
    QDate previousDate(validYear-1, 12, 31);
    QDate lastDate(validYear, 12, 31);
    float tmin = NODATA;
    float tmax = NODATA;
    float tavg = NODATA;
    float prec = NODATA;
    float waterTable = NODATA;

    const float tmin_min = -50;
    const float tmin_max = 40;

    const float tmax_min = -40;
    const float tmax_max = 60;

    do
    {
        getValue(query.value("date"), &date);
        getValue(query.value("tmin"), &tmin);
        getValue(query.value("tmax"), &tmax);
        getValue(query.value("prec"), &prec);

        // Watertable depth [m]
        getValue(query.value("watertable"), &waterTable);
        if (waterTable < 0.f) waterTable = NODATA;

        if (tmin < tmin_min || tmin > tmin_max)
        {
            tmin = NODATA;
        }
        if (tmax < tmax_min || tmax > tmax_max)
        {
            tmax = NODATA;
        }
        if (isEqual(tmin, NODATA) || isEqual(tmax, NODATA))
        {
            tavg = NODATA;
        }
        else
        {
            tavg = (tmin + tmax) * 0.5f;
        }
        if (prec < 0)
        {
            prec = NODATA;
        }

        Crit3DDate myDate = getCrit3DDate(date);
        meteoPoint->setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);
        meteoPoint->setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);
        meteoPoint->setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);
        meteoPoint->setMeteoPointValueD(myDate, dailyPrecipitation, prec);
        meteoPoint->setMeteoPointValueD(myDate, dailyWaterTableDepth, waterTable);
    }
    while(query.next());

    QDate firstDate(validYear, 1, 1);
    int daysInYear = firstDate.daysInYear();
    float prevTmin = NODATA;
    float prevTmax = NODATA;
    float nextTmin = NODATA;
    float nextTmax = NODATA;

    // fill NODATA values with average values of day before and next.
    for (int i = 0; i < daysInYear; i++)
    {
        date = firstDate.addDays(i);
        tmin = meteoPoint->getMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureMax);
        tavg = meteoPoint->getMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureAvg);
        if (isEqual(tmin, NODATA))
        {
            if (i!=0 && i!=(daysInYear-1))
            {
                prevTmin = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(-1)), dailyAirTemperatureMin);
                nextTmin = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(1)), dailyAirTemperatureMin);
                tmin = (prevTmin + nextTmin) * 0.5f;
            }
            else if (i==0)
            {
                nextTmin = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(1)), dailyAirTemperatureMin);
                tmin = nextTmin;
            }
            else if (i==(daysInYear-1))
            {
                prevTmin = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(-1)), dailyAirTemperatureMin);
                tmin = prevTmin;
            }

            meteoPoint->setMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureMin, tmin);

        }
        if (isEqual(tmax, NODATA))
        {
            if (i!=0 && i!=(daysInYear-1))
            {
                prevTmax = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(-1)), dailyAirTemperatureMax);
                nextTmax = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(1)), dailyAirTemperatureMax);
                tmax = (prevTmax + nextTmax) * 0.5f;
            }
            else if (i==0)
            {
                nextTmax = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(1)), dailyAirTemperatureMax);
                tmax = nextTmax;
            }
            else if (i==(daysInYear-1))
            {
                prevTmax = meteoPoint->getMeteoPointValueD(getCrit3DDate(date.addDays(-1)), dailyAirTemperatureMax);
                tmax = prevTmax;
            }

            meteoPoint->setMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureMax, tmax);
        }
        if (isEqual(tavg, NODATA))
        {
            tavg = (tmin + tmax) * 0.5f;
            meteoPoint->setMeteoPointValueD(getCrit3DDate(date), dailyAirTemperatureAvg, tavg);
        }
        if (isEqual(prec, NODATA))
        {
            prec = 0;
        }
    }
    return true;
}


/*!
 * \brief read daily meteo data from a table in the criteria-1D format
 * \brief (`date`,`tmin`,`tmax`,`tavg`,`prec`,`et0`,`watertable`)
 * \details mandatory: date, tmin, tmax, prec
 * \details not mandatory: tavg, et0, watertable
 * \details date format: "yyyy-mm-dd"
 * \return true if data are correctly loaded
 * \note meteoPoint have to be initialized BEFORE function
 * \note maxNrDays = NODATA: read all data
 */
bool readDailyDataCriteria1D(QSqlQuery &query, Crit3DMeteoPoint &meteoPoint, int maxNrDays, QString &error)
{
    const int MAX_MISSING_DAYS = 3;
    QDate myDate, expectedDate, previousDate;
    Crit3DDate date;
    QString meteoID = QString::fromStdString(meteoPoint.id);

    float tmin = NODATA;
    float tmax = NODATA;
    float tmed = NODATA;
    float prec = NODATA;            // [mm]
    float et0 = NODATA;             // [mm]
    float waterTable = NODATA;      // [m]
    float previousTmin = NODATA;
    float previousTmax = NODATA;
    float previousWaterTable = NODATA;
    int nrMissingData = 0;
    std::vector<QString> wrongDate;

    // first date
    query.first();
    myDate = query.value("date").toDate();
    expectedDate = myDate;
    previousDate = myDate.addDays(-1);

    QList<QString> fieldList = getFieldsUpperCase(query);
    bool existEt0 = fieldList.contains("ET0");
    bool existWatertable = fieldList.contains("WATERTABLE");
    bool existTavg = fieldList.contains("TAVG");

    int currentIndex = 0;
    do
    {
        myDate = query.value("date").toDate();

        if (! myDate.isValid())
        {
            if (maxNrDays != NODATA && currentIndex < (maxNrDays-1))
            {
                error = meteoID + "Wrong date format: " + query.value("date").toString();
                return false;
            }
            else
            {
                // last value is wrong - skip
                continue;
            }
        }

        if (myDate != previousDate)
        {
            if (myDate != expectedDate)
            {
                if (expectedDate.daysTo(myDate) > MAX_MISSING_DAYS)
                {
                    error = meteoID + " wrong METEO: too many missing data." + expectedDate.toString();
                    return false;
                }
                else
                {
                    // fill missing data
                    while (myDate != expectedDate)
                    {
                        tmin = previousTmin;
                        tmax = previousTmax;
                        tmed = (tmin + tmax) * 0.5f;
                        prec = 0;
                        et0 = NODATA;
                        waterTable = previousWaterTable;

                        date = getCrit3DDate(expectedDate);
                        meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureMin, tmin);
                        meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureMax, tmax);
                        meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureAvg, tmed);
                        meteoPoint.setMeteoPointValueD(date, dailyPrecipitation, prec);
                        meteoPoint.setMeteoPointValueD(date, dailyReferenceEvapotranspirationHS, et0);
                        meteoPoint.setMeteoPointValueD(date, dailyWaterTableDepth, waterTable);

                        expectedDate = expectedDate.addDays(1);
                    }
                }
            }

            previousTmax = tmax;
            previousTmin = tmin;
            previousWaterTable = waterTable;

            // mandatory variables
            getValue(query.value("tmin"), &tmin);
            getValue(query.value("tmax"), &tmax);
            getValue(query.value("prec"), &prec);

            // check
            if (prec < 0.f) prec = NODATA;
            if ((tmin < -60.f) || (tmin > 50.f)) tmin = NODATA;
            if ((tmax < -50.f) || (tmax > 60.f)) tmax = NODATA;

            if (isEqual(tmin, NODATA) || isEqual(tmax, NODATA) || isEqual(prec, NODATA))
            {
                wrongDate.push_back(myDate.toString("yyyy-MM-dd"));
                if (nrMissingData < MAX_MISSING_DAYS)
                {
                    if (isEqual(tmin, NODATA)) tmin = previousTmin;
                    if (isEqual(tmax, NODATA)) tmax = previousTmax;
                    if (isEqual(prec, NODATA)) prec = 0;
                    nrMissingData++;
                }
                else
                {
                    error = meteoID + " wrong METEO: too many missing data " + myDate.toString();
                    return false;
                }
            }
            else nrMissingData = 0;

            // NOT mandatory variables

            // TAVG [°C]
            if (existTavg)
            {
                getValue(query.value("tavg"), &tmed);
                if (isEqual(tmed, NODATA) || (tmed < -40.f) || (tmed > 50.f))
                     tmed = (tmin + tmax) * 0.5f;
            }
            else tmed = (tmin + tmax) * 0.5f;

            // ET0 [mm]
            if (existEt0)
            {
                getValue(query.value("et0"), &et0);
                if ((et0 < 0.f) || (et0 > 10.f))
                    et0 = NODATA;
            }
            else et0 = NODATA;

            // Watertable depth [m]
            if (existWatertable)
            {
                getValue(query.value("watertable"), &waterTable);
                if (waterTable < 0.f)
                    waterTable = NODATA;
            }
            else waterTable = NODATA;

            date = getCrit3DDate(myDate);
            if (meteoPoint.obsDataD[0].date.daysTo(date) < meteoPoint.nrObsDataDaysD)
            {
                meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureMin, float(tmin));
                meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureMax, float(tmax));
                meteoPoint.setMeteoPointValueD(date, dailyAirTemperatureAvg, float(tmed));
                meteoPoint.setMeteoPointValueD(date, dailyPrecipitation, float(prec));
                meteoPoint.setMeteoPointValueD(date, dailyReferenceEvapotranspirationHS, float(et0));
                meteoPoint.setMeteoPointValueD(date, dailyWaterTableDepth, waterTable);
            }
            else
            {
                error = meteoID + " wrong METEO: index out of range.";
                return false;
            }

            previousDate = myDate;
            expectedDate = myDate.addDays(1);
            currentIndex++;
        }

    } while(query.next() && (currentIndex < maxNrDays || maxNrDays == NODATA));

    if (wrongDate.size() > 0)
    {
        error = "WARNING on meteo table: " + meteoID + "\n";
        error += "Wrong or missing data: ";
        for (unsigned int i = 0; i < wrongDate.size(); i++)
        {
            error += wrongDate[i];
            if (i < (wrongDate.size() - 1))
                error += ", ";
        }
    }

    return true;
}

