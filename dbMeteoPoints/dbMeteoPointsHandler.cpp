#include "dbMeteoPointsHandler.h"
#include "commonConstants.h"
#include "meteo.h"
#include "utilities.h"
#include "basicMath.h"

#include <QtSql>


Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler(QString provider_, QString host_, QString dbname_, int port_,
                                                       QString user_, QString pass_)
{
    error = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase(provider_, QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);

    if (provider_ != "QSQLITE")
    {
        _db.setHostName(host_);
        _db.setPort(port_);
        _db.setUserName(user_);
        _db.setPassword(pass_);
    }

    if (!_db.open())
       error = _db.lastError().text();    

}

Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler(QString dbname_)
{
    error = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);

    if (!_db.open())
       error = _db.lastError().text();

}

Crit3DMeteoPointsDbHandler::~Crit3DMeteoPointsDbHandler()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        QString connection = _db.connectionName();
        _db.close();
        _db = QSqlDatabase();
        QSqlDatabase::removeDatabase(connection);
    }
}


QString Crit3DMeteoPointsDbHandler::getDbName()
{
    return _db.databaseName();
}

QString Crit3DMeteoPointsDbHandler::getDatasetURL(QString dataset)
{
    QSqlQuery qry(_db);
    QString url = nullptr;

    qry.prepare( "SELECT URL FROM datasets WHERE dataset = :dataset");
    qry.bindValue(":dataset", dataset);

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        if (qry.next())
            url = qry.value(0).toString();

        else
            qDebug( "Error: dataset not found" );
    }

    return url;
}


QStringList Crit3DMeteoPointsDbHandler::getDatasetsActive()
{
    QStringList activeList;
    QSqlQuery qry(_db);

    qry.prepare( "SELECT dataset FROM datasets WHERE active = 1" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        while (qry.next())
        {
            QString active = qry.value(0).toString();
            activeList << active;

        }

    }

    return activeList;

}

void Crit3DMeteoPointsDbHandler::setDatasetsActive(QString active)
{
    QString statement = QString("UPDATE datasets SET active = 0");
    _db.exec(statement);

    statement = QString("UPDATE datasets SET active = 1 WHERE dataset IN ( %1 )").arg(active);
    _db.exec(statement);
}


QStringList Crit3DMeteoPointsDbHandler::getDatasetsList()
{
    QStringList datasetList;
    QSqlQuery qry(_db);

    qry.prepare( "SELECT * FROM datasets" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        while (qry.next())
        {
            QString dataset = qry.value(0).toString();
            datasetList << dataset;
        }
    }

    return datasetList;
}


QDateTime Crit3DMeteoPointsDbHandler::getFirstDate(frequencyType frequency)
{
    QSqlQuery qry(_db);
    QStringList tables;
    QDateTime firstDate;

    QString dayHour;
    if (frequency == daily)
        dayHour = "D";
    else if (frequency == hourly)
        dayHour = "H";

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table' AND name like :dayHour ESCAPE '^'");
    qry.bindValue(":dayHour",  "%^" + dayHour  + "%");

    if( !qry.exec() )
    {
        error = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            QString table = qry.value(0).toString();
            tables << table;
        }
    }

    QDateTime date;
    QString dateStr, statement;
    foreach (QString table, tables)
    {
        statement = QString( "SELECT MIN(date_time) FROM `%1` AS dateTime").arg(table);
        if(qry.exec(statement) )
        {
            if (qry.next())
            {
                dateStr = qry.value(0).toString();
                if (!dateStr.isEmpty())
                {
                    if (frequency == daily)
                    {
                        date = QDateTime::fromString(dateStr,"yyyy-MM-dd");
                    }
                    else if (frequency == hourly)
                    {
                        date = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");
                    }

                    if (firstDate.isNull() || date < firstDate)
                    {
                        firstDate = date;
                    }
                }
            }
        }
    }

    return firstDate;
}


QDateTime Crit3DMeteoPointsDbHandler::getLastDate(frequencyType frequency)
{
    QSqlQuery qry(_db);
    QStringList tables;
    QDateTime lastDate;

    QString dayHour;
    if (frequency == daily)
        dayHour = "D";
    else if (frequency == hourly)
        dayHour = "H";

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table' AND name like :dayHour ESCAPE '^'");
    qry.bindValue(":dayHour",  "%^_" + dayHour  + "%");

    if( !qry.exec() )
    {
        error = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            QString table = qry.value(0).toString();
            tables << table;
        }
    }

    QDateTime date;
    QString dateStr, statement;
    foreach (QString table, tables)
    {
        statement = QString( "SELECT MAX(date_time) FROM `%1` AS dateTime").arg(table);
        if(qry.exec(statement))
        {
            if (qry.next())
            {
                dateStr = qry.value(0).toString();
                if (!dateStr.isEmpty())
                {
                    if (frequency == daily)
                    {
                        date = QDateTime::fromString(dateStr,"yyyy-MM-dd");
                    }
                    else if (frequency == hourly)
                    {
                        date = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");
                    }

                    if (lastDate.isNull() || date > lastDate)
                    {
                        lastDate = date;
                    }
                }
            }
        }
    }

    return lastDate;
}


// return a null datetime if the table doesn't exist or table is void
QDateTime Crit3DMeteoPointsDbHandler::getFirstDate(frequencyType frequency, std::string idMeteoPoint)
{
    QDateTime firstDate;

    QString tableName;
    if (frequency == daily)
        tableName = QString::fromStdString(idMeteoPoint + "_D");
    else if (frequency == hourly)
        tableName = QString::fromStdString(idMeteoPoint + "_H");

    QSqlQuery qry(_db);
    QString statement = QString( "SELECT MIN(date_time) FROM `%1` AS dateTime").arg(tableName);

    if(qry.exec(statement))
    {
        if (qry.next())
        {
            QString dateStr = qry.value(0).toString();
            if (!dateStr.isEmpty())
            {
                if (frequency == daily)
                {
                    firstDate = QDateTime::fromString(dateStr,"yyyy-MM-dd");
                }
                else if (frequency == hourly)
                {
                    firstDate = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");
                }
            }
        }
    }

    return firstDate;
}


// return a null datetime if the table doesn't exist or table is void
QDateTime Crit3DMeteoPointsDbHandler::getLastDate(frequencyType frequency, std::string idMeteoPoint)
{
    QDateTime lastDate;

    QString tableName;
    if (frequency == daily)
        tableName = QString::fromStdString(idMeteoPoint + "_D");
    else if (frequency == hourly)
        tableName = QString::fromStdString(idMeteoPoint + "_H");

    QSqlQuery qry(_db);
    QString statement = QString( "SELECT MAX(date_time) FROM `%1` AS dateTime").arg(tableName);

    if(qry.exec(statement))
    {
        if (qry.next())
        {
            QString dateStr = qry.value(0).toString();
            if (!dateStr.isEmpty())
            {
                if (frequency == daily)
                {
                    lastDate = QDateTime::fromString(dateStr,"yyyy-MM-dd");
                }
                else if (frequency == hourly)
                {
                    lastDate = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");
                }
            }
        }
    }

    return lastDate;
}


bool Crit3DMeteoPointsDbHandler::existData(Crit3DMeteoPoint *meteoPoint, frequencyType myFreq)
{
    QSqlQuery myQuery(_db);
    QString tableName = QString::fromStdString(meteoPoint->id) + ((myFreq == daily) ?  "_D" : "_H");
    QString statement = QString( "SELECT 1 FROM `%1`").arg(tableName);

    if (myQuery.exec(statement))
        if (myQuery.next())
            return true;

    return false;
}

bool Crit3DMeteoPointsDbHandler::loadDailyData(Crit3DDate dateStart, Crit3DDate dateEnd, Crit3DMeteoPoint *meteoPoint)
{
    QString dateStr;
    meteoVariable variable;
    QDate d;
    int idVar;
    float value;

    int numberOfDays = difference(dateStart, dateEnd) +1;
    QString startDate = QString::fromStdString(dateStart.toStdString());
    QString endDate = QString::fromStdString(dateEnd.toStdString());

    QSqlQuery myQuery(_db);

    meteoPoint->initializeObsDataD(numberOfDays, dateStart);

    QString tableName = QString::fromStdString(meteoPoint->id) + "_D";

    QString statement = QString( "SELECT * FROM `%1` WHERE date_time >= DATE('%2') AND date_time < DATE('%3', '+1 day')")
                                .arg(tableName).arg(startDate).arg(endDate);

    if( !myQuery.exec(statement) )
    {
        return false;
    }
    else
    {
        while (myQuery.next())
        {
            dateStr = myQuery.value(0).toString();
            d = QDate::fromString(dateStr, "yyyy-MM-dd");

            idVar = myQuery.value(1).toInt();
            variable = _mapIdMeteoVar.at(idVar);

            value = myQuery.value(2).toFloat();

            meteoPoint->setMeteoPointValueD(Crit3DDate(d.day(), d.month(), d.year()), variable, value);
        }
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::loadHourlyData(Crit3DDate dateStart, Crit3DDate dateEnd, Crit3DMeteoPoint *meteoPoint)
{
    QString dateStr;
    meteoVariable variable;
    QDateTime d;
    int idVar;
    float value;

    int numberOfDays = difference(dateStart, dateEnd)+1;
    int myHourlyFraction = 1;
    QString startDate = QString::fromStdString(dateStart.toStdString());
    QString endDate = QString::fromStdString(dateEnd.toStdString());

    QSqlQuery qry(_db);

    meteoPoint->initializeObsDataH(myHourlyFraction, numberOfDays, dateStart);

    QString tableName = QString::fromStdString(meteoPoint->id) + "_H";

    QString statement = QString( "SELECT * FROM `%1` WHERE date_time >= DATETIME('%2 01:00:00') AND date_time <= DATETIME('%3 00:00:00', '+1 day')")
                                 .arg(tableName).arg(startDate).arg(endDate);
    if( !qry.exec(statement) )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
    {
        while (qry.next())
        {
            dateStr = qry.value(0).toString();
            d = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");

            idVar = qry.value(1).toInt();
            try {
                variable = _mapIdMeteoVar.at(idVar);
            }
            catch (const std::out_of_range& ) {
                variable = noMeteoVar;
            }

            if (variable != noMeteoVar)
            {
                value = qry.value(2).toFloat();
                meteoPoint->setMeteoPointValueH(Crit3DDate(d.date().day(), d.date().month(), d.date().year()),
                                                       d.time().hour(), d.time().minute(), variable, value);

                // copy scalar intensity to vector intensity (instantaneous values are equivalent, following WMO)
                // should be removed when when we hourly averages are available
                if (variable == windScalarIntensity)
                    meteoPoint->setMeteoPointValueH(Crit3DDate(d.date().day(), d.date().month(), d.date().year()),
                                                           d.time().hour(), d.time().minute(), windVectorIntensity, value);
            }
        }
    }
    return true;
}


std::vector<float> Crit3DMeteoPointsDbHandler::loadDailyVar(QString *myError, meteoVariable variable, Crit3DDate dateStart, Crit3DDate dateEnd, QDate* firstDateDB, Crit3DMeteoPoint *meteoPoint)
{
    QString dateStr;
    QDate d, previousDate;
    float value;
    std::vector<float> dailyVarList;
    bool firstRow = true;

    int idVar = getIdfromMeteoVar(variable);
    QString startDate = QString::fromStdString(dateStart.toStdString());
    QString endDate = QString::fromStdString(dateEnd.toStdString());

    QSqlQuery myQuery(_db);

    QString tableName = QString::fromStdString(meteoPoint->id) + "_D";

    QString statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATE('%4') AND date_time < DATE('%5', '+1 day')")
                                .arg(tableName).arg(FIELD_METEO_VARIABLE).arg(idVar).arg(startDate).arg(endDate);

    if( !myQuery.exec(statement) )
    {
        *myError = myQuery.lastError().text();
        return dailyVarList;
    }
    else
    {
        while (myQuery.next())
        {
            if (firstRow)
            {
                dateStr = myQuery.value(0).toString();
                *firstDateDB = QDate::fromString(dateStr, "yyyy-MM-dd");
                previousDate = *firstDateDB;

                value = myQuery.value(2).toFloat();

                dailyVarList.push_back(value);

                firstRow = false;
            }
            else
            {
                dateStr = myQuery.value(0).toString();
                d = QDate::fromString(dateStr, "yyyy-MM-dd");

                int missingDate = previousDate.daysTo(d);
                for (int i =1; i<missingDate; i++)
                {
                    dailyVarList.push_back(NODATA);
                }

                value = myQuery.value(2).toFloat();

                dailyVarList.push_back(value);
                previousDate = d;

            }

        }
    }

    return dailyVarList;
}

std::vector<float> Crit3DMeteoPointsDbHandler::loadHourlyVar(QString *myError, meteoVariable variable, Crit3DDate dateStart, Crit3DDate dateEnd, QDateTime* firstDateDB, Crit3DMeteoPoint *meteoPoint)
{
    QString dateStr;
    QDateTime d, previousDate;
    float value;
    std::vector<float> hourlyVarList;
    bool firstRow = true;

    int idVar = getIdfromMeteoVar(variable);
    QString startDate = QString::fromStdString(dateStart.toStdString());
    QString endDate = QString::fromStdString(dateEnd.toStdString());

    QSqlQuery qry(_db);

    QString tableName = QString::fromStdString(meteoPoint->id) + "_H";

    QString statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATETIME('%4 01:00:00') AND date_time <= DATETIME('%5 00:00:00', '+1 day')")
                                 .arg(tableName).arg(FIELD_METEO_VARIABLE).arg(idVar).arg(startDate).arg(endDate);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return hourlyVarList;
    }
    else
    {
        while (qry.next())
        {
            if (firstRow)
            {
                dateStr = qry.value(0).toString();
                *firstDateDB = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");
                previousDate = *firstDateDB;

                value = qry.value(2).toFloat();

                hourlyVarList.push_back(value);
                firstRow = false;
            }
            else
            {
                dateStr = qry.value(0).toString();
                d = QDateTime::fromString(dateStr,"yyyy-MM-dd HH:mm:ss");

                int missingDate = previousDate.daysTo(d);
                for (int i =1; i<missingDate; i++)
                {
                    hourlyVarList.push_back(NODATA);
                }
                value = qry.value(2).toFloat();

                hourlyVarList.push_back(value);
                previousDate = d;
            }
        }
    }
    return hourlyVarList;
}

QSqlDatabase Crit3DMeteoPointsDbHandler::getDb() const
{
    return _db;
}

void Crit3DMeteoPointsDbHandler::setDb(const QSqlDatabase &db)
{
    _db = db;
}

std::map<int, meteoVariable> Crit3DMeteoPointsDbHandler::getMapIdMeteoVar() const
{
    return _mapIdMeteoVar;
}

QList<Crit3DMeteoPoint> Crit3DMeteoPointsDbHandler::getPropertiesFromDb(const gis::Crit3DGisSettings& gisSettings, QString *errorString)
{
    QList<Crit3DMeteoPoint> meteoPointsList;
    Crit3DMeteoPoint meteoPoint;
    QSqlQuery qry(_db);
    bool isPositionOk;

    qry.prepare( "SELECT id_point, name, dataset, latitude, longitude, utm_x, utm_y, altitude, state, region, province, municipality, is_active, is_utc, orog_code from point_properties ORDER BY id_point" );

    if( !qry.exec() )
    {
        *errorString = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            //initialize
            meteoPoint = *(new Crit3DMeteoPoint());

            meteoPoint.id = qry.value("id_point").toString().toStdString();
            meteoPoint.name = qry.value("name").toString().toStdString();
            meteoPoint.dataset = qry.value("dataset").toString().toStdString();

            if (qry.value("latitude") != "")
                meteoPoint.latitude = qry.value("latitude").toDouble();
            if (qry.value("longitude") != "")
                meteoPoint.longitude = qry.value("longitude").toDouble();
            if (qry.value("utm_x") != "")
                meteoPoint.point.utm.x = qry.value("utm_x").toDouble();
            if (qry.value("utm_y") != "")
                meteoPoint.point.utm.y = qry.value("utm_y").toDouble();
            if (qry.value("altitude") != "")
                meteoPoint.point.z = qry.value("altitude").toDouble();

            // check position
            isPositionOk = false;
            if ((meteoPoint.latitude != NODATA || meteoPoint.longitude != NODATA)
                && (meteoPoint.point.utm.x != NODATA && meteoPoint.point.utm.y != NODATA))
            {
                isPositionOk = true;
            }
            else if ((meteoPoint.latitude == NODATA || meteoPoint.longitude == NODATA)
                && (meteoPoint.point.utm.x != NODATA && meteoPoint.point.utm.y != NODATA))
            {
                gis::getLatLonFromUtm(gisSettings, meteoPoint.point.utm.x, meteoPoint.point.utm.y,
                                        &(meteoPoint.latitude), &(meteoPoint.longitude));
                isPositionOk = true;
            }
            else if ((meteoPoint.latitude != NODATA || meteoPoint.longitude != NODATA)
                && (meteoPoint.point.utm.x == NODATA && meteoPoint.point.utm.y == NODATA))
            {
                gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint.latitude, meteoPoint.longitude,
                                          &(meteoPoint.point.utm.x), &(meteoPoint.point.utm.y));
                isPositionOk = true;
            }

            if (isPositionOk)
            {
                meteoPoint.state = qry.value("state").toString().toStdString();
                meteoPoint.region = qry.value("region").toString().toStdString();
                meteoPoint.province = qry.value("province").toString().toStdString();
                meteoPoint.municipality = qry.value("municipality").toString().toStdString();
                meteoPoint.active = qry.value("is_active").toBool();
                meteoPoint.isUTC = qry.value("is_utc").toBool();
                meteoPoint.lapseRateCode = lapseRateCodeType((qry.value("orog_code").toInt()));
                meteoPointsList << meteoPoint;
            }
        }
    }

    return meteoPointsList;
}


bool Crit3DMeteoPointsDbHandler::writePointProperties(Crit3DMeteoPoint *myPoint)
{

    QSqlQuery qry(_db);

    qry.prepare( "INSERT INTO point_properties (id_point, name, dataset, latitude, longitude, latInt, lonInt, utm_x, utm_y, altitude, state, region, province, municipality)"
                                      " VALUES (:id_point, :name, :dataset, :latitude, :longitude, :latInt, :lonInt, :utm_x, :utm_y, :altitude, :state, :region, :province, :municipality)" );

    qry.bindValue(":id_point", QString::fromStdString(myPoint->id));
    qry.bindValue(":name", QString::fromStdString(myPoint->name));
    qry.bindValue(":dataset", QString::fromStdString(myPoint->dataset));
    qry.bindValue(":latitude", myPoint->latitude);
    qry.bindValue(":longitude", myPoint->longitude);
    qry.bindValue(":latInt", myPoint->latInt);
    qry.bindValue(":lonInt", myPoint->lonInt);
    qry.bindValue(":utm_x", myPoint->point.utm.x);
    qry.bindValue(":utm_y", myPoint->point.utm.y);
    qry.bindValue(":altitude", myPoint->point.z);
    qry.bindValue(":state", QString::fromStdString(myPoint->state));
    qry.bindValue(":region", QString::fromStdString(myPoint->region));
    qry.bindValue(":province", QString::fromStdString(myPoint->province));
    qry.bindValue(":municipality", QString::fromStdString(myPoint->municipality));

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
        return true;

}


bool Crit3DMeteoPointsDbHandler::loadVariableProperties()
{
    QSqlQuery qry(_db);

    QString tableName = "variable_properties";
    int id_variable;
    QString variable;
    std::string varStdString;
    meteoVariable meteoVar;
    std::pair<std::map<int, meteoVariable>::iterator,bool> ret;

    QString statement = QString( "SELECT * FROM `%1` ").arg(tableName);
    if( !qry.exec(statement) )
    {
        error = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_variable"), &id_variable);
            getValue(qry.value("variable"), &variable);
            varStdString = variable.toStdString();
            try {
              meteoVar = MapDailyMeteoVar.at(varStdString);
            }
            catch (const std::out_of_range& ) {
                try {
                    meteoVar = MapHourlyMeteoVar.at(varStdString);
                }
                catch (const std::out_of_range& ) {
                    meteoVar = noMeteoVar;
                }
            }
            if (meteoVar != noMeteoVar)
            {
                ret = _mapIdMeteoVar.insert(std::pair<int, meteoVariable>(id_variable, meteoVar));
                if (ret.second==false)
                {
                    error = "element 'z' already existed";
                }
            }
            else
            {
                error = "Wrong variable: " + variable;
                return false;
            }
        }
    }
    return true;
}


int Crit3DMeteoPointsDbHandler::getIdfromMeteoVar(meteoVariable meteoVar)
{
    std::map<int, meteoVariable>::const_iterator it;
    int key = NODATA;

    for (it = _mapIdMeteoVar.begin(); it != _mapIdMeteoVar.end(); ++it)
    {
        if (it->second == meteoVar)
        {
            key = it->first;
            break;
        }
    }
    return key;
}


bool Crit3DMeteoPointsDbHandler::existIdPoint(const QString& idPoint)
{
    QSqlQuery qry(_db);
    QString queryStr = "SELECT EXISTS(SELECT 1 FROM point_properties WHERE id_point='" + idPoint + "')";
    qry.prepare(queryStr);

    if (! qry.exec()) return false;
    qry.last();
    return (qry.value(0) > 0);
}


bool Crit3DMeteoPointsDbHandler::createTable(const QString& tableName, bool deletePrevious)
{
    QString queryStr;
    if (deletePrevious)
    {
        queryStr = "DROP TABLE IF EXISTS " + tableName;
        _db.exec(queryStr);
    }

    queryStr = "CREATE TABLE IF NOT EXISTS " + tableName + " (date_time TEXT(20), id_variable INTEGER, value REAL, PRIMARY KEY(date_time, id_variable))";
    QSqlQuery qry(_db);
    qry.prepare(queryStr);

    return qry.exec();
}


QString Crit3DMeteoPointsDbHandler::getNewDataEntry(int pos, const QStringList& dataStr, const QString& dateTimeStr,
                                                const QString& idVarStr, meteoVariable myVar,
                                                int* nrMissingData, int* nrWrongData, Crit3DQuality* dataQuality)
{
    if (dataStr.length() <= pos || dataStr.at(pos) == "")
    {
        (*nrMissingData)++;
        return "";
    }

    bool isNumber = false;
    float value = dataStr.at(pos).toFloat(&isNumber);
    if (! isNumber)
    {
        (*nrWrongData)++;
        return "";
    }

    if (dataQuality->syntacticQualitySingleValue(myVar, value) != quality::accepted)
    {
        (*nrWrongData)++;
        return "";
    }

    QString newEntry = "('" + dateTimeStr + "','" + idVarStr + "'," + QString::number(double(value)) + "),";
    return newEntry;
}


bool Crit3DMeteoPointsDbHandler::importHourlyMeteoData(QString csvFileName, bool deletePreviousData, QString* log)
{
    QString fileName = getFileName(csvFileName);
    *log = "\nInput file: " + fileName;

    // check point code
    QString pointCode = fileName.left(fileName.length()-4);
    if (! existIdPoint(pointCode))
    {
        *log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }

    // check input file
    QFile myFile(csvFileName);
    if(! myFile.open (QIODevice::ReadOnly))
    {
        *log += myFile.errorString();
        return false;
    }

    QTextStream myStream (&myFile);
    if (myStream.atEnd())
    {
        *log += "\nFile is void.";
        myFile.close();
        return false;
    }
    else
    {
        // skip first row (header)
        QStringList header = myStream.readLine().split(',');
    }

    // create table
    QString tableName = pointCode + "_H";
    if (! createTable(tableName, deletePreviousData))
    {
        *log += "\nError in create table: " + tableName + _db.lastError().text();
        myFile.close();
        return false;
    }

    QString idTavg = QString::number(getIdfromMeteoVar(airTemperature));
    QString idPrec = QString::number(getIdfromMeteoVar(precipitation));
    QString idRH = QString::number(getIdfromMeteoVar(airRelHumidity));
    QString idRad = QString::number(getIdfromMeteoVar(globalIrradiance));
    QString idWind = QString::number(getIdfromMeteoVar(windScalarIntensity));

    Crit3DQuality dataQuality;
    QStringList line;
    QDate currentDate, previousDate;
    int hour, previousHour = 0;
    QString dateTimeStr;
    int nrWrongDateTime = 0;
    int nrWrongData = 0;
    int nrMissingData = 0;
    QString queryStr = "INSERT INTO " + tableName + " VALUES";

    while(!myStream.atEnd())
    {
        line = myStream.readLine().split(',');

        // skip void lines
        if (line.length() <= 2) continue;

        // check date
        currentDate = QDate::fromString(line.at(0),"yyyy-MM-dd");
        if (! currentDate.isValid())
        {
            *log += "\nWrong dateTime: " + line.at(0) + " h" + line.at(1);
            nrWrongDateTime++;
            continue;
        }

        // check hour
        bool isNumber = false;
        hour = line.at(1).toInt(&isNumber);
        if (! isNumber || hour < 0 || hour > 23)
        {
            *log += "\nWrong dateTime: " + line.at(0) + " h" + line.at(1);
            nrWrongDateTime++;
            continue;
        }

        // don't use QDateTime because it has a bug at the end of March (vs2015 version)
        char timeStr[9];
        sprintf (timeStr, " %02d:00:00", hour);
        dateTimeStr = currentDate.toString("yyyy-MM-dd") + timeStr;

        // check duplicate
        if ((currentDate < previousDate) ||
            (currentDate == previousDate && hour <= previousHour))
        {
            *log += "\nDuplicate dateTime: " + dateTimeStr;
            nrWrongDateTime++;
            continue;
        }
        previousHour = hour;
        previousDate = currentDate;

        queryStr.append(getNewDataEntry(2, line, dateTimeStr, idTavg, airTemperature, &nrMissingData, &nrWrongData, &dataQuality));
        queryStr.append(getNewDataEntry(3, line, dateTimeStr, idPrec, precipitation, &nrMissingData, &nrWrongData, &dataQuality));
        queryStr.append(getNewDataEntry(4, line, dateTimeStr, idRH, airRelHumidity, &nrMissingData, &nrWrongData, &dataQuality));
        queryStr.append(getNewDataEntry(5, line, dateTimeStr, idRad, globalIrradiance, &nrMissingData, &nrWrongData, &dataQuality));
        queryStr.append(getNewDataEntry(6, line, dateTimeStr, idWind, windScalarIntensity, &nrMissingData, &nrWrongData, &dataQuality));
    }
    myFile.close();

    if (queryStr != "")
    {
        // remove the trailing comma
        queryStr.chop(1);

        // exec query
        QSqlQuery qry(_db);
        qry.prepare(queryStr);
        if (! qry.exec())
        {
            *log += "\nError in execute query: " + qry.lastError().text();
            return false;
        }
    }

    *log += "\nData imported successfully.";
    *log += "\nWrong date/time: " + QString::number(nrWrongDateTime);
    *log += "\nMissing data: " + QString::number(nrMissingData);
    *log += "\nWrong values: " + QString::number(nrWrongData);

    return true;
}

