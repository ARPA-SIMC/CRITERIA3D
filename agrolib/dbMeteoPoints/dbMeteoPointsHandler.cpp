#include "dbMeteoPointsHandler.h"
#include "commonConstants.h"
#include "meteo.h"
#include "utilities.h"
#include "basicMath.h"

#include <QtSql>


Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler()
{

}

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


QList<QString> Crit3DMeteoPointsDbHandler::getDatasetsActive()
{
    QList<QString> activeList;
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




QList<QString> Crit3DMeteoPointsDbHandler::getAllDatasetsList()
{
    QList<QString> datasetList;
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
    QList<QString> tables;
    QDateTime firstDate;
    QDate myDate;
    QTime myTime;

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
                        myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                        myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                        date = QDateTime(myDate, myTime, Qt::UTC);
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
    QList<QString> tables;
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
    QDate myDate;
    QTime myTime;
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
                        myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                        myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                        date = QDateTime(myDate, myTime, Qt::UTC);
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
    QDate myDate;
    QTime myTime;
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
                    myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                    myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                    firstDate = QDateTime(myDate, myTime, Qt::UTC);
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
    QDate myDate;
    QTime myTime;
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
                    myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                    myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                    lastDate = QDateTime(myDate, myTime, Qt::UTC);
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

bool Crit3DMeteoPointsDbHandler::deleteData(QString pointCode, frequencyType myFreq, QDate first, QDate last)
{
    QString tableName = pointCode + ((myFreq == daily) ?  "_D" : "_H");
    QSqlQuery qry(_db);
    QString statement;
    if (myFreq == daily)
    {
        QString firstStr = first.toString("yyyy-MM-dd");
        QString lastStr = last.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATE('%2') AND DATE('%3')")
                                .arg(tableName).arg(firstStr).arg(lastStr);
    }
    else
    {
        QString firstStr = first.toString("yyyy-MM-dd");
        QString lastStr = last.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATETIME('%2 00:00:00') AND DATETIME('%3 23:30:00')")
                                .arg(tableName).arg(firstStr).arg(lastStr);
    }

    return qry.exec(statement);
}


bool Crit3DMeteoPointsDbHandler::deleteData(QString pointCode, frequencyType myFreq, QList<meteoVariable> varList, QDate first, QDate last)
{
    QString tableName = pointCode + ((myFreq == daily) ?  "_D" : "_H");
    QString idList;
    QString id;
    for (int i = 0; i<varList.size(); i++)
    {
        id = QString::number(getIdfromMeteoVar(varList[i]));
        idList += id + ",";
    }
    idList = idList.left(idList.length() - 1);

    QSqlQuery qry(_db);
    QString statement;
    if (myFreq == daily)
    {
        QString firstStr = first.toString("yyyy-MM-dd");
        QString lastStr = last.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATE('%2') AND DATE('%3') AND `%4` IN (%5)")
                                .arg(tableName).arg(firstStr).arg(lastStr).arg(FIELD_METEO_VARIABLE).arg(idList);
    }
    else
    {
        QString firstStr = first.toString("yyyy-MM-dd");
        QString lastStr = last.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time "
                            "BETWEEN DATETIME('%2 00:00:00') "
                            "AND DATETIME('%3 23:30:00') "
                            "AND `%4` IN (%5)")
                            .arg(tableName).arg(firstStr).arg(lastStr).arg(FIELD_METEO_VARIABLE).arg(idList);
    }

    return qry.exec(statement);
}


bool Crit3DMeteoPointsDbHandler::deleteAllData(frequencyType myFreq)
{
    QSqlQuery qry(_db);
    QList<QString> tables;

    QString dayHour;
    if (myFreq == daily)
        dayHour = "D";
    else if (myFreq == hourly)
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

    QString statement;
    foreach (QString table, tables)
    {
        statement = QString( "DELETE FROM `%1`").arg(table);
        if( !qry.exec(statement) )
        {
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoPointsDbHandler::deleteAllPointsFromDataset(QList<QString> datasets)
{
    QList<QString> idList = getIdListGivenDataset(datasets);
    if (!idList.isEmpty())
    {
        return deleteAllPointsFromIdList(idList);
    }
    else
    {
        return false;
    }
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
    meteoVariable variable;
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
                                 .arg(tableName, startDate, endDate);
    if( !qry.exec(statement) )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
    {
        while (qry.next())
        {
            QDateTime d = qry.value(0).toDateTime();
            Crit3DDate myDate = Crit3DDate(d.date().day(), d.date().month(), d.date().year());
            //myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
            //myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
            //QDateTime d(QDateTime(myDate, myTime, Qt::UTC));

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
                meteoPoint->setMeteoPointValueH(myDate, d.time().hour(), d.time().minute(), variable, value);

                // copy scalar intensity to vector intensity (instantaneous values are equivalent, following WMO)
                // should be removed when hourly averages are available
                if (variable == windScalarIntensity)
                    meteoPoint->setMeteoPointValueH(myDate, d.time().hour(), d.time().minute(), windVectorIntensity, value);
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
    QDateTime previousDate;
    QDate myDate;
    QTime myTime;
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
                myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                QDateTime d(QDateTime(myDate, myTime, Qt::UTC));

                *firstDateDB = d;
                previousDate = *firstDateDB;

                value = qry.value(2).toFloat();

                hourlyVarList.push_back(value);
                firstRow = false;
            }
            else
            {
                dateStr = qry.value(0).toString();
                myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                QDateTime d(QDateTime(myDate, myTime, Qt::UTC));

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


bool Crit3DMeteoPointsDbHandler::setAndOpenDb(QString dbname_)
{
    error = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
        return false;
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);

    if (!_db.open())
    {
       error = _db.lastError().text();
       return false;
    }
    return true;
}

std::map<int, meteoVariable> Crit3DMeteoPointsDbHandler::getMapIdMeteoVar() const
{
    return _mapIdMeteoVar;
}

bool Crit3DMeteoPointsDbHandler::getPropertiesFromDb(QList<Crit3DMeteoPoint>& meteoPointsList,
                                        const gis::Crit3DGisSettings& gisSettings, QString& errorString)
{
    Crit3DMeteoPoint meteoPoint;
    QSqlQuery qry(_db);
    bool isLocationOk;

    qry.prepare( "SELECT id_point, name, dataset, latitude, longitude, utm_x, utm_y, altitude, state, region, province, municipality, is_active, is_utc, orog_code from point_properties ORDER BY id_point" );

    if( !qry.exec() )
    {
        errorString = qry.lastError().text();
        return false;
    }

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
        if ((int(meteoPoint.latitude) != int(NODATA) && int(meteoPoint.longitude) != int(NODATA))
            && (int(meteoPoint.point.utm.x) != int(NODATA) && int(meteoPoint.point.utm.y) != int(NODATA)))
        {
            double xTemp, yTemp;
            gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint.latitude, meteoPoint.longitude, &xTemp, &yTemp);
            if (fabs(xTemp - meteoPoint.point.utm.x) < 100 && fabs(yTemp - meteoPoint.point.utm.y) < 100)
            {
                isLocationOk = true;
            }
            else
            {
                errorString += "\nWrong location! "
                               + QString::fromStdString(meteoPoint.id) + " "
                               + QString::fromStdString(meteoPoint.name);
                isLocationOk = false;
            }
        }
        else if ((int(meteoPoint.latitude) == int(NODATA) || int(meteoPoint.longitude) == int(NODATA))
            && (int(meteoPoint.point.utm.x) != int(NODATA) && int(meteoPoint.point.utm.y) != int(NODATA)))
        {
            gis::getLatLonFromUtm(gisSettings, meteoPoint.point.utm.x, meteoPoint.point.utm.y,
                                    &(meteoPoint.latitude), &(meteoPoint.longitude));
            isLocationOk = true;
        }
        else if ((int(meteoPoint.latitude) != int(NODATA) && int(meteoPoint.longitude) != int(NODATA))
                 && (int(meteoPoint.point.utm.x) == int(NODATA) || int(meteoPoint.point.utm.y) == int(NODATA)))
        {
            gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint.latitude, meteoPoint.longitude,
                                      &(meteoPoint.point.utm.x), &(meteoPoint.point.utm.y));
            isLocationOk = true;
        }
        else
        {
            errorString += "\nMissing location (lat/lon or UTM): "
                           + QString::fromStdString(meteoPoint.id) + " "
                           + QString::fromStdString(meteoPoint.name);
            isLocationOk = false;
        }

        if (isLocationOk)
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

    return true;
}

bool Crit3DMeteoPointsDbHandler::getPropertiesGivenId(QString id, Crit3DMeteoPoint* meteoPoint,
                                        const gis::Crit3DGisSettings& gisSettings, QString& errorString)
{

    QSqlQuery qry(_db);
    bool isLocationOk;

    qry.prepare( "SELECT id_point, name, dataset, latitude, longitude, utm_x, utm_y, altitude, state, region, province, municipality, is_active, is_utc, orog_code from point_properties WHERE id_point = :id_point" );
    qry.bindValue(":id_point", id);

    if( !qry.exec() )
    {
        errorString = qry.lastError().text();
        return false;
    }

    while (qry.next())
    {
        meteoPoint->id = qry.value("id_point").toString().toStdString();
        meteoPoint->name = qry.value("name").toString().toStdString();
        meteoPoint->dataset = qry.value("dataset").toString().toStdString();

        if (qry.value("latitude") != "")
            meteoPoint->latitude = qry.value("latitude").toDouble();
        if (qry.value("longitude") != "")
            meteoPoint->longitude = qry.value("longitude").toDouble();
        if (qry.value("utm_x") != "")
            meteoPoint->point.utm.x = qry.value("utm_x").toDouble();
        if (qry.value("utm_y") != "")
            meteoPoint->point.utm.y = qry.value("utm_y").toDouble();
        if (qry.value("altitude") != "")
            meteoPoint->point.z = qry.value("altitude").toDouble();

        // check position
        if ((int(meteoPoint->latitude) != int(NODATA) && int(meteoPoint->longitude) != int(NODATA))
            && (int(meteoPoint->point.utm.x) != int(NODATA) && int(meteoPoint->point.utm.y) != int(NODATA)))
        {
            double xTemp, yTemp;
            gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint->latitude, meteoPoint->longitude, &xTemp, &yTemp);
            if (fabs(xTemp - meteoPoint->point.utm.x) < 100 && fabs(yTemp - meteoPoint->point.utm.y) < 100)
            {
                isLocationOk = true;
            }
            else
            {
                errorString += "\nWrong location! "
                               + id + " "
                               + QString::fromStdString(meteoPoint->name);
                isLocationOk = false;
            }
        }
        else if ((int(meteoPoint->latitude) == int(NODATA) || int(meteoPoint->longitude) == int(NODATA))
            && (int(meteoPoint->point.utm.x) != int(NODATA) && int(meteoPoint->point.utm.y) != int(NODATA)))
        {
            gis::getLatLonFromUtm(gisSettings, meteoPoint->point.utm.x, meteoPoint->point.utm.y,
                                    &(meteoPoint->latitude), &(meteoPoint->longitude));
            isLocationOk = true;
        }
        else if ((int(meteoPoint->latitude) != int(NODATA) && int(meteoPoint->longitude) != int(NODATA))
                 && (int(meteoPoint->point.utm.x) == int(NODATA) || int(meteoPoint->point.utm.y) == int(NODATA)))
        {
            gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint->latitude, meteoPoint->longitude,
                                      &(meteoPoint->point.utm.x), &(meteoPoint->point.utm.y));
            isLocationOk = true;
        }
        else
        {
            errorString += "\nMissing location (lat/lon or UTM): "
                           + id + " "
                           + QString::fromStdString(meteoPoint->name);
            isLocationOk = false;
        }

        if (isLocationOk)
        {
            meteoPoint->state = qry.value("state").toString().toStdString();
            meteoPoint->region = qry.value("region").toString().toStdString();
            meteoPoint->province = qry.value("province").toString().toStdString();
            meteoPoint->municipality = qry.value("municipality").toString().toStdString();
            meteoPoint->active = qry.value("is_active").toBool();
            meteoPoint->isUTC = qry.value("is_utc").toBool();
            meteoPoint->lapseRateCode = lapseRateCodeType((qry.value("orog_code").toInt()));
        }
    }

    return true;
}

QString Crit3DMeteoPointsDbHandler::getNameGivenId(QString id)
{

    QSqlQuery qry(_db);
    QString name = "";

    qry.prepare( "SELECT name from point_properties WHERE id_point = :id_point" );
    qry.bindValue(":id_point", id);

    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return name;
    }

    if(qry.next())
    {
        getValue(qry.value("name"), &name);
    }

    return name;
}

double Crit3DMeteoPointsDbHandler::getAltitudeGivenId(QString id)
{

    QSqlQuery qry(_db);
    double altitude = NODATA;

    qry.prepare( "SELECT altitude from point_properties WHERE id_point = :id_point" );
    qry.bindValue(":id_point", id);

    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return altitude;
    }

    if(qry.next())
    {
        getValue(qry.value("altitude"), &altitude);
    }

    return altitude;
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

bool Crit3DMeteoPointsDbHandler::updatePointProperties(QList<QString> columnList, QList<QString> valueList)
{

    if (columnList.size() != valueList.size())
    {
        qDebug() << "invalid input";
        return false;
    }
    QSqlQuery qry(_db);

    QString queryStr = QString("CREATE TABLE IF NOT EXISTS `%1`"
                               "(id_point TEXT(20), name TEXT(20), dataset TEXT(20), latitude NUMERIC, longitude REAL, latInt INTEGER, lonInt INTEGER, utm_x NUMERIC, utm_y NUMERIC,"
                               " altitude REAL, state TEXT(20), region TEXT(20), province TEXT(20), municipality TEXT(20), is_active INTEGER DEFAULT 1, is_utc INTEGER DEFAULT 1, "
                               "orog_code TEXT(20), PRIMARY KEY(id_point))").arg("point_properties");

    qry.prepare(queryStr);
    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return false;
    }

    queryStr = "INSERT OR REPLACE INTO point_properties (";
    for (int i = 0; i<columnList.size(); i++)
    {
        queryStr += columnList[i]+",";
    }
    queryStr.chop(1); // remove last ,
    queryStr += ") VALUES (";
    for (int i = 0; i<columnList.size(); i++)
    {
        queryStr += ":"+columnList[i]+",";
    }
    queryStr.chop(1); // remove last ,
    queryStr += ")";

    qry.prepare(queryStr);

    for (int i = 0; i<valueList.size(); i++)
    {
        qry.bindValue(":"+columnList[i], valueList[i]);
    }

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
        return true;

}

bool Crit3DMeteoPointsDbHandler::updatePointPropertiesGivenId(QString id, QList<QString> columnList, QList<QString> valueList)
{

    if (columnList.size() != valueList.size())
    {
        qDebug() << "invalid input";
        return false;
    }
    QSqlQuery qry(_db);

    QString queryStr = "UPDATE point_properties SET ";
    for (int i = 0; i<columnList.size(); i++)
    {
        valueList[i] = valueList[i].replace("'", "''");
        queryStr += columnList[i]+" = '" + valueList[i] + "',";
    }
    queryStr.chop(1); // remove last ,
    queryStr += " WHERE id_point = " + id;

    if( !qry.exec(queryStr) )
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

            meteoVar = getMeteoVar(varStdString);

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

bool Crit3DMeteoPointsDbHandler::getNameColumn(QString tableName, QList<QString>* columnList)
{
    QSqlQuery qry(_db);

    std::string varStdString;
    std::pair<std::map<int, meteoVariable>::iterator,bool> ret;

    QString statement = QString( "PRAGMA table_info('%1')").arg(tableName);
    if( !qry.exec(statement) )
    {
        error = qry.lastError().text();
        return false;
    }
    else
    {
        QString name;
        while (qry.next())
        {
            getValue(qry.value("name"), &name);
            *columnList << name;
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
    return (qry.value(0).toInt() > 0);
}


bool Crit3DMeteoPointsDbHandler::createTable(const QString& tableName, bool deletePrevious)
{
    QString queryStr;
    if (deletePrevious)
    {
        queryStr = "DROP TABLE IF EXISTS " + tableName;
        _db.exec(queryStr);
    }

    queryStr = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(date_time TEXT(20), id_variable INTEGER, value REAL, PRIMARY KEY(date_time, id_variable))").arg(tableName);
    QSqlQuery qry(_db);
    qry.prepare(queryStr);

    return qry.exec();
}


QString Crit3DMeteoPointsDbHandler::getNewDataEntry(int pos, const QList<QString>& dataStr, const QString& dateTimeStr,
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


/*!
    \name importHourlyMeteoData
    \brief import hourly meteo data from .csv files
    \details fixed format:
    DATE(yyyy-mm-dd), HOUR, TAVG, PREC, RHAVG, RAD, W_SCAL_INT
*/
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
        QString header = myStream.readLine();
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
    QList<QString> line;
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
        // fixed (GA 11/2021)
        char timeStr[10];
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
            *log += "\nError in execute query: " + qry.lastError().text() +"\n";
            *log += "Maybe there are missing or wrong data values.";
            return false;
        }
    }

    *log += "\nData imported successfully.";
    *log += "\nWrong date/time: " + QString::number(nrWrongDateTime);
    *log += "\nMissing data: " + QString::number(nrMissingData);
    *log += "\nWrong values: " + QString::number(nrWrongData);

    return true;
}

bool Crit3DMeteoPointsDbHandler::writeDailyDataList(QString pointCode, QList<QString> listEntries, QString* log)
{
    if (!existIdPoint(pointCode))
    {
        *log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }
    // create table
    bool deletePreviousData = false;
    QString tableName = pointCode + "_D";
    if (! createTable(tableName, deletePreviousData))
    {
        *log += "\nError in create table: " + tableName + _db.lastError().text();
        return false;
    }

    QString queryStr = QString(("INSERT OR REPLACE INTO `%1`"
                                " VALUES ")).arg(tableName);

    queryStr = queryStr + listEntries.join(",");

    // exec query
    QSqlQuery qry(_db);
    qry.prepare(queryStr);
    if (! qry.exec())
    {
        *log += "\nError in execute query: " + qry.lastError().text();
        return false;
    }
    else
    {
        return true;
    }
}

bool Crit3DMeteoPointsDbHandler::writeHourlyDataList(QString pointCode, QList<QString> listEntries, QString* log)
{
    if (!existIdPoint(pointCode))
    {
        *log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }
    // create table
    bool deletePreviousData = false;
    QString tableName = pointCode + "_H";
    if (! createTable(tableName, deletePreviousData))
    {
        *log += "\nError in create table: " + tableName + _db.lastError().text();
        return false;
    }

    QString queryStr = QString(("INSERT OR REPLACE INTO `%1`"
                                " VALUES ")).arg(tableName);

    queryStr = queryStr + listEntries.join(",");

    // exec query
    QSqlQuery qry(_db);
    qry.prepare(queryStr);
    if (! qry.exec())
    {
        *log += "\nError in execute query: " + qry.lastError().text();
        return false;
    }
    else
    {
        return true;
    }
}

bool Crit3DMeteoPointsDbHandler::setAllPointsActive()
{
    QSqlQuery qry(_db);

    qry.prepare( "UPDATE point_properties SET is_active = 1" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
    {
        return true;
    }
}

bool Crit3DMeteoPointsDbHandler::setAllPointsNotActive()
{
    QSqlQuery qry(_db);

    qry.prepare( "UPDATE point_properties SET is_active = 0" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
    {
        return true;
    }
}

/*
bool Crit3DMeteoPointsDbHandler::setGeoPointsListActiveState(QList<gis::Crit3DGeoPoint> pointList, bool activeState)
{
    QSqlQuery qry(_db);
    for (int i = 0; i<pointList.size(); i++)
    {
        double latitude = pointList.at(i).latitude;
        double longitude = pointList.at(i).longitude;
        qry.prepare( "UPDATE point_properties SET is_active = :activeState WHERE latitude = :latitude AND longitude = :longitude" );
        qry.bindValue(":activeState", activeState);
        qry.bindValue(":latitude", latitude);
        qry.bindValue(":longitude", longitude);

        if( !qry.exec() )
        {
            qDebug() << "(lon,lat)" << longitude << latitude << qry.lastError();
            return false;
        }
    }
    return true;
}


bool Crit3DMeteoPointsDbHandler::deleteAllPointsFromGeoPointList(const QList<gis::Crit3DGeoPoint> &pointList)
{
    QSqlQuery qry(_db);
    QList<QString> idPointList;
    QString idPoint;

    for (int i = 0; i<pointList.size(); i++)
    {
        double latitude = pointList.at(i).latitude;
        double longitude = pointList.at(i).longitude;

        qry.prepare( "SELECT * from point_properties WHERE latitude = :latitude AND longitude = :longitude" );
        qry.bindValue(":latitude", latitude);
        qry.bindValue(":longitude", longitude);

        if( !qry.exec() )
        {
            qDebug() << qry.lastError();
            return false;
        }
        else
        {
            while (qry.next())
            {
                getValue(qry.value("id_point"), &idPoint);
                idPointList << idPoint;
            }
        }
    }

    return deleteAllPointsFromIdList(idPointList);
}
*/


bool Crit3DMeteoPointsDbHandler::setActiveStatePointList(const QList<QString>& pointList, bool activeState)
{
    error = "";
    QString idList = "";
    for (int i = 0; i < pointList.size(); i++)
    {
        QString id_point = pointList.at(i);
        if (id_point != "")
        {
            if (idList != "")
                idList += ",";
            idList += "'" + id_point + "'";
        }
    }

    QString sqlStr = "UPDATE point_properties SET is_active = " + QString::number(activeState);
    sqlStr+= " WHERE id_point IN (" + idList + ")";

    QSqlQuery qry(_db);
    if( !qry.exec(sqlStr))
    {
        error = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::deleteAllPointsFromIdList(const QList<QString>& pointList)
{
    QSqlQuery qry(_db);

    error = "";
    for (int i = 0; i < pointList.size(); i++)
    {
        QString id_point = pointList[i];
        qry.prepare( "DELETE FROM point_properties WHERE id_point = :id_point" );
        qry.bindValue(":id_point", id_point);
        if( !qry.exec() )
        {
            error += id_point + " " + qry.lastError().text();
            return false;
        }

        // remove also tables
        QString table = id_point + "_H";
        QString queryStr = "DROP TABLE IF EXISTS '" + table +"'";
        if( !qry.exec(queryStr))
        {
            error += "\n" + qry.lastError().text();
        }

        table = id_point + "_D";
        queryStr = "DROP TABLE IF EXISTS '" + table +"'";
        if( !qry.exec(queryStr))
        {
            error += "\n" + qry.lastError().text();
        }
    }

    return true;
}


QList<QString> Crit3DMeteoPointsDbHandler::getMunicipalityList()
{
    QList<QString> municipalityList;
    QSqlQuery qry(_db);
    QString municipality;

    qry.prepare( "SELECT municipality from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return municipalityList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("municipality"), &municipality);
            if (!municipalityList.contains(municipality))
            {
                municipalityList << municipality;
            }
        }
    }
    return municipalityList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getProvinceList()
{
    QList<QString> provinceList;
    QSqlQuery qry(_db);
    QString province;

    qry.prepare( "SELECT province from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return provinceList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("province"), &province);
            if (!provinceList.contains(province))
            {
                provinceList << province;
            }
        }
    }
    return provinceList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getRegionList()
{
    QList<QString> regionList;
    QSqlQuery qry(_db);
    QString region;

    qry.prepare( "SELECT region from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return regionList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("region"), &region);
            if (!regionList.contains(region))
            {
                regionList << region;
            }
        }
    }
    return regionList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getStateList()
{
    QList<QString> stateList;
    QSqlQuery qry(_db);
    QString state;

    qry.prepare( "SELECT state from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return stateList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("state"), &state);
            if (!stateList.contains(state))
            {
                stateList << state;
            }
        }
    }
    return stateList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getDatasetList()
{
    QList<QString> datasetList;
    QSqlQuery qry(_db);
    QString dataset;

    qry.prepare( "SELECT dataset from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return datasetList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("dataset"), &dataset);
            if (!datasetList.contains(dataset))
            {
                datasetList << dataset;
            }
        }
    }
    return datasetList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getIdList()
{
    QList<QString> idList;
    QSqlQuery qry(_db);
    QString id;

    qry.prepare( "SELECT id_point from point_properties" );

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return idList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_point"), &id);
            if (!idList.contains(id))
            {
                idList << id;
            }
        }
    }
    return idList;
}

QList<QString> Crit3DMeteoPointsDbHandler::getIdListGivenDataset(QList<QString> datasets)
{
    QList<QString> idList;
    QSqlQuery qry(_db);
    QString id;

    QString datasetList;
    for (int i = 0; i < datasets.size(); i++)
    {
        QString dataset = datasets.at(i);
        if (dataset != "")
        {
            if (datasetList != "")
                datasetList += ",";
            datasetList += "'" + dataset + "'";
        }
    }
    QString statement = "SELECT id_point from point_properties WHERE dataset IN  (" + datasetList + ")";

    if( !qry.exec(statement) )
    {
        qDebug() << qry.lastError();
        return idList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_point"), &id);
            if (!idList.contains(id))
            {
                idList << id;
            }
        }
    }
    return idList;
}

QString Crit3DMeteoPointsDbHandler::getDatasetFromId(const QString& idPoint)
{

    QSqlQuery qry(_db);
    QString dataset;
    dataset.clear();

    qry.prepare( "SELECT dataset from point_properties WHERE id_point = :id_point");
    qry.bindValue(":id_point", idPoint);

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return dataset;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("dataset"), &dataset);
        }
    }
    return dataset;
}

int Crit3DMeteoPointsDbHandler::getArkIdFromVar(const QString& variable)
{

    QSqlQuery qry(_db);
    int arkId = NODATA;

    qry.prepare( "SELECT id_arkimet from variable_properties WHERE variable = :variable");
    qry.bindValue(":variable", variable);

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return arkId;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_arkimet"), &arkId);
        }
    }
    return arkId;
}

bool Crit3DMeteoPointsDbHandler::setActiveStateIfCondition(bool activeState, QString condition)
{
    QSqlQuery qry(_db);
    QString statement;

    if (activeState)
    {
        statement = QString("UPDATE point_properties SET is_active = 1 WHERE %1 ").arg(condition);
    }
    else
    {
        statement = QString("UPDATE point_properties SET is_active = 0 WHERE %1 ").arg(condition);
    }

    if( !qry.exec(statement) )
    {
        qDebug() << qry.lastError();
        return false;
    }
    else
    {
        return true;
    }

}

bool Crit3DMeteoPointsDbHandler::setOrogCode(QString id, int orogCode)
{
    QSqlQuery qry(_db);

    qry.prepare( "UPDATE point_properties SET orog_code = :orogCode WHERE id_point = :id" );
    qry.bindValue(":orogCode", orogCode);
    qry.bindValue(":id", id);

    if( !qry.exec() )
    {
        error += id + " " + qry.lastError().text();
        return false;
    }
    else
    {
        return true;
    }

}

QList<QString> Crit3DMeteoPointsDbHandler::getJointStations(const QString& idPoint)
{

    QSqlQuery qry(_db);
    QList<QString> stationsList;
    QString station;

    qry.prepare( "SELECT joint_station from joint_stations WHERE id_point = :id_point");
    qry.bindValue(":id_point", idPoint);

    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
        return stationsList;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("joint_station"), &station);
            if (!stationsList.contains(station))
            {
                stationsList << station;
            }
        }
    }
    return stationsList;
}

bool Crit3DMeteoPointsDbHandler::setJointStations(const QString& idPoint, QList<QString> stationsList)
{

    QSqlQuery qry(_db);

    QString queryStr;
    queryStr = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(id_point TEXT, joint_station TEXT, PRIMARY KEY(id_point, joint_station))").arg("joint_stations");
    qry.prepare(queryStr);
    if( !qry.exec() )
    {
        error += idPoint + " " + qry.lastError().text();
        return false;
    }

    qry.prepare( "DELETE FROM joint_stations WHERE id_point = :id_point" );
    qry.bindValue(":id_point", idPoint);
    if( !qry.exec() )
    {
        error += idPoint + " " + qry.lastError().text();
        return false;
    }

    error.clear();
    for (int i = 0; i < stationsList.size(); i++)
    {
        qry.prepare( "INSERT INTO joint_stations (id_point, joint_station) VALUES (:id_point, :joint_station)" );

        qry.bindValue(":id_point", idPoint);
        qry.bindValue(":joint_station", stationsList[i]);
        if( !qry.exec() )
        {
            error += idPoint + "," + stationsList[i] + " " + qry.lastError().text();
        }
    }
    if (error.isEmpty())
    {
        return true;
    }
    else
    {
        return false;
    }
}
