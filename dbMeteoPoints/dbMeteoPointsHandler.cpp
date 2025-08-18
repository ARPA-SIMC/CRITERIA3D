#include "dbMeteoPointsHandler.h"
#include "commonConstants.h"
#include "meteo.h"
#include "utilities.h"
#include "basicMath.h"

#include <QtSql>


Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler()
{
    _errorStr = "";
    _mapIdMeteoVar.clear();
}

Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler(QString provider_, QString host_, QString dbname_, int port_,
                                                       QString user_, QString pass_)
{
    _errorStr = "";
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

    if (! _db.open())
    {
       _errorStr = "Error in opening " + _db.connectionName() + "\n" + _db.lastError().text() ;
    }
}


Crit3DMeteoPointsDbHandler::Crit3DMeteoPointsDbHandler(QString _dbName)
{
    _errorStr = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _dbName << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(_dbName);

    if (! _db.open())
    {
       _errorStr = "Error in opening " + _dbName + "\n" +_db.lastError().text();
    }
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


QString Crit3DMeteoPointsDbHandler::getDatasetURL(QString dataset, bool &isOk)
{
    _errorStr = "";
    QString queryStr = QString("SELECT URL FROM datasets WHERE dataset = '%1' OR dataset = '%2'").arg(dataset, dataset.toUpper());

    QSqlQuery qry(_db);
    if(! qry.exec(queryStr))
    {
        isOk = false;
        _errorStr = qry.lastError().text();
        return "";
    }
    else
    {
        if (qry.next())
        {
            isOk = true;
            return qry.value(0).toString();
        }
        else
        {
            isOk = false;
            _errorStr = "dataset " + dataset + " not found in the table 'datasets'";
            return "";
        }
    }
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
    QSqlQuery myQuery(_db);
    myQuery.exec(statement);

    statement = QString("UPDATE datasets SET active = 1 WHERE dataset IN ( %1 )").arg(active);
    myQuery.exec(statement);
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
    QString dayHour;
    if (frequency == daily)
        dayHour = "D";
    else if (frequency == hourly)
        dayHour = "H";

    QSqlQuery qry(_db);
    QList<QString> tableList;

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table' AND name like :dayHour ESCAPE '^'");
    qry.bindValue(":dayHour",  "%^" + dayHour  + "%");
    QDateTime firstDateTime;

    if(! qry.exec() )
    {
        _errorStr = qry.lastError().text();
        return firstDateTime;
    }
    else
    {
        while (qry.next())
        {
            QString table = qry.value(0).toString();
            tableList << table;
        }
    }

    QDateTime dateTime;
    QDate myDate;
    QTime myTime;
    int maxTableNr = 300;
    int step = std::max(1, int(tableList.size() / maxTableNr));
    int count = 0;

    foreach (QString table, tableList)
    {
        count++;
        if ((count % step) != 0) continue;

        QString statement = QString( "SELECT MIN(date_time) FROM `%1` AS dateTime").arg(table);
        if(qry.exec(statement) )
        {
            if (qry.next())
            {
                QString dateStr = qry.value(0).toString();
                if (! dateStr.isEmpty())
                {
                    if (frequency == daily)
                    {
                        dateTime = QDateTime::fromString(dateStr,"yyyy-MM-dd");
                    }
                    else if (frequency == hourly)
                    {
                        myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                        myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                        dateTime = QDateTime(myDate, myTime, Qt::UTC);
                    }

                    if (firstDateTime.isNull() || dateTime < firstDateTime)
                    {
                        firstDateTime = dateTime;
                    }
                }
            }
        }
    }

    return firstDateTime;
}


QDateTime Crit3DMeteoPointsDbHandler::getLastDate(frequencyType frequency)
{
    QSqlQuery qry(_db);
    QList<QString> tableList;

    QString dayHour;
    if (frequency == daily)
        dayHour = "D";
    else if (frequency == hourly)
        dayHour = "H";

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table' AND name like :dayHour ESCAPE '^'");
    qry.bindValue(":dayHour",  "%^_" + dayHour  + "%");

    QDateTime lastDateTime;
    if(! qry.exec() )
    {
        _errorStr = qry.lastError().text();
        return lastDateTime;
    }
    else
    {
        while (qry.next())
        {
            QString table = qry.value(0).toString();
            tableList << table;
        }
    }

    QDateTime dateTime;
    QDate myDate;
    QTime myTime;
    int maxTableNr = 300;
    int step = std::max(1, int(tableList.size() / maxTableNr));
    int count = 0;
    foreach (QString table, tableList)
    {
        count++;
        if ((count % step) != 0) continue;

        QString statement = QString( "SELECT MAX(date_time) FROM `%1` AS dateTime").arg(table);
        if(qry.exec(statement))
        {
            if (qry.next())
            {
                QString dateStr = qry.value(0).toString();
                if (! dateStr.isEmpty())
                {
                    if (frequency == daily)
                    {
                        myDate = QDate::fromString(dateStr, "yyyy-MM-dd");
                        myTime = QTime(12, 0, 0, 0);
                        dateTime = QDateTime(myDate, myTime, Qt::UTC);
                    }
                    else if (frequency == hourly)
                    {
                        myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
                        myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
                        dateTime = QDateTime(myDate, myTime, Qt::UTC);
                    }

                    if (lastDateTime.isNull() || dateTime > lastDateTime)
                    {
                        lastDateTime = dateTime;
                    }
                }
            }
        }
    }

    return lastDateTime;
}


// return a null datetime if the table doesn't exist or table is void
QDateTime Crit3DMeteoPointsDbHandler::getFirstDate(frequencyType frequency, const std::string &idMeteoPoint)
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
QDateTime Crit3DMeteoPointsDbHandler::getLastDate(frequencyType frequency, const std::string &idMeteoPoint)
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


bool Crit3DMeteoPointsDbHandler::existData(const Crit3DMeteoPoint &meteoPoint, frequencyType myFreq)
{
    QSqlQuery query(_db);
    QString tableName = QString::fromStdString(meteoPoint.id) + ((myFreq == daily) ?  "_D" : "_H");
    QString statement = QString( "SELECT 1 FROM `%1`").arg(tableName);

    if (query.exec(statement))
        if (query.next())
            return true;

    return false;
}


bool Crit3DMeteoPointsDbHandler::deleteData(const QString &pointCode, frequencyType frequency, const QDate &firstDate, const QDate& lastDate)
{
    QString tableName = pointCode + ((frequency == daily) ?  "_D" : "_H");
    if (! _db.tables().contains(tableName))
    {
        // table doesn't exist
        return true;
    }

    QSqlQuery qry(_db);
    QString statement;
    if (frequency == daily)
    {
        QString firstStr = firstDate.toString("yyyy-MM-dd");
        QString lastStr = lastDate.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATE('%2') AND DATE('%3')")
                                .arg(tableName, firstStr, lastStr);
    }
    else
    {
        QString firstStr = firstDate.toString("yyyy-MM-dd");
        QString lastStr = lastDate.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATETIME('%2 00:00:00') AND DATETIME('%3 23:30:00')")
                                .arg(tableName, firstStr, lastStr);
    }

    return qry.exec(statement);
}


bool Crit3DMeteoPointsDbHandler::deleteData(const QString &pointCode, frequencyType frequency, const QList<meteoVariable> &varList, const QDate &firstDate, const QDate &lastDate)
{
    QString tableName = pointCode + ((frequency == daily) ?  "_D" : "_H");
    QString idList;
    QString id;
    for (int i = 0; i < varList.size(); i++)
    {
        id = QString::number(getIdfromMeteoVar(varList[i]));
        idList += id + ",";
    }
    idList = idList.left(idList.length() - 1);

    QSqlQuery qry(_db);
    QString statement;
    if (frequency == daily)
    {
        QString firstStr = firstDate.toString("yyyy-MM-dd");
        QString lastStr = lastDate.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time BETWEEN DATE('%2') AND DATE('%3') AND `%4` IN (%5)")
                                .arg(tableName, firstStr, lastStr, FIELD_METEO_VARIABLE, idList);
    }
    else
    {
        QString firstStr = firstDate.toString("yyyy-MM-dd");
        QString lastStr = lastDate.toString("yyyy-MM-dd");
        statement = QString( "DELETE FROM `%1` WHERE date_time "
                            "BETWEEN DATETIME('%2 00:00:00') "
                            "AND DATETIME('%3 23:30:00') "
                            "AND `%4` IN (%5)")
                            .arg(tableName, firstStr, lastStr, FIELD_METEO_VARIABLE, idList);
    }

    return qry.exec(statement);
}


bool Crit3DMeteoPointsDbHandler::deleteAllData(frequencyType frequency)
{
    QSqlQuery qry(_db);
    QList<QString> tables;

    QString dayHour;
    if (frequency == daily)
        dayHour = "D";
    else if (frequency == hourly)
        dayHour = "H";

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table' AND name like :dayHour ESCAPE '^'");
    qry.bindValue(":dayHour",  "%^" + dayHour  + "%");

    if( !qry.exec() )
    {
        _errorStr = qry.lastError().text();
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


bool Crit3DMeteoPointsDbHandler::loadDailyData(const Crit3DDate &firstDate, const Crit3DDate &lastDate, Crit3DMeteoPoint &meteoPoint)
{
    // check dates
    if (firstDate > lastDate)
    {
        _errorStr = "wrong dates: first > last";
        return false;
    }

    int numberOfDays = difference(firstDate, lastDate) + 1;
    meteoPoint.initializeObsDataD(numberOfDays, firstDate);

    QString firstDateStr = QString::fromStdString(firstDate.toISOString());
    QString lastDateStr = QString::fromStdString(lastDate.toISOString());
    QString tableName = QString::fromStdString(meteoPoint.id) + "_D";

    QString statement;
    if (numberOfDays == 1)
    {
        statement = QString( "SELECT * FROM `%1` WHERE date_time = DATE('%2')").arg(tableName, firstDateStr);
    }
    else
    {
        statement = QString( "SELECT * FROM `%1` WHERE date_time >= DATE('%2') AND date_time < DATE('%3', '+1 day')")
                                .arg(tableName, firstDateStr, lastDateStr);
    }

    QSqlQuery query(_db);
    if(! query.exec(statement))
    {
        return false;
        _errorStr = query.lastError().text();
    }

    while (query.next())
    {
        // date
        QString dateStr = query.value(0).toString();
        QDate d = QDate::fromString(dateStr, "yyyy-MM-dd");

        // variable
        meteoVariable variable = noMeteoVar;
        int idVar = query.value(1).toInt();
        if (idVar != NODATA)
        {
            variable = _mapIdMeteoVar.at(idVar);
        }

        // value
        float value = query.value(2).toFloat();

        meteoPoint.setMeteoPointValueD(Crit3DDate(d.day(), d.month(), d.year()), variable, value);
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::loadHourlyData(const Crit3DDate &firstDate, const Crit3DDate &lastDate, Crit3DMeteoPoint &meteoPoint)
{
    // check dates
    if (firstDate > lastDate)
    {
        _errorStr = "wrong dates: firstDate > lastDate";
        return false;
    }

    // initialize obs data
    int numberOfDays = difference(firstDate, lastDate) + 1;
    int myHourlyFraction = 1;
    meteoPoint.initializeObsDataH(myHourlyFraction, numberOfDays, firstDate);

    QString startDateStr = QString::fromStdString(firstDate.toISOString());
    QString endDateStr = QString::fromStdString(lastDate.toISOString());
    QString tableName = QString::fromStdString(meteoPoint.id) + "_H";

    QString statement = QString( "SELECT * FROM `%1` WHERE date_time >= DATETIME('%2 01:00:00') AND date_time <= DATETIME('%3 00:00:00', '+1 day')")
                                 .arg(tableName, startDateStr, endDateStr);
    QSqlQuery qry(_db);
    if(! qry.exec(statement) )
    {
        _errorStr = qry.lastError().text();
        return false;
    }

    while (qry.next())
    {
        Crit3DTime dateTime;
        if (getValueCrit3DTime(qry.value(0), &dateTime))
        {
            int hour = dateTime.getHour();
            int minute = dateTime.getMinutes();

            meteoVariable variable;
            int idVar = qry.value(1).toInt();
            try
            {
                variable = _mapIdMeteoVar.at(idVar);
            }
            catch (const std::out_of_range& )
            {
                variable = noMeteoVar;
            }

            if (variable != noMeteoVar)
            {
                float value = qry.value(2).toFloat();
                meteoPoint.setMeteoPointValueH(dateTime.date, hour, minute, variable, value);

                // copy scalar intensity to vector intensity (instantaneous values are equivalent, following WMO)
                // should be removed when hourly averages are available
                if (variable == windScalarIntensity)
                {
                    meteoPoint.setMeteoPointValueH(dateTime.date, hour, minute, windVectorIntensity, value);
                }
            }
        }
    }

    return true;
}


std::vector<float> Crit3DMeteoPointsDbHandler::loadDailyVar(meteoVariable variable, const Crit3DDate &dateStart,
                                                            const Crit3DDate &dateEnd, const QString &idStr, QDate &firstDateDB)
{
    int idVar = getIdfromMeteoVar(variable);
    QString startDate = QString::fromStdString(dateStart.toISOString());
    QString endDate = QString::fromStdString(dateEnd.toISOString());

    QSqlQuery query(_db);
    QString tableName = idStr + "_D";
    QString statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATE('%4') AND date_time < DATE('%5', '+1 day')")
                                .arg(tableName, FIELD_METEO_VARIABLE).arg(idVar).arg(startDate, endDate);

    std::vector<float> outputList;
    if(! query.exec(statement))
    {
        _errorStr = query.lastError().text();
        return outputList;
    }

    QString dateStr;
    QDate previousDate;
    bool isFirstRow = true;
    while (query.next())
    {
        if (isFirstRow)
        {
            dateStr = query.value(0).toString();
            firstDateDB = QDate::fromString(dateStr, "yyyy-MM-dd");
            previousDate = firstDateDB;

            float value = query.value(2).toFloat();

            outputList.push_back(value);

            isFirstRow = false;
        }
        else
        {
            dateStr = query.value(0).toString();
            QDate currentDate = QDate::fromString(dateStr, "yyyy-MM-dd");

            int missingDate = previousDate.daysTo(currentDate);
            for (int i=1; i < missingDate; i++)
            {
                outputList.push_back(NODATA);
            }

            float value = query.value(2).toFloat();

            outputList.push_back(value);
            previousDate = currentDate;
        }
    }

    return outputList;
}


std::vector<float> Crit3DMeteoPointsDbHandler::exportAllDataVar(QString *myError, frequencyType freq, meteoVariable variable, QString id, QDateTime myFirstTime, QDateTime myLastTime, std::vector<QString> &dateStr)
{
    std::vector<float> allDataVarList;

    int idVar = getIdfromMeteoVar(variable);

    QSqlQuery query(_db);
    QString tableName, startDate, endDate, statement;

    if (freq == daily)
    {
        tableName = id + "_D";
        startDate = myFirstTime.date().toString("yyyy-MM-dd");
        endDate = myLastTime.date().toString("yyyy-MM-dd");
        statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATE('%4') AND date_time <= DATE('%5')")
                                .arg(tableName, FIELD_METEO_VARIABLE).arg(idVar).arg(startDate, endDate);
    }
    else if (freq == hourly)
    {
        tableName = id + "_H";
        startDate = myFirstTime.date().toString("yyyy-MM-dd") + " " + myFirstTime.time().toString("hh:mm");
        endDate = myLastTime.date().toString("yyyy-MM-dd") + " " + myLastTime.time().toString("hh:mm");
        statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATETIME('%4') AND date_time <= DATETIME('%5')")
                                .arg(tableName, FIELD_METEO_VARIABLE).arg(idVar).arg(startDate, endDate);
    }
    else
    {
        *myError = "Frequency should be daily or hourly";
        return allDataVarList;
    }

    if(! query.exec(statement) )
    {
        *myError = query.lastError().text();
        return allDataVarList;
    }

    QDate date;
    float value;
    QString myDateStr;
    Crit3DTime dateTime;

    while (query.next())
    {
        if (freq == daily)
        {
            if (! getValue(query.value(0), &date))
            {
                *myError = "Missing date_time";
                return allDataVarList;
            }
            myDateStr = date.toString("yyyy-MM-dd");
        }
        else if (freq == hourly)
        {
            if (! getValueCrit3DTime(query.value(0), &dateTime))
            {
                *myError = "Missing date_time";
                return allDataVarList;
            }
            myDateStr = QString::fromStdString(dateTime.toISOString());
        }
        dateStr.push_back(myDateStr);
        value = query.value(2).toFloat();
        allDataVarList.push_back(value);
    }

    return allDataVarList;
}


std::vector<float> Crit3DMeteoPointsDbHandler::loadHourlyVar(meteoVariable variable, const QString& meteoPointId,
                                                             const QDateTime& startTime, const QDateTime& endTime,
                                                             QDateTime &firstDateDB, QString &myError)
{
    std::vector<float> hourlyVarList;

    int idVar = getIdfromMeteoVar(variable);
    QString tableName = meteoPointId + "_H";

    QString statement = QString( "SELECT * FROM `%1` WHERE `%2` = %3 AND date_time >= DATETIME('%4') AND date_time <= DATETIME('%5') ORDER BY date_time")
                                 .arg(tableName, FIELD_METEO_VARIABLE).arg(idVar).arg(startTime.toString("yyyy-MM-dd hh:mm:ss"), endTime.toString("yyyy-MM-dd hh:mm:ss"));

    QSqlQuery qry(_db);
    if(! qry.exec(statement))
    {
        myError = qry.lastError().text();
        return hourlyVarList;
    }

    bool isFirstRow = true;
    QDateTime previousDate;

    while (qry.next())
    {
        QString dateStr = qry.value(0).toString();
        QDate myDate = QDate::fromString(dateStr.mid(0,10), "yyyy-MM-dd");
        QTime myTime = QTime::fromString(dateStr.mid(11,8), "HH:mm:ss");
        QDateTime currentDate(QDateTime(myDate, myTime, Qt::UTC));

        if (isFirstRow)
        {
            firstDateDB = currentDate;
            isFirstRow = false;
        }
        else
        {
            int deltaHours = (currentDate.currentSecsSinceEpoch() - previousDate.currentSecsSinceEpoch()) / 3600;
            if (deltaHours > 1)
            {
                // fill missing hours
                for (int i=1; i < deltaHours; i++)
                {
                    hourlyVarList.push_back(NODATA);
                }
            }
        }

        previousDate = currentDate;
        float value = qry.value(2).toFloat();
        hourlyVarList.push_back(value);
    }

    return hourlyVarList;
}


bool Crit3DMeteoPointsDbHandler::setAndOpenDb(QString dbname_)
{
    _errorStr = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
        return false;
    }

    if (! QFile(dbname_).exists())
    {
        _errorStr = "Meteo points DB does not exists:\n" + dbname_;
        return false;
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);

    if (!_db.open())
    {
       _errorStr = _db.lastError().text();
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

    if(!qry.exec() )
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


bool Crit3DMeteoPointsDbHandler::getPropertiesGivenId(const QString &id, Crit3DMeteoPoint &meteoPoint,
                                        const gis::Crit3DGisSettings &gisSettings, QString &errorString)
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
                               + id + " "
                               + QString::fromStdString(meteoPoint.name);
                isLocationOk = false;
            }
        }
        else if ( (int(meteoPoint.latitude) == int(NODATA) || int(meteoPoint.longitude) == int(NODATA))
                 && (int(meteoPoint.point.utm.x) != int(NODATA) && int(meteoPoint.point.utm.y) != int(NODATA)) )
        {
            gis::getLatLonFromUtm(gisSettings, meteoPoint.point.utm.x, meteoPoint.point.utm.y,
                                    &(meteoPoint.latitude), &(meteoPoint.longitude));
            isLocationOk = true;
        }
        else if ( (int(meteoPoint.latitude) != int(NODATA) && int(meteoPoint.longitude) != int(NODATA))
                 && (int(meteoPoint.point.utm.x) == int(NODATA) || int(meteoPoint.point.utm.y) == int(NODATA)))
        {
            gis::latLonToUtmForceZone(gisSettings.utmZone, meteoPoint.latitude, meteoPoint.longitude,
                                      &(meteoPoint.point.utm.x), &(meteoPoint.point.utm.y));
            isLocationOk = true;
        }
        else
        {
            errorString += "\nMissing location (lat/lon or UTM): "
                           + id + " "
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
        _errorStr = qry.lastError().text();
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
        _errorStr = qry.lastError().text();
        return altitude;
    }

    if(qry.next())
    {
        getValue(qry.value("altitude"), &altitude);
    }

    return altitude;
}


bool Crit3DMeteoPointsDbHandler::writePointProperties(const Crit3DMeteoPoint &myPoint)
{

    QSqlQuery qry(_db);

    qry.prepare( "INSERT INTO point_properties (id_point, name, dataset, latitude, longitude, latInt, lonInt, utm_x, utm_y, altitude, state, region, province, municipality)"
                                      " VALUES (:id_point, :name, :dataset, :latitude, :longitude, :latInt, :lonInt, :utm_x, :utm_y, :altitude, :state, :region, :province, :municipality)" );

    qry.bindValue(":id_point", QString::fromStdString(myPoint.id));
    qry.bindValue(":name", QString::fromStdString(myPoint.name));
    qry.bindValue(":dataset", QString::fromStdString(myPoint.dataset));
    qry.bindValue(":latitude", myPoint.latitude);
    qry.bindValue(":longitude", myPoint.longitude);
    qry.bindValue(":latInt", myPoint.latInt);
    qry.bindValue(":lonInt", myPoint.lonInt);
    qry.bindValue(":utm_x", myPoint.point.utm.x);
    qry.bindValue(":utm_y", myPoint.point.utm.y);
    qry.bindValue(":altitude", myPoint.point.z);
    qry.bindValue(":state", QString::fromStdString(myPoint.state));
    qry.bindValue(":region", QString::fromStdString(myPoint.region));
    qry.bindValue(":province", QString::fromStdString(myPoint.province));
    qry.bindValue(":municipality", QString::fromStdString(myPoint.municipality));

    if( ! qry.exec() )
    {
        _errorStr = qry.lastError().text();
        return false;
    }
    else
        return true;
}


bool Crit3DMeteoPointsDbHandler::updatePointProperties(const QList<QString> &columnList, const QList<QString> &valueList)
{
    if (columnList.size() != valueList.size())
    {
        _errorStr = "invalid input: nr columns != nr values";
        return false;
    }
    QSqlQuery qry(_db);

    QString queryStr = QString("CREATE TABLE IF NOT EXISTS `%1`"
                               "(id_point TEXT(20), name TEXT(20), dataset TEXT(20), latitude NUMERIC, longitude REAL, latInt INTEGER, lonInt INTEGER, utm_x NUMERIC, utm_y NUMERIC,"
                               " altitude REAL, state TEXT(20), region TEXT(20), province TEXT(20), municipality TEXT(20), is_active INTEGER DEFAULT 1, is_utc INTEGER DEFAULT 1, "
                               "orog_code TEXT(20), PRIMARY KEY(id_point))").arg("point_properties");

    qry.prepare(queryStr);
    if( ! qry.exec() )
    {
        _errorStr = qry.lastError().text();
        return false;
    }

    queryStr = "INSERT OR REPLACE INTO point_properties (";
    for (int i = 0; i < columnList.size(); i++)
    {
        queryStr += columnList[i]+",";
    }
    queryStr.chop(1);                       // remove last comma
    queryStr += ") VALUES (";
    for (int i = 0; i<columnList.size(); i++)
    {
        queryStr += ":" + columnList[i] + ",";
    }
    queryStr.chop(1);                       // remove last comma
    queryStr += ")";

    qry.prepare(queryStr);

    for (int i = 0; i < valueList.size(); i++)
    {
        qry.bindValue(":" + columnList[i], valueList[i]);
    }

    if(! qry.exec())
    {
        _errorStr = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::updatePointPropertiesGivenId(const QString &idStr, const QList<QString>& columnList, QList<QString>& valueList)
{
    if (columnList.size() != valueList.size())
    {
        _errorStr = "Error in updatePointProperties: " + idStr + "\n" + "invalid input";
        return false;
    }

    QSqlQuery qry(_db);

    QString queryStr = "UPDATE point_properties SET ";
    for (int i = 0; i<columnList.size(); i++)
    {
        valueList[i] = valueList[i].replace("'", "''");
        queryStr += columnList[i] + " = '" + valueList[i] + "',";
    }

    queryStr.chop(1);       // remove last comma
    queryStr += " WHERE id_point = " + idStr;

    if(! qry.exec(queryStr))
    {
        _errorStr = "Error in updatePointProperties Id: " + idStr + "\n" + qry.lastError().text();
        return false;
    }

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
        _errorStr = qry.lastError().text();
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
                    _errorStr = "element 'z' already existed";
                }
            }
            else
            {
                _errorStr = "Wrong variable: " + variable;
                return false;
            }
        }
    }
    return true;
}


bool Crit3DMeteoPointsDbHandler::getFieldList(const QString &tableName, QList<QString> &fieldList)
{
    QSqlQuery qry(_db);
    QString statement = QString( "PRAGMA table_info('%1')").arg(tableName);

    if( !qry.exec(statement) )
    {
        _errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        QString name;
        while (qry.next())
        {
            getValue(qry.value("name"), &name);
            fieldList << name;
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

    QString newEntry = "('" + dateTimeStr + "'," + idVarStr + "," + QString::number(double(value)) + "),";
    return newEntry;
}


/*!
    \name importHourlyMeteoData
    \brief import hourly meteo data from .csv files
    \details fixed format:
    DATE(yyyy-mm-dd), HOUR, TAVG, PREC, RHAVG, RAD, W_SCAL_INT
    - the filename must be equal to pointcode
    - header is mandatory
*/
bool Crit3DMeteoPointsDbHandler::importHourlyMeteoData(const QString &csvFileName, bool deletePreviousData, QString &log)
{
    QString fileName = getFileName(csvFileName);
    log = "";

    // check point code
    QString pointCode = fileName.left(fileName.length()-4);
    if (! existIdPoint(pointCode))
    {
        log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }

    // check input file
    QFile myFile(csvFileName);
    if(! myFile.open (QIODevice::ReadOnly))
    {
        log += myFile.errorString();
        return false;
    }

    QTextStream myStream (&myFile);
    if (myStream.atEnd())
    {
        log += "\nFile is void.";
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
        log += "\nError in create table: " + tableName + _db.lastError().text();
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
    QString queryStr = "INSERT INTO '" + tableName + "' VALUES";

    while(! myStream.atEnd())
    {
        line = myStream.readLine().split(',');

        // skip void lines
        if (line.length() <= 2) continue;

        // check date
        currentDate = QDate::fromString(line.at(0),"yyyy-MM-dd");
        if (! currentDate.isValid())
        {
            log += "\nWrong dateTime: " + line.at(0) + " h" + line.at(1);
            nrWrongDateTime++;
            continue;
        }

        // check hour
        bool isNumber = false;
        hour = line.at(1).toInt(&isNumber);
        if (! isNumber || (hour < 0) || (hour > 23))
        {
            log += "\nWrong dateTime: " + line.at(0) + " h" + line.at(1);
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
            log += "\nDuplicate dateTime: " + dateTimeStr;
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
            log += "\nError in execute query: " + qry.lastError().text() +"\n";
            log += "Maybe there are missing or wrong data values.";
            return false;
        }
    }

    if (nrWrongDateTime > 0 || nrWrongDateTime > 0 || nrWrongData > 0)
    {
        log += "\nWrong date/time: " + QString::number(nrWrongDateTime);
        log += "\nMissing data: " + QString::number(nrMissingData);
        log += "\nWrong values: " + QString::number(nrWrongData);
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::writeDailyDataList(const QString &pointCode, const QList<QString> &listEntries, QString& log)
{
    if (! existIdPoint(pointCode))
    {
        log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }

    // create table
    bool deletePreviousData = false;
    QString tableName = pointCode + "_D";
    if (! createTable(tableName, deletePreviousData))
    {
        log += "\nError in create table: " + tableName + _db.lastError().text();
        return false;
    }

    QString queryStr = QString(("INSERT OR REPLACE INTO `%1`"
                                " VALUES ")).arg(tableName);

    queryStr = queryStr + listEntries.join(",");

    QSqlQuery qry(_db);
    qry.prepare(queryStr);
    if (! qry.exec())
    {
        log += "\nError in execute query: " + qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::writeHourlyDataList(const QString &pointCode, const QList<QString> &listEntries, QString &log)
{
    if (! existIdPoint(pointCode))
    {
        log += "\nID " + pointCode + " is not present in the point properties table.";
        return false;
    }

    // create table
    bool deletePreviousData = false;
    QString tableName = pointCode + "_H";
    if (! createTable(tableName, deletePreviousData))
    {
        log += "\nError in create table: " + tableName + _db.lastError().text();
        return false;
    }

    QString queryStr = QString("INSERT OR REPLACE INTO `%1` VALUES ").arg(tableName);

    queryStr = queryStr + listEntries.join(",");

    QSqlQuery qry(_db);
    qry.prepare(queryStr);
    if (! qry.exec())
    {
        log += "\nError in execute query: " + qry.lastError().text();
        return false;
    }

    return true;
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
    _errorStr = "";
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
        _errorStr = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::deleteAllPointsFromIdList(const QList<QString>& pointList)
{
    QSqlQuery qry(_db);

    _errorStr = "";
    for (int i = 0; i < pointList.size(); i++)
    {
        QString id_point = pointList[i];
        qry.prepare( "DELETE FROM point_properties WHERE id_point = :id_point" );
        qry.bindValue(":id_point", id_point);
        if( !qry.exec() )
        {
            _errorStr += id_point + " " + qry.lastError().text();
            return false;
        }

        // remove also tables
        QString table = id_point + "_H";
        QString queryStr = "DROP TABLE IF EXISTS '" + table +"'";
        if( !qry.exec(queryStr))
        {
            _errorStr += "\n" + qry.lastError().text();
        }

        table = id_point + "_D";
        queryStr = "DROP TABLE IF EXISTS '" + table +"'";
        if( !qry.exec(queryStr))
        {
            _errorStr += "\n" + qry.lastError().text();
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
    QString statement = "SELECT id_point from point_properties WHERE UPPER(dataset) IN  (" + datasetList.toUpper() + ")";

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


bool Crit3DMeteoPointsDbHandler::setActiveStateIfCondition(bool activeState, const QString &condition)
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

    if(! qry.exec(statement))
    {
        _errorStr += "\nError in SET is_active: " + condition + " " + qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoPointsDbHandler::setOrogCode(QString id, int orogCode)
{
    QSqlQuery qry(_db);

    qry.prepare( "UPDATE point_properties SET orog_code = :orogCode WHERE id_point = :id" );
    qry.bindValue(":orogCode", orogCode);
    qry.bindValue(":id", id);

    if(! qry.exec())
    {
        _errorStr += "\nError in SET orog_code, ID: " + id + " " + qry.lastError().text();
        return false;
    }

    return true;
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


bool Crit3DMeteoPointsDbHandler::setJointStations(const QString& idPoint, const QList<QString> &stationsList)
{
    QSqlQuery qry(_db);

    QString queryStr;
    queryStr = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(id_point TEXT, joint_station TEXT, PRIMARY KEY(id_point, joint_station))").arg("joint_stations");
    qry.prepare(queryStr);
    if(! qry.exec() )
    {
        _errorStr += idPoint + " " + qry.lastError().text();
        return false;
    }

    qry.prepare( "DELETE FROM joint_stations WHERE id_point = :id_point" );
    qry.bindValue(":id_point", idPoint);
    if(! qry.exec() )
    {
        _errorStr += idPoint + " " + qry.lastError().text();
        return false;
    }

    _errorStr.clear();
    for (int i = 0; i < stationsList.size(); i++)
    {
        qry.prepare( "INSERT INTO joint_stations (id_point, joint_station) VALUES (:id_point, :joint_station)" );

        qry.bindValue(":id_point", idPoint);
        qry.bindValue(":joint_station", stationsList[i]);
        if(! qry.exec() )
        {
            _errorStr += idPoint + "," + stationsList[i] + " " + qry.lastError().text();
        }
    }

    return _errorStr.isEmpty();
}


bool Crit3DMeteoPointsDbHandler::getPointListWithCriteria(QList<QString> &selectedPointsList, const QString &condition)
{
    QSqlQuery qry(_db);
    QString queryString;

    queryString = QString("SELECT id_point FROM point_properties WHERE %1 ").arg(condition);

    if(! qry.exec(queryString))
    {
        _errorStr += "\nError in getting the ids of the points which satisfy the: " + condition + " " + qry.lastError().text();
        return false;
    }

    while (qry.next()) {
        QString idPoint = qry.value("id_point").toString();;
        selectedPointsList.append(idPoint);
    }

    return true;
}
