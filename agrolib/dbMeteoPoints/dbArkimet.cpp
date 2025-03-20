#include "commonConstants.h"
#include "dbArkimet.h"
#include "dbMeteoPointsHandler.h"

#include <QSqlQuery>
#include <QSqlError>
#include <QFile>
#include <QDebug>


DbArkimet::DbArkimet(QString dbName) : Crit3DMeteoPointsDbHandler(dbName)
{
    queryString = "";
}


QList<VariablesList> DbArkimet::getAllVariableProperties()
{
    QList<VariablesList> variableList;

    QString statement = QString("SELECT * FROM variable_properties");
    QSqlQuery qry(statement, _db);

    if(! qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        while (qry.next())
        {
            variableList.append(VariablesList(qry.value("id_variable").toInt(), qry.value("id_arkimet").toInt(), qry.value("variable").toString(), qry.value("frequency").toInt() ));
        }
    }

    return variableList;
}


QList<VariablesList> DbArkimet::getVariableProperties(QList<int> id)
{
    QList<VariablesList> variableList;
    if (! (id.size() > 0))
        return variableList;

    QString idlist = QString("(%1").arg(id[0]);

    for (int i = 1; i < id.size(); i++)
    {
        idlist += QString(",%1").arg(id[i]);
    }
    idlist += QString(")");

    QString statement = QString("SELECT * FROM variable_properties WHERE id_arkimet IN %1").arg(idlist);

    QSqlQuery qry(statement, _db);

    if(! qry.exec() )
        qDebug() << qry.lastError();
    else
    {
        while (qry.next())
        {
            variableList.append(VariablesList(qry.value("id_variable").toInt(), qry.value("id_arkimet").toInt(), qry.value("variable").toString(), qry.value("frequency").toInt() ));
        }
    }

    return variableList;
}


QString DbArkimet::getVarName(int id)
{
    QString varName = nullptr;
    QSqlQuery qry(_db);

    qry.prepare( "SELECT variable FROM variable_properties WHERE id_arkimet = :id" );
    qry.bindValue(":id", id);

    if(! qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        qDebug( "getVarName Selected!" );

        if (qry.next())
            varName = qry.value(0).toString();

        else
            qDebug( "Error: dataset not found" );
    }

    return varName;
}


QList<int> DbArkimet::getId(QString VarName)
{
    QList<int> idList;
    QSqlQuery qry(_db);

    QString myQuery = QString("SELECT `id_arkimet` FROM `variable_properties` WHERE `variable`='%1'").arg(VarName);

    if( !qry.exec(myQuery))
    {
        this->setErrorString("Error in execute query:\n" + myQuery + "\n" + qry.lastError().text());
    }
    else
    {
        while (qry.next())
        {
            int id = qry.value(0).toInt();
            idList << id;
        }
    }

    return idList;
}


QList<int> DbArkimet::getDailyVar()
{
    QList<int> dailyVarList;
    QSqlQuery qry(_db);

    qry.prepare( "SELECT id_arkimet FROM variable_properties WHERE frequency = 86400" );

    if(! qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        qDebug( "getDailyVar Selected!" );

        while (qry.next())
        {
            int id = qry.value(0).toInt();
            dailyVarList << id;
        }
    }

    return dailyVarList;
}


QList<int> DbArkimet::getHourlyVar()
{
    QList<int> hourlyVarList;
    QSqlQuery qry(_db);

    qry.prepare( "SELECT id_arkimet FROM variable_properties WHERE frequency < 86400" );

    if(! qry.exec() )
    {
        qDebug() << qry.lastError();
    }
    else
    {
        while (qry.next())
        {
            int id = qry.value(0).toInt();
            hourlyVarList << id;
        }
    }

    return hourlyVarList;
}


void DbArkimet::initStationsDailyTables(const QDate &startDate, const QDate &endDate,
                                        const QList<QString> &stationList, const QList<QString> &idVarList)
{
    QList<QString> varList;
    for (int i=0; i < idVarList.size(); i++)
    {
        varList.append("'" + idVarList[i] + "'");
    }

    for (int i = 0; i < stationList.size(); i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_D` "
                                    "(date_time TEXT, id_variable INTEGER, value REAL, "
                                    "PRIMARY KEY(date_time,id_variable))").arg(stationList[i]);

        QSqlQuery qry(statement, _db);
        qry.exec();

        statement = QString("DELETE FROM `%1_D` WHERE date_time >= DATE('%2') "
                            "AND date_time < DATE('%3', '+1 day') AND id_variable IN (%4)")
                        .arg(stationList[i], startDate.toString("yyyy-MM-dd"), endDate.toString("yyyy-MM-dd"), varList.join(","));

        qry = QSqlQuery(statement, _db);
        qry.exec();
    }
}


void DbArkimet::initStationsHourlyTables(const QDate &startDate, const QDate &endDate,
                                         const QList<QString> &stationList, const QList<QString> &idVarList)
{
    // start from 01:00
    QDateTime startTime(startDate, QTime(1,0,0), Qt::UTC);

    QDateTime endTime(endDate, QTime(0,0,0), Qt::UTC);
    endTime = endTime.addSecs(3600 * 24);

    QList<QString> varList;
    for (int i=0; i < idVarList.size(); i++)
    {
        varList.append("'" + idVarList[i] + "'");
    }

    for (int i = 0; i < stationList.size(); i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_H` "
                                    "(date_time TEXT, id_variable INTEGER, value REAL, "
                                    "PRIMARY KEY(date_time,id_variable))").arg(stationList[i]);

        QSqlQuery qry(statement, _db);
        qry.exec();

        statement = QString("DELETE FROM `%1_H` WHERE date_time >= DATETIME('%2') "
                            "AND date_time <= DATETIME('%3') AND id_variable IN (%4)")
                        .arg(stationList[i], startTime.toString("yyyy-MM-dd hh:mm:ss"),
                             endTime.toString("yyyy-MM-dd hh:mm:ss"), varList.join(","));

        qry = QSqlQuery(statement, _db);
        qry.exec();
    }
}


void DbArkimet::createTmpTableHourly()
{
    this->deleteTmpTableHourly();

    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE TmpHourlyData (date_time TEXT, id_point TEXT, id_variable INTEGER, value REAL)");
    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
    }
}


bool DbArkimet::createTmpTableDaily()
{
    this->deleteTmpTableDaily();

    QSqlQuery qry(_db);

    qry.prepare("CREATE TABLE TmpDailyData (date TEXT, id_point TEXT, id_variable INTEGER, value REAL)");
    if(! qry.exec())
    {
        setErrorString(qry.lastError().text());
        return false;
    }

    return true;
}


void DbArkimet::deleteTmpTableHourly()
{
    QSqlQuery qry(_db);

    qry.prepare( "DROP TABLE TmpHourlyData" );

    if( !qry.exec() )
    {
        //qDebug() << "DROP TABLE TmpHourlyData" << qry.lastError();
    }
}


void DbArkimet::deleteTmpTableDaily()
{
    QSqlQuery qry(_db);

    qry.prepare("DROP TABLE TmpDailyData");

    if(! qry.exec() )
    {
        //qDebug() << "DROP TABLE TmpDailyData" << qry.lastError();
    }
}


void DbArkimet::appendQueryHourly(const QString &dateTime, const QString &idPoint, const QString &idVar, const QString &value, bool isFirstData)
{
    if (isFirstData)
    {
        queryString = "INSERT INTO TmpHourlyData VALUES ";
    }
    else
    {
        queryString += ",";
    }
    queryString += "('" + dateTime + "'"
            + ",'" + idPoint + "'"
            + "," + idVar
            + "," + value
            + ")";
}


void DbArkimet::appendQueryDaily(const QString &date, const QString &idPoint, const QString &idVar, const QString &value, bool isFirstData)
{
    if (isFirstData)
    {
        queryString = "INSERT INTO TmpDailyData VALUES ";
    }
    else
    {
        queryString += ",";
    }
    queryString += "('" + date + "'"
            + ",'" + idPoint + "'"
            + "," + idVar
            + "," + value
            + ")";
}


bool DbArkimet::saveDailyData()
{
    if (queryString == "")
        return false;

    // insert data into tmpTable
    _db.exec(queryString);

    // query stations with data
    QString statement = QString("SELECT DISTINCT id_point FROM TmpDailyData");
    QSqlQuery qryStations = _db.exec(statement);

    // create data stations list
    QList<QString> stations;
    while (qryStations.next())
    {
        stations.append(qryStations.value(0).toString());
    }

    // insert data
    foreach (QString id_point, stations)
    {
        statement = QString("INSERT INTO `%1_D` ").arg(id_point);
        statement += QString("SELECT date, id_variable, value FROM TmpDailyData ");
        statement += QString("WHERE id_point = %1").arg(id_point);

        _db.exec(statement);
        if (_db.lastError().type() != QSqlError::NoError)
        {
            setErrorString(_db.lastError().text());
            return false;
        }
    }

    return true;
}


bool DbArkimet::saveHourlyData()
{

    if (queryString == "")
        return false;

    // insert data into tmpTable
    _db.exec(queryString);

    // query stations with data
    QString statement = QString("SELECT DISTINCT id_point FROM TmpHourlyData");
    QSqlQuery qry = _db.exec(statement);

    // create data stations list
    QList<QString> stations;
    while (qry.next())
        stations.append(qry.value(0).toString());

    // insert data (only timestamp HH::00)
    foreach (QString id_point, stations)
    {
        statement = QString("INSERT INTO `%1_H` ").arg(id_point);
        statement += QString("SELECT date_time, id_variable, value FROM TmpHourlyData ");
        statement += QString("WHERE id_point = %1 AND strftime('%M', date_time) = '00'").arg(id_point);

        _db.exec(statement);
        if (_db.lastError().type() != QSqlError::NoError)
        {
            qDebug() << _db.lastError();
            return false;
        }
    }

    return true;
}


bool DbArkimet::readVmDataDaily(const QString &vmFileName, QString &errorString)
{
    QFile myFile(vmFileName);
    if (! myFile.open(QIODevice::ReadOnly))
    {
        errorString = "Open failed: " + vmFileName + "\n " + myFile.errorString();
        return false;
    }

    QTextStream myStream (&myFile);
    if (myStream.atEnd())
    {
        errorString = "File is empty: " + vmFileName;
        myFile.close();
        return false;
    }

    // list variables
    QList<VariablesList> variableList = this->getAllVariableProperties();
    if (variableList.isEmpty())
    {
        errorString = "table 'variable_properties' is missing or empty,";
        myFile.close();
        return false;
    }

    if (! this->createTmpTableDaily())
    {
        errorString = "Error in creating daily tmp table: " + this->getErrorString();
        myFile.close();
        return false;
    }

    bool isFirstData = true;

    while(! myStream.atEnd())
    {
        QList<QString> fields = myStream.readLine().split(',');

        // warning: reference date arkimet: hour 00 of day+1
        QString dateStr = fields[0];
        QDate myDate = QDate::fromString(dateStr.left(8), "yyyyMMdd").addDays(-1);
        dateStr = myDate.toString("yyyy-MM-dd");

        QString idPoint = fields[1];
        QString flag = fields[6];

        if (idPoint != "" && flag.left(1) != "1" && flag.left(3) != "054")
        {
            QString valueStr;
            if (flag.left(1) == "2")
                valueStr = fields[4];
            else
                valueStr = fields[3];

            if (valueStr != "")
            {
                bool isNumber;
                double value = valueStr.toDouble(&isNumber);
                if (isNumber)
                {
                    int idArkimet = fields[2].toInt();

                    // conversion from average daily radiation to integral radiation
                    if (idArkimet == RAD_ID)
                    {
                        value *= DAY_SECONDS / 1000000.;
                    }

                    // variable
                    int i = 0;
                    while (i < variableList.size() && variableList[i].arkId() != idArkimet)
                        i++;

                    if (i < variableList.size())
                    {
                        int idVariable = variableList[i].id();
                        this->appendQueryDaily(dateStr, idPoint, QString::number(idVariable), QString::number(value), isFirstData);
                        isFirstData = false;
                    }
                }
            }
        }
    }

    if (! this->saveDailyData())
    {
        errorString = this->getErrorString();
        return false;
    }

    return true;
}

