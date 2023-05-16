#include "commonConstants.h"
#include "dbArkimet.h"

#include <QtSql>


DbArkimet::DbArkimet(QString dbName) : Crit3DMeteoPointsDbHandler(dbName)
{
    queryString = "";
}


QList<VariablesList> DbArkimet::getVariableProperties(QList<int> id)
{
    QList<VariablesList> variableList;
    if (! (id.size() > 0))
        return variableList;

    QString idlist = QString("(%1").arg(id[0]);

    for (int i = 1; i < id.size(); i++)
    {
        idlist = idlist % QString(",%1").arg(id[i]);
    }
    idlist = idlist % QString(")");

    QString statement = QString("SELECT * FROM variable_properties WHERE id_arkimet IN %1").arg(idlist);

    QSqlQuery qry(statement, _db);

    if( !qry.exec() )
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

    if( !qry.exec() )
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
        error = "Error in execute query:\n" + myQuery + "\n" + qry.lastError().text();
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

    if( !qry.exec() )
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

    if( !qry.exec() )
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



void DbArkimet::initStationsDailyTables(QDate startDate, QDate endDate, QList<QString> stations, QList<QString> idVar)
{

    for (short i=0; i<idVar.size(); i++)
    {
        idVar[i].insert(0,"'");
        idVar[i].insert(idVar[i].size(),"'");
    }

    for (int i = 0; i < stations.size(); i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_D` "
                                    "(date_time TEXT, id_variable INTEGER, value REAL, PRIMARY KEY(date_time,id_variable))").arg(stations[i]);

        QSqlQuery qry(statement, _db);
        qry.exec();

        statement = QString("DELETE FROM `%1_D` WHERE date_time >= DATE('%2') AND date_time < DATE('%3', '+1 day') AND id_variable IN (%4)")
                        .arg(stations[i]).arg(startDate.toString("yyyy-MM-dd")).arg(endDate.toString("yyyy-MM-dd")).arg(idVar.join(","));

        qry = QSqlQuery(statement, _db);
        qry.exec();
    }

}


void DbArkimet::initStationsHourlyTables(QDate startDate, QDate endDate, QList<QString> stations, QList<QString> idVar)
{
    // start from 01:00
    QDateTime startTime(startDate, QTime(1,0,0), Qt::UTC);

    QDateTime endTime(endDate, QTime(0,0,0), Qt::UTC);
    endTime = endTime.addSecs(3600 * 24);

    for (short i=0; i<idVar.size(); i++)
    {
        idVar[i].insert(0,"'");
        idVar[i].insert(idVar[i].size(),"'");
    }

    for (int i = 0; i < stations.size(); i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_H` (date_time TEXT, id_variable INTEGER, value REAL, PRIMARY KEY(date_time,id_variable))").arg(stations[i]);

        QSqlQuery qry(statement, _db);
        qry.exec();

        statement = QString("DELETE FROM `%1_H` WHERE date_time >= DATETIME('%2') AND date_time <= DATETIME('%3') AND id_variable IN (%4)")
                        .arg(stations[i]).arg(startTime.toString("yyyy-MM-dd hh:mm:ss")).arg(endTime.toString("yyyy-MM-dd hh:mm:ss")).arg(idVar.join(","));

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


void DbArkimet::createTmpTableDaily()
{
    this->deleteTmpTableDaily();

    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE TmpDailyData (date TEXT, id_point TEXT, id_variable INTEGER, value REAL)");
    if( !qry.exec() )
    {
        qDebug() << qry.lastError();
    }
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

    qry.prepare( "DROP TABLE TmpDailyData" );

    if( !qry.exec() )
    {
        //qDebug() << "DROP TABLE TmpDailyData" << qry.lastError();
    }
}


void DbArkimet::appendQueryHourly(QString dateTime, QString idPoint, QString idVar, QString value, bool isFirstData)
{
    if (isFirstData)
        queryString = "INSERT INTO TmpHourlyData VALUES ";
    else
        queryString += ",";

    queryString += "('" + dateTime + "'"
            + ",'" + idPoint + "'"
            + "," + idVar
            + "," + value
            + ")";
}


void DbArkimet::appendQueryDaily(QString date, QString idPoint, QString idVar, QString value, bool isFirstData)
{
    if (isFirstData)
        queryString = "INSERT INTO TmpDailyData VALUES ";
    else
        queryString += ",";

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
    QSqlQuery qry = _db.exec(statement);

    // create data stations list
    QList<QString> stations;
    while (qry.next())
        stations.append(qry.value(0).toString());

    // insert data
    foreach (QString id_point, stations)
    {
        statement = QString("INSERT INTO `%1_D` ").arg(id_point);
        statement += QString("SELECT date, id_variable, value FROM TmpDailyData ");
        statement += QString("WHERE id_point = %1").arg(id_point);

        _db.exec(statement);
        if (_db.lastError().type() != QSqlError::NoError)
        {
            qDebug() << _db.lastError();
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

