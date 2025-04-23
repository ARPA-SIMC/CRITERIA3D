#include "commonConstants.h"
#include "waterTableDb.h"
#include <QtSql>


Crit1DWaterTableParameters::Crit1DWaterTableParameters()
{
    id = "";
    lat = NODATA;
    lon = NODATA;

    nrDaysPeriod = NODATA;
    alpha = NODATA;
    h0 = NODATA;
    avgDailyCWB = NODATA;

    isLoaded = false;
}


WaterTableDb::WaterTableDb(QString dbName, QString &errorString)
{
    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    if (! QFile::exists(dbName))
    {
        errorString = "waterTable DB doesn't exist:\n" + dbName;
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbName);

    if (!_db.open())
    {
        errorString = _db.lastError().text();
    }
}


WaterTableDb::~WaterTableDb()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        _db.close();
    }
}



bool WaterTableDb::readSingleWaterTableParameters(const QString &id, Crit1DWaterTableParameters &wtParameters, QString &error)
{
    QString queryString = "SELECT * FROM wellProperties WHERE ID_WATERTABLE='" + id + "'";

    QSqlQuery query = _db.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (! query.lastError().text().isEmpty())
        {
            error = "Error in reading wellProperties in waterTable db.\n" + query.lastError().text();
        }
        else
        {
            error = "Missing waterTable ID in wellProperties table: " + id;
        }

        return false;
    }

    wtParameters.id = query.value("ID_WATERTABLE").toString();
    wtParameters.lat = query.value("lat").toDouble();
    wtParameters.lon = query.value("lon").toDouble();
    wtParameters.alpha = query.value("alpha").toDouble();
    wtParameters.h0  = query.value("h0").toDouble();
    wtParameters.avgDailyCWB = query.value("avgDailyCWB").toDouble();
    wtParameters.nrDaysPeriod = query.value("nrDays").toInt();

    wtParameters.isLoaded = true;

    return true;
}
