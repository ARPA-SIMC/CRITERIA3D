#include "waterTableDb.h"
#include <QtSql>


WaterTableDb::WaterTableDb(QString dbName, QString &error)
{
    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbName);

    if (!_db.open())
    {
        error = _db.lastError().text();
    }
}


WaterTableDb::~WaterTableDb()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        _db.close();
    }
}



bool WaterTableDb::readSingleWaterTableParameters(const QString &id, waterTableParameters &waterTable, QString &error)
{
    QString queryString = "SELECT * FROM wellProperties WHERE ID_WATERTABLE='" + id + "'";

    QSqlQuery query = _db.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (! query.lastError().text().isEmpty())
        {
            error = "Error in reading wellProperties.\n" + query.lastError().text();
        }
        else
        {
            error = "Missing parameters for the id: " + id;
        }

        return false;
    }

    waterTable.id = query.value("ID_WATERTABLE").toString();
    waterTable.lat = query.value("lat").toDouble();
    waterTable.lon = query.value("lon").toDouble();
    waterTable.alpha = query.value("alpha").toDouble();
    waterTable.h0  = query.value("h0").toDouble();
    waterTable.avgDailyCWB = query.value("avgDailyCWB").toDouble();
    waterTable.nrDaysPeriod = query.value("nrDays").toInt();

    return true;
}
