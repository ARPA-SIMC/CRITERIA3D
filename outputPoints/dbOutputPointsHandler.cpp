#include "dbOutputPointsHandler.h"
#include "commonConstants.h"
#include "meteo.h"

#include <QtSql>


// Only SQLite database
Crit3DOutputPointsDbHandler::Crit3DOutputPointsDbHandler(QString dbname_)
{
    if(_db.isOpen())
    {
        _db.close();
    }
    errorString = "";

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);
    if (! _db.open())
    {
        errorString = _db.lastError().text();
    }
}


Crit3DOutputPointsDbHandler::~Crit3DOutputPointsDbHandler()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        QString connection = _db.connectionName();
        _db.close();
        _db = QSqlDatabase();
        QSqlDatabase::removeDatabase(connection);
    }
}


QString Crit3DOutputPointsDbHandler::getDbName()
{
    return _db.databaseName();
}

QString Crit3DOutputPointsDbHandler::getErrorString()
{
    return errorString;
}
