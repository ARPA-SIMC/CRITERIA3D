#include "dbOutputPointsHandler.h"
#include "commonConstants.h"
#include "utilities.h"

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


bool Crit3DOutputPointsDbHandler::isOpen()
{
    return _db.isOpen();
}

QString Crit3DOutputPointsDbHandler::getDbName()
{
    return _db.databaseName();
}

QString Crit3DOutputPointsDbHandler::getErrorString()
{
    return errorString;
}


bool Crit3DOutputPointsDbHandler::createTable(QString tableName)
{
    QString queryString = "CREATE TABLE IF NOT EXISTS '" + tableName + "'";
    queryString += " (DATE_TIME TEXT, PRIMARY KEY(DATE_TIME))";

    QSqlQuery myQuery = _db.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        errorString = "Error in creating table: " + tableName + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::addColumn(QString tableName, meteoVariable myVar)
{
    // column name
    QString newField = QString::fromStdString(getMeteoVarName(myVar));
    if (newField == "")
    {
        errorString = "Missing variable name.";
        return false;
    }

    // column exists already
    QList<QString> fieldList = getFields(&_db, tableName);
    if (fieldList.contains(newField))
        return true;

    // add column
    QString queryString = "ALTER TABLE '" + tableName + "'";
    queryString += " ADD " + newField + " REAL";

    QSqlQuery myQuery = _db.exec(queryString);
    if (myQuery.lastError().isValid())
    {
        errorString = "Error in add column: " + newField + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::saveHourlyData(QString tableName, QDateTime myTime,
                                                  std::vector<meteoVariable> varList, std::vector<float> values)
{
    QString timeStr = myTime.toString("yyyy-MM-dd HH:mm:ss");
    // todo delete row

    // todo elenco field
    QString fieldList = "DATE_TIME";
    // todo elenco values
    QString valuesList = "";

    QString queryString = "INSERT INTO '" + tableName + "'"
                          + " (" + fieldList + ")"
                          + " VALUES (" + valuesList +")";

    QSqlQuery myQuery = _db.exec(queryString);
    if (myQuery.lastError().isValid())
    {
        errorString = "Error saving values in table: " + tableName + "\n"
                      + timeStr + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


