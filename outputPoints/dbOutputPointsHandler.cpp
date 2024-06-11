#include "dbOutputPointsHandler.h"
#include "commonConstants.h"
#include "utilities.h"
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


bool Crit3DOutputPointsDbHandler::createTable(const QString& tableName, QString& errorStr)
{
    QString queryString = "CREATE TABLE IF NOT EXISTS '" + tableName + "'";
    queryString += " (DATE_TIME TEXT, PRIMARY KEY(DATE_TIME))";

    QSqlQuery myQuery = _db.exec(queryString);

    if (myQuery.lastError().isValid())
    {
        errorStr = "Error in creating table: " + tableName + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::addColumn(const QString &tableName, meteoVariable myVar, QString& errorStr)
{
    // column name
    QString newField = QString::fromStdString(getMeteoVarName(myVar));
    if (newField == "")
    {
        errorStr = "Missing variable name.";
        return false;
    }

    // column exists already
    QList<QString> fieldList = getFields(&_db, tableName);
    if (fieldList.contains(newField))
    {
        return true;
    }

    // add column
    QString queryString = "ALTER TABLE '" + tableName + "'";
    queryString += " ADD " + newField + " REAL";

    QSqlQuery myQuery = _db.exec(queryString);
    if (myQuery.lastError().isValid())
    {
        errorStr = "Error in add column: " + newField + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


// depth [cm]
bool Crit3DOutputPointsDbHandler::addCriteria3DColumn(const QString &tableName, criteria3DVariable myVar, int depth, QString& errorStr)
{
    // variable name
    QString variableString = QString::fromStdString(getCriteria3DVarName(myVar));
    if (variableString == "")
    {
        errorStr = "Missing variable name.";
        return false;
    }

    // column name
    QString newField = variableString + "_" + QString::number(depth);

    // column exists already
    QList<QString> fieldList = getFields(&_db, tableName);
    if ( fieldList.contains(newField) )
    {
        return true;
    }

    // add column
    QString queryString = "ALTER TABLE '" + tableName + "'";
    queryString += " ADD " + newField + " REAL";

    QSqlQuery myQuery = _db.exec(queryString);
    if (myQuery.lastError().isValid())
    {
        errorStr = "Error in add column: " + newField + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::saveHourlyMeteoData(const QString &tableName, const QDateTime &myTime,
                                                 const std::vector<meteoVariable> &varList,
                                                 const std::vector<float> &values,
                                                 QString &errorStr)
{
    if (varList.size() != values.size())
    {
        errorStr = "Error saving values: number of variables is different from values";
        return false;
    }

    QSqlQuery qry(_db);
    QString timeStr = myTime.toString("yyyy-MM-dd HH:mm:ss");
    QString queryString = QString("DELETE FROM '%1' WHERE DATE_TIME ='%2'").arg(tableName, timeStr);

    if (! qry.exec(queryString))
    {
        errorStr = QString("Error deleting values in table:%1 Time:%2\n%3")
                            .arg(tableName, timeStr, qry.lastError().text());
        return false;
    }

    // field list
    QString fieldList = "'DATE_TIME'";
    for (unsigned int i = 0; i < varList.size(); i++)
    {
        QString newField = QString::fromStdString(getMeteoVarName(varList[i]));
        if (newField != "")
        {
            fieldList += ",'" + newField + "'";
        }
        else
        {
            errorStr = "Error saving values: missing variable name.";
            return false;
        }
    }

    // values list
    QString valuesList = "'" + timeStr + "'";
    for (unsigned int i = 0; i < varList.size(); i++)
    {
        valuesList += "," + QString::number(values[i], 'f', 2);
    }

    queryString = QString("INSERT INTO '%1' (%2) VALUES (%3)").arg(tableName, fieldList, valuesList);
    if (! qry.exec(queryString))
    {
        errorStr = QString("Error saving values in table:%1 Time:%2\n%3")
                              .arg(tableName, timeStr, qry.lastError().text());
        return false;
    }

    return true;
}


// layerDepth  [m]
bool Crit3DOutputPointsDbHandler::saveHourlyCriteria3D_Data(const QString &tableName, const QDateTime& myTime,
                                                        const std::vector<criteria3DVariable>& varList,
                                                        const std::vector<float>& values,
                                                        const std::vector <double>& layerDepth,
                                                        QString &errorStr)
{
    int nrSoilLayers = int(layerDepth.size()) - 1;
    if (nrSoilLayers <= 0)
    {
        errorStr = "Error saving values: missing soil layers.";
        return false;
    }

    int nrValues = int(varList.size()) * nrSoilLayers;
    if (nrValues != values.size())
    {
        errorStr = "Error saving values: number of values is not as expected.";
        return false;
    }

    QString timeStr = myTime.toString("yyyy-MM-dd HH:mm:ss");

    // field list
    QString setList = "";
    for (unsigned int i = 0; i < varList.size(); i++)
    {
        QString variableString = QString::fromStdString(getCriteria3DVarName(varList[i]));
        if (variableString == "")
        {
            errorStr = "Missing variable name.";
            return false;
        }

        for (int layer = 1; layer <= nrSoilLayers; layer++)
        {
            int depth = round(layerDepth[layer] * 100);         // [cm]

            QString fieldName = variableString + "_" + QString::number(depth);
            setList += "'" + fieldName + "'=";

            int index = i * nrSoilLayers + layer - 1;
            QString valueStr = QString::number(values[index], 'f', 3);
            setList += valueStr;

            if (index < (nrValues - 1))
                setList += ",";
        }
    }

    QSqlQuery qry(_db);
    QString queryString = QString("UPDATE '%1' SET %2 WHERE DATE_TIME ='%3'").arg(tableName, setList, timeStr);
    if (! qry.exec(queryString))
    {
        errorStr = QString("Error in query: " + queryString + "\n" + qry.lastError().text());
        return false;
    }

    return true;
}

