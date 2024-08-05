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

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname_);
    _db.open();
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
    if (depth != NODATA)
    {
        variableString += "_" + QString::number(depth);
    }

    // column exists already
    QList<QString> fieldList = getFields(&_db, tableName);
    if ( fieldList.contains(variableString) )
    {
        return true;
    }

    // add column
    QString queryString = "ALTER TABLE '" + tableName + "'";
    queryString += " ADD " + variableString + " REAL";

    QSqlQuery myQuery = _db.exec(queryString);
    if (myQuery.lastError().isValid())
    {
        errorStr = "Error in add column: " + variableString + "\n" + myQuery.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::saveHourlyMeteoData(const QString &tableName, const QDateTime &myTime,
                                                 const std::vector<meteoVariable> &varList,
                                                 const std::vector<float> &valuesList,
                                                 QString &errorStr)
{
    if (varList.size() != valuesList.size())
    {
        errorStr = "Error saving values: number of variables is different from values";
        return false;
    }

    QString timeStr = myTime.toString("yyyy-MM-dd HH:mm:ss");
    QString queryString = QString("SELECT * FROM '%1' WHERE DATE_TIME='%2'").arg(tableName, timeStr);
    QSqlQuery query(_db);

    if (! query.exec(queryString))
    {
        errorStr = QString("Error in reading table: %1 \nTime: %2 \n%3")
                            .arg(tableName, timeStr, query.lastError().text());
        return false;
    }

    query.last();
    int querySize = query.at() + 1;     // SQLITE doesn't support SIZE
    if (querySize > 0)
    {
        return saveHourlyMeteoData_update(tableName, timeStr, varList, valuesList, errorStr);
    }
    else
    {
        return saveHourlyMeteoData_insert(tableName, timeStr, varList, valuesList, errorStr);
    }

}


bool Crit3DOutputPointsDbHandler::saveHourlyMeteoData_update(const QString &tableName, const QString timeStr,
                                                             const std::vector<meteoVariable> &varList,
                                                             const std::vector<float> &values,
                                                             QString &errorStr)
{
    QList<QString> assignList;
    for (unsigned int i = 0; i < varList.size(); i++)
    {
        QString fieldStr = QString::fromStdString(getMeteoVarName(varList[i]));
        if (fieldStr.isEmpty())
        {
            errorStr = QString("Error saving values in table:%1 \nMissing variable name").arg(tableName);
            return false;
        }

        QString valueStr = QString::number(values[i], 'f', 2);

        assignList.push_back("'" + fieldStr + "'=" + valueStr);
    }

    QSqlQuery qry(_db);
    QString queryString = QString("UPDATE '%1' SET %2 WHERE DATE_TIME = DATETIME('%3')").arg(tableName, assignList.join(','), timeStr);
    if (! qry.exec(queryString))
    {
        errorStr = QString("Error saving values in table:%1 Time:%2\n%3")
                       .arg(tableName, timeStr, qry.lastError().text());
        return false;
    }

    return true;
}


bool Crit3DOutputPointsDbHandler::saveHourlyMeteoData_insert(const QString &tableName, const QString timeStr,
                                                             const std::vector<meteoVariable> &varList,
                                                             const std::vector<float> &values,
                                                             QString &errorStr)
{
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
            errorStr = QString("Error saving values in table:%1 \nMissing variable name").arg(tableName);
            return false;
        }
    }

    // values list
    QString valuesList = "'" + timeStr + "'";
    for (unsigned int i = 0; i < values.size(); i++)
    {
        valuesList += "," + QString::number(values[i], 'f', 2);
    }

    QSqlQuery qry(_db);
    QString queryString = QString("INSERT INTO '%1' (%2) VALUES (%3)").arg(tableName, fieldList, valuesList);
    if (! qry.exec(queryString))
    {
        errorStr = QString("Error saving values in table:%1 Time:%2\n%3")
                       .arg(tableName, timeStr, qry.lastError().text());
        return false;
    }

    return true;
}



// variableDepth  [cm]
bool Crit3DOutputPointsDbHandler::saveHourlyCriteria3D_Data(const QString &tableName, const QDateTime& myTime,
                                                            const std::vector<float>& values,
                                                            const std::vector<int>& waterContentDepth,
                                                            const std::vector<int>& waterPotentialDepth,
                                                            const std::vector<int>& degreeOfSaturationDepth,
                                                            const std::vector<int>& factorOfSafetyDepth,
                                                            QString &errorStr)
{
    int nrValues = int(waterContentDepth.size() + waterPotentialDepth.size()
                       + degreeOfSaturationDepth.size() + factorOfSafetyDepth.size());

    if (nrValues != int(values.size()))
    {
        errorStr = "Error saving values: number of values is not as expected.";
        return false;
    }

    QString timeStr = myTime.toString("yyyy-MM-dd HH:mm:ss");

    QList<QString> valueList;
    int firstIndex = 0;
    appendCriteria3DOutputValue(volumetricWaterContent, waterContentDepth, values, firstIndex, valueList);
    appendCriteria3DOutputValue(waterMatricPotential, waterPotentialDepth, values, firstIndex, valueList);
    appendCriteria3DOutputValue(degreeOfSaturation, degreeOfSaturationDepth, values, firstIndex, valueList);
    appendCriteria3DOutputValue(factorOfSafety, factorOfSafetyDepth, values, firstIndex, valueList);


    QSqlQuery qry(_db);
    QString queryString = QString("UPDATE '%1' SET %2 WHERE DATE_TIME = DATETIME('%3')").arg(tableName, valueList.join(','), timeStr);
    if (! qry.exec(queryString))
    {
        errorStr = QString("Error in query: " + queryString + "\n" + qry.lastError().text());
        return false;
    }

    return true;
}


void Crit3DOutputPointsDbHandler::appendCriteria3DOutputValue(criteria3DVariable myVar, const std::vector<int> &depthList,
                                                              const std::vector<float>& values, int &firstIndex,
                                                              QList<QString> &outputList)
{
    QString variableString = QString::fromStdString(getCriteria3DVarName(myVar));

    for (int l = 0; l < depthList.size(); l++)
    {
        float depth_cm = depthList[l];
        QString fieldName = variableString + "_" + QString::number(depth_cm);

        int index = firstIndex + l;
        QString valueStr = QString::number(values[index], 'f', 3);

        QString assignStr = "'" + fieldName + "'=" + valueStr;

        outputList.push_back(assignStr);
    }

    firstIndex += int(depthList.size());
}

