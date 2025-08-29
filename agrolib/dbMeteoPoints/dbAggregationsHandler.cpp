#include "dbAggregationsHandler.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "meteo.h"

#include <QtSql>


Crit3DAggregationsDbHandler::Crit3DAggregationsDbHandler(QString dbname)
{
    _error = "";
    _mapIdMeteoVar.clear();

    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname);

    if (!_db.open())
       _error = _db.lastError().text();
}

Crit3DAggregationsDbHandler::~Crit3DAggregationsDbHandler()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        QString connection = _db.connectionName();
        _db.close();
        _db = QSqlDatabase();
        _db.removeDatabase(connection);
    }
}


bool Crit3DAggregationsDbHandler::saveAggregationData(const std::vector<int> &idZoneVector, const QString &aggrType,
                                                        const QString &periodType, const QDate &startDate,
                                                        const QDate &endDate, meteoVariable variable,
                                                        const std::vector<std::vector<float> > &aggregatedValues)
{
    initAggregatedTables(idZoneVector, aggrType, periodType, startDate, endDate, variable);
    createTmpAggrTable();

    int idVariable = getIdfromMeteoVar(variable);
    long nrDays = long(startDate.daysTo(endDate)) + 1;
    QSqlQuery qry(_db);

    for (unsigned int i = 0; i < idZoneVector.size(); i++)
    {
        QString queryString = QString("REPLACE INTO `%1_%2_%3` VALUES").arg(QString::number(idZoneVector[i]), aggrType, periodType);

        if (periodType == "H")
        {
            // hourly data
            for (int day = 0; day < nrDays; day++)
            {
                for (int hour = 0; hour < 24; hour++)
                {
                    QString valueString = "NULL";
                    if (! isEqual(aggregatedValues[day*24 + hour][i], NODATA))
                    {
                        valueString = QString::number(double(aggregatedValues[day*24 + hour][i]), 'f', 1);
                    }

                    // the data refers to the past hour
                    int currentDay = day;
                    int currentHour = hour + 1;
                    if (currentHour == 24)
                    {
                        currentHour = 0;
                        currentDay++;
                    }
                    QDateTime currentDateTime;
                    currentDateTime.setTimeZone(QTimeZone::utc());
                    currentDateTime.setDate(startDate.addDays(currentDay));
                    currentDateTime.setTime(QTime(currentHour, 0, 0, 0));

                    QString dateString = currentDateTime.toString("yyyy-MM-dd hh:00:00");
                    QString varString = QString::number(idVariable);

                    queryString += "('" + dateString + "'," + varString + "," + valueString + "),";
                }
            }
        }
        else
        {
            // daily data
            for (unsigned int day = 0; day < unsigned(nrDays); day++)
            {
                QString valueString = "NULL";
                if (! isEqual(aggregatedValues[day][i], NODATA))
                {
                    valueString = QString::number(double(aggregatedValues[day][i]), 'f', 1);
                }

                QString dateString = startDate.addDays(day).toString("yyyy-MM-dd");
                QString varString = QString::number(idVariable);

                queryString += "('" + dateString + "'," + varString + "," + valueString + "),";
            }
        }

        // removes the last comma
        queryString = queryString.left(queryString.length() - 1);

        if (! qry.exec(queryString))
        {
            _error = qry.lastError().text();
            return false;
        }
    }

    deleteTmpAggrTable();
    return true;
}


bool Crit3DAggregationsDbHandler::writeAggregationZonesTable(QString name, QString filename, QString field)
{
    QSqlQuery qry(_db);

    qry.prepare( "INSERT INTO aggregation_zones (name, filename, shape_field)"
                                      " VALUES (:name, :filename, :shape_field)" );

    qry.bindValue(":name", name);
    qry.bindValue(":filename", filename);
    qry.bindValue(":shape_field", field);

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DAggregationsDbHandler::writeRasterName(QString rasterName)
{
    QSqlQuery qry(_db);

    qry.prepare( "INSERT INTO zones (name) VALUES (:name)" );
    qry.bindValue(":name", rasterName);

    if(! qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DAggregationsDbHandler::getRasterName(QString* rasterName)
{
    QSqlQuery qry(_db);

    qry.prepare( "SELECT * FROM zones" );

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
    {
        if (qry.next())
        {
            getValue(qry.value("name"), rasterName);
            return true;
        }
        else
        {
            _error = "name not found";
            return false;
        }
    }
}


bool Crit3DAggregationsDbHandler::getAggregationZonesReference(QString name, QString* filename, QString* field)
{

    QSqlQuery qry(_db);

    qry.prepare( "SELECT * FROM aggregation_zones WHERE name = :name");
    qry.bindValue(":name", name);

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
    {
        if (qry.next())
        {
            getValue(qry.value("filename"), filename);
            getValue(qry.value("shape_field"), field);
            return true;
        }
        else
        {
            _error = "name not found";
            return false;
        }
    }
}


void Crit3DAggregationsDbHandler::initAggregatedTables(const std::vector<int> &idZoneVector, const QString &aggrType, const QString &periodType, const QDate &startDate, const QDate &endDate, meteoVariable variable)
{
    int idVariable = getIdfromMeteoVar(variable);
    for (int i = 0; i < (int)idZoneVector.size(); i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_%2_%3` "
                                    "(date_time TEXT, id_variable INTEGER, value REAL, PRIMARY KEY(date_time,id_variable))")
                                    .arg(idZoneVector[i]).arg(aggrType, periodType);

        QSqlQuery qry(statement, _db);
        if(! qry.exec() )
        {
            _error = qry.lastError().text();
        }

        statement = QString("DELETE FROM `%1_%2_%3` WHERE date_time >= DATE('%4') "
                            "AND date_time < DATE('%5', '+1 day') AND id_variable = %6")
                            .arg(idZoneVector[i]).arg(aggrType, periodType, startDate.toString("yyyy-MM-dd"), endDate.toString("yyyy-MM-dd")).arg(idVariable);

        qry = QSqlQuery(statement, _db);
        if(! qry.exec() )
        {
            _error = qry.lastError().text();
        }
    }
}


bool Crit3DAggregationsDbHandler::existIdPoint(const QString& idPoint)
{
    QSqlQuery qry(_db);
    QString queryStr = "SELECT EXISTS(SELECT 1 FROM point_properties WHERE id_point='" + idPoint + "')";
    qry.prepare(queryStr);

    if (! qry.exec()) return false;
    qry.last();
    return (qry.value(0).toInt() > 0);
}


bool Crit3DAggregationsDbHandler::writeAggregationPointProperties(const QString &aggrType, const std::vector<int> &idZoneVector,
                                                                  const std::vector<double> &lonVector, const std::vector<double> &latVector)
{
    if (! _db.tables().contains(QLatin1String("point_properties")) )
    {
        return false;
    }

    QSqlQuery qry(_db);
    for (int i = 0; i < (int)idZoneVector.size(); i++)
    {
        QString id = QString::number(idZoneVector[i]) + "_" + aggrType;
        QString name = id;

        if (! existIdPoint(id))
        {
            qry.prepare( "INSERT INTO point_properties (id_point, name, latitude, longitude, altitude, is_active)"
                                              " VALUES (:id_point, :name, :latitude, :longitude, :altitude, :is_active)" );

            qry.bindValue(":id_point", id);
            qry.bindValue(":name", name);
            qry.bindValue(":latitude", latVector[i]);
            qry.bindValue(":longitude", lonVector[i]);
            qry.bindValue(":altitude", 0);
            qry.bindValue(":is_active", 1);

            if(! qry.exec() )
            {
                _error = qry.lastError().text();
                return false;
            }
        }
    }

    return true;
}


void Crit3DAggregationsDbHandler::createTmpAggrTable()
{
    this->deleteTmpAggrTable();

    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE TmpAggregationData (date_time TEXT, zone TEXT, id_variable INTEGER, value REAL)");
    if(! qry.exec() )
    {
        _error = qry.lastError().text();
    }
}


void Crit3DAggregationsDbHandler::deleteTmpAggrTable()
{
    QSqlQuery qry(_db);

    qry.prepare("DROP TABLE TmpAggregationData");
    qry.exec();
}


std::vector<float> Crit3DAggregationsDbHandler::getAggrData(QString aggrType, QString periodType, int zone, QDate startDate, QDate endDate, meteoVariable variable)
{

    int idVariable = getIdfromMeteoVar(variable);
    unsigned int nrDays = unsigned(startDate.daysTo(endDate) + 1);
    std::vector<float> values(nrDays, NODATA);
    QDate date;
    float value;

    QString statement = QString( "SELECT * FROM `%1_%2_%3` WHERE date_time >= DATE('%4') AND date_time < DATE('%5', '+1 day') AND id_variable = '%6'")
                                .arg(zone).arg(aggrType).arg(periodType).arg(startDate.toString("yyyy-MM-dd")).arg(endDate.toString("yyyy-MM-dd")).arg(idVariable);
    QSqlQuery qry(statement, _db);

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return values;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("date_time"), &date);
            getValue(qry.value("value"), &value);
            values[startDate.daysTo(date)] = value;
        }
    }
    return values;
}


bool Crit3DAggregationsDbHandler::loadVariableProperties()
{
    QSqlQuery qry(_db);

    QString tableName = "variable_properties";
    int id_variable;
    QString variable;
    std::string stdVar;
    meteoVariable meteoVar;
    std::pair<std::map<int, meteoVariable>::iterator,bool> ret;

    QString statement = QString( "SELECT * FROM `%1` ").arg(tableName);
    if( !qry.exec(statement) )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_variable"), &id_variable);
            getValue(qry.value("variable"), &variable);
            stdVar = variable.toStdString();

            meteoVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, stdVar);
            if (meteoVar == noMeteoVar)
            {
                meteoVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, stdVar);
            }

            if (meteoVar != noMeteoVar)
            {
                ret = _mapIdMeteoVar.insert(std::pair<int, meteoVariable>(id_variable,meteoVar));
                if (ret.second==false)
                {
                    _error = "element 'z' already existed";
                }
            }
        }
    }
    return true;
}


int Crit3DAggregationsDbHandler::getIdfromMeteoVar(meteoVariable meteoVar)
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


QList<QString> Crit3DAggregationsDbHandler::getAggregations()
{
    QSqlQuery qry(_db);

    qry.prepare( "SELECT * FROM aggregations");
    QString aggregation;
    QList<QString> aggregationList;

    if(! qry.exec() )
    {
        _error = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("aggregation"), &aggregation);
            aggregationList.append(aggregation);
        }
    }

    if (aggregationList.isEmpty())
    {
        _error = "aggregation table is empty.";
    }

    return aggregationList;
}


bool Crit3DAggregationsDbHandler::renameColumn(QString oldColumn, QString newColumn)
{
    QSqlQuery qry(_db);

    qry.prepare( "SELECT name FROM sqlite_master WHERE type='table'  AND name like '%_D' OR name like '%_H';");
    QString table;
    QList<QString> tablesList;

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value(0), &table);
            tablesList.append(table);
        }
    }
    if (tablesList.isEmpty())
    {
        _error = "name not found";
        return false;
    }
    foreach (QString table, tablesList)
    {
        QString statement = QString( "ALTER TABLE `%1` RENAME COLUMN %2 TO %3").arg(table).arg(oldColumn).arg(newColumn);
        if(!qry.exec(statement) )
        {
            _error = qry.lastError().text();
            return false;
        }
    }
    return true;

}

bool Crit3DAggregationsDbHandler::writeDroughtDataList(QList<QString> listEntries, QString* log)
{
    // create table
    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE IF NOT EXISTS `drought` "
                "(year INTEGER, month INTEGER, id_point TEXT, yearRefStart INTEGER, yearRefEnd INTEGER, drought_index TEXT, timescale INTEGER, value REAL, "
                "PRIMARY KEY(year, month, id_point, yearRefStart, yearRefEnd, drought_index, timescale));");

    if( !qry.exec() )
    {
        *log += "\nError in execute query: " + qry.lastError().text();
        return false;
    }

    QString queryStr = QString(("INSERT OR REPLACE INTO `drought` (year, month, id_point, yearRefStart, yearRefEnd, drought_index, timescale, value) VALUES "));
    queryStr = queryStr + listEntries.join(",");

    // exec query
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
