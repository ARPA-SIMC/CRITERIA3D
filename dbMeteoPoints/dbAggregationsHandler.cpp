#include "dbAggregationsHandler.h"
#include "commonConstants.h"
#include "utilities.h"

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

    _db = QSqlDatabase::addDatabase("QSQLITE", "Aggregation");
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

QSqlDatabase Crit3DAggregationsDbHandler::db() const
{
    return _db;
}


QString Crit3DAggregationsDbHandler::error() const
{
    return _error;
}

std::map<int, meteoVariable> Crit3DAggregationsDbHandler::mapIdMeteoVar() const
{
return _mapIdMeteoVar;
}

bool Crit3DAggregationsDbHandler::saveAggrData(int nZones, QString aggrType, QString periodType, QDateTime startDate, QDateTime endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues)
{
    initAggregatedTables(nZones, aggrType, periodType, QDateTime(startDate), QDateTime(endDate));
    createTmpAggrTable();
    insertTmpAggr(QDateTime(startDate), QDateTime(endDate), variable, aggregatedValues, nZones);
    if (!saveTmpAggrData(aggrType, periodType, nZones))
    {
        return false;
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
    else
        return true;

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

void Crit3DAggregationsDbHandler::initAggregatedTables(int numZones, QString aggrType, QString periodType, QDateTime startDate, QDateTime endDate)
{

    for (int i = 1; i < numZones; i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_%2_%3` "
                                    "(date_time TEXT, id_variable INTEGER, value REAL, PRIMARY KEY(date_time,id_variable))").arg(i).arg(aggrType).arg(periodType);

        QSqlQuery qry(statement, _db);
        if( !qry.exec() )
        {
            _error = qry.lastError().text();
        }
        statement = QString("DELETE FROM `%1_%2_%3` WHERE date_time >= DATE('%4') AND date_time < DATE('%5', '+1 day')")
                        .arg(i).arg(aggrType).arg(periodType).arg(startDate.toString("yyyy-MM-dd hh:mm:ss")).arg(endDate.toString("yyyy-MM-dd hh:mm:ss"));

        qry = QSqlQuery(statement, _db);
        if( !qry.exec() )
        {
            _error = qry.lastError().text();
        }
    }

}

void Crit3DAggregationsDbHandler::createTmpAggrTable()
{
    this->deleteTmpAggrTable();

    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE TmpAggregationData (date_time TEXT, zone TEXT, id_variable INTEGER, value REAL)");
    if( !qry.exec() )
    {
        _error = qry.lastError().text();
    }
}

void Crit3DAggregationsDbHandler::deleteTmpAggrTable()
{
    QSqlQuery qry(_db);

    qry.prepare( "DROP TABLE TmpAggregationData" );

    qry.exec();
}


bool Crit3DAggregationsDbHandler::insertTmpAggr(QDateTime startDate, QDateTime endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues, int nZones)
{
    int idVariable = getIdfromMeteoVar(variable);
    int nrDays = int(startDate.daysTo(endDate) + 1);
    QSqlQuery qry(_db);
    qry.prepare( "INSERT INTO `TmpAggregationData` (date_time, zone, id_variable, value)"
                                          " VALUES (?, ?, ?, ?)" );
    //QString dateTime;
    QVariantList dateTimeList;
    QVariantList zoneList;
    QVariantList idVariableList;
    QVariantList valueList;


    for (int day = 0; day < nrDays; day++)
    {

        // LC NB le zone partono da 1, a 0 è NODATA
        for (int zone = 1; zone < nZones; zone++)
        {
            float value = aggregatedValues[day][zone];
            if (value != NODATA)
            {
                dateTimeList << (startDate.addDays(day)).toString("yyyy-MM-dd hh:mm:ss");
                zoneList << zone;
                idVariableList << idVariable;
                valueList << value;

            }
        }
    }

    qry.addBindValue(dateTimeList);
    qry.addBindValue(zoneList);
    qry.addBindValue(idVariableList);
    qry.addBindValue(valueList);

    if( !qry.execBatch() )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
        return true;

}


bool Crit3DAggregationsDbHandler::saveTmpAggrData(QString aggrType, QString periodType, int nZones)
{

    QString statement;

    for (int zone = 1; zone < nZones; zone++)
    {
        statement = QString("INSERT INTO `%1_%2_%3` ").arg(zone).arg(aggrType).arg(periodType);
        statement += QString("SELECT date_time, id_variable, value FROM TmpAggregationData ");
        statement += QString("WHERE zone = %1").arg(zone);

        _db.exec(statement);
        if (_db.lastError().type() != QSqlError::NoError)
        {
            _error = _db.lastError().text();
            return false;
        }

    }
    return true;
}

std::vector<float> Crit3DAggregationsDbHandler::getAggrData(QString aggrType, QString periodType, int zone, QDateTime startDate, QDateTime endDate, meteoVariable variable)
{

    int idVariable = getIdfromMeteoVar(variable);
    unsigned int nrDays = unsigned(startDate.daysTo(endDate) + 1);
    std::vector<float> values(nrDays, NODATA);
    QDateTime date;
    float value;

    QString statement = QString( "SELECT * FROM `%1_%2_%3` WHERE date_time >= DATE('%4') AND date_time < DATE('%5', '+1 day') AND id_variable = '%6'")
                                .arg(zone).arg(aggrType).arg(periodType).arg(startDate.toString("yyyy-MM-dd hh:mm:ss")).arg(endDate.toString("yyyy-MM-dd hh:mm:ss")).arg(idVariable);
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
            try {
              meteoVar = MapDailyMeteoVar.at(stdVar);
            }
            catch (const std::out_of_range& ) {
                try {
                    meteoVar = MapHourlyMeteoVar.at(stdVar);
                }
                catch (const std::out_of_range& ) {
                    meteoVar = noMeteoVar;
                }
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

    if( !qry.exec() )
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
        _error = "name not found";
    }
    return aggregationList;
}

