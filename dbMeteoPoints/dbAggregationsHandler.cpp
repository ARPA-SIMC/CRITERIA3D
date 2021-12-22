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

bool Crit3DAggregationsDbHandler::saveAggrData(int nZones, QString aggrType, QString periodType, QDate startDate, QDate endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues)
{
    initAggregatedTables(nZones, aggrType, periodType, startDate, endDate, variable);
    createTmpAggrTable();
    insertTmpAggr(startDate, endDate, variable, aggregatedValues, nZones);
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

bool Crit3DAggregationsDbHandler::writeRasterName(QString rasterName)
{
    QSqlQuery qry(_db);

    qry.prepare( "INSERT INTO zones (name)"
                                      " VALUES (:name)" );

    qry.bindValue(":name", rasterName);

    if( !qry.exec() )
    {
        _error = qry.lastError().text();
        return false;
    }
    else
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

void Crit3DAggregationsDbHandler::initAggregatedTables(int numZones, QString aggrType, QString periodType, QDate startDate, QDate endDate, meteoVariable variable)
{

    int idVariable = getIdfromMeteoVar(variable);
    for (int i = 1; i <= numZones; i++)
    {
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1_%2_%3` "
                                    "(date TEXT, id_variable INTEGER, value REAL, PRIMARY KEY(date,id_variable))").arg(i).arg(aggrType).arg(periodType);

        QSqlQuery qry(statement, _db);
        if( !qry.exec() )
        {
            _error = qry.lastError().text();
        }
        statement = QString("DELETE FROM `%1_%2_%3` WHERE date >= DATE('%4') AND date < DATE('%5', '+1 day') AND id_variable = %6")
                        .arg(i).arg(aggrType).arg(periodType).arg(startDate.toString("yyyy-MM-dd")).arg(endDate.toString("yyyy-MM-dd")).arg(idVariable);

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
    qry.prepare("CREATE TABLE TmpAggregationData (date TEXT, zone TEXT, id_variable INTEGER, value REAL)");
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


bool Crit3DAggregationsDbHandler::insertTmpAggr(QDate startDate, QDate endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues, int nZones)
{
    int idVariable = getIdfromMeteoVar(variable);
    int nrDays = int(startDate.daysTo(endDate) + 1);
    QSqlQuery qry(_db);
    qry.prepare( "INSERT INTO `TmpAggregationData` (date, zone, id_variable, value)"
                                          " VALUES (?, ?, ?, ?)" );
    //QString dateTime;
    QVariantList dateTimeList;
    QVariantList zoneList;
    QVariantList idVariableList;
    QVariantList valueList;


    for (int day = 0; day < nrDays; day++)
    {

        // LC NB le zone partono da 1
        for (int zone = 1; zone <= nZones; zone++)
        {
            if (aggregatedValues[day][zone-1] != NODATA)
            {
                QString value = QString::number(aggregatedValues[day][zone-1], 'f', 1);
                if (value != "nan")
                {
                    dateTimeList << (startDate.addDays(day)).toString("yyyy-MM-dd");
                    zoneList << zone;
                    idVariableList << idVariable;
                    valueList << value;

                }
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

    for (int zone = 1; zone <= nZones; zone++)
    {
        statement = QString("REPLACE INTO `%1_%2_%3` ").arg(zone).arg(aggrType).arg(periodType);
        statement += QString("SELECT date, id_variable, value FROM TmpAggregationData ");
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

std::vector<float> Crit3DAggregationsDbHandler::getAggrData(QString aggrType, QString periodType, int zone, QDate startDate, QDate endDate, meteoVariable variable)
{

    int idVariable = getIdfromMeteoVar(variable);
    unsigned int nrDays = unsigned(startDate.daysTo(endDate) + 1);
    std::vector<float> values(nrDays, NODATA);
    QDate date;
    float value;

    QString statement = QString( "SELECT * FROM `%1_%2_%3` WHERE date >= DATE('%4') AND date < DATE('%5', '+1 day') AND id_variable = '%6'")
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
            getValue(qry.value("date"), &date);
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

