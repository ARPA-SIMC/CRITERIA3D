#include "dbClimate.h"
#include "utilities.h"
#include "commonConstants.h"

#include <QtSql>


// LC saveDailyElab is a bit more efficient respect saveDailyElabSingleValue, making prepare just once.
// Anyway all the process for each id_point is only 2seconds.
bool saveDailyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_daily` (TimeIndex INTEGER, id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(TimeIndex,id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_daily` (TimeIndex smallint(5), id_point varchar(10), elab varchar(80), value float(6,1), PRIMARY KEY(TimeIndex,id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_daily` (TimeIndex, id_point, elab, value)"
                                      " VALUES (?, ?, ?, ?)" );



    for (unsigned int i = 0; i < allResults.size(); i++)
    {
        qry.addBindValue(i+1);
        qry.addBindValue(id);
        qry.addBindValue(elab);
        qry.addBindValue(QString::number(allResults[i],'f',3));

        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }


    return true;
}

bool deleteElab(QSqlDatabase db, QString *myError, QString table, QString elab)
{
    QSqlQuery qry(db);
    QString statement = QString("DELETE FROM `%1`").arg(table);
    qry.prepare( statement + " WHERE elab = :elab" );

    qry.bindValue(":elab", elab);

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }

    return true;
}


float readClimateElab(const QSqlDatabase &db, const QString &table, const int &timeIndex,
                      const QString &id, const QString &elab, QString *myError)
{
    *myError = "";
    QSqlQuery qry(db);

    QString statement = QString("SELECT * FROM `%1` ").arg(table);
    if (table == "climate_annual" || table == "climate_generic")
    {
        qry.prepare( statement + " WHERE id_point = :id_point AND elab = :elab" );
    }
    else
    {
        qry.prepare( statement + " WHERE TimeIndex= :TimeIndex AND id_point = :id_point AND elab = :elab" );
        qry.bindValue(":TimeIndex", timeIndex);
    }

    qry.bindValue(":id_point", id);
    qry.bindValue(":elab", elab);

    if(! qry.exec() )
    {
        *myError = qry.lastError().text();
        return NODATA;
    }

    float value = NODATA;
    if (qry.next())
    {
        getValue(qry.value("value"), &value);
    }

    return value;
}


QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab)
{
    QSqlQuery qry(db);
    QString id;
    QList<QString> idList;

    QString statement = QString("SELECT distinct(id_point) FROM `%1`").arg(table);
    qry.prepare( statement + " WHERE elab = :elab AND value != -9999.0" );

    qry.bindValue(":elab", elab);

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("id_point"), &id);
            idList.append(id);
        }
    }

    return idList;
}

QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab, int index)
{
    QSqlQuery qry(db);
    QString id;
    QList<QString> idList;
    int i;

    QString statement = QString("SELECT * FROM `%1`").arg(table);
    qry.prepare( statement + " WHERE elab = :elab AND value != -9999.0" );

    qry.bindValue(":elab", elab);

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value(0), &i);
            getValue(qry.value("id_point"), &id);
            if (i == index)
            {
                idList.append(id);
            }
        }
    }

    return idList;
}


bool saveDecadalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_decadal` (TimeIndex INTEGER, id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(TimeIndex,id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_decadal` (TimeIndex smallint(5), id_point varchar(10), elab  varchar(80), value float(6,1), PRIMARY KEY(TimeIndex,id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_decadal` (TimeIndex, id_point, elab, value)"
                                      " VALUES (?, ?, ?, ?)" );

    for (unsigned int i = 0; i < allResults.size(); i++)
    {
        qry.addBindValue(i+1);
        qry.addBindValue(id);
        qry.addBindValue(elab);
        qry.addBindValue(QString::number(allResults[i],'f',3));

        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool saveMonthlyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_monthly` (TimeIndex INTEGER, id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(TimeIndex,id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_monthly` (TimeIndex smallint(5), id_point varchar(10), elab  varchar(80), value float(6,1), PRIMARY KEY(TimeIndex,id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_monthly` (TimeIndex, id_point, elab, value)"
                                      " VALUES (?, ?, ?, ?)" );

    for (unsigned int i = 0; i < allResults.size(); i++)
    {
        qry.addBindValue(i+1);
        qry.addBindValue(id);
        qry.addBindValue(elab);
        qry.addBindValue(QString::number(allResults[i],'f',3));

        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool saveSeasonalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_seasonal` (TimeIndex INTEGER, id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(TimeIndex,id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_seasonal` (TimeIndex smallint(5), id_point varchar(10), elab  varchar(80), value float(6,1), PRIMARY KEY(TimeIndex,id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_seasonal` (TimeIndex, id_point, elab, value)"
                                      " VALUES (?, ?, ?,?)" );

    for (unsigned int i = 0; i < allResults.size(); i++)
    {
        qry.addBindValue(i+1);
        qry.addBindValue(id);
        qry.addBindValue(elab);
        qry.addBindValue(QString::number(allResults[i],'f',3));

        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool saveAnnualElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_annual` (id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_annual` (id_point varchar(10), elab  varchar(80), value float(6,1), PRIMARY KEY(id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_annual` (id_point, elab, value)"
                                      " VALUES (:id_point, :elab, :value)" );

    qry.bindValue(":id_point", id);
    qry.bindValue(":elab", elab);
    qry.bindValue(":value", QString::number(result,'f',3));

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }

    return true;
}

bool saveGenericElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab)
{
    QSqlQuery qry(db);
    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_generic` (id_point TEXT, elab TEXT, value REAL, PRIMARY KEY(id_point,elab));");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("CREATE TABLE IF NOT EXISTS `climate_generic` (id_point varchar(10), elab  varchar(80), value float(6,1), PRIMARY KEY(id_point,elab) );");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    qry.prepare( "REPLACE INTO `climate_generic` (id_point, elab, value)"
                                      " VALUES (:id_point, :elab, :value)" );

    qry.bindValue(":id_point", id);
    qry.bindValue(":elab", elab);
    qry.bindValue(":value", QString::number(result,'f',3));

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }

    return true;
}

bool selectVarElab(QSqlDatabase db, QString *myError, QString table, QString variable, QList<QString>* listElab)
{
    QSqlQuery qry(db);
    QString elab;

    bool found = false;
    variable = variable.remove("_"); // db save variable without "_"

    QString statement = QString("SELECT DISTINCT elab from `%1` WHERE `elab` LIKE '%%2%%'").arg(table).arg(variable);

    qry.prepare(statement);

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {

        while (qry.next())
        {
            if (getValue(qry.value("elab"), &elab))
            {
                listElab->append(elab);
                found = true;
            }
            else
            {
                *myError = qry.lastError().text();
            }
        }
    }

    return found;
}

bool getClimateFieldsFromTable(QSqlDatabase db, QString *myError, QString climateTable, QList<QString>* fieldList)
{
    QSqlQuery qry(db);
    QString elab;

    bool found = false;

    QString statement = QString("SELECT DISTINCT elab from `%1` ").arg(climateTable);

    qry.prepare(statement);

    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {

        while (qry.next())
        {
            if (getValue(qry.value("elab"), &elab))
            {
                fieldList->append(elab);
                found = true;
            }
            else
            {
                *myError = qry.lastError().text();
            }
        }
    }

    return found;
}

bool getClimateTables(QSqlDatabase db, QString *myError, QList<QString>* climateTables)
{
    QSqlQuery qry(db);
    QString table;
    QString type;

    if (db.driverName() == "QSQLITE")
    {
        qry.prepare("SELECT * FROM sqlite_master");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
        else
        {
            while (qry.next())
            {
                if (getValue(qry.value("name"), &table) && getValue(qry.value("type"),&type))
                {
                    if ( type == "table" && table.contains("climate_") )
                    {
                       climateTables->append(table);
                    }

                }
                else
                {
                    *myError = qry.lastError().text();
                    return false;
                }
            }
        }
    }
    else if (db.driverName() == "QMYSQL")
    {
        qry.prepare("SHOW TABLES LIKE 'climate_%'");
        if( !qry.exec() )
        {
            *myError = qry.lastError().text();
            return false;
        }
        else
        {
            while (qry.next())
            {
                if (getValue(qry.value(0), &table))
                {
                    climateTables->append(table);
                }
                else
                {
                    *myError = qry.lastError().text();
                    return false;
                }
            }
        }
    }

    return true;
}

