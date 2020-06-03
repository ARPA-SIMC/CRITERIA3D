#include "criteria1DUnit.h"
#include "commonConstants.h"
#include <QSqlQuery>
#include <QSqlError>
#include <QVariant>
#include <QUuid>


Crit1DUnit::Crit1DUnit()
{
    this->idCase = "";
    this->idCrop = "";
    this->idSoil = "";
    this->idMeteo = "";

    this->idCropClass = "";
    this->idForecast = "";
    this->idSoilNumber = NODATA;
    this->idCropNumber = NODATA;
}


bool Crit1DUnit::load(QSqlDatabase* dbUnits, QString idCase, QString *error)
{
    QString queryString = "SELECT * FROM units WHERE ID_CASE='" + idCase +"'";

    QSqlQuery query = dbUnits->exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (query.lastError().nativeErrorCode() != "")
            *error = "dbUnits error: " + query.lastError().nativeErrorCode() + " - " + query.lastError().text();
        else
            *error = "Missing units";
        return false;
    }

    idCase = query.value("ID_CASE").toString();
    idCropClass = query.value("ID_CROP").toString();
    idMeteo = query.value("ID_METEO").toString();
    idForecast = query.value("ID_METEO").toString();
    idSoilNumber = query.value("ID_SOIL").toInt();

    return true;
}


bool openDbUnits(QString dbName, QSqlDatabase* dbUnits, QString* error)
{

    *dbUnits = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbUnits->setDatabaseName(dbName);

    if (!dbUnits->open())
    {
       *error = "Connection with database fail";
       return false;
    }

    return true;
}

