#include "criteria1DUnit.h"
#include "commonConstants.h"
#include <QFile>
#include <QSqlQuery>
#include <QSqlError>
#include <QVariant>


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


// load computation units list
bool loadUnitList(QString dbUnitsName, std::vector<Crit1DUnit> &unitList, QString &myError)
{
    if (! QFile(dbUnitsName).exists())
    {
        myError = "DB units doesn't exist.";
        return false;
    }

    QSqlDatabase dbUnits = QSqlDatabase::addDatabase("QSQLITE", "units");
    dbUnits.setDatabaseName(dbUnitsName);
    if (! dbUnits.open())
    {
        myError = "Open DB Units failed.";
        return false;
    }

    QString queryString = "SELECT DISTINCT ID_CASE, ID_CROP, ID_SOIL, ID_METEO FROM units";
    queryString += " ORDER BY ID_CROP, ID_SOIL, ID_METEO";

    QSqlQuery query = dbUnits.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (query.lastError().nativeErrorCode() != "")
        {
            myError = "dbUnits error: " + query.lastError().nativeErrorCode() + " - " + query.lastError().text();
        }
        else
        {
            myError = "Missing units";
        }
        return false;
    }

    int nrUnits = query.at() + 1;     // SQLITE doesn't support SIZE
    unitList.clear();
    unitList.resize(nrUnits);

    int i = 0;
    query.first();
    do
    {
        unitList[i].idCase = query.value("ID_CASE").toString();
        unitList[i].idCropClass = query.value("ID_CROP").toString();
        unitList[i].idMeteo = query.value("ID_METEO").toString();
        unitList[i].idForecast = query.value("ID_METEO").toString();
        unitList[i].idSoilNumber = query.value("ID_SOIL").toInt();
        i++;

    } while(query.next());

    return true;
}

