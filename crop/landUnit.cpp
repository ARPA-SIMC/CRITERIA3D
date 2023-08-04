#include "landUnit.h"
#include "commonConstants.h"
#include <QtSql>


Crit3DLandUnit::Crit3DLandUnit()
{
    id = 0;

    name = "DEFAULT";
    description = "Default land use";
    idCrop = "FALLOW";
    idLandUse = "FALLOW";

    roughness = 0.05;
    pond = 0.002;
}


bool loadLandUnitList(const QSqlDatabase &dbCrop, std::vector<Crit3DLandUnit> &landUnitList, QString &errorStr)
{
    landUnitList.clear();

    QString queryString = "SELECT * FROM land_units";
    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = query.lastError().text();
        else
            errorStr = "the table is empty";

        return false;
    }

    int nrUnits = query.at() + 1;     // SQLITE doesn't support SIZE
    landUnitList.resize(nrUnits);

    int i = 0;
    query.first();
    do
    {
        landUnitList[i].id = query.value("id_unit").toInt();

        landUnitList[i].name = query.value("name").toString();
        landUnitList[i].description = query.value("description").toString();
        landUnitList[i].idCrop = query.value("id_crop").toString();
        landUnitList[i].idLandUse = query.value("id_landuse").toString();

        landUnitList[i].roughness = query.value("roughness").toDouble();
        landUnitList[i].pond = query.value("pond").toDouble();

        i++;
    }
    while(query.next());

    return true;
}


int getLandUnitIndex(const std::vector<Crit3DLandUnit> &landUnitList, int idLandUnit)
{
    for (int index = 0; index < landUnitList.size(); index++)
        if (landUnitList[index].id == idLandUnit)
            return index;

    return NODATA;
}
