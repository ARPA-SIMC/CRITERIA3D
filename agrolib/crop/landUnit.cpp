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
    landUseType = LANDUSE_FALLOW;

    roughness = 0.05;
    pond = 0.002;
}


bool loadLandUnitList(const QSqlDatabase &dbCrop, std::vector<Crit3DLandUnit> &landUnitList, QString &errorStr)
{
    landUnitList.clear();

    QSqlQuery query(dbCrop);
    query.prepare("SELECT * FROM land_units");
    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return false;
    }

    query.last();
    if (! query.isValid())
    {
        errorStr = "the table land_units is empty";
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
        landUnitList[i].idLandUse = query.value("id_landuse").toString().toUpper();

        try
        {
            landUnitList[i].landUseType = MapLandUseFromString.at(landUnitList[i].idLandUse.toStdString());
        }
        catch (const std::out_of_range& outOfErrorStr)
        {
            errorStr = QString("%1 is not a valid landUse type" ).arg(landUnitList[i].idLandUse);
            landUnitList[i].landUseType = LANDUSE_FALLOW;
        }

        landUnitList[i].roughness = query.value("roughness").toDouble();
        landUnitList[i].pond = query.value("pond").toDouble();

        i++;
    }
    while( query.next() );

    return true;
}
