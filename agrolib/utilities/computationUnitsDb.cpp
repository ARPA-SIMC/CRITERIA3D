#include "computationUnitsDb.h"
#include "commonConstants.h"
#include "utilities.h"

#include <QtSql>


Crit1DCompUnit::Crit1DCompUnit()
{
    idCase = "";
    idCropClass = "";
    idCrop = "";
    idWaterTable = "";

    idSoil = "";
    idSoilNumber = NODATA;

    idMeteo = "";
    idForecast = "";

    // default values
    isNumericalInfiltration = false;
    isComputeLateralDrainage = true;
    isGeometricLayers = false;
    isOptimalIrrigation = false;
    useWaterTableData = true;
    useWaterRetentionData = true;
    slope = 0.01;
}


ComputationUnitsDB::ComputationUnitsDB(QString dbname, QString &error)
{
    if(_db.isOpen())
    {
        qDebug() << _db.connectionName() << "is already open";
        _db.close();
    }

    _db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    _db.setDatabaseName(dbname);

    if (!_db.open())
    {
       error = _db.lastError().text();
    }
}


ComputationUnitsDB::~ComputationUnitsDB()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        _db.close();
    }
}


bool ComputationUnitsDB::writeListToCompUnitsTable(QList<QString> &idCase, QList<QString> &idCrop,
                                                   QList<QString> &idMeteo, QList<QString> &idSoil,
                                                   QList<double> &hectares, QString &error)
{
    QSqlQuery qry(_db);
    qry.prepare("CREATE TABLE computational_units (ID_CASE TEXT, ID_CROP TEXT, ID_METEO TEXT, ID_SOIL TEXT, HECTARES NUMERIC, PRIMARY KEY(ID_CASE))");
    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return false;
    }
    qry.clear();

    QString myQuery = "INSERT INTO computational_units (ID_CASE, ID_CROP, ID_METEO, ID_SOIL, HECTARES) VALUES ";

    for (int i = 0; i < idCase.size(); i++)
    {
        myQuery += "('" + idCase[i] + "','" + idCrop[i] + "','" + idMeteo[i] + "','" + idSoil[i];
        myQuery += "','" + QString::number(hectares[i]) +"')";
        if (i < (idCase.size()-1))
            myQuery += ",";
    }

    if( !qry.exec(myQuery))
    {
        error = qry.lastError().text();
        myQuery.clear();
        return false;
    }
    else
    {
        myQuery.clear();
        return true;
    }
}


// load computation units list
bool ComputationUnitsDB::readComputationUnitList(std::vector<Crit1DCompUnit> &unitList, QString &error)
{
    QString compUnitsTable = "computational_units";
    QList<QString> fieldList = getFields(&_db, compUnitsTable);
    bool existWaterTableId = fieldList.contains("ID_WATERTABLE");
    bool existNumericalInfiltration = fieldList.contains("numerical_solution");
    bool existComputeLateralDrainage = fieldList.contains("compute_lateral_drainage");
    bool existWaterRetentionData = fieldList.contains("water_retention_fitting");
    bool existUseWaterTable = fieldList.contains("use_water_table");
    bool existOptimalIrrigation = fieldList.contains("optimal_irrigation");
    bool existSlope = fieldList.contains("slope");
    // TODO others

    QString queryString = "SELECT * FROM " + compUnitsTable;
    queryString += " ORDER BY ID_CROP, ID_SOIL, ID_METEO";

    QSqlQuery query = _db.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (query.lastError().nativeErrorCode() != "")
        {
            error = "Error in reading computational units.\n" + query.lastError().text();
        }
        else error = "Missing computational units data";

        return false;
    }

    unsigned int nrUnits = unsigned(query.at() + 1);     // SQLITE doesn't support SIZE
    unitList.clear();
    unitList.resize(nrUnits);

    unsigned int i = 0;
    query.first();
    do
    {
        unitList[i].idCase = query.value("ID_CASE").toString();
        unitList[i].idCropClass = query.value("ID_CROP").toString();
        unitList[i].idMeteo = query.value("ID_METEO").toString();
        unitList[i].idForecast = query.value("ID_METEO").toString();

        bool isNumber = false;
        unitList[i].idSoilNumber = query.value("ID_SOIL").toInt(&isNumber);
        if (! isNumber)
        {
            // read soilCode
            unitList[i].idSoil = query.value("ID_SOIL").toString();
            unitList[i].idSoilNumber = NODATA;
        }

        if (existWaterTableId)
        {
            unitList[i].idWaterTable = query.value("ID_WATERTABLE").toString();
        }

        if (existNumericalInfiltration)
            unitList[i].isNumericalInfiltration = query.value("numerical_solution").toBool();
        if (existComputeLateralDrainage)
            unitList[i].isComputeLateralDrainage = query.value("compute_lateral_drainage").toBool();
        if (existUseWaterTable)
            unitList[i].useWaterTableData = query.value("use_water_table").toBool();
        if (existOptimalIrrigation)
            unitList[i].isOptimalIrrigation = query.value("optimal_irrigation").toBool();
        if (existWaterRetentionData)
            unitList[i].useWaterRetentionData = query.value("water_retention_fitting").toBool();
        if (existSlope)
        {
            double slope;
            if (getValue(query.value("slope"), &slope))
                unitList[i].slope = slope;
        }

        i++;
    }
    while(query.next());

    return true;
}


bool readComputationUnitList(QString dbComputationUnitsName, std::vector<Crit1DCompUnit> &unitList, QString &error)
{
    ComputationUnitsDB dbCompUnits(dbComputationUnitsName, error);
    if (error != "")
    {
        return false;
    }
    if (! dbCompUnits.readComputationUnitList(unitList, error))
    {
        return false;
    }

    return true;
}

