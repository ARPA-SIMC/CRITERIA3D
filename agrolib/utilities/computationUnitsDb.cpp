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
    slope = 0.02;                       // [-] default: 2%
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


bool ComputationUnitsDB::writeListToCompUnitsTable(const QList<QString> &idCase, const QList<QString> &idCrop,
                                                   const QList<QString> &idMeteo, const QList<QString> &idSoil,
                                                   const QList<QString> &idWaterTable, const QList<double> &hectares,
                                                   QString &errorStr)
{
    // check size
    if (idCase.size() != idCrop.size() ||
        idCase.size() != idMeteo.size() ||
        idCase.size() != idSoil.size() ||
        idCase.size() != hectares.size())
    {
        errorStr = "Input list sizes mismatch.";
        return false;
    }

    QSqlQuery qry(_db);
    qry.prepare(
        "CREATE TABLE IF NOT EXISTS computational_units ("
        "ID_CASE TEXT PRIMARY KEY, "
        "ID_CROP TEXT, "
        "ID_METEO TEXT, "
        "ID_SOIL TEXT, "
        "ID_WATERTABLE TEXT, "
        "HECTARES NUMERIC, "
        "use_water_table INTEGER DEFAULT 1, "
        "numerical_solution INTEGER DEFAULT 0)");

    if(! qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    // SINGLE BULK INSERT
    QString queryStr =
        "INSERT INTO computational_units ("
        "ID_CASE, ID_CROP, ID_METEO, ID_SOIL, HECTARES, "
        "ID_WATERTABLE, use_water_table) VALUES ";

    QStringList rows;

    const int n = idCase.size();

    for (int i = 0; i < n; ++i)
    {
        rows << "(?, ?, ?, ?, ?, ?, ?)";
    }

    queryStr += rows.join(",");

    if (! _db.transaction())
    {
        errorStr = _db.lastError().text();
        return false;
    }

    qry.prepare(queryStr);

    for (int i = 0; i < n; ++i)
    {
        qry.addBindValue(idCase[i]);
        qry.addBindValue(idCrop[i]);
        qry.addBindValue(idMeteo[i]);
        qry.addBindValue(idSoil[i]);
        qry.addBindValue(hectares[i]);

        // watertable on/off
        if (i < idWaterTable.size())
        {
            qry.addBindValue(idWaterTable[i]);
            qry.addBindValue(1);
        }
        else
        {
            qry.addBindValue(QString());
            qry.addBindValue(0);
        }
    }

    if (! qry.exec())
    {
        _db.rollback();
        errorStr = qry.lastError().text();
        return false;
    }

    if (! _db.commit())
    {
        errorStr = _db.lastError().text();
        return false;
    }

    return true;
}


// load computation units list
bool ComputationUnitsDB::readComputationUnitList(std::vector<Crit1DCompUnit> &unitList, QString &errorStr)
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
            errorStr = "Error in reading computational units.\n" + query.lastError().text();
        }
        else errorStr = "Missing computational units data";

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


bool readComputationUnitList(QString dbComputationUnitsName, std::vector<Crit1DCompUnit> &unitList, QString &errorStr)
{
    ComputationUnitsDB dbCompUnits(dbComputationUnitsName, errorStr);
    if (errorStr != "")
    {
        return false;
    }
    if (! dbCompUnits.readComputationUnitList(unitList, errorStr))
    {
        return false;
    }

    return true;
}

