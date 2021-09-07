#include "computationUnitsDb.h"
#include "commonConstants.h"
#include "utilities.h"

#include <QtSql>


Crit1DUnit::Crit1DUnit()
{
    idCase = "";
    idCropClass = "";
    idCrop = "";

    idSoil = "";
    idSoilNumber = NODATA;

    idMeteo = "";
    idForecast = "";

    // default values
    isNumericalInfiltration = false;
    isGeometricLayers = false;
    isOptimalIrrigation = false;
    useWaterTableData = true;
    useWaterRetentionData = true;
    slope = 0.01;
}


ComputationUnitsDB::ComputationUnitsDB(QString dbname, QString &error)
{
    if(db.isOpen())
    {
        qDebug() << db.connectionName() << "is already open";
        db.close();
    }

    db = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    db.setDatabaseName(dbname);

    if (!db.open())
    {
       error = db.lastError().text();
    }
}


ComputationUnitsDB::~ComputationUnitsDB()
{
    if ((db.isValid()) && (db.isOpen()))
    {
        db.close();
    }
}


bool ComputationUnitsDB::writeListToUnitsTable(QList<QString> idCase, QList<QString> idCrop, QList<QString> idMeteo,
                                  QList<QString> idSoil, QList<double> hectares, QString &error)
{
    QSqlQuery qry(db);
    qry.prepare("CREATE TABLE units (ID_CASE TEXT, ID_CROP TEXT, ID_METEO TEXT, ID_SOIL TEXT, hectares NUMERIC, PRIMARY KEY(ID_CASE))");
    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return false;
    }
    qry.clear();

    QString myQuery = "INSERT INTO units (ID_CASE, ID_CROP, ID_METEO, ID_SOIL, hectares) VALUES ";

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
bool ComputationUnitsDB::readUnitList(std::vector<Crit1DUnit> &unitList, QString &error)
{
    QString unitsTable = "units";
    QList<QString> fieldList = getFields(&db, unitsTable);
    bool existNumericalInfiltration = fieldList.contains("numerical_solution");
    bool existWaterRetentionData = fieldList.contains("water_retention_fitting");
    bool existWaterTable = fieldList.contains("use_water_table");
    bool existOptimalIrrigation = fieldList.contains("optimal_irrigation");
    bool existSlope = fieldList.contains("slope");
    // TODO others

    QString queryString = "SELECT * FROM " + unitsTable;
    queryString += " ORDER BY ID_CROP, ID_SOIL, ID_METEO";

    QSqlQuery query = db.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (query.lastError().nativeErrorCode() != "")
        {
            error = "dbUnits error: " + query.lastError().nativeErrorCode() + " - " + query.lastError().text();
        }
        else error = "Missing units data";

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
        unitList[i].idSoilNumber = query.value("ID_SOIL").toInt();

        if (existNumericalInfiltration)
            unitList[i].isNumericalInfiltration = query.value("numerical_solution").toBool();
        if (existWaterTable)
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


bool readUnitList(QString dbUnitsName, std::vector<Crit1DUnit> &unitList, QString &error)
{
    ComputationUnitsDB dbUnits(dbUnitsName, error);
    if (error != "")
    {
        return false;
    }
    if (! dbUnits.readUnitList(unitList, error))
    {
        return false;
    }

    return true;
}

