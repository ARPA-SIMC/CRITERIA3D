#include "computationUnitsDb.h"
#include "commonConstants.h"
#include <QtSql>


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


ComputationUnitsDB::ComputationUnitsDB(QString dbname, QString &error)
{
    if(db.isOpen())
    {
        qDebug() << db.connectionName() << "is already open";
        db.close();
    }

    db = QSqlDatabase::addDatabase("QSQLITE", "Units");
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


bool ComputationUnitsDB::writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                  QStringList idSoil, QList<double> ha, QString &error)
{
    QSqlQuery qry(db);
    qry.prepare("CREATE TABLE units (ID_CASE TEXT, ID_CROP TEXT, ID_METEO TEXT, ID_SOIL TEXT, HA NUMERIC, PRIMARY KEY(ID_CASE))");
    if( !qry.exec() )
    {
        error = qry.lastError().text();
        return false;
    }
    qry.clear();

    QString myQuery = "INSERT INTO units (ID_CASE, ID_CROP, ID_METEO, ID_SOIL, HA) VALUES ";

    for (int i = 0; i < idCase.size(); i++)
    {
        myQuery += "('" + idCase[i] + "','" + idCrop[i] + "','" + idMeteo[i] + "','" + idSoil[i];
        myQuery += "','" + QString::number(ha[i]) +"')";
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
    QString queryString = "SELECT DISTINCT ID_CASE, ID_CROP, ID_SOIL, ID_METEO FROM units";
    queryString += " ORDER BY ID_CROP, ID_SOIL, ID_METEO";

    QSqlQuery query = db.exec(queryString);
    query.last();
    if (! query.isValid())
    {
        if (query.lastError().nativeErrorCode() != "")
        {
            error = "dbUnits error: " + query.lastError().nativeErrorCode() + " - " + query.lastError().text();
        }
        else error = "Missing units";

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

