#include "ucmDb.h"
#include <QtSql>


UcmDb::UcmDb(QString dbname)
{
    error = "";

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
    else
    {
        createUnitsTable();
    }

}

UcmDb::~UcmDb()
{
    if ((db.isValid()) && (db.isOpen()))
    {
        QString connection = db.connectionName();
        db.close();
        db = QSqlDatabase();
        QSqlDatabase::removeDatabase(connection);
    }
}

void UcmDb::createUnitsTable()
{

    QSqlQuery qry(db);
    qry.prepare("CREATE TABLE units (ID_CASE TEXT, ID_CROP TEXT, ID_METEO TEXT, ID_SOIL TEXT, HA NUMERIC, PRIMARY KEY(ID_CASE))");
    if( !qry.exec() )
    {
        error = qry.lastError().text();
    }
}


bool UcmDb::writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                  QStringList idSoil, QList<double> ha)
{

    QString myString = "INSERT INTO units (ID_CASE, ID_CROP, ID_METEO, ID_SOIL, HA) VALUES ";

    for (int i = 0; i < idCase.size(); i++)
    {
        myString += "('" + idCase[i] + "','" + idCrop[i] + "','" + idMeteo[i] + "','" + idSoil[i];
        myString += "','" + QString::number(ha[i]) +"')";
        if (i < (idCase.size()-1))
            myString += ",";
    }

    QSqlQuery myQuery = db.exec(myString);
    myString.clear();

    if (myQuery.lastError().type() != QSqlError::NoError)
    {
        error = myQuery.lastError().text();
        return false;
    }

    return true;
}


QString UcmDb::getError() const
{
    return error;
}
