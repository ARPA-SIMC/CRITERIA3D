#include "cropDbQuery.h"
#include "commonConstants.h"
#include "utilities.h"

#include <QtSql>


bool getCropIdList(const QSqlDatabase &dbCrop, QList<QString>& cropIdList, QString& errorStr)
{
    QString queryString = "SELECT id_crop FROM crop";
    QSqlQuery query = dbCrop.exec(queryString);

    query.first();
    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return false;
    }

    do
    {
        QString cropId;
        getValue(query.value("id_crop"), &cropId);
        if (cropId != "")
        {
            cropIdList.append(cropId);
        }
    }
    while(query.next());

    return true;
}


QString getIdCropFromName(QSqlDatabase* dbCrop, QString cropName, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM crop WHERE crop_name='" + cropName +"' COLLATE NOCASE";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        *myError = query.lastError().text();
        return "";
    }

    QString idCrop;
    getValue(query.value("id_crop"), &idCrop);

    return idCrop;
}


QString getCropFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField, QString idCropClass, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM " + cropClassTable
                          + " WHERE " + cropClassField + " = '" + idCropClass + "'"
                          + " COLLATE NOCASE";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


QString getCropFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


float getIrriRatioFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField, QString idCropClass, QString *myError)
{
    *myError = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropClassField + " = '" + idCropClass + "'";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}


float getIrriRatioFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString *myError)
{
    *myError = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}
