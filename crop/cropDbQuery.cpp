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


QString getIdCropFromName(const QSqlDatabase &dbCrop, QString cropName, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM crop WHERE crop_name='" + cropName +"' COLLATE NOCASE";

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return "";
    }

    QString idCrop;
    getValue(query.value("id_crop"), &idCrop);

    return idCrop;
}


QString getCropFromClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField, QString idCropClass, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM " + cropClassTable
                          + " WHERE " + cropClassField + " = '" + idCropClass + "'"
                          + " COLLATE NOCASE";

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


QString getCropFromId(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


float getIrriRatioFromClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField, QString idCropClass, QString &errorStr)
{
    errorStr = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropClassField + " = '" + idCropClass + "'";

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}


float getIrriRatioFromId(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString &errorStr)
{
    errorStr = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}
