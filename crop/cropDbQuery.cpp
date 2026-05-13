#include "cropDbQuery.h"
#include "commonConstants.h"
#include "utilities.h"

#include <QtSql>


bool getCropIdList(const QSqlDatabase &dbCrop, QList<QString> &cropIdList, QString &errorStr)
{
    errorStr.clear();
    cropIdList.clear();

    QSqlQuery query(dbCrop);

    if (! query.exec("SELECT id_crop FROM crop"))
    {
        errorStr = query.lastError().text();
        return false;
    }

    while (query.next())
    {
        QString cropId = query.value(0).toString();

        if (! cropId.isEmpty())
        {
            cropIdList.append(cropId);
        }
    }

    return true;
}


QString getIdCropFromName(const QSqlDatabase &dbCrop, const QString &cropName, QString& errorStr)
{
    errorStr.clear();
    QSqlQuery query(dbCrop);

    QString queryString = QString("SELECT id_crop FROM crop WHERE crop_name COLLATE NOCASE = :cropName ");
    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return {};
    }

    query.bindValue(":cropName", cropName);

    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return {};
    }

    if (! query.next())
    {
        errorStr = "Missing id_crop for crop name: " + cropName;
        return {};
    }

    QString idCropStr;
    getValue(query.value("id_crop"), &idCropStr);

    return idCropStr;
}


QString getIdCropFromClass(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                           const QString &cropClassField, const QString &idCropClass, QString &errorStr)
{
    errorStr.clear();

    QSqlQuery query(dbCrop);

    QString queryString = QString("SELECT id_crop FROM %1 WHERE %2 = :idCropClass COLLATE NOCASE")
                                .arg(cropClassTable, cropClassField);

    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return {};
    }

    query.bindValue(":idCropClass", idCropClass);

    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return {};
    }

    if (! query.next())
    {
        errorStr = "Missing id_crop for crop class: " + idCropClass;
        return {};
    }

    return query.value("id_crop").toString();
}


QString getIdCropFromField(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                           const QString &cropIdField, int cropIdNumber, QString &errorStr)
{
    errorStr.clear();

    QSqlQuery query(dbCrop);

    QString queryString = QString("SELECT id_crop FROM %1 WHERE %2 = :cropId")
                                    .arg(cropClassTable, cropIdField);

    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return {};
    }

    query.bindValue(":cropId", cropIdNumber);

    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return {};
    }

    if (! query.next())
    {
        errorStr = "Missing id crop for crop nr: " + QString::number(cropIdNumber);
        return {};
    }

    return query.value("id_crop").toString();
}


float getIrriRatioFromCropClass(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                                const QString &cropClassField, const QString &idCropClass, QString &errorStr)
{
    errorStr.clear();

    QSqlQuery query(dbCrop);

    QString queryString = QString("SELECT irri_ratio FROM %1 WHERE %2 = :idCropClass")
                                    .arg(cropClassTable, cropClassField);

    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return NODATA;
    }

    query.bindValue(":idCropClass", idCropClass);

    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return NODATA;
    }

    if (! query.next())
    {
        errorStr = "Missing irri_ratio for crop class: " + idCropClass;
        return NODATA;
    }

    bool isNumber = false;
    float irriRatio = query.value("irri_ratio").toFloat(&isNumber);

    if (! isNumber)
    {
        errorStr = "Wrong irri_ratio value for crop class: " + idCropClass;
        return NODATA;
    }

    return irriRatio;
}


float getIrriRatioFromCropId(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                             const QString &cropIdField, int cropIdNumber, QString &errorStr)
{
    errorStr.clear();

    QSqlQuery query(dbCrop);

    QString queryString = QString("SELECT irri_ratio FROM %1 WHERE %2 = :cropId")
                              .arg(cropClassTable, cropIdField);

    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return NODATA;
    }

    query.bindValue(":cropId", cropIdNumber);

    if (! query.exec())
    {
        errorStr = query.lastError().text();
        return NODATA;
    }

    if (! query.next())
    {
        errorStr = "Missing irri_ratio for crop nr. " + QString::number(cropIdNumber);
        return NODATA;
    }

    float irriRatio = 0;
    if (! getValue(query.value("irri_ratio"), &irriRatio))
    {
        errorStr = "Invalid irri_ratio value";
        return NODATA;
    }

    return irriRatio;
}


bool getCropListFromType(const QSqlDatabase &dbCrop, const QString &cropType,
                         QList<QString> &cropList, QString &errorStr)
{
    errorStr.clear();
    cropList.clear();

    QSqlQuery query(dbCrop);
    query.setForwardOnly(true);

    const QString queryString = "SELECT id_crop FROM crop WHERE type = :cropType";

    if (! query.prepare(queryString))
    {
        errorStr = query.lastError().text();
        return false;
    }

    query.bindValue(":cropType", cropType);

    if (! query.exec())
    {
        errorStr = "Error reading crop list from crop type: " + cropType + "\n"
                   + query.lastError().text();
        return false;
    }

    while (query.next())
    {
        const QString cropId = query.value(0).toString();

        if (! cropId.isEmpty())
            cropList.append(cropId);
    }

    if (cropList.isEmpty())
    {
        errorStr = "Missing crop type: " + cropType;
        return false;
    }

    return true;
}
