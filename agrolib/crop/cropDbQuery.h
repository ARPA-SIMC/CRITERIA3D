#ifndef CROPDBQUERY_H
#define CROPDBQUERY_H

    #include <QString>
    class QSqlDatabase;

    bool getCropIdList(const QSqlDatabase &dbCrop, QList<QString>& cropIdList, QString& errorStr);

    bool getCropListFromType(const QSqlDatabase &dbCrop, const QString &cropType,
                             QList<QString> &cropList, QString &errorStr);

    QString getIdCropFromName(const QSqlDatabase &dbCrop, const QString &cropName, QString& errorStr);

    QString getIdCropFromClass(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                               const QString &cropClassField, const QString &idCropClass, QString &errorStr);

    QString getIdCropFromField(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                               const QString &cropIdField, int cropIdNumber, QString &errorStr);

    float getIrriRatioFromCropClass(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                                    const QString &cropClassField, const QString &idCropClass, QString& errorStr);

    float getIrriRatioFromCropId(const QSqlDatabase &dbCrop, const QString &cropClassTable,
                                 const QString &cropIdField, int cropId, QString &errorStr);


#endif // CROPDBQUERY_H
