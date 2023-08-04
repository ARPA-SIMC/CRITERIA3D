#ifndef CROPDBQUERY_H
#define CROPDBQUERY_H

    #include <QString>
    class QSqlDatabase;

    bool getCropIdList(const QSqlDatabase &dbCrop, QList<QString>& cropIdList, QString& errorStr);

    QString getIdCropFromName(const QSqlDatabase &dbCrop, QString cropName, QString& errorStr);

    QString getIdCropFromClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField,
                             QString idCropClass, QString& errorStr);

    QString getIdCropFromField(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField,
                          int cropId, QString& errorStr);

    float getIrriRatioFromCropClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField,
                                QString idCropClass, QString& errorStr);

    float getIrriRatioFromCropId(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField,
                         int cropId, QString& errorStr);


#endif // CROPDBQUERY_H
