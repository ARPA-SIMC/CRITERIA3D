#ifndef CROPDBQUERY_H
#define CROPDBQUERY_H

    #include <QString>
    class QSqlDatabase;

    bool getCropIdList(const QSqlDatabase &dbCrop, QList<QString>& cropIdList, QString& errorStr);

    QString getIdCropFromName(const QSqlDatabase &dbCrop, QString cropName, QString& errorStr);

    QString getCropFromClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField,
                             QString idCropClass, QString& errorStr);

    QString getCropFromId(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField,
                          int cropId, QString& errorStr);

    float getIrriRatioFromClass(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropClassField,
                                QString idCropClass, QString& errorStr);

    float getIrriRatioFromId(const QSqlDatabase &dbCrop, QString cropClassTable, QString cropIdField,
                         int cropId, QString& errorStr);


#endif // CROPDBQUERY_H
