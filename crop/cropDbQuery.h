#ifndef CROPDBQUERY_H
#define CROPDBQUERY_H

    #include <QString>
    class QSqlDatabase;

    bool getCropIdList(QSqlDatabase* dbCrop, QList<QString>* cropIdList, QString* error);

    QString getIdCropFromName(QSqlDatabase* dbCrop, QString cropName, QString *myError);

    QString getCropFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField,
                             QString idCropClass, QString *myError);

    QString getCropFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField,
                          int cropId, QString *myError);

    float getIrriRatioFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField,
                                QString idCropClass, QString *myError);

    float getIrriRatioFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField,
                         int cropId, QString *myError);


#endif // CROPDBQUERY_H
