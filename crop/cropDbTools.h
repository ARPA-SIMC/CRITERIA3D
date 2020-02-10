#ifndef CROPDBTOOLS_H
#define CROPDBTOOLS_H

    class Crit3DCrop;
    class QString;
    class QStringList;
    class QSqlDatabase;

    bool openDbCrop(QString dbName, QSqlDatabase* dbCrop, QString* error);

    bool getCropNameList(QSqlDatabase* dbCrop, QStringList* cropNameList, QString* error);

    QString getIdCropFromName(QSqlDatabase* dbCrop, QString cropName, QString *myError);

    QString getCropFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField,
                             QString idCropClass, QString *myError);
    QString getCropFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField,
                          int cropId, QString *myError);
    float getIrriRatioFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField,
                                QString idCrop, QString *myError);
    float getIrriRatioFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField,
                             int cropId, QString *myError);
    bool loadCropParameters(QString idCrop, Crit3DCrop* myCrop, QSqlDatabase* dbCrop, QString *myError);


#endif // CROPDBTOOLS_H
