#ifndef CROPDBTOOLS_H
#define CROPDBTOOLS_H

    class QSqlDatabase;
    class QString;
    class Crit3DCrop;

    bool openDbCrop(QSqlDatabase& dbCrop, const QString& dbName, QString& errorStr);

    bool deleteCropData(QSqlDatabase &dbCrop, const QString& cropName, QString& errorStr);

    bool loadCropParameters(const QSqlDatabase &dbCrop, QString idCrop, Crit3DCrop &myCrop, QString& errorStr);

    bool updateCropLAIparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr);

    bool updateCropRootparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr);

    bool updateCropIrrigationparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr);


#endif // CROPDBTOOLS_H
