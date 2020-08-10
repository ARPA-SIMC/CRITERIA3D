#ifndef CROPDBTOOLS_H
#define CROPDBTOOLS_H

    class QSqlDatabase;
    class QString;
    class Crit3DCrop;

    bool openDbCrop(QSqlDatabase* dbCrop, QString dbName, QString* error);

    bool deleteCropData(QSqlDatabase* dbCrop, QString cropName, QString *error);

    bool loadCropParameters(QSqlDatabase* dbCrop, QString idCrop, Crit3DCrop* myCrop, QString *myError);

    bool updateCropLAIparam(QSqlDatabase* dbCrop, Crit3DCrop* myCrop, QString *error);

    bool updateCropRootparam(QSqlDatabase* dbCrop, Crit3DCrop* myCrop, QString *error);

    bool updateCropIrrigationparam(QSqlDatabase* dbCrop, Crit3DCrop* myCrop, QString *error);


#endif // CROPDBTOOLS_H
