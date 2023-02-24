#ifndef SOILDBTOOLS_H
#define SOILDBTOOLS_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #include <QString>

    class QSqlDatabase;

    bool loadSoilData(QSqlDatabase* dbSoil, QString soilCode, soil::Crit3DSoil *mySoil, QString *myError);

    bool loadSoil(QSqlDatabase* dbSoil, QString soilCode, soil::Crit3DSoil* mySoil,
                  soil::Crit3DTextureClass* textureClassList,
                  soil::Crit3DFittingOptions *fittingOptions, QString* error);

    bool updateSoilData(QSqlDatabase* dbSoil, QString soilCode, soil::Crit3DSoil* mySoil, QString *error);

    bool updateWaterRetentionData(QSqlDatabase* dbSoil, QString soilCode, soil::Crit3DSoil* mySoil, int horizon, QString *error);

    bool insertSoilData(QSqlDatabase* dbSoil, int soilID, QString soilCode, QString soilName, QString soilInfo, QString *error);

    bool deleteSoilData(QSqlDatabase* dbSoil, QString soilCode, QString *error);

    bool loadVanGenuchtenParameters(QSqlDatabase* dbSoil, soil::Crit3DTextureClass* textureClassList, QString *error);

    bool loadDriessenParameters(QSqlDatabase* dbSoil, soil::Crit3DTextureClass* textureClassList, QString *error);

    QString getIdSoilString(QSqlDatabase* dbSoil, int idSoilNumber, QString *myError);

    bool openDbSoil(QString dbName, QSqlDatabase* dbSoil, QString* error);

    bool loadAllSoils(QString dbSoilName, std::vector <soil::Crit3DSoil> *soilList,
                      soil::Crit3DTextureClass *textureClassList,
                      soil::Crit3DFittingOptions *fittingOptions, QString* error);

    bool loadAllSoils(QSqlDatabase* dbSoil, std::vector <soil::Crit3DSoil> *soilList,
                      soil::Crit3DTextureClass *textureClassList,
                      soil::Crit3DFittingOptions *fittingOptions, QString* error);

    bool loadSoilInfo(QSqlDatabase* dbSoil, QString soilCode, soil::Crit3DSoil* mySoil, QString *error);
    bool getSoilList(QSqlDatabase* dbSoil, QList<QString>* soilList, QString* error);


#endif // SOILDBTOOLS_H
