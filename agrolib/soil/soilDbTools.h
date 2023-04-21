#ifndef SOILDBTOOLS_H
#define SOILDBTOOLS_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif

    #include <QString>
    class QSqlDatabase;

    bool loadSoilData(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString &errorStr);

    bool loadSoil(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil,
                  const std::vector<soil::Crit3DTextureClass> &textureClassList,
                  const soil::Crit3DFittingOptions &fittingOptions, QString &errorStr);

    bool updateSoilData(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString& errorStr);

    bool updateWaterRetentionData(QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, int horizon, QString& errorStr);

    bool insertSoilData(QSqlDatabase &dbSoil, int soilID, const QString &soilCode,
                        const QString &soilName, const QString &soilInfo, QString &errorStr);

    bool deleteSoilData(QSqlDatabase &dbSoil, const QString &soilCode, QString &errorStr);

    bool loadVanGenuchtenParameters(const QSqlDatabase &dbSoil, std::vector<soil::Crit3DTextureClass> &textureClassList, QString &errorStr);

    bool loadDriessenParameters(const QSqlDatabase &dbSoil, std::vector<soil::Crit3DTextureClass> &textureClassList, QString &errorStr);

    QString getIdSoilString(const QSqlDatabase &dbSoil, int idSoilNumber, QString &errorStr);
    int getIdSoilNumeric(const QSqlDatabase &dbSoil, QString soilCode, QString &errorStr);

    bool openDbSoil(const QString &dbSoilName, QSqlDatabase &dbSoil, QString &errorStr);

    bool loadAllSoils(const QString &dbSoilName, std::vector <soil::Crit3DSoil> &soilList,
                      std::vector<soil::Crit3DTextureClass> &textureClassList,
                      const soil::Crit3DFittingOptions &fittingOptions, QString &errorStr);

    bool loadAllSoils(const QSqlDatabase &dbSoil, std::vector <soil::Crit3DSoil> &soilList,
                      std::vector<soil::Crit3DTextureClass> &textureClassList,
                      const soil::Crit3DFittingOptions &fittingOptions, QString& errorStr);

    bool loadSoilInfo(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString &errorStr);

    bool getSoilList(const QSqlDatabase &dbSoil, QList<QString> &soilList, QString &errorStr);


#endif // SOILDBTOOLS_H
