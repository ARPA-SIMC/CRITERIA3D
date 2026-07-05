/*!
* \brief computational unit for CRITERIA-1D
* \note computational unit = distinct combination of crop, soil and meteo
*/

#ifndef COMPUTATIONUNITSDB_H
#define COMPUTATIONUNITSDB_H

    #include <QString>
    #include <QSqlDatabase>
    #include <vector>

    class QSqlDatabase;

    class Crit1DCompUnit
    {
    public:
        QString idCase;
        QString idCropClass;
        QString idCrop;
        QString idWaterTable;

        QString idMeteo;
        QString idForecast;

        QString idSoil;
        int idSoilNumber;

        bool isNumericalInfiltration;
        bool isComputeLateralDrainage;
        bool isGeometricLayers;
        bool isOptimalIrrigation;
        bool useWaterTableData;
        bool useWaterRetentionData;
        double slope;                           // [m m-1]

        Crit1DCompUnit();
    };

    class ComputationUnitsDB
    {
    public:
        ComputationUnitsDB(QString dbname, QString &error);
        ~ComputationUnitsDB();

        bool writeListToCompUnitsTable(const QList<QString> &idCase, const QList<QString> &idCrop,
                                       const QList<QString> &idMeteo, const QList<QString> &idSoil,
                                       const QList<QString> &idWaterTable,
                                       const QList<double> &hectares, QString &errorStr);

        bool readComputationUnitList(std::vector<Crit1DCompUnit> &unitList, QString &errorStr);

    private:
        QSqlDatabase _db;
    };

    bool readComputationUnitList(QString dbComputationUnitsName, std::vector<Crit1DCompUnit> &unitList, QString &errorStr);


#endif // COMPUTATIONUNITSDB_H
