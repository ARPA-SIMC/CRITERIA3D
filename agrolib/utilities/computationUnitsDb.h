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

        bool writeListToCompUnitsTable(QList<QString> &idCase, QList<QString> &idCrop, QList<QString> &idMeteo,
                                   QList<QString> &idSoil, QList<double> &hectares, QString &error);

        bool readComputationUnitList(std::vector<Crit1DCompUnit> &unitList, QString &error);

    private:
        QSqlDatabase _db;
    };

    bool readComputationUnitList(QString dbComputationUnitsName, std::vector<Crit1DCompUnit> &unitList, QString &error);


#endif // COMPUTATIONUNITSDB_H
