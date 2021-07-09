/*!
* \brief computation unit for CRITERIA-1D
* \note Unit = distinct combination of crop, soil and meteo
*/

#ifndef COMPUTATIONUNITSDB_H
#define COMPUTATIONUNITSDB_H

    #include <QString>
    #include <QSqlDatabase>
    #include <vector>

    class Crit1DUnit
    {
    public:
        QString idCase;
        QString idCropClass;
        QString idCrop;

        QString idMeteo;
        QString idForecast;

        QString idSoil;
        int idSoilNumber;

        bool isNumericalInfiltration;
        bool isGeometricLayers;
        bool isOptimalIrrigation;
        bool useWaterTableData;
        bool useWaterRetentionData;
        double slope;                           // [m m-1]

        Crit1DUnit();
    };

    class ComputationUnitsDB
    {
    public:
        ComputationUnitsDB(QString dbname, QString &error);
        ~ComputationUnitsDB();

        bool writeListToUnitsTable(QList<QString> idCase, QList<QString> idCrop, QList<QString> idMeteo,
                                   QList<QString> idSoil, QList<double> hectares, QString &error);

        bool readUnitList(std::vector<Crit1DUnit> &unitList, QString &error);

    private:
        QSqlDatabase db;
    };


    bool readUnitList(QString dbUnitsName, std::vector<Crit1DUnit> &unitList, QString &error);


#endif // COMPUTATIONUNITSDB_H
