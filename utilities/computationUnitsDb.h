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
        double slope;

        Crit1DUnit();
    };

    class ComputationUnitsDB
    {
    public:
        ComputationUnitsDB(QString dbname, QString &error);
        ~ComputationUnitsDB();

        bool writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                   QStringList idSoil, QList<double> hectares, QString &error);

        bool readUnitList(std::vector<Crit1DUnit> &unitList, QString &error);

    private:
        QSqlDatabase db;
    };


    bool readUnitList(QString dbUnitsName, std::vector<Crit1DUnit> &unitList, QString &error);


#endif // COMPUTATIONUNITSDB_H
