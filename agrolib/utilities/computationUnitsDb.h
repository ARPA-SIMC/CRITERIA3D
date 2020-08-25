#ifndef COMPUTATIONUNITSDB_H
#define COMPUTATIONUNITSDB_H

    #include <QString>
    #include <QSqlDatabase>
    #include <vector>

    /*!
    * \brief computation unit of Criteria1D
    * \note Unit = distinct crop, soil, meteo
    */
    class Crit1DUnit
    {
    public:
        QString idCase;
        QString idCrop;
        QString idSoil;
        QString idMeteo;
        QString idForecast;
        QString idCropClass;
        int idCropNumber;
        int idSoilNumber;

        Crit1DUnit();
    };

    class ComputationUnitsDB
    {
    public:
        ComputationUnitsDB(QString dbname, QString &error);
        ~ComputationUnitsDB();

        bool writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                   QStringList idSoil, QList<double> ha, QString &error);

        bool readUnitList(std::vector<Crit1DUnit> &unitList, QString &error);

    private:
        QSqlDatabase db;
    };


#endif // COMPUTATIONUNITSDB_H
