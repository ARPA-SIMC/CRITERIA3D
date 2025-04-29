/*!
* \brief water table parameters
*/

#ifndef WATERTABLEDB_H
#define WATERTABLEDB_H

    #include <QString>
    #include <QSqlDatabase>
    #include <vector>

    class QSqlDatabase;

    class Crit1DWaterTableParameters
    {
        public:
        Crit1DWaterTableParameters();

        QString id;
        double lat, lon;

        int nrDaysPeriod;           // [days]
        double alpha;               // [-]
        double h0;                  // unit of observed watertable data, usually [cm]
        double avgDailyCWB;         // [mm]

        bool isLoaded;
    };

    class WaterTableDb
    {
    public:
        WaterTableDb(QString dbName, QString &errorString);
        ~WaterTableDb();

        bool readSingleWaterTableParameters(const QString &id, Crit1DWaterTableParameters &wtParameters, QString &error);

    private:
        QSqlDatabase _db;
    };


#endif // WATERTABLEDB_H
