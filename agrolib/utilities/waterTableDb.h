/*!
* \brief water table parameters
*/

#ifndef WATERTABLEDB_H
#define WATERTABLEDB_H

    #include <QString>
    #include <QSqlDatabase>
    #include <vector>

    class QSqlDatabase;

    struct waterTableParameters
    {
        QString id;
        double lat, lon;

        int nrDaysPeriod;           // [days]
        double alpha;               // [-]
        double h0;                  // unit of observed watertable data, usually [cm]
        double avgDailyCWB;         // [mm]
    };

    class WaterTableDb
    {
    public:
        WaterTableDb(QString dbName, QString &error);
        ~WaterTableDb();

        //bool readWaterTableList(std::vector<waterTableParameters> &waterTableList, QString &error);
        bool readSingleWaterTableParameters(const QString &id, waterTableParameters &waterTableList, QString &error);

    private:
        QSqlDatabase _db;
    };


#endif // WATERTABLEDB_H
