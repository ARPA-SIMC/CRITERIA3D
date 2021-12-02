#ifndef DBOUTPUTPOINTSHANDLER_H
#define DBOUTPUTPOINTSHANDLER_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif

    #include <QSqlDatabase>
    #include <QDateTime>

    class Crit3DOutputPointsDbHandler
    {
    public:
        explicit Crit3DOutputPointsDbHandler(QString dbname_);
        ~Crit3DOutputPointsDbHandler();

        void closeDatabase();
        QSqlDatabase getDb() const;
        QString getDbName();
        QString getErrorString();
        bool isOpen();

        bool createTable(QString tableName);
        bool addColumn(QString tableName, meteoVariable myVar);
        bool saveHourlyData(QString tableName, QDateTime myTime,
                            std::vector<meteoVariable> varList, std::vector<float> values);

    private:

        QSqlDatabase _db;
        QString errorString;
    };


#endif // DBOUTPUTPOINTSHANDLER_H
