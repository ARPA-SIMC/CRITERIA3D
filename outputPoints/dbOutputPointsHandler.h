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

        bool createTable(QString tableName, QString &errorStr);
        bool addColumn(QString tableName, meteoVariable myVar, QString &errorString);
        bool addCriteria3DColumn(const QString &tableName, criteria3DVariable myVar, int depth, QString& errorStr);

        bool saveHourlyMeteoData(QString tableName, const QDateTime &myTime,
                            const std::vector<meteoVariable> &varList,
                            const std::vector<float> &values, QString& errorStr);

    private:

        QSqlDatabase _db;
        QString errorString;
    };


#endif // DBOUTPUTPOINTSHANDLER_H
