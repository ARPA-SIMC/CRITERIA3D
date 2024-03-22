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

        QString getDbName() {
            return _db.databaseName(); }

        QString getErrorString() {
            return errorString; }

        bool isOpen() {
            return _db.isOpen(); }

        bool createTable(const QString &tableName, QString &errorStr);

        bool addColumn(const QString &tableName, meteoVariable myVar, QString &errorString);

        bool addCriteria3DColumn(const QString &tableName, criteria3DVariable myVar, int depth, QString &errorStr);

        bool saveHourlyMeteoData(const QString &tableName, const QDateTime &myTime,
                                const std::vector<meteoVariable> &varList,
                                const std::vector<float> &values, QString &errorStr);

        bool saveHourlyCriteria3D_Data(const QString &tableName, const QDateTime &myTime,
                                        const std::vector<criteria3DVariable> &varList,
                                        const std::vector<float> &values,
                                        const std::vector <double> &layerDepth, QString &errorStr);

    private:
        QSqlDatabase _db;
        QString errorString;
    };


#endif // DBOUTPUTPOINTSHANDLER_H
