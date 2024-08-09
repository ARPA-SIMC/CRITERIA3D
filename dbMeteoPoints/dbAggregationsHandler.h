#ifndef DBAGGREGATIONSHANDLER_H
#define DBAGGREGATIONSHANDLER_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
    #ifndef QDATETIME_H
        #include <QDateTime>
    #endif
    #ifndef _VECTOR_
        #include <vector>
    #endif

    class Crit3DAggregationsDbHandler
    {
    public:
        Crit3DAggregationsDbHandler(QString dbname);
        ~Crit3DAggregationsDbHandler();

        QString error() const { return _error; }
        QString name() const { return _db.databaseName(); }
        QSqlDatabase db() const { return _db; }

        std::map<int, meteoVariable> mapIdMeteoVar() const { return _mapIdMeteoVar; }

        bool existIdPoint(const QString& idPoint);
        bool writeAggregationZonesTable(QString name, QString filename, QString field);
        bool getAggregationZonesReference(QString name, QString* filename, QString* field);
        void initAggregatedTables(int numZones, QString aggrType, QString periodType, QDate startDate, QDate endDate, meteoVariable variable);

        bool writeAggregationPointProperties(int nrPoints, QString aggrType, std::vector <double> lonVector, std::vector <double> latVector);

        bool saveAggrData(int nZones, QString aggrType, QString periodType, QDate startDate, QDate endDate,
                          meteoVariable variable, std::vector< std::vector<float> > aggregatedValues);

        bool insertTmpAggr(QDate startDate, QDate endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues, int nZones);
        bool saveTmpAggrData(QString aggrType, QString periodType, int nZones);

        std::vector<float> getAggrData(QString aggrType, QString periodType, int zone, QDate startDate, QDate endDate, meteoVariable variable);

        bool loadVariableProperties();
        int getIdfromMeteoVar(meteoVariable meteoVar);

        QList<QString> getAggregations();

        bool writeRasterName(QString rasterName);
        bool getRasterName(QString* rasterName);
        bool renameColumn(QString oldColumn, QString newColumn);
        bool writeDroughtDataList(QList<QString> listEntries, QString* log);

    private:
        QSqlDatabase _db;
        std::map<int, meteoVariable> _mapIdMeteoVar;
        QString _error;

        void createTmpAggrTable();
        void deleteTmpAggrTable();
    };

#endif // DBAGGREGATIONSHANDLER_H

