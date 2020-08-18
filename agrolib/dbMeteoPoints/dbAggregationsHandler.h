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
        QSqlDatabase db() const;
        QString error() const;

        bool writeAggregationZonesTable(QString name, QString filename, QString field);
        bool getAggregationZonesReference(QString name, QString* filename, QString* field);
        void initAggregatedTables(int numZones, QString aggrType, QString periodType, QDateTime startDate, QDateTime endDate);
        bool saveAggrData(int nZones, QString aggrType, QString periodType, QDateTime startDate, QDateTime endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues);
        void createTmpAggrTable();
        void deleteTmpAggrTable();
        bool insertTmpAggr(QDateTime startDate, QDateTime endDate, meteoVariable variable, std::vector< std::vector<float> > aggregatedValues, int nZones);
        bool saveTmpAggrData(QString aggrType, QString periodType, int nZones);
        std::vector<float> getAggrData(QString aggrType, QString periodType, int zone, QDateTime startDate, QDateTime endDate, meteoVariable variable);
        std::map<int, meteoVariable> mapIdMeteoVar() const;
        bool loadVariableProperties();
        int getIdfromMeteoVar(meteoVariable meteoVar);

    private:
        QSqlDatabase _db;
        std::map<int, meteoVariable> _mapIdMeteoVar;
        QString _error;
    };

#endif // DBAGGREGATIONSHANDLER_H

