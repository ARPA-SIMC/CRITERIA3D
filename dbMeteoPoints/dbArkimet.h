#ifndef DBARKIMET_H
#define DBARKIMET_H

    #ifndef VARIABLESLIST_H
        #include "variablesList.h"
    #endif
    #ifndef DBMETEOPOINTS_H
        #include "dbMeteoPointsHandler.h"
    #endif

    #ifndef QDATE_H
        #include <QDate>
    #endif


    #define PREC_ID 250
    #define RAD_ID 706

    class DbArkimet : public Crit3DMeteoPointsDbHandler
    {
        public:
            explicit DbArkimet(QString dbName);
            void dbManager();
            QString queryString;

            QString getVarName(int id);
            QList<int> getDailyVar();
            QList<int> getHourlyVar();
            QList<int> getId(QString VarName);
            QList<VariablesList> getVariableProperties(QList<int> id);

            void initStationsDailyTables(const QDate &startDate, const QDate &endDate,
                                         const QList<QString> &stationList, const QList<QString> &idVarList);
            void initStationsHourlyTables(const QDate &startDate, const QDate &endDate,
                                          const QList<QString> &stationList, const QList<QString> &idVarList);

            void createTmpTableHourly();
            void deleteTmpTableHourly();
            bool createTmpTableDaily(QString &errorStr);
            void deleteTmpTableDaily();

            bool saveHourlyData();
            bool saveDailyData();

            void appendQueryHourly(QString dateTime, QString idPoint, QString idVar, QString value, bool isFirstData);
            void appendQueryDaily(QString date, QString idPoint, QString idVar, QString value, bool isFirstData);
    signals:

        protected slots:

    };


#endif // DBARKIMET_H
