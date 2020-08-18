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

            void initStationsDailyTables(QDate startDate, QDate endDate, QStringList stations);
            void initStationsHourlyTables(QDate startDate, QDate endDate, QStringList stations);

            void createTmpTableHourly();
            void deleteTmpTableHourly();
            void createTmpTableDaily();
            void deleteTmpTableDaily();

            bool saveHourlyData();
            bool saveDailyData(QDate startDate, QDate endDate);

            void appendQueryHourly(QString dateTime, QString idPoint, QString idVar, QString value, bool isFirstData);
            void appendQueryDaily(QString date, QString idPoint, QString idVar, QString value, bool isFirstData);
    signals:

        protected slots:

    };


#endif // DBARKIMET_H
