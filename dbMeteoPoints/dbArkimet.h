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

            QString queryString;

            QString getVarName(int id);
            QList<int> getDailyVar();
            QList<int> getHourlyVar();
            QList<int> getId(QString VarName);
            QList<VariablesList> getVariableProperties(QList<int> id);
            QList<VariablesList> getAllVariableProperties();

            void initStationsDailyTables(const QDate &startDate, const QDate &endDate,
                                         const QList<QString> &stationList, const QList<QString> &idVarList);
            void initStationsHourlyTables(const QDate &startDate, const QDate &endDate,
                                          const QList<QString> &stationList, const QList<QString> &idVarList);

            void deleteTmpTable();
            bool createTmpTable();

            void appendTmpData(const QString &dateTime, const QString &idPoint, const QString &idVar, const QString &value, bool isFirstData);

            bool saveHourlyData();
            bool saveDailyData();

            bool readVmDataDaily(const QString &vmFileName, bool isPrec0024, QString &errorString);

    signals:

        protected slots:

    };


#endif // DBARKIMET_H
