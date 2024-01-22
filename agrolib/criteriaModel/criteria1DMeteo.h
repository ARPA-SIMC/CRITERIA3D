#ifndef DBMETEOCRITERIA1D_H
#define DBMETEOCRITERIA1D_H

    #define MAX_MISSING_TOT_DAYS 30
    #define MAX_MISSING_CONSECUTIVE_DAYS_T 1
    #define MAX_MISSING_CONSECUTIVE_DAYS_PREC 7

    #include <QString>

    class QSqlDatabase;
    class QSqlQuery;
    class QDate;
    class Crit3DMeteoPoint;

    bool openDbMeteo(QString dbName, QSqlDatabase &dbMeteo, QString &error);
    bool getMeteoPointList(const QSqlDatabase &dbMeteo, QList<QString> &idMeteoList, QString &errorStr);
    bool getYearList(QSqlDatabase* dbMeteo, QString table, QList<QString>* yearList, QString *error);
    bool getLatLonFromIdMeteo(QSqlDatabase &dbMeteo, QString idMeteo, QString &lat, QString &lon, QString &errorStr);
    bool updateLatFromIdMeteo(QSqlDatabase &dbMeteo, QString idMeteo, QString lat, QString &error);
    QString getTableNameFromIdMeteo(QSqlDatabase &dbMeteo, QString idMeteo, QString &errorStr);

    bool checkYear(QSqlDatabase* dbMeteo, QString table, QString year, QString *error);
    bool checkYearMeteoGridFixedFields(QSqlDatabase dbMeteo, QString tableD, QString fieldTime, QString fieldTmin, QString fieldTmax, QString fieldPrec, QString year, QString *error);
    bool checkYearMeteoGrid(const QSqlDatabase &dbMeteo, const QString &tableD, const QString &fieldTime,
                            int varCodeTmin, int varCodeTmax, int varCodePrec, const QString &year, QString &error);

    bool getLastDate(QSqlDatabase* dbMeteo, QString table, QString year, QDate* date, QString *error);
    bool getLastDateGrid(QSqlDatabase dbMeteo, QString table, QString fieldTime, QString year, QDate* date, QString *error);

    bool fillDailyTempPrecCriteria1D(QSqlDatabase* dbMeteo, QString table, Crit3DMeteoPoint *meteoPoint, int validYear, QString *error);
    bool readDailyDataCriteria1D(QSqlQuery &query, Crit3DMeteoPoint &meteoPoint, int maxNrDays, QString &error);


#endif // DBMETEOCRITERIA1D_H
