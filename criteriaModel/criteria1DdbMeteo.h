#ifndef DBMETEOCRITERIA1D_H
#define DBMETEOCRITERIA1D_H

    class QSqlDatabase;
    class QSqlQuery;
    class QStringList;
    class QString;
    class Crit3DMeteoPoint;

    bool openDbMeteo(QString dbName, QSqlDatabase* dbMeteo, QString* error);
    bool getMeteoPointList(QSqlDatabase* dbMeteo, QStringList* idMeteoList, QString* error);
    bool getYearList(QSqlDatabase* dbMeteo, QString table, QStringList* yearList, QString *error);
    bool getLatLonFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString* lat, QString* lon, QString *error);
    bool updateLatLonFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString lat, QString lon, QString *error);
    bool updateLatFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString lat, QString *error);
    QString getTableNameFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString *error);

    bool checkYear(QSqlDatabase* dbMeteo, QString table, QString year, QString *error);

    bool fillDailyTempCriteria1D(QSqlDatabase* dbMeteo, QString table, Crit3DMeteoPoint *meteoPoint, QString validYear, QString *error);
    bool readDailyDataCriteria1D(QSqlQuery *query, Crit3DMeteoPoint *meteoPoint, QString *myError);


#endif // DBMETEOCRITERIA1D_H
