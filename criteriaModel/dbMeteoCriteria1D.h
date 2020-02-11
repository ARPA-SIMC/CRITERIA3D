#ifndef DBMETEOCRITERIA1D_H
#define DBMETEOCRITERIA1D_H

#include <QSqlDatabase>

    class QSqlQuery;
    class QString;
    class Crit3DMeteoPoint;

    bool openDbMeteo(QString dbName, QSqlDatabase* dbMeteo, QString* error);
    bool getIdMeteoList(QSqlDatabase* dbMeteo, QStringList* idMeteoList, QString* error);
    QString getLatFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString *myError);
    QString getLonFromIdMeteo(QSqlDatabase* dbMeteo, QString idMeteo, QString *myError);
    bool readDailyDataCriteria1D(QSqlQuery *query, Crit3DMeteoPoint *meteoPoint, QString *myError);

#endif // DBMETEOCRITERIA1D_H
