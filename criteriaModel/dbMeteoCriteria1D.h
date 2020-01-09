#ifndef DBMETEOCRITERIA1D_H
#define DBMETEOCRITERIA1D_H

    class QSqlQuery;
    class QString;
    class Crit3DMeteoPoint;

    bool readDailyDataCriteria1D(QSqlQuery *query, Crit3DMeteoPoint *meteoPoint, QString *myError);

#endif // DBMETEOCRITERIA1D_H
