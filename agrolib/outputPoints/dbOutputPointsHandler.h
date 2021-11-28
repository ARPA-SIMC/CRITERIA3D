#ifndef DBOUTPUTPOINTSHANDLER_H
#define DBOUTPUTPOINTSHANDLER_H

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    class Crit3DOutputPointsDbHandler
    {
    public:
        explicit Crit3DOutputPointsDbHandler(QString dbname_);
        ~Crit3DOutputPointsDbHandler();

        void closeDatabase();
        QSqlDatabase getDb() const;
        QString getDbName();
        QString getLastError();

    protected:

        QSqlDatabase _db;
    };


#endif // DBOUTPUTPOINTSHANDLER_H
