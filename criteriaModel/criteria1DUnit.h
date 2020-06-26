#ifndef CRITERIA1DUNIT_H
#define CRITERIA1DUNIT_H

    #ifndef QSTRING_H
        #include <QString>
    #endif
    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    class Crit1DUnit
    {
    public:
        QString idCase;
        QString idCrop;
        QString idSoil;
        QString idMeteo;
        QString idForecast;
        QString idCropClass;
        int idCropNumber;
        int idSoilNumber;

        Crit1DUnit();

        bool load(QSqlDatabase* dbUnits, QString idCase, QString *error);
    };

    bool openDbUnits(QString dbName, QSqlDatabase* dbUnits, QString* error);


#endif // CRITERIA1DUNIT_H
