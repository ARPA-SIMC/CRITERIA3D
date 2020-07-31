#ifndef CRITERIA1DUNIT_H
#define CRITERIA1DUNIT_H

    #ifndef QSTRING_H
        #include <QString>
    #endif
    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    #include <vector>

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
    };


    bool loadUnitList(QString dbUnitsName, std::vector<Crit1DUnit> &unitList, QString &myError);


#endif // CRITERIA1DUNIT_H
