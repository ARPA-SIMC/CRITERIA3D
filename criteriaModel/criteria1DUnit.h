#ifndef CRITERIA1DUNIT_H
#define CRITERIA1DUNIT_H

    #include <QString>
    #include <vector>

    /*!
    * \brief computation unit of Criteria1D
    * \note Unit = distinct crop, soil, meteo
    */
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
