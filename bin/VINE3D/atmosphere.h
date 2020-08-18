#ifndef ATMOSPHERE_H
#define ATMOSPHERE_H

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif

    class Vine3DProject;
    class QDate;
    class QString;

    bool vine3DInterpolationDem(Vine3DProject* myProject, meteoVariable myVar, const Crit3DTime& myTime, bool loadData);

    bool interpolationProjectDemMain(Vine3DProject* myProject, meteoVariable myVar, const Crit3DTime& myTime, bool isLoadData);

    bool interpolateAndSaveHourlyMeteo(Vine3DProject* myProject, meteoVariable myVar,
                            const Crit3DTime& myCrit3DTime, const QString& myOutputPath,
                            bool isSave, const QString& myArea);

    bool loadDailyMeteoMap(Vine3DProject* myProject, meteoVariable myDailyVar, QDate myDate,
                           const QString& myArea);

    void qualityControl(Vine3DProject* myProject, meteoVariable myVar, const Crit3DTime& myCrit3DTime);

    bool checkLackOfData(Vine3DProject* myProject, meteoVariable myVar, Crit3DTime myDateTime, long* nrReplacedData);

#endif // ATMOSPHERE_H
