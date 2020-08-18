#ifndef MODELCORE_H
#define MODELCORE_H

    #ifndef QSTRING_H
        #include <QString>
    #endif
    #ifndef QDATETIME_H
        #include <QDateTime>
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif

    class Vine3DProject;
    struct TstatePlant;

    bool assignIrrigation(Vine3DProject* myProject, Crit3DTime myTime);

    bool modelDailyCycle(bool isInitialState, Crit3DDate myDate, int nrHours, Vine3DProject* myProject,
                         const QString& myOutputPath, bool isSave, const QString& myArea);

    bool modelCycleOld(QDateTime dStart, QDateTime dStop,  int secondsPerStep, int indexPoint, TstatePlant* statePlant);

#endif // MODELCORE_H
