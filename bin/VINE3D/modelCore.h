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

    QString grapevineError(Crit3DTime myTime, long row, long col, QString errorIni);

    bool modelDailyCycle(bool isInitialState, Crit3DDate myDate, int nrHours, Vine3DProject* myProject,
                         const QString& myOutputPath, bool isSave);

#endif // MODELCORE_H
