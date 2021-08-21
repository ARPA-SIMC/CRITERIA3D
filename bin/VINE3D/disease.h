#ifndef DESEASE_H
#define DESEASE_H

    #ifndef PROJECT_H
        #include "vine3DProject.h"
    #endif
    #ifndef QDATETIME_H
        #include <QDateTime>
    #endif

    bool computePowderyMildew(Vine3DProject* myProject);
    bool computeDownyMildew(Vine3DProject* myProject, QDate firstDate, QDate lastDate, unsigned lastHour, QString myArea);

#endif // DESEASE_H
