#ifndef DESEASE_H
#define DESEASE_H

    #ifndef PROJECT_H
        #include "vine3DProject.h"
    #endif

    bool computePowderyMildew(Vine3DProject* myProject);
    bool computeDownyMildew(Vine3DProject* myProject, QDate firstDate, QDate lastDate, unsigned lastHour);


#endif // DESEASE_H
