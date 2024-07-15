#ifndef DATAHANDLER_H
#define DATAHANDLER_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef PLANT_H
        #include "plant.h"
    #endif

    class QString;

    float getTimeStepFromHourlyInterval(int myHourlyIntervals);

    int getMeteoVarIndex(meteoVariable myVar);

    QString getVarNameFromPlantVariable(plantVariable myVar);

#endif // DATAHANDLER_H
