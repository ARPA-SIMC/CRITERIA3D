#ifndef MODELCORE_H
#define MODELCORE_H

    class Crit3DDate;
    class Crit1DIrrigationForecast;
    class Crit1DCase;
    class Crit1DUnit;
    class QString;
    class Crit3DCrop;
    class Crit3DMeteoPoint;
    class Crit1DOutput;

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #include <vector>

    bool runModel(Crit1DIrrigationForecast* irrForecast, const Crit1DUnit &myUnit, QString *myError);

    bool computeDailyModel(Crit3DDate myDate, Crit1DCase* myCase, std::string *myError);

    bool computeDailyModel(Crit3DDate myDate, Crit3DMeteoPoint* meteoPoint, Crit3DCrop* myCrop,
                           std::vector<soil::Crit3DLayer>* soilLayers, Crit1DOutput* myOutput, bool optimizeIrrigation,
                           std::string *myError);

#endif // MODELCORE_H
