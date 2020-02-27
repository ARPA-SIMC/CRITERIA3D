#ifndef MODELCORE_H
#define MODELCORE_H

    class Crit3DDate;
    class CriteriaModel;
    class CriteriaUnit;
    class QString;
    class Crit3DCrop;
    class Crit3DMeteoPoint;
    class CriteriaModelOutput;

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #include <vector>

    bool runModel(CriteriaModel* myCase, CriteriaUnit *myUnit, QString *myError);

    bool computeDailyModel(Crit3DDate myDate, Crit3DMeteoPoint* meteoPoint, Crit3DCrop* myCrop,
                           std::vector<soil::Crit3DLayer>* soilLayers, CriteriaModelOutput* myOutput, bool optimizeIrrigation,
                           std::string *myError);

    double getCropReadilyAvailableWater(CriteriaModel* myCase);

    double getSoilWaterDeficit(CriteriaModel* myCase);

    //bool computeModel(CriteriaModel* myCase, const Crit3DDate& firstDate, const Crit3DDate& lastDate, QString *myError);


#endif // MODELCORE_H
