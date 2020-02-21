#ifndef CROPPINGSYSTEM_H
#define CROPPINGSYSTEM_H

    class CriteriaModel;
    class Crit3DDate;
    class QString;

    #define MAX_EVAPORATION_DEPTH 0.15

    void initializeCrop(CriteriaModel* myCase, int currentDoy);
    bool updateCrop(CriteriaModel* myCase, Crit3DDate myDate,
                    float tmin, float tmax, double waterTableDepth, QString *myError);

    bool optimalIrrigation(CriteriaModel* myCase, float myIrrigation);

    bool computeEvaporation(CriteriaModel* myCase);

    double getCropReadilyAvailableWater(CriteriaModel* myCase);
    double getSoilWaterDeficit(CriteriaModel* myCase);


#endif // CROPPINGSYSTEM_H
