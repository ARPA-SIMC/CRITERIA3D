#ifndef CROPPINGSYSTEM_H
#define CROPPINGSYSTEM_H

    class CriteriaModel;
    class Crit3DDate;
    class QString;

    #define MAX_EVAPORATION_DEPTH 0.15

    void initializeCrop(CriteriaModel* myCase, int currentDoy);
    bool updateCrop(CriteriaModel* myCase, Crit3DDate myDate,
                    float tmin, float tmax, double waterTableDepth, QString *myError);

    float cropIrrigationDemand(CriteriaModel* myCase, int doy, float myPrec, float nextPrec);
    bool cropWaterDemand(CriteriaModel* myCase);
    bool optimalIrrigation(CriteriaModel* myCase, float myIrrigation);

    bool evaporation(CriteriaModel* myCase);
    double cropTranspiration(CriteriaModel* myCase, bool getWaterStress);

    double getCropReadilyAvailableWater(CriteriaModel* myCase);
    double getTotalReadilyAvailableWater(CriteriaModel* myCase);
    double getCropWaterDeficit(CriteriaModel* myCase);


#endif // CROPPINGSYSTEM_H
