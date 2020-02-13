#ifndef CROPPINGSYSTEM_H
#define CROPPINGSYSTEM_H

    class CriteriaModel;
    class Crit3DDate;
    class Crit3DCrop;
    class QString;

    #define MIN_EMERGENCE_DAYS 7
    #define MAX_EVAPORATION_DEPTH 0.15

    void initializeCrop(CriteriaModel* myCase, int currentDoy);
    void initializeCrop(Crit3DCrop* myCrop, double latitude, int nrLayers, double totalSoilDepth, int currentDoy);

    bool updateCrop(CriteriaModel* myCase, QString *myError, Crit3DDate myDate, float tmin, float tmax, float waterTableDepth);
    bool updateLAI(CriteriaModel* myCase, int myDoy);

    float cropIrrigationDemand(CriteriaModel* myCase, int doy, float myPrec, float nextPrec);
    bool cropWaterDemand(CriteriaModel* myCase);
    bool optimalIrrigation(CriteriaModel* myCase, float myIrrigation);

    bool evaporation(CriteriaModel* myCase);
    double cropTranspiration(CriteriaModel* myCase, bool getWaterStress);

    double getCropReadilyAvailableWater(CriteriaModel* myCase);
    double getTotalReadilyAvailableWater(CriteriaModel* myCase);
    double getCropWaterDeficit(CriteriaModel* myCase);


#endif // CROPPINGSYSTEM_H
