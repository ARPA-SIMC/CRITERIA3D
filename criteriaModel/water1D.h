#ifndef WATER1D_H
#define WATER1D_H

    #include <vector>

    #ifndef SOIL_H
        #include "soil.h"
    #endif

    #define MAX_EVAPORATION_DEPTH 0.15

    class CriteriaModel;

    void initializeWater(CriteriaModel* myCase);
    bool computeInfiltration(CriteriaModel* myCase, double prec, double sprayIrrigation);
    bool computeEvaporation(CriteriaModel* myCase);
    bool computeSurfaceRunoff(CriteriaModel* myCase);
    bool computeLateralDrainage(CriteriaModel* myCase);

    double computeOptimalIrrigation(std::vector<soil::Crit3DLayer>* soilLayers, double myIrrigation);
    double computeCapillaryRise(std::vector<soil::Crit3DLayer> *soilLayers, double waterTableDepth);

    double getSoilWaterContent(CriteriaModel* myCase);


#endif // WATER1D_H

