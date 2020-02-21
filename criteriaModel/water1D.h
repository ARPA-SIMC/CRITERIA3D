#ifndef WATER1D_H
#define WATER1D_H

    class CriteriaModel;

    void initializeWater(CriteriaModel* myCase);
    bool computeInfiltration(CriteriaModel* myCase, double prec, double sprayIrrigation);
    bool computeSurfaceRunoff(CriteriaModel* myCase);
    bool computeLateralDrainage(CriteriaModel* myCase);
    bool computeCapillaryRise(CriteriaModel* myCase, double waterTableDepth);

    double getSoilWaterContent(CriteriaModel* myCase);


#endif // WATER1D_H

