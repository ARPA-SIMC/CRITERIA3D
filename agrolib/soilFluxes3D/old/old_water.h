#ifndef WATER_H
#define WATER_H

    #include "old_macro.h"

    struct TlinkedNode;

    bool waterFlowComputation_stdTreads(double deltaT);
    double getWaterExchange(long index, TlinkedNode *link, double deltaT);
    bool computeWaterFluxes(double maxTime, double *acceptedTime);
    void restoreWater();

#endif  // WATER_H
