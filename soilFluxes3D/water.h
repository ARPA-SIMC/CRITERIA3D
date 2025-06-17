#ifndef WATER_H
#define WATER_H

    #ifndef MACRO_H
        #include "macro.h"
    #endif

    struct TlinkedNode;

    bool waterFlowComputation_stdTreads(double deltaT);
    double getWaterExchange(long index, TlinkedNode *link, double deltaT);
    bool computeWaterFluxes(double maxTime, double *acceptedTime);
    __SF3DINLINE void restoreWater();

#endif  // WATER_H
