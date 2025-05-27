#ifndef WATER_H
#define WATER_H

    struct TlinkedNode;

    bool waterFlowComputation(double deltaT);
    double getWaterExchange(long index, TlinkedNode *link, double deltaT);
    bool computeWater(double maxTime, double *acceptedTime);
    void restoreWater();

#endif  // WATER_H
