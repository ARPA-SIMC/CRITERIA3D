#ifndef BALANCE_H
#define BALANCE_H

    #ifndef MACRO_H
        #include "soilFluxes3D_new/macro.h"
    #endif

    struct TlinkedNode;

    __SF3DINLINE void halveTimeStep();
    __SF3DINLINE bool getForcedHalvedTime();
    __SF3DINLINE void setForcedHalvedTime(bool isForced);
    double computeTotalWaterContent();
    double getMatrixValue(long i, TlinkedNode *link);
    void InitializeBalanceWater();
    bool waterBalance(double deltaT, int approxNr);
    void updateBalanceWaterWholePeriod();

#endif  // BALANCE_H
