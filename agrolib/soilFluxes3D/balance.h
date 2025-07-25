#ifndef BALANCE_H
#define BALANCE_H

    #ifndef MACRO_H
        #include "macro.h"
    #endif

    struct TlinkedNode;

    void halveTimeStep();
    bool getForcedHalvedTime();
    void setForcedHalvedTime(bool isForced);
    double computeTotalWaterContent();
    double getMatrixValue(long i, TlinkedNode *link);
    void InitializeBalanceWater();
    bool waterBalance(double deltaT, int approxNr);
    void updateBalanceWaterWholePeriod();

#endif  // BALANCE_H
