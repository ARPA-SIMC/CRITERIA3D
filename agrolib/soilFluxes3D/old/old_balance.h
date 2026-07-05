#ifndef BALANCE_H
#define BALANCE_H

    #include "old_macro.h"

    struct TlinkedNode;

    void halveTimeStep();
    bool getForcedHalvedTime();
    void setForcedHalvedTime(bool isForced);
    double computeTotalWaterContent();
    double getMatrixValue(long i, TlinkedNode *link);
    void InitializeBalanceWater();
    bool waterBalance(double deltaT, int approxNr);
    void updateBalanceWaterWholePeriod();
    void restoreBestApproximation(double deltaT);

#endif  // BALANCE_H
