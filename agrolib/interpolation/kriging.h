#ifndef KRIGING_H
#define KRIGING_H

    bool matrixInversion(double *A);

    bool krigingVariogram(double *myPos, double *mtVal, int nrItems, short myMode,
                         double myRange, double myNugget, double mySill, double mySlope);

    bool krigingSetWeight(double x_p, double y_p);

    double krigingResult();

    double krigingRMSE();

    void krigingFreeMemory();

#endif // KRIGING_H
