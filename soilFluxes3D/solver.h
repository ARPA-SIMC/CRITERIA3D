#ifndef SOLVER_H
#define SOLVER_H

    #ifndef MACRO_H
        #include "macro.h"
    #endif

    __SF3DINLINE double distance(unsigned long index1, unsigned long index2);

    __SF3DINLINE double distance2D(unsigned long index1, unsigned long index2);

    double computeMean(double v1, double v2);

    __SF3DINLINE double arithmeticMean(double v1, double v2);

    bool solveLinearSystem(int approximation, double residualTolerance, int computationType);

#endif  // SOLVER_H

