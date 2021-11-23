#ifndef SOLVER_H
#define SOLVER_H

    inline double square(double x) {return ((x)*(x));}

    double distance(unsigned long index1, unsigned long index2);

    double distance2D(unsigned long index1, unsigned long index2);

    double computeMean(double v1, double v2);

    double arithmeticMean(double v1, double v2);

    bool GaussSeidelRelaxation (int myApproximation, double myResidualTolerance, int myProcess);

#endif  // SOLVER_H

