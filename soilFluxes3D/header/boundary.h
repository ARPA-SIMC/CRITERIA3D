#ifndef BOUNDARY_H
#define BOUNDARY_H

    struct Tboundary;

    void updateConductance();
    bool updateBoundaryHeat(double timeStep, double &reducedTimeStep);
    void updateBoundaryWater(double deltaT);
    void initializeBoundary(Tboundary *myBoundary, int myType, float slope, float boundaryArea);

#endif  // BOUNDARY_H


