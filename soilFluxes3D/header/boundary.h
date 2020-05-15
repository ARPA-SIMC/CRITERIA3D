#ifndef BOUNDARY_H
#define BOUNDARY_H

    struct Tboundary;

    void updateBoundary();
    void updateBoundaryHeat();
    void updateBoundaryWater(double deltaT);
    void initializeBoundary(Tboundary *myBoundary, int myType, float slope);

#endif  // BOUNDARY_H


