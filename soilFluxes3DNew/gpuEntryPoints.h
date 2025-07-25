#ifndef GPUENTRYPOINTS_H
#define GPUENTRYPOINTS_H

#include "../soilFluxes3D/types.h"

void runCUSPARSEiteration(TmatrixElement **A, double* b, double* x, double* &vecSol, uint64_t numNodes);

#endif // GPUENTRYPOINTS_H
