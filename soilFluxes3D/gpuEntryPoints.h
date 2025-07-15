#ifndef GPUENTRYPOINTS_H
#define GPUENTRYPOINTS_H

#include "../types.h"

void runCUSPARSEiteration(TmatrixElement **A, double* b, double* x, double* &vecSol, uint64_t numNodes);

#endif // GPUENTRYPOINTS_H
