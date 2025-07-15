#ifndef SOILFLUXES3D_CUDAFUNCTIONS_H
#define SOILFLUXES3D_CUDAFUNCTIONS_H

#include "types_gpu.h"

using namespace soilFluxes3D::New;

void runCUSPARSEtest(double* &vecSol);

void copyMatrixVectorFromOld_k(MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2, TmatrixElement **matA, double *vecB, double *vecX, uint64_t numNodes);
void run_k(cusparseHandle_t libHandle, MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2);
void gatherOutput_k(VectorGPU &tempSolution1, double* &vecX);


void destructCUSPARSEobjects(cusparseHandle_t &libHandle, MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2);

#endif // SOILFLUXES3D_CUDAFUNCTIONS_H
