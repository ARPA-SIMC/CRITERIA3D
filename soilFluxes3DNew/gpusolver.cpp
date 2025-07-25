#include <cmath>

#include "cudaFunctions.h"
#include "gpusolver.h"

namespace soilFluxes3D::New
{
void GPUSolver::copyMatrixVectorFromOld(TmatrixElement **matA, double *vecB, double *vecX, uint64_t numNodes)
{
    if(_status != Created)
        return;

    copyMatrixVectorFromOld_k(iterationMatrix, constantTerm, tempSolution1, tempSolution2, matA, vecB, vecX, numNodes);

    _status = Inizialized;
}

GPUSolver::GPUSolver(numericalMethod method)
{
    cusparseCreate(&libHandle);
    _type = GPU;
    _method = method;
    _status = Created;
}

GPUSolver::~GPUSolver()
{
    destructCUSPARSEobjects(libHandle, iterationMatrix, constantTerm, tempSolution1, tempSolution2);
}

SF3Derror_t GPUSolver::inizialize()
{
    return SF3Dok;
}

SF3Derror_t GPUSolver::run(double maxTimeStep, double &acceptedTimeStep, processType process)
{
    if(_status != Inizialized)
        return SolverError;

    run_k(libHandle, iterationMatrix, constantTerm, tempSolution1, tempSolution2);

    _status = Terminated;
    return SF3Dok;
}

SF3Derror_t GPUSolver::gatherOutput(double*& vecX)
{
    if(_status != Terminated)
        return SolverError;

    gatherOutput_k(tempSolution1, vecX);
    return SF3Dok;
}
}
