#include <cmath>

#include "cudaFunctions.h"
#include "gpusolver.h"

using namespace soilFluxes3D::New;

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

void GPUSolver::inizialize()
{

}

void GPUSolver::run()
{
    if(_status != Inizialized)
        return;

    run_k(libHandle, iterationMatrix, constantTerm, tempSolution1, tempSolution2);

    _status = Terminated;
}

void GPUSolver::gatherOutput(double*& vecX)
{
    if(_status != Terminated)
        return;

    gatherOutput_k(tempSolution1, vecX);
}
