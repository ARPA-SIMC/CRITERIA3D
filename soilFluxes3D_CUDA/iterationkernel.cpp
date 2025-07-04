#include "iterationkernel.h"

#include <iostream>

using namespace soilFluxes3D_CUDA;

cudaError_t iterationKernel::launchKernel()
{
    if(currentStatus != kernelInizialized)
        return cudaErrorIllegalInstruction;

    cudaCheck(launchKernelFunction(d_iterationMatrix, d_unknownVector, d_constantTerm, numIteration, numThreadperBlock));
    currentStatus = kernelLaunched;
    return cudaSuccess;
}

cudaError_t iterationKernel::uploadData()
{
    uint32_t systemSize = h_iterationMatrix->numRow;
    uint32_t totalNumberofElemets = systemSize * h_iterationMatrix->maxNumCols;

    CUDAoptimizedMatrix<double> *tempMatrix;
    CUDAoptimizedVector<double> *tempVector;

    tempMatrix = new CUDAoptimizedMatrix<double>();
    tempMatrix->numRow = systemSize;

    cudaCheck(cudaMalloc(&(tempMatrix->numCol), systemSize * sizeof(uint32_t)));
    cudaCheck(cudaMemcpy(tempMatrix->numCol, h_iterationMatrix->numCol, systemSize * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&(tempMatrix->elementIndeces), totalNumberofElemets * sizeof(uint32_t)));
    cudaCheck(cudaMemcpy(tempMatrix->elementIndeces, h_iterationMatrix->elementIndeces, totalNumberofElemets * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&(tempMatrix->elementValues), totalNumberofElemets * sizeof(double)));
    cudaCheck(cudaMemcpy(tempMatrix->elementValues, h_iterationMatrix->elementValues, totalNumberofElemets * sizeof(double), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&(d_iterationMatrix), sizeof(CUDAoptimizedMatrix<double>)));
    cudaCheck(cudaMemcpy(d_iterationMatrix, tempMatrix, sizeof(CUDAoptimizedMatrix<double>), cudaMemcpyHostToDevice));

    //delete(tempMatrix);

    tempVector = new CUDAoptimizedVector<double>();
    tempVector->numElement = systemSize;
    cudaCheck(cudaMalloc(&(tempVector->values), systemSize * sizeof(double)));
    cudaCheck(cudaMemcpy(tempVector->values, h_constantTerm->values, systemSize * sizeof(double), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&(d_constantTerm), sizeof(CUDAoptimizedVector<double>)));
    cudaCheck(cudaMemcpy(d_constantTerm, tempVector, sizeof(CUDAoptimizedVector<double>), cudaMemcpyHostToDevice));

    //delete(tempVector);

    tempVector = new CUDAoptimizedVector<double>();
    tempVector->numElement = systemSize;
    cudaCheck(cudaMalloc(&(tempVector->values), systemSize * sizeof(double)));
    cudaCheck(cudaMemcpy(tempVector->values, h_unknownVector->values, systemSize * sizeof(double), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&(d_unknownVector), sizeof(CUDAoptimizedVector<double>)));
    cudaCheck(cudaMemcpy(d_unknownVector, tempVector, sizeof(CUDAoptimizedVector<double>), cudaMemcpyHostToDevice));

    //delete(tempVector);

    currentStatus = kernelInizialized;
    return cudaSuccess;
}

cudaError_t iterationKernel::downloadData()
{
    cudaDeviceSynchronize();

    CUDAoptimizedVector<double> *tempVector = new CUDAoptimizedVector<double>();

    cudaCheck(cudaMemcpy(tempVector, d_unknownVector, sizeof(CUDAoptimizedVector<double>), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_unknownVector->values, tempVector->values, h_unknownVector->numElement*sizeof(double), cudaMemcpyDeviceToHost));

    return cudaSuccess;
}
