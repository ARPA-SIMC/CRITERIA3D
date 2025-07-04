#include "iterationkernel.h"
#include "kernels.h"

#include <stdlib.h>



extern "C"
    __global__ void updateTempVector(CUDAoptimizedVector<double> *x, double *tempVector, const uint32_t size)
{
    uint32_t globalIdx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(globalIdx >= size)
        return;

    tempVector[globalIdx] = x->values[globalIdx];
    //x->values[globalIdx] = tempVector[globalIdx];
}

extern "C"
    __global__ void JacobiIteration(CUDAoptimizedMatrix<double> *A, CUDAoptimizedVector<double> *x, CUDAoptimizedVector<double> *b, double *tempVector, const uint32_t size)
{
    uint32_t globalIdx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(globalIdx >= size)
        return;

    x->values[globalIdx] = b->values[globalIdx];

    for (uint8_t colIdx = 0; colIdx < A->numCol[globalIdx]; ++colIdx)
    {
        uint32_t linerMatrixPosition = size * colIdx + globalIdx;
        x->values[globalIdx] -= A->elementValues[linerMatrixPosition] * tempVector[A->elementIndeces[linerMatrixPosition]];
    }

}

extern "C"
    __global__ void launchKernelTail(CUDAoptimizedMatrix<double> *A, CUDAoptimizedVector<double> *x, CUDAoptimizedVector<double> *b, double *tempVector, const uint32_t size, const uint32_t nB, const uint32_t nTpB, kernelNameCalls kernelName)
{
    switch(kernelName)
    {
        case JacobiMain:
            JacobiIteration<<<nB, nTpB, 0, cudaStreamTailLaunch>>>(A, x, b, tempVector, size);
            break;
        case updateVectorX:
            updateTempVector<<<nB, nTpB, 0, cudaStreamTailLaunch>>>(x, tempVector, size);
            break;
        default:
            return;

    }
}

extern "C"
    __global__ void mainKernel(CUDAoptimizedMatrix<double> *A, CUDAoptimizedVector<double> *x, CUDAoptimizedVector<double> *b, double *tempVector, const uint32_t numIter, const uint32_t nTpB)
{
    uint32_t systemSize = A->numRow;
    uint32_t numBlocks = (systemSize / nTpB) + 1;

    for(uint32_t iterIdx = 0; iterIdx < numIter; ++iterIdx)
    {
        launchKernelTail<<<1, 1>>>(A, x, b, tempVector, systemSize, numBlocks, nTpB, JacobiMain);
        launchKernelTail<<<1, 1>>>(A, x, b, tempVector, systemSize, numBlocks, nTpB, updateVectorX);
    }
}

extern "C"
cudaError_t launchKernelFunction(CUDAoptimizedMatrix<double> *A, CUDAoptimizedVector<double> *x, CUDAoptimizedVector<double> *b, const uint32_t numIter, const uint32_t nTpB)
{
    uint32_t size = 3;
    double *tempVector;

    cudaCheck(cudaMalloc(&tempVector, size*sizeof(double)));

    cudaMemcpy(tempVector, x, size*sizeof(double), cudaMemcpyDeviceToDevice);

    mainKernel<<<1, 1>>>(A, x, b, tempVector, numIter, nTpB);

    return cudaGetLastError();
}
