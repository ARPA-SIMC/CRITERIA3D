#ifndef SOILFLUXES3DCUDA_ITERATIONKERNEL_INL_H
#define SOILFLUXES3DCUDA_ITERATIONKERNEL_INL_H

#include "iterationkernel.h"
#include <iostream>

using namespace soilFluxes3D_CUDA;

template<class T>
cudaError_t iterationKernel::inizializeKernelData(std::vector<std::vector<T>> matrix, std::vector<T> vector)
{
    uint32_t systemSize = vector.size();

    if(systemSize == 0)
        return cudaErrorInvalidValue;

    if(matrix.size() != systemSize)
        return cudaErrorInvalidValue;

    T** matrixRaw = (T**) calloc(systemSize, sizeof(T*));
    uint32_t rowIdx = 0, colIdx = 0;
    for (auto row : matrix)
    {
        if(row.size() != systemSize)
            return cudaErrorInvalidValue; //need to free the pointers

        matrixRaw[rowIdx] = (T*) calloc(systemSize, sizeof(T));
        colIdx = 0;
        for (auto element : row)
        {
            matrixRaw[rowIdx][colIdx] = element;
            colIdx++;
        }
        rowIdx++;
    }

    T* vectorRaw = (T*) calloc(systemSize, sizeof(T));
    uint32_t elemIdx = 0;
    for (auto element : vector)
    {
        vectorRaw[elemIdx] = element;
        ++elemIdx;
    }

    return iterationKernel::inizializeKernelData<T>(matrixRaw, vectorRaw, systemSize);
}

template<class T>
cudaError_t iterationKernel::inizializeKernelData(T **matrixRaw, T *vectorRaw, const uint32_t systemSize)
{
    if(systemSize == 0)
        return cudaErrorInvalidValue;

    h_iterationMatrix = new CUDAoptimizedMatrix<double>();
    h_iterationMatrix->numRow = systemSize;
    h_iterationMatrix->numCol = (uint32_t*) calloc(systemSize, sizeof(uint32_t));
    h_iterationMatrix->elementIndeces = (uint32_t*) calloc(systemSize*h_iterationMatrix->maxNumCols, sizeof(uint32_t));
    h_iterationMatrix->elementValues = (double*) calloc(systemSize*h_iterationMatrix->maxNumCols, sizeof(double));
    for(uint32_t row = 0; row < systemSize; ++row)
    {
        uint32_t numColCurrRow = 0;
        for (uint32_t col = 0; col < systemSize; ++col)
        {
            if (matrixRaw[row][col] == 0)
                continue;

            uint32_t currIdx = numColCurrRow * systemSize + row;

            h_iterationMatrix->elementValues[currIdx] = (double) matrixRaw[row][col];
            h_iterationMatrix->elementIndeces[currIdx] = col;
            numColCurrRow++;
        }
        if(numColCurrRow > MAXNUMCOLS)
            return cudaErrorInvalidValue;

        h_iterationMatrix->numCol[row] = numColCurrRow;
    }

    h_constantTerm = new CUDAoptimizedVector<double>();
    h_constantTerm->numElement = systemSize;
    h_constantTerm->values = (double*) calloc(systemSize, sizeof(double));
    for(uint32_t idx = 0; idx < systemSize; ++idx)
        h_constantTerm->values[idx] = (double) vectorRaw[idx];

    h_unknownVector = new CUDAoptimizedVector<double>();
    h_unknownVector->numElement = systemSize;
    h_unknownVector->values = (double*) calloc(systemSize, sizeof(double));
    for(uint32_t idx = 0; idx < systemSize; ++idx)
        h_constantTerm->values[idx] = (double) vectorRaw[idx];

    return uploadData();
}


template<class T>
cudaError_t iterationKernel::getKernelOutput(T *&vectorRaw)
{
    cudaCheck(downloadData());

    uint32_t size = 3; //h_unknownVector->numElement;
    destrucHostPointer(vectorRaw);

    vectorRaw = (T*) malloc(size * sizeof(T));
    for (uint32_t idx = 0; idx < size; ++idx)
        vectorRaw[idx] = h_unknownVector->values[idx];

    return cudaSuccess;
}

#endif // SOILFLUXES3DCUDA_ITERATIONKERNEL_INL_H
