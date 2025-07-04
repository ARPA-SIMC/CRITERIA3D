#ifndef KERNELS_H
#define KERNELS_H

#include "dataTypes.h"
#include <cuda.h>
#include <cuda_runtime.h>

enum kernelNameCalls {JacobiMain, updateVectorX};

extern "C"
    __global__ void mainKernel(CUDAoptimizedMatrix<double>*, CUDAoptimizedVector<double>*, CUDAoptimizedVector<double>*, double*, const uint32_t, const uint32_t);

extern "C"
    __global__ void updateTempVector(CUDAoptimizedVector<double>*, double*, const uint32_t);

extern "C"
    __global__ void JacobiIteration(CUDAoptimizedMatrix<double>*, CUDAoptimizedVector<double>*, CUDAoptimizedVector<double>*, double*, const uint32_t);

extern "C"
    __global__ void launchKernelTail(CUDAoptimizedMatrix<double>*, CUDAoptimizedVector<double>*, CUDAoptimizedVector<double>*, double*, const uint32_t, const uint32_t, const uint32_t, kernelNameCalls);


#endif // KERNELS_H
