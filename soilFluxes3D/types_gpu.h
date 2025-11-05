#pragma once

#define NOMINMAX

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cub/device/device_reduce.cuh>

#include <cstdint>
#include "types.h"

namespace soilFluxes3D::v2
{
    using i64_t = std::int64_t;

    struct MatrixGPU
    {
        cusparseSpMatDescr_t cusparseDescriptor;

        i64_t numRows;
        i64_t numCols;
        uint16_t* d_numColsInRow = nullptr;
        i64_t numNonZeroElement;
        i64_t totValuesSize;
        i64_t sliceSize;
        i64_t* d_offsets = nullptr;
        i64_t* d_columnIndeces = nullptr;
        double* d_values = nullptr;
        double* d_diagonalValues = nullptr;
        const cusparseIndexType_t offsetType = CUSPARSE_INDEX_64I;
        const cusparseIndexType_t colIdxType = CUSPARSE_INDEX_64I;
        const cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        const cudaDataType valueType = CUDA_R_64F;
    };

    struct VectorGPU
    {
        cusparseDnVecDescr_t cusparseDescriptor;
        i64_t numElements;
        double *d_values = nullptr;
        const cudaDataType valueType = CUDA_R_64F;
    };

    template<typename T>
    inline cudaError_t allocDevicePointer(T*& ptr, const std::size_t count)
    {
        if(ptr != nullptr)
            return cudaErrorIllegalAddress;

        cudaError_t err;

        err = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
        if(err != cudaSuccess)
            return err;

        err = cudaMemset(ptr, 0, count * sizeof(T));
        if(err != cudaSuccess)
            return err;

        return err;
    }

    template<typename T>
    __global__ void fillDevicePointer(T*& ptr, const std::size_t count, const T value)
    {
        if(ptr == nullptr)
            return;

        SF3Duint_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(idx >= count)
            return;

        ptr[idx] = value;
        return;
    }

    template<typename T>
    inline cudaError_t resetDevicePointer(T*& ptr, const std::size_t count)
    {
        if(ptr == nullptr)
            return cudaErrorIllegalAddress;

        cudaError_t err = cudaMemset(ptr, 0, count * sizeof(T));
        return err;
    }

    template<typename T>
    inline cudaError_t freeDevicePointer(T*& ptr)
    {
        if(ptr == nullptr)
            return cudaSuccess;

        cudaError_t err = cudaFree(ptr);
        if(err != cudaSuccess)
            return err;

        ptr = nullptr;
        return err;
    }

    template<typename T, typename Func>
    __global__ void conditionalCopyDevicePointer(T*& dst, const T*& src, const std::size_t count, Func cond)
    {
        if(src == nullptr || dst == nullptr)
            return;

        SF3Duint_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(idx >= count)
            return;

        if(cond(idx))
            dst[idx] = src[idx];

        return;
    }


    template <typename T>
    inline void movePointerToDevice(T*& ptr, const std::size_t count, const cudaStream_t& stream)
    {
        T* tmp = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&tmp), count * sizeof(T));
        cudaMemcpyAsync(tmp, ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        std::free(ptr);
        ptr = tmp;
    }

    template <typename T>
    inline void movePointerToHost(T*& ptr, const std::size_t count, const cudaStream_t& stream)
    {
        T* tmp = reinterpret_cast<T*>(std::calloc(count, sizeof(T)));
        cudaMemcpyAsync(tmp, ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaFree(ptr);
        ptr = tmp;
    }

    template <typename KernelFunc, typename... Args>
    inline cudaError_t launchGPUKernel(KernelFunc kernel, dim3 numBlocks, dim3 numThreadsPerBlock, Args&&... args)
    {
        kernel<<<numBlocks, numThreadsPerBlock>>>(std::forward<Args>(args)...);

        cudaError_t err = cudaDeviceSynchronize();
        return err;
    }
}
