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

    template <typename KernelFunc, typename... Args>
    inline cudaError_t launchGPUKernel(KernelFunc kernel, dim3 numBlocks, dim3 numThreadsPerBlock, Args&&... args)
    {
        kernel<<<numBlocks, numThreadsPerBlock>>>(std::forward<Args>(args)...);

        cudaError_t err = cudaDeviceSynchronize();
        return err;
    }

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
    __global__ void fillDevicePointer_k(T* ptr, const std::size_t count, const T value)
    {
        SF3Duint_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(idx >= count)
            return;

        ptr[idx] = value;
        return;
    }

    template<typename T>
    cudaError_t fillDevicePointer(T* ptr, const std::size_t count, const T value)
    {
        if(ptr == nullptr)
            return cudaErrorIllegalAddress;

        std::size_t nTpB = SF3Dmin(count, 64);
        std::size_t nB = static_cast<std::size_t>(std::ceil((double) count / nTpB));

        return launchGPUKernel(fillDevicePointer_k<T>, dim3(nB), dim3(nTpB), ptr, count, value);
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

    enum class condition : u8_t {notSurface};

    template<condition cond>
    struct ConditionWrapper;

    template<>
    struct ConditionWrapper<condition::notSurface>
    {
        bool* isSurfPtr = nullptr;
        ConditionWrapper(bool* ptr) : isSurfPtr(ptr) {}
        __cudaSpec bool operator() (std::size_t idx) const {return isSurfPtr[idx];}
    };

    template<typename T, class CW>
    __global__ void conditionalCopyDevicePointer_k(T* dst, const T* src, const std::size_t count, const CW& cond)
    {
        SF3Duint_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(idx >= count)
            return;

        if(cond(idx))
            dst[idx] = src[idx];

        return;
    }

    template<typename T, class CW>
    cudaError_t conditionalCopyDevicePointer(T* dst, const T* src, const std::size_t size, const CW& cond)
    {
        if(src == nullptr || dst == nullptr)
            return cudaErrorIllegalAddress;

        std::size_t nTpB = SF3Dmin(size, 64);
        std::size_t nB = static_cast<std::size_t>(std::ceil((double) size / nTpB));

        return launchGPUKernel(conditionalCopyDevicePointer_k<T, CW>, dim3(nB), dim3(nTpB), dst, src, size, cond);
    }

    enum class reduceOperation_t : u8_t {Sum, Max};

    template<reduceOperation_t op, typename T>
    struct ReduceOpWrapper;

    template<typename T>
    struct ReduceOpWrapper<reduceOperation_t::Sum, T>
    {
        static void apply(void* temp, std::size_t& tempSize, const T* in, T* out, std::size_t n)
        {
            cub::DeviceReduce::Sum(temp, tempSize, in, out, n);
        }
    };

    template<typename T>
    struct ReduceOpWrapper<reduceOperation_t::Max, T>
    {
        static void apply(void* temp, std::size_t& tempSize, const T* in, T* out, std::size_t n)
        {
            cub::DeviceReduce::Max(temp, tempSize, in, out, n);
        }
    };

    template<reduceOperation_t op, typename T>
    inline T reduceDeviceVector(const T* d_ptr, const std::size_t size)
    {
        T *d_value = nullptr, h_value = 0;

        void* d_tempStorage = nullptr;
        std::size_t tempStorageSize = 0;

        deviceAlloc(d_value, 1);

        ReduceOpWrapper<op, T>::apply(d_tempStorage, tempStorageSize, d_ptr, d_value, size);
        cudaMalloc(&d_tempStorage, tempStorageSize);

        ReduceOpWrapper<op, T>::apply(d_tempStorage, tempStorageSize, d_ptr, d_value, size);
        cudaDeviceSynchronize();

        deviceFree(d_tempStorage);
        cudaMemcpy(&h_value, d_value, sizeof(T), cudaMemcpyDeviceToHost);
        deviceFree(d_value);
        return h_value;
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
}
