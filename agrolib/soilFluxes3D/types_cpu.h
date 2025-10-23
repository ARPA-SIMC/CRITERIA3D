#pragma once

#include "types.h"

namespace soilFluxes3D::v2
{
    struct MatrixCPU
    {
        SF3Duint_t numRows;
        u8_t maxColumns = maxMatrixColumns;
        u8_t* numColumns = nullptr;
        SF3Duint_t** colIndeces = nullptr;
        double** values = nullptr;
    };

    struct VectorCPU
    {
        SF3Duint_t numElements;
        double* values;
    };

    template<typename T>
    inline SF3Derror_t allocHostPointer(T*& ptr, const std::size_t count)
    {
        if(ptr != nullptr)
            return SF3Derror_t::MemoryError;

        ptr = reinterpret_cast<T*>(std::calloc(count, sizeof(T)));

        if(ptr == nullptr)
            return SF3Derror_t::MemoryError;

        return SF3Derror_t::SF3Dok;
    }

    template<typename T>
    inline void freeHostPointer(T*& ptr)
    {
        if(ptr == nullptr)
            return;

        std::free(ptr);
        ptr = nullptr;
    }

    template<typename T>
    inline SF3Derror_t resetHostPointer(T*& ptr, const std::size_t count)
    {
        if(ptr == nullptr)
            return SF3Derror_t::MemoryError;

        std::memset(ptr, 0, count * sizeof(T));
        return SF3Derror_t::SF3Dok;
    }

    template<typename T>
    inline SF3Derror_t fillHostPointer(T*& ptr, const std::size_t count, const T value)
    {
        if(ptr == nullptr)
            return SF3Derror_t::MemoryError;

        std::fill(ptr, ptr + count, value);
        return SF3Derror_t::SF3Dok;
    }

}
