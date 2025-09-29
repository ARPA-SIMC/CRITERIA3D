#ifndef SOILFLUXES3D_TYPES_CPU_H
#define SOILFLUXES3D_TYPES_CPU_H

#include "types.h"

namespace soilFluxes3D::New
{
    struct MatrixCPU
    {
        uint64_t numRows;
        uint8_t maxColumns = maxMatrixColumns;
        uint8_t* numColumns = nullptr;
        uint64_t** colIndeces = nullptr;
        double** values = nullptr;
    };

    struct VectorCPU
    {
        uint64_t numElements;
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


#endif // SOILFLUXES3D_TYPES_CPU_H
