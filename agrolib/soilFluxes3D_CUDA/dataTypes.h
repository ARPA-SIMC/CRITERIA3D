#ifndef SOILFLUXES3DCUDA_DATATYPES_H
#define SOILFLUXES3DCUDA_DATATYPES_H

#include <cstdint>
#include <iostream>
// #include <stdfloat>      -> move to float64_t instead of double?

#define destrucHostPointer(ptr) {if(ptr != nullptr){free(ptr); ptr = nullptr;}}
#define destrucCUDAPointer(ptr) {if(ptr != nullptr){cudaFree(ptr); ptr = nullptr;}}

#define dbg(x) {std::cout << x << std::endl;}

namespace soilFluxes3D_CUDA
{
    static uint32_t MAXNUMCOLS = 11;
    enum numericalMethod {Jacobi, GaussSeidel};
    enum kernelStatus {kernelError, kernelCreated, kernelInizialized, kernelLaunched, kernelExecuted};

    template<typename T>
    struct CUDAoptimizedVector
    {
        uint32_t numElement = 0;
        T* values = nullptr;
    };

    template<typename T>
    struct CUDAoptimizedMatrix
    {
        const uint32_t maxNumCols = MAXNUMCOLS;
        uint32_t numRow = 0;
        uint32_t *numCol = nullptr;
        uint32_t *elementIndeces = nullptr;

        T* elementValues  = nullptr;
    };

}

#endif // SOILFLUXES3DCUDA_DATATYPES_H
