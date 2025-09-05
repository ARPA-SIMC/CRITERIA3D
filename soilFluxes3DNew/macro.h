#ifndef SOILFLUXES3D_MACRO_H
#define SOILFLUXES3D_MACRO_H

// Uncomment to compile as win32 dll
// #define BUILD_DLL 1

#ifdef BUILD_DLL
    #define DLL_EXPORT __declspec(dllexport)
    #define __EXTERN extern "C"
    #define __STDCALL __stdcall
#else
    #define DLL_EXPORT
    #define __EXTERN
    #define __STDCALL
#endif

#ifdef CUDA_ENABLED
    #include <cuda.h>
    #include <cuda_runtime.h>
    #define __cudaMngd __managed__
    #define __cudaSpec __host__ __device__

    #define __ompStatus (*solver)->getOMPstatus()

    #define SF3Dmax(v1, v2) ((v1 > v2) ? v1 : v2)
    #define SF3Dmin(v1, v2) ((v1 < v2) ? v1 : v2)
#else
    #define __cudaMngd
    #define __cudaSpec

    #define __ompStatus (solver)->getOMPstatus()

    #define SF3Dmax(v1, v2) std::max(v1, v2)
    #define SF3Dmin(v1, v2) std::min(v1, v2)

#endif


#define SF3DatomicMax(ptr, value) atomicMaxDouble(ptr, value)

inline __cudaSpec void atomicMaxDouble(double *ptr, double value)
{
    #ifdef __CUDA_ARCH__
        using ull = unsigned long long int;
        ull* addrAsULL = reinterpret_cast<ull*>(ptr);
        ull newValue = __double_as_longlong(value);
        ull oldValue = *addrAsULL, assumed;
        do
        {
            assumed = oldValue;
            if(newValue <= assumed)
                break;
            oldValue = atomicCAS(addrAsULL, assumed, newValue);
        }
        while(assumed != oldValue);

    #else
        #pragma omp critical
        *ptr = (*ptr > value) ? *ptr : value;
    #endif
}




//TO DO: move all macro into inline funtions + errorCheck

#define hostAlloc(ptr, type, size) {if(ptr != nullptr) {return MemoryError;} ptr = static_cast<type*>(std::calloc(size, sizeof(type))); if(ptr==nullptr) {return MemoryError;}}
#define hostFill(ptr, size, value) {std::fill(ptr, ptr + size, value);}
#define hostReset(ptr, size) {std::memset(ptr, 0, size);}
#define hostFree(ptr) {if(ptr != nullptr){std::free(ptr); ptr = nullptr;}}

#define hostSolverAlloc(ptr, type, size) {if(ptr != nullptr) {_status = Error; return SolverError;} ptr = static_cast<type*>(std::calloc(size, sizeof(type))); if(ptr==nullptr) {_status = Error; return SolverError;}}
#define hostSolverFree(ptr) {if(ptr != nullptr){std::free(ptr); ptr = nullptr;}}


#define destructDevicePointer(ptr) {if(ptr != nullptr) {cudaFree(ptr); ptr = nullptr;}}
#define destructHostPointer(ptr) {if(ptr != nullptr) {free(ptr); ptr = nullptr;}}



#define deviceSolverAlloc(ptr, type, count) {cudaCheck(cudaMalloc((void**) &(ptr), count * sizeof(type))); cudaCheck(cudaMemset(ptr, 0, count * sizeof(type)));}
#define deviceSolverFree(ptr) {if(ptr != nullptr) {cudaCheck(cudaFree(ptr)); ptr = nullptr;}}

/*TO DO:
 * - use a pinned buffer
 * - use async copy
 */

#define moveToDevice(ptr, type, count) { type *tmp;                                                                     \
                                         cudaMalloc((void**) &(tmp), count * sizeof(type));                  \
                                         cudaMemcpy(tmp, ptr, count * sizeof(type), cudaMemcpyHostToDevice); \
                                         std::free(ptr); ptr = tmp; }

#define moveToHost(ptr, type, count) { type *tmp = static_cast<type*>(std::calloc(count, sizeof(type)));                     \
                                       cudaMemcpy(tmp, ptr, count * sizeof(type), cudaMemcpyDeviceToHost);   \
                                       cudaFree(ptr); ptr = tmp; }


#define launchKernel(kernel, ...) { kernel<<<numBlocks, numThreadsPerBlock>>>(__VA_ARGS__);  \
                                    cudaDeviceSynchronize(); }


#define solverCheck(retValue) {if(retValue != SF3Dok) {_status = Error; return SolverError;}}


#define cudaCheckCritical(retValue) {if(retValue != cudaSuccess) {_status = Error; exit(1);}}
#define cudaCheckSolver(retValue) {if(retValue != cudaSuccess) {_status = Error; return;}}
#define cudaCheck(retValue) {if(retValue != cudaSuccess) {return MemoryError;}}
#define cuspCheck(retValue) {if(retValue != CUSPARSE_STATUS_SUCCESS) {_status = Error; return SolverError;}}


#endif // SOILFLUXES3D_MACRO_H
