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
    #include <cuda_runtime.h>
    #define __cudaMngd __managed__
    #define __cudaSpec __host__ __device__

    #define SF3Dmax(v1, v2) ((v1 > v2) ? v1 : v2)
    #define SF3Dmin(v1, v2) ((v1 < v2) ? v1 : v2)
#else
    #define __cudaMngd
    #define __cudaSpec

    #define SF3Dmax(v1, v2) std::max(v1, v2)
    #define SF3Dmin(v1, v2) std::min(v1, v2)
#endif

#define hostAlloc(ptr, type, size) {if(ptr != nullptr) {return MemoryError;} ptr = static_cast<type*>(calloc(size, sizeof(type))); if(ptr==nullptr) {return MemoryError;}}
#define hostFill(ptr, size, value) {std::fill(ptr, ptr + size, value);}
#define hostReset(ptr, size) {std::memset(ptr, 0, size);}
#define hostFree(ptr) {if(ptr != nullptr){free(ptr); ptr = nullptr;}}

#define hostSolverAlloc(ptr, type, size) {if(ptr != nullptr) {_status = Error; return SolverError;} ptr = static_cast<type*>(calloc(size, sizeof(type))); if(ptr==nullptr) {_status = Error; return SolverError;}}
#define hostSolverFree(ptr) {if(ptr != nullptr){free(ptr); ptr = nullptr;}}


#define destructDevicePointer(ptr) {if(ptr != nullptr) {cudaFree(ptr); ptr = nullptr;}}
#define destructHostPointer(ptr) {if(ptr != nullptr) {free(ptr); ptr = nullptr;}}



#define deviceSolverAlloc(ptr, type, count) {cudaCheck(cudaMalloc((void**) &(ptr), count * sizeof(type))); cudaCheck(cudaMemset(ptr, 0, count * sizeof(type)));}
#define deviceSolverFree(ptr) {if(ptr != nullptr) {cudaCheck(cudaFree(ptr)); ptr = nullptr;}}

/*TO DO:
 * - use a pinned buffer
 * - use async copy
 */
#define moveToDevice(ptr, type, count) { type *tmp;                                                                     \
                                         cudaCheck(cudaMalloc((void**) &(tmp), count * sizeof(type)));                  \
                                         cudaCheck(cudaMemcpy(tmp, ptr, count * sizeof(type), cudaMemcpyHostToDevice)); \
                                         free(ptr); ptr = tmp; }

#define moveToHost(ptr, type, count) { type *tmp = static_cast<type*>(calloc(count, sizeof(type)));                                         \
                                       cudaCheck(cudaMemcpy(tmp, ptr, count * sizeof(type), cudaMemcpyDeviceToHost));   \
                                       cudaCheck(cudaFree(ptr)); ptr = tmp; }


#define launchKernel(kernel, ...) {kernel<<<numBlocks, numThreadsPerBlock>>>(__VA_ARGS__);}


#define solverCheck(retValue) {if(retValue != SF3Dok) {_status = Error; return SolverError;}}


#define cudaCheckSolver(retValue) {if(retValue != cudaSuccess) {_status = Error; return;}}
#define cudaCheck(retValue) {if(retValue != cudaSuccess) {return MemoryError;}}
#define cuspCheck(retValue) {if(retValue != CUSPARSE_STATUS_SUCCESS) {_status = Error; return SolverError;}}

#endif // SOILFLUXES3D_MACRO_H
