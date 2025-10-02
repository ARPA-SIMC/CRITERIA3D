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

//Generic macroes
#define toStr(var) #var
#define expStr(var) toStr(var)

//Generic functions
#define toUnderlyingT(enumValue) castToUnderlyingType(enumValue)

//CPU base
#define hostAlloc(ptr, count) allocHostPointer(ptr, count)
#define hostFill(ptr, count, value) fillHostPointer(ptr, count, value)
#define hostReset(ptr, count) resetHostPointer(ptr, count)
#define hostFree(ptr) freeHostPointer(ptr)

//CPU solver
#define hostSolverAlloc(ptr, count) solverHostCheckError(hostAlloc(ptr, count), _status)
#define hostSolverFree(ptr) hostFree(ptr)


//GPU base
#define deviceAlloc(ptr, count) allocDevicePointer(ptr, count)
#define deviceReset(ptr, count) resetDevicePointer(ptr, count)
#define deviceFree(ptr) freeDevicePointer(ptr)

#define moveToDevice(ptr, count) movePointerToDevice(ptr, count, moveStreams[(currStreamIdx++) % 32])
#define moveToHost(ptr, count) movePointerToHost(ptr, count, moveStreams[(currStreamIdx++) % 32])

//GPU Solver
#define deviceSolverAlloc(ptr, count) solverDeviceCheckError(deviceAlloc(ptr, count), _status, SF3Derror_t::MemoryError)
#define deviceSolverFree(ptr) solverDeviceCheckError(deviceFree(ptr), _status, SF3Derror_t::MemoryError)

#define launchKernel(kernel, ...) launchGPUKernel(kernel, dim3(numBlocks), dim3(numThreadsPerBlock), __VA_ARGS__)
#define cuspCheck(retValue) solverDeviceCheckError(retValue, _status, SF3Derror_t::SolverError)

//CUDA runtime
#ifdef CUDA_ENABLED
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

//openMP directives
#ifdef _OPENMP  // Defined automatically when compiling with -fopenmp. Move to a custom macro?
    #define __parfor            _Pragma(expStr(omp parallel for if(__ompStatus)))
    #define __parforop(op, var) _Pragma(expStr(omp parallel for if(__ompStatus) reduction(op:var)))
#else
    #define __parfor
    #define __parforsum(var)
    #define __parformax(var)
#endif


//Log
#ifdef MCR_ENABLED
    #include "logFunctions.h"
    using namespace soilFluxes3D::Log;
    #define logStruct logNodeGridStruct(nodeGrid, getSolverType())
    #define logSystem //createCurrStepLog(matrixA, vectorB, vectorX, isStepValid)
#else
    #define logStruct
    #define logSystem
#endif




/* OLD
#define SF3DatomicMax(ptr, value) atomicMaxDouble(ptr, value)

inline __cudaSpec void atomicMaxDouble(double *ptr, double value)
{
    #if defined(__CUDA_ARCH__) && defined(CUDA_ENABLED)
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
*/

#endif // SOILFLUXES3D_MACRO_H
