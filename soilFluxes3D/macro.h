#pragma once

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
#define deviceFill(ptr, count, value) fillDevicePointer(ptr, count, value)
#define deviceReset(ptr, count) resetDevicePointer(ptr, count)
#define deviceFree(ptr) freeDevicePointer(ptr)

#define deviceConditionalCopy(dst, src, count, condition) conditionalCopyDevicePointer(dst, src, count, condition)
#define deviceSum(ptr, count) reduceDeviceVector<reduceOperation_t::Sum>(ptr, count)
#define deviceMax(ptr, count) reduceDeviceVector<reduceOperation_t::Max>(ptr, count)

#define moveToDevice(ptr, count) movePointerToDevice(ptr, count, moveStreams[(currStreamIdx++) % 32])
#define moveToHost(ptr, count) movePointerToHost(ptr, count, moveStreams[(currStreamIdx++) % 32])

#define launchKernel(kernel, ...) launchGPUKernel(kernel, dim3(numBlocks), dim3(numThreadsPerBlock), __VA_ARGS__)

//GPU Solver
#define deviceSolverAlloc(ptr, count) solverDeviceCheckError(deviceAlloc(ptr, count), _status, SF3Derror_t::MemoryError)
#define deviceSolverFree(ptr) solverDeviceCheckError(deviceFree(ptr), _status, SF3Derror_t::MemoryError)

#define cuspCheck(retValue) solverDeviceCheckError(retValue, _status, SF3Derror_t::SolverError)

//CUDA runtime
#ifdef CUDA_ENABLED
    #include <cuda_runtime.h>
    #define __cudaMngd __managed__
    #define __cudaSpec __host__ __device__
    #define __ompStatus (*solver)->getOMPstatus()

    #define SF3Dmax(v1, v2) ((v1 > v2) ? (v1) : (v2))
    #define SF3Dmin(v1, v2) ((v1 < v2) ? (v1) : (v2))
#else
    #define __cudaMngd
    #define __cudaSpec
    #define __ompStatus (solver)->getOMPstatus()

    #define SF3Dmax(v1, v2) std::max(v1, v2)
    #define SF3Dmin(v1, v2) std::min(v1, v2)
#endif

//openMP directives
#ifdef _OPENMP  // Defined automatically when compiling with openmp flag. Move to a custom macro?
    #define __parfor(cond)              _Pragma(expStr(omp parallel for if(cond)))
    #ifndef _MSVC_LANG
        #define __parforop(cond, op, var)   _Pragma(expStr(omp parallel for if(cond) reduction(op:var)))
    #else
        #define __parforop(cond, op, var)
    #endif
#else
    #define __parfor(cond)
    #define __parforop(cond, op, var)
#endif

//Log
#ifdef MCR_ENABLED
    #include "logFunctions.h"
    using namespace soilFluxes3D::v2::Log;
    #define logStruct logNodeGridStruct(nodeGrid, getSolverType())
    #define logSystem //createCurrStepLog(matrixA, vectorB, vectorX, isStepValid)
#else
    #define logStruct
    #define logSystem
#endif

