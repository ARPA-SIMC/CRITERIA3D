#ifndef SOILFLUXES3D_MACRO_H
#define SOILFLUXES3D_MACRO_H

#include <algorithm>

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

#define hostAlloc(ptr, type, size) {if(ptr != nullptr) {return MemoryError;} ptr = static_cast<type*>(calloc(size, sizeof(type))); if(ptr==nullptr) {return MemoryError;}}
#define hostFill(ptr, size, value) {std::fill(ptr, ptr + size, value);}     //TO DO: move to a #pragma omp for condizionato

#define hostFree(ptr) {if(ptr != nullptr){free(ptr);} ptr = nullptr;}

#define cudaCheck(retValue) {if(retValue != cudaSuccess) {return retValue;}}
#define cuspCheck(retValue) {if(retValue!=CUSPARSE_STATUS_SUCCESS) {exit(1);}}

#endif // SOILFLUXES3D_MACRO_H
