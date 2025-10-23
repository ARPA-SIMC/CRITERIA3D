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

#endif // SOILFLUXES3D_MACRO_H
