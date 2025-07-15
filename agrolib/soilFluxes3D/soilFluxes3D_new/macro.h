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

    // Uncomment to inline functions -> not working
    #define __SF3DINLINE //inline

    #define cudaCheck(retValue) {if(retValue != cudaSuccess) {return retValue;}}
    #define cuspCheck(retValue) {if(retValue!=CUSPARSE_STATUS_SUCCESS) {exit(1);}}

#endif // SOILFLUXES3D_MACRO_H
