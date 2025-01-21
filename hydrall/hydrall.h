#ifndef HYDRALL_H
#define HYDRALL_H

    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif

    #define UPSCALINGFUNC(z,LAI) ((1.0 - exp(-(z)*(LAI))) / (z))

    // Tree-plant properties
    #define FORM   0.5          // stem form factor
    #define RHOF   0.1          // [KgDM m-3] foliage density
    #define RHOS   750          // [KgDM m-3] wood-stem density

    // Hydraulic properties
    #define H50     0.4         // height for 50% maturation of xylem cells (m) [not relevant]
    #define KR      4.0E-7      // root specific conductance (m3 MPa-1 s-1 kg-1) [not relevant]
    #define KSMAX   2.5E-3      // max. sapwood specific conductivity (m2 MPa-1 s-1) [not relevant]
    #define PSITHR  -2.5        // water potential threshold for cavitation (MPa) [not relevant]


    #define NOT_INITIALIZED_VINE -1

    bool computeHydrall(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep);
    double getCO2(Crit3DDate myDate, double myTemperature, double myElevation);
    double getPressureFromElevation(double myTemperature, double myElevation);
    double getLAI();
    double meanLastMonthTemperature(double previousLastMonthTemp, double simulationStepInSeconds, double myInstantTemp);

#endif // HYDRALL_H
