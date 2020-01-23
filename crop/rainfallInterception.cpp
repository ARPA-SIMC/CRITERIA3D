#include <stdio.h>
#include <math.h>

#include "rainfallInterception.h"


namespace canopy {

    double plantCover(double lai, double extinctionCoefficient,double laiMin)
    {
        if (lai < laiMin) lai=laiMin;
        return (1-exp(-extinctionCoefficient*(lai))); //Van Dijk  A.I.J.M, Bruijnzeel L.A., 2001
    }

    double waterStorageCapacity(double lai,double leafStorage,double stemStorage)
    {
        return leafStorage*lai + stemStorage; //Van Dijk  A.I.J.M, Bruijnzeel L.A., 2001
    }

    double freeRainThroughfall(double rainfall,double cover)
    {
        return (rainfall*(1 - cover));
    }

    double rainfallInterceptedByCanopy(double rainfall,double cover)
    {
        return (rainfall*cover);
    }
    double evaporationFromCanopy(double waterFreeEvaporation, double storage,double grossStorage)
    {
        if (grossStorage < 0.1*storage) return grossStorage;

        if (grossStorage >= storage) return waterFreeEvaporation;
        else
        {
            return waterFreeEvaporation*grossStorage/storage ;
        }
    }

    double drainageFromTree(double grossStorage, double storage)
    {
        if (grossStorage > storage) return (grossStorage - storage);
        else return 0;
    }

    double stemFlowRate(double maxStemFlowRate)
    {
        return maxStemFlowRate;
    }

    bool waterManagementCanopy(double* storedWater, double* throughfallWater, double rainfall, double waterFreeEvaporation, double lai, double extinctionCoefficient, double leafStorage, double stemStorage,double maxStemFlowRate, double* freeRainfall, double *drainage, double* stemFlow, double laiMin)
    {
        // input variables:
        // double* storedWater: state variable: water stored within the canopy (mm)
        // double rainfall: hourly rainfall amount (mm)
        // double waterFreeEvaporation: water evaporated from a free water surface, e.g. a lake) (mm)
        // double lai leaf area index (m2 m-2) // it must include the cover provided by stems too
        // input parameters:
        // double laiMin minimal LAI (m2 m-2) // it must include the cover provided by stems too
        // double extinctionCoefficient light extinction coefficient of the plant (-)
        // double leaf storage (mm)
        // double stemstorge (mm)
        // double maxStemFlowRate (mm)
        // output:
        // double* freeRainfall (mm)
        // double *drainage (mm)
        // double* stemFlow (mm)
        // double* throughfallWater: (mm) water fallen on the ground composed by 2 components: interception free water + intercpted and fallen from leaves and branches (not from the trunk)

        double actualCover,actualStorage,grossStorage;
        double interception;
        actualCover = plantCover(lai,extinctionCoefficient,laiMin);
        actualStorage = waterStorageCapacity(lai,leafStorage,stemStorage);
        *freeRainfall = freeRainThroughfall(rainfall,actualCover);
        interception = rainfallInterceptedByCanopy(rainfall,actualCover);
        grossStorage = *storedWater + interception;
        *drainage = drainageFromTree(grossStorage,actualStorage);
        *stemFlow = (*drainage)*stemFlowRate(maxStemFlowRate);
        *throughfallWater = *freeRainfall + (*drainage)-(*stemFlow);
        grossStorage -= evaporationFromCanopy(waterFreeEvaporation,actualStorage,grossStorage);
        *storedWater = grossStorage - (*drainage);
        return true;
    }
}
