#include <math.h>

#include "rainfallInterception.h"


namespace canopy {


    double canopyInterceptionHydrall(double laiCanopy,double laiUnderstorey, double prec)
    {
        double canopyCapacity;
        double interception;
        double maxInterception;
        double upperBoundPrec = 20;
        if (prec > upperBoundPrec) maxInterception = 0.15 * upperBoundPrec;
        else maxInterception = 0.15 * prec;
        canopyCapacity = 0.07 * (laiCanopy + laiUnderstorey);
        if (canopyCapacity > maxInterception) interception = maxInterception ;
        else interception = canopyCapacity;
        return interception;
    }

    double canopyNoInterceptedRainfallHydrall(double laiCanopy,double laiUnderstorey, double prec)
    {
        double interception = canopyInterceptionHydrall(laiCanopy, laiUnderstorey, prec);
        return (prec - interception);
    }

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
        if (grossStorage < 0.01*storage) return grossStorage;

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

    bool waterManagementCanopy(double* storedWater, double rainfall, double waterFreeEvaporation, double lai,
                               double laiMin, double extinctionCoefficient, double leafStorage, double stemStorage,
                               double maxStemFlowRate, double* freeRainfall, double *drainage, double* stemFlow,
                               double* throughfallWater, double* soilWater)
    {
        // state variable (input and output):
        // double* storedWater: state variable: water stored within the canopy (mm)
        // input variables:
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
        // double* throughfallWater: (mm) water fallen on the ground composed by 2 components: interception free water + intercepted and fallen from leaves and branches (not from the trunk)
        // double *soilWater (mm) total rainfall water falling on soil


        double actualCover,actualStorage,grossStorage;
        double interception;
        actualCover = plantCover(lai,extinctionCoefficient,laiMin);
        actualStorage = waterStorageCapacity(lai,leafStorage,stemStorage);
        *freeRainfall = freeRainThroughfall(rainfall,actualCover);
        interception = rainfallInterceptedByCanopy(rainfall,actualCover);
        grossStorage = *storedWater + interception;
        grossStorage -= evaporationFromCanopy(waterFreeEvaporation,actualStorage,grossStorage);
        *drainage = drainageFromTree(grossStorage,actualStorage);
        *stemFlow = (*drainage)*stemFlowRate(maxStemFlowRate);
        *soilWater = *freeRainfall + (*drainage);
        *throughfallWater = *soilWater - (*stemFlow);
        *storedWater = grossStorage - (*drainage);
        return true;
    }

    bool waterManagementCanopy(double* storedWater, double rainfall, double waterFreeEvaporation, double lai, double laiMin,
                               double extinctionCoefficient, double leafStorage, double stemStorage,
                               double maxStemFlowRate, double* soilWater)
    {
        // state variable (input and output):
        // double* storedWater: state variable: water stored within the canopy (mm)

        // input variables:
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
        // double *soilWater (mm) total rainfall water falling on soil

        double actualCover,actualStorage,grossStorage,drainage;
        double interception, freeRainfall;

        actualCover = plantCover(lai,extinctionCoefficient,laiMin);
        actualStorage = waterStorageCapacity(lai,leafStorage,stemStorage);
        freeRainfall = freeRainThroughfall(rainfall,actualCover);
        interception = rainfallInterceptedByCanopy(rainfall,actualCover);
        grossStorage = *storedWater + interception;
        grossStorage -= evaporationFromCanopy(waterFreeEvaporation,actualStorage,grossStorage);
        drainage = drainageFromTree(grossStorage,actualStorage);

        // TODO check: not used
        // double stemFlow = (drainage)*stemFlowRate(maxStemFlowRate);

        *soilWater = freeRainfall + drainage;
        *storedWater = grossStorage - (drainage);

        return true;
    }
}
