#ifndef RAINFALLINTERCEPTION_H
#define RAINFALLINTERCEPTION_H

namespace canopy
{
    double plantCover(double lai, double extinctionCoefficient, double laiMin);
    double waterStorageCapacity(double lai,double leafStorage,double stemStorage);
    double freeRainThroughfall(double rainfall,double cover);
    double rainfallInterceptedByCanopy(double rainfall,double cover);
    double evaporationFromCanopy(double waterFreeEvaporation, double storage, double grossStorage);
    double drainageFromTree(double grossStorage, double storage);
    double stemFlowRate(double maxStemFlowRate);

    bool waterManagementCanopy(double* StoredWater, double rainfall, double waterFreeEvaporation, double lai, double laiMin, double extinctionCoefficient, double leafStorage, double stemStorage, double maxStemFlowRate, double* freeRainfall, double* drainage, double *stemFlow, double* throughfallWater, double *soilWater);
    bool waterManagementCanopy(double* storedWater, double rainfall, double waterFreeEvaporation, double lai, double laiMin, double extinctionCoefficient, double leafStorage, double stemStorage,double maxStemFlowRate, double *drainage);

    double canopyInterceptionHydrall(double laiCanopy,double laiUnderstorey, double prec);
    double canopyNoInterceptedRainfallHydrall(double laiCanopy,double laiUnderstorey, double prec);
}


#endif // RAINFALLINTERCEPTION_H
