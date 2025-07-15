#ifndef HEAT_H
#define HEAT_H

    #ifndef MACRO_H
        #include "soilFluxes3D_new/macro.h"
    #endif

    struct TlinkedNode;

    __SF3DINLINE bool isHeatNode(long i);
    double ThermalVaporFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater);
    double ThermalLiquidFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater);
    __SF3DINLINE double IsothermalVaporConductivity(long i, double h, double myT);
    __SF3DINLINE double IsothermalVaporFlux(long i, TlinkedNode *myLink, double timeStep, double timeStepWater);
    __SF3DINLINE double SoilRelativeHumidity(double h, double myT);
    double SoilHeatCapacity(long i, double h, double T);
    double SoilHeatConductivity(long i, double T, double h);
    double VaporFromPsiTemp(double h, double T);
    __SF3DINLINE double VaporThetaV(double h, double T, long i);
    void restoreHeat();
    void initializeBalanceHeat();
    void updateBalanceHeatWholePeriod();
    void initializeHeatFluxes(bool initHeat, bool initWater);
    void saveWaterFluxes(double dtHeat, double timeStepWater);
    void saveHeatFlux(TlinkedNode* myLink, int fluxType, double myValue);
    float readHeatFlux(TlinkedNode* myLink, int fluxType);
    bool isGoodHeatBalance(double timeStepHeat, double timeStepWater, double &newtimeStepHeat);
    bool HeatComputation(double timeStepHeat, double timeStepWater);
    void computeHeatBalance(double timeStepHeat, double timeStepWater);

#endif
