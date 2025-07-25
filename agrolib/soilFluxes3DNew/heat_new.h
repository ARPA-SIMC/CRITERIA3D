#ifndef SOILFLUXES3D_HEAT_H
#define SOILFLUXES3D_HEAT_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Heat
{

    double computeThermalLiquidFlux();
    double computeThermalVaporFlux();

    // //Heat simulation functions
    // void initializeBalanceHeat();
    // __EXTERN __SF3DINLINE void DLL_EXPORT __STDCALL initializeHeat(short saveHeatFluxes_, bool computeAdvectiveHeat, bool computeLatentHeat);
    // void initializeHeatFluxes(bool initHeat, bool initWater);           //Move to soilFluxes3D?
    // bool isGoodHeatBalance(double timeStepHeat, double timeStepWater, double &newtimeStepHeat);

    // bool HeatComputation(double timeStepHeat, double timeStepWater);
    // void computeHeatBalance(double timeStepHeat, double timeStepWater);
    // double ThermalVaporFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater);
    // double ThermalLiquidFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater);
    // __SF3DINLINE double IsothermalVaporFlux(long i, TlinkedNode *myLink, double timeStep, double timeStepWater);

    // //Node property
    // __SF3DINLINE bool isHeatNode(long i);
    // __SF3DINLINE double IsothermalVaporConductivity(long i, double h, double myT);
    // __SF3DINLINE double SoilRelativeHumidity(double h, double myT);

    // double SoilHeatCapacity(long i, double h, double T);
    // double SoilHeatConductivity(long i, double T, double h);
    // double VaporFromPsiTemp(double h, double T);
    // __SF3DINLINE double VaporThetaV(double h, double T, long i);

    // //IO functions
    // float readHeatFlux(TlinkedNode* myLink, int fluxType);                  //why it use float? move to double?
    // void saveHeatFlux(TlinkedNode* myLink, int fluxType, double myValue);
    // void saveWaterFluxes(double dtHeat, double timeStepWater);              //move to water?

    // void restoreHeat();
    // void updateBalanceHeatWholePeriod();
    // bool updateBoundaryHeat(double timeStep, double &reducedTimeStep);
    // void updateConductance();
}

#endif // SOILFLUXES3D_HEAT_H
