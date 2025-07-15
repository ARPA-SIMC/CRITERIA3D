#ifndef SOILFLUXES3D_WATER_H
#define SOILFLUXES3D_WATER_H

#include "macro.h"
#include "types_cpu.h"

using namespace soilFluxes3D::New;

namespace soilFluxes3D::Water
{
    //Water simulation functions
    SF3Derror_t initializeWaterBalance();                       //TO DO: move outside
    double computeTotalWaterContent_new();                      //TO DO: move outside

    void computeCurrentMassBalance(double deltaT);
    double computeWaterSinkSourceFlowsSum(double deltaT);


    double getMatrixValue(uint64_t rowIndex, uint64_t columnIndex);
    void updateLinkFlux(uint64_t nodeIndex, uint8_t linkIndex, double deltaT);

    void acceptStep(double deltaT);
    balanceResult_t evaluateWaterBalance(uint8_t approxNr, double& bestMBRerror, SolverParameters& parameters);




    void updateBoundaryWater(double deltaT);

    bool computeWaterFluxes(double maxTime, double& acceptedTime);
    bool waterFlowComputation_stdTreads(double deltaT);     //TO DO: refactor

    //IO functions
    __SF3DINLINE void restoreWater();
    void updateBalanceWaterWholePeriod();
}

#endif // SOILFLUXES3D_WATER_H
