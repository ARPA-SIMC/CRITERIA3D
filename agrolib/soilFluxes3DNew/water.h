#ifndef SOILFLUXES3D_WATER_H
#define SOILFLUXES3D_WATER_H

#include "macro.h"
#include "types_cpu.h"

using namespace soilFluxes3D::New;

namespace soilFluxes3D::Water
{
    //Water simulation functions
    SF3Derror_t initializeWaterBalance();
    double computeTotalWaterContent();                               //TO DO: check pragma vs device code

    void computeCurrentMassBalance(double deltaT);
    double computeWaterSinkSourceFlowsSum(double deltaT);
    void updateBoundaryWaterData(double deltaT);

    void updateWaterBalanceDataWholePeriod();

    __cudaSpec double getMatrixElement(uint64_t rowIndex, uint64_t columnIndex);
    __cudaSpec void updateLinkFlux(uint64_t nodeIndex, uint8_t linkIndex, double deltaT);

    void acceptStep(double deltaT);
    void saveBestStep();
    void restoreBestStep(double deltaT);
    balanceResult_t evaluateWaterBalance(uint8_t approxNr, double& bestMBRerror, double deltaT, SolverParameters& parameters);

    void restorePressureHead();

    void computeCapacity(VectorCPU& vectorC);

    void computeLinearSystemElement(MatrixCPU &matrixA, VectorCPU& vectorB, const VectorCPU& vectorC, uint8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType);
    __cudaSpec bool computeLinkFluxes(double &matrixElement, uint64_t &matrixIndex, uint64_t nodeIndex, uint8_t linkIndex, uint8_t approxNum, double deltaT, double lateralVerticalRatio, linkType_t linkType, meanType_t meanType);

    __cudaSpec double runoff(uint64_t rowIdx, uint64_t colIdx, uint8_t approxNum, double deltaT, double flowArea);
    __cudaSpec double infiltration(uint64_t surfNodeIdx, uint64_t soilNodeIdx, double deltaT, double flowArea, meanType_t meanType);
    __cudaSpec double redistribution(uint64_t rowIdx, uint64_t colIdx, double lateralVerticalRatio, double flowArea, linkType_t linkType, meanType_t meanType);

    double JacobiWaterCPU(VectorCPU& vectorX, const MatrixCPU &matrixA, const VectorCPU& vectorB);
    double GaussSeidelWaterCPU(VectorCPU& vectorX, const MatrixCPU &matrixA, const VectorCPU& vectorB);
}

#endif // SOILFLUXES3D_WATER_H
