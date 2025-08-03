#ifndef SOILFLUXES3D_WATER_H
#define SOILFLUXES3D_WATER_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Water
{
    //Water simulation functions
    soilFluxes3D::New::SF3Derror_t initializeWaterBalance();
    double computeTotalWaterContent();                               //TO DO: check pragma vs device code

    void computeCurrentMassBalance(double deltaT);
    double computeWaterSinkSourceFlowsSum(double deltaT);
    void updateBoundaryWaterData(double deltaT);

    __cudaSpec double getMatrixElement(uint64_t rowIndex, uint64_t columnIndex);
    __cudaSpec void updateLinkFlux(uint64_t nodeIndex, uint8_t linkIndex, double deltaT);

    void acceptStep(double deltaT);
    void saveBestStep();
    void restoreBestStep(double deltaT);
    soilFluxes3D::New::balanceResult_t evaluateWaterBalance(uint8_t approxNr, double& bestMBRerror, double deltaT, soilFluxes3D::New::SolverParameters& parameters);

    void restorePressureHead();

    void computeCapacity(soilFluxes3D::New::VectorCPU& vectorC);

    void computeLinearSystemElement(soilFluxes3D::New::MatrixCPU &matrixA, soilFluxes3D::New::VectorCPU& vectorB, const soilFluxes3D::New::VectorCPU& vectorC, uint8_t approxNum, double deltaT, double lateralVerticalRatio, soilFluxes3D::New::meanType_t meanType);
    __cudaSpec bool computeLinkFluxes(double &matrixElement, uint64_t &matrixIndex, uint64_t nodeIndex, uint8_t linkIndex, uint8_t approxNum, double deltaT, double lateralVerticalRatio, soilFluxes3D::New::linkType_t linkType, soilFluxes3D::New::meanType_t meanType);

    __cudaSpec double runoff(uint64_t rowIdx, uint64_t colIdx, uint8_t approxNum, double deltaT, double flowArea);
    __cudaSpec double infiltration(uint64_t surfNodeIdx, uint64_t soilNodeIdx, double deltaT, double flowArea, soilFluxes3D::New::meanType_t meanType);
    __cudaSpec double redistribution(uint64_t rowIdx, uint64_t colIdx, double lateralVerticalRatio, double flowArea, soilFluxes3D::New::linkType_t linkType, soilFluxes3D::New::meanType_t meanType);

    double JacobiWaterCPU(soilFluxes3D::New::VectorCPU& vectorX, const soilFluxes3D::New::MatrixCPU &matrixA, const soilFluxes3D::New::VectorCPU& vectorB);
    double GaussSeidelWaterCPU(soilFluxes3D::New::VectorCPU& vectorX, const soilFluxes3D::New::MatrixCPU &matrixA, const soilFluxes3D::New::VectorCPU& vectorB);
}

#endif // SOILFLUXES3D_WATER_H
