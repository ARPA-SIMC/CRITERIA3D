#pragma once

#include "macro.h"
#include "types_cpu.h"

using namespace soilFluxes3D::v2;

namespace soilFluxes3D::v2::Heat
{
    __cudaSpec bool isHeatNode(SF3Duint_t nodeIndex);

    double computeCurrentHeatSinkSource(double dtHeat);
    double computeCurrentHeatStorage(double dtWater = noDataD, double dtHeat = noDataD);
    SF3Derror_t initializeHeatBalance();
    void evaluateHeatBalance(double dtHeat, double dtWater);
    void updateHeatBalanceData();
    void updateHeatBalanceDataWholePeriod();

    SF3Derror_t resetFluxValues(bool flagHeat, bool flagWater);
    SF3Derror_t saveWaterFluxValues(double dtHeat, double dtWater);
    __cudaSpec SF3Derror_t saveNodeWaterFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    SF3Derror_t saveHeatFluxValues(double dtHeat, double dtWater);
    __cudaSpec SF3Derror_t saveNodeHeatFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    __cudaSpec SF3Derror_t saveNodeHeatSpecificFlux(SF3Duint_t nIdx, u8_t lIdx, fluxTypes_t fluxType, double fluxValue);

    SF3Derror_t updateConductance();

    bool updateBoundaryHeatData(double maxTimeStep, double& actualTimeStep);

    __cudaSpec bool computeHeatLinkFluxes(double& matrixElement, SF3Duint_t& matrixIndex, SF3Duint_t nodeIndex, u8_t linkIndex, double dtHeat, double dtWater);

    __cudaSpec double computeThermalLiquidFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat = noDataD, double dtWater = noDataD);
    __cudaSpec double computeThermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat = noDataD, double dtWater = noDataD);
    __cudaSpec double computeIsothermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    __cudaSpec double computeIsothermalLatentHeatFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    __cudaSpec double computeAdvectiveFlux(SF3Duint_t nIdx, u8_t lIdx);     //originally double&: check if needed somewhere
    __cudaSpec double getLinkHeatFlux(const linkData_t &linkData, SF3Duint_t srcIndex, fluxTypes_t fluxType);

    __cudaSpec double conduction(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);

    double GaussSeidelHeatCPU(VectorCPU& vectorX, const MatrixCPU& matrixA, const VectorCPU& vectorB);

    __cudaSpec double getNodeH_fromTimeSteps(SF3Duint_t nodeIndex, double dtHeat, double dtWater);

    __cudaSpec double computeNodeHeatSoilConductivity(SF3Duint_t nodeIndex, double T, double h);
    __cudaSpec double computeNodeHeatAirConductivity(SF3Duint_t nodeIndex, double T, double h);
    __cudaSpec double computeNodeThermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h);
    __cudaSpec double computeNodeIsothermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h);
    __cudaSpec double computeNodeHeatCapacity(SF3Duint_t nodeIndex, double h, double T);
    __cudaSpec double computeNodeVaporThetaV(SF3Duint_t nodeIndex, double h, double T);
    __cudaSpec double computeNodeAerodynamicConductance(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeAtmosphericSensibleHeatFlux(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeAtmosphericLatentHeatFlux(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeAtmosphericLatentVaporFlux(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeAtmosphericLatentSurfaceWaterFlux(SF3Duint_t nodeIndex);

    //Move to soilPhysics (update comment in mathFunctions/physics.h)
    __cudaSpec double estimateNodeBulkDensity(SF3Duint_t nodeIndex);
    __cudaSpec double estimateSoilParticleDensity(double organicMatter);

    __cudaSpec double computeVapor_fromPsiTemp(double h, double T);
    __cudaSpec double computeLatentVaporizationHeat(double T);
    __cudaSpec double computeWaterReturnFlowFactor(double theta, double T, double clayFraction);
    __cudaSpec double computePressure_fromAltitude(double height);
    __cudaSpec double computeSoilVaporDiffusivity(double thetaS, double theta, double T);
    __cudaSpec double computeSoilRelativeHumidity(double h, double T);
    __cudaSpec double computeSoilSurfaceResistance(double thetaTop);
    __cudaSpec double computeSaturationVaporPressure(double T);
    __cudaSpec double computeSVPSlope(double T, double svp);

    __cudaSpec double computeAirMolarDensity(double pressure, double T);
    __cudaSpec double computeAirVolumetricSpecificHeat(double pressure, double T);
    __cudaSpec double computeVaporPressure_fromConcentration(double concentration, double T);
    __cudaSpec double computeVaporConcentration_fromPressure(double pressure, double T);
    __cudaSpec double computeVaporBinaryDiffusivity(double T);

    __cudaSpec double computeThermalLiquidConductivity(double T, double h, double ILK);
}
