#pragma once

#include "macro.h"
#include "types_cpu.h"

using namespace soilFluxes3D::v2;

namespace soilFluxes3D::v2::Heat
{
    bool isHeatNode(SF3Duint_t nodeIndex);

    double computeCurrentHeatSinkSource(double dtHeat);
    double computeCurrentHeatStorage(double dtWater = noDataD, double dtHeat = noDataD);
    SF3Derror_t initializeHeatBalance();
    void evaluateHeatBalance(double dtHeat, double dtWater);
    void updateHeatBalanceData();
    void updateHeatBalanceDataWholePeriod();

    SF3Derror_t resetFluxValues(bool flagHeat, bool flagWater);
    SF3Derror_t saveWaterFluxValues(double dtHeat, double dtWater);
    SF3Derror_t saveNodeWaterFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    SF3Derror_t saveHeatFluxValues(double dtHeat, double dtWater);
    SF3Derror_t saveNodeHeatFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    SF3Derror_t saveNodeHeatSpecificFlux(SF3Duint_t nIdx, u8_t lIdx, fluxTypes_t fluxType, double fluxValue);

    SF3Derror_t updateConductance();

    bool updateBoundaryHeatData(double maxTimeStep, double& actualTimeStep);


    bool computeHeatLinkFluxes(double& matrixElement, SF3Duint_t& matrixIndex, SF3Duint_t nodeIndex, u8_t linkIndex, double dtHeat, double dtWater);


    double computeThermalLiquidFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat = noDataD, double dtWater = noDataD);
    double computeThermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat = noDataD, double dtWater = noDataD);
    double computeIsothermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    double computeIsothermalLatentHeatFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);
    double computeAdvectiveFlux(SF3Duint_t nIdx, u8_t lIdx);     //originally double&: check if needed somewhere
    double getLinkHeatFlux(const linkData_t &linkData, SF3Duint_t srcIndex, fluxTypes_t fluxType);

    double conduction(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater);

    double GaussSeidelHeatCPU(VectorCPU& vectorX, const MatrixCPU& matrixA, const VectorCPU& vectorB);

    double getNodeH_fromTimeSteps(SF3Duint_t nodeIndex, double dtHeat, double dtWater);

    double computeNodeHeatSoilConductivity(SF3Duint_t nodeIndex, double T, double h);
    double computeNodeHeatAirConductivity(SF3Duint_t nodeIndex, double T, double h);
    double computeNodeThermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h);
    double computeNodeIsothermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h);
    double computeNodeHeatCapacity(SF3Duint_t nodeIndex, double h, double T);
    double computeNodeVaporThetaV(SF3Duint_t nodeIndex, double h, double T);
    double computeNodeAerodynamicConductance(SF3Duint_t nodeIndex);
    double computeNodeAtmosphericSensibleHeatFlux(SF3Duint_t nodeIndex);
    double computeNodeAtmosphericLatentHeatFlux(SF3Duint_t nodeIndex);
    double computeNodeAtmosphericLatentVaporFlux(SF3Duint_t nodeIndex);
    double computeNodeAtmosphericLatentSurfaceWaterFlux(SF3Duint_t nodeIndex);

    //Move to soilPhysics (update comment in mathFunctions/physics.h
    double estimateNodeBulkDensity(SF3Duint_t nodeIndex);
    double estimateSoilParticleDensity(double organicMatter);

    double computeVapor_fromPsiTemp(double h, double T);
    double computeLatentVaporizationHeat(double T);
    double computeWaterReturnFlowFactor(double theta, double T, double clayFraction);
    double computePressure_fromAltitude(double height);
    double computeSoilVaporDiffusivity(double thetaS, double theta, double T);
    double computeSoilRelativeHumidity(double h, double T);
    double computeSoilSurfaceResistance(double thetaTop);
    double computeSaturationVaporPressure(double T);
    double computeSVPSlope(double T, double svp);

    double computeAirMolarDensity(double pressure, double T);
    double computeAirVolumetricSpecificHeat(double pressure, double T);
    double computeVaporPressure_fromConcentration(double concentration, double T);
    double computeVaporConcentration_fromPressure(double pressure, double T);
    double computeVaporBinaryDiffusivity(double T);

    double computeThermalLiquidConductivity(double T, double h, double ILK);
}
