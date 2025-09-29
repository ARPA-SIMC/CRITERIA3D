#ifndef SOILFLUXES3D_HEAT_H
#define SOILFLUXES3D_HEAT_H

#include "macro.h"
#include "types.h"

using namespace soilFluxes3D::New;

namespace soilFluxes3D::Heat
{
    bool isHeatNode(uint64_t nodeIndex);

    double computeCurrentHeatSinkSource(double dtHeat);
    double computeCurrentHeatStorage(double dtWater = noData, double dtHeat = noData);
    SF3Derror_t initializeHeatBalance();
    void evaluateHeatBalance(double dtHeat, double dtWater);
    void updateHeatBalanceData();
    void updateHeatBalanceDataWholePeriod();

    SF3Derror_t resetFluxValues(bool flagWater, bool flagHeat);
    SF3Derror_t saveWaterFluxValues(double dtHeat, double dtWater);
    SF3Derror_t saveNodeWaterFluxes(uint64_t nIdx, uint8_t lIdx, double dtHeat, double dtWater);
    SF3Derror_t saveHeatFluxValues(double dtHeat, double dtWater);
    SF3Derror_t saveNodeHeatFluxes(uint64_t nIdx, uint8_t lIdx, double dtHeat, double dtWater);
    SF3Derror_t saveNodeHeatSpecificFlux(uint64_t nIdx, uint8_t lIdx, fluxTypes_t fluxType, double fluxValue);

    SF3Derror_t updateConductance();

    bool updateBoundaryHeatData(double maxTimeStep, double& actualTimeStep);


    bool computeHeatLinkFluxes(double &matrixElement, uint64_t &matrixIndex, uint64_t nodeIndex, uint8_t linkIndex, double dtHeat, double dtWater);


    double computeThermalLiquidFlux(uint64_t nIdx, uint8_t lIdx, processType process, double dtHeat = noData, double dtWater = noData);
    double computeThermalVaporFlux(uint64_t nIdx, uint8_t lIdx, processType process, double dtHeat = noData, double dtWater = noData);
    double computeIsothermalVaporFlux(uint64_t nIdx, uint8_t lIdx, double dtHeat, double dtWater);
    double computeIsothermalLatentHeatFlux(uint64_t nIdx, uint8_t lIdx, double dtHeat, double dtWater);
    double computeAdvectiveFlux(uint64_t nIdx, uint8_t lIdx);     //originally double&: check if needed somewhere
    double getLinkHeatFlux(const linkData_t &linkData, uint64_t srcIndex, fluxTypes_t fluxType);

    double conduction(uint64_t nIdx, uint8_t lIdx, double dtHeat, double dtWater);

    double GaussSeidelHeatCPU(VectorCPU& vectorX, const MatrixCPU& matrixA, const VectorCPU& vectorB);

    double getNodeH_fromTimeSteps(uint64_t nodeIndex, double dtHeat, double dtWater);

    double computeNodeHeatSoilConductivity(uint64_t nodeIndex, double T, double h);
    double computeNodeHeatAirConductivity(uint64_t nodeIndex, double T, double h);
    double computeNodeThermalVaporConductivity(uint64_t nodeIndex, double T, double h);
    double computeNodeIsothermalVaporConductivity(uint64_t nodeIndex, double T, double h);
    double computeNodeHeatCapacity(uint64_t nodeIndex, double h, double T);
    double computeNodeVaporThetaV(uint64_t nodeIndex, double h, double T);
    double computeNodeAerodynamicConductance(uint64_t nodeIndex);
    double computeNodeAtmosphericSensibleHeatFlux(uint64_t nodeIndex);
    double computeNodeAtmosphericLatentHeatFlux(uint64_t nodeIndex);
    double computeNodeAtmosphericLatentSurfaceWaterFlux(uint64_t nodeIndex);

    //Move to soilPhysics (update comment in mathFunctions/physics.h
    double estimateNodeBulkDensity(uint64_t nodeIndex);
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

#endif // SOILFLUXES3D_HEAT_H
