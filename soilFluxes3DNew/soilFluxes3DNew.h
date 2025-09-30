#ifndef SOILFLUXES3DNEW_H
#define SOILFLUXES3DNEW_H

#include "macro.h"
#include "types.h"

namespace soilFluxes3D::New
{
    //Inizializazion and memory management
    SF3Derror_t initializeSF3D(uint64_t nrNodes, uint16_t nrLayers, uint8_t nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes, heatFluxSaveMode_t HFsm = heatFluxSaveMode_t::None);
    SF3Derror_t initializeBalance();
    SF3Derror_t initializeLog(const std::string& logPath, const std::string& projectName);

    SF3Derror_t cleanSF3D();
    SF3Derror_t closeLog();

    SF3Derror_t initializeHeatFlag(heatFluxSaveMode_t saveModeHeat, bool isComputeAdvectiveFlux, bool isComputeLatentHeat);

    uint32_t setThreadsNumber(uint32_t nrThreads);

    //Create types
    SF3Derror_t setSoilProperties(uint16_t nrSoil, uint16_t nrHorizon, double VG_alpha, double VG_n, double VG_m, double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay);
    SF3Derror_t setSurfaceProperties(uint16_t surfaceIndex, double roughness);

    //Set core data
    SF3Derror_t setNumericalParameters(double minDeltaT, double maxDeltaT, uint16_t maxIterationNumber, uint16_t maxApproximationsNumber, uint8_t ResidualToleranceExponent, uint8_t MBRThresholdExponent);
    SF3Derror_t setHydraulicProperties(WRCModel waterRetentionCurve, meanType_t conductivityMeanType, float conductivityHorizVertRatio);

    //Set topology
    SF3Derror_t setCulvert(uint64_t nodeIndex, double roughness, double slope, double width, double height);
    SF3Derror_t setNode(uint64_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope = 0, double boundaryArea = 0);
    SF3Derror_t setNodeLink(uint64_t nodeIndex, uint64_t linkIndex, linkType_t direction, double interfaceArea);
    SF3Derror_t setNodeBoundary(uint64_t nodeIndex, boundaryType_t boundaryType, double slope, double boundaryArea);

    //Set soil data
    SF3Derror_t setNodeSoil(uint64_t nodeIndex, uint16_t soilIndex, uint16_t horizonIndex);
    SF3Derror_t setNodeSurface(uint64_t nodeIndex, uint16_t surfaceIndex);

    //Set water data
    SF3Derror_t setNodePond(uint64_t nodeIndex, double pond);
    SF3Derror_t setNodeWaterContent(uint64_t nodeIndex, double waterContent);
    SF3Derror_t setNodeDegreeOfSaturation(uint64_t nodeIndex, double degreeOfSaturation);
    SF3Derror_t setNodeMatricPotential(uint64_t nodeIndex, double matricPotential);
    /*not used*/ SF3Derror_t setNodeTotalPotential(uint64_t nodeIndex, double totalPotential);
    SF3Derror_t setNodeWaterSinkSource(uint64_t nodeIndex, double waterSinkSource);
    /*not used*/ SF3Derror_t setNodePrescribedTotalPotential(uint64_t nodeIndex, double prescribedTotalPotential);

    //Get water data
    double getNodeWaterContent(uint64_t nodeIndex);
    double getNodeMaximumWaterContent(uint64_t nodeIndex);
    double getNodeAvailableWaterContent(uint64_t nodeIndex);
    double getNodeWaterDeficit(uint64_t nodeIndex, double fieldCapacity);
    double getNodeDegreeOfSaturation(uint64_t nodeIndex);
    /*not used*/ double getNodeWaterConductivity(uint64_t nodeIndex);
    double getNodeMatricPotential(uint64_t nodeIndex);
    double getNodeTotalPotential(uint64_t nodeIndex);
    double getNodePond(uint64_t nodeIndex);
    /*not used*/ double getNodeMaxWaterFlow(uint64_t nodeIndex, linkType_t linkDirection);
    /*not used*/ double getNodeSumLateralWaterFlow(uint64_t nodeIndex);
    double getNodeSumLateralWaterFlowIn(uint64_t nodeIndex);
    double getNodeSumLateralWaterFlowOut(uint64_t nodeIndex);
    double getNodeBoundaryWaterFlow(uint64_t nodeIndex);

    double getTotalBoundaryWaterFlow(boundaryType_t boundaryType);
    double getTotalWaterContent();
    double getWaterStorage();
    double getWaterMBR();

    //Set heat data
    SF3Derror_t setNodeHeatSinkSource(uint64_t nodeIndex, double heatSinkSource);
    SF3Derror_t setNodeTemperature(uint64_t nodeIndex, double temperature);
    SF3Derror_t setNodeBoundaryFixedTemperature(uint64_t nodeIndex, double fixedTemperature, double depth);
    SF3Derror_t setNodeBoundaryHeightWind(uint64_t nodeIndex, double heightWind);
    SF3Derror_t setNodeBoundaryHeightTemperature(uint64_t nodeIndex, double heightTemperature);
    SF3Derror_t setNodeBoundaryNetIrradiance(uint64_t nodeIndex, double netIrradiance);
    SF3Derror_t setNodeBoundaryTemperature(uint64_t nodeIndex, double temperature);
    SF3Derror_t setNodeBoundaryRelativeHumidity(uint64_t nodeIndex, double relativeHumidity);
    SF3Derror_t setNodeBoundaryRoughness(uint64_t nodeIndex, double roughness);
    SF3Derror_t setNodeBoundaryWindSpeed(uint64_t nodeIndex, double windSpeed);

    //Get heat data
    double getNodeTemperature(uint64_t nodeIndex);
    double getNodeHeatConductivity(uint64_t nodeIndex);
    double getNodeVapor(uint64_t nodeIndex);
    /*Rename?*/ double getNodeHeat(uint64_t nodeIndex, double h);                                           //nodeHeatStorage
    /*Rename?*/ double getNodeHeatFlux(uint64_t nodeIndex, linkType_t linkDirection, fluxTypes_t fluxType); //nodeMaxHeatFlux
    double getNodeBoundaryAdvectiveFlux(uint64_t nodeIndex);
    double getNodeBoundaryLatentFlux(uint64_t nodeIndex);
    double getNodeBoundaryRadiativeFlux(uint64_t nodeIndex);
    double getNodeBoundarySensibleFlux(uint64_t nodeIndex);
    double getNodeBoundaryAerodynamicConductance(uint64_t nodeIndex);
    double getNodeBoundarySoilConductance(uint64_t nodeIndex);
    double getHeatMBR();
    double getHeatMBE();

    //Computations
    void computePeriod(double timePeriod);      //move to a SF3Derror_t return
    double computeStep(double maxTimeStep);

}
#endif // SOILFLUXES3DNEW_H
