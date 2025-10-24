#pragma once

#include "macro.h"
#include "types.h"

namespace soilFluxes3D { inline namespace v2
{
    //Inizializazion and memory management
    SF3Derror_t initializeSF3D(SF3Duint_t nrNodes, u16_t nrLayers, u8_t nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes, heatFluxSaveMode_t HFsm = heatFluxSaveMode_t::None);
    SF3Derror_t initializeBalance();
    SF3Derror_t initializeLog(const std::string& logPath, const std::string& projectName);

    SF3Derror_t cleanSF3D();
    SF3Derror_t closeLog();

    SF3Derror_t initializeHeatFlag(heatFluxSaveMode_t saveModeHeat, bool isComputeAdvectiveFlux, bool isComputeLatentHeat);

    u32_t setThreadsNumber(u32_t nrThreads);

    //Create types
    SF3Derror_t setSoilProperties(u16_t nrSoil, u16_t nrHorizon, double VG_alpha, double VG_n, double VG_m, double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay);
    SF3Derror_t setSurfaceProperties(u16_t surfaceIndex, double roughness);

    //Set core data
    SF3Derror_t setNumericalParameters(double minDeltaT, double maxDeltaT, u16_t maxIterationNumber, u16_t maxApproximationsNumber, u8_t ResidualToleranceExponent, u8_t MBRThresholdExponent);
    SF3Derror_t setHydraulicProperties(WRCModel waterRetentionCurve, meanType_t conductivityMeanType, float conductivityHorizVertRatio);

    //Set topology
    SF3Derror_t setCulvert(SF3Duint_t nodeIndex, double roughness, double slope, double width, double height);
    SF3Derror_t setNode(SF3Duint_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope = 0, double boundaryArea = 0);
    SF3Derror_t setNodeLink(SF3Duint_t nodeIndex, SF3Duint_t linkIndex, linkType_t direction, double interfaceArea);
    SF3Derror_t setNodeBoundary(SF3Duint_t nodeIndex, boundaryType_t boundaryType, double slope, double boundaryArea);

    //Set soil data
    SF3Derror_t setNodeSoil(SF3Duint_t nodeIndex, u16_t soilIndex, u16_t horizonIndex);
    SF3Derror_t setNodeSurface(SF3Duint_t nodeIndex, u16_t surfaceIndex);

    //Set water data
    SF3Derror_t setNodePond(SF3Duint_t nodeIndex, double pond);
    SF3Derror_t setNodeWaterContent(SF3Duint_t nodeIndex, double waterContent);
    SF3Derror_t setNodeDegreeOfSaturation(SF3Duint_t nodeIndex, double degreeOfSaturation);
    SF3Derror_t setNodeMatricPotential(SF3Duint_t nodeIndex, double matricPotential);
    /*not used*/ SF3Derror_t setNodeTotalPotential(SF3Duint_t nodeIndex, double totalPotential);
    SF3Derror_t setNodeWaterSinkSource(SF3Duint_t nodeIndex, double waterSinkSource);
    /*not used*/ SF3Derror_t setNodePrescribedTotalPotential(SF3Duint_t nodeIndex, double prescribedTotalPotential);

    //Get water data
    double getNodeWaterContent(SF3Duint_t nodeIndex);
    double getNodeMaximumWaterContent(SF3Duint_t nodeIndex);
    double getNodeAvailableWaterContent(SF3Duint_t nodeIndex);
    double getNodeWaterDeficit(SF3Duint_t nodeIndex, double fieldCapacity);
    double getNodeDegreeOfSaturation(SF3Duint_t nodeIndex);
    /*not used*/ double getNodeWaterConductivity(SF3Duint_t nodeIndex);
    double getNodeMatricPotential(SF3Duint_t nodeIndex);
    double getNodeTotalPotential(SF3Duint_t nodeIndex);
    double getNodePond(SF3Duint_t nodeIndex);
    /*not used*/ double getNodeMaxWaterFlow(SF3Duint_t nodeIndex, linkType_t linkDirection);
    /*not used*/ double getNodeSumLateralWaterFlow(SF3Duint_t nodeIndex);
    double getNodeSumLateralWaterFlowIn(SF3Duint_t nodeIndex);
    double getNodeSumLateralWaterFlowOut(SF3Duint_t nodeIndex);
    double getNodeBoundaryWaterFlow(SF3Duint_t nodeIndex);

    double getTotalBoundaryWaterFlow(boundaryType_t boundaryType);
    double getTotalWaterContent();
    double getWaterStorage();
    double getWaterMBR();

    //Set heat data
    SF3Derror_t setNodeHeatSinkSource(SF3Duint_t nodeIndex, double heatSinkSource);
    SF3Derror_t setNodeTemperature(SF3Duint_t nodeIndex, double temperature);
    SF3Derror_t setNodeBoundaryFixedTemperature(SF3Duint_t nodeIndex, double fixedTemperature, double depth);
    SF3Derror_t setNodeBoundaryHeightWind(SF3Duint_t nodeIndex, double heightWind);
    SF3Derror_t setNodeBoundaryHeightTemperature(SF3Duint_t nodeIndex, double heightTemperature);
    SF3Derror_t setNodeBoundaryNetIrradiance(SF3Duint_t nodeIndex, double netIrradiance);
    SF3Derror_t setNodeBoundaryTemperature(SF3Duint_t nodeIndex, double temperature);
    SF3Derror_t setNodeBoundaryRelativeHumidity(SF3Duint_t nodeIndex, double relativeHumidity);
    SF3Derror_t setNodeBoundaryRoughness(SF3Duint_t nodeIndex, double roughness);
    SF3Derror_t setNodeBoundaryWindSpeed(SF3Duint_t nodeIndex, double windSpeed);

    //Get heat data
    double getNodeTemperature(SF3Duint_t nodeIndex);
    double getNodeHeatConductivity(SF3Duint_t nodeIndex);
    __cudaSpec double getNodeVapor(SF3Duint_t nodeIndex);
    double getNodeHeatStorage(SF3Duint_t nodeIndex, double h);
    double getNodeHeatMaxFlux(SF3Duint_t nodeIndex, linkType_t linkDirection, fluxTypes_t fluxType);
    double getNodeBoundaryAdvectiveFlux(SF3Duint_t nodeIndex);
    double getNodeBoundaryLatentFlux(SF3Duint_t nodeIndex);
    double getNodeBoundaryRadiativeFlux(SF3Duint_t nodeIndex);
    double getNodeBoundarySensibleFlux(SF3Duint_t nodeIndex);
    double getNodeBoundaryAerodynamicConductance(SF3Duint_t nodeIndex);
    double getNodeBoundarySoilConductance(SF3Duint_t nodeIndex);
    double getHeatMBR();
    double getHeatMBE();

    //Computations
    void computePeriod(double timePeriod);      //move to a SF3Derror_t return
    double computeStep(double maxTimeStep);
}}
