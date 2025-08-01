#ifndef SOILFLUXES3D_NEW_H
#define SOILFLUXES3D_NEW_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::New
{
    //Inizializazion and memory management
    SF3Derror_t initializeSF3D(uint64_t nrNodes, uint16_t nrLayers, uint8_t nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes);
    SF3Derror_t inizializeBalance();

    SF3Derror_t cleanSF3D();

    uint32_t setThreadsNumber(uint32_t nrThreads);

    //Create types
    SF3Derror_t setSoilProperties(uint16_t nrSoil, uint16_t nrHorizon, double VG_alpha, double VG_n, double VG_m, double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay);
    SF3Derror_t setSurfaceProperties(uint16_t surfaceIndex, double roughness);

    //Set core data
    SF3Derror_t setNumericalParameters(double minDeltaT, double maxDeltaT, uint16_t maxIterationNumber, uint16_t maxApproximationsNumber, uint8_t ResidualToleranceExponent, uint8_t MBRThresholdExponent);
    SF3Derror_t setHydraulicProperties(WRCModel waterRetentionCurve, meanType_t conductivityMeanType, float conductivityHorizVertRatio);

    //Set topology
    SF3Derror_t setNode(uint64_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope = 0, double boundaryArea = 0);
    SF3Derror_t setNodeLink(uint64_t nodeIndex, uint64_t linkIndex, linkType_t direction, double interfaceArea);
    /* not used */ SF3Derror_t setCulvert(uint64_t nodeIndex, double roughness, double slope, double width, double height);

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
    /*not used*/ double getWaterMBR();

    //Computations
    /*not used*/ void computePeriod();
    double computeStep(double maxTimeStep);

}
#endif // SOILFLUXES3D_NEW_H
