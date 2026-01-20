#include "soilFluxes3D.h"
#include "heat.h"
#include "solver.h"
#include "types_cpu.h"
#include "water.h"
#include "soilPhysics.h"
#include "otherFunctions.h"
#include "commonConstants.h"

using namespace soilFluxes3D::v2;
using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Water;
using namespace soilFluxes3D::v2::Math;

namespace soilFluxes3D::v2
{
    extern __cudaMngd Solver* solver;
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;
    extern __cudaMngd simulationFlags_t simulationFlags;
}

namespace soilFluxes3D::v2::Heat
{
    __cudaSpec bool isHeatNode(SF3Duint_t nodeIndex)
    {
        return (simulationFlags.computeHeat && !nodeGrid.surfaceFlag[nodeIndex]);
    }

    SF3Derror_t initializeHeatBalance()
    {
        balanceDataWholePeriod.heatSinkSource = 0.;
        balanceDataCurrentPeriod.heatSinkSource = 0.;
        balanceDataCurrentTimeStep.heatSinkSource = 0.;
        balanceDataPreviousTimeStep.heatSinkSource = 0.;

        balanceDataWholePeriod.heatMBE = 0.;
        balanceDataCurrentPeriod.heatMBE = 0.;
        balanceDataCurrentTimeStep.heatMBE = 0.;

        balanceDataWholePeriod.heatMBE = 0.;
        balanceDataCurrentPeriod.heatMBE = 0.;
        balanceDataCurrentTimeStep.heatMBE = 0.;

        double heatStorage = computeCurrentHeatStorage();
        balanceDataWholePeriod.heatStorage = heatStorage;
        balanceDataCurrentPeriod.heatStorage = heatStorage;
        balanceDataCurrentTimeStep.heatStorage = heatStorage;
        balanceDataPreviousTimeStep.heatStorage = heatStorage;

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t resetFluxValues(bool flagHeat, bool flagWater)
    {
        if(!simulationFlags.computeHeat)
            return SF3Derror_t::MissingDataError;

        if(flagHeat)
        {
            switch(simulationFlags.HFsaveMode)
            {
                case heatFluxSaveMode_t::All:
                    for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                        for(auto index : heatFluxIndeces)
                            hostFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(index)], nodeGrid.numNodes, noDataD);
                    break;

                case heatFluxSaveMode_t::Total:
                    for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                        hostFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::HeatTotal)], nodeGrid.numNodes, noDataD);
                    break;

                default:
                    break;
            }
        }

        if(flagWater)
        {
            switch(simulationFlags.HFsaveMode)
            {
                case heatFluxSaveMode_t::All:
                    for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                        for(auto index : waterFluxIndeces)
                            hostFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(index)], nodeGrid.numNodes, noDataD);
                    break;

                default:
                    break;
            }
        }

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t saveWaterFluxValues(double dtHeat, double dtWater)
    {
        __parfor(__ompStatus)    //Try swap loop order and/or use collapse(2)
        for (SF3Duint_t nIdx = 0; nIdx < nodeGrid.numNodes; ++nIdx)
            for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                if(nodeGrid.linkData[lIdx].linkType[nIdx] != linkType_t::NoLink)
                    saveNodeWaterFluxes(nIdx, lIdx, dtHeat, dtWater);

        return SF3Derror_t::SF3Dok;
    }

    __cudaSpec SF3Derror_t saveNodeWaterFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater)
    {
        SF3Duint_t dIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        double scrAvgH = getNodeH_fromTimeSteps(nIdx, dtHeat, dtWater);
        double dstAvgH = getNodeH_fromTimeSteps(dIdx, dtHeat, dtWater);

        //Flux value as saved in the iteration matrix for water process
        double matrixValue = getMatrixElement(nIdx, dIdx);

        double isothermalLiquidFlux = matrixValue * (scrAvgH - dstAvgH);

        bool deepLink = !nodeGrid.surfaceFlag[nIdx] && !nodeGrid.surfaceFlag[dIdx];
        double isothermalVaporFlux = deepLink ? computeIsothermalVaporFlux(nIdx, lIdx, dtHeat, dtWater)                 : 0.;
        double thermalLiquidFlux   = deepLink ? computeThermalLiquidFlux(nIdx, lIdx, processType::Heat, dtHeat, dtWater): 0.;
        double thermalVaporflux    = deepLink ? computeThermalVaporFlux(nIdx, lIdx, processType::Heat, dtHeat, dtWater) : 0.;

        nodeGrid.linkData[lIdx].waterFlux[nIdx] = static_cast<float>(isothermalLiquidFlux - isothermalVaporFlux / WATER_DENSITY + thermalLiquidFlux);
        nodeGrid.linkData[lIdx].vaporFlux[nIdx] = static_cast<float>(isothermalVaporFlux + thermalVaporflux);

        if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::All)
        {
            nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::WaterLiquidIsothermal)][nIdx] = static_cast<float>(isothermalLiquidFlux);
            nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::WaterLiquidThermal)][nIdx] = static_cast<float>(thermalLiquidFlux);
            nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::WaterVaporIsothermal)][nIdx] = static_cast<float>(isothermalVaporFlux);
            nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::WaterVaporThermal)][nIdx] = static_cast<float>(thermalVaporflux);
        }

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t saveHeatFluxValues(double dtHeat, double dtWater)
    {
        if(!simulationFlags.computeHeat)
            return SF3Derror_t::SF3Dok;

        if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::None)
            return SF3Derror_t::SF3Dok;

        __parfor(__ompStatus)
        for (SF3Duint_t nIdx = 0; nIdx < nodeGrid.numNodes; ++nIdx)
        {
            if(nodeGrid.surfaceFlag[nIdx])
                continue;

            for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                saveNodeHeatFluxes(nIdx, lIdx, dtHeat, dtWater);
        }
        return SF3Derror_t::SF3Dok;
    }

    __cudaSpec SF3Derror_t saveNodeHeatFluxes(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater)
    {
        if(nodeGrid.linkData[lIdx].linkType[nIdx] == linkType_t::NoLink)
            return SF3Derror_t::ParameterError;     //Check if is correct

        SF3Duint_t linkedIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        if(!isHeatNode(nIdx) || !isHeatNode(linkedIdx))
            return SF3Derror_t::ParameterError;     //Check if is correct

        double matrixValue = getMatrixElement(nIdx, linkedIdx);

        double heatWF = solver->getHeatWF();
        double heatDiff = matrixValue * (nodeGrid.heatData.temperature[nIdx] - nodeGrid.heatData.temperature[linkedIdx]) * heatWF +
                          matrixValue * (nodeGrid.heatData.oldTemperature[nIdx] - nodeGrid.heatData.oldTemperature[linkedIdx]) * (1. - heatWF);

        switch(simulationFlags.HFsaveMode)
        {
            case heatFluxSaveMode_t::Total:
                saveNodeHeatSpecificFlux(nIdx, lIdx, fluxTypes_t::HeatTotal, heatDiff);
                break;

            case heatFluxSaveMode_t::All:
                if(simulationFlags.computeHeatVapor)
                {
                    double thermalLatentFlux = computeThermalVaporFlux(nIdx, lIdx, processType::Heat, dtHeat, dtWater) * computeLatentVaporizationHeat(nodeGrid.heatData.temperature[nIdx] - ZEROCELSIUS);
                    saveNodeHeatSpecificFlux(nIdx, lIdx, fluxTypes_t::HeatLatentThermal, thermalLatentFlux);
                    heatDiff -= thermalLatentFlux;
                }
                saveNodeHeatSpecificFlux(nIdx, lIdx, fluxTypes_t::HeatDiffusive, heatDiff);
                break;
            default:
                return SF3Derror_t::ParameterError;
        }

        return SF3Derror_t::SF3Dok;
    }

    __cudaSpec SF3Derror_t saveNodeHeatSpecificFlux(SF3Duint_t nIdx, u8_t lIdx, fluxTypes_t fluxType, double fluxValue)
    {
        if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::None)
            return SF3Derror_t::ParameterError;

        fluxValue = static_cast<float>(fluxValue);

        double& totalFlux = nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::HeatTotal)][nIdx];
        totalFlux = static_cast<float>((totalFlux == noDataD) ? fluxValue : totalFlux + fluxValue);

        if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::All)
            nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxType)][nIdx] = fluxValue;

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t updateConductance()
    {
        if(!simulationFlags.computeHeat)
            return SF3Derror_t::MissingDataError;

        __parfor(__ompStatus)
        for (SF3Duint_t nIdx = 0; nIdx < nodeGrid.numNodes; ++nIdx)
        {
            if(nodeGrid.boundaryData.boundaryType[nIdx] != boundaryType_t::HeatSurface)
                continue;

            //check if is a heat node?

            nodeGrid.boundaryData.aerodynamicConductance[nIdx] = computeNodeAerodynamicConductance(nIdx);

            if(simulationFlags.computeWater)
            {
                double theta = computeNodeTheta_fromSignedPsi(nIdx, nodeGrid.waterData.pressureHead[nIdx] - nodeGrid.z[nIdx]);
                nodeGrid.boundaryData.soilConductance[nIdx] = 1. / computeSoilSurfaceResistance(theta);
            }
        }

        return SF3Derror_t::SF3Dok;
    }

    bool updateBoundaryHeatData(double maxTimeStep, double& actualTimeStep)
    {
        double heatBoundaryCourantValue = 0.;
        double* tempCourantValues = nullptr;
        hostAlloc(tempCourantValues, nodeGrid.numNodes);

        __parfor(__ompStatus)
        for(SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            if(!isHeatNode(nodeIndex))
                continue;

            nodeGrid.heatData.heatFlux[nodeIndex] = nodeGrid.heatData.heatSinkSource[nodeIndex];

            if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
                continue;

            double upLinkArea = nodeGrid.linkData[0].interfaceArea[nodeIndex];  //linkType_t::Up

            switch(nodeGrid.boundaryData.boundaryType[nodeIndex])
            {
                case boundaryType_t::HeatSurface:
                    nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex] = 0.;
                    nodeGrid.boundaryData.sensibleFlux[nodeIndex] = 0.;
                    nodeGrid.boundaryData.latentFlux[nodeIndex] = 0.;
                    nodeGrid.boundaryData.radiativeFlux[nodeIndex] = 0.;

                    if(nodeGrid.boundaryData.netIrradiance[nodeIndex] != noDataD)
                        nodeGrid.boundaryData.radiativeFlux[nodeIndex] = nodeGrid.boundaryData.netIrradiance[nodeIndex];

                    nodeGrid.boundaryData.sensibleFlux[nodeIndex] += computeNodeAtmosphericSensibleHeatFlux(nodeIndex);

                    if(simulationFlags.computeWater && simulationFlags.computeHeatVapor)
                        nodeGrid.boundaryData.latentFlux[nodeIndex] += computeNodeAtmosphericLatentHeatFlux(nodeIndex) / upLinkArea;

                    if(simulationFlags.computeWater && simulationFlags.computeHeatAdvection)
                    {
                        double advT = nodeGrid.boundaryData.temperature[nodeIndex];

                        //Advective heat from rain
                        double waterFlux = nodeGrid.linkData[0].waterFlux[nodeIndex];   //linkType_t::Up
                        if(waterFlux > 0.)
                            nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex] = waterFlux * HEAT_CAPACITY_WATER * advT / upLinkArea;

                        //Advective heat from evaporation/condensation
                        if(nodeGrid.boundaryData.waterFlowRate[nodeIndex] < 0.)
                            advT = nodeGrid.heatData.temperature[nodeIndex];

                        nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex] += nodeGrid.boundaryData.waterFlowRate[nodeIndex] * WATER_DENSITY * HEAT_CAPACITY_WATER_VAPOR * advT / upLinkArea;
                    }

                    nodeGrid.heatData.heatFlux[nodeIndex] += upLinkArea * (nodeGrid.boundaryData.radiativeFlux[nodeIndex] +
                                                                           nodeGrid.boundaryData.sensibleFlux[nodeIndex] +
                                                                           nodeGrid.boundaryData.latentFlux[nodeIndex] +
                                                                           nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex]);
                    double heatCapacity;
                    heatCapacity = computeNodeHeatCapacity(nodeIndex, nodeGrid.waterData.oldPressureHead[nodeIndex], nodeGrid.heatData.oldTemperature[nodeIndex]);
                    tempCourantValues[nodeIndex] = std::fabs(nodeGrid.heatData.heatFlux[nodeIndex]) * maxTimeStep / (heatCapacity * nodeGrid.size[nodeIndex]);
                    break;

                case boundaryType_t::FreeDrainage:
                case boundaryType_t::PrescribedTotalWaterPotential:
                    if(simulationFlags.computeWater && simulationFlags.computeHeatAdvection)
                    {
                        double waterFlux = nodeGrid.boundaryData.waterFlowRate[nodeIndex];
                        double advT = (waterFlux < 0) ? nodeGrid.heatData.temperature[nodeIndex] : nodeGrid.boundaryData.fixedTemperatureValue[nodeIndex];

                        nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex] = waterFlux * HEAT_CAPACITY_WATER * advT / upLinkArea;
                        nodeGrid.heatData.heatFlux[nodeIndex] += upLinkArea * nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex];
                    }

                    if(nodeGrid.boundaryData.fixedTemperatureValue[nodeIndex] != noDataD)
                    {
                        double avgH = computeMean(nodeGrid.waterData.pressureHead[nodeIndex], nodeGrid.waterData.oldPressureHead[nodeIndex], meanType_t::Arithmetic);
                        double boundaryHeatK = computeNodeHeatSoilConductivity(nodeIndex, nodeGrid.heatData.temperature[nodeIndex], avgH - nodeGrid.z[nodeIndex]);
                        double deltaT = nodeGrid.boundaryData.fixedTemperatureValue[nodeIndex] - nodeGrid.heatData.temperature[nodeIndex];

                        nodeGrid.heatData.heatFlux[nodeIndex] += boundaryHeatK * deltaT / nodeGrid.boundaryData.fixedTemperatureDepth[nodeIndex] * upLinkArea;
                    }
                    break;
                default:
                    break;
            }
        }

        //Reduction
        __parforop(__ompStatus, max, heatBoundaryCourantValue)
        for(SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
            heatBoundaryCourantValue = SF3Dmax(heatBoundaryCourantValue, tempCourantValues[nodeIndex]);

        hostFree(tempCourantValues);

        //Check Value and operation on deltaT
        double minTimeStep = solver->getMinTimeStep();
        if(heatBoundaryCourantValue > 1. && maxTimeStep > minTimeStep)
        {
            actualTimeStep = SF3Dmax(minTimeStep, maxTimeStep / heatBoundaryCourantValue);
            if(actualTimeStep > 1.)
                actualTimeStep = std::floor(actualTimeStep);

            return false;
        }
        return true;
    }

    double computeCurrentHeatStorage(double dtWater, double dtHeat)
    {
        double heatStorage = 0.;
        __parforop(__ompStatus, +, heatStorage)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
        {
            if(nodeGrid.surfaceFlag[nodeIdx])
                continue;

            double nodeH = (dtHeat != noDataD && dtWater != noDataD) ? getNodeH_fromTimeSteps(nodeIdx, dtHeat, dtWater) : nodeGrid.waterData.pressureHead[nodeIdx];
            heatStorage += getNodeHeatStorage(nodeIdx, nodeH - nodeGrid.z[nodeIdx]);
        }
        return heatStorage;
    }

    double computeCurrentHeatSinkSource(double dtHeat)
    {
        double heatSinkSource = 0.;
        __parforop(__ompStatus, +, heatSinkSource)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
        {
            if(nodeGrid.surfaceFlag[nodeIdx])
                continue;

            if(nodeGrid.heatData.heatFlux[nodeIdx] != 0.)
                heatSinkSource += nodeGrid.heatData.heatFlux[nodeIdx] * dtHeat;
        }
        return heatSinkSource;
    }

    void evaluateHeatBalance(double dtHeat, double dtWater)
    {
        //Heat sink/source
        double heatSinkSource = computeCurrentHeatSinkSource(dtHeat);
        balanceDataCurrentTimeStep.heatSinkSource = heatSinkSource;

        //Heat storage
        double heatStorage = computeCurrentHeatStorage(dtWater, dtHeat);
        balanceDataCurrentTimeStep.heatStorage = heatStorage;

        //Heat MBE
        double deltaHeatStorage = balanceDataCurrentTimeStep.heatStorage - balanceDataPreviousTimeStep.heatStorage;
        balanceDataCurrentTimeStep.heatMBE = deltaHeatStorage - balanceDataCurrentTimeStep.heatSinkSource;

        //Heat MBR
        double referenceHeat = SF3Dmax(1., std::fabs(balanceDataCurrentTimeStep.heatSinkSource));
        balanceDataCurrentTimeStep.heatMBR = balanceDataCurrentTimeStep.heatMBE / referenceHeat;
    }

    void updateHeatBalanceData()
    {
        balanceDataPreviousTimeStep.heatStorage = balanceDataCurrentTimeStep.heatStorage;
        balanceDataPreviousTimeStep.heatSinkSource = balanceDataCurrentTimeStep.heatSinkSource;
        balanceDataCurrentPeriod.heatSinkSource += balanceDataCurrentTimeStep.heatSinkSource;
    }

    void updateHeatBalanceDataWholePeriod()
    {
        balanceDataWholePeriod.heatSinkSource += balanceDataCurrentPeriod.heatSinkSource;
        double deltaStoragePeriod = balanceDataCurrentTimeStep.heatStorage - balanceDataCurrentPeriod.heatStorage;
        double deltaStorageHistorical = balanceDataCurrentTimeStep.heatStorage - balanceDataWholePeriod.heatStorage;

        balanceDataCurrentPeriod.heatMBE = deltaStoragePeriod - balanceDataCurrentPeriod.heatSinkSource;
        balanceDataWholePeriod.heatMBE = deltaStorageHistorical - balanceDataWholePeriod.heatSinkSource;

        double referenceHeat = SF3Dmax(1., std::fabs(balanceDataWholePeriod.heatSinkSource));
        balanceDataWholePeriod.heatMBR = balanceDataWholePeriod.heatMBE / referenceHeat;

        balanceDataCurrentPeriod.heatStorage = balanceDataCurrentTimeStep.heatStorage;
    }


    __cudaSpec bool computeHeatLinkFluxes(double& matrixElement, SF3Duint_t& matrixIndex, SF3Duint_t nodeIndex, u8_t linkIndex, double dtHeat, double dtWater)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return false;

        SF3Duint_t linkedNodeIndex = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];

        if(!isHeatNode(linkedNodeIndex))
            return false;

        matrixElement = conduction(nodeIndex, linkIndex, dtHeat, dtWater);
        matrixIndex = linkedNodeIndex;

        if(!simulationFlags.computeWater)
            return true;

        double advectiveFlux = 0., latentFlux = 0.;

        if(simulationFlags.computeHeatVapor)
        {
            latentFlux = computeIsothermalLatentHeatFlux(nodeIndex, linkIndex, dtHeat, dtWater);
            saveNodeHeatSpecificFlux(nodeIndex, linkIndex, fluxTypes_t::HeatLatentIsothermal, latentFlux);
        }

        if(simulationFlags.computeHeatAdvection)
        {
            advectiveFlux = computeAdvectiveFlux(nodeIndex, linkIndex);
            saveNodeHeatSpecificFlux(nodeIndex, linkIndex, fluxTypes_t::HeatAdvective, advectiveFlux);
        }

        nodeGrid.waterData.invariantFluxes[nodeIndex] += advectiveFlux + latentFlux;
        return true;
    }


    /*!
     * \brief compute the thermal liquid flux at lIdx link of the nIdx node
     * \param dtHeat: time step for the heat calculation [s]
     * \param dtWater: time step for the water calculation [s]
     * \param process: type of calculation that call the function [Water/Heat]
     * \return Thermal liquid flux [m3 s-1]
     */
    __cudaSpec double computeThermalLiquidFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat, double dtWater)
    {
        SF3Duint_t dIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];
        double srcAvgT, dstAvgT, srcAvgH, dstAvgH;
        switch(process)
        {
            case processType::Water:
                if(!simulationFlags.computeWater)
                    return noDataD;

                srcAvgT = getNodeMeanTemperature(nIdx);
                dstAvgT = getNodeMeanTemperature(dIdx);
                srcAvgH = nodeGrid.waterData.pressureHead[nIdx] - nodeGrid.z[nIdx];
                dstAvgH = nodeGrid.waterData.pressureHead[dIdx] - nodeGrid.z[dIdx];
                break;
            case processType::Heat:
                if(!simulationFlags.computeHeat)
                    return noDataD;

                srcAvgT = nodeGrid.heatData.temperature[nIdx];
                dstAvgT = nodeGrid.heatData.temperature[dIdx];
                if(dtHeat != dtWater)
                {
                    srcAvgH = computeMean(getNodeH_fromTimeSteps(nIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[nIdx], meanType_t::Arithmetic) - nodeGrid.z[nIdx];
                    dstAvgH = computeMean(getNodeH_fromTimeSteps(dIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[dIdx], meanType_t::Arithmetic) - nodeGrid.z[dIdx];
                }
                else
                {
                    srcAvgH = computeMean(nodeGrid.waterData.pressureHead[nIdx], nodeGrid.waterData.oldPressureHead[nIdx], meanType_t::Arithmetic) - nodeGrid.z[nIdx];
                    dstAvgH = computeMean(nodeGrid.waterData.pressureHead[dIdx], nodeGrid.waterData.oldPressureHead[dIdx], meanType_t::Arithmetic) - nodeGrid.z[dIdx];
                }
                break;
            default:
                return noDataD;
        }

        //Thermal liquid conductivity
        double scrTLK = computeThermalLiquidConductivity(srcAvgT - ZEROCELSIUS, srcAvgH, nodeGrid.waterData.waterConductivity[nIdx]);
        double dstTLK = computeThermalLiquidConductivity(dstAvgT - ZEROCELSIUS, dstAvgH, nodeGrid.waterData.waterConductivity[dIdx]);
        double avgTLK = computeMean(scrTLK, dstTLK);

        //Flow density [m s-1]
        double flowDensity = avgTLK * (dstAvgT - srcAvgT) / nodeDistance3D(nIdx, dIdx);
        //Flow [m3 s-1]
        double flow = flowDensity * nodeGrid.linkData[lIdx].interfaceArea[nIdx];

        return flow;
    }

    /*!
     * \brief compute the thermal vapor flux at lIdx link of the nIdx node
     * \param dtHeat: time step for the heat calculation [s]
     * \param dtWater: time step for the water calculation [s]
     * \param process: type of calculation that call the function [Water/Heat]
     * \return Thermal vapor flux [m3 s-1]
     */
    __cudaSpec double computeThermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, processType process, double dtHeat, double dtWater)
    {
        SF3Duint_t dIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];
        double srcAvgT, dstAvgT, srcAvgH, dstAvgH;
        switch(process)
        {
            case processType::Water:
                if(!simulationFlags.computeWater)
                    return noDataD;

                srcAvgT = getNodeMeanTemperature(nIdx);
                dstAvgT = getNodeMeanTemperature(dIdx);
                srcAvgH = nodeGrid.waterData.pressureHead[nIdx] - nodeGrid.z[nIdx];
                dstAvgH = nodeGrid.waterData.pressureHead[dIdx] - nodeGrid.z[dIdx];
                break;
            case processType::Heat:
                if(!simulationFlags.computeHeat)
                    return noDataD;

                srcAvgT = nodeGrid.heatData.temperature[nIdx];
                dstAvgT = nodeGrid.heatData.temperature[dIdx];
                srcAvgH = computeMean(getNodeH_fromTimeSteps(nIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[nIdx], meanType_t::Arithmetic) - nodeGrid.z[nIdx];
                dstAvgH = computeMean(getNodeH_fromTimeSteps(dIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[dIdx], meanType_t::Arithmetic) - nodeGrid.z[dIdx];
                break;
            default:
                return noDataD;
        }

        //Thermal liquid conductivity
        double scrTVK = computeNodeThermalVaporConductivity(nIdx, srcAvgT, srcAvgH);
        double dstTVK = computeNodeThermalVaporConductivity(dIdx, dstAvgT, dstAvgH);
        double avgTVK = computeMean(scrTVK, dstTVK);

        //Flow density [m s-1]
        double flowDensity = avgTVK * (dstAvgT - srcAvgT) / nodeDistance3D(nIdx, dIdx);
        //Flow [m3 s-1]
        double flow = flowDensity * nodeGrid.linkData[lIdx].interfaceArea[nIdx];

        return flow;
    }

    /*!
     * \brief compute the isothermal vapor flux at lIdx link of the nIdx node
     * \param dtHeat: time step for the heat calculation [s]
     * \param dtWater: time step for the water calculation [s]
     * \return Isothermal vapor flux [kg s-1]
     */
    __cudaSpec double computeIsothermalVaporFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater)
    {
        SF3Duint_t dIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        double srcAvgH = computeMean(getNodeH_fromTimeSteps(nIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[nIdx], meanType_t::Arithmetic) - nodeGrid.z[nIdx];
        double dstAvgH = computeMean(getNodeH_fromTimeSteps(dIdx, dtHeat, dtWater), nodeGrid.waterData.oldPressureHead[dIdx], meanType_t::Arithmetic) - nodeGrid.z[dIdx];

        //Isothermal vapor conductivity [kg s m-3]
        double scrIVK = computeNodeIsothermalVaporConductivity(nIdx, nodeGrid.heatData.temperature[nIdx], srcAvgH);
        double dstIVK = computeNodeIsothermalVaporConductivity(dIdx, nodeGrid.heatData.temperature[dIdx], dstAvgH);
        double avgIVK = computeMean(scrIVK, dstIVK);

        //Water matric potential [J kg-1] = [m2 s-2]
        double srcPsi = srcAvgH * GRAVITY;
        double dstPsi = dstAvgH * GRAVITY;
        double deltaPsi = dstPsi - srcPsi;

        //Flux [kg s-1]
        double flux = avgIVK * deltaPsi / nodeDistance3D(nIdx, dIdx) *  nodeGrid.linkData[lIdx].interfaceArea[nIdx];

        return flux;
    }

    /*!
     * \brief compute the isothermal latent heat flux at lIdx link of the nIdx node
     * \param dtHeat: time step for the heat calculation [s]
     * \param dtWater: time step for the water calculation [s]
     * \return Isothermal latent heat flux [W]
     */
    __cudaSpec double computeIsothermalLatentHeatFlux(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater)
    {
        SF3Duint_t dIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        //Latent heat of vaporization [J kg-1]
        double nLambda = computeLatentVaporizationHeat(nodeGrid.heatData.temperature[nIdx] - ZEROCELSIUS);
        double lLambda = computeLatentVaporizationHeat(nodeGrid.heatData.temperature[dIdx] - ZEROCELSIUS);
        double avgLambda = computeMean(nLambda, lLambda, meanType_t::Arithmetic);

        return avgLambda * computeIsothermalVaporFlux(nIdx, lIdx, dtHeat, dtWater);
    }

    /*!
     * \brief compute the advective liquid water heat flux at lIdx link of the nIdx node
     * \return Advective liquid water heat flux [W]
     */
    __cudaSpec double computeAdvectiveFlux(SF3Duint_t nIdx, u8_t lIdx) //TO DO: change param lIdx with a linkData_t const ref
    {
        SF3Duint_t linkedNodeIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        double liquidWaterFlux = nodeGrid.linkData[lIdx].waterFlux[nIdx];
        double liquidAdvT = nodeGrid.heatData.temperature[(liquidWaterFlux < 0.) ? nIdx : linkedNodeIdx];

        double AdvFluxCourant = HEAT_CAPACITY_WATER * liquidWaterFlux;

        double vaporWaterFlux = nodeGrid.linkData[lIdx].vaporFlux[nIdx];
        double vaporAdvT = nodeGrid.heatData.temperature[(vaporWaterFlux < 0.) ? nIdx : linkedNodeIdx];

        double vapFluxCourant = HEAT_CAPACITY_WATER_VAPOR * vaporWaterFlux;

        return AdvFluxCourant * liquidAdvT + vapFluxCourant * vaporAdvT;
    }

    __cudaSpec double getLinkHeatFlux(const linkData_t& linkData, SF3Duint_t srcIndex, fluxTypes_t fluxType)
    {
        if(!simulationFlags.computeHeat)
            return noDataD;

        switch(simulationFlags.HFsaveMode)
        {
            case heatFluxSaveMode_t::Total:
                if(fluxType == fluxTypes_t::HeatTotal)
                    return linkData.fluxes[toUnderlyingT(fluxType)][srcIndex];
                return noDataD;
                break;
            case heatFluxSaveMode_t::All:
                return linkData.fluxes[toUnderlyingT(fluxType)][srcIndex];
                break;
            default:
                return noDataD;
        }
    }

    __cudaSpec double conduction(SF3Duint_t nIdx, u8_t lIdx, double dtHeat, double dtWater)
    {
        SF3Duint_t linkedNodeIdx = nodeGrid.linkData[lIdx].linkIndex[nIdx];

        double distance = nodeDistance3D(nIdx, linkedNodeIdx);
        double zeta = nodeGrid.linkData[lIdx].interfaceArea[nIdx] / distance;

        double nodeH = getNodeH_fromTimeSteps(nIdx, dtHeat, dtWater);
        double linkH = getNodeH_fromTimeSteps(linkedNodeIdx, dtHeat, dtWater);

        double nodeAvgH = computeMean(nodeH, nodeGrid.waterData.oldPressureHead[nIdx], meanType_t::Arithmetic) - nodeGrid.z[nIdx];
        double linkAvgH = computeMean(linkH, nodeGrid.waterData.oldPressureHead[linkedNodeIdx], meanType_t::Arithmetic) - nodeGrid.z[linkedNodeIdx];

        double nodeK = computeNodeHeatSoilConductivity(nIdx, nodeGrid.heatData.temperature[nIdx], nodeAvgH);
        double linkK = computeNodeHeatSoilConductivity(linkedNodeIdx, nodeGrid.heatData.temperature[linkedNodeIdx], linkAvgH);

        double meanK = computeMean(nodeK, linkK);
        return zeta * meanK;
    }


    double GaussSeidelHeatCPU(VectorCPU& vectorX, const MatrixCPU& matrixA, const VectorCPU& vectorB)
    {
        double infinityNorm = -1;
        for(SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            if(nodeGrid.surfaceFlag[rowIdx])
                continue;

            if(matrixA.values[rowIdx][0] == 0.)
                continue;

            double newXvalue = vectorB.values[rowIdx];
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                newXvalue -= matrixA.values[rowIdx][colIdx] * vectorX.values[matrixA.columnIndeces[rowIdx][colIdx]];

            double deltaX = std::fabs(newXvalue - vectorX.values[rowIdx]);
            vectorX.values[rowIdx] = newXvalue;
            infinityNorm = std::max(infinityNorm, deltaX);
        }

        return infinityNorm;
    }

    /* TODO
     *
     */
    __cudaSpec double getNodeH_fromTimeSteps(SF3Duint_t nodeIndex, double dtHeat, double dtWater)
    {
        double deltaH = nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.waterData.oldPressureHead[nodeIndex];
        return nodeGrid.waterData.oldPressureHead[nodeIndex] + deltaH * dtHeat / dtWater;
    }

    /*!
     * \brief compute the nodeIndex node thermal soil conductivity according to Campbell et al. Soil Sci. 158:307-313
     * \param T: node temperature [K]
     * \param h: node water matric potential [m]
     * \return soil thermal conductivity [W m-1 K-1]
     */
    __cudaSpec double computeNodeHeatSoilConductivity(SF3Duint_t nodeIndex, double T, double h)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        double celsiusT = T - ZEROCELSIUS;

        //Volume fraction of water          [m3 m-3]
        double wVolFrac = computeNodeTheta_fromSignedPsi(nodeIndex, h);

        //Volume fraction of solids         [m3 m-3]
        double sVolFrac = 1. - nodeSoil.Theta_s;

        //Volume fraction of air            [m3 m-3]
        double aVolFrac = nodeSoil.Theta_s - wVolFrac;

        //Water return flow factor          []  (same in air conductivity)
        double wRetFlowFactor = computeWaterReturnFlowFactor(wVolFrac, T, nodeSoil.clay);

        //Thermal conductivity of water     [W m-1 K-1]
        double wThermalK = 0.554 + 0.0024 * celsiusT - 0.00000987 * celsiusT * celsiusT;

        //Thermal conductivity of air       [W m-1 K-1]
        double aThermalK = computeNodeHeatAirConductivity(nodeIndex, T, h);

        //Thermal conductivity of fluids    [W m-1 K-1]
        double fThermalK = aThermalK + wRetFlowFactor * (wThermalK - aThermalK);

        //deVries shape factor              []  (assume same for all mineral soils)
        double ga = 0.088;
        //Shape factor                      []
        double gc = 1. - 2. * ga;

        //Air weighting factor              []
        double aWFactor = (2. / (1. + (aThermalK / fThermalK - 1.) * ga) + 1. / (1. + (aThermalK / fThermalK - 1.) * gc)) / 3.;
        //Water weighting factor            []
        double wWFactor = (2. / (1. + (wThermalK / fThermalK - 1.) * ga) + 1. / (1. + (wThermalK / fThermalK - 1.) * gc)) / 3.;
        //Solid weighting factor            []
        double sWFactor = (2. / (1. + (mineralHK / fThermalK - 1.) * ga) + 1. / (1. + (mineralHK / fThermalK - 1.) * gc)) / 3.;

        //Total thermal conductivity[W m-1 K-1]
        double nodeConductivity = (wVolFrac * wWFactor * wThermalK + aVolFrac * aWFactor * aThermalK + sVolFrac * sWFactor * mineralHK)
                                    / (wWFactor * wVolFrac + aWFactor * aVolFrac + sWFactor * sVolFrac);
        return nodeConductivity;
    }

    /*!
     * \brief compute the nodeIndex node thermal air conductivity
     * \param T: node temperature [K]
     * \param h: node water matric potential [m]
     * \return air thermal conductivity [W m-1 K-1]
     */
    __cudaSpec double computeNodeHeatAirConductivity(SF3Duint_t nodeIndex, double T, double h)
    {
        double celsiusT = T - ZEROCELSIUS;

        //Thermal conductivity of (dry) air     [W m-1 K-1]
        double aThermalK = 0.024 + 0.0000773 * celsiusT - 0.000000026 * celsiusT * celsiusT;

        if(simulationFlags.computeWater)
        {
            //Latent heat of vaporization       [J kg-1]
            double lamdba = computeLatentVaporizationHeat(celsiusT);

            //Non isothermal vapor conductivity [kg m-1 s-1 K-1]
            double niVK = computeNodeThermalVaporConductivity(nodeIndex, T, h);

            //Thermal conductivity of air     [W m-1 K-1]
            aThermalK += lamdba * niVK;
        }

        return aThermalK;
    }

    /*!
     * \brief compute the nodeIndex node thermal vapor conductivity
     * \param T: node temperature [K]
     * \param h: node water matric potential [m]
     * \return thermal vapor conductivity [kg m-1 s-1 K-1]
     */
    __cudaSpec double computeNodeThermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        double celsiusT = T - ZEROCELSIUS;

        //Total air pressure        [Pa]
        double aPressure = computePressure_fromAltitude(nodeGrid.z[nodeIndex]);

        //Volumetric water content  [m3 m-3]
        double theta = computeNodeTheta_fromSignedPsi(nodeIndex, h);

        //Vapor diffusivity         [m2 s-1]
        double vDiff = computeSoilVaporDiffusivity(nodeSoil.Theta_s, theta, T);

        //Saturation vapor pressure [Pa]
        double svPressure = computeSaturationVaporPressure(celsiusT);

        //Slope of saturation vapor pressure        [Pa K-1]
        double svpSlope = computeSVPSlope(celsiusT, svPressure / 1000);

        //Slope of saturation vapor concentration   [kg m-3 K-1]
        double svcSlope = svpSlope * MH2O * computeAirMolarDensity(aPressure, T) / aPressure;

        //Vapor concentration       [kg m-3]
        double vConcentration = computeVapor_fromPsiTemp(h, T);

        //Vapor pressure            [Pa]
        double vPressure = computeVaporPressure_fromConcentration(vConcentration, T);

        //Relative humidity         []
        double rH = vPressure / svPressure;

        //Degree of saturation      []
        double satDegree = theta / nodeSoil.Theta_s;

        //Enhancement factor        [] (Cass et al. 1984)
        double eta = 9.5 + 3. * satDegree - 8.5 * std::exp(-std::pow((1. + 2.6 / std::sqrt(nodeSoil.clay)) * satDegree, 4));

        return eta * vDiff * svcSlope * rH;
    }

    /*!
     * \brief compute the nodeIndex node isothermal vapor conductivity
     * \param T: node temperature [K]
     * \param h: node water matric potential [m]
     * \return isothermal vapor conductivity [kg s m-3]
     */
   __cudaSpec double computeNodeIsothermalVaporConductivity(SF3Duint_t nodeIndex, double T, double h)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        //Volumetric water content  [m3 m-3]
        double theta = computeNodeTheta_fromSignedPsi(nodeIndex, h);

        //Vapor diffusivity         [m2 s-1]
        double vDiff = computeSoilVaporDiffusivity(nodeSoil.Theta_s, theta, T);

        //Vapor concentration       [kg m-3]
        double vConc = computeVapor_fromPsiTemp(h, T);

        return (vDiff * vConc * MH2O) / (R_GAS * T);
    }

    /*!
     * \brief compute the nodeIndex node volumetric heat capacity
     * \param h: node water matric potential [m]
     * \param T: node temperature [K]
     * \return volumetric heat capacity [J m-3 K-1]
     */
    __cudaSpec double computeNodeHeatCapacity(SF3Duint_t nodeIndex, double h, double T)
    {
        double theta = computeNodeTheta_fromSignedPsi(nodeIndex, h);

        double bulkDensity = estimateNodeBulkDensity(nodeIndex);
        double heatCapacity = (bulkDensity / QUARTZ_DENSITY) * HEAT_CAPACITY_MINERAL + theta * HEAT_CAPACITY_WATER;

        if(simulationFlags.computeHeatVapor)
            heatCapacity += computeNodeVaporThetaV(nodeIndex, h, T) * HEAT_CAPACITY_AIR;

        return heatCapacity;
    }

    /*!
     * \brief compute the nodeIndex node vapor volumetric water equivalent
     * \param h: node water matric potential [m]
     * \param T: node temperature [K]
     * \return vapor volumetric water equivalent [m3 m-3]
     */
    __cudaSpec double computeNodeVaporThetaV(SF3Duint_t nodeIndex, double h, double T)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        double theta = computeNodeTheta_fromSignedPsi(nodeIndex, h);
        double vaporConcentration = computeVapor_fromPsiTemp(h, T);

        return vaporConcentration / WATER_DENSITY * (nodeSoil.Theta_s - theta);
    }


    /*!
     * \brief compute the nodeIndex node aerodynamic conductance for heat and vapor (Campbell Norman, 1998)
     * \return aerodynamic conductance [m s-1]
     */
    __cudaSpec double computeNodeAerodynamicConductance(SF3Duint_t nodeIndex)
    {
        //Node parameters
        double heightTemperature = nodeGrid.boundaryData.heightTemperature[nodeIndex];
        double heightWind = nodeGrid.boundaryData.heightWind[nodeIndex];
        double soilSurfaceTemperature = nodeGrid.heatData.temperature[nodeIndex];
        double rHeight = nodeGrid.boundaryData.roughnessHeight[nodeIndex];
        double airTemperature = nodeGrid.boundaryData.temperature[nodeIndex];
        double windSpeed = SF3Dmax(nodeGrid.boundaryData.windSpeed[nodeIndex], 0.01);

        //Zero place displacement   [m]
        double zeroPlane = 0.77 * rHeight;

        //Surface roughness parameters for momentum and heat
        double rMomentum = 0.13 * rHeight;
        double rHeat = 0.2 * rMomentum;

        //Diabatic correction factors for momentum and for heat
        double psiM = 0.;
        double psiH = 0.;

        // Volumetric specific heat of air [J m-3 K-1]
        double cH = computeAirVolumetricSpecificHeat(computePressure_fromAltitude(heightWind), airTemperature);

        bool isFirstIteration = true;
        double dH = 1000;           // [W m-2]
        double oldH = noDataD;      // [W m-2]
        double K = noDataD;         // [m s-1]
        for(u8_t counter = 0; counter < 100; ++counter)
        {
            // Friction velocity [m s-1]
            double uStar = VON_KARMAN_CONST * windSpeed / (std::log((heightWind - zeroPlane + rMomentum) / rMomentum) + psiM);

            // Aereodynamic conductance [m s-1]
            K = VON_KARMAN_CONST * uStar / (std::log((heightTemperature - zeroPlane + rHeat) / rHeat) + psiH);

            // Sensible heat flux [W m-2]
            double H = K * cH * (soilSurfaceTemperature - airTemperature);

            // Stability parameter
            double sP = -VON_KARMAN_CONST * heightWind * GRAVITY * H / (cH * airTemperature * (std::pow(uStar, 3)));

            // Check stability
            if(sP > 0)
            {
                psiH = 6 * std::log(1 + sP);
                psiM = psiH;
            }
            else
            {
                psiH = -2 * std::log((1 + std::sqrt(1 - 16 * sP)) / 2);
                psiM = 0.6 * psiH;
            }

            if(isFirstIteration)
            {
                isFirstIteration = false;
            }
            else
            {
                dH = std::fabs(H - oldH);
                if(dH < 0.01)
                    break;
            }

            oldH = H;
        }

        return K;
    }

    /*!
     * \brief compute the nodeIndex node atmospheric sensible heat flux
     * \return atmospheric sensible flux [W m-2]
     */
    __cudaSpec double computeNodeAtmosphericSensibleHeatFlux(SF3Duint_t nodeIndex)
    {
        SF3Duint_t nodeLinkUpIndex = nodeGrid.linkData[0].linkIndex[nodeIndex];    //linkType_t::Up
        if(!nodeGrid.surfaceFlag[nodeLinkUpIndex])
            return 0.;

        double pressure = computePressure_fromAltitude(nodeGrid.z[nodeIndex]);
        double deltaT = nodeGrid.boundaryData.temperature[nodeIndex] - nodeGrid.heatData.temperature[nodeIndex];
        double airVSH = computeAirVolumetricSpecificHeat(pressure, nodeGrid.boundaryData.temperature[nodeIndex]);

        return airVSH * deltaT * nodeGrid.boundaryData.aerodynamicConductance[nodeIndex];
    }

    /*!
     * \brief compute the nodeIndex node atmospheric latent heat flux (evaporation/condensation)
     * \return atmospheric latent flux [W]
     */
    __cudaSpec double computeNodeAtmosphericLatentHeatFlux(SF3Duint_t nodeIndex)
    {
        SF3Duint_t upIndex = nodeGrid.linkData[0].linkIndex[nodeIndex];    //linkType_t::Up
        if(!nodeGrid.surfaceFlag[upIndex])
            return 0.;

        double lambda = computeLatentVaporizationHeat(nodeGrid.heatData.temperature[nodeIndex] - ZEROCELSIUS);
        return nodeGrid.boundaryData.waterFlowRate[nodeIndex] * WATER_DENSITY * lambda;
    }

    /*!
     * \brief compute the nodeIndex node boundary vapor flux (evaporation/condensation)
     * \return vapor flux       [kg m-2 s-1]
     */
    __cudaSpec double computeNodeAtmosphericLatentVaporFlux(SF3Duint_t nodeIndex)
    {
        SF3Duint_t upIndex = nodeGrid.linkData[0].linkIndex[nodeIndex];    //linkType_t::Up
        if(!nodeGrid.surfaceFlag[upIndex])
            return 0.;

        //Atmospheric vapor content [kg m-3]
        double satPressure = computeSaturationVaporPressure(nodeGrid.boundaryData.temperature[nodeIndex] - ZEROCELSIUS);
        double satConcentration = computeVaporConcentration_fromPressure(satPressure, nodeGrid.boundaryData.temperature[nodeIndex]);
        double boundaryVapor = satConcentration * (nodeGrid.boundaryData.relativeHumidity[nodeIndex] / 100.);

        //Surface water vapor content [kg m-3]
        double deltaVapor = boundaryVapor - getNodeVapor(nodeIndex);

        //Total conductance [m s-1]
        double totalConductance = 1. / ((1. / nodeGrid.boundaryData.aerodynamicConductance[nodeIndex]) + (1. / nodeGrid.boundaryData.soilConductance[nodeIndex]));

        //Vapor flux [kg m-2 s-1]
        return deltaVapor * totalConductance;
    }

    /*!
     * \brief compute the nodeIndex node boundary vapor flux from surface water
     * \return vapor flux       [kg m-2 s-1]
     */
    __cudaSpec double computeNodeAtmosphericLatentSurfaceWaterFlux(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.surfaceFlag[nodeIndex])
            return 0.;

        if(nodeGrid.linkData[1].linkType[nodeIndex] == linkType_t::NoLink)
            return 0.;

        SF3Duint_t downIndex = nodeGrid.linkData[1].linkIndex[nodeIndex];      //linkType_t::Down

        if(nodeGrid.boundaryData.boundaryType[downIndex] != boundaryType_t::HeatSurface)
            return 0.;

        //Atmospheric vapor content [kg m-3]
        double satPressure = computeSaturationVaporPressure(nodeGrid.boundaryData.temperature[downIndex] - ZEROCELSIUS);
        double satConcentration = computeVaporConcentration_fromPressure(satPressure, nodeGrid.boundaryData.temperature[downIndex]);
        double boundaryVapor = satConcentration * (nodeGrid.boundaryData.relativeHumidity[downIndex] / 100.);

        //Surface water vapor content [kg m-3] (assuming water temperature is the same of atmosphere)
        double deltaVapor = boundaryVapor - satConcentration;

        //Vapor flux [kg m-2 s-1] (using aerodynamic conductance of index below, boundary for heat)
        return deltaVapor * nodeGrid.boundaryData.aerodynamicConductance[downIndex];
    }


    /*!
     * \brief estimate the nodeIndex node bulk density
     * \return bulk density [Mg m-3]
     */
    __cudaSpec double estimateNodeBulkDensity(SF3Duint_t nodeIndex)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        double particleDensity = estimateSoilParticleDensity(nodeSoil.organicMatter);
        double totalPorosity = nodeSoil.Theta_s;

        return (1. - totalPorosity) * particleDensity;
    }

    /*!
     * \brief estimate soil particle density (Driessen, 1986)
     * \param organicMatter: fraction of organic matter
     * \return soil particle density [Mg m-3]
     */
    __cudaSpec double estimateSoilParticleDensity(double organicMatter)
    {
        if(organicMatter == noDataD)
            organicMatter = 0.02;

        return 1. / ((1. - organicMatter) / QUARTZ_DENSITY + organicMatter / 1.43);
    }

    /*!
     * \brief compute vapor concentration from matric potential and temperature
     * \param h: matric potential [J kg-1]
     * \param T: temperature [K]
     * \return vapor concentration [kg m-3]
     */
    __cudaSpec double computeVapor_fromPsiTemp(double h, double T)
    {
        double svp = computeSaturationVaporPressure(T - ZEROCELSIUS);
        double svc = computeVaporConcentration_fromPressure(svp, T);
        double rh = computeSoilRelativeHumidity(h, T);

        return svc * rh;
    }

    /*!
     * \brief compute latent heat of vaporization as function of temperature
     * \param T: temperature [°C]
     * \return latent heat of vaporization [J kg-1]
     */
    __cudaSpec double computeLatentVaporizationHeat(double T)
    {
        return (2501000. - 2369.2 * T);
    }

    /*!
     * \brief compute the water return flow factor (Campbell, 1994)
     * \param theta: volumetric water content [m3 m-3]
     * \param T: temperature [K]
     * \param clayFraction: fraction of clay in the soil mixrure
     * \return water return flow factor [-]
     */
    __cudaSpec double computeWaterReturnFlowFactor(double theta, double T, double clayFraction)
    {
        //Cutoff water content []
        double wc0 = 0.078 + 0.33 * clayFraction;

        if(theta < 0.01 * wc0)
            return 0.;

        //Power []
        double q0 = 2.52 + 7.25 * clayFraction;
        double q = q0 * std::pow(T / 303., 2.);

        return 1. / (1. + std::pow(theta / wc0, -q));
    }

    /*!
     * \brief compute the atmospheric pressure at a fixed height (Allen et al., 1994)
     * \param height: altitude above the sea level [m]
     * \return atmospheric pressure [Pa]
     */
    __cudaSpec double computePressure_fromAltitude(double height)
    {
        return P0 * std::pow(1 + height * LAPSE_RATE_MOIST_AIR / TP0, -GRAVITY / (LAPSE_RATE_MOIST_AIR * R_DRY_AIR));
    }

    /*!
     * \brief compute the soil vapor diffusivity
     * \param T: temperature [K]
     * \return vapor diffusivity [m2 s-1]
     */
    __cudaSpec double computeSoilVaporDiffusivity(double thetaS, double theta, double T)
    {
        const double beta = 0.66;   // [] Penman 1940
        const double m = 1.;        // [] Penman 1940

        double binaryDiffusivity = computeVaporBinaryDiffusivity(T);	// [m2 s-1]
        double airFilledPorosity = thetaS - theta;                      // [m3 m-3]

        return binaryDiffusivity * beta * std::pow(airFilledPorosity, m);
    }

    /*!
     * \brief compute the soil relative humidity
     * \param h: pressure head [m]
     * \param T: temperature [K]
     * \return soil relative humidity [-]
     */
    __cudaSpec double computeSoilRelativeHumidity(double h, double T)
    {
        return std::exp(MH2O * h * GRAVITY / (R_GAS * T));
    }

    /*!
     * \brief compute the soil surface resistance (Van De Griend and Owe, 1994)
     * \param thetaTop:
     * \return soil surface resistance [s m-1]
     */
    __cudaSpec double computeSoilSurfaceResistance(double thetaTop)
    {
        return 10 * std::exp(0.3563 * (THETAMIN - thetaTop) * 100);
    }

    /*!
     * \brief compute the saturation vapor pressure as function of the temperature (semplified August-Roche-Magnus)
     * \param T: temperature [°C]
     * \return saturation vapor pressure [Pa]
     */
    __cudaSpec double computeSaturationVaporPressure(double T)
    {
        return 611 * std::exp(17.502 * T / (T + 240.97));
    }

    /*!
     * \brief compute the slope of saturation vapor pressure curve
     * \param T: temperature [°C]
     * \param svp: saturation vapor pressure [kPa]
     * \return slope [kPa °C-1]
     */
    __cudaSpec double computeSVPSlope(double T, double svp)
    {
        return (4098. * svp / ((237.3 + T) * (237.3 + T)));
    }

    /*!
     * \brief compute the air molar density (Boyle-Charles law)
     * \param pressure: air pressure [Pa]
     * \param T: temperature [K]
     * \return air molar density [mol m-3]
     */
    __cudaSpec double computeAirMolarDensity(double pressure, double T)
    {
        return 44.65 * (pressure / P0) * (ZEROCELSIUS / T);
    }

    /*!
     * \brief compute the volumetric specific heat of air
     * \param pressure: air pressure [Pa]
     * \param T: temperature [K]
     * \return air volumetric specific heat [J m-3 K-1]
     */
    __cudaSpec double computeAirVolumetricSpecificHeat(double pressure, double T)
    {
        return HEAT_CAPACITY_AIR_MOLAR * computeAirMolarDensity(pressure, T);
    }

    /*!
     * \brief compute the vapor pressure at a fixed temperature as function of the vapor concentration
     * \param concentration: vapor concentration [kg m-3]
     * \param T: temperature [K]
     * \return vapor pressure [Pa]
     */
    __cudaSpec double computeVaporPressure_fromConcentration(double concentration, double T)
    {
        return (concentration * R_GAS * T / MH2O);
    }

    /*!
     * \brief compute the vapor concentration at a fixed temperature as function of the vapor pressure
     * \param pressure: vapor pressure [Pa]
     * \param T: temperature [K]
     * \return vapor concentration [kg m-3]
     */
    __cudaSpec double computeVaporConcentration_fromPressure(double pressure, double T)
    {
        return (pressure * MH2O / (R_GAS * T));
    }

    /*!
     * \brief compute the binary vapor diffusivity (Do) (Bittelli, 2008)
     *          or the vapor diffusion coefficient in air (Dva) (Monteith, 1973)
     * \param T: temperature [K]
     * \return binary vapor diffusivity [m2 s-1]
     */
    __cudaSpec double computeVaporBinaryDiffusivity(double T)
    {
        return VAPOR_DIFFUSIVITY0 * std::pow(T / ZEROCELSIUS, 2.);
    }

    /*!
     * \brief compute the thermal liquid conductivity
     * \param T: temperature [°C]
     * \param h: pressure head [m]
     * \param ILK: isothermal liquid conductivity [m s-1]
     * \return thermal liquid conductivity [m2 s-1 K-1]
     */
    __cudaSpec double computeThermalLiquidConductivity(double T, double h, double ILK)
    {
        //Gain factor (temperature dependence of soil water retention curve) []
        double Gwt = 4.;
        //Derivative of surface tension with respect to temperature [g s-2 K-1]
        double dGammadT = -0.1425 - 0.000576 * T;

        return SF3Dmax(0., ILK * h * Gwt * dGammadT / GAMMA0);
    }


}
