#include <cassert>
#include <iostream>

#ifdef CUDA_ENABLED
    #include "gpusolver.h"
#endif
#include "cpusolver.h"

#include "water.h"
#include "soilPhysics.h"
#include "heat.h"
#include "otherFunctions.h"

#define EPSILON_CUSTOM 0.00001
#define DBL_EPSILON_CUSTOM 2.2204460492503131e-016

using namespace soilFluxes3D::New;
using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Math;
using namespace soilFluxes3D::Heat;

namespace soilFluxes3D::New
{
    extern __cudaMngd Solver* solver;
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;
    extern __cudaMngd simulationFlags_t simulationFlags;
}

namespace soilFluxes3D::Water
{
    /*!
     * \brief initializes the water balance variables
     * \return Ok/Error
     */
    SF3Derror_t initializeWaterBalance()
    {
        double twc = computeTotalWaterContent();
        balanceDataWholePeriod.waterStorage = twc;
        balanceDataCurrentPeriod.waterStorage = twc;
        balanceDataCurrentTimeStep.waterStorage = twc;
        balanceDataPreviousTimeStep.waterStorage = twc;

        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        //Reset link water flows
        for (uint8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
            hostReset(nodeGrid.linkData[linkIndex].waterFlowSum, nodeGrid.numNodes);

        //Reset boundary water flows
        hostReset(nodeGrid.boundaryData.waterFlowSum, nodeGrid.numNodes);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief computes the total water content
     * \return total water content [m3]
     */
    double computeTotalWaterContent()
    {
        if(!nodeGrid.isInitialized)
            return -1;

        double sum = 0.0;

        #pragma omp parallel for if(__ompStatus) reduction(+:sum)
        for (uint64_t idx = 0; idx < nodeGrid.numNodes; ++idx)
        {
            double theta = nodeGrid.surfaceFlag[idx] ? (nodeGrid.waterData.pressureHead[idx] - nodeGrid.z[idx]) : computeNodeTheta(idx);
            sum += theta * nodeGrid.size[idx];
        }

        return sum;
    }

    /*!
     * \brief computes the mass balance error of the current time step
     * \param deltaT    [s]
     */
    void computeCurrentMassBalance(double deltaT)
    {
        balanceDataCurrentTimeStep.waterStorage = computeTotalWaterContent();
        double deltaStorage = balanceDataCurrentTimeStep.waterStorage - balanceDataPreviousTimeStep.waterStorage;

        balanceDataCurrentTimeStep.waterSinkSource = computeWaterSinkSourceFlowsSum(deltaT);
        balanceDataCurrentTimeStep.waterMBE = deltaStorage - balanceDataCurrentTimeStep.waterSinkSource;

        // minimum reference water storage [m3] as % of current storage
        double timePercentage = 0.01 * SF3Dmax(deltaT, 60.) / HOUR_SECONDS;
        double minRefWaterStorage = balanceDataCurrentTimeStep.waterStorage * timePercentage;
        minRefWaterStorage = SF3Dmax(minRefWaterStorage, 0.001);

        // Reference water for computation of mass balance error ratio
        // when the water sink/source is too low, use the reference water storage
        double referenceWater = SF3Dmax(std::fabs(balanceDataCurrentTimeStep.waterSinkSource), minRefWaterStorage);     // [m3]

        balanceDataCurrentTimeStep.waterMBR = balanceDataCurrentTimeStep.waterMBE / referenceWater;
    }

    /*!
     * \brief computes sum of water sink/source flows
     * \param deltaT    [s]
     * \return sum of water sink/source [m3]
     */
    double computeWaterSinkSourceFlowsSum(double deltaT)
    {
        double sum = 0;

        #pragma omp parallel for if(__ompStatus) reduction(+:sum)
        for (uint64_t idx = 0; idx < nodeGrid.numNodes; ++idx)
            if(nodeGrid.waterData.waterFlow[idx] != 0)
                sum += nodeGrid.waterData.waterFlow[idx] * deltaT;

        return sum;
    }


    void updateWaterBalanceDataWholePeriod()
    {
        balanceDataWholePeriod.waterSinkSource += balanceDataCurrentPeriod.waterSinkSource;
        double deltaStoragePeriod = balanceDataCurrentTimeStep.waterStorage - balanceDataCurrentPeriod.waterStorage;
        double deltaStorageHistorical = balanceDataCurrentTimeStep.waterStorage - balanceDataWholePeriod.waterStorage;

        balanceDataCurrentPeriod.waterMBE = deltaStoragePeriod - balanceDataCurrentPeriod.waterSinkSource;
        balanceDataWholePeriod.waterMBE = deltaStorageHistorical - balanceDataWholePeriod.waterSinkSource;

        double referenceWater = SF3Dmax(0.001, std::fabs(balanceDataWholePeriod.waterSinkSource));
        balanceDataCurrentTimeStep.waterMBR = balanceDataWholePeriod.waterMBE / referenceWater;

        balanceDataCurrentPeriod.waterStorage = balanceDataCurrentTimeStep.waterStorage;
    }

    /*!
     * \brief evalutate the current water balance
     * \param approxNr number of iteration performed
     * \param bestMBRerror best mass balance ratio error achieved in the previous iterations
     * \param parameters solver parameters
     * \return evaluations of water balance
     */
    balanceResult_t evaluateWaterBalance(uint8_t approxNr, double& bestMBRerror, double deltaT, SolverParameters& parameters)
    {
        computeCurrentMassBalance(deltaT);

        double currMBRerror = std::fabs(balanceDataCurrentTimeStep.waterMBR);

        //Optimal error
        if(currMBRerror < parameters.MBRThreshold)
        {
            acceptStep(deltaT);

            //Check Stability (Courant)
            double currCWL = nodeGrid.waterData.CourantWaterLevel;
            // if((currCWL < parameters.CourantWaterThreshold) && (approxNr <= 3) && (currMBRerror < (0.5 * parameters.MBRThreshold)))
            // {
            //     //increase deltaT
            //     parameters.deltaTcurr = (currCWL < 0.5) ? (2 * parameters.deltaTcurr) : (parameters.deltaTcurr / currCWL);
            //     parameters.deltaTcurr = SF3Dmin(parameters.deltaTcurr, parameters.deltaTmax);
            //     if(parameters.deltaTcurr > 1.)
            //         parameters.deltaTcurr = std::floor(parameters.deltaTcurr);
            // }

            if((currCWL < parameters.CourantWaterThreshold) && (approxNr <= 3))
                parameters.deltaTcurr = 2 * parameters.deltaTcurr;

            return balanceResult_t::stepAccepted;
        }

        //Good error or first approximation
        if (approxNr == 0 || currMBRerror < bestMBRerror)
        {
            saveBestStep();
            bestMBRerror = currMBRerror;
        }

        //Critical error (unstable system) or last approximation
        if (approxNr == (parameters.maxApproximationsNumber - 1) || currMBRerror > (bestMBRerror * parameters.instabilityFactor))
        {
            if(deltaT > parameters.deltaTmin)
            {
                parameters.deltaTcurr = SF3Dmax(parameters.deltaTcurr / 2, parameters.deltaTmin);
                return balanceResult_t::stepHalved;
            }

            restoreBestStep(deltaT);
            acceptStep(deltaT);
            return balanceResult_t::stepAccepted;
        }

        return balanceResult_t::stepRefused;
    }


    void acceptStep(double deltaT)
    {
        /*! set current time step balance data as the previous one */
        balanceDataPreviousTimeStep.waterStorage = balanceDataCurrentTimeStep.waterStorage;
        balanceDataPreviousTimeStep.waterSinkSource = balanceDataCurrentTimeStep.waterSinkSource;

        /*! update balance data of current period */
        balanceDataCurrentPeriod.waterSinkSource += balanceDataCurrentTimeStep.waterSinkSource;

        /*! update sum of flow */
        #pragma omp parallel for if(__ompStatus)
        for (uint64_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            //Update link flows
            for(uint8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
                updateLinkFlux(nodeIndex, linkIndex, deltaT);

            //Update boundary flow
            if (nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::NoBoundary)
                nodeGrid.boundaryData.waterFlowSum[nodeIndex] += nodeGrid.boundaryData.waterFlowRate[nodeIndex] * deltaT;
        }
    }

    void saveBestStep() //TO DO: remove
    {
        std::memcpy(nodeGrid.waterData.bestPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double));
    }

    void restoreBestStep(double deltaT)
    {
        std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.bestPressureHeads, nodeGrid.numNodes * sizeof(double));

        #pragma omp parallel for if(__ompStatus)
        for (uint64_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
            if(!nodeGrid.surfaceFlag[nodeIndex])
            {
                nodeGrid.waterData.saturationDegree[nodeIndex] = computeNodeSe(nodeIndex);
                nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);
            }

        updateBoundaryWaterData(deltaT);
        computeCurrentMassBalance(deltaT);
    }

    __cudaSpec void updateLinkFlux(uint64_t nodeIndex, uint8_t linkIndex, double deltaT)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return;

        uint64_t linkedNodeIndex = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];
        double matrixValue = getMatrixElement(nodeIndex, linkedNodeIndex);
        nodeGrid.linkData[linkIndex].waterFlowSum[nodeIndex] += matrixValue * (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.waterData.pressureHead[linkedNodeIndex]) * deltaT;
    }

    void restorePressureHead() //TO DO: remove
    {
        std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHeads, nodeGrid.numNodes * sizeof(double));
    }

    void computeCapacity(VectorCPU& vectorC)
    {
        #pragma omp parallel for if(__ompStatus)
        for (uint64_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            nodeGrid.waterData.invariantFluxes[nodeIndex] = 0.;
            if(nodeGrid.surfaceFlag[nodeIndex])
                continue;

            //Compute hydraulic conductivity
            nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);

            double dThetadH = computeNodedThetadH(nodeIndex);
            vectorC.values[nodeIndex] = nodeGrid.size[nodeIndex] * dThetadH;

            if(simulationFlags.computeHeat && simulationFlags.computeHeatVapor)
                vectorC.values[nodeIndex] += nodeGrid.size[nodeIndex] * computeNodedThetaVdH(nodeIndex, getNodeMeanTemperature(nodeIndex), dThetadH);
        }
    }

    //TO DO: move to a CPUSolver method
    void computeLinearSystemElement(MatrixCPU& matrixA, VectorCPU& vectorB, const VectorCPU& vectorC, uint8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType)
    {
        #pragma omp parallel for if(__ompStatus)
        for (uint64_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            uint8_t linkIdx = 1;
            bool isLinked;

            //Compute flux up
            isLinked = computeLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 0, approxNum, deltaT, lateralVerticalRatio, linkType_t::Up, meanType);
            if(isLinked)
                linkIdx++;

            //Compute flox down
            isLinked = computeLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 1, approxNum, deltaT, lateralVerticalRatio, linkType_t::Down, meanType);
            if(isLinked)
                linkIdx++;

            //Compute flux lateral
            for(uint8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
            {
                isLinked = computeLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 2 + latIdx, approxNum, deltaT, lateralVerticalRatio, linkType_t::Lateral, meanType);
                if(isLinked)
                    linkIdx++;
            }

            matrixA.numColumns[rowIdx] = linkIdx;

            //TO DO: need to fill the not used columns of the row?

            //Compute diagonal element
            double sum = 0.;
            for(uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
            {
                sum += matrixA.values[rowIdx][colIdx];
                matrixA.values[rowIdx][colIdx] *= -1.;
            }
            matrixA.colIndeces[rowIdx][0] = rowIdx;
            matrixA.values[rowIdx][0] = (vectorC.values[rowIdx] / deltaT) + sum;

            //Compute b element
            vectorB.values[rowIdx] = ((vectorC.values[rowIdx] / deltaT) * nodeGrid.waterData.oldPressureHeads[rowIdx]) + nodeGrid.waterData.waterFlow[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx];

            //Preconditioning
            for(uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
                matrixA.values[rowIdx][colIdx] /= matrixA.values[rowIdx][0];

            vectorB.values[rowIdx] /= matrixA.values[rowIdx][0];
        }
    }

    __cudaSpec bool computeLinkFluxes(double& matrixElement, uint64_t& matrixIndex, uint64_t nodeIndex, uint8_t linkIndex, uint8_t approxNum, double deltaT, double lateralVerticalRatio, linkType_t linkType, meanType_t meanType)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return false;

        uint64_t linkedNodeIndex = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];
        double flowArea = nodeGrid.linkData[linkIndex].interfaceArea[nodeIndex];
        matrixIndex = linkedNodeIndex;

        if(nodeGrid.surfaceFlag[nodeIndex] && nodeGrid.surfaceFlag[linkedNodeIndex])
            matrixElement = runoff(nodeIndex, linkedNodeIndex, approxNum, deltaT, flowArea);
        else if (nodeGrid.surfaceFlag[nodeIndex] && !nodeGrid.surfaceFlag[linkedNodeIndex])
            matrixElement = infiltration(nodeIndex, linkedNodeIndex, deltaT, flowArea, meanType);
        else if (!nodeGrid.surfaceFlag[nodeIndex] && nodeGrid.surfaceFlag[linkedNodeIndex])
            matrixElement = infiltration(linkedNodeIndex, nodeIndex, deltaT, flowArea, meanType);
        else if (!nodeGrid.surfaceFlag[nodeIndex] && !nodeGrid.surfaceFlag[linkedNodeIndex])
            matrixElement = redistribution(nodeIndex, linkedNodeIndex, lateralVerticalRatio, flowArea, linkType, meanType);
        else
            return false;

        if(nodeGrid.surfaceFlag[nodeIndex] || nodeGrid.surfaceFlag[linkedNodeIndex])
            return true;

        if(!simulationFlags.computeHeat)
            return true;

        double thermalLiquidFlux = computeThermalLiquidFlux(nodeIndex, linkIndex, processType::Water);
        nodeGrid.waterData.invariantFluxes[nodeIndex] += thermalLiquidFlux;

        if(!simulationFlags.computeHeatVapor)
            return true;

        double thermalVaporFlux = computeThermalVaporFlux(nodeIndex, linkIndex, processType::Water);
        nodeGrid.waterData.invariantFluxes[nodeIndex] += thermalVaporFlux;

        return true;
    }

    __cudaSpec double runoff(uint64_t rowIdx, uint64_t colIdx, uint8_t approxNum, double deltaT, double flowArea)
    {
        double flux_i = (nodeGrid.waterData.waterFlow[rowIdx] * deltaT) / nodeGrid.size[rowIdx];
        double flux_j = (nodeGrid.waterData.waterFlow[colIdx] * deltaT) / nodeGrid.size[colIdx];

        double H_i = (approxNum != 0) ? nodeGrid.waterData.pressureHead[rowIdx] : nodeGrid.waterData.oldPressureHeads[rowIdx] + 0.5 * flux_i;
        double H_j = (approxNum != 0) ? nodeGrid.waterData.pressureHead[colIdx] : nodeGrid.waterData.oldPressureHeads[colIdx] + 0.5 * flux_j;

        double dH = std::fabs(H_i - H_j);

        if(dH < DBL_EPSILON_CUSTOM)
            return 0.;

        double z_i = nodeGrid.z[rowIdx] + nodeGrid.waterData.pond[rowIdx];
        double z_j = nodeGrid.z[colIdx] + nodeGrid.waterData.pond[colIdx];

        double H_max = SF3Dmax(H_i, H_j);
        double z_max = SF3Dmax(z_i, z_j);

        double H_s = H_max - z_max;

        if(H_s < 0.0001)
            return 0.;

        // Land depression
        if((H_i > H_j && z_i < z_j) || ((H_i < H_j && z_i > z_j)))
            H_s = SF3Dmin(H_s, dH);

        double cellDistance = nodeDistance2D(rowIdx, colIdx);
        double slope = dH / cellDistance;

        if(slope < EPSILON_CUSTOM)
            return 0.;

        double roughness = 0.5 * (nodeGrid.soilSurfacePointers[rowIdx].surfacePtr->roughness + nodeGrid.soilSurfacePointers[colIdx].surfacePtr->roughness);

        double v = std::pow(H_s, 2./3.) * std::sqrt(slope) / roughness;

        nodeGrid.waterData.partialCourantWaterLevels[rowIdx] = SF3Dmax(nodeGrid.waterData.partialCourantWaterLevels[rowIdx], v * deltaT / cellDistance);
        //atomicMaxDouble(&(nodeGrid.waterData.CourantWaterLevel), nodeGrid.waterData.tempCourantVector[maxTotalLink*rowIdx + colIdx]);

        return v * flowArea * H_s / dH;
    }

    __cudaSpec double infiltration(uint64_t surfNodeIdx, uint64_t soilNodeIdx, double deltaT, double flowArea, meanType_t meanType)
    {
        double cellDistance = nodeGrid.z[surfNodeIdx] - nodeGrid.z[soilNodeIdx];
        soilData_t& soilData = *(nodeGrid.soilSurfacePointers[soilNodeIdx].soilPtr);

        double boundaryFactor = 1.;
        switch(nodeGrid.boundaryData.boundaryType[soilNodeIdx])
        {
            case boundaryType_t::Urban:
                boundaryFactor = 0.1;
                break;
            case boundaryType_t::Road:
                boundaryFactor = 0.;        //TO DO: maybe can transformed in return 0.;
                break;
            default:
                break;
        }

        //Soil node saturated
        if(nodeGrid.waterData.pressureHead[soilNodeIdx] > nodeGrid.z[surfNodeIdx])
                return (soilData.K_sat * boundaryFactor * flowArea) / cellDistance;

        double surfH = 0.5 * (nodeGrid.waterData.pressureHead[surfNodeIdx] + nodeGrid.waterData.oldPressureHeads[surfNodeIdx]);
        double soilH = 0.5 * (nodeGrid.waterData.pressureHead[soilNodeIdx] + nodeGrid.waterData.oldPressureHeads[soilNodeIdx]);

        double surfaceWater = SF3Dmax(surfH - nodeGrid.z[surfNodeIdx], 0.);                            // [m]
        double prec_evapRate = nodeGrid.waterData.waterFlow[surfNodeIdx] / nodeGrid.size[surfNodeIdx];  // [m s-1]

        double maxInfRate = (surfaceWater / deltaT) + prec_evapRate;
        if(maxInfRate < DBL_EPSILON)
            return 0.;

        double dH = surfH - soilH;
        double maxK = maxInfRate * (cellDistance / dH);
        double meanK = computeMean(soilData.K_sat, nodeGrid.waterData.waterConductivity[soilNodeIdx], meanType);

        //TO DO: check if needed
        switch(nodeGrid.boundaryData.boundaryType[soilNodeIdx])
        {
            case boundaryType_t::Urban:
                meanK *= 0.1;
                break;
            case boundaryType_t::Road:
                meanK = 0.;
                break;
            default:
                break;
        }

        return (SF3Dmin(boundaryFactor * meanK, maxK) * flowArea) / cellDistance;
    }

    __cudaSpec double redistribution(uint64_t rowIdx, uint64_t colIdx, double lateralVerticalRatio, double flowArea, linkType_t linkType, meanType_t meanType)
    {
        double cellDistance;
        double rowK = nodeGrid.waterData.waterConductivity[rowIdx];
        double colK = nodeGrid.waterData.waterConductivity[colIdx];
        if(linkType == linkType_t::Lateral)
        {
            cellDistance = nodeDistance3D(rowIdx, colIdx);
            rowK *= lateralVerticalRatio;
            colK *= lateralVerticalRatio;
        }
        else
        {
            cellDistance = std::fabs(nodeGrid.z[rowIdx] - nodeGrid.z[colIdx]);
        }

        return (computeMean(rowK, colK, meanType) * flowArea) / cellDistance;
    }


    double JacobiWaterCPU(VectorCPU& vectorX, const MatrixCPU& matrixA, const VectorCPU& vectorB)
    {
        double infinityNorm = -1;

        double* tempX = nullptr;
        hostAlloc(tempX, double, vectorX.numElements);
        std::memcpy(tempX, vectorB.values, vectorB.numElements * sizeof(double));

        #pragma omp parallel for if(__ompStatus) reduction(max:infinityNorm)
        for(uint64_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            for(uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
                tempX[rowIdx] -= matrixA.values[rowIdx][colIdx] * vectorX.values[matrixA.colIndeces[rowIdx][colIdx]];

            if(nodeGrid.surfaceFlag[rowIdx] && tempX[rowIdx] < nodeGrid.z[rowIdx])
                tempX[rowIdx] = nodeGrid.z[rowIdx];

            double currentNorm = std::fabs(tempX[rowIdx] - vectorX.values[rowIdx]);

            double psi = std::fabs(tempX[rowIdx] - nodeGrid.z[rowIdx]);
            if(psi > 1.)
                currentNorm /= psi;

            if(currentNorm > infinityNorm)
                infinityNorm = currentNorm;
        }

        std::memcpy(vectorX.values, tempX, vectorX.numElements * sizeof(double));
        hostFree(tempX);
        return infinityNorm;
    }

    double GaussSeidelWaterCPU(VectorCPU& vectorX, const MatrixCPU &matrixA, const VectorCPU& vectorB)
    {
        double currentNorm = -1, infinityNorm = -1;

        for (uint64_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            double newCurrValue = vectorB.values[rowIdx];
            for (uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
                newCurrValue -= matrixA.values[rowIdx][colIdx] * vectorX.values[matrixA.colIndeces[rowIdx][colIdx]];

            if(nodeGrid.surfaceFlag[rowIdx] && newCurrValue < nodeGrid.z[rowIdx])
                newCurrValue = nodeGrid.z[rowIdx];

            currentNorm = std::fabs(newCurrValue - vectorX.values[rowIdx]);
            vectorX.values[rowIdx] = newCurrValue;

            double psi = newCurrValue - nodeGrid.z[rowIdx];
            if(psi > 1.)
                currentNorm /= psi;

            if(currentNorm > infinityNorm)
                infinityNorm = currentNorm;
        }

        return infinityNorm;
    }


    void updateBoundaryWaterData(double deltaT)
    {
        #pragma omp parallel for if(__ompStatus)
        for (uint64_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
        {
            //initialize: water sink.source
            nodeGrid.waterData.waterFlow[nodeIdx] = nodeGrid.waterData.waterSinkSource[nodeIdx];   //TO DO: evaluate move to a memcpy

            if(nodeGrid.boundaryData.boundaryType[nodeIdx] == boundaryType_t::NoBoundary)
                continue;

            nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0;   //TO DO: evaluate move to a memset

            switch(nodeGrid.boundaryData.boundaryType[nodeIdx])
            {
                case boundaryType_t::Runoff:
                    double avgH, hs, maxFlow, v, flow;
                    avgH = 0.5 * (nodeGrid.waterData.pressureHead[nodeIdx] + nodeGrid.waterData.oldPressureHeads[nodeIdx]);

                    hs = SF3Dmax(0., avgH - (nodeGrid.z[nodeIdx] + nodeGrid.waterData.pond[nodeIdx]));
                    if(hs < EPSILON_RUNOFF)
                        break;

                    // Maximum flow available during the time step [m3 s-1]
                    maxFlow = (hs * nodeGrid.size[nodeIdx]) / deltaT;

                    //Manning equation
                    assert(nodeGrid.surfaceFlag[nodeIdx]);
                    v = std::pow(hs, 2./3.) * std::sqrt(nodeGrid.boundaryData.boundarySlope[nodeIdx]) / nodeGrid.soilSurfacePointers[nodeIdx].surfacePtr->roughness;

                    flow = hs * v * nodeGrid.boundaryData.boundarySize[nodeIdx];
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -SF3Dmin(flow, maxFlow);
                    break;

                case boundaryType_t::FreeDrainage:
                    //Darcy unit gradient (use link node up)
                    assert(nodeGrid.linkData[0].linkType[nodeIdx] != linkType_t::NoLink);
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = - nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.linkData[0].interfaceArea[nodeIdx];
                    break;

                case boundaryType_t::FreeLateraleDrainage:
                    //Darcy gradient = slope
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = - nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.boundaryData.boundarySize[nodeIdx]
                                                                            * nodeGrid.boundaryData.boundarySlope[nodeIdx] * solver->getLVRatio();
                    break;

                case boundaryType_t::PrescribedTotalWaterPotential:
                    double L, boundaryPsi, boundaryZ, boundaryK, meanK, dH;
                    L = 1.;     // [m]
                    boundaryZ = nodeGrid.z[nodeIdx] - L;

                    boundaryPsi = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - boundaryZ;

                    boundaryK = (boundaryPsi >= 0)  ? nodeGrid.soilSurfacePointers[nodeIdx].soilPtr->K_sat
                                                    : computeMualemSoilConductivity(*(nodeGrid.soilSurfacePointers[nodeIdx].soilPtr), computeNodeSe_fromPsi(nodeIdx, std::fabs(boundaryPsi)));

                    meanK = computeMean(boundaryK, nodeGrid.waterData.waterConductivity[nodeIdx], solver->getMeanType());
                    dH = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - nodeGrid.waterData.pressureHead[nodeIdx];

                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = meanK * nodeGrid.boundaryData.boundarySize[nodeIdx] * (dH / L);
                    break;

                case boundaryType_t::HeatSurface:
                    if(!simulationFlags.computeHeat && !simulationFlags.computeHeatVapor)
                        break;

                    uint64_t upIndex;
                    upIndex = nodeGrid.linkData[toUnderlyingT(linkType_t::Up)].linkIndex[nodeIdx];
                    double surfaceWaterFraction;
                    surfaceWaterFraction = 0.;

                    if(upIndex != toUnderlyingT(linkType_t::NoLink))
                        surfaceWaterFraction = getNodeSurfaceWaterFraction(upIndex);

                    double soilEvaporation;
                    soilEvaporation = computeNodeAtmosphericLatentHeatFlux(nodeIdx) / WATER_DENSITY * nodeGrid.size[nodeIdx];

                    if(surfaceWaterFraction > 0.) //check upIndex != toUnderlyingT(linkType_t::NoLink) superfluo.
                    {
                        double surfEvaporation = computeNodeAtmosphericLatentSurfaceWaterFlux(nodeIdx) / WATER_DENSITY * nodeGrid.size[nodeIdx];

                        soilEvaporation *= (1. - surfaceWaterFraction);
                        surfEvaporation *= surfaceWaterFraction;

                        double waterVolume = (nodeGrid.waterData.pressureHead[nodeIdx] - nodeGrid.z[nodeIdx]) * nodeGrid.size[nodeIdx];
                        surfEvaporation = SF3Dmax(surfEvaporation, -waterVolume / deltaT);

                        if(nodeGrid.boundaryData.boundaryType[upIndex] != boundaryType_t::NoBoundary)
                            nodeGrid.boundaryData.waterFlowRate[upIndex] = surfEvaporation;
                        else
                            nodeGrid.waterData.waterFlow[upIndex] += surfEvaporation;
                    }

                    double thetaR, thetaS, thetaV;
                    thetaR = nodeGrid.soilSurfacePointers[nodeIdx].soilPtr->Theta_r;
                    thetaS = nodeGrid.soilSurfacePointers[nodeIdx].soilPtr->Theta_s;
                    thetaV = computeNodeTheta(nodeIdx);

                    soilEvaporation = (soilEvaporation < 0.) ? SF3Dmax(soilEvaporation, -(thetaV - thetaR) * nodeGrid.size[nodeIdx] / deltaT)
                                                             : SF3Dmin(soilEvaporation, (thetaS - thetaR) * nodeGrid.size[nodeIdx] / deltaT);

                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = soilEvaporation;
                    break;

                default:
                    assert(false);
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
                    break;
            }

            if(std::fabs(nodeGrid.boundaryData.waterFlowRate[nodeIdx]) < DBL_EPSILON)
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
            else
                nodeGrid.waterData.waterFlow[nodeIdx] += nodeGrid.boundaryData.waterFlowRate[nodeIdx];
        }

        //Culvert -> need to be moved in the switch block?
        if(!nodeGrid.culvertData.isActive)
            return;

        uint64_t culvertIndex = nodeGrid.culvertData.index;

        double culvertHeight = nodeGrid.culvertData.height;
        double culvertWidth = nodeGrid.culvertData.width;
        double culvertRoughness = nodeGrid.culvertData.roughness;
        double culvertSlope = nodeGrid.culvertData.slope;
        double waterLevel = 0.5 * (nodeGrid.waterData.pressureHead[culvertIndex] - nodeGrid.waterData.oldPressureHeads[culvertIndex]) - nodeGrid.z[culvertIndex];
        double flow = 0.;

        if(waterLevel >= 1.5 * culvertHeight)
        {
            //Pressure flow (Hazen-Williams equation) (roughness = 70. - rough concrete)
            double equivalentDiameter = std::sqrt(4. * culvertWidth * culvertHeight / PI);
            flow = (70. * std::pow(culvertSlope, 0.54)) * std::pow(equivalentDiameter, 2.63) / 3.591;
        }
        else if(waterLevel >= culvertHeight)
        {
            //Mixed flow: open channel and pressure
            double wettedPerimeter = culvertWidth + 2. * culvertHeight;
            double hydraulicRadius = nodeGrid.boundaryData.boundarySize[culvertIndex] / wettedPerimeter;

            //Maximum Mannig flow [m3 s-1]
            double ManningFlow = (nodeGrid.boundaryData.boundarySize[culvertIndex] / culvertRoughness) * std::sqrt(culvertSlope) * std::pow(hydraulicRadius, 2./3.);
            //Pressure flow (Hazen-Williams equation) (roughness = 70. - rough concrete)
            double equivalentDiameter = std::sqrt(4. * culvertWidth * culvertHeight / PI);
            double pressureFlow = (70. * std::pow(culvertSlope, 0.54)) * std::pow(equivalentDiameter, 2.63) / 3.591;

            double weight = (waterLevel - culvertHeight) / (0.5 * culvertHeight);
            flow = weight * pressureFlow + (1. - weight) * ManningFlow;
        }
        else if(waterLevel > nodeGrid.waterData.pond[culvertIndex])
        {
            //Open channel flow
            double boundaryArea = culvertWidth * waterLevel;
            double wettedPerimeter = culvertWidth + 2. * waterLevel;
            double hydraulicRadius = boundaryArea / wettedPerimeter;

            flow = (boundaryArea / culvertRoughness) * std::sqrt(culvertSlope) * std::pow(hydraulicRadius, 2./3.);
        }

        nodeGrid.boundaryData.waterFlowRate[culvertIndex] = - flow;
        nodeGrid.waterData.waterFlow[culvertIndex] -= flow;

        return;
    }

    __cudaSpec double getMatrixElement(uint64_t rowIndex, uint64_t columnIndex)
    {
        #ifdef CUDA_ENABLED
            switch(solver->getSolverType())
            {
                case solverType::CPU:
                    return solver->getMatrixElementValue<CPUSolver>(rowIndex, columnIndex);
                    break;
                case solverType::GPU:
                    return solver->getMatrixElementValue<GPUSolver>(rowIndex, columnIndex);
                    break;
                default:
                    return 0.;
            }
        #else
            return solver->getMatrixElementValue<CPUSolver>(rowIndex, columnIndex);
        #endif

    }

}//namespace
