#ifdef CUDA_ENABLED
    #include "gpusolver.h"
#endif
#include "cpusolver.h"

#include <cassert>

#include "water.h"
#include "soilPhysics.h"
#include "heat.h"
#include "otherFunctions.h"

#define EPSILON_CUSTOM 0.00001
#define DBL_EPSILON_CUSTOM 2.2204460492503131e-016

using namespace soilFluxes3D::v2;
using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Math;
using namespace soilFluxes3D::v2::Heat;

namespace soilFluxes3D::v2
{
    extern __cudaMngd Solver* solver;
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;
    extern __cudaMngd simulationFlags_t simulationFlags;
}

namespace soilFluxes3D::v2::Water
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

        balanceDataCurrentTimeStep.waterSinkSource = 0.;
        balanceDataPreviousTimeStep.waterSinkSource = 0.;
        balanceDataCurrentPeriod.waterSinkSource = 0.;
        balanceDataWholePeriod.waterSinkSource = 0.;

        balanceDataCurrentTimeStep.waterMBR = 0.;
        balanceDataWholePeriod.waterMBR = 0.;

        balanceDataCurrentTimeStep.waterMBE = 0.;
        balanceDataWholePeriod.waterMBE = 0.;

        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        //Reset link water flows
        for (u8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
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

        __parforop(__ompStatus, +, sum)
        for (SF3Duint_t idx = 0; idx < nodeGrid.numNodes; ++idx)
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
        // [m3]
        balanceDataCurrentTimeStep.waterStorage = computeTotalWaterContent();
        // [m3]
        double deltaStorage = balanceDataCurrentTimeStep.waterStorage - balanceDataPreviousTimeStep.waterStorage;
        // [m3]
        balanceDataCurrentTimeStep.waterSinkSource = computeWaterSinkSourceFlowsSum(deltaT);
        //[ m3]
        balanceDataCurrentTimeStep.waterMBE = deltaStorage - balanceDataCurrentTimeStep.waterSinkSource;

        // minimum reference water storage [m3] as % of current storage
        double timePercentage = 0.005 * SF3Dmax(deltaT, 30.) / HOUR_SECONDS;
        double minRefWaterStorage = balanceDataCurrentTimeStep.waterStorage * timePercentage;
        // [m3] minimum 1 liter
        minRefWaterStorage = SF3Dmax(minRefWaterStorage, 0.001);

        // Reference water [m3] for computation of mass balance error ratio
        // when the water sink/source is too low, use the reference water storage
        double referenceWater = SF3Dmax(std::fabs(balanceDataCurrentTimeStep.waterSinkSource), minRefWaterStorage);

        balanceDataCurrentTimeStep.waterMBR = balanceDataCurrentTimeStep.waterMBE / referenceWater;
    }

    /*!
     * \brief computes sum of water sink/source flows
     * \param deltaT    [s]
     * \return sum of water sink/source [m3]
     */
    double computeWaterSinkSourceFlowsSum(double deltaT)
    {
        double sum = 0.;

        __parforop(__ompStatus, +, sum)
        for (SF3Duint_t idx = 0; idx < nodeGrid.numNodes; ++idx)
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

        double referenceWater = SF3Dmax(0.001, balanceDataWholePeriod.waterSinkSource);
        balanceDataWholePeriod.waterMBR = balanceDataWholePeriod.waterMBE / referenceWater;

        balanceDataCurrentPeriod.waterStorage = balanceDataCurrentTimeStep.waterStorage;
    }

    /*!
     * \brief evalutate the current water balance
     * \param approxNr number of iteration performed
     * \param bestMBRerror best mass balance ratio error achieved in the previous iterations
     * \param parameters solver parameters
     * \return evaluations of water balance
     */
    balanceResult_t evaluateWaterBalance(u8_t approxNr, double& bestMBRerror, double deltaT, SolverParameters& parameters)
    {
        computeCurrentMassBalance(deltaT);

        double currMBRerror = std::fabs(balanceDataCurrentTimeStep.waterMBR);

        //Optimal error
        if(currMBRerror < parameters.MBRThreshold)
        {
            acceptStep(deltaT);

            //Check Stability (Courant)
            double currCWL = nodeGrid.waterData.CourantWaterLevel;

            if((currCWL < parameters.CourantWaterThreshold) && (approxNr <= 3))
                parameters.deltaTcurr = std::min(parameters.deltaTmax, parameters.deltaTcurr * 2);

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
        __parfor(__ompStatus)
        for (SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            //Update link flows
            for(u8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
                updateLinkFlux(nodeIndex, linkIndex, deltaT);

            //Update boundary flow
            if (nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::NoBoundary)
                nodeGrid.boundaryData.waterFlowSum[nodeIndex] += nodeGrid.boundaryData.waterFlowRate[nodeIndex] * deltaT;
        }
    }

    void saveBestStep() //TO DO: remove
    {
        std::memcpy(nodeGrid.waterData.bestPressureHead, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double));
    }

    void restoreBestStep(double deltaT)
    {
        std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.bestPressureHead, nodeGrid.numNodes * sizeof(double));

        __parfor(__ompStatus)
        for (SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
            if(!nodeGrid.surfaceFlag[nodeIndex])
            {
                nodeGrid.waterData.saturationDegree[nodeIndex] = computeNodeSe(nodeIndex);
                nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);
            }

        updateBoundaryWaterData(deltaT);
        computeCurrentMassBalance(deltaT);
    }

    __cudaSpec void updateLinkFlux(SF3Duint_t nodeIndex, u8_t linkIndex, double deltaT)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return;

        SF3Duint_t linkedNodeIndex = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];
        double matrixValue = getMatrixElement(nodeIndex, linkedNodeIndex);
        nodeGrid.linkData[linkIndex].waterFlowSum[nodeIndex] += matrixValue * (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.waterData.pressureHead[linkedNodeIndex]) * deltaT;
    }

    void restorePressureHead() //TO DO: remove
    {
        std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHead, nodeGrid.numNodes * sizeof(double));
    }

    void computeCapacity(VectorCPU& vectorC)
    {
        __parfor(__ompStatus)
        for (SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            nodeGrid.waterData.invariantFluxes[nodeIndex] = 0.;     //move to memset
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
    void computeLinearSystemElement(MatrixCPU& matrixA, VectorCPU& vectorB, const VectorCPU& vectorC, u8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType)
    {
        __parfor(__ompStatus)
        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            u8_t colIdx = 1;
            bool isLinked;

            // flux up
            u8_t linkIndex = 0;
            isLinked = computeLinkFluxes(matrixA.values[rowIdx][colIdx], matrixA.columnIndeces[rowIdx][colIdx], rowIdx, linkIndex, approxNum, deltaT, lateralVerticalRatio, linkType_t::Up, meanType);
            if(isLinked)
                colIdx++;

            // flux lateral
            for(u8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
            {
                linkIndex = 2 + latIdx;
                isLinked = computeLinkFluxes(matrixA.values[rowIdx][colIdx], matrixA.columnIndeces[rowIdx][colIdx], rowIdx, linkIndex, approxNum, deltaT, lateralVerticalRatio, linkType_t::Lateral, meanType);
                if(isLinked)
                    colIdx++;
            }

            // flux down
            linkIndex = 1;
            isLinked = computeLinkFluxes(matrixA.values[rowIdx][colIdx], matrixA.columnIndeces[rowIdx][colIdx], rowIdx, linkIndex, approxNum, deltaT, lateralVerticalRatio, linkType_t::Down, meanType);
            if(isLinked)
                colIdx++;

            matrixA.numColsInRow[rowIdx] = colIdx;

            //TO DO: need to fill the not used columns of the row?

            //Compute diagonal element
            double sum = 0.;
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
            {
                sum += matrixA.values[rowIdx][colIdx];
                matrixA.values[rowIdx][colIdx] *= -1.;
            }
            matrixA.columnIndeces[rowIdx][0] = rowIdx;
            matrixA.values[rowIdx][0] = (vectorC.values[rowIdx] / deltaT) + sum;

            //Compute b element
            vectorB.values[rowIdx] = ((vectorC.values[rowIdx] / deltaT) * nodeGrid.waterData.oldPressureHead[rowIdx]) + nodeGrid.waterData.waterFlow[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx];

            //Preconditioning
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                matrixA.values[rowIdx][colIdx] /= matrixA.values[rowIdx][0];

            vectorB.values[rowIdx] /= matrixA.values[rowIdx][0];
        }
    }

    __cudaSpec bool computeLinkFluxes(double& matrixElement, SF3Duint_t& matrixIndex, SF3Duint_t nodeIndex, u8_t linkIndex, u8_t approxNum, double deltaT, double lateralVerticalRatio, linkType_t linkType, meanType_t meanType)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return false;

        SF3Duint_t linkedNodeIndex = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];
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

        double thermalVaporFlux = computeThermalVaporFlux(nodeIndex, linkIndex, processType::Water) / WATER_DENSITY;
        nodeGrid.waterData.invariantFluxes[nodeIndex] += thermalVaporFlux;

        return true;
    }


    __cudaSpec double runoff(SF3Duint_t rowIdx, SF3Duint_t colIdx, u8_t approxNum, double deltaT, double flowArea)
    {
        double flux_i = (nodeGrid.waterData.waterFlow[rowIdx] * deltaT) / nodeGrid.size[rowIdx];
        double flux_j = (nodeGrid.waterData.waterFlow[colIdx] * deltaT) / nodeGrid.size[colIdx];

        double H_i = (approxNum != 0) ? nodeGrid.waterData.pressureHead[rowIdx] : nodeGrid.waterData.oldPressureHead[rowIdx] + 0.5 * flux_i;
        double H_j = (approxNum != 0) ? nodeGrid.waterData.pressureHead[colIdx] : nodeGrid.waterData.oldPressureHead[colIdx] + 0.5 * flux_j;

        double dH = std::fabs(H_i - H_j);

        if(dH < DBL_EPSILON_CUSTOM)
            return 0.;

        double z_i = nodeGrid.z[rowIdx] + nodeGrid.waterData.pond[rowIdx];
        double z_j = nodeGrid.z[colIdx] + nodeGrid.waterData.pond[colIdx];

        double H_max = SF3Dmax(H_i, H_j);
        double z_max = SF3Dmax(z_i, z_j);
        double H_s = H_max - z_max;

        if(H_s < EPSILON_CUSTOM)
            return 0.;

        // Warning: cause underestimation of flow in lowland water bodies
        // use only in land depressions (disabled: produces mass balance error)
        //if((H_i > H_j && z_i < z_j) || ((H_i < H_j && z_i > z_j)))
        H_s = SF3Dmin(H_s, dH);

        double cellDistance = nodeDistance2D(rowIdx, colIdx);
        double slope = dH / cellDistance;
        double roughness = 0.5 * (nodeGrid.soilSurfacePointers[rowIdx].surfacePtr->roughness + nodeGrid.soilSurfacePointers[colIdx].surfacePtr->roughness);

        double v = std::pow(H_s, 2./3.) * std::sqrt(slope) / roughness;

        nodeGrid.waterData.partialCourantWaterLevels[rowIdx] = SF3Dmax(nodeGrid.waterData.partialCourantWaterLevels[rowIdx], v * deltaT / cellDistance);

        return v * flowArea * H_s / dH;
    }


    __cudaSpec double infiltration(SF3Duint_t surfNodeIdx, SF3Duint_t soilNodeIdx, double deltaT, double flowArea, meanType_t meanType)
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

        double surfH = 0.5 * (nodeGrid.waterData.pressureHead[surfNodeIdx] + nodeGrid.waterData.oldPressureHead[surfNodeIdx]);
        double soilH = 0.5 * (nodeGrid.waterData.pressureHead[soilNodeIdx] + nodeGrid.waterData.oldPressureHead[soilNodeIdx]);

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

    __cudaSpec double redistribution(SF3Duint_t rowIdx, SF3Duint_t colIdx, double lateralVerticalRatio, double flowArea, linkType_t linkType, meanType_t meanType)
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
        hostAlloc(tempX, vectorX.numElements);
        std::memcpy(tempX, vectorB.values, vectorB.numElements * sizeof(double));

        double* normVector = nullptr;
        hostAlloc(normVector, vectorX.numElements);

        __parfor(__ompStatus)
        for(SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                tempX[rowIdx] -= matrixA.values[rowIdx][colIdx] * vectorX.values[matrixA.columnIndeces[rowIdx][colIdx]];

            if(nodeGrid.surfaceFlag[rowIdx] && tempX[rowIdx] < nodeGrid.z[rowIdx])
                tempX[rowIdx] = nodeGrid.z[rowIdx];

            double currentNorm = std::fabs(tempX[rowIdx] - vectorX.values[rowIdx]);

            double psi = std::fabs(tempX[rowIdx] - nodeGrid.z[rowIdx]);
            if(psi > 1.)
                currentNorm /= psi;
            
            normVector[rowIdx] = currentNorm;
        }

        __parforop(__ompStatus, max, infinityNorm)
        for(SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
            infinityNorm = SF3Dmax(infinityNorm, normVector[rowIdx]);

        hostFree(normVector);

        std::memcpy(vectorX.values, tempX, vectorX.numElements * sizeof(double));
        hostFree(tempX);
        return infinityNorm;
    }

    double GaussSeidelWaterCPU(VectorCPU& vectorX, const MatrixCPU &matrixA, const VectorCPU& vectorB)
    {
        double currentNorm = -1, infinityNorm = -1;

        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            double newCurrValue = vectorB.values[rowIdx];
            for (u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                newCurrValue -= matrixA.values[rowIdx][colIdx] * vectorX.values[matrixA.columnIndeces[rowIdx][colIdx]];

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
        __parfor(__ompStatus)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
        {
            //initialize: water sink.source
            nodeGrid.waterData.waterFlow[nodeIdx] = nodeGrid.waterData.waterSinkSource[nodeIdx];

            if(nodeGrid.boundaryData.boundaryType[nodeIdx] == boundaryType_t::NoBoundary)
                continue;

            nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0;   //TO DO: evaluate move to a memset
            switch(nodeGrid.boundaryData.boundaryType[nodeIdx])
            {
                case boundaryType_t::Runoff:
                    double avgH, hs, maxFlow, v, valFlow;
                    avgH = 0.5 * (nodeGrid.waterData.pressureHead[nodeIdx] + nodeGrid.waterData.oldPressureHead[nodeIdx]);

                    hs = SF3Dmax(0., avgH - (nodeGrid.z[nodeIdx] + nodeGrid.waterData.pond[nodeIdx]));
                    if(hs < EPSILON_RUNOFF)
                        break;

                    // Maximum flow available during the time step [m3 s-1]
                    maxFlow = (hs * nodeGrid.size[nodeIdx]) / deltaT;

                    //Manning equation
                    assert(nodeGrid.surfaceFlag[nodeIdx]);
                    v = std::pow(hs, 2./3.) * std::sqrt(nodeGrid.boundaryData.boundarySlope[nodeIdx]) / nodeGrid.soilSurfacePointers[nodeIdx].surfacePtr->roughness;

                    valFlow = hs * v * nodeGrid.boundaryData.boundarySize[nodeIdx];
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -SF3Dmin(valFlow, maxFlow);
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
                    if(!simulationFlags.computeHeat || !simulationFlags.computeHeatVapor)
                        break;

                    bool isUpLinked;
                    SF3Duint_t upIndex;
                    double surfaceWaterFraction;
                    isUpLinked = (nodeGrid.linkData[0].linkType[nodeIdx] != linkType_t::NoLink);
                    upIndex = isUpLinked ? nodeGrid.linkData[0].linkIndex[nodeIdx] : noDataU;
                    surfaceWaterFraction = isUpLinked ? getNodeSurfaceWaterFraction(upIndex) : 0.;

                    double soilEvaporation;
                    soilEvaporation = computeNodeAtmosphericLatentVaporFlux(nodeIdx) / WATER_DENSITY * nodeGrid.linkData[0].interfaceArea[nodeIdx];

                    if(surfaceWaterFraction > 0.) //check isUpLinked superfluo.
                    {
                        double surfEvaporation = computeNodeAtmosphericLatentSurfaceWaterFlux(upIndex) / WATER_DENSITY * nodeGrid.linkData[0].interfaceArea[nodeIdx];

                        soilEvaporation *= (1. - surfaceWaterFraction);
                        surfEvaporation *= surfaceWaterFraction;

                        double waterVolume = (nodeGrid.waterData.pressureHead[upIndex] - nodeGrid.z[upIndex]) * nodeGrid.size[upIndex];
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

                case boundaryType_t::Culvert:
                    if(nodeGrid.culvertPtr[nodeIdx] == nullptr)
                        assert(false); //? -> throwing error?

                    double culvertHeight, culvertWidth, culvertRoughness, culvertSlope;
                    culvertHeight = nodeGrid.culvertPtr[nodeIdx]->height;
                    culvertWidth = nodeGrid.culvertPtr[nodeIdx]->width;
                    culvertRoughness = nodeGrid.culvertPtr[nodeIdx]->roughness;
                    culvertSlope = nodeGrid.boundaryData.boundarySlope[nodeIdx];

                    double waterLevel, flow;
                    waterLevel = 0.5 * (nodeGrid.waterData.pressureHead[nodeIdx] - nodeGrid.waterData.oldPressureHead[nodeIdx]) - nodeGrid.z[nodeIdx];
                    flow = 0.;

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
                        double hydraulicRadius = nodeGrid.boundaryData.boundarySize[nodeIdx] / wettedPerimeter;

                        //Maximum Mannig flow [m3 s-1]
                        double ManningFlow = (nodeGrid.boundaryData.boundarySize[nodeIdx] / culvertRoughness) * std::sqrt(culvertSlope) * std::pow(hydraulicRadius, 2./3.);
                        //Pressure flow (Hazen-Williams equation) (roughness = 70. - rough concrete)
                        double equivalentDiameter = std::sqrt(4. * culvertWidth * culvertHeight / PI);
                        double pressureFlow = (70. * std::pow(culvertSlope, 0.54)) * std::pow(equivalentDiameter, 2.63) / 3.591;

                        double weight = (waterLevel - culvertHeight) / (0.5 * culvertHeight);
                        flow = weight * pressureFlow + (1. - weight) * ManningFlow;
                    }
                    else if(waterLevel > nodeGrid.waterData.pond[nodeIdx])
                    {
                        //Open channel flow
                        double boundaryArea = culvertWidth * waterLevel;
                        double wettedPerimeter = culvertWidth + 2. * waterLevel;
                        double hydraulicRadius = boundaryArea / wettedPerimeter;

                        flow = (boundaryArea / culvertRoughness) * std::sqrt(culvertSlope) * std::pow(hydraulicRadius, 2./3.);
                    }

                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -flow;
                    break;
                default:
                    nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
                    assert(false);
                    break;
            }

            if(std::fabs(nodeGrid.boundaryData.waterFlowRate[nodeIdx]) < DBL_EPSILON)
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
            else
                nodeGrid.waterData.waterFlow[nodeIdx] += nodeGrid.boundaryData.waterFlowRate[nodeIdx];
        }

        return;
    }

    __cudaSpec double getMatrixElement(SF3Duint_t rowIndex, SF3Duint_t columnIndex)
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
