#include "gpusolver.h"
#include "soilFluxes3D.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"
#include "otherFunctions.h"

#include <cassert>
#include <type_traits>
#include <omp.h>

using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Math;
using namespace soilFluxes3D::v2::Heat;
using namespace soilFluxes3D::v2::Water;

namespace soilFluxes3D::v2
{
    extern __cudaMngd Solver* solver;
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;
    extern __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;
    extern std::vector<soilData_t> soilList;
    extern std::vector<surfaceData_t> surfaceList;
    extern std::vector<culvertData_t> culvertList;

    SF3Derror_t GPUSolver::initialize()
    {
        if(_status != solverStatus::Created)
            return SF3Derror_t::SolverError;

        if(_parameters.deltaTcurr == noDataD)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;                           //TO DO: (nodeGrid.numNodes > ...);
        if(_parameters.enableOMP)
            omp_set_num_threads(static_cast<int>(_parameters.numThreads));

        numThreadsPerBlock = SF3Dmin(nodeGrid.numNodes, 64);    //Default value must be a small multiple of warp-size(32)
        numBlocks = static_cast<SF3Duint_t>(std::ceil((double) nodeGrid.numNodes / numThreadsPerBlock));

        cuspCheck(cusparseCreate(&libHandle));

        //initialize matrix data
        iterationMatrix.numRows = static_cast<i64_t>(nodeGrid.numNodes);
        iterationMatrix.numCols = static_cast<i64_t>(nodeGrid.numNodes);
        iterationMatrix.sliceSize = static_cast<i64_t>(numThreadsPerBlock);
        SF3Duint_t numSlice = numBlocks;
        SF3Duint_t tnv = numSlice * iterationMatrix.sliceSize * maxTotalLink;
        iterationMatrix.totValuesSize = static_cast<i64_t>(tnv);

        deviceSolverAlloc(iterationMatrix.d_numColsInRow, iterationMatrix.numRows);

        deviceSolverAlloc(iterationMatrix.d_offsets, (numSlice + 1));
        deviceSolverAlloc(iterationMatrix.d_columnIndeces, tnv);
        deviceSolverAlloc(iterationMatrix.d_values, tnv);

        deviceSolverAlloc(iterationMatrix.d_diagonalValues, iterationMatrix.numRows);

        //initialize vectors data
        constantTerm.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(constantTerm.d_values, constantTerm.numElements);

        unknownTerm.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(unknownTerm.d_values, unknownTerm.numElements);

        tempSolution.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(tempSolution.d_values, tempSolution.numElements);

        //initialize cuSPARSE descriptor
        createCUsparseDescriptors();

        //initialize raw capacity vector
        deviceSolverAlloc(d_Cvalues, nodeGrid.numNodes);

        _status = solverStatus::initialized;
        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::createCUsparseDescriptors()
    {
        //Calc total nnz element
        iterationMatrix.numNonZeroElement = deviceSum(iterationMatrix.d_numColsInRow, iterationMatrix.numRows);

        //Setup offsets vector
        i64_t* h_offsets = static_cast<i64_t*>(std::calloc((numBlocks + 1), sizeof(i64_t)));
        for (i64_t idx = 0; idx < (numBlocks + 1); ++idx)
            h_offsets[idx] = idx * numThreadsPerBlock * maxTotalLink;

        cudaMemcpy(iterationMatrix.d_offsets, h_offsets, (numBlocks + 1) * sizeof(i64_t), cudaMemcpyHostToDevice);

        //Create matrix descriptor
        cuspCheck(cusparseCreateSlicedEll(&(iterationMatrix.cusparseDescriptor),
                                            iterationMatrix.numRows, iterationMatrix.numCols, iterationMatrix.numNonZeroElement, iterationMatrix.totValuesSize, iterationMatrix.sliceSize,
                                            iterationMatrix.d_offsets, iterationMatrix.d_columnIndeces,  iterationMatrix.d_values,
                                            iterationMatrix.offsetType, iterationMatrix.colIdxType, iterationMatrix.idxBase, iterationMatrix.valueType));

        //Create vector descriptor
        cuspCheck(cusparseCreateDnVec(&(constantTerm.cusparseDescriptor), constantTerm.numElements, constantTerm.d_values, constantTerm.valueType));
        cuspCheck(cusparseCreateDnVec(&(unknownTerm.cusparseDescriptor), unknownTerm.numElements, unknownTerm.d_values, unknownTerm.valueType));
        cuspCheck(cusparseCreateDnVec(&(tempSolution.cusparseDescriptor), tempSolution.numElements, tempSolution.d_values, tempSolution.valueType));

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::run(double maxTimeStep, double &acceptedTimeStep, processType process)
    {
        if(_status != solverStatus::initialized)
            return SF3Derror_t::SolverError;

        if(maxTimeStep == HOUR_SECONDS)
            upCopyData();

        switch (process)
        {
            case processType::Water:
                waterMainLoop(maxTimeStep, acceptedTimeStep);
                break;
            case processType::Heat:
                throw std::runtime_error("Heat not available with GPUSolver");
                break;
            case processType::Solutes:
                throw std::runtime_error("Solutes not available with GPUSolver");
                break;
            default:
                break;
        }

        _status = solverStatus::Terminated;
        if(acceptedTimeStep == maxTimeStep)
            downCopyData();

        _status = solverStatus::initialized;
        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::clean()
    {
        if(_status == solverStatus::Created)
            return SF3Derror_t::SF3Dok;

        if((_status != solverStatus::Terminated) && (_status != solverStatus::initialized))
            return SF3Derror_t::SolverError;

        deviceSolverFree(iterationMatrix.d_numColsInRow);
        deviceSolverFree(iterationMatrix.d_offsets);
        deviceSolverFree(iterationMatrix.d_columnIndeces);
        deviceSolverFree(iterationMatrix.d_values);
        deviceSolverFree(iterationMatrix.d_diagonalValues);

        cuspCheck(cusparseDestroySpMat(iterationMatrix.cusparseDescriptor));

        deviceSolverFree(constantTerm.d_values);
        cuspCheck(cusparseDestroyDnVec(constantTerm.cusparseDescriptor));

        deviceSolverFree(unknownTerm.d_values);
        cuspCheck(cusparseDestroyDnVec(unknownTerm.cusparseDescriptor));

        deviceSolverFree(tempSolution.d_values);
        cuspCheck(cusparseDestroyDnVec(tempSolution.cusparseDescriptor));

        cuspCheck(cusparseDestroy(libHandle));

        deviceSolverFree(d_Cvalues);

        deviceSolverFree(d_surfaceList);
        deviceSolverFree(d_soilList);

        _status = solverStatus::Created;
        return SF3Derror_t::SF3Dok;
    }

    void GPUSolver::waterMainLoop(double maxTimeStep, double &acceptedTimeStep)
    {
        balanceResult_t stepStatus = balanceResult_t::stepRefused;
        while(stepStatus != balanceResult_t::stepAccepted)
        {
            acceptedTimeStep = SF3Dmin(_parameters.deltaTcurr, maxTimeStep);

            //Save instantaneus H values
            cudaMemcpy(nodeGrid.waterData.oldPressureHead, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

            //initialize the solution vector with the current pressure head
            assert(unknownTerm.numElements == nodeGrid.numNodes);
            cudaMemcpy(unknownTerm.d_values, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

            launchKernel(initializeCapacityAndSaturationDegree_k, d_Cvalues);

            //Update aereodynamic and soil conductance
            updateConductance();

            //Update boundary
            launchKernel(updateBoundaryWaterData_k, acceptedTimeStep);

            //Effective computation step
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            if(stepStatus != balanceResult_t::stepAccepted)
            {
                //  restore old pressureHead
                cudaMemcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHead,
                           nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
            }
        }
    }

    balanceResult_t GPUSolver::waterApproximationLoop(double deltaT)
    {
        balanceResult_t balanceResult;

        for(u8_t approxIdx = 0; approxIdx < _parameters.maxApproximationsNumber; ++approxIdx)
        {
            //Compute capacity vector elements
            launchKernel(computeCapacityWater_k, d_Cvalues);

            //Update boundary water
            launchKernel(updateBoundaryWaterData_k, deltaT);

            //Reset Courant data
            nodeGrid.CourantWaterLevel = 0.;
            deviceReset(nodeGrid.waterData.partialCourantWaterLevels, nodeGrid.numNodes);

            //Compute linear system elements
            launchKernel(computeWaterLinearSystemElement_k, iterationMatrix, constantTerm, d_Cvalues, approxIdx, deltaT, _parameters.lateralVerticalRatio, _parameters.meanType);

            //Courant data reduction
            nodeGrid.CourantWaterLevel = deviceMax(nodeGrid.waterData.partialCourantWaterLevels, nodeGrid.numNodes);

            //Check Courant
            if((nodeGrid.CourantWaterLevel > 1.01) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / nodeGrid.CourantWaterLevel);
                if(_parameters.deltaTcurr > 1.)
                    _parameters.deltaTcurr = std::floor(_parameters.deltaTcurr);

                return balanceResult_t::stepHalved;
            }

            //Try solve linear system
            bool isStepValid = solveLinearSystem(approxIdx, processType::Water);

            //Reduce step tipe if system resolution failed
            if((!isStepValid) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / 2.);
                return balanceResult_t::stepHalved;
            }

            //Update potential
            cudaMemcpy(nodeGrid.waterData.pressureHead, unknownTerm.d_values, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
            //Update degree of saturation
            launchKernel(updateSaturationDegree_k);

            //Check water balance
            balanceResult = evaluateWaterBalance_m(approxIdx, _bestMBRerror, deltaT);

            if((balanceResult == balanceResult_t::stepAccepted) || (balanceResult == balanceResult_t::stepHalved))
                return balanceResult;
        }

        return balanceResult;
    }

    void GPUSolver::heatLoop(double timeStepHeat, double timeStepWater)
    {
        resetFluxValues_m(true, false);

        //initialize vector X
        cudaMemcpy(unknownTerm.d_values, nodeGrid.heatData.temperature, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
        //Save current temperatures
        cudaMemcpy(nodeGrid.heatData.oldTemperature, nodeGrid.heatData.temperature, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

        //inizialize vector C
        launchKernel(computeCapacityHeat_k, d_Cvalues, timeStepHeat, timeStepWater);

        //Compute linear system elements
        deviceReset(nodeGrid.waterData.invariantFluxes, nodeGrid.numNodes);

        launchKernel(computeHeatLinearSystemElement_k, iterationMatrix, constantTerm, d_Cvalues, timeStepHeat, timeStepWater);

        solveLinearSystem(_parameters.maxApproximationsNumber - 1, processType::Heat);

        //Retrive new temperatures
        deviceConditionalCopy(nodeGrid.heatData.temperature, unknownTerm.d_values, nodeGrid.numNodes,
                              ConditionWrapper<condition::notSurface>(nodeGrid.surfaceFlag));

        evaluateHeatBalance_m(timeStepHeat, timeStepWater);
        updateHeatBalanceData();
        saveHeatFluxValues_m(timeStepHeat, timeStepWater);

        //Save new temperatures
        deviceConditionalCopy(nodeGrid.heatData.oldTemperature, nodeGrid.heatData.temperature, nodeGrid.numNodes,
                              ConditionWrapper<condition::notSurface>(nodeGrid.surfaceFlag));

        return;
    }

    bool GPUSolver::solveLinearSystem(u8_t approximationNumber, processType computationType) //Implemented only Jacobi
    {
        std::size_t bufSize;
        const double alpha = -1, beta = 1;
        void *externalBuffer = nullptr;

        cusparseSpMV_bufferSize(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, unknownTerm.cusparseDescriptor, &beta, tempSolution.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
        cudaDeviceSynchronize();
        cudaMalloc(&externalBuffer, bufSize);

        bool status = true;
        double bestErrorNorm = static_cast<double>(std::numeric_limits<float>::max());

        double *d_tempVector = nullptr;
        deviceAlloc(d_tempVector, unknownTerm.numElements);

        //TO DO: implement cusparseSpMV_preprocess

        u32_t currMaxIterationNum = calcCurrentMaxIterationNumber(approximationNumber);
        u32_t GPUfactor = 1;           //TO DO: test and optimize
        for (std::size_t iterationNumber = 0; iterationNumber < currMaxIterationNum; ++iterationNumber)
        {
            for(std::size_t internalCounter = 0; internalCounter < GPUfactor; ++internalCounter)
            {
                cudaMemcpy(tempSolution.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();

                cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, unknownTerm.cusparseDescriptor, &beta, tempSolution.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
                cudaDeviceSynchronize();

                //TEMP: single iteration to mimic CPU behaviour
                cudaMemcpy(d_tempVector, unknownTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpy(unknownTerm.d_values, tempSolution.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpy(tempSolution.d_values, d_tempVector, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);

                /* //Optimal path: double iteration to value swap
                 * cudaMemcpy(unknownTerm.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                 * cudaDeviceSynchronize();
                 *
                 * cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, tempSolution.cusparseDescriptor, &beta, unknownTerm.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
                 * cudaDeviceSynchronize();
                 */
            }

            //Calcolo della norma dell'errore
            double *d_normVector = nullptr;
            deviceAlloc(d_normVector, unknownTerm.numElements);
            launchKernel(computeNormalizedError, d_normVector, unknownTerm.d_values, tempSolution.d_values);

            double currErrorNorm = deviceMax(d_normVector, unknownTerm.numElements);
            deviceFree(d_normVector);

            if(currErrorNorm < _parameters.residualTolerance)
                break;

            if(currErrorNorm > (bestErrorNorm * 10))
            {
                status = false;
                break;
            }

            if(currErrorNorm < bestErrorNorm)
                bestErrorNorm = currErrorNorm;
        }

        deviceFree(externalBuffer);
        deviceFree(d_tempVector);

        return status;
    }

    //------------------------------- TEMP FUNCTIONS -------------------------------
    balanceResult_t GPUSolver::evaluateWaterBalance_m(u8_t approxNr, double& bestMBRerror, double deltaT)
    {
        computeCurrentMassBalance_m(deltaT);

        double currMBRerror = std::fabs(balanceDataCurrentTimeStep.waterMBR);

        //Optimal error
        if(currMBRerror < _parameters.MBRThreshold)
        {
            acceptStep_m(deltaT);

            //Check Stability (Courant)
            if((nodeGrid.CourantWaterLevel < _parameters.CourantWaterThreshold) && (approxNr <= 3))
                _parameters.deltaTcurr = SF3Dmin(_parameters.deltaTmax, _parameters.deltaTcurr * 2);

            return balanceResult_t::stepAccepted;
        }

        //Good error or first approximation
        if (approxNr == 0 || currMBRerror < bestMBRerror)
        {
            //saveBestStep();
            cudaMemcpy(nodeGrid.waterData.bestPressureHead, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
            bestMBRerror = currMBRerror;
        }

        //Critical error (unstable system) or last approximation
        if (approxNr == (_parameters.maxApproximationsNumber - 1) || currMBRerror > (bestMBRerror * _parameters.instabilityFactor))
        {
            if(deltaT > _parameters.deltaTmin)
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTcurr / 2, _parameters.deltaTmin);
                return balanceResult_t::stepHalved;
            }

            restoreBestStep_m(deltaT);
            acceptStep_m(deltaT);
            return balanceResult_t::stepAccepted;
        }

        return balanceResult_t::stepRefused;
    }

    void GPUSolver::computeCurrentMassBalance_m(double deltaT)
    {
        double* d_waterContentVector = nullptr;
        deviceAlloc(d_waterContentVector, nodeGrid.numNodes);
        launchKernel(computeWaterContent_k, d_waterContentVector);

        balanceDataCurrentTimeStep.waterStorage = deviceSum(d_waterContentVector, nodeGrid.numNodes);
        double deltaStorage = balanceDataCurrentTimeStep.waterStorage - balanceDataPreviousTimeStep.waterStorage;

        balanceDataCurrentTimeStep.waterSinkSource = deviceSum(nodeGrid.waterData.waterFlow, nodeGrid.numNodes) * deltaT;;
        balanceDataCurrentTimeStep.waterMBE = deltaStorage - balanceDataCurrentTimeStep.waterSinkSource;

        // minimum reference water storage [m3] as % of current storage
        double timePercentage = 0.01 * SF3Dmax(deltaT, 30.) / HOUR_SECONDS;
        double minRefWaterStorage = balanceDataCurrentTimeStep.waterStorage * timePercentage;
        // [m3] minimum 1 liter
        minRefWaterStorage = SF3Dmax(minRefWaterStorage, 0.001);

        // Reference water for computation of mass balance error ratio
        // when the water sink/source is too low, use the reference water storage
        double referenceWater = SF3Dmax(std::fabs(balanceDataCurrentTimeStep.waterSinkSource), minRefWaterStorage);     // [m3]

        balanceDataCurrentTimeStep.waterMBR = balanceDataCurrentTimeStep.waterMBE / referenceWater;
    }

    void GPUSolver::acceptStep_m(double deltaT)
    {
        /*! set current time step balance data as the previous one */
        balanceDataPreviousTimeStep.waterStorage = balanceDataCurrentTimeStep.waterStorage;
        balanceDataPreviousTimeStep.waterSinkSource = balanceDataCurrentTimeStep.waterSinkSource;

        /*! update balance data of current period */
        balanceDataCurrentPeriod.waterSinkSource += balanceDataCurrentTimeStep.waterSinkSource;

        launchKernel(updateWaterFlows_k, deltaT);
    }

    void GPUSolver::restoreBestStep_m(double deltaT)
    {
        cudaMemcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.bestPressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

        launchKernel(updateSaturationDegree_k);
        launchKernel(updateWaterConductivity_k);

        launchKernel(updateBoundaryWaterData_k, deltaT);
        computeCurrentMassBalance_m(deltaT);
    }

    void GPUSolver::evaluateHeatBalance_m(double dtHeat, double dtWater)
    {
        //Heat sink/source
        double* d_heatSinkSourceVector = nullptr;
        deviceAlloc(d_heatSinkSourceVector, nodeGrid.numNodes);
        launchKernel(computeCurrentHeatSinkSource_k, d_heatSinkSourceVector, dtHeat);
        double heatSinkSource = deviceSum(d_heatSinkSourceVector, nodeGrid.numNodes);

        balanceDataCurrentTimeStep.heatSinkSource = heatSinkSource;

        //Heat storage
        double* d_heatStorageVector = nullptr;
        deviceAlloc(d_heatStorageVector, nodeGrid.numNodes);
        launchKernel(computeCurrentHeatStorage_k, d_heatStorageVector, dtWater, dtHeat);
        double heatStorage = deviceSum(d_heatStorageVector, nodeGrid.numNodes);

        balanceDataCurrentTimeStep.heatStorage = heatStorage;

        //Heat MBE
        double deltaHeatStorage = balanceDataCurrentTimeStep.heatStorage - balanceDataPreviousTimeStep.heatStorage;
        balanceDataCurrentTimeStep.heatMBE = deltaHeatStorage - balanceDataCurrentTimeStep.heatSinkSource;

        //Heat MBR
        double referenceHeat = SF3Dmax(1., std::fabs(balanceDataCurrentTimeStep.heatSinkSource));
        balanceDataCurrentTimeStep.heatMBR = balanceDataCurrentTimeStep.heatMBE / referenceHeat;
    }

    SF3Derror_t GPUSolver::resetFluxValues_m(bool flagHeat, bool flagWater)
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
                            deviceFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(index)], nodeGrid.numNodes, noDataD);
                    break;

                case heatFluxSaveMode_t::Total:
                    for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                        deviceFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(fluxTypes_t::HeatTotal)], nodeGrid.numNodes, noDataD);
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
                            deviceFill(nodeGrid.linkData[lIdx].fluxes[toUnderlyingT(index)], nodeGrid.numNodes, noDataD);
                    break;

                default:
                    break;
            }
        }

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::saveHeatFluxValues_m(double dtHeat, double dtWater)
    {
        if(!simulationFlags.computeHeat)
            return SF3Derror_t::SF3Dok;

        if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::None)
            return SF3Derror_t::SF3Dok;

        launchKernel(saveHeatFluxValues_k, dtHeat, dtWater);

        return SF3Derror_t::SF3Dok;
    }

    //------------------------------- MOVE FUNCTIONS -------------------------------
    SF3Derror_t GPUSolver::upCopyData()
    {
        //Streams creations
        cudaStream_t moveStreams[32];
        for(auto& stream : moveStreams)
            cudaStreamCreate(&(stream));

        std::size_t currStreamIdx = 0;

        upMoveVectorPtrs();      //Need to be done before moving nodeGrid.surfaceFlag

        //Topology Data
        moveToDevice(nodeGrid.size, nodeGrid.numNodes);
        moveToDevice(nodeGrid.x, nodeGrid.numNodes);
        moveToDevice(nodeGrid.y, nodeGrid.numNodes);
        moveToDevice(nodeGrid.z, nodeGrid.numNodes);
        moveToDevice(nodeGrid.surfaceFlag, nodeGrid.numNodes);

        //Soil/surface properties pointers
        moveToDevice(nodeGrid.soilSurfacePointers, nodeGrid.numNodes);

        //Boundary data
        moveToDevice(nodeGrid.boundaryData.boundaryType, nodeGrid.numNodes);
        moveToDevice(nodeGrid.boundaryData.boundarySlope, nodeGrid.numNodes);
        moveToDevice(nodeGrid.boundaryData.boundarySize, nodeGrid.numNodes);
        if(simulationFlags.computeWater)
        {
            moveToDevice(nodeGrid.boundaryData.waterFlowRate, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.waterFlowSum, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.prescribedWaterPotential, nodeGrid.numNodes);
        }

        //Link data
        moveToDevice(nodeGrid.numLateralLink, nodeGrid.numNodes);
        for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
        {
            moveToDevice(nodeGrid.linkData[lIdx].linkType, nodeGrid.numNodes);
            moveToDevice(nodeGrid.linkData[lIdx].linkIndex, nodeGrid.numNodes);
            moveToDevice(nodeGrid.linkData[lIdx].interfaceArea, nodeGrid.numNodes);

            //if(simulationFlags.computeWater)
            moveToDevice(nodeGrid.linkData[lIdx].waterFlowSum, nodeGrid.numNodes);

            if(simulationFlags.computeHeat)
            {
                moveToDevice(nodeGrid.linkData[lIdx].waterFlux, nodeGrid.numNodes);
                moveToDevice(nodeGrid.linkData[lIdx].vaporFlux, nodeGrid.numNodes);
                for(u8_t fIdx = 0; fIdx < numTotalFluxTypes; ++fIdx)
                    moveToDevice(nodeGrid.linkData[lIdx].fluxes[fIdx], nodeGrid.numNodes);
            }
        }

        //Water data
        moveToDevice(nodeGrid.waterData.saturationDegree, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.waterConductivity, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.waterFlow, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.pressureHead, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.waterSinkSource, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.pond, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.invariantFluxes, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.oldPressureHead, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.bestPressureHead, nodeGrid.numNodes);
        moveToDevice(nodeGrid.waterData.partialCourantWaterLevels, nodeGrid.numNodes);

        //Heat data
        if(simulationFlags.computeHeat)
        {
            moveToDevice(nodeGrid.boundaryData.heightWind, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.heightTemperature, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.roughnessHeight, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.aerodynamicConductance, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.soilConductance, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.temperature, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.relativeHumidity, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.windSpeed, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.netIrradiance, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.sensibleFlux, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.latentFlux, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.radiativeFlux, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.advectiveHeatFlux, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.fixedTemperatureValue, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.fixedTemperatureDepth, nodeGrid.numNodes);
        }

        cudaDeviceSynchronize();
        for(auto& stream : moveStreams)
            cudaStreamDestroy(stream);

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::upMoveVectorPtrs()
    {
        static_assert(std::is_trivially_copyable<soilData_t>::value, "Soil structures are not trivially copyable");
        static_assert(std::is_trivially_copyable<surfaceData_t>::value, "Surface structures are not trivially copyable");
        static_assert(std::is_trivially_copyable<culvertData_t>::value, "Culvert structures are not trivially copyable");

        //Move surface data
        auto surfSize = surfaceList.size() * sizeof(surfaceData_t);
        deviceSolverAlloc(d_surfaceList, surfaceList.size());
        cudaMemcpy(d_surfaceList, surfaceList.data(), surfSize, cudaMemcpyHostToDevice);

        //Move soil data
        auto soilSize = soilList.size() * sizeof(soilData_t);
        deviceSolverAlloc(d_soilList, soilList.size());
        cudaMemcpy(d_soilList, soilList.data(), soilSize, cudaMemcpyHostToDevice);

        //Move Culvert data
        auto culvSize = culvertList.size() * sizeof(culvertData_t);
        isCulvertActive = (culvSize > 0);
        if(isCulvertActive)
        {
            deviceSolverAlloc(d_culverList, culvertList.size());
            cudaMemcpy(d_culverList, culvertList.data(), culvSize, cudaMemcpyHostToDevice);
        }

        //Update pointers in nodeGrid
        __parfor(_parameters.enableOMP)
        for(SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            auto& SSPointer = nodeGrid.soilSurfacePointers[nodeIndex];
            if(nodeGrid.surfaceFlag[nodeIndex])
                SSPointer.surfacePtr = d_surfaceList + (SSPointer.surfacePtr - surfaceList.data());
            else
                SSPointer.soilPtr = d_soilList + (SSPointer.soilPtr - soilList.data());

            auto& CPointer = nodeGrid.culvertPtr[nodeIndex];
            if(isCulvertActive && CPointer)
                CPointer = d_culverList + (CPointer - culvertList.data());
        }

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::downCopyData()
    {
        //Streams creations
        cudaStream_t moveStreams[32];
        for(auto& stream : moveStreams)
            cudaStreamCreate(&(stream));

        std::size_t currStreamIdx = 0;

        moveToHost(nodeGrid.size, nodeGrid.numNodes);
        moveToHost(nodeGrid.x, nodeGrid.numNodes);
        moveToHost(nodeGrid.y, nodeGrid.numNodes);
        moveToHost(nodeGrid.z, nodeGrid.numNodes);
        moveToHost(nodeGrid.surfaceFlag, nodeGrid.numNodes);

        //Soil/surface properties pointers
        moveToHost(nodeGrid.soilSurfacePointers, nodeGrid.numNodes);
        downMoveVectorPtrs();      //Need to be done after moving nodeGrid.surfaceFlag

        //Boundary data
        moveToHost(nodeGrid.boundaryData.boundaryType, nodeGrid.numNodes);
        moveToHost(nodeGrid.boundaryData.boundarySlope, nodeGrid.numNodes);
        moveToHost(nodeGrid.boundaryData.boundarySize, nodeGrid.numNodes);
        if(simulationFlags.computeWater)
        {
            moveToHost(nodeGrid.boundaryData.waterFlowRate, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.waterFlowSum, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.prescribedWaterPotential, nodeGrid.numNodes);
        }

        //Link data
        moveToHost(nodeGrid.numLateralLink, nodeGrid.numNodes);
        for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
        {
            moveToHost(nodeGrid.linkData[lIdx].linkType, nodeGrid.numNodes);
            moveToHost(nodeGrid.linkData[lIdx].linkIndex, nodeGrid.numNodes);
            moveToHost(nodeGrid.linkData[lIdx].interfaceArea, nodeGrid.numNodes);

            //if(simulationFlags.computeWater)
            moveToHost(nodeGrid.linkData[lIdx].waterFlowSum, nodeGrid.numNodes);

            if(simulationFlags.computeHeat)
            {
                moveToHost(nodeGrid.linkData[lIdx].waterFlux, nodeGrid.numNodes);
                moveToHost(nodeGrid.linkData[lIdx].vaporFlux, nodeGrid.numNodes);
                for(u8_t fIdx = 0; fIdx < numTotalFluxTypes; ++fIdx)
                    moveToHost(nodeGrid.linkData[lIdx].fluxes[fIdx], nodeGrid.numNodes);
            }
        }

        //Water data
        moveToHost(nodeGrid.waterData.saturationDegree, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.waterConductivity, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.waterFlow, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.pressureHead, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.waterSinkSource, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.pond, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.invariantFluxes, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.oldPressureHead, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.bestPressureHead, nodeGrid.numNodes);
        moveToHost(nodeGrid.waterData.partialCourantWaterLevels, nodeGrid.numNodes);

        //Heat data
        if(simulationFlags.computeHeat)
        {
            moveToHost(nodeGrid.boundaryData.heightWind, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.heightTemperature, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.roughnessHeight, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.aerodynamicConductance, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.soilConductance, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.temperature, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.relativeHumidity, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.windSpeed, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.netIrradiance, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.sensibleFlux, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.latentFlux, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.radiativeFlux, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.advectiveHeatFlux, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.fixedTemperatureValue, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.fixedTemperatureDepth, nodeGrid.numNodes);
        }

        cudaDeviceSynchronize();
        for(auto& stream : moveStreams)
            cudaStreamDestroy(stream);

        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t GPUSolver::downMoveVectorPtrs()
    {
        __parfor(_parameters.enableOMP)
        for(SF3Duint_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            auto& SSPointer = nodeGrid.soilSurfacePointers[nodeIndex];
            if(nodeGrid.surfaceFlag[nodeIndex])
                SSPointer.surfacePtr = surfaceList.data() + (SSPointer.surfacePtr - d_surfaceList);
            else
                SSPointer.soilPtr = soilList.data() + (SSPointer.soilPtr - d_soilList);

            auto& CPointer = nodeGrid.culvertPtr[nodeIndex];
            if(isCulvertActive && CPointer)
                CPointer = d_culverList + (CPointer - culvertList.data());
        }

        deviceSolverFree(d_surfaceList);
        deviceSolverFree(d_soilList);
        if(isCulvertActive)
            deviceSolverFree(d_culverList);

        return SF3Derror_t::SF3Dok;
    }


    //-------------------------------- CUDA KERNELS --------------------------------
    extern __cudaMngd Solver* solver;

    __global__ void initializeCapacityAndSaturationDegree_k(double* Cvalues)    //TO DO: refactor with host code in single __host__ __device__ function
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            Cvalues[nodeIdx] = nodeGrid.size[nodeIdx];
        else
            nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
    }

    __global__ void computeCapacityWater_k(double* Cvalues)                          //TO DO: merge with host code in single __host__ __device__ function
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        nodeGrid.waterData.invariantFluxes[nodeIdx] = 0.;
        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        //Compute hydraulic conductivity
        nodeGrid.waterData.waterConductivity[nodeIdx] = computeNodeK(nodeIdx);

        double dThetadH = computeNodedThetadH(nodeIdx);
        Cvalues[nodeIdx] = nodeGrid.size[nodeIdx] * dThetadH;

        // if(simulationFlags.computeHeat && simulationFlags.computeHeatVapor)
        //     vectorC[nodeIdx] += nodeGrid.size[nodeIdx] * computeNodedThetaVdH(nodeIdx, getNodeMeanTemperature(nodeIdx), dThetadH);
    }

    __global__ void updateBoundaryWaterData_k(double deltaT)                    //TO DO: merge with host code in single __host__ __device__ function
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        nodeGrid.waterData.waterFlow[nodeIdx] = nodeGrid.waterData.waterSinkSource[nodeIdx];

        if(nodeGrid.boundaryData.boundaryType[nodeIdx] == boundaryType_t::NoBoundary)
            return;

        nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0;

        switch(nodeGrid.boundaryData.boundaryType[nodeIdx])
        {
            case boundaryType_t::Runoff:
                double avgH, hs, maxFlow, v, flow;
                avgH = 0.5 * (nodeGrid.waterData.pressureHead[nodeIdx] + nodeGrid.waterData.oldPressureHead[nodeIdx]);

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
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.linkData[0].interfaceArea[nodeIdx];
                break;

            case boundaryType_t::FreeLateraleDrainage:
                //Darcy gradient = slope
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.boundaryData.boundarySize[nodeIdx]
                                                               * nodeGrid.boundaryData.boundarySlope[nodeIdx] * (*solver).getLVRatio();
                break;

            case boundaryType_t::PrescribedTotalWaterPotential:
                double L, boundaryPsi, boundaryZ, boundaryK, meanK, dH;
                L = 1.;     // [m]
                boundaryZ = nodeGrid.z[nodeIdx] - L;

                boundaryPsi = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - boundaryZ;

                boundaryK = (boundaryPsi >= 0)  ? nodeGrid.soilSurfacePointers[nodeIdx].soilPtr->K_sat
                                               : computeMualemSoilConductivity(*(nodeGrid.soilSurfacePointers[nodeIdx].soilPtr), computeNodeSe_fromPsi(nodeIdx, fabs(boundaryPsi)));

                meanK = computeMean(boundaryK, nodeGrid.waterData.waterConductivity[nodeIdx], (*solver).getMeanType());
                dH = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - nodeGrid.waterData.pressureHead[nodeIdx];

                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = meanK * nodeGrid.boundaryData.boundarySize[nodeIdx] * (dH / L);
                break;

            case boundaryType_t::HeatSurface:
                if(!simulationFlags.computeHeat && !simulationFlags.computeHeatVapor)
                    break;
                //TO DO: complete

                break;

            default:
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
                break;
        }

        if(abs(nodeGrid.boundaryData.waterFlowRate[nodeIdx]) < DBL_EPSILON)
            nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0.;
        else
            nodeGrid.waterData.waterFlow[nodeIdx] += nodeGrid.boundaryData.waterFlowRate[nodeIdx];
    }

    __global__ void computeCapacityHeat_k(double *Cvalues, double timeStepHeat, double timeStepWater)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        double nodeH = getNodeH_fromTimeSteps(nodeIdx, timeStepHeat, timeStepWater);
        double avgH = computeMean(nodeGrid.waterData.oldPressureHead[nodeIdx], nodeH, meanType_t::Arithmetic) - nodeGrid.z[nodeIdx];
        Cvalues[nodeIdx] = computeNodeHeatCapacity(nodeIdx, avgH, nodeGrid.heatData.temperature[nodeIdx]) * nodeGrid.size[nodeIdx];

        return;
    }

    __global__ void computeNormalizedError(double *vectorNorm, double *vectorX, const double *previousX)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx] && vectorX[nodeIdx] < nodeGrid.z[nodeIdx])
            vectorX[nodeIdx] = nodeGrid.z[nodeIdx];

        double currentNorm = std::fabs(vectorX[nodeIdx] - previousX[nodeIdx]);
        double psi = std::fabs(vectorX[nodeIdx] - nodeGrid.z[nodeIdx]);
        if(psi > 1.)
            currentNorm /= psi;

        vectorNorm[nodeIdx] = currentNorm;
    }

    __global__ void computeWaterLinearSystemElement_k(MatrixGPU matrixA, VectorGPU vectorB, const double* Cvalues, u8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType)
    {
        SF3Duint_t sliceIdx = blockIdx.x;
        SF3Duint_t rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(rowIdx >= nodeGrid.numNodes)
            return;

        SF3Duint_t baseRowIndex = (sliceIdx * maxTotalLink * blockDim.x) + threadIdx.x;

        u32_t numLinks = 0;
        SF3Duint_t currentElementIndex = baseRowIndex, unsignedTmpColIdx = 0;
        bool isLinked = false;

        // flux up
        isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 0, approxNum, deltaT, lateralVerticalRatio, linkType_t::Up, meanType);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        // flux down
        isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 1, approxNum, deltaT, lateralVerticalRatio, linkType_t::Down, meanType);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        // flux lateral
        for(u32_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
        {
            isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 2 + latIdx, approxNum, deltaT, lateralVerticalRatio, linkType_t::Lateral, meanType);
            if(isLinked)
            {
                matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
                numLinks++;
                currentElementIndex += blockDim.x;
            }
        }

        //Save num cols in current row
        matrixA.d_numColsInRow[rowIdx] = numLinks;

        //Fill not used columns with 0 (values) and -1 (indeces)
        for(u32_t colIdx = numLinks; colIdx < maxTotalLink; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            matrixA.d_values[currentElementIndex] = 0;
            matrixA.d_columnIndeces[currentElementIndex] = -1;
        }

        //Compute diagonal element
        double sum = 0.;
        for(u32_t colIdx = 0; colIdx < numLinks; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            sum += matrixA.d_values[currentElementIndex];
            matrixA.d_values[currentElementIndex] *= -1.;
        }

        matrixA.d_diagonalValues[rowIdx] = (Cvalues[rowIdx] / deltaT) + sum;

        //Compute b element
        vectorB.d_values[rowIdx] = ((Cvalues[rowIdx] / deltaT) * nodeGrid.waterData.oldPressureHead[rowIdx]) + nodeGrid.waterData.waterFlow[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx];

        //Preconditioning
        for(u32_t colIdx = 0; colIdx < matrixA.d_numColsInRow[rowIdx]; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            matrixA.d_values[currentElementIndex] /= matrixA.d_diagonalValues[rowIdx];
        }

        vectorB.d_values[rowIdx] /= matrixA.d_diagonalValues[rowIdx];
    }

    __global__ void computeHeatLinearSystemElement_k(MatrixGPU matrixA, VectorGPU vectorB, const double* Cvalues, double timeStepHeat, double timeStepWater)
    {
        SF3Duint_t rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(rowIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[rowIdx])
            return;

        SF3Duint_t sliceIdx = blockIdx.x;
        SF3Duint_t baseRowIndex = (sliceIdx * maxTotalLink * blockDim.x) + threadIdx.x;

        double nodeT = nodeGrid.heatData.temperature[rowIdx];
        double nodeH = getNodeH_fromTimeSteps(rowIdx, timeStepHeat, timeStepWater);

        double dTheta = computeNodeTheta_fromSignedPsi(rowIdx, nodeH - nodeGrid.z[rowIdx]) -
                        computeNodeTheta_fromSignedPsi(rowIdx, nodeGrid.waterData.oldPressureHead[rowIdx] - nodeGrid.z[rowIdx]);

        double heatCapacity = dTheta * HEAT_CAPACITY_WATER * nodeT;

        if(simulationFlags.computeHeatVapor)
        {
            double dThetaV = computeNodeVaporThetaV(rowIdx, nodeH - nodeGrid.z[rowIdx], nodeT) -
                             computeNodeVaporThetaV(rowIdx, nodeGrid.waterData.oldPressureHead[rowIdx] - nodeGrid.z[rowIdx], nodeGrid.heatData.oldTemperature[rowIdx]);

            heatCapacity += dThetaV * HEAT_CAPACITY_AIR * nodeT;
            heatCapacity += dThetaV * computeLatentVaporizationHeat(nodeT - ZEROCELSIUS) * WATER_DENSITY;
        }

        heatCapacity *= nodeGrid.size[rowIdx];

        u32_t numLinks = 0;
        SF3Duint_t currentElementIndex = baseRowIndex, unsignedTmpColIdx = 0;
        bool isLinked = false;

        isLinked = computeHeatLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 0, timeStepHeat, timeStepWater);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        //Compute flox down
        isLinked = computeHeatLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 1, timeStepHeat, timeStepWater);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        //Compute flux lateral
        for(u32_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
        {
            isLinked = computeHeatLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 2 + latIdx, timeStepHeat, timeStepWater);
            if(isLinked)
            {
                matrixA.d_columnIndeces[currentElementIndex] = static_cast<i64_t>(unsignedTmpColIdx);
                numLinks++;
                currentElementIndex += blockDim.x;
            }
        }

        //Save num cols in current row
        matrixA.d_numColsInRow[rowIdx] = numLinks;

        //Fill not used columns with 0 (values) and -1 (indeces)
        for(u32_t colIdx = numLinks; colIdx < maxTotalLink; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            matrixA.d_values[currentElementIndex] = 0;
            matrixA.d_columnIndeces[currentElementIndex] = -1;
        }

        //Compute diagonal element
        double hWF = solver->getHeatWF();
        double sumDP = 0., sumF0 = 0.;
        for(u32_t colIdx = 1; colIdx < matrixA.d_numColsInRow[rowIdx]; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            sumDP += matrixA.d_values[currentElementIndex] * hWF;
            double dT0 = nodeGrid.heatData.oldTemperature[matrixA.d_columnIndeces[currentElementIndex]] - nodeGrid.heatData.oldTemperature[rowIdx];
            sumF0 += matrixA.d_values[currentElementIndex] * (1. - hWF) * dT0;
            matrixA.d_values[currentElementIndex] *= -(hWF);
        }

        matrixA.d_diagonalValues[rowIdx] = sumDP + (Cvalues[rowIdx] / timeStepHeat);

        //Computeb element
        vectorB.d_values[rowIdx] = Cvalues[rowIdx] * nodeGrid.heatData.oldTemperature[rowIdx] / timeStepHeat - heatCapacity / timeStepHeat +
                                 nodeGrid.heatData.heatFlux[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx] + sumF0;

        //Preconditioning
        if(matrixA.d_diagonalValues[rowIdx] > 0)
        {
            vectorB.d_values[rowIdx] /= matrixA.d_diagonalValues[rowIdx];

            for(u32_t colIdx = 1; colIdx < matrixA.d_numColsInRow[rowIdx]; ++colIdx)
            {
                currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
                matrixA.d_values[currentElementIndex] /= matrixA.d_diagonalValues[rowIdx];
            }
        }

        return;
    }

    __global__ void updateSaturationDegree_k()
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(!nodeGrid.surfaceFlag[nodeIdx])
            nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
    }

    __global__ void updateWaterConductivity_k()
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(!nodeGrid.surfaceFlag[nodeIdx])
            nodeGrid.waterData.waterConductivity[nodeIdx] = computeNodeK(nodeIdx);
    }

    __global__ void updateWaterFlows_k(double deltaT)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        //Update link flows
        for(u8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
            updateLinkFlux(nodeIdx, linkIndex, deltaT);

        //Update boundary flow
        if (nodeGrid.boundaryData.boundaryType[nodeIdx] != boundaryType_t::NoBoundary)
            nodeGrid.boundaryData.waterFlowSum[nodeIdx] += nodeGrid.boundaryData.waterFlowRate[nodeIdx] * deltaT;
    }

    __global__ void computeWaterContent_k(double* outVector)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        double theta = nodeGrid.surfaceFlag[nodeIdx] ? (nodeGrid.waterData.pressureHead[nodeIdx] - nodeGrid.z[nodeIdx]) : computeNodeTheta(nodeIdx);
        outVector[nodeIdx] = theta * nodeGrid.size[nodeIdx];
    }

    __global__ void computeCurrentHeatSinkSource_k(double* d_heatSinkSourceVector, double dtHeat)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        if(nodeGrid.heatData.heatFlux[nodeIdx] != 0.)
            d_heatSinkSourceVector[nodeIdx] = nodeGrid.heatData.heatFlux[nodeIdx] * dtHeat;

        return;
    }

    __global__ void computeCurrentHeatStorage_k(double* d_heatSinkSourceVector, double dtWater,double dtHeat)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        double nodeH = (dtHeat != noDataD && dtWater != noDataD) ? getNodeH_fromTimeSteps(nodeIdx, dtHeat, dtWater) : nodeGrid.waterData.pressureHead[nodeIdx];
        d_heatSinkSourceVector[nodeIdx] = getNodeHeatStorage(nodeIdx, nodeH - nodeGrid.z[nodeIdx]);
    }

    __global__ void saveHeatFluxValues_k(double dtHeat, double dtWater)
    {
        SF3Duint_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        for(u32_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
            saveNodeHeatFluxes(nodeIdx, lIdx, dtHeat, dtWater);

        return;
    }

}
