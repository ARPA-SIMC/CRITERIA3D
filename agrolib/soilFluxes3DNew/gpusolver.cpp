#include <cassert>
#include <type_traits>

#include "gpusolver.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"
#include "otherFunctions.h"

#include "mat.h"
#include "matrix.h"

using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Math;
using namespace soilFluxes3D::Heat;
using namespace soilFluxes3D::Water;

namespace soilFluxes3D::New
{
    extern __cudaMngd Solver* solver;
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;
    extern __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;
    extern std::vector<std::vector<soilData_t>> soilList;
    extern std::vector<surfaceData_t> surfaceList;

    SF3Derror_t GPUSolver::inizialize()
    {
        if(_status != Created)
            return SolverError;

        if(_parameters.deltaTcurr == noData)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;                   //TO DO: (nodeGrid.numNodes > ...);
        if(_parameters.enableOMP)
            omp_set_num_threads(static_cast<int>(_parameters.numThreads));

        numThreadsPerBlock = 64;                        //Must be multiple of warp-size(32)
        numBlocks = (uint64_t) std::ceil((double) nodeGrid.numNodes / numThreadsPerBlock);

        cuspCheck(cusparseCreate(&libHandle));

        //Inizialize matrix data
        iterationMatrix.numRows = static_cast<int64_t>(nodeGrid.numNodes);
        iterationMatrix.numCols = static_cast<int64_t>(nodeGrid.numNodes);
        iterationMatrix.sliceSize = static_cast<int64_t>(numThreadsPerBlock);
        uint64_t numSlice = numBlocks;
        uint64_t tnv = numSlice * iterationMatrix.sliceSize * maxTotalLink;
        iterationMatrix.totValuesSize = static_cast<int64_t>(tnv);

        deviceSolverAlloc(iterationMatrix.d_numColsInRow, uint16_t, iterationMatrix.numRows);

        deviceSolverAlloc(iterationMatrix.d_offsets, int64_t, (numSlice + 1));
        deviceSolverAlloc(iterationMatrix.d_columnIndeces, int64_t, tnv);
        deviceSolverAlloc(iterationMatrix.d_values, double, tnv);

        deviceSolverAlloc(iterationMatrix.d_diagonalValues, double, iterationMatrix.numRows);

        //Inizialize vectors data
        constantTerm.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(constantTerm.d_values, double, constantTerm.numElements);

        unknownTerm.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(unknownTerm.d_values, double, unknownTerm.numElements);

        tempSolution.numElements = nodeGrid.numNodes;
        deviceSolverAlloc(tempSolution.d_values, double, tempSolution.numElements);

        //Inizialize cuSPARSE descriptor
        createCUsparseDescriptors();

        //Inizialize raw capacity vector
        deviceSolverAlloc(d_Cvalues, double, nodeGrid.numNodes);

        _status = Inizialized;
        return SF3Dok;
    }

    SF3Derror_t GPUSolver::createCUsparseDescriptors()
    {
        int64_t *d_nnz = nullptr;
        cudaMalloc((void**) &d_nnz, sizeof(int64_t));

        void* d_tempStorage = nullptr;
        size_t tempStorageSize = 0;
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, iterationMatrix.d_numColsInRow, d_nnz, iterationMatrix.numRows);
        cudaMalloc(&d_tempStorage, tempStorageSize);
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, iterationMatrix.d_numColsInRow, d_nnz, iterationMatrix.numRows);
        cudaMemcpy(&(iterationMatrix.numNonZeroElement), d_nnz, sizeof(int64_t), cudaMemcpyDeviceToHost);

        cudaFree(d_tempStorage);
        cudaFree(d_nnz);

        //Setup offsets vector
        int64_t* h_offsets = static_cast<int64_t*>(std::calloc((numBlocks + 1), sizeof(int64_t)));
        for (int64_t idx = 0; idx < (numBlocks + 1); ++idx)
            h_offsets[idx] = idx * numThreadsPerBlock * maxTotalLink;

        cudaMemcpy(iterationMatrix.d_offsets, h_offsets, (numBlocks + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);


        //Create matrix descriptor
        cuspCheck(cusparseCreateSlicedEll(&(iterationMatrix.cusparseDescriptor),
                                            iterationMatrix.numRows, iterationMatrix.numCols, iterationMatrix.numNonZeroElement, iterationMatrix.totValuesSize, iterationMatrix.sliceSize,
                                            iterationMatrix.d_offsets, iterationMatrix.d_columnIndeces,  iterationMatrix.d_values,
                                            iterationMatrix.offsetType, iterationMatrix.colIdxType, iterationMatrix.idxBase, iterationMatrix.valueType));

        //Create vector descriptor
        cuspCheck(cusparseCreateDnVec(&(constantTerm.cusparseDescriptor), constantTerm.numElements, constantTerm.d_values, constantTerm.valueType));
        cuspCheck(cusparseCreateDnVec(&(unknownTerm.cusparseDescriptor), unknownTerm.numElements, unknownTerm.d_values, unknownTerm.valueType));
        cuspCheck(cusparseCreateDnVec(&(tempSolution.cusparseDescriptor), tempSolution.numElements, tempSolution.d_values, tempSolution.valueType));

        return SF3Dok;
    }

    SF3Derror_t GPUSolver::run(double maxTimeStep, double &acceptedTimeStep, processType process)
    {
        if(_status != Inizialized)
            return SolverError;

        if(maxTimeStep == HOUR_SECONDS)
            upCopyData();

        switch (process)
        {
            case Water:
                waterMainLoop(maxTimeStep, acceptedTimeStep);
                break;
            case Heat:
                break;
            case Solutes:
                break;
            default:
                break;
        }

        _status = Terminated;
        if(acceptedTimeStep == maxTimeStep)
            downCopyData();

        _status = Inizialized;
        return SF3Dok;
    }

    SF3Derror_t GPUSolver::clean()
    {
        if(_status == Created)
            return SF3Dok;

        if((_status != Terminated) && (_status != Inizialized))
            return SolverError;

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

        _status = Created;
        return SF3Dok;
    }

    void GPUSolver::waterMainLoop(double maxTimeStep, double &acceptedTimeStep)
    {
        balanceResult_t stepStatus = stepRefused;
        while(stepStatus != stepAccepted)
        {
            acceptedTimeStep = SF3Dmin(_parameters.deltaTcurr, maxTimeStep);

            //Save instantaneus H values
            //std::memcpy(nodeGrid.waterData.oldPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double));
            cudaMemcpy(nodeGrid.waterData.oldPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

            //Inizialize the solution vector with the current pressure head
            assert(unknownTerm.numElements == nodeGrid.numNodes);
            //cudaMemcpy(vectorX.values, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(unknownTerm.d_values, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

            launchKernel(inizializeCapacityAndSaturationDegree_k, d_Cvalues);

            //Update aereodynamic and soil conductance
            //updateConductance();      //TO DO (Heat)

            //Update boundary
            launchKernel(updateBoundaryWaterData_k, acceptedTimeStep);

            //Effective computation step
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            if(stepStatus != stepAccepted)  //old restorePressureHead();
                cudaMemcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHeads, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }

    balanceResult_t GPUSolver::waterApproximationLoop(double deltaT)
    {
        balanceResult_t balanceResult;

        for(uint8_t approxIdx = 0; approxIdx < _parameters.maxApproximationsNumber; ++approxIdx)
        {
            //Compute capacity vector elements
            launchKernel(computeCapacity_k, d_Cvalues);

            //Update boundary water
            launchKernel(updateBoundaryWaterData_k, deltaT);

            //Update Courant data
            nodeGrid.waterData.CourantWaterLevel = 0.;

            //Compute linear system elements
            launchKernel(computeLinearSystemElement_k, iterationMatrix, constantTerm, d_Cvalues, approxIdx, deltaT, _parameters.lateralVerticalRatio, _parameters.meantype);

            //Check Courant
            if((nodeGrid.waterData.CourantWaterLevel > 1.) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / nodeGrid.waterData.CourantWaterLevel);
                if(_parameters.deltaTcurr > 1.)
                    _parameters.deltaTcurr = std::floor(_parameters.deltaTcurr);

                return stepHalved;
            }

            //Try solve linear system
            bool isStepValid = solveLinearSystem(approxIdx, Water);

            //Reduce step tipe if system resolution failed
            if((!isStepValid) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / 2.);
                return stepHalved;
            }

            //Update potential
            cudaMemcpy(nodeGrid.waterData.pressureHead, unknownTerm.d_values, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

            //Update degree of saturation
            launchKernel(updateSaturationDegree_k);

            //Check water balance
            balanceResult = evaluateWaterBalance_m(approxIdx, _bestMBRerror, deltaT);

            if((balanceResult == stepAccepted) || (balanceResult == stepHalved))
                return balanceResult;
        }

        return balanceResult;
    }

    bool GPUSolver::solveLinearSystem(uint8_t approximationNumber, processType computationType) //Implemented only Jacobi Water
    {
        size_t bufSize;
        const double alpha = -1, beta = 1;
        void *externalBuffer = nullptr;

        cusparseSpMV_bufferSize(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, unknownTerm.cusparseDescriptor, &beta, tempSolution.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
        cudaDeviceSynchronize();
        cudaMalloc(&externalBuffer, bufSize);

        bool status = true;
        double bestErrorNorm = (double) std::numeric_limits<float>::max();

        double *d_tempVector = nullptr;
        cudaMalloc((void**) &d_tempVector, unknownTerm.numElements * sizeof(double));

        //TO DO: implement cusparseSpMV_preprocess

        uint32_t currMaxIterationNum = calcCurrentMaxIterationNumber(approximationNumber);
        uint32_t GPUfactor = 1;           //TO DO: test and optimize
        for (size_t iterationNumber = 0; iterationNumber < currMaxIterationNum; ++iterationNumber)
        {
            for(size_t internalCounter = 0; internalCounter < GPUfactor; ++internalCounter)
            {
                cudaMemcpy(tempSolution.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();

                cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, unknownTerm.cusparseDescriptor, &beta, tempSolution.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
                cudaDeviceSynchronize();

                //TEMP: single iteration to mimic CPU behaviour

                cudaMemcpy(d_tempVector, unknownTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpy(unknownTerm.d_values, tempSolution.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpy(tempSolution.d_values, d_tempVector, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);

                //cudaMemcpy(unknownTerm.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
                //cudaDeviceSynchronize();

                //cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, tempSolution.cusparseDescriptor, &beta, unknownTerm.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
                //cudaDeviceSynchronize();
            }

            //Calcolo della norma dell'errore
            double currErrorNorm = 0;
            double *d_normValue = nullptr, *d_normVector = nullptr;
            cudaMalloc((void**) &d_normValue, sizeof(double));
            cudaMalloc((void**) &d_normVector, unknownTerm.numElements * sizeof(double));
            launchKernel(computeNormalizedError, d_normVector, unknownTerm.d_values, tempSolution.d_values);

            void* d_tempStorage = nullptr;
            size_t tempStorageSize = 0;
            cub::DeviceReduce::Max(d_tempStorage, tempStorageSize, d_normVector, d_normValue, unknownTerm.numElements);
            cudaMalloc(&d_tempStorage, tempStorageSize);
            cub::DeviceReduce::Max(d_tempStorage, tempStorageSize, d_normVector, d_normValue, unknownTerm.numElements);
            cudaMemcpy(&currErrorNorm, d_normValue, sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(d_tempStorage);
            cudaFree(d_normValue);
            cudaFree(d_normVector);

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

        cudaFree(externalBuffer); externalBuffer = nullptr;
        cudaFree(d_tempVector); d_tempVector = nullptr;

        return status;
    }


    //------------------------------- TEMP FUNCTIONS -------------------------------
    balanceResult_t GPUSolver::evaluateWaterBalance_m(uint8_t approxNr, double& bestMBRerror, double deltaT)
    {
        computeCurrentMassBalance_m(deltaT);

        double currMBRerror = std::fabs(balanceDataCurrentTimeStep.waterMBR);

        //Optimal error
        if(currMBRerror < _parameters.MBRThreshold)
        {
            acceptStep_m(deltaT);

            //Check Stability (Courant)
            double currCWL = nodeGrid.waterData.CourantWaterLevel;
            if((currCWL < _parameters.CourantWaterThreshold) && (approxNr <= 3) && (currMBRerror < (0.5 * _parameters.MBRThreshold)))     //TO DO: change constant with _parameters
            {
                //increase deltaT
                _parameters.deltaTcurr = (currCWL < 0.5) ? (2 * _parameters.deltaTcurr) : (_parameters.deltaTcurr / currCWL);
                _parameters.deltaTcurr = SF3Dmin(_parameters.deltaTcurr, _parameters.deltaTmax);
                if(_parameters.deltaTcurr > 1.)
                    _parameters.deltaTcurr = std::floor(_parameters.deltaTcurr);
            }
            return stepAccepted;
        }

        //Good error or first approximation
        if (approxNr == 0 || currMBRerror < bestMBRerror)
        {
            //saveBestStep();
            cudaMemcpy(nodeGrid.waterData.bestPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);
            bestMBRerror = currMBRerror;
        }

        //Critical error (unstable system) or last approximation
        if (approxNr == (_parameters.maxApproximationsNumber - 1) || currMBRerror > (bestMBRerror * _parameters.instabilityFactor))
        {
            if(deltaT > _parameters.deltaTmin)
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTcurr / 2, _parameters.deltaTmin);
                return stepHalved;
            }

            restoreBestStep_m(deltaT);
            acceptStep_m(deltaT);
            return stepAccepted;
        }

        return stepRefused;
    }

    void GPUSolver::computeCurrentMassBalance_m(double deltaT)
    {
        balanceDataCurrentTimeStep.waterStorage = computeTotalWaterContent_m();
        double deltaStorage = balanceDataCurrentTimeStep.waterStorage - balanceDataPreviousTimeStep.waterStorage;

        balanceDataCurrentTimeStep.waterSinkSource = computeWaterSinkSourceFlowsSum_m(deltaT);
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

    double GPUSolver::computeTotalWaterContent_m()
    {
        if(!nodeGrid.isInizialized)
            return -1;

        double* d_waterContentVector = nullptr;
        cudaMalloc((void**) &d_waterContentVector, sizeof(double));

        launchKernel(computeWaterContent_k, d_waterContentVector);

        double *d_sum = nullptr, h_sum = 0;
        cudaMalloc((void**) &d_sum, sizeof(double));

        void* d_tempStorage = nullptr;
        size_t tempStorageSize = 0;
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, d_waterContentVector, d_sum, nodeGrid.numNodes);
        cudaMalloc(&d_tempStorage, tempStorageSize);
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, d_waterContentVector, d_sum, nodeGrid.numNodes);
        cudaMemcpy(&(h_sum), d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_waterContentVector);
        cudaFree(d_tempStorage);
        cudaFree(d_sum);
        return h_sum;
    }

    double GPUSolver::computeWaterSinkSourceFlowsSum_m(double deltaT)
    {
        double *d_sum = nullptr, h_sum = 0;
        cudaMalloc((void**) &d_sum, sizeof(double));

        void* d_tempStorage = nullptr;
        size_t tempStorageSize = 0;
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, nodeGrid.waterData.waterFlow, d_sum, nodeGrid.numNodes);
        cudaMalloc(&d_tempStorage, tempStorageSize);
        cub::DeviceReduce::Sum(d_tempStorage, tempStorageSize, nodeGrid.waterData.waterFlow, d_sum, nodeGrid.numNodes);
        cudaMemcpy(&(h_sum), d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_tempStorage);
        cudaFree(d_sum);
        return h_sum * deltaT;
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
        cudaMemcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.bestPressureHeads, nodeGrid.numNodes * sizeof(double), cudaMemcpyDeviceToDevice);

        launchKernel(updateSaturationDegree_k);

        computeCurrentMassBalance_m(deltaT);
    }


    //------------------------------- MOVE FUNCTIONS -------------------------------

    SF3Derror_t GPUSolver::upCopyData()
    {
        solverCheck(upMoveSoilSurfacePtr());      //Need to be done before moving nodeGrid.surfaceFlag

        //Topology Data
        moveToDevice(nodeGrid.size, double, nodeGrid.numNodes);
        moveToDevice(nodeGrid.x, double, nodeGrid.numNodes);
        moveToDevice(nodeGrid.y, double, nodeGrid.numNodes);
        moveToDevice(nodeGrid.z, double, nodeGrid.numNodes);
        moveToDevice(nodeGrid.surfaceFlag, bool, nodeGrid.numNodes);

        //Soil/surface properties pointers
        moveToDevice(nodeGrid.soilSurfacePointers, soil_surface_ptr, nodeGrid.numNodes);

        //Boundary data
        moveToDevice(nodeGrid.boundaryData.boundaryType, boundaryType_t, nodeGrid.numNodes);
        moveToDevice(nodeGrid.boundaryData.boundarySlope, double, nodeGrid.numNodes);
        moveToDevice(nodeGrid.boundaryData.boundarySize, double, nodeGrid.numNodes);
        if(simulationFlags.computeWater)
        {
            moveToDevice(nodeGrid.boundaryData.waterFlowRate, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.waterFlowSum, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.boundaryData.prescribedWaterPotential, double, nodeGrid.numNodes);
        }

        //Link data
        moveToDevice(nodeGrid.numLateralLink, uint8_t, nodeGrid.numNodes);
        for(uint8_t idx = 0; idx < maxTotalLink; ++idx)
        {
            moveToDevice(nodeGrid.linkData[idx].linktype, linkType_t, nodeGrid.numNodes);
            moveToDevice(nodeGrid.linkData[idx].linkIndex, uint64_t, nodeGrid.numNodes);
            moveToDevice(nodeGrid.linkData[idx].interfaceArea, double, nodeGrid.numNodes);
            if(simulationFlags.computeWater)
                moveToDevice(nodeGrid.linkData[idx].waterFlowSum, double, nodeGrid.numNodes);
        }

        //Water data
        if(simulationFlags.computeWater)
        {
            moveToDevice(nodeGrid.waterData.saturationDegree, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.waterConductivity, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.waterFlow, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.pressureHead, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.waterSinkSource, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.pond, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.invariantFluxes, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.oldPressureHeads, double, nodeGrid.numNodes);
            moveToDevice(nodeGrid.waterData.bestPressureHeads, double, nodeGrid.numNodes);
        }

        return SF3Dok;
    }

    SF3Derror_t GPUSolver::upMoveSoilSurfacePtr()
    {
        static_assert(std::is_trivially_copyable<soilData_t>::value, "...");
        static_assert(std::is_trivially_copyable<surfaceData_t>::value, "...");

        //Move surface data
        size_t surfaceSize = surfaceList.size() * sizeof(surfaceData_t);
        cudaCheck(cudaMalloc((void**) &d_surfaceList, surfaceSize));
        cudaCheck(cudaMemcpy(d_surfaceList, surfaceList.data(), surfaceSize, cudaMemcpyHostToDevice));

        //Flat soil data
        size_t soilSize = 0;
        std::vector<soilData_t> soilList1D;
        std::vector<size_t> listOffsets = {0};
        listOffsets.reserve(soilList.size() + 1);
        for(const auto& soilRow : soilList)
        {
            soilSize += soilRow.size();
            soilList1D.insert(soilList1D.end(), soilRow.begin(), soilRow.end());
            listOffsets.push_back(soilSize);
        }

        //Move soil data
        soilSize *= sizeof(soilData_t);
        cudaCheck(cudaMalloc((void**) &d_soilList, soilSize));
        cudaCheck(cudaMemcpy(d_soilList, soilList1D.data(), soilSize, cudaMemcpyHostToDevice));

        #pragma omp parallel for if(_parameters.enableOMP)
        for(uint64_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            auto& pointer = nodeGrid.soilSurfacePointers[nodeIndex];
            ptrdiff_t currOffset;
            if(nodeGrid.surfaceFlag[nodeIndex])
            {
                currOffset = pointer.surfacePtr - surfaceList.data();
                pointer.surfacePtr = d_surfaceList + currOffset;
            }
            else
            {
                uint16_t soilRowIndex = nodeGrid.soilRowIndeces[nodeIndex];
                currOffset = listOffsets[soilRowIndex] + (pointer.soilPtr - soilList[soilRowIndex].data());
                pointer.soilPtr = d_soilList + currOffset;
            }
        }

        return SF3Dok;
    }

    SF3Derror_t GPUSolver::downCopyData()
    {
        moveToHost(nodeGrid.size, double, nodeGrid.numNodes);
        moveToHost(nodeGrid.x, double, nodeGrid.numNodes);
        moveToHost(nodeGrid.y, double, nodeGrid.numNodes);
        moveToHost(nodeGrid.z, double, nodeGrid.numNodes);
        moveToHost(nodeGrid.surfaceFlag, bool, nodeGrid.numNodes);

        //Soil/surface properties pointers
        moveToHost(nodeGrid.soilSurfacePointers, soil_surface_ptr, nodeGrid.numNodes);
        solverCheck(downMoveSoilSurfacePtr());      //Need to be done after moving nodeGrid.surfaceFlag

        //Boundary data
        moveToHost(nodeGrid.boundaryData.boundaryType, boundaryType_t, nodeGrid.numNodes);
        moveToHost(nodeGrid.boundaryData.boundarySlope, double, nodeGrid.numNodes);
        moveToHost(nodeGrid.boundaryData.boundarySize, double, nodeGrid.numNodes);
        if(simulationFlags.computeWater)
        {
            moveToHost(nodeGrid.boundaryData.waterFlowRate, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.waterFlowSum, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.boundaryData.prescribedWaterPotential, double, nodeGrid.numNodes);
        }

        //Link data
        moveToHost(nodeGrid.numLateralLink, uint8_t, nodeGrid.numNodes);
        for(uint8_t idx = 0; idx < maxTotalLink; ++idx)
        {
            moveToHost(nodeGrid.linkData[idx].linktype, linkType_t, nodeGrid.numNodes);
            moveToHost(nodeGrid.linkData[idx].linkIndex, uint64_t, nodeGrid.numNodes);
            moveToHost(nodeGrid.linkData[idx].interfaceArea, double, nodeGrid.numNodes);
            if(simulationFlags.computeWater)
                moveToHost(nodeGrid.linkData[idx].waterFlowSum, double, nodeGrid.numNodes);
        }

        //Water data
        if(simulationFlags.computeWater)
        {
            moveToHost(nodeGrid.waterData.saturationDegree, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.waterConductivity, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.waterFlow, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.pressureHead, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.waterSinkSource, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.pond, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.invariantFluxes, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.oldPressureHeads, double, nodeGrid.numNodes);
            moveToHost(nodeGrid.waterData.bestPressureHeads, double, nodeGrid.numNodes);
        }

        return SF3Dok;
    }

    SF3Derror_t GPUSolver::downMoveSoilSurfacePtr()
    {
        //Flat soil data
        size_t soilSize = 0;
        std::vector<size_t> listOffsets;
        listOffsets.reserve(soilList.size());
        for (auto soilRow : soilList)
        {
            listOffsets.push_back(soilSize);
            soilSize += soilRow.size();
        }

        #pragma omp parallel for if(_parameters.enableOMP)
        for(uint64_t nodeIndex = 0; nodeIndex < nodeGrid.numNodes; ++nodeIndex)
        {
            ptrdiff_t currOffset;
            if(nodeGrid.surfaceFlag[nodeIndex])
            {
                currOffset = nodeGrid.soilSurfacePointers[nodeIndex].surfacePtr - d_surfaceList;
                nodeGrid.soilSurfacePointers[nodeIndex].surfacePtr = surfaceList.data() + currOffset;
            }
            else
            {
                uint16_t soilRowIndex = nodeGrid.soilRowIndeces[nodeIndex];
                currOffset = nodeGrid.soilSurfacePointers[nodeIndex].soilPtr - (d_soilList + listOffsets[soilRowIndex]);
                nodeGrid.soilSurfacePointers[nodeIndex].soilPtr = soilList[soilRowIndex].data() + currOffset;
            }
        }

        deviceSolverFree(d_surfaceList);
        deviceSolverFree(d_soilList);
        return SF3Dok;
    }


    //-------------------------------- CUDA KERNELS --------------------------------

    extern __cudaMngd Solver* solver;

    __global__ void inizializeCapacityAndSaturationDegree_k(double* vectorC)    //TO DO: refactor with host code in single __host__ __device__ function
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(nodeGrid.surfaceFlag[nodeIdx])
            vectorC[nodeIdx] = nodeGrid.size[nodeIdx];
        else
            nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
    }

    __global__ void computeCapacity_k(double* vectorC)                          //TO DO: merge with host code in single __host__ __device__ function
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        nodeGrid.waterData.invariantFluxes[nodeIdx] = 0.;
        if(nodeGrid.surfaceFlag[nodeIdx])
            return;

        //Compute hydraulic conductivity
        nodeGrid.waterData.waterConductivity[nodeIdx] = computeNodeK(nodeIdx);

        double dThetadH = computeNodedThetadH(nodeIdx);
        vectorC[nodeIdx] = nodeGrid.size[nodeIdx] * dThetadH;

        // if(simulationFlags.computeHeat && simulationFlags.computeHeatVapor)
        //     vectorC[nodeIdx] += nodeGrid.size[nodeIdx] * computeNodedThetaVdH(nodeIdx, getNodeMeanTemperature(nodeIdx), dThetadH);
    }

    __global__ void updateBoundaryWaterData_k(double deltaT)                    //TO DO: merge with host code in single __host__ __device__ function
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        nodeGrid.waterData.waterFlow[nodeIdx] = nodeGrid.waterData.waterSinkSource[nodeIdx];

        if(nodeGrid.boundaryData.boundaryType[nodeIdx] == NoBoundary)
            return;

        nodeGrid.boundaryData.waterFlowRate[nodeIdx] = 0;

        switch(nodeGrid.boundaryData.boundaryType[nodeIdx])
        {
            case Runoff:
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

            case FreeDrainage:
                //Darcy unit gradient (use link node up)
                assert(nodeGrid.linkData[0].linktype[nodeIdx] != NoLink);
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.linkData[0].interfaceArea[nodeIdx];
                break;

            case FreeLateraleDrainage:
                //Darcy gradient = slope
                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = -nodeGrid.waterData.waterConductivity[nodeIdx] * nodeGrid.boundaryData.boundarySize[nodeIdx]
                                                               * nodeGrid.boundaryData.boundarySlope[nodeIdx] * (*solver).getLVRatio();
                break;

            case PrescribedTotalWaterPotential:
                double L, boundaryPsi, boundaryZ, boundaryK, meanK, dH;
                L = 1.;     // [m]
                boundaryZ = nodeGrid.z[nodeIdx] - L;

                boundaryPsi = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - boundaryZ;

                boundaryK = (boundaryPsi >= 0)  ? nodeGrid.soilSurfacePointers[nodeIdx].soilPtr->K_sat
                                               : computeNodeK_Mualem(*(nodeGrid.soilSurfacePointers[nodeIdx].soilPtr), computeNodeSe_fromPsi(nodeIdx, fabs(boundaryPsi)));

                meanK = computeMean(boundaryK, nodeGrid.waterData.waterConductivity[nodeIdx], (*solver).getMeanType());
                dH = nodeGrid.boundaryData.prescribedWaterPotential[nodeIdx] - nodeGrid.waterData.pressureHead[nodeIdx];

                nodeGrid.boundaryData.waterFlowRate[nodeIdx] = meanK * nodeGrid.boundaryData.boundarySize[nodeIdx] * (dH / L);
                break;

            case HeatSurface:
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

    __global__ void computeNormalizedError(double *vectorNorm, double *vectorX, const double *previousX)
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
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

    __global__ void computeLinearSystemElement_k(MatrixGPU matrixA, VectorGPU vectorB, const double* Cvalues, uint8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType)
    {
        uint64_t sliceIdx = blockIdx.x;
        uint64_t rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(rowIdx >= nodeGrid.numNodes)
            return;

        uint64_t baseRowIndex = (sliceIdx * maxTotalLink * blockDim.x) + threadIdx.x;

        uint32_t numLinks = 0;
        uint64_t currentElementIndex = baseRowIndex, unsignedTmpColIdx = 0;
        bool isLinked;

        //Compute flux up
        isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 0, approxNum, deltaT, lateralVerticalRatio, Up, meanType);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<int64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        //Compute flox down
        isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 1, approxNum, deltaT, lateralVerticalRatio, Down, meanType);
        if(isLinked)
        {
            matrixA.d_columnIndeces[currentElementIndex] = static_cast<int64_t>(unsignedTmpColIdx);
            numLinks++;
            currentElementIndex += blockDim.x;
        }

        //Compute flux lateral
        for(uint32_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
        {
            isLinked = computeLinkFluxes(matrixA.d_values[currentElementIndex], unsignedTmpColIdx, rowIdx, 2 + latIdx, approxNum, deltaT, lateralVerticalRatio, Lateral, meanType);
            if(isLinked)
            {
                matrixA.d_columnIndeces[currentElementIndex] = static_cast<int64_t>(unsignedTmpColIdx);
                numLinks++;
                currentElementIndex += blockDim.x;
            }
        }

        //Save num cols in current row
        matrixA.d_numColsInRow[rowIdx] = numLinks;

        //Fill not used columns with 0 (values) and -1 (indeces)
        for(uint32_t colIdx = numLinks; colIdx < maxTotalLink; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            matrixA.d_values[currentElementIndex] = 0;
            matrixA.d_columnIndeces[currentElementIndex] = -1;
        }

        //Compute diagonal element
        double sum = 0.;
        for(uint32_t colIdx = 0; colIdx < numLinks; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            sum += matrixA.d_values[currentElementIndex];
            matrixA.d_values[currentElementIndex] *= -1.;
        }

        matrixA.d_diagonalValues[rowIdx] = (Cvalues[rowIdx] / deltaT) + sum;

        //Compute b element
        vectorB.d_values[rowIdx] = ((Cvalues[rowIdx] / deltaT) * nodeGrid.waterData.oldPressureHeads[rowIdx]) + nodeGrid.waterData.waterFlow[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx];

        //Preconditioning
        for(uint32_t colIdx = 0; colIdx < matrixA.d_numColsInRow[rowIdx]; ++colIdx)
        {
            currentElementIndex = baseRowIndex + (colIdx * blockDim.x);
            matrixA.d_values[currentElementIndex] /= matrixA.d_diagonalValues[rowIdx];
        }

        vectorB.d_values[rowIdx] /= matrixA.d_diagonalValues[rowIdx];
    }

    __global__ void updateSaturationDegree_k()
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        if(!nodeGrid.surfaceFlag[nodeIdx])
            nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
    }

    __global__ void updateWaterFlows_k(double deltaT)
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        //Update link flows
        for(uint8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
            updateLinkFlux(nodeIdx, linkIndex, deltaT);

        //Update boundary flow
        if (nodeGrid.boundaryData.boundaryType[nodeIdx] != NoBoundary)
            nodeGrid.boundaryData.waterFlowSum[nodeIdx] += nodeGrid.boundaryData.waterFlowRate[nodeIdx] * deltaT;
    }

    __global__ void computeWaterContent_k(double* outVector)
    {
        uint64_t nodeIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(nodeIdx >= nodeGrid.numNodes)
            return;

        double theta = nodeGrid.surfaceFlag[nodeIdx] ? (nodeGrid.waterData.pressureHead[nodeIdx] - nodeGrid.z[nodeIdx]) : computeNodeTheta(nodeIdx);
        outVector[nodeIdx] = theta * nodeGrid.size[nodeIdx];
    }
}
