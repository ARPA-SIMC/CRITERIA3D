#include <cstring>
#include <cassert>
#include <iostream>

#include "cpusolver.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"
#include "otherFunctions.h"

using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Water;
using namespace soilFluxes3D::Heat;
using namespace soilFluxes3D::Math;

namespace soilFluxes3D::New
{
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;

    SF3Derror_t CPUSolver::initialize()
    {
        if(_status != solverStatus::Created)
            return SF3Derror_t::SolverError;

        if(_parameters.deltaTcurr == noDataD)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;       //TO DO: (nodeGrid.numNodes > ...);
        if(_parameters.enableOMP)
            omp_set_num_threads(static_cast<int>(_parameters.numThreads));

        //initialize matrix structure
        matrixA.numRows = nodeGrid.numNodes;
        hostSolverAlloc(matrixA.numColumns, matrixA.numRows);
        hostSolverAlloc(matrixA.colIndeces, matrixA.numRows);
        hostSolverAlloc(matrixA.values, matrixA.numRows);

        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            hostSolverAlloc(matrixA.colIndeces[rowIdx], matrixA.maxColumns);
            hostSolverAlloc(matrixA.values[rowIdx], matrixA.maxColumns);
        }

        //initialize vector data
        vectorX.numElements = nodeGrid.numNodes;
        hostSolverAlloc(vectorX.values, vectorX.numElements);

        vectorB.numElements = nodeGrid.numNodes;
        hostSolverAlloc(vectorB.values, vectorB.numElements);

        vectorC.numElements = nodeGrid.numNodes;
        hostSolverAlloc(vectorC.values, vectorC.numElements);

        _status = solverStatus::initialized;
        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t CPUSolver::run(double maxTimeStep, double& acceptedTimeStep, processType process)
    {
        if(_status != solverStatus::initialized)
            return SF3Derror_t::SolverError;

        switch (process)
        {
            case processType::Water:
                waterMainLoop(maxTimeStep, acceptedTimeStep);
                break;
            case processType::Heat:
                break;
            case processType::Solutes:
                throw std::runtime_error("Solutes not available with GPUSolver");
                break;
            default:
                break;
        }

        _status = solverStatus::Terminated;
        _status = solverStatus::initialized;
        return SF3Derror_t::SF3Dok;
    }

    SF3Derror_t CPUSolver::clean()
    {
        if(_status == solverStatus::Created)
            return SF3Derror_t::SF3Dok;

        if((_status != solverStatus::Terminated) && (_status != solverStatus::initialized))
            return SF3Derror_t::SolverError;

        //Destruct matrix variable
        #pragma omp parallel for if(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            hostSolverFree(matrixA.colIndeces[rowIdx]);
            hostSolverFree(matrixA.values[rowIdx]);
        }
        hostSolverFree(matrixA.numColumns);
        hostSolverFree(matrixA.colIndeces);
        hostSolverFree(matrixA.values);

        //Destruct matrix variable
        hostSolverFree(vectorX.values);

        hostSolverFree(vectorB.values);

        hostSolverFree(vectorC.values);

        _status = solverStatus::Created;
        return SF3Derror_t::SF3Dok;
    }

    void CPUSolver::waterMainLoop(double maxTimeStep, double &acceptedTimeStep)
    {
        balanceResult_t stepStatus = balanceResult_t::stepRefused;
        while(stepStatus != balanceResult_t::stepAccepted)
        {
            acceptedTimeStep = SF3Dmin(_parameters.deltaTcurr, maxTimeStep);

            //Save instantaneus H values
            std::memcpy(nodeGrid.waterData.oldPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double));

            //initialize the solution vector with the current pressure head
            assert(vectorX.numElements == nodeGrid.numNodes);
            std::memcpy(vectorX.values, nodeGrid.waterData.pressureHead, vectorX.numElements * sizeof(double));

            //Assign vectorC surface values and compute subsurface saturation degree
            #pragma omp parallel for if(_parameters.enableOMP)
            for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
            {
                if(nodeGrid.surfaceFlag[nodeIdx])
                    vectorC.values[nodeIdx] = nodeGrid.size[nodeIdx];
                else
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
            }

            //Update aereodynamic and soil conductance
            updateConductance();

            //Update boundary
            updateBoundaryWaterData(acceptedTimeStep);

            //Effective computation step
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            if(stepStatus != balanceResult_t::stepAccepted)
                restorePressureHead();
        }
    }

    balanceResult_t CPUSolver::waterApproximationLoop(double deltaT)
    {
        balanceResult_t balanceResult;

        for(uint8_t approxIdx = 0; approxIdx < _parameters.maxApproximationsNumber; ++approxIdx)
        {
            //Compute capacity vector elements
            computeCapacity(vectorC);
            logStruct;

            //Update boundary water
            updateBoundaryWaterData(deltaT);
            logStruct;

            //Reset Courant data
            nodeGrid.waterData.CourantWaterLevel = 0.;
            std::memset(nodeGrid.waterData.partialCourantWaterLevels, 0, nodeGrid.numNodes * sizeof(double));

            //Compute linear system elements
            computeLinearSystemElement(matrixA, vectorB, vectorC, approxIdx, deltaT, _parameters.lateralVerticalRatio, _parameters.meanType);

            //Courant data reduction
            double tempMax = 0;
            #pragma omp parallel for reduction(max:tempMax)
            for(SF3Duint_t idx = 0; idx < nodeGrid.numNodes; ++idx)
                tempMax = SF3Dmax(tempMax, nodeGrid.waterData.partialCourantWaterLevels[idx]);

            nodeGrid.waterData.CourantWaterLevel = tempMax;
            logStruct;

            //Check Courant
            if((nodeGrid.waterData.CourantWaterLevel > 1.) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / nodeGrid.waterData.CourantWaterLevel);
                if(_parameters.deltaTcurr > 1.)
                    _parameters.deltaTcurr = std::floor(_parameters.deltaTcurr);

                return balanceResult_t::stepHalved;
            }


            //Try solve linear system
            bool isStepValid = solveLinearSystem(approxIdx, processType::Water);
            logStruct;

            //Log system data
            logSystem;

            //Reduce step tipe if system resolution failed
            if((!isStepValid) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / 2.);
                return balanceResult_t::stepHalved;
            }

            //Update potential
            assert(vectorX.numElements == nodeGrid.numNodes);
            std::memcpy(nodeGrid.waterData.pressureHead, vectorX.values, vectorX.numElements * sizeof(double));

            //Update degree of saturation   //TO DO: make a function
            #pragma omp parallel for if(_parameters.enableOMP)
            for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
                if(!nodeGrid.surfaceFlag[nodeIdx])
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);

            //Check water balance
            balanceResult = evaluateWaterBalance(approxIdx, _bestMBRerror, deltaT, _parameters);
            logStruct;

            if((balanceResult == balanceResult_t::stepAccepted) || (balanceResult == balanceResult_t::stepHalved))
                return balanceResult;

        }

        return balanceResult;
    }

    void CPUSolver::heatLoop(double timeStepHeat, double timeStepWater)
    {
        resetFluxValues(true, false);

        //initialize vector X
        std::memcpy(vectorX.values, nodeGrid.heatData.temperature, nodeGrid.numNodes * sizeof(double));
        //Save current temperatures
        std::memcpy(nodeGrid.heatData.oldTemperature, nodeGrid.heatData.temperature, nodeGrid.numNodes * sizeof(double));

        //initialize vector C
        #pragma omp parallel for if(_parameters.enableOMP)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
        {
            double nodeH = getNodeH_fromTimeSteps(nodeIdx, timeStepHeat, timeStepWater);
            double avgH = computeMean(nodeGrid.waterData.oldPressureHeads[nodeIdx], nodeH, meanType_t::Arithmetic);
            vectorC.values[nodeIdx] = computeNodeHeatCapacity(nodeIdx, avgH, nodeGrid.heatData.temperature[nodeIdx]) * nodeGrid.size[nodeIdx];
        }

        //Compute linear system elements
        hostReset(nodeGrid.waterData.invariantFluxes, nodeGrid.numNodes);

        #pragma omp parallel for if(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.numNodes; ++rowIdx)
        {
            double nodeT = nodeGrid.heatData.temperature[rowIdx];
            double nodeH = getNodeH_fromTimeSteps(rowIdx, timeStepHeat, timeStepWater);

            double dTheta = computeNodeTheta_fromSignedPsi(rowIdx, nodeH - nodeGrid.z[rowIdx]) -
                            computeNodeTheta_fromSignedPsi(rowIdx, nodeGrid.waterData.oldPressureHeads[rowIdx] - nodeGrid.z[rowIdx]);

            double heatCapacity = dTheta * HEAT_CAPACITY_WATER * nodeT;

            if(simulationFlags.computeHeatVapor)
            {
                double dThetaV = computeNodeVaporThetaV(rowIdx, nodeH - nodeGrid.z[rowIdx], nodeT) -
                                 computeNodeVaporThetaV(rowIdx, nodeH - nodeGrid.z[rowIdx], nodeGrid.heatData.oldTemperature[rowIdx]);

                heatCapacity += dThetaV * HEAT_CAPACITY_AIR * nodeT;
                heatCapacity += dThetaV * computeLatentVaporizationHeat(nodeT - ZEROCELSIUS) * WATER_DENSITY;
            }

            heatCapacity *= nodeGrid.size[rowIdx];

            //Create matrix elements
            uint8_t linkIdx = 1;
            bool isLinked = false;

            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 0, timeStepHeat, timeStepWater);
            if(isLinked)
                linkIdx++;

            //Compute flox down
            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 1, timeStepHeat, timeStepWater);
            if(isLinked)
                linkIdx++;

            //Compute flux lateral
            for(uint8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
            {
                isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.colIndeces[rowIdx][linkIdx], rowIdx, 2 + latIdx, timeStepHeat, timeStepWater);
                if(isLinked)
                    linkIdx++;
            }

            matrixA.numColumns[rowIdx] = linkIdx; //TO DO: need to fill the not used columns of the row?

            //Compute diagonal element
            double sumDP = 0., sumF0 = 0.;
            for(uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
            {
                sumDP += matrixA.values[rowIdx][colIdx] * _parameters.heatWeightFactor;
                double dT0 = nodeGrid.heatData.oldTemperature[matrixA.colIndeces[rowIdx][colIdx]] - nodeGrid.heatData.oldTemperature[rowIdx];
                sumF0 += matrixA.values[rowIdx][colIdx] * (1. - _parameters.heatWeightFactor) * dT0;
                matrixA.values[rowIdx][colIdx] *= -(_parameters.heatWeightFactor);
            }
            matrixA.colIndeces[rowIdx][0] = rowIdx;
            matrixA.values[rowIdx][0] = sumDP + (vectorC.values[rowIdx] / timeStepHeat);  //Check if C values is correct

            //Computeb element
            vectorB.values[rowIdx] = (vectorC.values[rowIdx] / timeStepHeat) * nodeGrid.heatData.oldTemperature[rowIdx] - heatCapacity / timeStepHeat +
                                        nodeGrid.heatData.heatFlux[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx] + sumF0;

            //Preconditioning
            if(matrixA.values[rowIdx][0] > 0)
            {
                vectorB.values[rowIdx] /= matrixA.values[rowIdx][0];

                for(uint8_t colIdx = 1; colIdx < matrixA.numColumns[rowIdx]; ++colIdx)
                    matrixA.values[rowIdx][colIdx] /= matrixA.values[rowIdx][0];
            }
        }

        //Solve linear system
        solveLinearSystem(_parameters.maxApproximationsNumber - 1, processType::Heat);

        //Retrive new temperatures
        std::memcpy(nodeGrid.heatData.temperature, vectorX.values, nodeGrid.numNodes * sizeof(double));

        //Compute balance
        evaluateHeatBalance(timeStepHeat, timeStepWater);

        //Update balance
        updateHeatBalanceData();

        //Update heat fluxes
        saveHeatFluxValues(timeStepHeat, timeStepWater);

        //Save new temperatures
        std::memcpy(nodeGrid.heatData.oldTemperature, nodeGrid.heatData.temperature, nodeGrid.numNodes * sizeof(double));

        return;
    }


    bool CPUSolver::solveLinearSystem(uint8_t approximationNumber, processType computationType)
    {
        double currErrorNorm = 0., bestErrorNorm = static_cast<double>(std::numeric_limits<float>::max());

        uint32_t currMaxIterationNum = calcCurrentMaxIterationNumber(approximationNumber);

        for(uint32_t iterationNumber = 0; iterationNumber < currMaxIterationNum; ++iterationNumber)
        {
            switch(computationType)
            {
                case processType::Water:
                    currErrorNorm = JacobiWaterCPU(vectorX, matrixA, vectorB);
                    break;
                case processType::Heat:
                    currErrorNorm = GaussSeidelHeatCPU(vectorX, matrixA, vectorB);
                    break;
                default:
                    throw std::runtime_error("Process not available");
            }

            if(currErrorNorm < _parameters.residualTolerance)
                break;

            if(currErrorNorm > (bestErrorNorm * 10))
                return false;

            if(currErrorNorm < bestErrorNorm)
                bestErrorNorm = currErrorNorm;
        }

        return true;
    }

} // namespace
