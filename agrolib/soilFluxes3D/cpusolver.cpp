#include <cstring>
#include <cassert>
#include <omp.h>
#include <iostream>

#include "soilFluxes3D.h"
#include "cpusolver.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"
#include "otherFunctions.h"
#include "linealia.hpp"

using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Water;
using namespace soilFluxes3D::v2::Heat;
using namespace soilFluxes3D::v2::Math;

namespace soilFluxes3D::v2
{
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;

    SF3Derror_t CPUSolver::initialize()
    {
        if(_status != solverStatus::Created)
            return SF3Derror_t::SolverError;

        if(_parameters.deltaTcurr == noDataD)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;       //TO DO: (nodeGrid.nrNodes > ...);
        if(_parameters.enableOMP)
            omp_set_num_threads(static_cast<int>(_parameters.numThreads));

        //initialize matrix structure
        matrixA.numRows = nodeGrid.nrNodes;
        hostSolverAlloc(matrixA.numColsInRow, matrixA.numRows);
        hostSolverAlloc(matrixA.columnIndeces, matrixA.numRows);
        hostSolverAlloc(matrixA.values, matrixA.numRows);

        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            hostSolverAlloc(matrixA.columnIndeces[rowIdx], matrixA.maxColumns);
            hostSolverAlloc(matrixA.values[rowIdx], matrixA.maxColumns);
        }

        //initialize vector data
        vectorX.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorX.values, vectorX.numElements);

        vectorB.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorB.values, vectorB.numElements);

        vectorC.numElements = nodeGrid.nrNodes;
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
                if (! waterMainLoop(maxTimeStep, acceptedTimeStep))
                    return SF3Derror_t::SolverError;
                break;
            case processType::Heat:
                heatLoop(maxTimeStep, acceptedTimeStep);
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
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            hostSolverFree(matrixA.columnIndeces[rowIdx]);
            hostSolverFree(matrixA.values[rowIdx]);
        }
        hostSolverFree(matrixA.numColsInRow);
        hostSolverFree(matrixA.columnIndeces);
        hostSolverFree(matrixA.values);

        //Destruct matrix variable
        hostSolverFree(vectorX.values);

        hostSolverFree(vectorB.values);

        hostSolverFree(vectorC.values);

        _status = solverStatus::Created;
        return SF3Derror_t::SF3Dok;
    }

    void CPUSolver::setThreads()
    {
        if(_parameters.enableOMP)
            omp_set_num_threads(static_cast<int>(_parameters.numThreads));
    }


    bool CPUSolver::waterMainLoop(double maxTimeStep, double &acceptedTimeStep)
    {
        balanceResult_t stepStatus = balanceResult_t::stepRefused;

        while(stepStatus != balanceResult_t::stepAccepted)
        {
            acceptedTimeStep = SF3Dmin(_parameters.deltaTcurr, maxTimeStep);

            //Save instantaneus H values
            std::memcpy(nodeGrid.waterData.oldPressureHead, nodeGrid.waterData.pressureHead, nodeGrid.nrNodes * sizeof(double));

            //initialize the solution vector with the current pressure head
            assert(vectorX.numElements == nodeGrid.nrNodes);
            std::memcpy(vectorX.values, nodeGrid.waterData.pressureHead, vectorX.numElements * sizeof(double));

            //Assign vectorC surface values and compute subsurface saturation degree
            __parfor(_parameters.enableOMP)
            for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.nrNodes; ++nodeIdx)
            {
                if(nodeGrid.surfaceFlag[nodeIdx])
                    vectorC.values[nodeIdx] = nodeGrid.size[nodeIdx];
                else
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
            }

            // update aereodynamic and soil conductance
            updateConductance();

            // update boundary water
            updateBoundaryWaterData(acceptedTimeStep);

            // main computation
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            if(stepStatus != balanceResult_t::stepAccepted)
            {
                // restore old pressureHead
                std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHead, nodeGrid.nrNodes * sizeof(double));
            }

            if (stepStatus == balanceResult_t::stepNan)
                return false;
        }

        return true;
    }


    bool CPUSolver::checkSurfaceElements(double deltaT)
    {
        //__parforop(_parameters.enableOMP, max, courantMax)
        for(SF3Duint_t index = 0; index <  nodeGrid.nrSurfaceNodes; ++index)
        {
            double x = vectorB.values[index];
            for(u8_t j = 1; j < matrixA.numColsInRow[index]; ++j)
            {
                double value = matrixA.values[index][j];
                if (value != 0.)
                    x -= value * vectorX.values[matrixA.columnIndeces[index][j]];
            }
            x /= matrixA.values[index][0];

            double h = x - nodeGrid.z[index];
            // avoid negative potentials
            if(h < -0.0001)
            {
                double h0 = vectorX.values[index] - nodeGrid.z[index];
                double ratio = 0.;
                if (std::abs(h0) > EPSILON)
                    ratio = std::max(0., 1. - std::abs(h / h0));

                for(u8_t j = 1; j < matrixA.numColsInRow[index]; ++j)
                {
                    double value = matrixA.values[index][j];
                    if (value != 0.)
                    {
                        // reduces water sinks
                        SF3Duint_t linked = matrixA.columnIndeces[index][j];
                        if (vectorX.values[linked] < vectorX.values[index])
                        {
                            matrixA.values[index][j] = value * ratio;

                            // symmetric value in linked row
                            for (u8_t k = 1; k < matrixA.numColsInRow[linked]; ++k)
                            {
                                if (matrixA.columnIndeces[linked][k] == index)
                                {
                                    matrixA.values[linked][k] = matrixA.values[index][j];
                                    break;
                                }
                            }
                            computeDiagonalElement(linked, deltaT);
                        }
                    }
                }
                computeDiagonalElement(index, deltaT);
            }
        }

        return true;
    }


    bool CPUSolver::checkCourant(double deltaT)
    {
        // search maximum Courant number
        double courantMax = 0;
        __parforop(_parameters.enableOMP, max, courantMax)
        for(SF3Duint_t i = 0; i < nodeGrid.nrSurfaceNodes; ++i)
            courantMax = SF3Dmax(courantMax, nodeGrid.waterData.partialCourantWater[i]);

        // more speed, less accuracy
        if (_parameters.MBRThreshold > 0.01)
            courantMax *= 0.5;

        nodeGrid.CourantWater = courantMax;

        // check Courant condition
        if(nodeGrid.CourantWater < 1.01 || deltaT <= _parameters.deltaTmin)
        {
            return true;
        }

        // Courant condition failed: update deltaT
        {
            _parameters.deltaTcurr /= nodeGrid.CourantWater;

            int multiply = 0;
            while (_parameters.deltaTcurr < 1.)
            {
                _parameters.deltaTcurr *= 10.;
                ++multiply;
            }
            _parameters.deltaTcurr = std::floor(_parameters.deltaTcurr);

            for (int i = 0; i < multiply; i++)
                _parameters.deltaTcurr /= 10.;

            _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr);
        }

        return false;
    }


    void CPUSolver::preconditioningMatrix()
    {
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t row = 0; row < matrixA.numRows; ++row)
        {
            auto& rowValues = matrixA.values[row];
            const u8_t nrCols = matrixA.numColsInRow[row];
            const double invDiag = 1.0 / matrixA.values[row][0];

            // Normalize off-diagonal elements
            u8_t nrLinkedFluxes = 0;
            for(u8_t col = 1; col < nrCols; ++col)
            {
                if (rowValues[col] != 0.0)
                {
                    rowValues[col] *= invDiag;
                    ++nrLinkedFluxes;
                }
            }

            // normalize RHS
            vectorB.values[row] *= invDiag;

            // set diagonal to 1
            rowValues[0] = 1.0;

            // handle isolated node (only diagonal term)
            if (nrLinkedFluxes == 0)
            {
                vectorX.values[row] = vectorB.values[row];
            }
        }
    }


    bool CPUSolver::isLinked(bool& isPrevious, double& matrixElement, SF3Duint_t& matrixIndex,
                             SF3Duint_t nodeIndex, u8_t linkIndex)
    {
        if(nodeGrid.linkData[linkIndex].linkType[nodeIndex] == linkType_t::NoLink)
            return false;

        SF3Duint_t index = nodeGrid.linkData[linkIndex].linkIndex[nodeIndex];
        isPrevious = (index < nodeIndex);

        if (! isPrevious)
            return true;

        for (u8_t j = 1; j < matrixA.numColsInRow[index]; ++j)
        {
            if (matrixA.columnIndeces[index][j] == nodeIndex)
            {
                matrixElement = matrixA.values[index][j];
                matrixIndex = index;
                return true;
            }
        }

        return false;
    }


    void CPUSolver::computeDiagonalElement(SF3Duint_t row, double deltaT)
    {
        u8_t nrElements = matrixA.numColsInRow[row];
        auto& rowValues = matrixA.values[row];

        double sum = 0.;
        for (size_t col = 1; col < nrElements; ++col)
            sum += rowValues[col];

        rowValues[0] = (vectorC.values[row] / deltaT) + sum;
    }


    void CPUSolver::computeLinearSystemElement(SF3Duint_t row, u8_t approxNum, double deltaT)
    {
        u8_t col = 1;

        // flux up
        u8_t linkIndex = 0;
        if ( computeLinkFluxes(matrixA.values[row][col], matrixA.columnIndeces[row][col], row,
                              linkIndex, approxNum, deltaT, _parameters.lateralVerticalRatio,
                              linkType_t::Up, _parameters.meanType) )
            col++;

        // flux lateral
        for(u8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
        {
            linkIndex = 2 + latIdx;
            if ( computeLinkFluxes(matrixA.values[row][col], matrixA.columnIndeces[row][col], row,
                                linkIndex, approxNum, deltaT, _parameters.lateralVerticalRatio,
                                linkType_t::Lateral, _parameters.meanType) )
                col++;
        }

        // flux down
        linkIndex = 1;
        if (computeLinkFluxes(matrixA.values[row][col], matrixA.columnIndeces[row][col], row,
                                linkIndex, approxNum, deltaT, _parameters.lateralVerticalRatio,
                                linkType_t::Down, _parameters.meanType))
            col++;

        matrixA.numColsInRow[row] = col;

        // diagonal
        computeDiagonalElement(row, deltaT);
        for(u8_t col = 1; col < matrixA.numColsInRow[row]; ++col)
        {
            matrixA.values[row][col] *= -1.;
        }
        matrixA.columnIndeces[row][0] = row;

        // Compute b element
        vectorB.values[row] = ((vectorC.values[row] / deltaT) * nodeGrid.waterData.oldPressureHead[row])
                              + nodeGrid.waterData.waterFlow[row] + nodeGrid.waterData.invariantFluxes[row];
    }


    balanceResult_t CPUSolver::waterApproximationLoop(double deltaT)
    {
        balanceResult_t balanceResult = balanceResult_t::stepRefused;
        _bestMBRerror = noDataD;

        for(u8_t approxIdx = 0; approxIdx < _parameters.maxApproximationsNumber; ++approxIdx)
        {
            // compute capacity vector elements
            computeCapacity(vectorC);

            // update boundary water
            updateBoundaryWaterData(deltaT);

            // reset Courant data to zero
            nodeGrid.CourantWater = 0.;
            std::memset(nodeGrid.waterData.partialCourantWater, 0, nodeGrid.nrSurfaceNodes * sizeof(double));

            // compute linear system
            {
                // suface elements
                __parfor(_parameters.enableOMP)
                for (SF3Duint_t row = 0; row < nodeGrid.nrSurfaceNodes; ++row)
                {
                    computeLinearSystemElement(row, approxIdx, deltaT);
                }

                if (! checkCourant(deltaT))
                {
                    // Courant condition is failed: reduces time step
                    return balanceResult_t::stepHalved;
                }
                //checkSurfaceElements(deltaT);

                // soil elements
                __parfor(_parameters.enableOMP)
                for (SF3Duint_t row = nodeGrid.nrSurfaceNodes; row < matrixA.numRows; ++row)
                {
                    computeLinearSystemElement(row, approxIdx, deltaT);
                }
            }

            // preconditioning the matrix
            preconditioningMatrix();

            // solve linear system
            bool isStepValid;
            if (useLineal)
                isStepValid = linealSolver(approxIdx);
            else
                isStepValid = solveLinearSystem(approxIdx, processType::Water);

            // reduce time step if system resolution failed
            if((! isStepValid) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = SF3Dmax(_parameters.deltaTmin, _parameters.deltaTcurr / 2.);
                return balanceResult_t::stepHalved;
            }

            // update water potential
            assert(vectorX.numElements == nodeGrid.nrNodes);
            std::memcpy(nodeGrid.waterData.pressureHead, vectorX.values, vectorX.numElements * sizeof(double));

            // update degree of saturation
            __parfor(_parameters.enableOMP)
            for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.nrNodes; ++nodeIdx)
                if(!nodeGrid.surfaceFlag[nodeIdx])
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);

            // check water balance
            balanceResult = evaluateWaterBalance(approxIdx, _bestMBRerror, deltaT, _parameters);

            if((balanceResult == balanceResult_t::stepAccepted) || (balanceResult == balanceResult_t::stepHalved)
                || (balanceResult == balanceResult_t::stepNan))
                return balanceResult;
        }

        return balanceResult;
    }


    void CPUSolver::heatLoop(double timeStepHeat, double timeStepWater)
    {
        resetFluxValues(true, false);

        //initialize vector X
        std::memcpy(vectorX.values, nodeGrid.heatData.temperature, nodeGrid.nrNodes * sizeof(double));
        //Save current temperatures
        std::memcpy(nodeGrid.heatData.oldTemperature, nodeGrid.heatData.temperature, nodeGrid.nrNodes * sizeof(double));

        //initialize vector C
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.nrNodes; ++nodeIdx)
        {
            if(nodeGrid.surfaceFlag[nodeIdx])
                continue;

            double nodeH = getNodeH_fromTimeSteps(nodeIdx, timeStepHeat, timeStepWater);
            double avgH = computeMean(nodeGrid.waterData.oldPressureHead[nodeIdx], nodeH, meanType_t::Arithmetic) - nodeGrid.z[nodeIdx];
            vectorC.values[nodeIdx] = computeNodeHeatCapacity(nodeIdx, avgH, nodeGrid.heatData.temperature[nodeIdx]) * nodeGrid.size[nodeIdx];
        }

        //Compute linear system elements
        hostReset(nodeGrid.waterData.invariantFluxes, nodeGrid.nrNodes);

        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
        {
            if(nodeGrid.surfaceFlag[rowIdx])
                continue;

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

            //Create matrix elements
            u8_t linkIdx = 1;
            bool isLinked = false;

            //Compute flox up
            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx], rowIdx, 0, timeStepHeat, timeStepWater);
            if(isLinked)
                linkIdx++;

            //Compute flox down
            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx], rowIdx, 1, timeStepHeat, timeStepWater);
            if(isLinked)
                linkIdx++;

            //Compute flux lateral
            for(u8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx) //TO DO: implement real num lat links
            {
                isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx], rowIdx, 2 + latIdx, timeStepHeat, timeStepWater);
                if(isLinked)
                    linkIdx++;
            }

            matrixA.numColsInRow[rowIdx] = linkIdx; //TO DO: need to fill the not used columns of the row?

            //Compute diagonal element
            double sumDP = 0., sumF0 = 0.;
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
            {
                sumDP += matrixA.values[rowIdx][colIdx] * _parameters.heatWeightFactor;
                double dT0 = nodeGrid.heatData.oldTemperature[matrixA.columnIndeces[rowIdx][colIdx]] - nodeGrid.heatData.oldTemperature[rowIdx];
                sumF0 += matrixA.values[rowIdx][colIdx] * (1. - _parameters.heatWeightFactor) * dT0;
                matrixA.values[rowIdx][colIdx] *= -(_parameters.heatWeightFactor);
            }
            matrixA.columnIndeces[rowIdx][0] = rowIdx;
            matrixA.values[rowIdx][0] = sumDP + (vectorC.values[rowIdx] / timeStepHeat);  //Check if C values is correct

            //Compute b element
            vectorB.values[rowIdx] = vectorC.values[rowIdx] * nodeGrid.heatData.oldTemperature[rowIdx] / timeStepHeat - heatCapacity / timeStepHeat +
                                        nodeGrid.heatData.heatFlux[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx] + sumF0;

            //Preconditioning
            if(matrixA.values[rowIdx][0] > 0)
            {
                vectorB.values[rowIdx] /= matrixA.values[rowIdx][0];

                for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                    matrixA.values[rowIdx][colIdx] /= matrixA.values[rowIdx][0];
            }
        }

        //Solve linear system
        solveLinearSystem(_parameters.maxApproximationsNumber - 1, processType::Heat);

        //Retrive new temperatures
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
            if(!nodeGrid.surfaceFlag[rowIdx])
                nodeGrid.heatData.temperature[rowIdx] = vectorX.values[rowIdx];

        //Compute balance
        evaluateHeatBalance(timeStepHeat, timeStepWater);

        //Update balance
        updateHeatBalanceData();

        //Update heat fluxes
        saveHeatFluxValues(timeStepHeat, timeStepWater);

        //Save new temperatures
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
            if(!nodeGrid.surfaceFlag[rowIdx])
                nodeGrid.heatData.oldTemperature[rowIdx] = nodeGrid.heatData.temperature[rowIdx];
    }


    bool CPUSolver::linealSolver(u8_t approximationNr)
    {
        u32_t maxNrIteration = calcCurrentMaxIterationNumber(approximationNr);

        LinealExecutionParams executionParams;
        executionParams.log = 0;

        LinealiaIterativeSolverParams iterativeParams;
        iterativeParams.max_iterations = maxNrIteration;
        iterativeParams.max_relative_residual_norm = _parameters.residualTolerance;

        LinealiaRelaxedPreconditionerParams relPcgParams;
        LinealiaPcgAmgParams pcgAmgParams;

        LinealiaMatrix A;
        A.num_rows = matrixA.numRows;
        A.num_columns = matrixA.numColsInRow;
        A.column_indices = matrixA.columnIndeces;
        A.values = matrixA.values;

        LinealiaVector x, b;
        x.num_elements = vectorX.numElements;
        x.values = vectorX.values;
        b.num_elements = vectorB.numElements;
        b.values = vectorB.values;

        LinealiaLib::instance().solveCG(A, x, b, executionParams, iterativeParams);
        //LinealiaLib::instance().solvePCG_SOR(A, x, b, executionParams, iterativeParams, relPcgParams);
        //LinealiaLib::instance().solvePCG_AMG_SOR(A, x, b, executionParams, iterativeParams, pcgAmgParams);

        // check surface potential (must be >= 0)
        bool isStepValid = true;
        #pragma omp parallel for reduction(&&:isStepValid) if(_parameters.enableOMP)
        for (SF3Duint_t row = 0; row < nodeGrid.nrSurfaceNodes; ++row)
        {
            double& xVal = x.values[row];
            const double zVal = nodeGrid.z[row];
            const double diff = zVal - xVal;

            if (diff > 0.)
            {
                if (diff > 1e-3)
                    isStepValid = false;

                xVal = zVal;
            }
        }

        return isStepValid;
    }


    bool CPUSolver::solveLinearSystem(u8_t approximationNr, processType computationType)
    {
        double currErrorNorm = 0., bestErrorNorm = 1.;

        u32_t currMaxIterationNum = calcCurrentMaxIterationNumber(approximationNr);

        for(u32_t iterationNumber = 0; iterationNumber < currMaxIterationNum; ++iterationNumber)
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
