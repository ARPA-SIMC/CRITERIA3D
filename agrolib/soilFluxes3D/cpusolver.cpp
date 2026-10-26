#include <cstring>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "soilFluxes3D.h"
#include "cpusolver.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"
#include "linealia.hpp"

using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Water;
using namespace soilFluxes3D::v2::Heat;
using namespace soilFluxes3D::v2::Math;

namespace soilFluxes3D::v2
{
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;
    extern __cudaMngd balanceData_t balanceDataCurrentTimeStep;

    SF3Derror_t CPUSolver::initialize()
    {
        if(_status != solverStatus::Created)
            return SF3Derror_t::SolverError;

        if(_parameters.deltaTcurr == noDataD)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;
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

        vectorNewX.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorNewX.values, vectorNewX.numElements);

        vectorB.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorB.values, vectorB.numElements);

        vectorC.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorC.values, vectorC.numElements);

        vectorDiagOriginal.numElements = nodeGrid.nrNodes;
        hostSolverAlloc(vectorDiagOriginal.values, vectorDiagOriginal.numElements);

        _status = solverStatus::initialized;
        return SF3Derror_t::SF3Dok;
    }


    SF3Derror_t CPUSolver::run(double maxTimeStep, double& acceptedTimeStep, processType process, double heatStartTime)
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
            {
                const double dtWater = acceptedTimeStep;
                double dtHeat = maxTimeStep;
                double sumHeatTime = 0.;
                while ((maxTimeStep - sumHeatTime) > 1e-9)
                {
                    dtHeat = SF3Dmin(dtHeat, maxTimeStep - sumHeatTime);

                    if (! heatLoop(dtHeat, dtWater, heatStartTime + sumHeatTime))
                    {
                        // step already at the minimum: halving again would never end
                        if (dtHeat <= _parameters.deltaTmin)
                            return SF3Derror_t::SolverError;

                        dtHeat = SF3Dmax(dtHeat * 0.5, _parameters.deltaTmin);
                    }
                    else
                        sumHeatTime += dtHeat;
                }
                break;
            }
            case processType::Solutes:
                throw std::runtime_error("Solutes not available");
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

        // Destruct matrix variable
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            hostSolverFree(matrixA.columnIndeces[rowIdx]);
            hostSolverFree(matrixA.values[rowIdx]);
        }
        hostSolverFree(matrixA.numColsInRow);
        hostSolverFree(matrixA.columnIndeces);
        hostSolverFree(matrixA.values);

        hostSolverFree(vectorDiagOriginal.values);

        hostSolverFree(vectorX.values);
        hostSolverFree(vectorNewX.values);

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

        // reset invariant fluxes
        std::memset(nodeGrid.waterData.invariantFluxes, 0, nodeGrid.nrNodes * sizeof(double));

        // surface capacity = cell size
        std::memcpy(vectorC.values, nodeGrid.size, nodeGrid.nrSurfaceNodes * sizeof(double));

        while(stepStatus != balanceResult_t::stepAccepted)
        {
            acceptedTimeStep = SF3Dmin(_parameters.deltaTcurr, maxTimeStep);

            // save current potential
            std::memcpy(nodeGrid.waterData.oldPressureHead, nodeGrid.waterData.pressureHead, nodeGrid.nrNodes * sizeof(double));

            // initialize the solution vector with the current pressure head
            assert(vectorX.numElements == nodeGrid.nrNodes);
            std::memcpy(vectorX.values, nodeGrid.waterData.pressureHead, vectorX.numElements * sizeof(double));

            // compute subsurface degree of saturation
            __parfor(_parameters.enableOMP)
            for (SF3Duint_t nodeIdx = nodeGrid.nrSurfaceNodes; nodeIdx < nodeGrid.nrNodes; ++nodeIdx)
            {
                nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
            }

            // main computation
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            // error in solver
            if (stepStatus == balanceResult_t::stepNan)
                return false;

            if(stepStatus != balanceResult_t::stepAccepted)
            {
                // restore old pressureHead
                std::memcpy(nodeGrid.waterData.pressureHead, nodeGrid.waterData.oldPressureHead, nodeGrid.nrNodes * sizeof(double));
            }
        }

        return true;
    }


    // NOT USED sostituito da taglio diretto nella soluzione del sistema lineare
    bool CPUSolver::checkSurfaceElements(double deltaT)
    {
        //__parforop(_parameters.enableOMP, max, courantMax)
        for(SF3Duint_t index = 0; index <  nodeGrid.nrSurfaceNodes; ++index)
        {
            double x = vectorB.values[index];
            for(u8_t j = 1; j < matrixA.numColsInRow[index]; ++j)
            {
                const double value = matrixA.values[index][j];
                if (value != 0.)
                    x -= value * vectorX.values[matrixA.columnIndeces[index][j]];
            }
            x /= matrixA.values[index][0];

            const double h = x - nodeGrid.z[index];
            // avoid negative potentials
            if(h < -0.0001)
            {
                const double h0 = vectorX.values[index] - nodeGrid.z[index];
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
        // search the maximum Courant number
        double courantMax = 0;
        __parforop(_parameters.enableOMP, max, courantMax)
        for(SF3Duint_t i = 0; i < nodeGrid.nrSurfaceNodes; ++i)
            courantMax = SF3Dmax(courantMax, nodeGrid.waterData.partialCourantWater[i]);

        nodeGrid.CourantWater = courantMax;

        // check Courant condition
        if(nodeGrid.CourantWater < 1.01 || deltaT <= _parameters.deltaTmin)
            return true;

        // condition is failed: update deltaT based on the Courant number
        {
            _parameters.deltaTcurr /= nodeGrid.CourantWater;

            // guard against non finite/zero values (the rescaling loop below would never end)
            if (!std::isfinite(_parameters.deltaTcurr) || _parameters.deltaTcurr <= 0.)
                _parameters.deltaTcurr = _parameters.deltaTmin;

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
            const double invDiag = 1.0 / rowValues[0];

            // normalize off-diagonal elements
            for(u8_t col = 1; col < nrCols; ++col)
            {
                rowValues[col] *= invDiag;
            }

            // normalize RHS
            vectorB.values[row] *= invDiag;

            // store the diagonal value
            vectorDiagOriginal.values[row] = rowValues[0];

            // set diagonal to 1
            rowValues[0] = 1.0;
        }
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
        if (computeLinkFluxes(matrixA.values[row][col], matrixA.columnIndeces[row][col], row,
                              linkIndex, approxNum, deltaT, _parameters.lateralVerticalRatio,
                              linkType_t::Up, _parameters.meanType) )
            col++;

        // flux lateral
        for(u8_t l = 0; l < nodeGrid.numLateralLink[row]; ++l)
        {
            linkIndex = l + 2;
            if (computeLinkFluxes(matrixA.values[row][col], matrixA.columnIndeces[row][col], row,
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
                // surface elements
                __parfor(_parameters.enableOMP)
                for (SF3Duint_t row = 0; row < nodeGrid.nrSurfaceNodes; ++row)
                {
                    computeLinearSystemElement(row, approxIdx, deltaT);
                }

                if (! checkCourant(deltaT))
                {
                    // Courant condition is failed -> reduces time step
                    return balanceResult_t::stepHalved;
                }

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
            for (SF3Duint_t nodeIdx = nodeGrid.nrSurfaceNodes; nodeIdx < nodeGrid.nrNodes; ++nodeIdx)
                if(! nodeGrid.surfaceFlag[nodeIdx])
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);

            // check water balance
            balanceResult = evaluateWaterBalance(approxIdx, _bestMBRerror, deltaT, _parameters);

            if((balanceResult == balanceResult_t::stepAccepted) || (balanceResult == balanceResult_t::stepHalved)
                || (balanceResult == balanceResult_t::stepNan))
                return balanceResult;
        }

        return balanceResult;
    }


    bool CPUSolver::heatLoop(double timeStepHeat, double timeStepWater, double heatStartTime)
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

            double avgH = getNodeAvgH_fromTimeSteps(nodeIdx, timeStepHeat, timeStepWater, heatStartTime);
            vectorC.values[nodeIdx] = computeNodeHeatCapacity(nodeIdx, avgH, nodeGrid.heatData.temperature[nodeIdx]) * nodeGrid.size[nodeIdx];
        }

        //Compute linear system elements
        hostReset(nodeGrid.waterData.invariantFluxes, nodeGrid.nrNodes);

        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
        {
            if(nodeGrid.surfaceFlag[rowIdx])
            {
                // Identity: 1.0 * T_new = T_old
                matrixA.numColsInRow[rowIdx] = 1;
                matrixA.columnIndeces[rowIdx][0] = rowIdx;

                matrixA.values[rowIdx][0] = 1.0;
                vectorDiagOriginal.values[rowIdx] = 1.0;

                // boundary temperature
                vectorB.values[rowIdx] = nodeGrid.heatData.oldTemperature[rowIdx];

                continue;
            }

            double nodeT = nodeGrid.heatData.temperature[rowIdx];

            const double hStart = getNodeH_fromTimeSteps(rowIdx, 0., timeStepWater, heatStartTime) - nodeGrid.z[rowIdx];
            const double hEnd   = getNodeH_fromTimeSteps(rowIdx, timeStepHeat, timeStepWater, heatStartTime) - nodeGrid.z[rowIdx];

            double dTheta = computeNodeTheta_fromSignedPsi(rowIdx, hEnd) - computeNodeTheta_fromSignedPsi(rowIdx, hStart);

            // Heat stored/released by the change of liquid water content: it is consistent only if the heat carried by the
            // liquid water flux is modelled (advective flux). Without advection the water content change must not exchange heat,
            // otherwise every wetting removes ~ Cw * T * dTheta (T in K) from the soil (spurious cooling of tens of degrees).
            double heatCapacity = 0.;
            if(simulationFlags.computeWater && simulationFlags.computeHeatAdvection)
                heatCapacity = dTheta * HEAT_CAPACITY_WATER * nodeT;

            if(simulationFlags.computeWater && simulationFlags.computeHeatVapor)
            {
                double dThetaV = computeNodeVaporThetaV(rowIdx, hEnd, nodeT) -
                                 computeNodeVaporThetaV(rowIdx, hStart, nodeGrid.heatData.oldTemperature[rowIdx]);

                heatCapacity += dThetaV * WATER_DENSITY * HEAT_CAPACITY_WATER_VAPOR * nodeT;
                heatCapacity += dThetaV * WATER_DENSITY * computeLatentVaporizationHeat(nodeT - ZEROCELSIUS);
            }

            heatCapacity *= nodeGrid.size[rowIdx];

            //Create matrix elements
            u8_t linkIdx = 1;
            bool isLinked = false;

            // Compute flux up
            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx], rowIdx,
                                             0, timeStepHeat, timeStepWater, heatStartTime);
            if(isLinked)
                linkIdx++;

            // Compute flux down
            isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx], rowIdx,
                                             1, timeStepHeat, timeStepWater, heatStartTime);
            if(isLinked)
                linkIdx++;

            // Compute flux lateral
            for(u8_t latIdx = 0; latIdx < maxLateralLink; ++latIdx)
            {
                isLinked = computeHeatLinkFluxes(matrixA.values[rowIdx][linkIdx], matrixA.columnIndeces[rowIdx][linkIdx],
                                                 rowIdx, 2 + latIdx, timeStepHeat, timeStepWater, heatStartTime);
                if(isLinked)
                    linkIdx++;
            }

            matrixA.numColsInRow[rowIdx] = linkIdx;

            // Compute diagonal elements
            double sumDP = 0., sumF0 = 0.;
            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
            {
                sumDP += matrixA.values[rowIdx][colIdx] * _parameters.heatWeightFactor;
                double dT0 = nodeGrid.heatData.oldTemperature[matrixA.columnIndeces[rowIdx][colIdx]] - nodeGrid.heatData.oldTemperature[rowIdx];
                sumF0 += matrixA.values[rowIdx][colIdx] * (1. - _parameters.heatWeightFactor) * dT0;
                matrixA.values[rowIdx][colIdx] *= -(_parameters.heatWeightFactor);
            }

            matrixA.columnIndeces[rowIdx][0] = rowIdx;
            matrixA.values[rowIdx][0] = sumDP + (vectorC.values[rowIdx] / timeStepHeat);

            // New code: implicit advection
            // positive = Inflow (enter in rowIdx), negative = Outflow
            if (simulationFlags.computeWater && simulationFlags.computeHeatAdvection)
            {
                for(u8_t lIdx = 0; lIdx < maxTotalLink; ++lIdx)
                {
                    if(nodeGrid.linkData[lIdx].linkType[rowIdx] == linkType_t::NoLink)
                        continue;

                    const SF3Duint_t linkedNode = nodeGrid.linkData[lIdx].linkIndex[rowIdx];
                    if(! isHeatNode(linkedNode))
                        continue;

                    // liquid water
                    const double qLiquid = nodeGrid.linkData[lIdx].waterFlux[rowIdx];
                    if (qLiquid < 0.)
                    {
                        // Outflow: il calore esce. Sommiamo alla diagonale di rowIdx (LHS)
                        matrixA.values[rowIdx][0] += std::fabs(qLiquid) * HEAT_CAPACITY_WATER;
                    }
                    else if (qLiquid > 0.)
                    {
                        // Inflow: il calore entra. Sottraiamo dall'extra-diagonale del nodo sorgente
                        double inflowC = qLiquid * HEAT_CAPACITY_WATER;
                        for(u8_t c = 1; c < matrixA.numColsInRow[rowIdx]; ++c)
                        {
                            if(matrixA.columnIndeces[rowIdx][c] == linkedNode)
                            {
                                matrixA.values[rowIdx][c] -= inflowC;
                                break;
                            }
                        }
                    }

                    // vapor
                    const double qVapor  = nodeGrid.linkData[lIdx].vaporFlux[rowIdx];
                    if (qVapor < 0.)
                    {
                        matrixA.values[rowIdx][0] += std::fabs(qVapor) * HEAT_CAPACITY_WATER_VAPOR;
                    }
                    else if (qVapor > 0.)
                    {
                        const double inflowC = qVapor * HEAT_CAPACITY_WATER_VAPOR;
                        for(u8_t c = 1; c < matrixA.numColsInRow[rowIdx]; ++c)
                        {
                            if(matrixA.columnIndeces[rowIdx][c] == linkedNode)
                            {
                                matrixA.values[rowIdx][c] -= inflowC;
                                break;
                            }
                        }
                    }
                }
            }

            // store the diagonal value
            vectorDiagOriginal.values[rowIdx] = matrixA.values[rowIdx][0];

            // Compute b elements
            vectorB.values[rowIdx] = vectorC.values[rowIdx] * nodeGrid.heatData.oldTemperature[rowIdx] / timeStepHeat - heatCapacity / timeStepHeat +
                                        nodeGrid.heatData.heatFlux[rowIdx] + nodeGrid.waterData.invariantFluxes[rowIdx] + sumF0;

            // Preconditioning
            {
            assert(matrixA.values[rowIdx][0] > 0);

            vectorB.values[rowIdx] /= matrixA.values[rowIdx][0];

            for(u8_t colIdx = 1; colIdx < matrixA.numColsInRow[rowIdx]; ++colIdx)
                matrixA.values[rowIdx][colIdx] /= matrixA.values[rowIdx][0];

            matrixA.values[rowIdx][0] = 1.0;
            }
        }

        // Solve linear system
        bool isStepValid = solveLinearSystem(_parameters.maxApproximationsNumber - 1, processType::Heat);

        if (!isStepValid)
        {
            // restore old temperatures
            __parfor(_parameters.enableOMP)
            for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
                if(! nodeGrid.surfaceFlag[rowIdx])
                    nodeGrid.heatData.temperature[rowIdx] = nodeGrid.heatData.oldTemperature[rowIdx];

            return false;
        }

        // store new temperatures
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
            if(! nodeGrid.surfaceFlag[rowIdx])
                nodeGrid.heatData.temperature[rowIdx] = vectorX.values[rowIdx];

        // compute balance
        evaluateHeatBalance(timeStepHeat, timeStepWater, heatStartTime);

        // check balance
        double currentMBR = balanceDataCurrentTimeStep.heatMBR;
        if ((!std::isfinite(currentMBR) || std::fabs(currentMBR) > 0.1))
        {
            if (timeStepHeat > _parameters.deltaTmin)
            {
                // restore old temperatures
                __parfor(_parameters.enableOMP)
                for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
                    if(! nodeGrid.surfaceFlag[rowIdx])
                        nodeGrid.heatData.temperature[rowIdx] = nodeGrid.heatData.oldTemperature[rowIdx];

                return false;
            }
            /*
            else
            {
                throw std::runtime_error("Instability: NaN or MBR > 1");
            }
            */
        }

        // Update balance
        updateHeatBalanceData();

        // Update heat fluxes
        saveHeatFluxValues(timeStepHeat, timeStepWater, heatStartTime);

        // Save new temperatures
        __parfor(_parameters.enableOMP)
        for (SF3Duint_t rowIdx = 0; rowIdx < nodeGrid.nrNodes; ++rowIdx)
            if(! nodeGrid.surfaceFlag[rowIdx])
                nodeGrid.heatData.oldTemperature[rowIdx] = nodeGrid.heatData.temperature[rowIdx];

        return true;
    }


    bool CPUSolver::linealSolver(u8_t approximationNr)
    {
        u32_t nrIterationMax = calcCurrentMaxIterationNumber(approximationNr);

        LinealExecutionParams executionParams;
        executionParams.log = 0;

        LinealiaIterativeSolverParams iterativeParams;
        iterativeParams.max_iterations = nrIterationMax;
        iterativeParams.max_relative_residual_norm = _parameters.residualTolerance;

        LinealiaRelaxedParams relaxParams;
        // over-relaxation
        //relaxParams.relax = 1.2;

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

        LinealiaIterativeResult result;

        switch (linealMethod) {
        case 0:
            result = LinealiaLib::instance().solveSSOR(A, x, b, executionParams, iterativeParams, relaxParams);
            break;
        case 1:
            result = LinealiaLib::instance().solveCG(A, x, b, executionParams, iterativeParams);
            break;
        case 2:
            result = LinealiaLib::instance().solvePCG_SOR(A, x, b, executionParams, iterativeParams, relPcgParams);
            break;
        case 3:
            result = LinealiaLib::instance().solvePCG_AMG_SOR(A, x, b, executionParams, iterativeParams, pcgAmgParams);
            break;
        default:
            result = LinealiaLib::instance().solveSSOR(A, x, b, executionParams, iterativeParams, relaxParams);
            break;
        }

        // Check surface water level (it must be ≥ 0)
        #pragma omp parallel for if(_parameters.enableOMP) schedule(static)
        for (SF3Duint_t row = 0; row < nodeGrid.nrSurfaceNodes; ++row)
        {
            double elevation = nodeGrid.z[row];
            double waterLevel = x.values[row] - elevation;
            if (waterLevel < 0.)
                x.values[row] = elevation;
        }

        return true;
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
                    currErrorNorm = JacobiWaterCPU(vectorX, vectorNewX, matrixA, vectorB);
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

} // end namespace
