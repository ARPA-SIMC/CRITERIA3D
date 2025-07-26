#include <cstring>
#include <cassert>
#include <iostream>

#include "cpusolver.h"
#include "soil_new.h"
#include "water_new.h"

using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Water;

namespace soilFluxes3D::New
{
    extern nodesData_t nodeGrid;
    extern simulationFlags_t simulationFlags;

    SF3Derror_t CPUSolver::inizialize()
    {
        if(_parameters.deltaTcurr == noData)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.enableOMP = true;       //TO DO: (nodeGrid.numNodes > ...);

        //Inizialize matrix structure
        matrixA.numRows = nodeGrid.numNodes;
        hostAlloc(matrixA.numColumns, uint8_t, matrixA.numRows);
        hostAlloc(matrixA.colIndeces, uint64_t*, matrixA.numRows);
        hostAlloc(matrixA.values, double*, matrixA.numRows);

        #pragma omp parallel for if(_parameters.enableOMP)
        for (uint64_t rowIdx = 0; rowIdx < matrixA.numRows; ++rowIdx)
        {
            matrixA.colIndeces[rowIdx] = static_cast<uint64_t*>(calloc(matrixA.maxColumns, sizeof(uint64_t)));
            matrixA.values[rowIdx] = static_cast<double*>(calloc(matrixA.maxColumns, sizeof(double)));
        }

        vectorX.numElements = nodeGrid.numNodes;
        hostAlloc(vectorX.values, double, vectorX.numElements);

        vectorB.numElements = nodeGrid.numNodes;
        hostAlloc(vectorB.values, double, vectorB.numElements);

        vectorC.numElements = nodeGrid.numNodes;
        hostAlloc(vectorC.values, double, vectorC.numElements);

        _status = Inizialized;
        return SF3Dok;
    }

    SF3Derror_t CPUSolver::run(double maxTimeStep, double& acceptedTimeStep, processType process)
    {
        if(_status != Inizialized)
            return SolverError;

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
        return SF3Dok;
    }

    void CPUSolver::waterMainLoop(double maxTimeStep, double &acceptedTimeStep)
    {
        balanceResult_t stepStatus = stepRefused;
        while(stepStatus != stepAccepted)
        {
            acceptedTimeStep = std::min(_parameters.deltaTcurr, maxTimeStep);

            //Save instantaneus H values
            std::memcpy(nodeGrid.waterData.oldPressureHeads, nodeGrid.waterData.pressureHead, nodeGrid.numNodes * sizeof(double));

            //Inizialize the solution vector with the current pressure head
            assert(vectorX.numElements == nodeGrid.numNodes);
            std::memcpy(vectorX.values, nodeGrid.waterData.pressureHead, vectorX.numElements * sizeof(double));

            //Assign vectorC surface values and compute subsurface saturation degree
            #pragma omp parallel for if(_parameters.enableOMP)
            for (uint64_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
            {
                if(nodeGrid.surfaceFlag[nodeIdx])
                    vectorC.values[nodeIdx] = nodeGrid.size[nodeIdx];
                else
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);
            }

            //Update aereodynamic and soil conductance
            //updateConductance();      //TO DO (Heat)

            //Update boundary
            updateBoundaryWaterData(acceptedTimeStep);

            //Effective computation step
            stepStatus = waterApproximationLoop(acceptedTimeStep);

            if(stepStatus != stepAccepted)
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

            //Update boundary water
            updateBoundaryWaterData(deltaT);

            //Update Courant data
            nodeGrid.waterData.CourantWaterLevel = 0.;

            //Compute linear system elements
            computeLinearSystemElement(matrixA, vectorB, vectorC, approxIdx, deltaT, _parameters.lateralVerticalRatio, _parameters.meantype);
            //Check Courant
            if((nodeGrid.waterData.CourantWaterLevel > 1.) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = std::max(_parameters.deltaTmin, _parameters.deltaTcurr / nodeGrid.waterData.CourantWaterLevel);
                if(_parameters.deltaTcurr > 1.)
                    _parameters.deltaTcurr = floor(_parameters.deltaTcurr);

                return stepHalved;
            }

            //Try solve linear system
            bool isStepValid = solveLinearSystem(approxIdx, Water);

            //Reduce step tipe if system resolution failed
            if((!isStepValid) && (deltaT > _parameters.deltaTmin))
            {
                _parameters.deltaTcurr = std::max(_parameters.deltaTmin, _parameters.deltaTcurr / 2.);
                return stepHalved;
            }

            //Update potential
            assert(vectorX.numElements == nodeGrid.numNodes);
            std::memcpy(nodeGrid.waterData.pressureHead, vectorX.values, vectorX.numElements * sizeof(double));

            //Update degree of saturation
            #pragma omp parallel for if(_parameters.enableOMP)
            for (uint64_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
                if(!nodeGrid.surfaceFlag[nodeIdx])
                    nodeGrid.waterData.saturationDegree[nodeIdx] = computeNodeSe(nodeIdx);

            //Check water balance
            balanceResult = evaluateWaterBalance(approxIdx, bestMBRerror, _parameters);

            if((balanceResult == stepAccepted) || (balanceResult == stepHalved))
                return balanceResult;
        }
        //TO DO: log functions

        return balanceResult;
    }

    bool CPUSolver::solveLinearSystem(uint8_t approximationNumber, processType computationType)
    {
        double currErrorNorm = 0., bestErrorNorm = (double) std::numeric_limits<float>::max();

        uint32_t currMaxIterationNum = calcCurrentMaxIterationNumber(approximationNumber);

        for(uint32_t iterationNumber = 0; iterationNumber < currMaxIterationNum; ++iterationNumber)
        {
            if(computationType == Water && _method == Jacobi)
                currErrorNorm = JacobiWaterCPU(vectorX, matrixA, vectorB);
            else if(computationType == Water && _method == GaussSeidel)
                currErrorNorm = 0.; //GaussSeidelWaterCPU();
            else
                std::exit(EXIT_FAILURE);

            if(currErrorNorm < _parameters.residualTolerance)
                break;

            if(currErrorNorm > (bestErrorNorm * 10))
                return false;

            if(currErrorNorm < bestErrorNorm)
                bestErrorNorm = currErrorNorm;
        }

        //TO DO: Log fuction
        return true;
    }

} // namespace
