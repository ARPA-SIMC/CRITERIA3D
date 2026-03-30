#pragma once

#include "solver.h"
#include "types_cpu.h"
#include "linealiaLib.h"

namespace soilFluxes3D::v2
{
    class CPUSolver : public Solver
    {
        private:
            MatrixCPU matrixA;
            VectorCPU vectorB, vectorX;
            VectorCPU vectorC;

            bool waterMainLoop(double maxTimeStep, double& acceptedTimeStep);
            balanceResult_t waterApproximationLoop(double deltaT);

            bool isLinked(bool& isPrevious, double& matrixElement, SF3Duint_t &matrixIndex, SF3Duint_t nodeIndex, u8_t linkIndex);
            void computeLinearSystemElement(SF3Duint_t row, u8_t approxNum, double deltaT);
            void computeDiagonalElement(SF3Duint_t row, double deltaT);
            void preconditioningMatrix();

            bool checkSurfaceElements(double deltaT);
            bool checkCourant(double deltaT);

            void heatLoop(double timeStepHeat, double timeStepWater);

            bool solveLinearSystem(u8_t approximationNr, processType computationType) override;
            bool linealSolver(u8_t approximationNr);

        public:
            CPUSolver() : Solver(solverType::CPU, numericalMethod::Jacobi) {}

            __cudaSpec double getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept;

            SF3Derror_t initialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t clean() override;
            void setThreads();
    };

    inline __cudaSpec double CPUSolver::getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept
    {
        assert(rowIndex != colIndex);
        //assert(matrixA.values != nullptr);
        u8_t cpuColIdx;
        for(cpuColIdx = 1; cpuColIdx < matrixA.numColsInRow[rowIndex]; ++cpuColIdx)
            if(matrixA.columnIndeces[rowIndex][cpuColIdx] == colIndex)
                break;

        return matrixA.values[rowIndex][cpuColIdx] * matrixA.values[rowIndex][0];
    }

    inline SF3Derror_t solverHostCheckError(SF3Derror_t retError, solverStatus& status)
    {
        if(retError != SF3Derror_t::SF3Dok)
            status = solverStatus::Error;

        return retError;
    }
}

