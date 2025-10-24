#pragma once

#include "solver.h"
#include "types_cpu.h"

namespace soilFluxes3D::v2
{
    class CPUSolver : public Solver
    {
        private:
            MatrixCPU matrixA;
            VectorCPU vectorB, vectorX;

            VectorCPU vectorC;

            void waterMainLoop(double maxTimeStep, double& acceptedTimeStep);
            balanceResult_t waterApproximationLoop(double deltaT);

            void heatLoop(double timeStepHeat, double timeStepWater);


            bool solveLinearSystem(u8_t approximationNumber, processType computationType) override;

        public:
            CPUSolver() : Solver(solverType::CPU, numericalMethod::Jacobi) {}
            __cudaSpec double getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept;

            SF3Derror_t initialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t clean() override;
    };

    inline __cudaSpec double CPUSolver::getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept
    {
        assert(rowIndex != colIndex);
        //assert(matrixA.values != nullptr);
        u8_t cpuColIdx;
        for(cpuColIdx = 1; cpuColIdx < matrixA.numColumns[rowIndex]; ++cpuColIdx)
            if(matrixA.colIndeces[rowIndex][cpuColIdx] == colIndex)
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

