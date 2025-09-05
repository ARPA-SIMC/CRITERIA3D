#ifndef SOILFLUXES3D_CPUSOLVER_H
#define SOILFLUXES3D_CPUSOLVER_H

#include "solver.h"

namespace soilFluxes3D::New
{
    class CPUSolver : public Solver
    {
        private:
            MatrixCPU matrixA;
            VectorCPU vectorB, vectorX;

            VectorCPU vectorC;

            void waterMainLoop(double maxTimeStep, double& acceptedTimeStep);
            balanceResult_t waterApproximationLoop(double deltaT);
            bool solveLinearSystem(uint8_t approximationNumber, processType computationType) override;

        public:
            CPUSolver() : Solver(CPU, Jacobi) {}
            __cudaSpec double getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) const noexcept;

            SF3Derror_t inizialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t clean() override;
    };

    inline __cudaSpec double CPUSolver::getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) const noexcept
    {
        //assert(matrixA.values != nullptr);
        uint8_t cpuColIdx;
        for(cpuColIdx = 0; cpuColIdx < matrixA.numColumns[rowIndex]; ++cpuColIdx)
            if(matrixA.colIndeces[rowIndex][cpuColIdx] == colIndex)
                break;

        //assert(cpuColIdx < matrixA.numColumns[rowIndex]);
        return matrixA.values[rowIndex][cpuColIdx] * matrixA.values[rowIndex][0];
    }

} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_CPUSOLVER_H
