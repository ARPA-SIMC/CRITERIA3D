#ifndef SOILFLUXES3D_CPUSOLVER_H
#define SOILFLUXES3D_CPUSOLVER_H

#include "solver_new.h"

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
            CPUSolver() {}
            double getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) override;

            SF3Derror_t inizialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
    };

    inline double CPUSolver::getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex)
    {
        //assert(matrixA.values != nullptr);
        uint8_t cpuColIdx;
        for(cpuColIdx = 0; cpuColIdx < matrixA.numColumns[rowIndex]; ++cpuColIdx)
            if(matrixA.colIndeces[rowIndex][cpuColIdx] == colIndex)
                break;

        //assert(cpuColIdx < matrixA.numColumns[rowIndex]);
        return matrixA.values[rowIndex][cpuColIdx];
    }

} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_CPUSOLVER_H
