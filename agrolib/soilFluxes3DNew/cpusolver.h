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

            SF3Derror_t inizialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
    };

} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_CPUSOLVER_H
