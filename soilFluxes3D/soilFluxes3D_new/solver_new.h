#ifndef SOILFLUXES3D_SOLVER_H
#define SOILFLUXES3D_SOLVER_H

#include "types_cpu.h"
#include "macro.h"

namespace soilFluxes3D::New
{
    class Solver
    {
        protected:
            numericalMethod _method;
            solverStatus _status;
            solverType _type;

            SolverParameters _parameters;

            double bestMBRerror;
            bool isHalfTimeStepForced = false;

        public:
            Solver() {}
            virtual void inizialize(){}
            virtual void run(){}
            virtual void gatherOutput(double*&){}
    };

    bool solveLinearSystem(int approximation, double residualTolerance, processType computationType);

    //Move to member?
    __SF3DINLINE void halveTimeStep();
    __SF3DINLINE bool getForcedHalvedTime();
    __SF3DINLINE void setForcedHalvedTime(bool isForced);

} // namespace soilFluxes3D

#endif // SOILFLUXES3D_SOLVER_H
