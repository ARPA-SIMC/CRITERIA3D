#ifndef SOILFLUXES3D_SOLVER_H
#define SOILFLUXES3D_SOLVER_H

#include "types_cpu.h"
#include "macro.h"
#include <algorithm>

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

            uint32_t calcCurrentMaxIterationNumber(uint8_t approxNumber);
            virtual bool solveLinearSystem(uint8_t approximationNumber, processType computationType) = 0;

        public:
            Solver() {}

            void updateParameters(const SolverParametersPartial &newParameters) noexcept;
            void setTimeStep(const double timeStep) noexcept;

            WRCModel getWRCModel() const noexcept;
            bool getOMPstatus() const noexcept;
            double getMaxTimeStep() const noexcept;
            double getLVRatio() const noexcept;
            meanType_t getMeanType() const noexcept;

            virtual SF3Derror_t inizialize() = 0;
            virtual SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) = 0;
    };

    inline uint32_t Solver::calcCurrentMaxIterationNumber(uint8_t approxNumber)
    {
        uint32_t maxCurrIterNum = uint32_t((approxNumber + 1) * (double(_parameters.maxIterationsNumber) / double(_parameters.maxApproximationsNumber)));
        return std::max(maxCurrIterNum, static_cast<uint32_t>(_parameters.maxIterationsNumber));
    }

    inline void Solver::updateParameters(const SolverParametersPartial &newParameters) noexcept
    {
        updateFromPartial(_parameters, newParameters, MBRThreshold);
        updateFromPartial(_parameters, newParameters, residualTolerance);
        updateFromPartial(_parameters, newParameters, deltaTmin);
        updateFromPartial(_parameters, newParameters, deltaTmax);
        updateFromPartial(_parameters, newParameters, deltaTcurr);
        updateFromPartial(_parameters, newParameters, maxApproximationsNumber);
        updateFromPartial(_parameters, newParameters, maxIterationsNumber);
        updateFromPartial(_parameters, newParameters, waterRetentionCurveModel);
        updateFromPartial(_parameters, newParameters, meantype);
        updateFromPartial(_parameters, newParameters, lateralVerticalRatio);
        updateFromPartial(_parameters, newParameters, heatWeightFactor);
        updateFromPartial(_parameters, newParameters, CourantWaterThreshold);
        updateFromPartial(_parameters, newParameters, instabilityFactor);
        updateFromPartial(_parameters, newParameters, enableOMP);
        updateFromPartial(_parameters, newParameters, numThreads);
    }
    inline void Solver::setTimeStep(const double timeStep) noexcept
    {
        if(timeStep < _parameters.deltaTmin)
            _parameters.deltaTcurr = _parameters.deltaTmin;

        if(timeStep > _parameters.deltaTmax)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.deltaTcurr = timeStep;
    }

    inline WRCModel Solver::getWRCModel() const noexcept
    {
        return _parameters.waterRetentionCurveModel;
    }
    inline bool Solver::getOMPstatus() const noexcept
    {
        return _parameters.enableOMP;
    }
    inline double Solver::getMaxTimeStep() const noexcept
    {
        return _parameters.deltaTmax;
    }
    inline double Solver::getLVRatio() const noexcept
    {
        return _parameters.lateralVerticalRatio;
    }
    inline meanType_t Solver::getMeanType() const noexcept
    {
        return _parameters.meantype;
    }
} // namespace soilFluxes3D

#endif // SOILFLUXES3D_SOLVER_H
