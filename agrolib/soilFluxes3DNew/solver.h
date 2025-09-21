#ifndef SOILFLUXES3D_SOLVER_H
#define SOILFLUXES3D_SOLVER_H

#include <cstdlib>
#include <cassert>

#include "types_cpu.h"
#include "macro.h"

namespace soilFluxes3D::New
{
    class Solver
    {
        protected:
            numericalMethod _method;
            solverStatus _status = Created;
            solverType _type;

            SolverParameters _parameters;
            double _bestMBRerror;

            __cudaSpec uint32_t calcCurrentMaxIterationNumber(uint8_t approxNumber);
            virtual bool solveLinearSystem(uint8_t approximationNumber, processType computationType) = 0;

        public:
            Solver(solverType type, numericalMethod method) : _type(type), _method(method) {}

            void updateParameters(const SolverParametersPartial &newParameters) noexcept;
            void setTimeStep(double timeStep) noexcept;

            __cudaSpec solverType getSolverType() const noexcept;
            __cudaSpec WRCModel getWRCModel() const noexcept;
            __cudaSpec bool getOMPstatus() const noexcept;
            __cudaSpec double getMaxTimeStep() const noexcept;
            __cudaSpec double getLVRatio() const noexcept;
            __cudaSpec meanType_t getMeanType() const noexcept;

            template<class Derived>
            __cudaSpec double getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) const noexcept;

            virtual SF3Derror_t inizialize() = 0;
            virtual SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) = 0;
            virtual SF3Derror_t clean() = 0;
    };

    inline __cudaSpec uint32_t Solver::calcCurrentMaxIterationNumber(uint8_t approxNumber)
    {
        uint32_t maxCurrIterNum = uint32_t((approxNumber + 1) * (float(_parameters.maxIterationsNumber) / float(_parameters.maxApproximationsNumber)));
        return SF3Dmax(maxCurrIterNum, uint32_t(20));
        //return SF3Dmax(maxCurrIterNum, static_cast<uint32_t>(_parameters.maxIterationsNumber));
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
    inline void Solver::setTimeStep(double timeStep) noexcept
    {
        if(timeStep < _parameters.deltaTmin)
            _parameters.deltaTcurr = _parameters.deltaTmin;

        if(timeStep > _parameters.deltaTmax)
            _parameters.deltaTcurr = _parameters.deltaTmax;

        _parameters.deltaTcurr = timeStep;
    }

    inline __cudaSpec solverType Solver::getSolverType() const noexcept
    {
        return _type;
    }
    inline __cudaSpec WRCModel Solver::getWRCModel() const noexcept
    {
        return _parameters.waterRetentionCurveModel;
    }
    inline __cudaSpec bool Solver::getOMPstatus() const noexcept
    {
        return _parameters.enableOMP;
    }
    inline __cudaSpec double Solver::getMaxTimeStep() const noexcept
    {
        return _parameters.deltaTmax;
    }
    inline __cudaSpec double Solver::getLVRatio() const noexcept
    {
        return _parameters.lateralVerticalRatio;
    }
    inline __cudaSpec meanType_t Solver::getMeanType() const noexcept
    {
        return _parameters.meantype;
    }

    template<class Derived>
    __cudaSpec double Solver::getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) const noexcept
    {
        return static_cast<const Derived*>(this)->getMatrixElementValue(rowIndex, colIndex);
    }
} // namespace soilFluxes3D

#endif // SOILFLUXES3D_SOLVER_H
