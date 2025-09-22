#ifndef SOILFLUXES3D_TYPES_OPT_H
#define SOILFLUXES3D_TYPES_OPT_H

#include <optional>
#include "types_cpu.h"

namespace soilFluxes3D::New
{
    #define updateFromPartial(total, partial, field) {if (partial.field) {total.field = *partial.field;}}

    //TO DO: remove not used fields
    struct SolverParametersPartial
    {
        std::optional<double> MBRThreshold;
        std::optional<double> residualTolerance;

        std::optional<double> deltaTmin;
        std::optional<double> deltaTmax;
        std::optional<double> deltaTcurr;

        std::optional<uint16_t> maxApproximationsNumber;
        std::optional<uint16_t> maxIterationsNumber;

        std::optional<WRCModel> waterRetentionCurveModel;
        std::optional<meanType_t> meanType;

        std::optional<float> lateralVerticalRatio;
        std::optional<double> heatWeightFactor;

        std::optional<double> CourantWaterThreshold;
        std::optional<double> instabilityFactor;

        std::optional<bool> enableOMP;
        std::optional<uint32_t> numThreads;
    };
}


#endif // SOILFLUXES3D_TYPES_OPT_H
