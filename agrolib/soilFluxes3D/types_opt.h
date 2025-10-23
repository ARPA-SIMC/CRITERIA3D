#pragma once

#include <optional>
#include "types.h"

namespace soilFluxes3D::v2
{
    #define updateFromPartial(total, partial, field) if (partial.field) {total.field = *(partial.field);}

    //TO DO: remove not used fields
    struct SolverParametersPartial
    {
        std::optional<double> MBRThreshold;
        std::optional<double> residualTolerance;

        std::optional<double> deltaTmin;
        std::optional<double> deltaTmax;
        std::optional<double> deltaTcurr;

        std::optional<u16_t> maxApproximationsNumber;
        std::optional<u16_t> maxIterationsNumber;

        std::optional<WRCModel> waterRetentionCurveModel;
        std::optional<meanType_t> meanType;

        std::optional<float> lateralVerticalRatio;
        std::optional<double> heatWeightFactor;

        std::optional<double> CourantWaterThreshold;
        std::optional<double> instabilityFactor;

        std::optional<bool> enableOMP;
        std::optional<u32_t> numThreads;
    };
}
