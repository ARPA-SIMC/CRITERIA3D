#ifndef PARAMETERS_H
#define PARAMETERS_H

    #ifndef COMMONCONSTANTS_H
        #include "../../mathFunctions/commonConstants.h"
    #endif

    struct TParameters
    {
        int numericalSolutionMethod;
        double MBRThreshold;
        double ResidualTolerance;
        double delta_t_min;
        double delta_t_max;
        double current_delta_t;
        int iterazioni_min;
        int iterazioni_max;
        int maxApproximationsNumber;
        int waterRetentionCurve;
        int meanType;
        float k_lateral_vertical_ratio;
        double heatWeightingFactor;

        void initialize()
        {
            numericalSolutionMethod = RELAXATION;
            delta_t_min = 1;
            delta_t_max = 600;
            current_delta_t = delta_t_max;
            iterazioni_max = 200;
            maxApproximationsNumber = 10;
            MBRThreshold = 1E-6;
            ResidualTolerance = 1E-10;
            waterRetentionCurve = MODIFIEDVANGENUCHTEN;
            meanType = MEAN_LOGARITHMIC;
            k_lateral_vertical_ratio = 10.;
            heatWeightingFactor = 0.5;
        }
    };

#endif  // PARAMETERS_H
