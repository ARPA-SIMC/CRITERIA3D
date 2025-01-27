#ifndef PARAMETERS_H
#define PARAMETERS_H

    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif

    struct TParameters
    {
        int numericalSolutionMethod;
        int threadsNumber;
        double MBRThreshold;
        double ResidualTolerance;
        double delta_t_min;
        double delta_t_max;
        double current_delta_t;
        int maxIterationsNumber;
        int maxApproximationsNumber;
        int waterRetentionCurve;
        int meanType;
        float k_lateral_vertical_ratio;
        double heatWeightingFactor;

        void initialize()
        {
            numericalSolutionMethod = GAUSS_SEIDEL;
            threadsNumber = 1;
            delta_t_min = 1;
            delta_t_max = 600;
            current_delta_t = delta_t_max;
            maxIterationsNumber = 200;
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
