#include "otherFunctions.h"

using namespace soilFluxes3D::v2;

namespace soilFluxes3D::v2::Math
{
    __cudaSpec double computeMean(double v1, double v2, meanType_t type)
    {
        switch(type)
        {
            case meanType_t::Arithmetic:
                return arithmeticMean(v1, v2);
            case meanType_t::Geometric:
                return geometricMean(v1, v2);
            case meanType_t::Logarithmic:
                return logarithmicMean(v1, v2);
            default:
                return logarithmicMean(v1, v2);
        }
    }

    __cudaSpec double arithmeticMean(double v1, double v2)
    {
        return (v1 + v2) * 0.5;
    }

    __cudaSpec double geometricMean(double v1, double v2)
    {
        int8_t sign = (v1 > 0) - (v1 < 0);
        return sign * std::sqrt(v1 * v2);
    }

    __cudaSpec double logarithmicMean(double v1, double v2)
    {
        if (v1 <= 0.0 || v2 <= 0.0)
            return 0.0;

        // protezione numerica: se i valori sono molto vicini,
        // la media logaritmica è numericamente instabile ma converge a quella aritmetica.
        const double diff = std::fabs(v1 - v2);
        if (diff < 1e-6 * std::max(v1, v2))
            return (v1 + v2) * 0.5;

        return (v1 - v2) / std::log(v1 / v2);
    }

    __cudaSpec double vectorNorm(const double *vector, std::size_t size)
    {
        double norm = 0;
        for (std::size_t idx = 0; idx < size; ++idx)
            norm += (vector[idx] * vector[idx]);

        return std::sqrt(norm);
    }

} //namespace
