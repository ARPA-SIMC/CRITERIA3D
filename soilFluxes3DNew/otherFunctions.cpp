#include "otherFunctions.h"

//TEMP
using namespace soilFluxes3D::New;

namespace soilFluxes3D::Math
{
    __cudaSpec double computeMean(double v1, double v2, meanType_t type)
    {
        switch(type)
        {
            case Arithmetic:
                return arithmeticMean(v1, v2);
            case Geometric:
                return geometricMean(v1, v2);
            case Logarithmic:
                return logaritmicMean(v1, v2);
            default:
                return logaritmicMean(v1, v2);  //
        }
    }

    __cudaSpec double arithmeticMean(double v1, double v2)
    {
        return (v1 + v2) / 2;
    }

    __cudaSpec double geometricMean(double v1, double v2)
    {
        int8_t sign = (v1 > 0) - (v1 < 0);
        return sign * std::sqrt(v1 * v2);
    }

    __cudaSpec double logaritmicMean(double v1, double v2)
    {
        return (v1 == v2) ? v1 : (v1 - v2) / std::log(v1/v2);
    }

    __cudaSpec double vectorNorm(double vector[], size_t size)
    {
        double norm = 0;
        for (size_t idx = 0; idx < size; ++idx)
            norm += std::pow(vector[idx], 2);

        return std::sqrt(norm);
    }
} //namespace
