#pragma once

#include "macro.h"
#include "types.h"

using namespace soilFluxes3D::v2;
namespace soilFluxes3D::v2::Math
{
    __cudaSpec double computeMean(double v1, double v2, meanType_t type = meanType_t::Logarithmic);
    __cudaSpec double arithmeticMean(double v1, double v2);
    __cudaSpec double geometricMean(double v1, double v2);
    __cudaSpec double logaritmicMean(double v1, double v2);

    __cudaSpec double vectorNorm(double vector[], std::size_t size);
}
