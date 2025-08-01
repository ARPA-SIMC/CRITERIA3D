#ifndef OTHERFUNCTIONS_H
#define OTHERFUNCTIONS_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Math    //move to mathFunctions
{
    __cudaSpec double computeMean(double v1, double v2, soilFluxes3D::New::meanType_t type = soilFluxes3D::New::Logarithmic);
    __cudaSpec double arithmeticMean(double v1, double v2);
    __cudaSpec double geometricMean(double v1, double v2);
    __cudaSpec double logaritmicMean(double v1, double v2);

    __cudaSpec double vectorNorm(double vector[], size_t size);
}
#endif // OTHERFUNCTIONS_H
