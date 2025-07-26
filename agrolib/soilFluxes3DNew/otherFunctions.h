#ifndef OTHERFUNCTIONS_H
#define OTHERFUNCTIONS_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Math    //move to mathFunctions
{
    double computeMean(double v1, double v2, soilFluxes3D::New::meanType_t type = soilFluxes3D::New::Logarithmic);
    double arithmeticMean(double v1, double v2);
    double geometricMean(double v1, double v2);
    double logaritmicMean(double v1, double v2);

    double vectorNorm(double vector[], size_t size);
}
#endif // OTHERFUNCTIONS_H
