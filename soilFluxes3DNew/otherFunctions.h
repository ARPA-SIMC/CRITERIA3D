#ifndef OTHERFUNCTIONS_H
#define OTHERFUNCTIONS_H

#include "macro.h"
#include "types_cpu.h"

using namespace soilFluxes3D::New;

namespace soilFluxes3D::Math    //move to mathFunctions
{
    double computeMean(double v1, double v2, meanType_t type = Logarithmic);
    double arithmeticMean(double v1, double v2);
    double geometricMean(double v1, double v2);
    double logaritmicMean(double v1, double v2);

    double vectorNorm(double vector[], size_t size);
}
#endif // OTHERFUNCTIONS_H
