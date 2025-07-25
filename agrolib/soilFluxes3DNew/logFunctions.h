#ifndef SOILFLUXES3D_LOGFUNCTIONS_H
#define SOILFLUXES3D_LOGFUNCTIONS_H

#include <QString>
#include "types_cpu.h"

#include "macro.h"

namespace soilFluxes3D::Log
{
    //Log di matrici e vettori
    __EXTERN QString getMatrixLog();
    __EXTERN QString getMatrixLog_formatted();

    __EXTERN QString getVectorLog();

    //Log dei parametri
    __EXTERN QString getLinSystLog();

    }
#endif // SOILFLUXES3D_LOGFUNCTIONS_H
