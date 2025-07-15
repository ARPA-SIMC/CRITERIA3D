#ifndef SOILFLUXES3D_H
#define SOILFLUXES3D_H

#include "macro.h"
#include "types.h"

namespace soilFluxes3D_New
{
    nodesData_t nodeGrid;
    simulationParameters_t simulationParameters;

    SF3Derror_t setNode_new(uint64_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope = 0, double boundaryArea = 0);

    SF3Derror_t setNodeLink_new(uint64_t nodeIndex, uint64_t linkIndex, linkType_t direction, double interfaceArea);
}

#endif // SOILFLUXES3D_H
