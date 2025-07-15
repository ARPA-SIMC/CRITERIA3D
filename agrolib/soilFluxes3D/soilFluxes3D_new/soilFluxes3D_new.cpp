#include "commonConstants.h"

#include "soilFluxes3D_new.h"

using namespace soilFluxes3D::New;

// SF3Derror_t setNode_new(uint64_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope = 0, double boundaryArea = 0)
// {
//     if(!nodeGrid.isInizialized)
//         return MemoryError;

//     if(index >= nodeGrid.numNodes)
//         return IndexError;

//     nodeGrid.size[index] = volume_or_area;
//     nodeGrid.x[index] = x;
//     nodeGrid.y[index] = y;
//     nodeGrid.z[index] = z;

//     nodeGrid.surfaceFlag[index] = isSurface;

//     nodeGrid.boundaryType[index] = boundaryType;
//     if(boundaryType != None)
//     {
//         nodeGrid.boundarySlope[index] = slope;
//         nodeGrid.boundarySize[index] = boundaryArea;
//     }

//     if(simulationParameters.computeWater)
//     {
//         nodeGrid.waterData->pond[index] = isSurface ? 0.0001f : noData;
//         nodeGrid.waterData->waterSinkSource[index] = 0;
//     }

//     if(simulationParameters.computeHeat && !isSurface)
//     {
//         nodeGrid.heatData->temperature[index] = ZEROCELSIUS + 20;
//         nodeGrid.heatData->oldTemperature[index] = ZEROCELSIUS + 20;
//         nodeGrid.heatData->heatFlux[index] = 0;
//         nodeGrid.heatData->heatSinkSource[index] = 0;
//     }

//     return Crit3Dok;
// }

// SF3Derror_t setNodeLink_new(uint64_t nodeIndex, uint64_t linkIndex, linkType_t direction, double interfaceArea)
// {
//     if(!nodeGrid.isInizialized)
//         return MemoryError;

//     if(nodeIndex >= nodeGrid.numNodes || linkIndex >= nodeGrid.numNodes)
//         return IndexError;


//     uint8_t idx;
//     switch (direction)
//     {
//     case Up:
//         idx = 0;
//         break;
//     case Down:
//         idx = 1;
//         break;
//     case Lateral:
//         if(nodeGrid.numLateralLink[nodeIndex] == maxLateralLink)
//             return TopographyError;
//         idx = 2 + nodeGrid.numLateralLink[nodeIndex];
//         break;
//     default:
//         return ParameterError;
//     }

//     nodeGrid.linkData[idx].linktype[nodeIndex] = direction;
//     nodeGrid.linkData[idx].index[nodeIndex] = linkIndex;
//     nodeGrid.linkData[idx].interfaceArea[nodeIndex] = interfaceArea;

//     if(simulationParameters.computeWater)
//         nodeGrid.linkData[idx].waterLinkData->linkWaterFlow[nodeIndex] = 0;

//     if(simulationParameters.computeHeat)
//     {
//         nodeGrid.linkData[idx].heatLinkData->waterFlux[nodeIndex] = 0;
//         nodeGrid.linkData[idx].heatLinkData->vaporFlux[nodeIndex] = 0;
//         //switch sul saveheatFluxType + conseguente inizializzazione
//     }

//     return Crit3Dok;
// }
