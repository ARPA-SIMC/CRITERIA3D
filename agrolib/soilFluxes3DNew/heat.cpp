#include "heat.h"
#include "soilPhysics.h"
#include "otherFunctions.h"

using namespace soilFluxes3D::New;
using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Math;

namespace soilFluxes3D::New
{
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd simulationFlags_t simulationFlags;
}

namespace soilFluxes3D::Heat
{

double computeThermalLiquidFlux(uint64_t srcIndex, uint64_t dstIndex, [[maybe_unused]] double timeStep, [[maybe_unused]] double timeStepWater, processType process)
{
    // TO DO: inserire time step water per calcolo più preciso

    double [[maybe_unused]] tavg, tavglink, havg, havglink;
    switch(process)
    {
        case processType::Water:
            if(!simulationFlags.computeWater)
                return noData;

            tavg = getNodeMeanTemperature(srcIndex);
            tavglink = getNodeMeanTemperature(dstIndex);
            havg = nodeGrid.waterData.pressureHead[srcIndex] - nodeGrid.z[srcIndex];
            havglink = nodeGrid.waterData.pressureHead[dstIndex] - nodeGrid.z[dstIndex];
            break;
        case processType::Heat:
            if(!simulationFlags.computeHeat)
                return noData;

            tavg = nodeGrid.heatData.temperature[srcIndex];
            tavglink = nodeGrid.heatData.temperature[dstIndex];
            //havg = computeMean(getH_timeStep(srcIndex, timeStep, timeStepWater), nodeGrid.waterData.oldPressureHeads[srcIndex], Arithmetic) - nodeGrid.z[srcIndex];
            //havglink = computeMean(getH_timeStep(dstIndex, timeStep, timeStepWater), nodeGrid.waterData.oldPressureHeads[dstIndex], Arithmetic) - nodeGrid.z[dstIndex];
            break;
        default:
            return noData;
    }

    //TO COMPLETE
    return 0;

}

double computeThermalVaporFlux([[maybe_unused]] uint64_t srcIndex, [[maybe_unused]] uint64_t dstIndex, [[maybe_unused]] double timeStep, [[maybe_unused]] double timeStepWater, [[maybe_unused]] processType process)
{
    //TO COMPLETE
    return 0;
}

}
