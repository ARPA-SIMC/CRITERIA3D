#include "water_new.h"
#include "soil_new.h"

using namespace soilFluxes3D::New;
using namespace soilFluxes3D::Soil;


/*extern*/ bool enableOMP = true;
/*extern*/ nodesData_t nodesData;
balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;

//TEMP, problemi di conflitti
namespace soilFluxes3D::Water
{

/*!
 * \brief inizializes the water balance variables
 * \return Ok/Error
 */
SF3Derror_t initializeWaterBalance()
{
    double twc = computeTotalWaterContent_new();
    balanceDataWholePeriod.waterStorage = twc;
    balanceDataCurrentPeriod.waterStorage = twc;
    balanceDataCurrentTimeStep.waterStorage = twc;
    balanceDataPreviousTimeStep.waterStorage = twc;

    if(!nodesData.isInizialized)
        return MemoryError;

    //link flow
    for (uint8_t linkIndex = 0; linkIndex < maxTotalLink; ++linkIndex)
        nodesData.linkData[linkIndex].waterFlow = (double*) calloc(nodesData.numNodes, sizeof(double));

    //boundary flow sum
    nodesData.boundaryData.waterFlowSum = (double*) calloc(nodesData.numNodes, sizeof(double));

    return SF3Dok;
}

/*!
 * \brief computes the total water content
 * \return total water content [m3]
 */
double computeTotalWaterContent_new()
{
    {
        if(!nodesData.isInizialized)
            return -1;

        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum) if(enableOMP)
        for (uint64_t idx = 0; idx < nodesData.numNodes; ++idx)
        {
            double theta = nodesData.surfaceFlag[idx] ? (nodesData.waterData.pressureHead[idx] - nodesData.z[idx]) : thetaFromSe(idx);
            sum += theta * nodesData.size[idx];
        }

        return sum;
    }
}

/*!
 * \brief computes the mass balance error of the current time step
 * \param deltaT    [s]
 */
void computeCurrentMassBalance(double deltaT)
{
    balanceDataCurrentTimeStep.waterStorage = computeTotalWaterContent_new();
    double deltaStorage = balanceDataCurrentTimeStep.waterStorage - balanceDataPreviousTimeStep.waterStorage;

    balanceDataCurrentTimeStep.waterSinkSource = computeWaterSinkSourceFlowsSum(deltaT);
    balanceDataCurrentTimeStep.waterMBE = deltaStorage - balanceDataCurrentTimeStep.waterSinkSource;

    // minimum reference water storage [m3] as % of current storage
    double timePercentage = 0.01 * std::max(deltaT, 1.) / HOUR_SECONDS;
    double minRefWaterStorage = balanceCurrentTimeStep.storageWater * timePercentage;
    minRefWaterStorage = std::max(minRefWaterStorage, 0.001);

    // Reference water for computation of mass balance error ratio
    // when the water sink/source is too low, use the reference water storage
    double referenceWater = std::max(fabs(balanceCurrentTimeStep.sinkSourceWater), minRefWaterStorage);     // [m3]

    balanceCurrentTimeStep.waterMBR = balanceCurrentTimeStep.waterMBE / referenceWater;
}

/*!
 * \brief computes sum of water sink/source flows
 * \param deltaT    [s]
 * \return sum of water sink/source [m3]
 */
double computeWaterSinkSourceFlowsSum(double deltaT)
{
    double sum = 0;

    #pragma omp parallel for reduction(+:sum) if(enableOMP)
    for (uint64_t idx = 0; idx < nodesData.numNodes; ++idx)
        if(nodesData.waterData.waterFlow[idx] != 0)     //TO DO: evaluate remove check
            sum += nodesData.waterData.waterFlow[idx] * deltaT;

    return sum;
}


bool evaluateWaterBalance(uint8_t approxNr, double& bestMBRerror, SolverParameters& parameters)
{
    //setForcedHalvedTime(false)            //TO DO: needed?
    computeCurrentMassBalance(parameters.deltaTcurr);

    double currMBRerror = fabs(balanceCurrentTimeStep.waterMBR);

    //Optimal error
    if(currMBRerror < parameters.MBRThreshold)
    {
        //acceptStep()                      //TO DO

        double currCWL = nodesData.waterData.CourantWaterLevel;
        //Check Stability (Courant)
        if(currCWL < parameters.CourantWaterThreshold)
        {
            //increase deltaT
            parameters.deltaTcurr = (currCWL > 0.5) ? parameters.deltaTcurr * 2 : parameters.deltaTcurr / currCWL;
            parameters.deltaTcurr = std::min(parameters.deltaTcurr, parameters.deltaTmax);
            if(parameters.deltaTcurr > 1.0)
                parameters.deltaTcurr = floor(parameters.deltaTcurr);
        }
        return true;
    }

    //Good error or first approximation
    if (approxNr == 0 || currMBRerror < bestMBRerror)
    {
        //saveBestStep()                    //TO DO
        bestMBRerror = currMBRerror;
    }

    //Critical error (unstable system) or last approximation
    if (approxNr == (parameters.maxApproximationsNumber - 1) || currMBRerror > (bestMBRerror*parameters.instabilityFactor))
    {
        if(parameters.deltaTcurr <= parameters.deltaTmin)
        {
            //restoreBestStep(deltaT);      //TO DO
            //acceptStep(deltaT);           //TO DO
            return true;
        }

        parameters.deltaTcurr = std::max(parameters.deltaTcurr / 2, parameters.deltaTmax);
        //setForcedHalvedTime(true)        //TO DO: needed?
    }

    return false;
}



}//namespace
