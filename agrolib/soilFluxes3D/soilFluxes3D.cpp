#include "commonConstants.h"
#include "soilFluxes3D.h"
#include "soilPhysics.h"
#include "water.h"
#include "heat.h"

#ifdef MCR_ENABLED
    #include "logFunctions.h"
    using namespace soilFluxes3D::v2::Log;
#endif

#include "cpusolver.h"
#ifdef CUDA_ENABLED
    #include "gpusolver.h"
#endif

#include <algorithm>
#include <new>

using namespace soilFluxes3D::v2::Soil;
using namespace soilFluxes3D::v2::Water;
using namespace soilFluxes3D::v2::Heat;

namespace soilFluxes3D::v2
{
    //Solver objects
    CPUSolver CPUSolverObject;
    #ifdef CUDA_ENABLED
        GPUSolver GPUSolverObject;
        bool CUDAactive = true;
    #endif

    //global variables
    __cudaMngd Solver* solver = nullptr;
    __cudaMngd nodesData_t nodeGrid;
    __cudaMngd simulationFlags_t simulationFlags;
    __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;

    std::vector<std::vector<u16_t>> soil1DIndeces = {};
    std::vector<soilData_t> soilList = {};
    std::vector<surfaceData_t> surfaceList = {};
    std::vector<culvertData_t> culvertList = {};

    /*!
     *  \brief initializes the node data grid and set simulation parameters
     *  \return Ok/Error
    */
    SF3Derror_t initializeSF3D(SF3Duint_t nrNodes, u16_t nrLayers, u8_t nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes, heatFluxSaveMode_t HFsm)
    {
        //Cleans all the data structures
        SF3Derror_t cleanResult = cleanSF3D();
        if(cleanResult != SF3Derror_t::SF3Dok)
            return cleanResult;

        //Set flags data
        simulationFlags.computeWater = isComputeWater;
        simulationFlags.computeHeat = isComputeHeat;
        if(isComputeHeat)
        {
            simulationFlags.computeHeatVapor = true;
            simulationFlags.computeHeatAdvection = true;
            simulationFlags.HFsaveMode = HFsm;
        }
        simulationFlags.computeSolutes = isComputeSolutes;

        //initialize data structures
        nodeGrid.numNodes = nrNodes;
        nodeGrid.numLayers = nrLayers;

        //Check with a define value. In CRITERIA3D nrLateralLink is a compile-time constant
        if(nrLateralLinks > maxLateralLink)
            return SF3Derror_t::ParameterError;

        //Topology data
        hostAlloc(nodeGrid.size, nrNodes);
        hostAlloc(nodeGrid.x, nrNodes);
        hostAlloc(nodeGrid.y, nrNodes);
        hostAlloc(nodeGrid.z, nrNodes);
        hostAlloc(nodeGrid.surfaceFlag, nrNodes);

        //Soil/Surface data
        hostAlloc(nodeGrid.soilSurfacePointers, nrNodes);

        //Boundary data
        hostAlloc(nodeGrid.boundaryData.boundaryType, nrNodes); //NoBoundary is equal 0, automatic set with calloc
        hostAlloc(nodeGrid.boundaryData.boundarySlope, nrNodes);
        hostAlloc(nodeGrid.boundaryData.boundarySize, nrNodes);
        //if(isComputeWater)
        hostAlloc(nodeGrid.boundaryData.waterFlowRate, nrNodes);
        hostAlloc(nodeGrid.boundaryData.waterFlowSum, nrNodes);
        hostAlloc(nodeGrid.boundaryData.prescribedWaterPotential, nrNodes);
        
        if(isComputeHeat)
        {
            hostAlloc(nodeGrid.boundaryData.heightWind, nrNodes);
            hostAlloc(nodeGrid.boundaryData.heightTemperature, nrNodes);
            hostAlloc(nodeGrid.boundaryData.roughnessHeight, nrNodes);
            hostAlloc(nodeGrid.boundaryData.aerodynamicConductance, nrNodes);
            hostAlloc(nodeGrid.boundaryData.soilConductance, nrNodes);
            hostAlloc(nodeGrid.boundaryData.temperature, nrNodes);
            hostAlloc(nodeGrid.boundaryData.relativeHumidity, nrNodes);
            hostAlloc(nodeGrid.boundaryData.windSpeed, nrNodes);
            hostAlloc(nodeGrid.boundaryData.netIrradiance, nrNodes);
            hostAlloc(nodeGrid.boundaryData.sensibleFlux, nrNodes);
            hostAlloc(nodeGrid.boundaryData.latentFlux, nrNodes);
            hostAlloc(nodeGrid.boundaryData.radiativeFlux, nrNodes);
            hostAlloc(nodeGrid.boundaryData.advectiveHeatFlux, nrNodes);
            hostAlloc(nodeGrid.boundaryData.fixedTemperatureValue, nrNodes);
            hostAlloc(nodeGrid.boundaryData.fixedTemperatureDepth, nrNodes);
        }

        //Link data
        hostAlloc(nodeGrid.numLateralLink, nrNodes);
        for(u8_t linkIdx = 0; linkIdx < maxTotalLink; ++linkIdx)
        {
            hostAlloc(nodeGrid.linkData[linkIdx].linkType, nrNodes);    //NoLink is equal 0, automatic set with calloc
            hostAlloc(nodeGrid.linkData[linkIdx].linkIndex, nrNodes);;
            hostAlloc(nodeGrid.linkData[linkIdx].interfaceArea, nrNodes);

            //if(isComputeWater)
            hostAlloc(nodeGrid.linkData[linkIdx].waterFlowSum, nrNodes);

            if(isComputeHeat)
            {
                hostAlloc(nodeGrid.linkData[linkIdx].waterFlux, nrNodes);
                hostAlloc(nodeGrid.linkData[linkIdx].vaporFlux, nrNodes);
                for(u8_t fluxIdx = 0; fluxIdx < numTotalFluxTypes; ++fluxIdx)     //maybe move to initHeatFlag
                    hostAlloc(nodeGrid.linkData[linkIdx].fluxes[fluxIdx], nrNodes);
            }
        }

        //Water data    //if(isComputeWater)
        hostAlloc(nodeGrid.waterData.saturationDegree, nrNodes);
        hostAlloc(nodeGrid.waterData.waterConductivity, nrNodes);
        hostAlloc(nodeGrid.waterData.waterFlow, nrNodes);
        hostAlloc(nodeGrid.waterData.pressureHead, nrNodes);
        hostAlloc(nodeGrid.waterData.waterSinkSource, nrNodes);
        hostAlloc(nodeGrid.waterData.pond, nrNodes);
        hostAlloc(nodeGrid.waterData.oldPressureHead, nrNodes);
        hostAlloc(nodeGrid.waterData.bestPressureHead, nrNodes);
        hostAlloc(nodeGrid.waterData.invariantFluxes, nrNodes);
        hostAlloc(nodeGrid.waterData.partialCourantWaterLevels, nrNodes);

        //Culvert pointers
        hostAlloc(nodeGrid.culvertPtr, nrNodes);

        //Heat data
        if(isComputeHeat)
        {
            hostAlloc(nodeGrid.heatData.temperature, nrNodes);      //needs to be initialized with non-zero value?
            hostAlloc(nodeGrid.heatData.oldTemperature, nrNodes);   //needs to be initialized with non-zero value?
            hostAlloc(nodeGrid.heatData.heatFlux, nrNodes);
            hostAlloc(nodeGrid.heatData.heatSinkSource, nrNodes);
        }
        nodeGrid.isInitialized = true;

        //initialize the solver pointer
        #ifdef CUDA_ENABLED
            if(CUDAactive)
            {
                GPUSolver* tmpPtr = nullptr;
                cudaMallocManaged(&tmpPtr, sizeof(GPUSolver));
                new (tmpPtr) GPUSolver(GPUSolverObject);
                solver = tmpPtr;
            }
            else
                solver = &(CPUSolverObject);
        #else
            solver = &(CPUSolverObject);
        #endif

        SF3Derror_t solverResult = solver->initialize();
        if(solverResult != SF3Derror_t::SF3Dok)
            return solverResult;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief initializes the balance variables
     *  \return Ok/Error
    */
    SF3Derror_t initializeBalance()
    {
        SF3Derror_t status;
        status = initializeWaterBalance();
        if(status != SF3Derror_t::SF3Dok)
            return status;

        if(simulationFlags.computeHeat)
            status = initializeHeatBalance();
        else
            balanceDataWholePeriod.heatMBR = 1.;    //Why?

        return status;
    }

    /*!
     *  \brief initializes the log data structures if enabled
     *  \return Ok/Error
    */
    SF3Derror_t initializeLog([[maybe_unused]] const std::string& logPath, [[maybe_unused]] const std::string& projectName)
    {
        #ifdef MCR_ENABLED
            SF3Derror_t logResult = initializeLogData(logPath, projectName);
            if(logResult != SF3Derror_t::SF3Dok)
                return logResult;
        #endif

        return SF3Derror_t::SF3Dok;
    }

        /*!
     *  \brief cleans all the data structures
        \return Ok/Error
    */
    SF3Derror_t cleanSF3D()
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::SF3Dok;

        //Topology data
        hostFree(nodeGrid.size);
        hostFree(nodeGrid.x);
        hostFree(nodeGrid.y);
        hostFree(nodeGrid.z);
        hostFree(nodeGrid.surfaceFlag);

        //Soil/Surface data
        hostFree(nodeGrid.soilSurfacePointers);

        //Boundary data
        hostFree(nodeGrid.boundaryData.boundaryType);
        hostFree(nodeGrid.boundaryData.boundarySlope);
        hostFree(nodeGrid.boundaryData.boundarySize);
        hostFree(nodeGrid.boundaryData.waterFlowRate);
        hostFree(nodeGrid.boundaryData.waterFlowSum);
        hostFree(nodeGrid.boundaryData.prescribedWaterPotential);
        hostFree(nodeGrid.boundaryData.heightWind);
        hostFree(nodeGrid.boundaryData.heightTemperature);
        hostFree(nodeGrid.boundaryData.roughnessHeight);
        hostFree(nodeGrid.boundaryData.aerodynamicConductance);
        hostFree(nodeGrid.boundaryData.soilConductance);
        hostFree(nodeGrid.boundaryData.temperature);
        hostFree(nodeGrid.boundaryData.relativeHumidity);
        hostFree(nodeGrid.boundaryData.windSpeed);
        hostFree(nodeGrid.boundaryData.netIrradiance);
        hostFree(nodeGrid.boundaryData.sensibleFlux);
        hostFree(nodeGrid.boundaryData.latentFlux);
        hostFree(nodeGrid.boundaryData.radiativeFlux);
        hostFree(nodeGrid.boundaryData.advectiveHeatFlux);
        hostFree(nodeGrid.boundaryData.fixedTemperatureValue);
        hostFree(nodeGrid.boundaryData.fixedTemperatureDepth);

        //Link data
        hostFree(nodeGrid.numLateralLink);
        for(u8_t linkIdx = 0; linkIdx < maxTotalLink; ++linkIdx)
        {
            hostFree(nodeGrid.linkData[linkIdx].linkType);
            hostFree(nodeGrid.linkData[linkIdx].linkIndex);
            hostFree(nodeGrid.linkData[linkIdx].interfaceArea);

            hostFree(nodeGrid.linkData[linkIdx].waterFlowSum);
            hostFree(nodeGrid.linkData[linkIdx].waterFlux);
            hostFree(nodeGrid.linkData[linkIdx].vaporFlux);
            for(u8_t fluxIdx = 0; fluxIdx < numTotalFluxTypes; ++fluxIdx)
                hostFree(nodeGrid.linkData[linkIdx].fluxes[fluxIdx]);
        }

        //Water data
        hostFree(nodeGrid.waterData.saturationDegree);
        hostFree(nodeGrid.waterData.waterConductivity);
        hostFree(nodeGrid.waterData.waterFlow);
        hostFree(nodeGrid.waterData.pressureHead);
        hostFree(nodeGrid.waterData.waterSinkSource);
        hostFree(nodeGrid.waterData.pond);
        hostFree(nodeGrid.waterData.oldPressureHead);
        hostFree(nodeGrid.waterData.bestPressureHead);
        hostFree(nodeGrid.waterData.invariantFluxes);
        hostFree(nodeGrid.waterData.partialCourantWaterLevels);

        //Culvert pointers
        hostFree(nodeGrid.culvertPtr);

        //Heat data
        hostFree(nodeGrid.heatData.temperature);
        hostFree(nodeGrid.heatData.oldTemperature);
        hostFree(nodeGrid.heatData.heatFlux);
        hostFree(nodeGrid.heatData.heatSinkSource);

        nodeGrid.isInitialized = false;

        //Clear the soil/surface data
        soilList.clear();
        surfaceList.clear();

        //Clean the solver
        SF3Derror_t solverResult = solver->clean();
        if(solverResult != SF3Derror_t::SF3Dok)
            return solverResult;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief initializes the log data structures if enabled
     *  \return Ok/Error
    */
    SF3Derror_t closeLog()
    {
        #ifdef MCR_ENABLED
            SF3Derror_t logResult = writeLogFile();
            if(logResult != SF3Derror_t::SF3Dok)
                return logResult;
        #endif

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief initializes the heat flags
     *  \return Ok/Error
    */
    SF3Derror_t initializeHeatFlag(heatFluxSaveMode_t saveModeHeat, bool isComputeAdvectiveFlux, bool isComputeLatentHeat)
    {
        simulationFlags.HFsaveMode = saveModeHeat;
        simulationFlags.computeHeatAdvection = isComputeAdvectiveFlux;
        simulationFlags.computeHeatVapor = isComputeLatentHeat;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief sets number of threads for parallel computing.
     *          if nrThreads < 1 or too large, hardware_concurrency get the number of logical processors
        \return setted number of threads
    */
    // TODO enableOmp
    u32_t setThreadsNumber(u32_t nrThreads)
    {
        u32_t nrHWthreads = std::thread::hardware_concurrency();
        if (nrThreads < 1 || nrThreads > nrHWthreads)
            nrThreads = nrHWthreads;

        SolverParametersPartial paramTemp;
        paramTemp.numThreads = nrThreads;
        if(solver)
            solver->updateParameters(paramTemp);

        //Versione c++20
        //solver->updateParameters(SolverParametersPartial{.numThreads = nrThreads});
        return nrThreads;
    }

    /*!
     * \brief sets the soil properties of the nrSoil-nrHorizon soil type
     * \param VG_alpha  [m-1]       Van Genutchen alpha parameter (warning: usually is kPa-1 in literature)
     * \param VG_n      [-]         Van Genutchen n parameter (1, 10]
     * \param VG_m      [-]         Van Genutchen m parameter (0, 1)
     * \param VG_he     [m]         Van Genutchen air-entry potential for modified formulation
     * \param ThetaR    [m3 m-3]    residual water content
     * \param ThetaS    [m3 m-3]    saturated water content
     * \param Ksat      [m s-1]     saturated hydraulic conductivity
     * \param L         [-]         tortuosity (Mualem equation)
     * \return Ok/Error
     */
    SF3Derror_t setSoilProperties(u16_t nrSoil, u16_t nrHorizon, double VG_alpha, double VG_n, double VG_m, double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay)
    {
        if (VG_alpha <= 0 || (ThetaR < 0) || (ThetaR >= 1) || (ThetaS <= 0) || (ThetaS > 1) || (ThetaR > ThetaS))
            return SF3Derror_t::ParameterError;

        //Check duplicato
        for(const auto& soil : soilList)
            if(soil.soilNumber == nrSoil && soil.horizonNumber == nrHorizon)
                return SF3Derror_t::ParameterError;

        //Creazione del nuovo elemento
        soilData_t currSoil;
        currSoil.VG_alpha = VG_alpha;
        currSoil.VG_n = VG_n;
        currSoil.VG_m = VG_m;
        currSoil.VG_he = VG_he;
        currSoil.VG_Sc = std::pow(1. + std::pow(VG_alpha * VG_he, VG_n), -VG_m);
        currSoil.Theta_r = ThetaR;
        currSoil.Theta_s = ThetaS;
        currSoil.K_sat = Ksat;
        currSoil.Mualem_L = L;
        currSoil.organicMatter = organicMatter;
        currSoil.clay = clay;

        soilList.push_back(currSoil);

        //Set dell'indice
        if(nrSoil >= soil1DIndeces.size())
            soil1DIndeces.resize(nrSoil + 1);

        if(nrHorizon >= soil1DIndeces[nrSoil].size())
            soil1DIndeces[nrSoil].resize(nrHorizon + 1);

        soil1DIndeces[nrSoil][nrHorizon] = static_cast<u16_t>(soilList.size() - 1);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the surface properties of the surfaceIndex surface type
     * \param roughness [s m-1/3]   Manning roughness
     * \return Ok/Error
     */
    SF3Derror_t setSurfaceProperties(u16_t surfaceIndex, double roughness)
    {
        if(roughness < 0)
            return SF3Derror_t::ParameterError;

        if(surfaceIndex >= surfaceList.size())
            surfaceList.resize(surfaceIndex + 1);

        surfaceList[surfaceIndex].roughness = roughness;
        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief sets numerical parameters of the solver
     *  \return Ok/Error
    */
    SF3Derror_t setNumericalParameters(double minDeltaT, double maxDeltaT, u16_t maxIterationNumber, u16_t maxApproximationsNumber, u8_t ResidualToleranceExponent, u8_t MBRThresholdExponent)
    {
        if (minDeltaT < 0.01)
            minDeltaT = 0.01;           // [s]
        if (minDeltaT > HOUR_SECONDS)
            minDeltaT = HOUR_SECONDS;

        if (maxDeltaT < 60)
            maxDeltaT = 60;             // [s]
        if (maxDeltaT > HOUR_SECONDS)
            maxDeltaT = HOUR_SECONDS;
        if (maxDeltaT < minDeltaT)
            maxDeltaT = minDeltaT;

        if (maxIterationNumber < 10)
            maxIterationNumber = 10;
        if (maxIterationNumber > MAX_NUMBER_ITERATIONS)
            maxIterationNumber = MAX_NUMBER_ITERATIONS;

        if (maxApproximationsNumber < 1)
            maxApproximationsNumber = 1;
        if (maxApproximationsNumber > MAX_NUMBER_APPROXIMATIONS)
            maxApproximationsNumber = MAX_NUMBER_APPROXIMATIONS;

        if (ResidualToleranceExponent < 5)
            ResidualToleranceExponent = 5;
        if (ResidualToleranceExponent > 16)
            ResidualToleranceExponent = 16;

        if (MBRThresholdExponent < 1)
            MBRThresholdExponent = 1;
        if (MBRThresholdExponent > 6)
            MBRThresholdExponent = 6;

        SolverParametersPartial paramTemp;
        paramTemp.MBRThreshold = std::pow(10.0, -MBRThresholdExponent);
        paramTemp.residualTolerance = std::pow(10.0, -ResidualToleranceExponent);
        paramTemp.deltaTmin = minDeltaT;
        paramTemp.deltaTmax = maxDeltaT;
        paramTemp.deltaTcurr = maxDeltaT;
        paramTemp.maxApproximationsNumber = maxApproximationsNumber;
        paramTemp.maxIterationsNumber = maxIterationNumber;
        solver->updateParameters(paramTemp);

        //Versione c++20
        // solver->updateParameters(SolverParametersPartial{.MBRThreshold = std::pow(10.0, -MBRThresholdExponent),
        //                                                  .residualTolerance = std::pow(10.0, -ResidualToleranceExponent),
        //                                                  .deltaTmin = minDeltaT, .deltaTmax = maxDeltaT, .deltaTcurr = maxDeltaT,
        //                                                  .maxApproximationsNumber = maxApproximationsNumber, .maxIterationsNumber = maxIterationNumber});

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the hydraulic proprerties used by the solver
     *      default values:
     *          waterRetentionCurve = ModifiedVanGenuchten
     *          conductivityMeanType = Logarithmic
     *          conductivityHorizVertRatio = 10.0
     * \return OK/Error
     */
    SF3Derror_t setHydraulicProperties(WRCModel waterRetentionCurve, meanType_t conductivityMeanType, float conductivityHorizVertRatio)
    {
        if((conductivityHorizVertRatio < 0.1) || (conductivityHorizVertRatio > 100))
            return SF3Derror_t::ParameterError;

        SolverParametersPartial paramTemp;
        paramTemp.waterRetentionCurveModel = waterRetentionCurve;
        paramTemp.meanType = conductivityMeanType;
        paramTemp.lateralVerticalRatio = conductivityHorizVertRatio;
        solver->updateParameters(paramTemp);

        //Versione c++20
        //solver->updateParameters(SolverParametersPartial{.waterRetentionCurveModel = waterRetentionCurve,
        //                                             .meantype = conductivityMeanType,
        //                                             .lateralVerticalRatio = conductivityHorizVertRatio});

        return SF3Derror_t::SF3Dok;
    }


    SF3Derror_t setCulvert(SF3Duint_t nodeIndex, double roughness, double slope, double width, double height)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes || !nodeGrid.surfaceFlag[nodeIndex])
            return SF3Derror_t::IndexError;

        //Update boundary condition
        setNodeBoundary(nodeIndex, boundaryType_t::Culvert, slope, width*height);

        //Obtain the culvertData_t pointer
        culvertData_t* culvertPtr = nullptr;
        for(std::size_t vIdx = 0; vIdx < culvertList.size(); ++vIdx)
        {
            culvertData_t& currCulvert = culvertList[vIdx];
            //Move to a index system for perfomance and flexibility
            if(currCulvert.roughness == roughness && currCulvert.width == width && currCulvert.height == height)
            {
                culvertPtr = &(currCulvert);
                break;
            }
        }

        if(!culvertPtr)
        {
            culvertData_t currCulvert;
            currCulvert.roughness = roughness;
            currCulvert.width = width;
            currCulvert.height = height;
            culvertList.push_back(currCulvert);
            culvertPtr = &(culvertList[culvertList.size() - 1]);
        }

        //Set the culvertData_t pointer
        nodeGrid.culvertPtr[nodeIndex] = culvertPtr;
        return SF3Derror_t::SF3Dok;

    }


    /*!
     *  \brief sets the principal data of the index node
     *  \return Ok/Error
    */
    SF3Derror_t setNode(SF3Duint_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope, double boundaryArea)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(index >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.x[index] = x;
        nodeGrid.y[index] = y;
        nodeGrid.z[index] = z;
        nodeGrid.size[index] = volume_or_area;

        nodeGrid.surfaceFlag[index] = isSurface;

        //Boundary data
        setNodeBoundary(index, boundaryType, slope, boundaryArea);

        if(simulationFlags.computeWater)
        {
            nodeGrid.waterData.pond[index] = isSurface ? 0.0001f : noDataD;
            nodeGrid.waterData.waterSinkSource[index] = 0.;
        }

        if(simulationFlags.computeHeat && !isSurface)
        {
            nodeGrid.heatData.temperature[index] = static_cast<double>(ZEROCELSIUS + 20);
            nodeGrid.heatData.oldTemperature[index] = static_cast<double>(ZEROCELSIUS + 20);
            nodeGrid.heatData.heatFlux[index] = 0.;
            nodeGrid.heatData.heatSinkSource[index] = 0.;
        }

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief sets the data of the link from nodeIndex to linkIndex nodes
     *  \return Ok/Error
    */
    SF3Derror_t setNodeLink(SF3Duint_t nodeIndex, SF3Duint_t linkIndex, linkType_t direction, double interfaceArea)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes || linkIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        u8_t idx;
        switch (direction)
        {
            case linkType_t::Up:
                idx = 0;
                break;
            case linkType_t::Down:
                idx = 1;
                break;
            case linkType_t::Lateral:
                if(nodeGrid.numLateralLink[nodeIndex] == maxLateralLink)
                    return SF3Derror_t::TopographyError;
                idx = 2 + nodeGrid.numLateralLink[nodeIndex];
                nodeGrid.numLateralLink[nodeIndex]++;
                break;
            default:
                return SF3Derror_t::ParameterError;
        }
        nodeGrid.linkData[idx].linkType[nodeIndex] = direction;
        nodeGrid.linkData[idx].linkIndex[nodeIndex] = linkIndex;
        nodeGrid.linkData[idx].interfaceArea[nodeIndex] = interfaceArea;

        if(simulationFlags.computeWater)
            nodeGrid.linkData[idx].waterFlowSum[nodeIndex] = 0.;

        if(simulationFlags.computeHeat)
        {
            nodeGrid.linkData[idx].waterFlux[nodeIndex] = 0.;
            nodeGrid.linkData[idx].vaporFlux[nodeIndex] = 0.;

            nodeGrid.linkData[idx].fluxes[0][nodeIndex] = noDataD;
            if(simulationFlags.HFsaveMode == heatFluxSaveMode_t::All)
                for(u8_t fluxIdx = 1; fluxIdx < numTotalFluxTypes; ++fluxIdx)
                    nodeGrid.linkData[idx].fluxes[fluxIdx][nodeIndex] = noDataD;
        }

        //TO DO: if(computeSolutes){}

        return SF3Derror_t::SF3Dok;
    }

    /*!
     *  \brief sets the nodeIndex node boundary data
     *  \return Ok/Error
    */
    SF3Derror_t setNodeBoundary(SF3Duint_t nodeIndex, boundaryType_t boundaryType, double slope, double boundaryArea)
    {
        nodeGrid.boundaryData.boundaryType[nodeIndex] = boundaryType;
        if(boundaryType == boundaryType_t::NoBoundary)
            return SF3Derror_t::SF3Dok;

        nodeGrid.boundaryData.boundarySlope[nodeIndex] = slope;
        nodeGrid.boundaryData.boundarySize[nodeIndex] = boundaryArea;

        if(simulationFlags.computeWater)
        {
            nodeGrid.boundaryData.waterFlowRate[nodeIndex] = 0.;
            nodeGrid.boundaryData.waterFlowSum[nodeIndex] = 0.;
            nodeGrid.boundaryData.prescribedWaterPotential[nodeIndex] = noDataD;
        }

        if(simulationFlags.computeHeat)
        {
            nodeGrid.boundaryData.heightWind[nodeIndex] = noDataD;
            nodeGrid.boundaryData.heightTemperature[nodeIndex] = noDataD;
            nodeGrid.boundaryData.roughnessHeight[nodeIndex] = noDataD;
            nodeGrid.boundaryData.aerodynamicConductance[nodeIndex] = noDataD;
            nodeGrid.boundaryData.soilConductance[nodeIndex] = noDataD;
            nodeGrid.boundaryData.temperature[nodeIndex] = noDataD;
            nodeGrid.boundaryData.relativeHumidity[nodeIndex] = noDataD;
            nodeGrid.boundaryData.windSpeed[nodeIndex] = noDataD;
            nodeGrid.boundaryData.netIrradiance[nodeIndex] = noDataD;
            nodeGrid.boundaryData.radiativeFlux[nodeIndex] = 0.;
            nodeGrid.boundaryData.latentFlux[nodeIndex] = 0.;
            nodeGrid.boundaryData.sensibleFlux[nodeIndex] = 0.;
            nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex] = 0.;
            nodeGrid.boundaryData.fixedTemperatureValue[nodeIndex] = noDataD;
            nodeGrid.boundaryData.fixedTemperatureDepth[nodeIndex] = noDataD;
        }

        return SF3Derror_t::SF3Dok;
    }


    /*!
     * \brief sets the soil data of the subsurface nodeIndex node
     * \param nodeIndex index of the node
     * \param soilIndex, horizonIndex indeces of the soil type in the soil list
     * \return Ok/Error
     */
    SF3Derror_t setNodeSoil(SF3Duint_t nodeIndex, u16_t soilIndex, u16_t horizonIndex)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return SF3Derror_t::IndexError;

        if(soilIndex >= soil1DIndeces.size() || horizonIndex >= soil1DIndeces[soilIndex].size())
            return SF3Derror_t::ParameterError;

        nodeGrid.soilSurfacePointers[nodeIndex].soilPtr = &(soilList[soil1DIndeces[soilIndex][horizonIndex]]);
        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the surface data of the surface nodeIndex node
     * \param nodeIndex index of the node
     * \param surfaceIndex index of the surface type in the surface list
     * \return Ok/Error
     */
    SF3Derror_t setNodeSurface(SF3Duint_t nodeIndex, u16_t surfaceIndex)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(surfaceIndex >= surfaceList.size())
            return SF3Derror_t::ParameterError;

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return SF3Derror_t::IndexError;      //surfaceFlags must be initialized before soil data

        nodeGrid.soilSurfacePointers[nodeIndex].surfacePtr = &(surfaceList[surfaceIndex]);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex surface node's pond data
     * \param nodeIndex
     * \param pond            maximum pond height [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodePond(SF3Duint_t nodeIndex, double pond)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return SF3Derror_t::IndexError;

        nodeGrid.waterData.pond[nodeIndex] = pond;
        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node water content and updates node pressure head and saturation degree accordingly
     * \param waterContent  [m] surface - [m3 m-3] sub-surface
     * \return Ok/Error
     */
    SF3Derror_t setNodeWaterContent(SF3Duint_t nodeIndex, double waterContent)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(waterContent < 0.)
            return SF3Derror_t::ParameterError;

        if(nodeGrid.surfaceFlag[nodeIndex])
        {
            nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] + waterContent;
            nodeGrid.waterData.oldPressureHead[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
            nodeGrid.waterData.saturationDegree[nodeIndex] = 1.;
            nodeGrid.waterData.waterConductivity[nodeIndex] = 0.;
        }
        else
        {
            if(waterContent > 1.)
                return SF3Derror_t::ParameterError;

            nodeGrid.waterData.saturationDegree[nodeIndex] = computeNodeSe_fromTheta(nodeIndex, waterContent);
            nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] - computeNodePsi(nodeIndex);
            nodeGrid.waterData.oldPressureHead[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
            nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);
        }

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node degree of saturation and updates node pressure head and water conducuctivy accordingly
     * \param degreeOfSaturation  [-] (only sub-surface)
     * \return Ok/Error
     */
    SF3Derror_t setNodeDegreeOfSaturation(SF3Duint_t nodeIndex, double degreeOfSaturation)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return SF3Derror_t::IndexError;

        if((degreeOfSaturation < 0.) || (degreeOfSaturation > 1.))
            return SF3Derror_t::ParameterError;

        nodeGrid.waterData.saturationDegree[nodeIndex] = degreeOfSaturation;
        nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] - computeNodePsi(nodeIndex);
        nodeGrid.waterData.oldPressureHead[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
        nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node pressure head based of the matric potential
     * \param matricPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodeMatricPotential(SF3Duint_t nodeIndex, double matricPotential)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] + matricPotential;
        nodeGrid.waterData.oldPressureHead[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];

        nodeGrid.waterData.saturationDegree[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? 1. : computeNodeSe(nodeIndex);
        nodeGrid.waterData.waterConductivity[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? noDataD : computeNodeK(nodeIndex);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node pressure head based of the total potential
     * \param totalPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodeTotalPotential(SF3Duint_t nodeIndex, double totalPotential)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.waterData.pressureHead[nodeIndex] = totalPotential;
        nodeGrid.waterData.oldPressureHead[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];

        nodeGrid.waterData.saturationDegree[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? 1. : computeNodeSe(nodeIndex);
        nodeGrid.waterData.waterConductivity[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? noDataD : computeNodeK(nodeIndex);

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary prescribed total potential
     * \param prescribedTotalPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodePrescribedTotalPotential(SF3Duint_t nodeIndex, double prescribedTotalPotential)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::PrescribedTotalWaterPotential)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.prescribedWaterPotential[nodeIndex] = prescribedTotalPotential;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node water sink source
     * \param waterSinkSource  [m3 sec-1]
     * \return Ok/Error
     */
    SF3Derror_t setNodeWaterSinkSource(SF3Duint_t nodeIndex, double waterSinkSource)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.waterData.waterSinkSource[nodeIndex] = waterSinkSource;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief gets the nodeIndex node water content
     * \return surface water level [m] if surface, volumetric water content [m3 m-3] if subsurface
     */
    double getNodeWaterContent(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.surfaceFlag[nodeIndex] ? (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex])
                                               : computeNodeTheta(nodeIndex);
    }

    /*!
     * \brief gets the nodeIndex sub-surface node maximum volumetric water content
     * \return theta_sat (maximum volumetric water content) [m3 m-3]
     */
    double getNodeMaximumWaterContent(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(nodeGrid.surfaceFlag[nodeIndex])
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.soilSurfacePointers[nodeIndex].soilPtr->Theta_s;
    }

    /*!
     * \brief gets the nodeIndex node available water content
     * \return surface water level [m] if surface, volumetric water content [m3 m-3] if subsurface
     */
    double getNodeAvailableWaterContent(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.surfaceFlag[nodeIndex] ? (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex])
                                               : SF3Dmax(0., computeNodeTheta(nodeIndex) - computeNodeTheta_fromSignedPsi(nodeIndex, -160));
    }

    /*!
     * \brief gets the nodeIndex node water deficit
     * \param fieldCapacity //TO ASK: what is this?
     * \return water deficit    [m3 m-3] (0 at surface)
     */
    double getNodeWaterDeficit(SF3Duint_t nodeIndex, double fieldCapacity)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(nodeGrid.surfaceFlag[nodeIndex])
            return 0.;

        return computeNodeTheta_fromSignedPsi(nodeIndex, -fieldCapacity) - computeNodeTheta(nodeIndex);
    }

    /*!
     * \brief gets the nodeIndex node degree of saturation
     * \note for surface node return 0 if there is no water and 1 if water > 0.001
     * \return degree of saturation [-]
     */
    double getNodeDegreeOfSaturation(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return nodeGrid.waterData.saturationDegree[nodeIndex];

        double curPot = nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex];
        double maxPot = 0.001;       // [m]

        return curPot <= 0 ? 0 : (curPot > maxPot ? 1. : curPot/maxPot);
    }

    /*!
     * \brief gets the nodeIndex node water conductivity
     * \return degree of saturation [-]
     */
    double getNodeWaterConductivity(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.waterData.waterConductivity[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node signed matric potential
     * \return matric potential     [m]
     */
    double getNodeMatricPotential(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node signed total potential
     * \return total potential     [m]
     */
    double getNodeTotalPotential(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.waterData.pressureHead[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex surface node pond
     * \return surface pond     [m3]
     */
    double getNodePond(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        return nodeGrid.waterData.pond[nodeIndex];
    }

    /*!
     * \brief gets the maximum integrated water flow fromnodeInxed node in the linkDirection direction
     * \param linkDirection    [Up/Down/Lateral]
     * \return integrated water flow     [m3]
     */
    double getNodeMaxWaterFlow(SF3Duint_t nodeIndex, linkType_t linkDirection)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        double maxFlow = 0.;
        switch (linkDirection)
        {
            case linkType_t::Up:
                if(nodeGrid.linkData[0].linkIndex[nodeIndex] == noDataU)
                    return getDoubleErrorValue(SF3Derror_t::IndexError);

                return nodeGrid.linkData[0].waterFlowSum[nodeIndex];
            case linkType_t::Down:
                if(nodeGrid.linkData[1].linkIndex[nodeIndex] == noDataU)
                    return getDoubleErrorValue(SF3Derror_t::IndexError);

                return nodeGrid.linkData[1].waterFlowSum[nodeIndex];
            case linkType_t::Lateral:
                for(u8_t linkIdx = 0; linkIdx < nodeGrid.numLateralLink[nodeIndex]; ++linkIdx)
                    if(nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != noDataU)
                        maxFlow = SF3Dmax(maxFlow, nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex]);

                return maxFlow;
            default:
                return getDoubleErrorValue(SF3Derror_t::IndexError);
        }
    }

    /*!
     * \brief gets the total integrated lateral water flow from nodeIndex node
     * \return integrated water flow     [m3]
     */
    double getNodeSumLateralWaterFlow(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        double sumFlow = 0.;
        for(u8_t linkIdx = 0; linkIdx < nodeGrid.numLateralLink[nodeIndex]; ++linkIdx)
            if(nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != noDataU)
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated lateral water inflow from nodeIndex node
     * \return integrated water inflow     [m3]
     */
    double getNodeSumLateralWaterFlowIn(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        double sumFlow = 0.;
        for(u8_t linkIdx = 0; linkIdx < nodeGrid.numLateralLink[nodeIndex]; ++linkIdx)
            if((nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != noDataU) && (nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex] > 0))
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated lateral water outflow from nodeIndex node
     * \return integrated water outflow     [m3]
     */
    double getNodeSumLateralWaterFlowOut(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        double sumFlow = 0.;
        for(u8_t linkIdx = 0; linkIdx < nodeGrid.numLateralLink[nodeIndex]; ++linkIdx)
            if((nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != noDataU) && (nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex] < 0))
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated boundary water flow from nodeIndex node
     * \return integrated boundary water flow     [m3]
     */
    double getNodeBoundaryWaterFlow(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.waterFlowSum[nodeIndex];
    }

    /*!
     * \brief gets the total integrated boundaryType boundary water flow
     * \return integrated boundaryType boundary water flow     [m3]
     */
    double getTotalBoundaryWaterFlow(boundaryType_t boundaryType)
    {
        double totalBoundaryWaterFlow = 0.0;

        __parforop(__ompStatus, +, totalBoundaryWaterFlow)
        for (SF3Duint_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
            if (nodeGrid.boundaryData.boundaryType[nodeIdx] == boundaryType)
                totalBoundaryWaterFlow += nodeGrid.boundaryData.waterFlowSum[nodeIdx];

        return totalBoundaryWaterFlow;
    }

    /*!
     * \brief gets the total water content
     * \return total water content  [m3]
     */
    double getTotalWaterContent()
    {
        return computeTotalWaterContent();
    }

    /*!
     * \brief gets the total water storage
     * \return water storage    [m3]
     */
    double getWaterStorage()
    {
        return balanceDataCurrentTimeStep.waterStorage;
    }

    /*!
     * \brief gets the water mass balance ratio
     * \return mass balance ratio   [m3 m-3]
     */
    double getWaterMBR()
    {
        return balanceDataWholePeriod.waterMBR;
    }

    /*!
     * \brief sets the nodeIndex node heat sink source
     * \return Ok/Error
     */
    SF3Derror_t setNodeHeatSinkSource(SF3Duint_t nodeIndex, double heatSinkSource)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.heatData.heatSinkSource[nodeIndex] = heatSinkSource;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node temperature
     * \return Ok/Error
     */
    SF3Derror_t setNodeTemperature(SF3Duint_t nodeIndex, double temperature)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        nodeGrid.heatData.temperature[nodeIndex] = temperature;
        nodeGrid.heatData.oldTemperature[nodeIndex] = temperature;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary fixed temperature
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryFixedTemperature(SF3Duint_t nodeIndex, double fixedTemperature, double depth)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::PrescribedTotalWaterPotential
            && nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::FreeDrainage)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.fixedTemperatureValue[nodeIndex] = fixedTemperature;
        nodeGrid.boundaryData.fixedTemperatureDepth[nodeIndex] = depth;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary height wind
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryHeightWind(SF3Duint_t nodeIndex, double heightWind)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.heightWind[nodeIndex] = heightWind;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary height temperature
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryHeightTemperature(SF3Duint_t nodeIndex, double heightTemperature)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.heightTemperature[nodeIndex] = heightTemperature;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary net irradiance
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryNetIrradiance(SF3Duint_t nodeIndex, double netIrradiance)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.netIrradiance[nodeIndex] = netIrradiance;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary temperature
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryTemperature(SF3Duint_t nodeIndex, double temperature)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.temperature[nodeIndex] = temperature;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary relative humidity
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryRelativeHumidity(SF3Duint_t nodeIndex, double relativeHumidity)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        nodeGrid.boundaryData.relativeHumidity[nodeIndex] = relativeHumidity;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary roughness
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryRoughness(SF3Duint_t nodeIndex, double roughness)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        if(roughness < 0)
            return SF3Derror_t::ParameterError;

        nodeGrid.boundaryData.roughnessHeight[nodeIndex] = roughness;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary wind speed
     * \return Ok/Error
     */
    SF3Derror_t setNodeBoundaryWindSpeed(SF3Duint_t nodeIndex, double windSpeed)
    {
        if(!nodeGrid.isInitialized)
            return SF3Derror_t::MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return SF3Derror_t::IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == boundaryType_t::NoBoundary)
            return SF3Derror_t::BoundaryError;

        if((windSpeed < 0.) || (windSpeed > 1000.))
            return SF3Derror_t::ParameterError;

        nodeGrid.boundaryData.windSpeed[nodeIndex] = windSpeed;

        return SF3Derror_t::SF3Dok;
    }

    /*!
     * \brief gets the nodeIndex node temperature
     * \return temperature      [K]
     */
    double getNodeTemperature(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!isHeatNode(nodeIndex))
            return getDoubleErrorValue(SF3Derror_t::TopographyError);

        return nodeGrid.heatData.temperature[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node heat conductivity
     * \return node heat conductivity       [W m-1 K-1]
     */
    double getNodeHeatConductivity(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!isHeatNode(nodeIndex))
            return getDoubleErrorValue(SF3Derror_t::TopographyError);

        return computeNodeHeatSoilConductivity(nodeIndex, nodeGrid.heatData.temperature[nodeIndex],
                                            nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]);
    }

    /*!
     * \brief gets the nodeIndex node vapor concentration
     * \return node vapor concentration     [kg m-3]
     */
    __cudaSpec double getNodeVapor(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeWater || !simulationFlags.computeHeat || !simulationFlags.computeHeatVapor)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        double h = nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex];
        double T = nodeGrid.heatData.temperature[nodeIndex];

        return computeVapor_fromPsiTemp(h, T);
    }

    /*!
     * \brief gets the nodeIndex node heat storage
     * \param pressureHead  [m]
     * \return heat storage [J]
     */
    double getNodeHeatStorage(SF3Duint_t nodeIndex, double h)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeHeat)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        double nodeT = nodeGrid.heatData.temperature[nodeIndex];
        double nodeSize = nodeGrid.size[nodeIndex];
        double nodeHeat = computeNodeHeatCapacity(nodeIndex, h, nodeT) * nodeSize * nodeT;

        if(simulationFlags.computeWater && simulationFlags.computeHeatVapor)
        {
            double nodeThetaV = computeNodeVaporThetaV(nodeIndex, h, nodeT);
            nodeHeat += nodeThetaV * computeLatentVaporizationHeat(nodeT - ZEROCELSIUS) * WATER_DENSITY * nodeSize;
        }

        return nodeHeat;
    }

    /*!
     * \brief gets the nodeIndex node instantaneous heat flux/water flux
     * \param linkDirection     [Up/Down/Lateral]
     * \param fluxType          [HeatTotal/HeatDiffusive/HeatLatentIsothermal/HeatLatentThermal/HeatAdvective/
     *                           WaterLiquidIsothermal/WaterLiquidThermal/WaterVaporIsothermal/WaterVaporThermal]
     * \return instantaneous heat flux  [W]
     * \return water flux               [m3 s-1]
     */
    double getNodeHeatMaxFlux(SF3Duint_t nodeIndex, linkType_t linkDirection, fluxTypes_t fluxType)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!isHeatNode(nodeIndex))
            return getDoubleErrorValue(SF3Derror_t::TopographyError);

        double nodeMaxFlux = 0.;
        switch(linkDirection)
        {
            case linkType_t::Up:
                return getLinkHeatFlux(nodeGrid.linkData[0], nodeIndex, fluxType);

            case linkType_t::Down:
                return getLinkHeatFlux(nodeGrid.linkData[1], nodeIndex, fluxType);

            case linkType_t::Lateral:
                for(u8_t linkIdx = 0; linkIdx < maxLateralLink; ++linkIdx)
                {
                    double currFlux = getLinkHeatFlux(nodeGrid.linkData[2 + linkIdx], nodeIndex, fluxType);
                    if(currFlux > std::fabs(nodeMaxFlux))
                        nodeMaxFlux = currFlux;
                }
                return nodeMaxFlux;

            default:
                return getDoubleErrorValue(SF3Derror_t::IndexError);
        }
    }

    /*!
     * \brief gets the nodeIndex node boundary heat advective flux
     * \return boundary advective heat flux     [W m-2]
     */
    double getNodeBoundaryAdvectiveFlux(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeWater || !simulationFlags.computeHeat || !simulationFlags.computeHeatVapor)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.advectiveHeatFlux[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node boundary heat latent flux
     * \return boundary latent heat flux     [W m-2]
     */
    double getNodeBoundaryLatentFlux(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeWater || !simulationFlags.computeHeat || !simulationFlags.computeHeatVapor)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.latentFlux[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node boundary heat radiative flux
     * \return boundary radiative heat flux     [W m-2]
     */
    double getNodeBoundaryRadiativeFlux(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeHeat)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.radiativeFlux[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node boundary heat sensible flux
     * \return boundary sensible heat flux     [W m-2]
     */
    double getNodeBoundarySensibleFlux(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeHeat)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.sensibleFlux[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node boundary heat aerodynamic conductance
     * \return boundary aerodynamic conductance     [m s-1]
     */
    double getNodeBoundaryAerodynamicConductance(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeHeat)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.aerodynamicConductance[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node boundary heat soil conductance
     * \return boundary soil conductance     [m s-1]
     */
    double getNodeBoundarySoilConductance(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.isInitialized)
            return getDoubleErrorValue(SF3Derror_t::MemoryError);

        if(nodeIndex >= nodeGrid.numNodes)
            return getDoubleErrorValue(SF3Derror_t::IndexError);

        if(!simulationFlags.computeHeat)
            return getDoubleErrorValue(SF3Derror_t::MissingDataError);

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != boundaryType_t::HeatSurface)
            return getDoubleErrorValue(SF3Derror_t::BoundaryError);

        return nodeGrid.boundaryData.soilConductance[nodeIndex];
    }

    /*!
     * \brief gets the heat mass balance ratio
     * \return heat mass balance ratio   [?]
     */
    double getHeatMBR()
    {
        return balanceDataWholePeriod.heatMBR;
    }

    /*!
     * \brief gets the heat mass balance error
     * \return heat mass balance error   [?]
     */
    double getHeatMBE()
    {
        return balanceDataWholePeriod.heatMBE;
    }

    /*!
     * \brief compute simulation for a specific time period
     * \details compute water and heat fluxes for a time period (maximum 1 hour) assiming constant meteo conditions
     * \param timePeriod     [s]
     */
    void computePeriod(double timePeriod)
    {
        double sumCurrentTime = 0.;

        balanceDataCurrentPeriod.waterSinkSource = 0.;
        balanceDataCurrentPeriod.heatSinkSource = 0.;

        while(sumCurrentTime < timePeriod)
            sumCurrentTime += computeStep(timePeriod - sumCurrentTime);

        if(simulationFlags.computeWater)
            updateWaterBalanceDataWholePeriod();

        if(simulationFlags.computeHeat)
            updateHeatBalanceDataWholePeriod();

        return;
    }

    /*!
     * \brief compute one simulation step
     * \details compute one step for active fluxes assuming constant meteo conditions
     * \param maxTimeStep       [s] (default HOUR_SECONDS = 3600)
     * \return computedTimeStep [s]
     */
    double computeStep(double maxTimeStep)
    {
        if(simulationFlags.computeHeat)
        {
            resetFluxValues(false, true);
            updateConductance();
        }

        double dtWater, dtHeat;

        if(simulationFlags.computeWater)
            solver->run(maxTimeStep, dtWater, processType::Water);
        else
            dtWater = SF3Dmin(maxTimeStep, solver->getMaxTimeStep());

        if(simulationFlags.computeHeat)
        {
            dtHeat = dtWater;
            saveWaterFluxValues(dtHeat, dtWater);

            double dtHeatAccumulator = 0.;
            while(dtHeatAccumulator < dtWater)
            {
                dtHeat = std::min(dtHeat, dtWater - dtHeatAccumulator);

                double reducedTimeStep;
                while(!updateBoundaryHeatData(dtHeat, reducedTimeStep))
                    dtHeat = reducedTimeStep;

                solver->run(dtHeat, dtWater, processType::Heat);

                dtHeatAccumulator += dtHeat;
            }
        }
        return dtWater;
    }

} //namespace
