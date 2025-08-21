#include <algorithm>
#include <iostream>

#include "commonConstants.h"
#include "soilFluxes3DNew.h"
#include "solver.h"
#include "soilPhysics.h"
#include "water.h"

#ifdef CUDA_ENABLED
    #include "gpusolver.h"
#endif
#include "cpusolver.h"

using namespace soilFluxes3D::Soil;
using namespace soilFluxes3D::Water;

namespace soilFluxes3D::New
{
    #ifdef CUDA_ENABLED
        GPUSolver GPUSolverObject;
        bool CUDAactive = true;
    #endif

    CPUSolver CPUSolverObject;

    //global variables
    __cudaMngd Solver* solver = nullptr;
    __cudaMngd nodesData_t nodeGrid;
    __cudaMngd simulationFlags_t simulationFlags;
    __cudaMngd balanceData_t balanceDataCurrentPeriod, balanceDataWholePeriod, balanceDataCurrentTimeStep, balanceDataPreviousTimeStep;

    std::vector<std::vector<soilData_t>> soilList;
    std::vector<surfaceData_t> surfaceList;

    /*!
     *  \brief inizializes the node data grid and set simulation parameters
     *  \return Ok/Error
    */
    SF3Derror_t initializeSF3D(uint64_t nrNodes, uint16_t nrLayers, uint8_t nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes)
    {
        //Cleans all the data structures
        SF3Derror_t cleanResult = cleanSF3D();
        if(cleanResult != SF3Dok)
            return cleanResult;

        simulationFlags.computeWater = isComputeWater;
        simulationFlags.computeHeat = isComputeHeat;
        if(isComputeHeat)
        {
            simulationFlags.computeHeatVapor = true;
            simulationFlags.computeHeatAdvection = true;
        }
        simulationFlags.computeSolutes = isComputeSolutes;

        nodeGrid.numNodes = nrNodes;
        nodeGrid.numLayers = nrLayers;

        //Check with a define value. In CRITERIA3D nrLateralLink is a compile-time constant
        if(nrLateralLinks > maxLateralLink)
            return ParameterError;


        //Inizialize data
        //Topology data
        hostAlloc(nodeGrid.size, double, nrNodes);
        hostAlloc(nodeGrid.x, double, nrNodes);
        hostAlloc(nodeGrid.y, double, nrNodes);
        hostAlloc(nodeGrid.z, double, nrNodes);
        hostAlloc(nodeGrid.surfaceFlag, bool, nrNodes);

        //Soil/Surface data
        hostAlloc(nodeGrid.soilRowIndeces, uint16_t, nrNodes);
        hostAlloc(nodeGrid.soilSurfacePointers, soil_surface_ptr, nrNodes);

        //Boundary data
        hostAlloc(nodeGrid.boundaryData.boundaryType, boundaryType_t, nrNodes); //NoBoundary is equal 0, automatic set with calloc
        hostAlloc(nodeGrid.boundaryData.boundarySlope, double, nrNodes);
        hostAlloc(nodeGrid.boundaryData.boundarySize, double, nrNodes);
        if(isComputeWater)      //TO DO: check if needs to be removed
        {
            hostAlloc(nodeGrid.boundaryData.waterFlowRate, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.waterFlowSum, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.prescribedWaterPotential, double, nrNodes);
        }
        if(isComputeHeat)
        {
            hostAlloc(nodeGrid.boundaryData.heightWind, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.heightTemperature, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.roughnessHeight, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.aerodynamicConductance, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.soilConductance, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.temperature, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.relativeHumidity, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.windSpeed, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.netIrradiance, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.sensibleFlux, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.latentFlux, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.radiativeFlux, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.advectiveHeatFlux, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.fixedTemperature, double, nrNodes);
            hostAlloc(nodeGrid.boundaryData.fixedTemperatureDepth, double, nrNodes);
        }

        //Link data
        hostAlloc(nodeGrid.numLateralLink, uint8_t, nrNodes);
        for(uint8_t idx = 0; idx < maxTotalLink; ++idx)
        {
            hostAlloc(nodeGrid.linkData[idx].linktype, linkType_t, nrNodes);    //NoLink is equal 0, automatic set with calloc
            hostAlloc(nodeGrid.linkData[idx].linkIndex, uint64_t, nrNodes);;
            hostAlloc(nodeGrid.linkData[idx].interfaceArea, double, nrNodes);
            if(isComputeWater)
                hostAlloc(nodeGrid.linkData[idx].waterFlowSum, double, nrNodes);
            //TO DO -- Heat and Solutes
        }


        //Water data //TO DO: check if needs to be moved under if
        hostAlloc(nodeGrid.waterData.saturationDegree, double, nrNodes);
        hostAlloc(nodeGrid.waterData.waterConductivity, double, nrNodes);
        hostAlloc(nodeGrid.waterData.waterFlow, double, nrNodes);
        hostAlloc(nodeGrid.waterData.pressureHead, double, nrNodes);
        hostAlloc(nodeGrid.waterData.waterSinkSource, double, nrNodes);
        hostAlloc(nodeGrid.waterData.pond, double, nrNodes);
        hostAlloc(nodeGrid.waterData.oldPressureHeads, double, nrNodes);
        hostAlloc(nodeGrid.waterData.bestPressureHeads, double, nrNodes);
        hostAlloc(nodeGrid.waterData.invariantFluxes, double, nrNodes);

        //Heat data
        if(isComputeHeat)
        {
            hostAlloc(nodeGrid.heatData.temperature, double, nrNodes);
            hostAlloc(nodeGrid.heatData.oldTemperature, double, nrNodes);
            hostAlloc(nodeGrid.heatData.heatFlux, double, nrNodes);
            hostAlloc(nodeGrid.heatData.heatSinkSource, double, nrNodes);
            //hostFill(nodeGrid.heatData.temperature, nrNodes, static_cast<double>(ZEROCELSIUS + 20));
            //hostFill(nodeGrid.heatData.oldTemperature, nrNodes, static_cast<double>(ZEROCELSIUS + 20));
        }
        nodeGrid.isInizialized = true;

        //Inizialize the solver pointer
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

        SF3Derror_t solverResult = solver->inizialize();
        if(solverResult != SF3Dok)
            return solverResult;

        return SF3Dok;
    }

    /*!
     *  \brief inizializes the balance variables
     *  \return Ok/Error
    */
    SF3Derror_t inizializeBalance()
    {
        SF3Derror_t status = initializeWaterBalance();
        if(status != SF3Dok)
            return status;

        if(simulationFlags.computeHeat)
            status; //= inizializeHeatBalance();    //TO DO
        else
            balanceDataWholePeriod.heatMBR = 1.;    //Why?

        return status;
    }

    /*!
     *  \brief cleans all the data structures
        \return Ok/Error
    */
    SF3Derror_t cleanSF3D()
    {
        if(!nodeGrid.isInizialized)
            return SF3Dok;

        //Topology data
        hostFree(nodeGrid.size);
        hostFree(nodeGrid.x);
        hostFree(nodeGrid.y);
        hostFree(nodeGrid.z);
        hostFree(nodeGrid.surfaceFlag);

        //Soil/Surface data
        hostFree(nodeGrid.soilRowIndeces);
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
        hostFree(nodeGrid.boundaryData.fixedTemperature);
        hostFree(nodeGrid.boundaryData.fixedTemperatureDepth);

        //Link data
        hostFree(nodeGrid.numLateralLink);
        for(uint8_t idx = 0; idx < maxTotalLink; ++idx)
        {
            hostFree(nodeGrid.linkData[idx].linktype);
            hostFree(nodeGrid.linkData[idx].linkIndex);
            hostFree(nodeGrid.linkData[idx].interfaceArea);
            hostFree(nodeGrid.linkData[idx].waterFlowSum);
            //TO DO -- Heat and Solutes
        }

        //Water data
        hostFree(nodeGrid.waterData.saturationDegree);
        hostFree(nodeGrid.waterData.waterConductivity);
        hostFree(nodeGrid.waterData.waterFlow);
        hostFree(nodeGrid.waterData.pressureHead);
        hostFree(nodeGrid.waterData.waterSinkSource);
        hostFree(nodeGrid.waterData.pond);
        hostFree(nodeGrid.waterData.oldPressureHeads);
        hostFree(nodeGrid.waterData.bestPressureHeads);
        hostFree(nodeGrid.waterData.invariantFluxes);

        //Heat data
        hostFree(nodeGrid.heatData.temperature);
        hostFree(nodeGrid.heatData.oldTemperature);
        hostFree(nodeGrid.heatData.heatFlux);
        hostFree(nodeGrid.heatData.heatSinkSource);

        nodeGrid.isInizialized = false;

        //Clean the solver
        SF3Derror_t solverResult = solver->clean();
        if(solverResult != SF3Dok)
            return solverResult;

        return SF3Dok;
    }

    /*!
     *  \brief sets number of threads for parallel computing.
     *          if nrThreads < 1 or too large, hardware_concurrency get the number of logical processors
        \return setted number of threads
    */
    uint32_t setThreadsNumber(uint32_t nrThreads)
    {
        uint32_t nrHWthreads = std::thread::hardware_concurrency();
        if (nrThreads < 1 || nrThreads > nrHWthreads)
            nrThreads = nrHWthreads;


        SolverParametersPartial paramTemp;
        paramTemp.numThreads = nrThreads;
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
    SF3Derror_t setSoilProperties(uint16_t nrSoil, uint16_t nrHorizon, double VG_alpha, double VG_n, double VG_m, double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay)
    {
        if (VG_alpha <= 0 || (ThetaR < 0) || (ThetaR >= 1) || (ThetaS <= 0) || (ThetaS > 1) || (ThetaR > ThetaS))
            return ParameterError;

        if(nrSoil >= soilList.size())
            soilList.resize(nrSoil + 1);

        if(nrHorizon >= soilList[nrSoil].size())
            soilList[nrSoil].resize(nrHorizon + 1);

        soilList[nrSoil][nrHorizon].VG_alpha = VG_alpha;
        soilList[nrSoil][nrHorizon].VG_n = VG_n;
        soilList[nrSoil][nrHorizon].VG_m = VG_m;

        soilList[nrSoil][nrHorizon].VG_he = VG_he;
        soilList[nrSoil][nrHorizon].VG_Sc = std::pow(1. + std::pow(VG_alpha * VG_he, VG_n), -VG_m);

        soilList[nrSoil][nrHorizon].Theta_r = ThetaR;
        soilList[nrSoil][nrHorizon].Theta_s = ThetaS;
        soilList[nrSoil][nrHorizon].K_sat = Ksat;
        soilList[nrSoil][nrHorizon].Mualem_L = L;

        soilList[nrSoil][nrHorizon].organicMatter = organicMatter;
        soilList[nrSoil][nrHorizon].clay = clay;

        return SF3Dok;
    }

    /*!
     * \brief sets the surface properties of the surfaceIndex surface type
     * \param roughness [s m-1/3]   Manning roughness
     * \return Ok/Error
     */
    SF3Derror_t setSurfaceProperties(uint16_t surfaceIndex, double roughness)
    {
        if(roughness < 0)
            return ParameterError;

        if(surfaceIndex >= surfaceList.size())
            surfaceList.resize(surfaceIndex + 1);

        surfaceList[surfaceIndex].roughness = roughness;
        return SF3Dok;
    }

    /*!
     *  \brief sets numerical parameters of the solver
     *  \return Ok/Error
    */
    SF3Derror_t setNumericalParameters(double minDeltaT, double maxDeltaT, uint16_t maxIterationNumber, uint16_t maxApproximationsNumber, uint8_t ResidualToleranceExponent, uint8_t MBRThresholdExponent)
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
        //solver->updateParameters(SolverParametersPartial{.MBRThreshold = std::pow(10.0, -MBRThresholdExponent),
        //                                             .residualTolerance = std::pow(10.0, -ResidualToleranceExponent),
        //                                             .deltaTmin = minDeltaT, .deltaTmax = maxDeltaT, .deltaTcurr = maxDeltaT,
        //                                             .maxApproximationsNumber = maxApproximationsNumber, .maxIterationsNumber = maxIterationNumber});

        return SF3Dok;
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
            return ParameterError;

        SolverParametersPartial paramTemp;
        paramTemp.waterRetentionCurveModel = waterRetentionCurve;
        paramTemp.meantype = conductivityMeanType;
        paramTemp.lateralVerticalRatio = conductivityHorizVertRatio;
        solver->updateParameters(paramTemp);

        //Versione c++20
        //solver->updateParameters(SolverParametersPartial{.waterRetentionCurveModel = waterRetentionCurve,
        //                                             .meantype = conductivityMeanType,
        //                                             .lateralVerticalRatio = conductivityHorizVertRatio});

        return SF3Dok;
    }

    /*!
     *  \brief sets the principal data of the index node
     *  \return Ok/Error
    */
    SF3Derror_t setNode(uint64_t index, double x, double y, double z, double volume_or_area, bool isSurface, boundaryType_t boundaryType, double slope, double boundaryArea)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(index >= nodeGrid.numNodes)
            return IndexError;

        nodeGrid.x[index] = x;
        nodeGrid.y[index] = y;
        nodeGrid.z[index] = z;
        nodeGrid.size[index] = volume_or_area;

        nodeGrid.surfaceFlag[index] = isSurface;

        nodeGrid.boundaryData.boundaryType[index] = boundaryType;
        if(boundaryType != NoBoundary)
        {
            nodeGrid.boundaryData.boundarySlope[index] = slope;
            nodeGrid.boundaryData.boundarySize[index] = boundaryArea;
            nodeGrid.boundaryData.prescribedWaterPotential[index] = noData;     //Check if 0 is okay
        }

        if(simulationFlags.computeWater)
        {
            nodeGrid.waterData.pond[index] = isSurface ? 0.0001f : noData;
            nodeGrid.waterData.waterSinkSource[index] = 0.;                     //Maybe useless: 0. is the value set by calloc
        }

        if(simulationFlags.computeHeat && !isSurface)
        {
            nodeGrid.heatData.temperature[index] = static_cast<double>(ZEROCELSIUS + 20);
            nodeGrid.heatData.oldTemperature[index] = static_cast<double>(ZEROCELSIUS + 20);
            nodeGrid.heatData.heatFlux[index] = 0.;                             //Maybe useless: 0. is the value set by calloc
            nodeGrid.heatData.heatSinkSource[index] = 0.;                       //Maybe useless: 0. is the value set by calloc
        }

        return SF3Dok;
    }

    /*!
     *  \brief sets the data of the link from nodeIndex to linkIndex nodes
     *  \return Ok/Error
    */
    SF3Derror_t setNodeLink(uint64_t nodeIndex, uint64_t linkIndex, linkType_t direction, double interfaceArea)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes || linkIndex >= nodeGrid.numNodes)
            return IndexError;

        uint8_t idx;
        switch (direction)
        {
            case Up:
                idx = 0;
                break;
            case Down:
                idx = 1;
                break;
            case Lateral:
                if(nodeGrid.numLateralLink[nodeIndex] == maxLateralLink)
                    return TopographyError;
                idx = 2 + nodeGrid.numLateralLink[nodeIndex];
                nodeGrid.numLateralLink[nodeIndex]++;
                break;
            default:
                return ParameterError;
        }
        nodeGrid.linkData[idx].linktype[nodeIndex] = direction;
        nodeGrid.linkData[idx].linkIndex[nodeIndex] = linkIndex;
        nodeGrid.linkData[idx].interfaceArea[nodeIndex] = interfaceArea;

        if(simulationFlags.computeWater)
            nodeGrid.linkData[idx].waterFlowSum[nodeIndex] = 0;

        //TO DO Heat
        //if(simulationFlags.computeHeat)

        return SF3Dok;
    }

    /*!
     * \brief sets the soil data of the subsurface nodeIndex node
     * \param nodeIndex index of the node
     * \param soilIndex, horizonIndex indeces of the soil type in the soil list
     * \return Ok/Error
     */
    SF3Derror_t setNodeSoil(uint64_t nodeIndex, uint16_t soilIndex, uint16_t horizonIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(soilIndex >= soilList.size() || horizonIndex >= soilList[soilIndex].size())
            return ParameterError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;      //surfaceFlags must be inizialized before soil data

        nodeGrid.soilRowIndeces[nodeIndex] = soilIndex;
        nodeGrid.soilSurfacePointers[nodeIndex].soilPtr = &(soilList[soilIndex][horizonIndex]);
        return SF3Dok;
    }

    /*!
     * \brief sets the surface data of the surface nodeIndex node
     * \param nodeIndex index of the node
     * \param surfaceIndex index of the surface type in the surface list
     * \return Ok/Error
     */
    SF3Derror_t setNodeSurface(uint64_t nodeIndex, uint16_t surfaceIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(surfaceIndex >= surfaceList.size())
            return ParameterError;

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;      //surfaceFlags must be inizialized before soil data

        nodeGrid.soilSurfacePointers[nodeIndex].surfacePtr = &(surfaceList[surfaceIndex]);

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex surface node's pond data
     * \param nodeIndex
     * \param pond            maximum pond height [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodePond(uint64_t nodeIndex, double pond)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;

        nodeGrid.waterData.pond[nodeIndex] = pond;
        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node water content and updates node pressure head and saturation degree accordingly
     * \param waterContent  [m] surface - [m3 m-3] sub-surface
     * \return Ok/Error
     */
    SF3Derror_t setNodeWaterContent(uint64_t nodeIndex, double waterContent)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(waterContent < 0.)
            return ParameterError;

        if(nodeGrid.surfaceFlag[nodeIndex])
        {
            nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] + waterContent;
            nodeGrid.waterData.oldPressureHeads[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
            nodeGrid.waterData.saturationDegree[nodeIndex] = 1.;
            nodeGrid.waterData.waterConductivity[nodeIndex] = 0.;
        }
        else
        {
            if(waterContent > 1.)
                return ParameterError;

            nodeGrid.waterData.saturationDegree[nodeIndex] = computeNodeSe_fromTheta(nodeIndex, waterContent);        //TO DO
            nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] + computeNodePsi(nodeIndex);    //TO DO
            nodeGrid.waterData.oldPressureHeads[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
            nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);                          //TO DO
        }

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node degree of saturation and updates node pressure head and water conducuctivy accordingly
     * \param degreeOfSaturation  [-] (only sub-surface)
     * \return Ok/Error
     */
    SF3Derror_t setNodeDegreeOfSaturation(uint64_t nodeIndex, double degreeOfSaturation)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;

        if((degreeOfSaturation < 0.) || (degreeOfSaturation > 1.))
            return ParameterError;

        nodeGrid.waterData.saturationDegree[nodeIndex] = degreeOfSaturation;
        nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] - computeNodePsi(nodeIndex);
        nodeGrid.waterData.oldPressureHeads[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];
        nodeGrid.waterData.waterConductivity[nodeIndex] = computeNodeK(nodeIndex);

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node pressure head based of the matric potential
     * \param matricPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodeMatricPotential(uint64_t nodeIndex, double matricPotential)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        nodeGrid.waterData.pressureHead[nodeIndex] = nodeGrid.z[nodeIndex] + matricPotential;
        nodeGrid.waterData.oldPressureHeads[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];

        nodeGrid.waterData.saturationDegree[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? 1. : computeNodeSe(nodeIndex);
        nodeGrid.waterData.waterConductivity[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? NODATA : computeNodeK(nodeIndex);   //TO DO: change default value

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node pressure head based of the total potential
     * \param totalPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodeTotalPotential(uint64_t nodeIndex, double totalPotential)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        nodeGrid.waterData.pressureHead[nodeIndex] = totalPotential;
        nodeGrid.waterData.oldPressureHeads[nodeIndex] = nodeGrid.waterData.pressureHead[nodeIndex];

        nodeGrid.waterData.saturationDegree[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? 1. : computeNodeSe(nodeIndex);
        nodeGrid.waterData.waterConductivity[nodeIndex] = nodeGrid.surfaceFlag[nodeIndex] ? NODATA : computeNodeK(nodeIndex);   //TO DO: change default value

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node boundary prescribed total potential
     * \param prescribedTotalPotential  [m]
     * \return Ok/Error
     */
    SF3Derror_t setNodePrescribedTotalPotential(uint64_t nodeIndex, double prescribedTotalPotential)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] != PrescribedTotalWaterPotential)
            return BoundaryError;

        nodeGrid.boundaryData.prescribedWaterPotential[nodeIndex] = prescribedTotalPotential;

        return SF3Dok;
    }

    /*!
     * \brief sets the nodeIndex node water sink source
     * \param waterSinkSource  [m3 sec-1]
     * \return Ok/Error
     */
    SF3Derror_t setNodeWaterSinkSource(uint64_t nodeIndex, double waterSinkSource)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        nodeGrid.waterData.waterSinkSource[nodeIndex] = waterSinkSource;

        return SF3Dok;
    }

    /*!
     * \brief gets the nodeIndex node water content
     * \return surface water level [m] if surface, volumetric water content [m3 m-3] if subsurface
     */
    double getNodeWaterContent(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        return nodeGrid.surfaceFlag[nodeIndex] ? (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex])
                                               : computeNodeTheta(nodeIndex);
    }

    /*!
     * \brief gets the nodeIndex sub-surface node maximum volumetric water content
     * \return theta_sat (maximum volumetric water content) [m3 m-3]
     */
    double getNodeMaximumWaterContent(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;

        return nodeGrid.soilSurfacePointers[nodeIndex].soilPtr->Theta_s;
    }

    /*!
     * \brief gets the nodeIndex node available water content
     * \return surface water level [m] if surface, volumetric water content [m3 m-3] if subsurface
     */
    double getNodeAvailableWaterContent(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        return nodeGrid.surfaceFlag[nodeIndex] ? (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex])
                                               : SF3Dmax(0., computeNodeTheta(nodeIndex) - computeNodeTheta_fromSignedPsi(nodeIndex, -160));
    }

    /*!
     * \brief gets the nodeIndex node water deficit
     * \param fieldCapacity //TO DO: what is this?
     * \return water deficit    [m3 m-3] (0 at surface)
     */
    double getNodeWaterDeficit(uint64_t nodeIndex, double fieldCapacity)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(nodeGrid.surfaceFlag[nodeIndex])
            return 0.;

        return computeNodeTheta_fromSignedPsi(nodeIndex, -fieldCapacity) - computeNodeTheta(nodeIndex);
    }

    /*!
     * \brief gets the nodeIndex node degree of saturation
     * \note for surface node return 0 if there is no water and 1 if water > 0.001
     * \return degree of saturation [-]
     */
    double getNodeDegreeOfSaturation(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

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
    double getNodeWaterConductivity(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        return nodeGrid.waterData.waterConductivity[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node signed matric potential
     * \return matric potential     [m]
     */
    double getNodeMatricPotential(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        return nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex node signed total potential
     * \return total potential     [m]
     */
    double getNodeTotalPotential(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        return nodeGrid.waterData.pressureHead[nodeIndex];
    }

    /*!
     * \brief gets the nodeIndex surface node pond
     * \return surface pond     [m3]
     */
    double getNodePond(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(!nodeGrid.surfaceFlag[nodeIndex])
            return IndexError;

        return nodeGrid.waterData.pond[nodeIndex];
    }

    /*!
     * \brief gets the maximum integrated water flow fromnodeInxed node in the linkDirection direction
     * \param linkDirection    [Up/Down/Lateral]
     * \return integrated water flow     [m3]
     */
    double getNodeMaxWaterFlow(uint64_t nodeIndex, linkType_t linkDirection)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        double maxFlow = 0.;
        switch (linkDirection)
        {
            case Up:
                if(nodeGrid.linkData[0].linkIndex[nodeIndex] == NoLink)
                    return IndexError;

                return nodeGrid.linkData[0].waterFlowSum[nodeIndex];
            case Down:
                if(nodeGrid.linkData[1].linkIndex[nodeIndex] == NoLink)
                    return IndexError;

                return nodeGrid.linkData[0].waterFlowSum[nodeIndex];
            case Lateral:
                for(uint8_t linkIdx = 0; linkIdx < maxLateralLink; ++linkIdx)   //TO DO: change limit to nodeGrid.numLateralLink[nodeIndex];
                    if(nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != NoLink)
                        maxFlow = SF3Dmax(maxFlow, nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex]);

                return maxFlow;
            default:
                return IndexError;
        }
    }

    /*!
     * \brief gets the total integrated lateral water flow from nodeIndex node
     * \return integrated water flow     [m3]
     */
    double getNodeSumLateralWaterFlow(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        double sumFlow = 0.;
        for(uint8_t linkIdx = 0; linkIdx < maxLateralLink; ++linkIdx)   //TO DO: change limit to nodeGrid.numLateralLink[nodeIndex];
            if(nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != NoLink)
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated lateral water inflow from nodeIndex node
     * \return integrated water inflow     [m3]
     */
    double getNodeSumLateralWaterFlowIn(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        double sumFlow = 0.;
        for(uint8_t linkIdx = 0; linkIdx < maxLateralLink; ++linkIdx)   //TO DO: change limit to nodeGrid.numLateralLink[nodeIndex];
            if((nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != NoLink) && (nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex] > 0))
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated lateral water outflow from nodeIndex node
     * \return integrated water outflow     [m3]
     */
    double getNodeSumLateralWaterFlowOut(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        double sumFlow = 0.;
        for(uint8_t linkIdx = 0; linkIdx < maxLateralLink; ++linkIdx)   //TO DO: change limit to nodeGrid.numLateralLink[nodeIndex];
            if((nodeGrid.linkData[2 + linkIdx].linkIndex[nodeIndex] != NoLink) && (nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex] < 0))
                sumFlow += nodeGrid.linkData[2 + linkIdx].waterFlowSum[nodeIndex];

        return sumFlow;
    }

    /*!
     * \brief gets the total integrated boundary water flow from nodeIndex node
     * \return integrated boundary water flow     [m3]
     */
    double getNodeBoundaryWaterFlow(uint64_t nodeIndex)
    {
        if(!nodeGrid.isInizialized)
            return MemoryError;

        if(nodeIndex >= nodeGrid.numNodes)
            return IndexError;

        if(nodeGrid.boundaryData.boundaryType[nodeIndex] == NoBoundary)
            return BoundaryError;

        return nodeGrid.boundaryData.waterFlowSum[nodeIndex];
    }

    /*!
     * \brief gets the total integrated boundaryType boundary water flow
     * \return integrated boundaryType boundary water flow     [m3]
     */
    double getTotalBoundaryWaterFlow(boundaryType_t boundaryType)
    {
        double totalBoundaryWaterFlow = 0.0;

        #pragma omp parallel for if(__ompStatus) reduction(+:totalBoundaryWaterFlow)
        for (uint64_t nodeIdx = 0; nodeIdx < nodeGrid.numNodes; ++nodeIdx)
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
     * \brief compute one simulation step
     * \details compute one step for active fluxes assuming constant meteo conditions
     * \param maxTimeStep       [s] (default HOUR_SECONDS = 3600)
     * \return computedTimeStep [s]
     */
    double computeStep(double maxTimeStep)
    {
        if(simulationFlags.computeHeat)
        {
            //TO DO: heat
        }

        double dtWater, dtHeat;

        if(simulationFlags.computeWater)
            solver->run(maxTimeStep, dtWater, Water);
        else
            dtWater = SF3Dmin(maxTimeStep, solver->getMaxTimeStep());

        if(simulationFlags.computeHeat)
        {
            //TO DO: heat;
            dtHeat = dtWater;
        }

        return dtWater;
    }

} //namespace
