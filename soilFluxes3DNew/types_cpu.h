#ifndef SOILFLUXES3D_TYPES_CPU_H
#define SOILFLUXES3D_TYPES_CPU_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <omp.h>

#include "commonConstants.h"

namespace soilFluxes3D::New
{
    #define maxLateralLink 8
    #define maxTotalLink (maxLateralLink + 2)
    #define maxMatrixColumns (maxTotalLink + 1)
    #define noData 0

    //Math
    enum meanType_t : uint8_t {Arithmetic, Geometric, Logarithmic};

    //Error Status
    enum SF3Derror_t : uint8_t {SF3Dok, IndexError, MemoryError, TopographyError, BoundaryError, MissingDataError, ParameterError, SolverError};

    //Process implemented
    enum processType : uint8_t {Water, Heat, Solutes};

    //Structure
    enum boundaryType_t : uint8_t {NoBoundary, Runoff, FreeDrainage, FreeLateraleDrainage, PrescribedTotalWaterPotential, Urban, Road, Culvert, HeatSurface, SoluteFlux};
    enum linkType_t : uint8_t {NoLink, Up, Down, Lateral};

    //Soil / surface
    struct soilData_t
    {
        double VG_alpha;            /*!< [m-1] Van Genutchen alpha parameter */
        double VG_n;                /*!< [-] Van Genutchen n parameter */
        double VG_m;                /*!< [-] Van Genutchen m parameter  ]0. , 1.[ */
        double VG_he;               /*!< [m] air-entry potential for modified VG formulation */
        double VG_Sc;               /*!< [-] reduction factor for modified VG formulation */
        double Theta_s;             /*!< [m3 m-3] saturated water content */
        double Theta_r;             /*!< [m3 m-3] residual water content */
        double K_sat;               /*!< [m sec-1] saturated hydraulic conductivity */
        double Mualem_L;            /*!< [-] Mualem tortuosity parameter */

        // for heat flux
        double organicMatter;       /*!< [-] fraction of organic matter */
        double clay;
    };

    struct surfaceData_t
    {
        double roughness;           /*!< [s m-1/3] surface: Manning roughness */
    };

    union soil_surface_ptr
    {
        soilData_t* soilPtr;
        surfaceData_t* surfacePtr;
    };

    //Water
    enum WRCModel : uint8_t {VanGenuchten, ModifiedVanGenuchten, Campbell};
    struct waterData_t
    {
        double *saturationDegree = nullptr;   //Se
        double *waterConductivity = nullptr;  //k
        double *waterFlow = nullptr;          //Qw
        double *pressureHead = nullptr;       //H
        double *waterSinkSource = nullptr;    //waterSinkSource
        double *pond = nullptr;               //pond
        double *invariantFluxes = nullptr;    //invariantFlux

        //Temp variables
        double *oldPressureHeads = nullptr;    //oldH
        double *bestPressureHeads = nullptr;   //bestH

        //Courant data
        double CourantWaterLevel = 0.;
    };

    //Heat and Solutes
    struct heatData_t
    {
        double *temperature = nullptr;      /*!< [K] node temperature */
        double *oldTemperature = nullptr;   /*!< [K] old node temperature */
        double *heatFlux = nullptr;         /*!< [W] heat flow */
        double *heatSinkSource = nullptr;   /*!< [W] heat sink/source */
    };

    //Simulation
    enum balanceResult_t {stepAccepted, stepRefused, stepHalved};
    struct balanceData_t
    {
        double waterStorage = 0.;
        double waterSinkSource = 0.;
        double waterMBE = 0., waterMBR = 0.;

        double heatStorage = 0.;
        double heatSinkSource = 0.;
        double heatMBE = 0., heatMBR = 0.;
    };

    struct simulationFlags_t
    {
        bool computeWater = true;
        bool computeHeat = false;
        bool computeSolutes = false;
        bool computeHeatVapor = false;
        bool computeHeatAdvection = false;
    };

    struct linkData_t
    {
        linkType_t *linktype = nullptr;
        uint64_t *linkIndex = nullptr;      /*!< index of linked elements */
        double *interfaceArea = nullptr;    /*!< interface area [m2] */

        //water data
        double *waterFlowSum = nullptr;     /*!< [m3] sum of flow(i,j) */

        //heat data     //TO DO
        // double *waterFlux;
        // double *vaporFlux;
        // double **fluxes;        //insert variable to count how many different fluxes store
    };

    struct boundaryData_t
    {
        boundaryType_t *boundaryType = nullptr;
        double *boundarySlope = nullptr;            /*!< [m m-1]   */
        double *boundarySize = nullptr;             /*!< [m2] (only for surface runoff [m]) */

        //water data
        double *waterFlowRate = nullptr;            /*!< [m3 s-1]   */
        double *waterFlowSum = nullptr;             /*!< [m3] sum of boundary water flow */
        double *prescribedWaterPotential = nullptr; /*!< [m] imposed total soil-water potential (H) */

        //heat data:
        double *heightWind = nullptr;               /*!< [m] reference height for wind measurement */
        double *heightTemperature = nullptr;        /*!< [m] reference height for temperature and humidity measurement */
        double *roughnessHeight = nullptr;          /*!< [m] surface roughness height */
        double *aerodynamicConductance = nullptr;   /*!< [m s-1] aerodynamic conductance for heat */
        double *soilConductance = nullptr;          /*!< [m s-1] soil conductance */

        double *temperature = nullptr;              /*!< [K] temperature of the boundary (ex. air temperature) */
        double *relativeHumidity = nullptr;         /*!< [%] relative humidity */
        double *windSpeed = nullptr;                /*!< [m s-1] wind speed */
        double *netIrradiance = nullptr;            /*!< [W m-2] net irradiance */

        double *sensibleFlux = nullptr;             /*!< [W m-2] boundary sensible heat flux density */
        double *latentFlux = nullptr;               /*!< [W m-2] boundary latent heat flux density */
        double *radiativeFlux = nullptr;            /*!< [W m-2] boundary net radiative flux density */
        double *advectiveHeatFlux = nullptr;        /*!< [W m-2] boundary advective heat flux density  */

        double *fixedTemperature = nullptr;         /*!< [K] fixed temperature */
        double *fixedTemperatureDepth = nullptr;    /*!< [m] depth of fixed temperature layer */
    };

    struct nodesData_t
    {
        bool isInizialized = false;

        uint64_t numNodes = 0;
        uint64_t numLayers = 0;

        //Topology data
        double *size = nullptr;                             //volume_area
        double *x = nullptr, *y = nullptr, *z = nullptr;    //x, y, z
        bool *surfaceFlag = nullptr;                        //isSurface

        //Soil/surface properties pointers
        uint16_t *soilRowIndeces = nullptr;                 //used for offsets in gpuCode
        soil_surface_ptr *soilSurfacePointers = nullptr;

        //Boundary data
        boundaryData_t boundaryData;

        //Link data
        uint8_t *numLateralLink = nullptr;
        linkData_t linkData[maxTotalLink];

        //Water quantities
        waterData_t waterData;

        //Heat and solutes quantities
        heatData_t heatData;

    };



    //Solver
    enum numericalMethod : uint8_t {Jacobi, GaussSeidel};
    enum solverType : uint8_t  {CPU, GPU};
    enum solverStatus : uint8_t {Error, Created, Inizialized, Launched, Terminated};

    struct SolverParameters
    {
        double MBRThreshold = 1e-4;
        double residualTolerance = 1e-10;

        double deltaTmin = 1;       // [s] (corretto?)
        double deltaTmax = 600;     // [s]
        double deltaTcurr = noData;

        uint16_t maxApproximationsNumber = 10;
        uint16_t maxIterationsNumber = 200;

        WRCModel waterRetentionCurveModel = ModifiedVanGenuchten;
        meanType_t meantype = Logarithmic;

        float lateralVerticalRatio = 10;    //why float?
        double heatWeightFactor = 0.5;      //???

        double CourantWaterThreshold = 0.8; //used for evaluate stability
        double instabilityFactor = 3.0;     //used for evaluate stability

        bool enableOMP = true;
        uint32_t numThreads = std::thread::hardware_concurrency();
    };


    struct MatrixCPU
    {
        uint64_t numRows;
        uint8_t maxColumns = maxMatrixColumns;
        uint8_t* numColumns = nullptr;
        uint64_t** colIndeces = nullptr;
        double** values = nullptr;
    };

    struct VectorCPU
    {
        uint64_t numElements;
        double* values;
    };

    // Log
    struct logData
    {
        std::string solverInfo;
        std::vector<uint32_t> numberApprox;
        std::vector<uint32_t> totIterationsNumbers;
        std::vector<uint32_t> maxIterationsNumbers;
    };
}

#include "types_opt.h"
#ifdef CUDA_ENABLED
    #include "types_gpu.h"
#endif

#endif // SOILFLUXES3D_TYPES_CPU_H
