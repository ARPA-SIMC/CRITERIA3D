#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <type_traits>
#include <limits>
#include <thread>

#include "commonConstants.h"
#include "macro.h"


namespace soilFluxes3D { inline namespace v2
{
    using SF3Duint_t = std::uint32_t;
    using u8_t  = std::uint8_t;
    using u16_t = std::uint16_t;
    using u32_t = std::uint32_t;

    #define maxLateralLink 8
    #define maxTotalLink (maxLateralLink + 2)
    #define maxMatrixColumns (maxTotalLink + 1)

    #define numTotalFluxTypes 9

    constexpr double doubleNoData_v = static_cast<double>(NODATA);
    constexpr SF3Duint_t uintNoData_v = std::numeric_limits<SF3Duint_t>::max();
    #define noDataD doubleNoData_v
    #define noDataU uintNoData_v

    //Math
    enum class meanType_t : u8_t {Arithmetic, Geometric, Logarithmic};

    //Error Status
    enum class SF3Derror_t : u8_t {SF3Dok, IndexError, MemoryError, TopographyError, BoundaryError, MissingDataError, ParameterError, SolverError, FileError};

    inline constexpr __cudaSpec double getDoubleErrorValue(const SF3Derror_t errorCode)
    {
        if(errorCode == SF3Derror_t::SF3Dok)
            return 0;

        switch(errorCode)
        {
            case SF3Derror_t::IndexError:
                return static_cast<double>(INDEX_ERROR);
            case SF3Derror_t::MemoryError:
                return static_cast<double>(MEMORY_ERROR);
            case SF3Derror_t::TopographyError:
                return static_cast<double>(TOPOGRAPHY_ERROR);
            case SF3Derror_t::BoundaryError:
                return static_cast<double>(BOUNDARY_ERROR);
            case SF3Derror_t::MissingDataError:
                return static_cast<double>(MISSING_DATA_ERROR);
            case SF3Derror_t::ParameterError:
                return static_cast<double>(PARAMETER_ERROR);
            default:
                return static_cast<double>(INDEX_ERROR);
        }

    }

    inline constexpr bool getSF3DerrorName(soilFluxes3D::SF3Derror_t errorCode, std::string& errorName)
    {
        if (errorCode == soilFluxes3D::SF3Derror_t::SF3Dok)
            return false;

        switch (errorCode)
        {
            case soilFluxes3D::SF3Derror_t::IndexError:
                errorName = "index error";
                break;
            case soilFluxes3D::SF3Derror_t::MemoryError:
                errorName = "memory error";
                break;
            case soilFluxes3D::SF3Derror_t::TopographyError:
                errorName = "topography error";
                break;
            case soilFluxes3D::SF3Derror_t::BoundaryError:
                errorName = "boundary error";
                break;
            case soilFluxes3D::SF3Derror_t::ParameterError:
                errorName = "parameter error";
                break;
            default:
                errorName = "generic error";
            }

        return true;
    }

    //Process implemented
    enum class processType : u8_t {Water, Heat, Solutes};

    //Structure
    /*enum class boundaryType_t : u8_t {NoBoundary = BOUNDARY_NONE,
                                       Runoff = BOUNDARY_RUNOFF,
                                       FreeDrainage = BOUNDARY_FREEDRAINAGE,
                                       FreeLateraleDrainage = BOUNDARY_FREELATERALDRAINAGE,
                                       PrescribedTotalWaterPotential = BOUNDARY_PRESCRIBEDTOTALPOTENTIAL,
                                       Urban = BOUNDARY_URBAN,
                                       Road = BOUNDARY_ROAD,
                                       Culvert = BOUNDARY_CULVERT,
                                       HeatSurface = BOUNDARY_HEAT_SURFACE,
                                       SoluteFlux = BOUNDARY_SOLUTEFLUX};*/

    enum class boundaryType_t : u8_t {NoBoundary, Runoff, FreeDrainage, FreeLateraleDrainage, PrescribedTotalWaterPotential, Urban, Road, Culvert, HeatSurface, SoluteFlux};

    enum class linkType_t : u8_t {NoLink, Up, Down, Lateral};

    //Soil / surface
    struct soilData_t
    {
        u16_t soilNumber;
        u8_t horizonNumber;
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
        double clay;                /*!< [-] fraction of clay */
    };

    struct surfaceData_t
    {
        double roughness;           /*!< [s m-1/3] surface: Manning roughness */
    };

    union soilSurface_ptr
    {
        soilData_t* soilPtr;
        surfaceData_t* surfacePtr;
    };

    //Water
    enum class WRCModel : u8_t {VanGenuchten, ModifiedVanGenuchten, Campbell};
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
        double *oldPressureHead = nullptr;    //oldH
        double *bestPressureHead = nullptr;   //bestH

        //Courant data
        double *partialCourantWaterLevels = nullptr;
        double CourantWaterLevel = 0.;
    };

    struct culvertData_t
    {
        double width;        /*!< [m] */
        double height;       /*!< [m] */
        double roughness;    /*!< [s m-1/3] Manning roughness */
    };

    //Heat
    struct heatData_t
    {
        double *temperature = nullptr;      /*!< [K] node temperature */
        double *oldTemperature = nullptr;   /*!< [K] old node temperature */
        double *heatFlux = nullptr;         /*!< [W] heat flow */
        double *heatSinkSource = nullptr;   /*!< [W] heat sink/source */
    };

    //Solutes
    // ...

    //Simulation
    enum class balanceResult_t : u8_t {stepAccepted, stepRefused, stepHalved, stepNan};
    struct balanceData_t
    {
        double waterStorage = 0.;
        double waterSinkSource = 0.;
        double waterMBE = 0., waterMBR = 0.;

        double heatStorage = 0.;
        double heatSinkSource = 0.;
        double heatMBE = 0., heatMBR = 0.;
    };

    enum class heatFluxSaveMode_t : uint8_t {None, Total, All};

    struct simulationFlags_t
    {
        bool computeWater = true;
        bool computeHeat = false;
        bool computeSolutes = false;
        bool computeHeatVapor = false;
        bool computeHeatAdvection = false;

        heatFluxSaveMode_t HFsaveMode = heatFluxSaveMode_t::None;
    };

    enum class fluxTypes_t : u8_t {HeatTotal, HeatDiffusive, HeatLatentIsothermal, HeatLatentThermal, HeatAdvective, WaterLiquidIsothermal, WaterLiquidThermal, WaterVaporIsothermal, WaterVaporThermal};
    constexpr fluxTypes_t heatFluxIndeces[] = {fluxTypes_t::HeatTotal, fluxTypes_t::HeatDiffusive, fluxTypes_t::HeatLatentIsothermal, fluxTypes_t::HeatLatentThermal, fluxTypes_t::HeatAdvective};
    constexpr fluxTypes_t waterFluxIndeces[] = {fluxTypes_t::WaterLiquidIsothermal, fluxTypes_t::WaterLiquidThermal, fluxTypes_t::WaterVaporIsothermal, fluxTypes_t::WaterVaporThermal};

    struct linkData_t
    {
        linkType_t *linkType = nullptr;
        SF3Duint_t *linkIndex = nullptr;        /*!< index of linked elements */
        double *interfaceArea = nullptr;        /*!< interface area [m2] */

        //water data
        double *waterFlowSum = nullptr;         /*!< [m3] sum of flow(i,j) */

        //heat data
        double *waterFlux = nullptr;                        /*!< [m3 s-1] water volume flux*/
        double *vaporFlux = nullptr;                        /*!< [kg s-1] vapor mass flux*/
        double *fluxes[numTotalFluxTypes] = {nullptr};      /*!< [W] for heat fluxes; [m3 s-1] for water fluxes */
    };

    //Boundary
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

        double *fixedTemperatureValue = nullptr;    /*!< [K] fixed temperature */
        double *fixedTemperatureDepth = nullptr;    /*!< [m] depth of fixed temperature layer */
    };

    struct nodesData_t
    {
        bool isInitialized = false;

        SF3Duint_t numNodes = 0;
        SF3Duint_t numLayers = 0;

        //Topology data
        double *size = nullptr;                             //volume_area
        double *x = nullptr, *y = nullptr, *z = nullptr;    //x, y, z
        bool *surfaceFlag = nullptr;                        //isSurface

        //Soil/surface properties pointers
        soilSurface_ptr *soilSurfacePointers = nullptr;

        //Boundary data
        boundaryData_t boundaryData;

        //Link data
        u8_t *numLateralLink = nullptr;
        linkData_t linkData[maxTotalLink];

        //Water quantities
        waterData_t waterData;

        //Culvert pointers
        culvertData_t* *culvertPtr = nullptr;

        //Heat and solutes quantities
        heatData_t heatData;
    };

    //Solver
    enum class numericalMethod : u8_t {Jacobi, GaussSeidel};
    enum class solverType : u8_t  {CPU, GPU};
    enum class solverStatus : u8_t {Error, Created, initialized, Launched, Terminated};

    struct SolverParameters
    {
        double MBRThreshold = 1e-3;
        double residualTolerance = 1e-10;

        double deltaTmin = 1;           // [s]
        double deltaTmax = 600;         // [s]
        double deltaTcurr = noDataD;

        u16_t maxApproximationsNumber = 10;
        u16_t maxIterationsNumber = 200;

        WRCModel waterRetentionCurveModel = WRCModel::ModifiedVanGenuchten;
        meanType_t meanType = meanType_t::Logarithmic;

        double lateralVerticalRatio = 10.;
        double heatWeightFactor = 0.5;          //???

        double CourantWaterThreshold = 0.5;     //used for evaluate stability
        double instabilityFactor = 10.;         //used for evaluate stability

        bool enableOMP = true;
        u32_t numThreads = std::thread::hardware_concurrency();
    };

    template<typename E>
    __cudaSpec constexpr auto castToUnderlyingType(E value)
    {
        static_assert(std::is_enum_v<E>, "type required to be enum to be casted");
        return static_cast<std::underlying_type_t<E>>(value);
    }
}}

#include "types_opt.h"
