#ifndef SOILFLUXES3D_TYPES_CPU_H
#define SOILFLUXES3D_TYPES_CPU_H

#include <cstdint>
#include <vector>

// using u64 = uint64_t;
// using u16 = uint16_t;

#include "../types.h"

namespace soilFluxes3D::New
{
    #define maxLateralLink 8
    #define maxTotalLink maxLateralLink + 2
    #define noData 0

    //Math
    enum meanType_t {Arithmetic, Geometric, Logarithmic};

    //Error Status
    enum SF3Derror_t {SF3Dok, IndexError, MemoryError, TopographyError, BoundaryError, MissingDataError, ParameterError};

    //Process implemented
    enum processType {Water, Heat, Solutes};

    //Structure
    enum boundaryType_t {None, Runoff, FreeDrainage, FreeLateraleDrainage, PrescribedTotalWaterPotential, Urban, Road, Culvert, HeatSurface, SoluteFlux};
    enum linkType_t {NoLink, Up, Down, Lateral};

    //Soil
    struct soilData_t
    {
        double *VG_alpha;            /*!< [m-1] Van Genutchen alpha parameter */
        double *VG_n;                /*!< [-] Van Genutchen n parameter */
        double *VG_m;                /*!< [-] Van Genutchen m parameter  ]0. , 1.[ */
        double *VG_he;               /*!< [m] air-entry potential for modified VG formulation */
        double *VG_Sc;               /*!< [-] reduction factor for modified VG formulation */
        double *Theta_s;             /*!< [m3 m-3] saturated water content */
        double *Theta_r;             /*!< [m3 m-3] residual water content */
        double *K_sat;               /*!< [m sec-1] saturated hydraulic conductivity */
        double *Mualem_L;            /*!< [-] Mualem tortuosity parameter */

        double *roughness;           /*!< [s m-1/3] surface: Manning roughness */

        // for heat flux
        double *organicMatter;       /*!< [-] fraction of organic matter */
        double *clay;
    };

    //Water
    enum WRCModel {VanGenuchten, ModifiedVanGenuchten, Campbell};
    struct waterData_t
    {
        double *saturationDegree;   //Se
        double *waterConductivity;  //k
        double *waterFlow;          //Qw
        double *pressureHead;       //H
        double *waterSinkSource;    //waterSinkSource
        double *pond;               //pond

        //Temp variables //TO DO: move in Solver?
        double *oldPressureHeads;    //oldH
        double *bestPressureHeads;   //bestH

        //Boundary data
        double *boundaryWaterFlow;
        double *boundaryTotalWaterFlow;
        double *prescribedWaterPotential;

        //Courant data
        double CourantWaterLevel = 0;
    };

    //Heat and Solutes
    struct heatData_t
    {
        double *temperature;        //T
        double *heatFlux;           //Qh
        double *heatSinkSource;

        double *oldTemperature;     //oldT

        //Boundary data
        double *boundaryWaterFlow;
        double *boundaryTotalWaterFlow;
        double *prescribedWaterPotential;
    };

    //Simulation
    struct balanceData_t
    {
        double waterStorage;
        double waterSinkSource = 0;
        double waterMBE = 0, waterMBR = 0;

        double heatStorage;
        double heatSinkSource;
        double heatMBE, heatMBR;
    };

    struct simulationParameters_t
    {
        bool computeWater = true;
        bool computeHeat = false;
        bool computeSolutes = false;
        bool computeHeatVapor = false;
        bool computeHeatAdvection = false;
    };

    struct linkData_t
    {
        linkType_t *linktype;
        uint64_t *linkIndex;
        double *interfaceArea;

        //water data
        double *waterFlow;

        //heat data
        // double *waterFlux;
        // double *vaporFlux;
        // double **fluxes;        //insert variable to count how many different fluxes store
    };

    struct boundaryData_t
    {
        boundaryType_t *boundaryType;
        double *boundarySlope;              /*!< [m m-1]   */

        //water data
        double *boundarySize;               /*!< [m2] (only for surface runoff [m]) */
        double *waterFlowRate;              /*!< [m3 s-1]   */
        double *waterFlowSum;               /*!< [m3] sum of boundary water flow */
        double *prescribedWaterPotential;   /*!< [m] imposed total soil-water potential (H) */

        //heat data: TO DO
    };

    struct nodesData_t
    {
        bool isInizialized = false;

        uint64_t numNodes;

        //Topology data
        double *size;               //volume_area
        double *x, *y, *z;          //x, y, z
        bool *surfaceFlag;          //isSurface

        //Boundary data
        boundaryData_t boundaryData;

        //Link data
        uint8_t *numLateralLink = nullptr;
        linkData_t linkData[maxTotalLink];

        //Soil quantities
        soilData_t soilData;

        //Water quantities
        waterData_t waterData;

        //Heat and solutes quantities
        heatData_t heatData;

    };



    //Solver
    enum numericalMethod {Jacobi, GaussSeidel};
    enum solverType {CPU, GPU};
    enum solverStatus {Error, Created, Inizialized, Launched, Terminated};

    struct SolverParameters
    {
        double MBRThreshold = 1e-6;
        double residualTolerance = 1e-10;

        double deltaTmin = 1;   // [s] (corretto?)
        double deltaTmax = 600; // [s]
        double deltaTcurr = NULL;

        uint16_t maxApproximationsNumber = 10;
        uint16_t maxIterationsNumber = 200;

        WRCModel waterRetentionCurveModel = ModifiedVanGenuchten;
        meanType_t meantype = Logarithmic;

        float lateralVerticalRatio = 10;    //why float?
        double heatWeightFactor = 0.5;      //???

        double CourantWaterThreshold = 0.8; //used for evaluate stability
        double instabilityFactor = 3.0;     //used for evaluate stability
    };

    struct MatrixElementCPU
    {
        uint64_t columnIndex;
        double elementValue;
    };

    struct MatrixCPU
    {
        uint64_t numRows;
        std::vector<std::vector<MatrixElementCPU>> elements;
    };

    // Log
    struct logData
    {
        uint16_t numberApprox = NULL;
        //std::vector<int> totIterationsNumbers = nullptr;
        //std::vector<int> maxIterationsNumbers = nullptr;
    };

}


#endif // SOILFLUXES3D_TYPES_CPU_H
