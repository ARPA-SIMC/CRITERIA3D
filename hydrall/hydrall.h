#ifndef HYDRALL_H
#define HYDRALL_H

    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    #define UPSCALINGFUNC(z,LAI) ((1.0 - exp(-(z)*(LAI))) / (z))

    // Tree-plant properties
    #define FORM   0.5          // stem form factor
    #define RHOF   0.1          // [KgDM m-3] foliage density
    #define RHOS   750          // [KgDM m-3] wood-stem density

    // Hydraulic properties
    #define H50     0.4         // height for 50% maturation of xylem cells (m) [not relevant]
    #define KR      4.0E-7      // root specific conductance (m3 MPa-1 s-1 kg-1) [not relevant]
    #define KSMAX   2.5E-3      // max. sapwood specific conductivity (m2 MPa-1 s-1) [not relevant]
    #define PSITHR  -2.5        // water potential threshold for cavitation (MPa) [not relevant]


    #define NOT_INITIALIZED_VINE -1


    struct TweatherDerivedVariable {
        double airVapourPressure;
        double emissivitySky;
        double longWaveIrradiance;
        double slopeSatVapPressureVSTemp;
        double myDirectIrradiance;
        double myDiffuseIrradiance;
        double myEmissivitySky;
        double myLongWaveIrradiance;

    };

    struct TweatherVariable {
        TweatherDerivedVariable derived;

        double myInstantTemp;
        double prec;
        double irradiance;
        double relativeHumidity;
        double windSpeed;
        double atmosphericPressure;
        //double meanDailyTemperature;
        double vaporPressureDeficit;


    };

    struct TbigLeaf
    {
        double absorbedPAR ;
        double isothermalNetRadiation;
        double leafAreaIndex ;
        double totalConductanceHeatExchange;
        double aerodynamicConductanceHeatExchange;
        double aerodynamicConductanceCO2Exchange ;
        double leafTemperature ;
        double darkRespiration ;
        double minimalStomatalConductance;
        double maximalCarboxylationRate,maximalElectronTrasportRate ;
        double carbonMichaelisMentenConstant, oxygenMichaelisMentenConstant ;
        double compensationPoint, convexityFactorNonRectangularHyperbola ;
        double quantumYieldPS2 ;
        double assimilation,transpiration,stomatalConductance ;


    };

    class Crit3DHydrallMaps
    {
    private:

    public:
        gis::Crit3DRasterGrid* mapLAI;
        gis::Crit3DRasterGrid* mapLast30DaysTavg;

        Crit3DHydrallMaps();
        ~Crit3DHydrallMaps();

        void initialize(const gis::Crit3DRasterGrid& DEM);
    };
    class Crit3D_Hydrall{
    public:

        // Crit3D_Hydrall();
        // ~Crit3D_Hydrall();

        void initialize();
        void initializeLeaf(TbigLeaf myLeaf);
        //gis::Crit3DRasterGrid* stateMaps;

        TbigLeaf sunlit,shaded;
        TweatherVariable weatherVariable;
        double myChlorophyllContent;
        double elevation;

        void radiationAbsorption(double mySunElevation, double leafAreaIndex);
        void setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex);
        bool setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex);
        void setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex);
        void setPlantVariables(double chlorophyllContent);

    };

    bool computeHydrallPoint(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep);
    double getCO2(Crit3DDate myDate, double myTemperature, double myElevation);
    double getPressureFromElevation(double myTemperature, double myElevation);
    double getLAI();
    double meanLastMonthTemperature(double previousLastMonthTemp, double simulationStepInSeconds, double myInstantTemp);
    double photosynthesisAndTranspiration();

#endif // HYDRALL_H
