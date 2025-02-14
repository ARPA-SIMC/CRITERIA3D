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

    /*!
     * Assign physical and miscellaneous constants
    */

    #define      CARBONFACTOR 0.5           /*!< coeff for conversion of carbon into DM, kgC kgDM-1  */
    #define      GAMMA  66.2                /*!< psychrometer constant, Pa K-1  */
    #define      LATENT  43956              /*!< latent heat of vaporization, J mol-1  */
    #define      H2OMOLECULARWEIGHT  0.018  /*!< molecular weight of H2O, kg mol-1  */
    #define      OSS 21176                  /*!< oxygen part pressure in the atmosphere, Pa  */

        /*!
        * Define additional photosynthetic parameters
        */
    #define      HARD       46.39           /*!< activation energy of RD0 (kJ mol-1)   */
    #define      HAVCM      65.33           /*!< activation energy of VCMOP (kJ mol-1)  */
    #define      HAJM       43.9            /*!< activation energy of JMOP (kJ mol-1 e-)  */
    #define      HAKC       79.43           /*!< activation energy of KCT0 (kJ mol-1)  */
    #define      HAKO       36.38           /*!< activation energy of KOT0 (kJ mol-1)  */
    #define      HAGSTAR    37.83           /*!< activation energy of Gamma_star (kJ mol-1)  */
    #define      HDEACTIVATION  200         /*!< deactivation energy from Kattge & Knorr 2007 (kJ mol-1)  */

    #define      CRD        18.72           /*!< scaling factor in RD0 response to temperature (-)  */
    #define      CVCM       26.35           /*!< scaling factor in VCMOP response to temperature (-)  */
    #define      CVOM       22.98           /*!< scaling factor in VOMOP response to temperature (-)  */
    #define      CGSTAR     19.02           /*!< scaling factor in Gamma_star response to temperature (-)  */
    #define      CKC        38.05           /*!< scaling factor in KCT0 response to temperature (-)  */
    #define      CKO        20.30           /*!< scaling factor in KOT0 response to temperature (-)  */
    #define      CJM        17.7            /*!< scaling factor in JMOP response to temperature (-)  */

        /*!
         * Define additional functional and structural parameters for stand and understorey
         */
    #define      CONV       0.8             //    dry matter conversion efficiency (growth resp.)(-)
    #define      MERCH      0.85            //    merchantable wood as fraction of stem biomass (-)
    #define      RADRT      1.E-3           //    root radius (m)
    #define      STH0       0.8561          //    intercept in self-thinning eq. (log(TREES) vs log(WST)) (m-2)
    #define      STH1       1.9551          //    slope in self-thinning eq. (log(TREES) vs log(WST)) (kgDM-1)
    #define      ALLRUND    0.5             //    coeff of allocation to roots in understorey (-)

        /*!
         * Define soil respiration parameters, partition soil C into young and old components
         * Note: initial steady-state conditions are assumed (Andren & Katterer 1997)
         * Reference respiration rates are for a 'tropical' soil (Andren & Katterer 1997, p. 1231).
        */
    #define      HUMCOEF    0.125           /*!< humification coefficient (-)  */
    #define      R0SLO      0.04556         /*!< resp per unit old soil carbon at ref conditions (kgDM kgDM-1 d-1) */
    #define      R0SLY      4.228           /*!< resp per unit young soil carbon at ref conditions (kgDM kgDM-1 d-1) */
    #define      CHLDEFAULT 500             /*!< [g cm-2]  */
    #define      RUEGRASS   1.0             /*!< maize: 1.5-2.0, vine: 0.6-1.0  */

    #define      SHADEDGRASS true
    #define      SUNLITGRASS false


    #define NOT_INITIALIZED_VINE -1


    struct TstatePlant
    {
        double treeNetPrimaryProduction;
        double treecumulatedBiomassFoliage;
        double treecumulatedBiomassRoot;
        double treecumulatedBiomassSapwood;
        double understoreycumulatedBiomass;
        double understoreycumulatedBiomassFoliage;
        double understoreycumulatedBiomassRoot;
    };

    struct TweatherDerivedVariable {
        double airVapourPressure;
        double emissivitySky;
        double longWaveIrradiance;
        double slopeSatVapPressureVSTemp;
        double myDirectIrradiance;
        double myDiffuseIrradiance;
        double myEmissivitySky;
        double myLongWaveIrradiance;
        double psychrometricConstant;

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
        double last30DaysTAvg;


    };

    struct TenvironmentalVariable {

        double CO2;
        double sineSolarElevation;
    };

    struct Tplant {

        double myChlorophyllContent;
        double height;
        double myLeafWidth;
        bool isAmphystomatic;
        double foliageLongevity;
        double sapwoodLongevity;
        double fineRootLongevity;


    };

    struct ThydrallSoil {

        int layersNr;
        double totalDepth;
        double temperature;
        std::vector <double> rootDensity;
        std::vector <double> stressCoefficient;
        std::vector <double> waterContent;
        std::vector <double> wiltingPoint;
        std::vector <double> fieldCapacity;
        std::vector <double> saturation;
    };

    struct TbigLeaf
    {
        double absorbedPAR ;
        double isothermalNetRadiation;
        double leafAreaIndex;
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
        double assimilation,transpiration,stomatalConductance;

    };

    struct TparameterWangLeuning
    {
        double optimalTemperatureForPhotosynthesis;
        double stomatalConductanceMin;
        double sensitivityToVapourPressureDeficit;
        double alpha;
        double psiLeaf;                 // kPa
        double waterStressThreshold;
        double maxCarboxRate;           // Vcmo at optimal temperature

    };

    struct TlightExtinctionCoefficient
    {
        double global;
        double par;
        double nir;

    };

    struct ThydrallDeltaTimeOutputs {

        double netAssimilation;
        double grossAssimilation ;
        double transpiration ;
        double interceptedWater ;
        double netDryMatter ;
        double absorbedPAR ;
        double respiration ;
        double transpirationGrass;
        double transpirationNoStress;
    };

    struct ThydrallNitrogen {
        double interceptLeaf, slopeLeaf, leafNitrogen;
        double leaf , stem , root;
    };

    struct ThydrallBiomass {

        double total ;
        double leaf ;
        double sapwood ;
        double fineRoot ;
    };

    struct TallocationCoefficient {

        double toFoliage;
        double toFineRoots;
        double toSapwood;
    };


    class Crit3DHydrallMaps
    {
    private:

    public:
        //sapwood, foliage, fine root
        gis::Crit3DRasterGrid* standBiomassMap;
        gis::Crit3DRasterGrid* rootBiomassMap;
        gis::Crit3DRasterGrid* mapLAI;
        gis::Crit3DRasterGrid* mapLast30DaysTavg;

        Crit3DHydrallMaps();
        ~Crit3DHydrallMaps();

        void initialize(const gis::Crit3DRasterGrid& DEM);
    };
    class Crit3D_Hydrall{
    public:

        //Crit3D_Hydrall();
        //~Crit3D_Hydrall();

        void initialize();
        bool firstDayOfMonth;
        int firstMonthVegetativeSeason;
        bool isFirstYearSimulation;

        TbigLeaf sunlit,shaded, understorey;
        TweatherVariable weatherVariable;
        TenvironmentalVariable environmentalVariable;
        TparameterWangLeuning parameterWangLeuning;
        Tplant plant;
        ThydrallSoil soil;
        TlightExtinctionCoefficient directLightExtinctionCoefficient;
        TlightExtinctionCoefficient diffuseLightExtinctionCoefficient;
        ThydrallDeltaTimeOutputs deltaTime;
        ThydrallNitrogen nitrogenContent;
        ThydrallBiomass treeBiomass, understoreyBiomass;
        TstatePlant statePlant;
        TallocationCoefficient allocationCoefficient;


        double elevation;
        int simulationStepInSeconds;
        double leafAreaIndex;

        //gasflux results
        std::vector<double> treeTranspirationRate;          //molH2O m^-2 s^-1
        double treeAssimilationRate;
        std::vector<double> understoreyTranspirationRate;
        double understoreyAssimilationRate;


        void radiationAbsorption();
        void setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, double CO2, double sunElevation);
        bool setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure);
        void setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex);
        void setPlantVariables(double chlorophyllContent);
        bool computeHydrallPoint(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep, double &AGBiomass, double &rootBiomass);
        double getCO2(Crit3DDate myDate, double myTemperature, double myElevation);
        //double getPressureFromElevation(double myTemperature, double myElevation);
        double getLAI();
        double meanLastMonthTemperature(double previousLastMonthTemp, double simulationStepInSeconds, double myInstantTemp);
        double photosynthesisAndTranspiration();
        double photosynthesisAndTranspirationUnderstorey();
        void leafTemperature();
        void aerodynamicalCoupling();
        double leafWidth();
        void upscale();
        double acclimationFunction(double Ha , double Hd, double leafTemp, double entropicTerm,double optimumTemp);
        void photosynthesisKernel(double COMP,double GAC,double GHR,double GSCD,double J,double KC,double KO
                                                  ,double RD,double RNI,double STOMWL,double VCmax,double *ASS,double *GSC,double *TR);
        void carbonWaterFluxesProfile();
        void cumulatedResults();
        double plantRespiration();
        double soilTemperatureModel();
        double temperatureMoistureFunction(double temperature);
        bool growthStand();
        void resetStandVariables();

    };


#endif // HYDRALL_H
