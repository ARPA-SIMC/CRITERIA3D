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
    #define RHOS   750          // [KgDM m-3] default wood-stem density
    #define LAIMIN 0.1          //[-]
    #define LAIMAX 1            //[-]

    // Hydraulic properties
    #define H50     0.4         // height for 50% maturation of xylem cells (m) [not relevant]
    #define KR      4.0E-7      // root specific conductance (m3 MPa-1 s-1 kg-1) [not relevant]
    #define KSMAX   2.5E-3      // max. sapwood specific conductivity (m2 MPa-1 s-1) [not relevant]
    #define PSITHR  -2.5        // water potential threshold for cavitation (MPa) [not relevant]

    /*!
     * Assign physical and miscellaneous constants
    */

    #define RESPIRATION_PARAMETER  0.00000147222 // to compute respiration
    #define CARBONFACTOR 0.5           /*!< coeff for conversion of carbon into DM, kgC kgDM-1  */
    #define GAMMA  66.2                /*!< psychrometer constant, Pa K-1  */
    #define LATENT  43956              /*!< latent heat of vaporization, J mol-1  */
    #define H2OMOLECULARWEIGHT  0.018  /*!< molecular weight of H2O, kg mol-1  */
    #define OSS 21176                  /*!< oxygen part pressure in the atmosphere, Pa  */

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

    struct TecophysiologicalParameter {
        std::string name; // name of the species
        double Vcmo; // Max carboxylation rate at 25°C rate of RuBiSCO activity (PSII photosynthesis))
        double mBallBerry; // empirical parameter of sensitivity to water stress to obtain stomatal closure
        bool isAmphystomatic;
        double rootShootRatio; //ratio of C allocated to roots and C allocated to aboveground biomass
        double wildfireDamage; //ratio of biomass lost in wildfire event
    };

    struct TLAIparam {
        std::string name;
        double lai_min;
        double lai_max;
    };

    struct TLAIphenology{
        std::string name;
        double emergence; // GDD with threshold 5°C
        double increase;  // GDD with threshold 5°C
        double decrease;  // GDD with threshold 5°C
    };

    struct TAnnualYield{
       std::string name;
       double carbon; // annual carbon biomass
    };

    class Crit3DHydrallState
    { 
    public:
        Crit3DHydrallState();

        double standBiomass;
        double rootBiomass;
    };


    class Crit3DHydrallStatePlant
    {
    public:
        Crit3DHydrallStatePlant();

        double treeNetPrimaryProduction; //SAVE
        double treeBiomassFoliage; //SAVE
        double treeBiomassRoot; //SAVE
        double treeBiomassSapwood; //SAVE
        double understoreyNetPrimaryProduction; //SAVE
        double understoreyBiomassFoliage; //SAVE
        double understoreyBiomassRoot; //SAVE
    };

    class Crit3DHydrallWeatherDerivedVariable {

    public:
        Crit3DHydrallWeatherDerivedVariable();

        double airVapourPressure;
        double emissivitySky;
        double longWaveIrradiance;
        double slopeSatVapPressureVSTemp;
        double myDirectIrradiance;
        double myDiffuseIrradiance;
        double myEmissivitySky;
        double myLongWaveIrradiance;
        double psychrometricConstant;
        double et0;

    };

    class Crit3DHydrallWeatherVariable {

    public:
        Crit3DHydrallWeatherVariable();

        Crit3DHydrallWeatherDerivedVariable derived;

        double getYearlyET0 () { return yearlyET0; };
        void setYearlyET0 (double myET) { yearlyET0 = myET; };
        double getYearlyPrec () { return yearlyPrec; };
        void setYearlyPrec (double myPrec) { yearlyPrec = myPrec; };

        double myInstantTemp;
        double prec;
        double irradiance;
        double relativeHumidity;
        double windSpeed;
        double atmosphericPressure;
        //double meanDailyTemperature;
        double vaporPressureDeficit;
        double last30DaysTAvg;
        double meanDailyTemp;


    private:
        double yearlyET0;
        double yearlyPrec;


    };

    class Crit3DHydrallEnvironmentalVariable {

    public:
        Crit3DHydrallEnvironmentalVariable();

        double CO2;
        double sineSolarElevation;
    };

    class Crit3DHydrallPlant {

    public:
        Crit3DHydrallPlant();

        // TODO Cate unità di misura
        std::vector<TAnnualYield> tableYield;
        std::vector<TecophysiologicalParameter> tableEcophysiologicalParameters;
        std::vector<TLAIparam> rangeLAI;
        std::vector<TLAIphenology> phenologyLAI;
        double myChlorophyllContent;
        double height; // in cm
        double hydraulicResistancePerFoliageArea; //(MPa s m2 m-3)
        double myLeafWidth;
        bool isAmphystomatic;
        double foliageLongevity;
        double sapwoodLongevity;
        double fineRootLongevity;
        double foliageDensity;
        double woodDensity;
        double specificLeafArea;
        double psiLeaf;
        double psiSoilCritical;
        double transpirationCritical;
        double psiLeafMinimum;
        double transpirationPerUnitFoliageAreaCritical;
        double standVolume; // maps referred to stand volume MUST be initialized
        double currentIncrementalVolume;
        double rootShootRatioRef;
        double mBallBerry;
        double wildfireDamage;
        int management;

        void setLAICanopy(double myLAI) { leafAreaIndexCanopy = myLAI; }
        double getLAICanopy() { return leafAreaIndexCanopy; }

        void setLAICanopyMin(double myLAIMin) { leafAreaIndexCanopyMin = myLAIMin; }
        double getLAICanopyMin() { return leafAreaIndexCanopyMin; }

        void setLAICanopyMax(double myLAIMax) { leafAreaIndexCanopyMax = myLAIMax; }
        double getLAICanopyMax() { return leafAreaIndexCanopyMax; }

    private:
        double leafAreaIndexCanopy;
        double leafAreaIndexCanopyMax;
        double leafAreaIndexCanopyMin;

    };

    class Crit3DHydrallSoil {
    public:
        Crit3DHydrallSoil();

        int layersNr;
        double totalDepth;
        double temperature;

        std::vector <double> stressCoefficient;
        std::vector <double> waterContent;
        std::vector <double> wiltingPoint;
        std::vector <double> fieldCapacity;
        std::vector <double> saturation;
        std::vector <double> hydraulicConductivity;
        std::vector <double> satHydraulicConductivity;
        std::vector <double> nodeThickness;
        std::vector <double> clay;
        std::vector <double> sand;
        std::vector <double> silt;
        std::vector <double> bulkDensity;
        std::vector <double> waterPotential;

        void setRootDensity(std::vector<double> myRD) { rootDensity = myRD; }
        std::vector<double> getRootDensity() { return rootDensity; }

    private:
        std::vector <double> rootDensity;

    };

    class Crit3DHydrallBigLeaf
    {
    public:
        Crit3DHydrallBigLeaf();

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

    class Crit3DHydrallParameterWangLeuning
    {
    public:
        Crit3DHydrallParameterWangLeuning();

        double optimalTemperatureForPhotosynthesis;
        double stomatalConductanceMin;
        double sensitivityToVapourPressureDeficit;
        double alpha;
        double psiLeaf;
        double waterStressThreshold;
        double maxCarboxRate;
    };


    class Crit3DHydrallLightExtinctionCoefficient
    {
    public:
        Crit3DHydrallLightExtinctionCoefficient();

        double global;
        double par;
        double nir;
    };


    class Crit3DHydrallDeltaTimeOutputs {
    public:
        Crit3DHydrallDeltaTimeOutputs();

        double netAssimilation;
        double grossAssimilation ;
        double transpiration ;
        double interceptedWater ;
        double netDryMatter ;
        double absorbedPAR ;
        double respiration ;
        double transpirationGrass;
        double transpirationNoStress;
        double evaporation;
        double evapoTranspiration;
        double understoreyNetAssimilation;
    };

    class Crit3DHydrallNitrogen {
    public:
        Crit3DHydrallNitrogen();

        double interceptLeaf, slopeLeaf;
        double leaf;
        double stem;
        double root;
    };

    class Crit3DHydrallBiomass {
    public:
        Crit3DHydrallBiomass();

        double total;
        double leaf;
        double sapwood;
        double fineRoot;
    };

    class Crit3DHydrallAllocationCoefficient {
    public:
        Crit3DHydrallAllocationCoefficient();

        double toFoliage;
        double toFineRoots;
        double toSapwood;
    };


    class Crit3DHydrallMaps
    {
    private:

    public:
        //sapwood, foliage, fine root
        bool isInitialized;
        gis::Crit3DRasterGrid treeSpeciesMap;
        gis::Crit3DRasterGrid plantHeight;
        gis::Crit3DRasterGrid* criticalTranspiration;
        gis::Crit3DRasterGrid* criticalSoilWaterPotential;
        gis::Crit3DRasterGrid* minLeafWaterPotential;

        gis::Crit3DRasterGrid* yearlyPrec;
        gis::Crit3DRasterGrid* yearlyET0;

        gis::Crit3DRasterGrid* treeNetPrimaryProduction; //SAVE
        gis::Crit3DRasterGrid* treeBiomassFoliage; //SAVE
        gis::Crit3DRasterGrid* treeBiomassRoot; //SAVE
        gis::Crit3DRasterGrid* treeBiomassSapwood; //SAVE
        gis::Crit3DRasterGrid* understoreyNetPrimaryProduction; //SAVE
        gis::Crit3DRasterGrid* understoreyBiomassFoliage; //SAVE
        gis::Crit3DRasterGrid* understoreyBiomassRoot; //SAVE

        gis::Crit3DRasterGrid* outputC;

        Crit3DHydrallMaps();
        ~Crit3DHydrallMaps();

        void initialize(const gis::Crit3DRasterGrid& DEM);
    };


    class Crit3DHydrall{
    public:

        Crit3DHydrall();
        //~Crit3DHydrall();

        void initialize();

        int firstMonthVegetativeSeason = 1;
        bool isFirstYearSimulation;
        Crit3DDate currentDate;
        Crit3DHydrallState stateVariable;
        Crit3DHydrallBigLeaf sunlit,shaded, understorey;
        Crit3DHydrallWeatherVariable weatherVariable;
        Crit3DHydrallEnvironmentalVariable environmentalVariable;
        Crit3DHydrallParameterWangLeuning parameterWangLeuning;
        Crit3DHydrallPlant plant;
        Crit3DHydrallSoil soil;
        Crit3DHydrallLightExtinctionCoefficient directLightExtinctionCoefficient;
        Crit3DHydrallLightExtinctionCoefficient diffuseLightExtinctionCoefficient;
        Crit3DHydrallDeltaTimeOutputs deltaTime;
        Crit3DHydrallNitrogen nitrogenContent;
        Crit3DHydrallBiomass treeBiomass, understoreyBiomass;
        Crit3DHydrallStatePlant statePlant;
        Crit3DHydrallAllocationCoefficient allocationCoefficient;
        bool printHourlyRecords = false;
        double maxIterationNumber;
        double understoreyLeafAreaIndexMax;
        double cover = 1; // TODO

        std::vector<int> conversionTableVector;

        double annualGrossStandGrowth;
        double internalCarbonStorage ; // [kgC m-2]
        double carbonStock;

        //gasflux results
        std::vector<double> treeTranspirationRate;          //molH2O m^-2 s^-1
        double treeAssimilationRate;
        std::vector<double> understoreyTranspirationRate;
        double understoreyAssimilationRate;
        double totalTranspirationRate; //molH2O m^-2 s^-1

        double getOutputC() { return outputC; }
        void setElevation(double myElevation) {elevation = myElevation;}
        void setYear(int myYear) { year = myYear;}

        double moistureCorrectionFactorOld(int index);
        double moistureCorrectionFactor(int index);
        double understoreyRespiration();
        void radiationAbsorption();
        void setSoilVariables(int iLayer, int currentNode, float checkFlag, double waterContent, double waterContentFC, double waterContentWP, double clay, double sand, double thickness, double bulkDensity, double waterContentSat, double kSat, double waterPotential);
        void setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, Crit3DDate currentDate, double sunElevation,double meanTemp30Days,double et0);
        bool setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, double meanTemp30Days,double et0);
        void setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex, double et0);
        void setPlantVariables(int forestIndex, double chlorophyllContent, double height, double psiMinimum);
        bool computeHydrallPoint();
        double getCO2(Crit3DDate myDate);
        //double getPressureFromElevation(double myTemperature, double myElevation);
        double computeLAI(Crit3DDate myDate);
        //double meanLastMonthTemperature(double previousLastMonthTemp, double myInstantTemp);
        double photosynthesisAndTranspiration();
        double photosynthesisAndTranspirationUnderstorey();
        void leafTemperature();
        void aerodynamicalCoupling();
        void preliminaryComputations(double diffuseIncomingPAR, double diffuseReflectionCoefficientPAR, double directIncomingPAR, double directReflectionCoefficientPAR,
                                                     double diffuseIncomingNIR, double diffuseReflectionCoefficientNIR, double directIncomingNIR, double directReflectionCoefficientNIR,
                                     double scatteringCoefPAR, double scatteringCoefNIR, std::vector<double> &dum);
        double leafWidth();
        void upscale();
        inline double acclimationFunction(double Ha , double Hd, double leafTemp, double entropicTerm,double optimumTemp);
        void photosynthesisKernel(double COMP, double GAC, double GHR, double GSCD, double J, double KC, double KO
                                  , double RD, double RNI, double STOMWL, double VCmax, double *ASS, double *GSC, double *TR);
        void carbonWaterFluxesProfile();
        void cumulatedResults();
        double plantRespiration();
        double computeEvaporation();
        inline double soilTemperatureModel();
        double temperatureFunction(double temperature);
        bool growthStand();
        bool simplifiedGrowthStand();
        void resetStandVariables();
        void optimal();
        void rootfind(double &allf, double &allr, double &alls, bool &sol);

        void setStateVariables(const Crit3DHydrallMaps &stateMap, int row, int col);
        void saveStateVariables(Crit3DHydrallMaps &stateMap, int row, int col);

        //void getPlantAndSoilVariables(Crit3DHydrallMaps &map, int row, int col);
        void updateCriticalPsi();
        double cavitationConditions();
        double getFirewoodLostSurfacePercentage(double percentageSurfaceLostByFirewoodAtReferenceYear, int simulationYear);

    private:
        double outputC;
        double elevation;
        int year;
        void nullPhotosynthesis();

    };


#endif // HYDRALL_H
