#ifndef GRAPEVINE_H
#define GRAPEVINE_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef BIOMASS_H
        #include "biomass.h"
    #endif
    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif

    #ifndef _VECTOR_
        #include <vector>
    #endif
    #ifndef _MAP_
        #include <map>
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


    enum phenoStage {endoDormancy, ecoDormancy , budBurst , flowering , fruitSet, veraison, physiologicalMaturity, vineSenescence};
    enum rootDistribution {CARDIOID_DISTRIBUTION, BLACKBODY_DISTRIBUTION , GAMMA_DISTRIBUTION};
    enum TfieldOperation {irrigationOperation, grassSowing, grassRemoving, trimming, leafRemoval,
                          clusterThinning, harvesting, tartaricAnalysis};
    enum Crit3DLanduse {landuse_nodata, landuse_bare, landuse_vineyard};

    const std::map<std::string, Crit3DLanduse> landuseNames = {
        { "UNDEFINED", landuse_nodata },
        { "BARESOIL", landuse_bare },
        { "VINEYARD", landuse_vineyard}
    };

    struct TstateGrowth{
        double cumulatedBiomass;
        double fruitBiomass;
        double shootLeafNumber;
        double meanTemperatureLastMonth;
        double tartaricAcid;
        double leafAreaIndex;
        double fruitBiomassIndex;
        int isHarvested;

        void initialize()
        {
            cumulatedBiomass = 0.;
            fruitBiomass = 0.;
            shootLeafNumber = 0.;
            meanTemperatureLastMonth = 10.;
            tartaricAcid = NODATA;
            isHarvested = 0;
        }
    };

    struct TstatePheno{
        double chillingState;
        double forceStateBudBurst;
        double forceStateVegetativeSeason;
        double meanTemperature;
        double degreeDaysFromFirstMarch;        // based on 0 Celsius
        double degreeDaysAtFruitSet;            // based on 0 Celsius
        double daysAfterBloom;
        double stage;
        double cumulatedRadiationFromFruitsetToVeraison;

        void initialize()
        {
            chillingState = 86.267 ;
            forceStateVegetativeSeason = 0.;
            forceStateBudBurst = 0.415;
            meanTemperature = 3.93;
            degreeDaysFromFirstMarch = 0.0;
            degreeDaysAtFruitSet = NODATA;
            daysAfterBloom = -1;
            stage = endoDormancy ;
            cumulatedRadiationFromFruitsetToVeraison = 0.;
        }
    };

    struct ToutputPlant
    {
        double transpirationNoStress;
        double grassTranspiration;
        double evaporation;
        double brixBerry;
        double brixMaximum;
    };

    struct TstatePlant
    {
        TstateGrowth stateGrowth;
        TstatePheno statePheno;
        ToutputPlant outputPlant;
    };

    struct TparameterBindiMiglietta
    {
        double radiationUseEfficiency ;
        double d,f ;
        double fruitBiomassOffset , fruitBiomassSlope ;
    };

    struct TparameterBindiMigliettaFix
    {
        double a,b,c;
        double shadedSurface;
        double extinctionCoefficient ;
        //double baseTemperature ;
        //double tempMaxThreshold ;

        void initialize()
        {
            a =  -0.28;
            b = 0.04;
            c = -0.015;
            //baseTemperature = 10; //Celsius deg
            //tempMaxThreshold = 35; //Celsius deg
            extinctionCoefficient = 0.5;
            shadedSurface = 0.8;
        }
    };

    struct TparameterWangLeuning{
        double sensitivityToVapourPressureDeficit;
        double alpha;
        double psiLeaf;                 // kPa
        double waterStressThreshold;
        double maxCarboxRate;           // Vcmo at optimal temperature
    };

    struct TparameterWangLeuningFix{
        double optimalTemperatureForPhotosynthesis;
        double stomatalConductanceMin;
    };

    struct TparameterPhenoVitis{
        double co1;
        double criticalChilling;
        double criticalForceStateFruitSet;
        double criticalForceStateFlowering;
        double criticalForceStateVeraison;
        double criticalForceStatePhysiologicalMaturity;
        double degreeDaysAtVeraison;
    };

    struct TparameterPhenoVitisFix{
        int startingDay;
        double a,optimalChillingTemp,co2;
        void initialize()
        {
            a = 0.005;
            optimalChillingTemp = 2.8;
            co2 = -0.015;
            startingDay = 244;
        }
    };

    struct TVineCultivar {
        int id;
        TparameterBindiMiglietta parameterBindiMiglietta;
        TparameterWangLeuning parameterWangLeuning;
        TparameterPhenoVitis parameterPhenoVitis;
    };

    struct TtrainingSystem {
        int id;
        float shootsPerPlant;
        float rowWidth;
        float rowHeight;
        float rowDistance;
        float plantDistance;
    };

    struct Crit3DModelCase {
        int id;
        Crit3DLanduse landuse;
        int soilIndex;

        float shootsPerPlant;
        float plantDensity;
        float maxLAIGrass;
        int trainingSystem;
        float maxIrrigationRate;        //[mm/h]

        int soilLayersNr;
        double soilTotalDepth;
        double* rootDensity;
        double* grassRootDensity;
        double* fallowRootDensity;

        TVineCultivar* cultivar;
    };

    struct TsoilProfileTest {
        double* waterContent;
        double* psi;
        double* temp;
    };

    struct Vine3D_Nitrogen {
        double interceptLeaf, slopeLeaf, leafNitrogen;
        double leaf , stem , root , shoot;
    };


    struct Vine3D_SunShade {

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

    struct Vine3D_DeltaTimeResults {

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

    struct Vine3D_Biomass {

        double total ;
        double leaf ;
        double sapwood ;
        double fineRoot ;
        double shoot ;
    };


    struct TqualityBerry {
        double acidity;
        double pH;
        double sugar;
    };

    class Vine3D_Grapevine {

    private:
        TstatePlant statePlant;

        double simulationStepInSeconds;

        int myDoy;
        int myYear;
        double myHour;

        double myAtmosphericPressure;
        double myPrec;
        double myMeanDailyTemperature;
        double myInstantTemp;
        double myRelativeHumidity;
        double myVaporPressureDeficit;
        double myAirVapourPressure;
        double myIrradiance, myDiffuseIrradiance, myDirectIrradiance, myLongWaveIrradiance;
        double myEmissivitySky;
        double mySlopeSatVapPressureVSTemp;
        double mySunElevation;
        double myCloudiness;
        double myWindSpeed;
        double mySoilTemp;

        int nrMaxLayers;

        double wiltingPoint;
        double psiSoilAverage;
        double psiFieldCapacityAverage;

        //double* layerRootDensity;
        double totalStomatalConductance, totalStomatalConductanceNoStress ;
        double transpirationInstant;
        double* currentProfile;
        double* transpirationInstantLayer;          //molH2O m^-2 s^-1
        double* transpirationLayer;                 //mm
        double* transpirationCumulatedGrass;
        double transpirationInstantNoStress;
        double* fractionTranspirableSoilWaterProfile;
        double* stressCoefficientProfile;
        double fractionTranspirableSoilWaterAverage;
        double assimilationInstant;

        TparameterBindiMigliettaFix parameterBindiMigliettaFix;
        TparameterWangLeuningFix parameterWangLeuningFix;
        TparameterPhenoVitisFix parameterPhenoVitisFix;

        double potentialBrix;
        double chlorophyllContent;
        double leafNumberRate ;
        double stepPhotosynthesis ;
        double myPlantHeight ;
        double myLeafWidth ;
        double directLightK, diffuseLightK, diffuseLightKPAR, diffuseLightKNIR, directLightKPAR, directLightKNIR;
        bool isAmphystomatic ;
        double specificLeafArea ;
        double alphaLeuning ;
        //double leafNitrogen ;
        //double entropicFactorCarboxyliation,entropicFactorElectronTransporRate ;

        Vine3D_SunShade shaded ;
        Vine3D_SunShade sunlit ;
        Vine3D_Nitrogen nitrogen ;
        Vine3D_DeltaTimeResults deltaTime ;
        Vine3D_Biomass biomass ;

    private: // functions
        void photosynthesisRadiationUseEfficiency(TVineCultivar* cultivar);
        double getCO2();
        double acclimationFunction(double Ha, double Hd, double leafTemp,double entropicTerm,double optimumTemp);
        double acclimationFunction2(double preFactor, double expFactor, double leafTemp,double optimumTemp);
        void weatherVariables();
        void radiationAbsorption ();
        void aerodynamicalCoupling ();
        void upscale(TVineCultivar* cultivar);
        void photosynthesisAndTranspiration(Crit3DModelCase *modelCase);
        void carbonWaterFluxes(TVineCultivar* cultivar);
        void carbonWaterFluxesProfile(Crit3DModelCase *modelCase);
        void carbonWaterFluxesProfileNoStress(Crit3DModelCase *modelCase);
        void photosynthesisKernel(TVineCultivar *cultivar, double COMP, double GAC, double GHR, double GSCD, double J, double KC, double KO
                                  , double RD, double RNI, double STOMWL, double VCMAX, double *ASS, double *GSC, double *TR);
        void photosynthesisKernelSimplified(TVineCultivar *cultivar, double COMP, double GSCD, double J, double KC, double KO
                             , double RD, double STOMWL, double VCmax, double *ASS, double *GSC, double *TR);
        void cumulatedResults(Crit3DModelCase *modelCase);
        double plantRespiration();
        double temperatureMoistureFunction(double temperature);
        void plantInterception(double fieldCoverByPlant);
        double meanLastMonthTemperature(double temperature);

        double* waterSuctionDistribution(int nrLayers, double *layerRootDensity, double *psiSoil);

        double chillingRate(double temp, double aParameter, double cParameter);
        double criticalForceState(double chillState,double co1 , double co2);
        double forceStateFunction(double forceState , double temp);
        double forceStateFunction(double forceState , double temp, double degDays);

        void computePhenology(bool computeDaily, bool* isVegSeason, TVineCultivar* cultivar);
        double leafWidth();
        void leafTemperature();
        double getLAIGrass(bool isShadow, double laiMax);
        void getLAIVine(Crit3DModelCase *vineField);

        double getWaterStressByPsiSoil(double myPsiSoil,double psiSoilStressParameter,double exponentialFactorForPsiRatio);
        double getWaterStressSawFunction(int index, TVineCultivar *cultivar);
        //bool getExtractedWaterFromGrassTranspirationandEvaporation(double* myWaterExtractionProfile);
        double getWaterStressSawFunctionAverage(TVineCultivar* cultivar);
        double getGrassTranspiration(double stress, double laiGrassMax, double sensitivityToVPD, double fieldCoverByPlant);
        double getFallowTranspiration(double stress, double laiGrassMax, double sensitivityToVPD);
        void grassTranspiration(Crit3DModelCase *modelCase);
        void fallowTranspiration(Crit3DModelCase *modelCase, double laiGrassMax, double sensitivityToVPD);
        void getFixSimulationParameters();
        double getLaiStressCoefficient();
        void getPotentialBrix();
        void initializeWaterStress(Crit3DModelCase *modelCase);
        double gompertzDistribution(double stage);
        double getTartaricAcid();
        double soilTemperatureModel();
        void fruitBiomassRateDecreaseDueToRainfall();

    public:
        Vine3D_Grapevine();

        //void initializeGrapevineModel(TVineCultivar* cultivar, double secondsPerStep);
        bool initializeLayers(int myMaxLayers);
        bool initializeStatePlant(int doy, Crit3DModelCase *vineField);
        void resetLayers();

        void setRootDensity(Crit3DModelCase *modelCase, soil::Crit3DSoil* mySoil, std::vector<double> layerDepth, std::vector<double> layerThickness,
                            int nrLayersWithRoot, int nrUpperLayersWithoutRoot, rootDistribution type, double mode, double mean);
        void setGrassRootDensity(Crit3DModelCase* modelCase, soil::Crit3DSoil *mySoil, std::vector<double> layerDepth, std::vector<double> layerThickness,
                                 double startRootDepth, double totalRootDepth);
        void setFallowRootDensity(Crit3DModelCase* modelCase, soil::Crit3DSoil* mySoil, std::vector<double> layerDepth, std::vector<double> layerThickness,
                                 double startRootDepth, double totalRootDepth);

        void setDate (Crit3DTime myTime);
        bool setWeather(double meanDailyTemp, double temp, double irradiance ,
                double prec , double relativeHumidity , double windSpeed, double atmosphericPressure);
        bool setDerivedVariables (double diffuseIrradiance, double directIrradiance,
                double cloudIndex, double sunElevation);
        bool setSoilProfile(Crit3DModelCase *modelCase, double* myWiltingPoint, double *myFieldCapacity,
                            double *myPsiSoilProfile , double *mySoilWaterContentProfile,
                            double* mySoilWaterContentFC, double* mySoilWaterContentWP);
        bool setStatePlant(TstatePlant myStatePlant, bool isVineyard);

        TstatePlant getStatePlant();
        ToutputPlant getOutputPlant();
        double* getExtractedWater(Crit3DModelCase* modelCase);
        //bool getOutputPlant(int hour, ToutputPlant *outputPlant);
        double getStressCoefficient();
        double getRealTranspirationGrapevine(Crit3DModelCase *modelCase);
        double getRealTranspirationGrass(Crit3DModelCase *modelCase);
        bool fieldBookAction(Crit3DModelCase* vineField, TfieldOperation action, float quantity);
        double getRootDensity(Crit3DModelCase *modelCase, int myLayer);

        bool compute(bool computeDaily, int secondsPerStep, Crit3DModelCase *vineField, double chlorophyll);
    };


#endif // GRAPEVINE_H
