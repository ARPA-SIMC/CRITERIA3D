#ifndef CARBON_NITROGEN_H
#define CARBON_NITROGEN_H

    #include "soil.h"

    // Translation of the whole chemical reactions algorithms from LEACHM (2013)
    // reviewed by Gabriele Antolini (2007-2009)
    // The Nitrogen cycle is substituted through the LEACHN model (Hutson & Wagenet 1992)

    // NO3 diffusion coefficient in water [m2 d-1] (Lide, 1997: 1.9 x 10-5 cm2 s-1)
    #define KDW_NO3 0.00016416

    #define FERTILIZER_UREA  9

    struct TactualRate
    {
        double C_humusMin;        //
        double C_litterToHumus;   //
        double C_litterToCO2;     //
        double C_litterToBiomass; //
        double N_litterMin;       // [] rate of N mineralization in litter
        double N_litterImm;       // [] rate of N immobilization in litter
        double N_nitrification;   //
        double N_denitrification; //
        double ureaHydr;          //
    };

    struct TFlag
    {
        int SOM;                        // 1: computes SO; 0: SO set at the default value
        int localOS;                    // 1: Initializes the profile of SO without keeping that of soil
        bool waterTableWashing;         // if true: the solute is completely leached in groundwater
        bool waterTableUpward;          // if true: capillary rise is allowed
    };

    struct TNitrogenTotalProfile
    {

        double humusGG;                //[g m-2] Nitrogen within humus
        double litterGG;               //[g m-2] Nitrogen within litter
        double NH4_adsorbedGG;         //[g m-2] adsorbed Ammonium in the current day
        double NH4_adsorbedBeforeGG;   //[g m-2] adsorbed Ammonium in the previous day
        double profileNO3;              //[g m-2] N-NO3 in the whole profile
        double profileNH4;              //[g m-2] N-NH4 in the whole profile
        double balanceFinalNO3;            //[g m-2] N-NO3: budget error
        double balanceFinalNH4;            //[g m-2] N-NH4: budget error
        //fluxes
        double prec_NO3GG;              //[g m-2] NO3 from rainfall
        double prec_NH4GG;              //[g m-2] NH4 from rainfall
        double NO3_fertGG;             //[g m-2] NO3 from fertilization
        double NH4_fertGG;             //[g m-2] NH4 from fertilization
        double min_litterGG;           //[g m-2] mineralized Nitrogen from litter


        double imm_l_NH4GG;            //[g m-2] NH4 immobilized in litter
        double imm_l_NO3GG;            //[g m-2] NO3 immobilized in litter

        double min_humusGG;            //[g m-2] mineralized Nitrogen from humus
        double litter_humusGG;         //[g m-2] Nitrogen from litter to humus
        double NH4_volGG;              //[g m-2] Volatilized NH4 in the whole profile
        double nitrifGG;               //[g m-2] Nitrogen from NH4 to NO3
        double urea_hydrGG;            //[g m-2] Hydrolyzed urea urea to NH4
        double flux_NO3GG;               //[g m-2] NO3 leaching flux
        double flux_NH4GG;               //[g m-2] NH4 leaching flux
        double NO3_runoff0GG;          //[g m-2] NO3 lost through surface run off
        double NH4_runoff0GG;          //[g m-2] NH4 lost through surface run off
        double NO3_runoffGG;           //[g m-2] NO3 lost through subsurface run off
        double NH4_runoffGG;           //[g m-2] NH4 lost through subsurface run off

        double uptakable;

        double cropToHarvest;          //[g m-2] Nitrogen absorbed in harvest
        double cropToResidues;         //[g m-2] Nitrogen absorbed in crop residues
        double roots;                  //[g m-2] Nitrogen absorbed in roots

        double ratioHarvested;         //[] ratio of harvested crop
        double ratioResidues;          //[] ratio of residues not harvested left above the soil
        double ratioRoots;             //[] ratio of living roots left at harvest

        double potentialDemandCum;     //[g m-2] cumulated potential Nitrogen at current date
        double dailyDemand;            //[g m-2] potential Nitrogen at current date
        double dailyDemandMaxCover;    //[g m-2] potential Nitrogen at max cover day (LAI_MAX)
        double uptakeMax;              //[g m-2] Max Nitrogen uptake
        double uptakeDeficit;          //[g m-2] Nitrogen deficit: not absorbed Nitrogen with respect to Nitrogen demand

        int deficit_max_days;          //[d] nr days with available deficit
        double NH4_uptakeGG;           //[g m-2] NH4 absorbed by the crop
        double NO3_uptakeGG;           //[g m-2] NO3 absorbed by the crop
        double denitrGG;               //[g m-2] Lost Nitrogen by denitrification
    };

    struct TCarbonTotalProfile
    {
        double humusGG;                //[g m-2] C in humus
        double litterGG;               //[g m-2] C in litter
        // flux
        double litter_humusGG;         //[g m-2] C from litter to humus
        double litter_litterGG;        //[g m-2] C recycled within litter
        double min_humusGG;            //[g m-2] C lost as CO2 by humus mineralization
        double min_litterGG;           //[g m-2] C lost as CO2 by litter mineralization
    };

    class Crit3DCarbonNitrogenLayer
    {
    public:

        Crit3DCarbonNitrogenLayer();

        // NITROGEN
        double N_NO3;               // [g m-2] Nitrogen in form of Nitrates
        double N_NH4;               // [g m-2] Nitrogen in form of Ammonium
        double N_NH4_Adsorbed;      // [g m-2] Nitrogen in form of adsorbed Ammonium
        double N_NH4_Sol;           // [g m-2] Nitrogen in form of dissolved Ammonium
        double N_urea;              // [g m-2] Nitrogen in form of Urea
        double N_humus;             // [g m-2] Nitrogen in humus
        double N_litter;            // [g m-2] Nitrogen litter
        double N_NO3_uptake;        // [g m-2] NO3 crop uptake
        double N_NH4_uptake;        // [g m-2] NH4 crop uptake

        // CARBON
        double C_humus;             // [g m-2] C in humus
        double C_litter;            // [g m-2] C in litter

        // ratios
        double ratio_CN_litter;     // [-] ratio C/N in litter
        double ratio_CN_humus;

        // correction factors
        double temperatureCorrectionFactor;         // [] correction factor for soil temperature
        double waterCorrecctionFactor;              // [] correction factor for soil water content
        double waterCorrecctionFactorDenitrification;     // [] correction factor for soil water content (denitrification)

        // NITROGEN
        double N_min_litter;        // [g m-2] mineralized Nitrogen in litter
        double N_imm_l_NH4;         // [g m-2] NH4 immobilized in litter
        double N_imm_l_NO3;         // [g m-2] NO3 immobilized in litter
        double N_min_humus;         // [g m-2] mineralized Nitrogen in humus
        double N_litter_humus;      // [g m-2] N from litter to humus
        double N_vol;               // [g m-2] volatilized NH4
        double N_denitr;            // [g m-2] denitrified N
        float N_nitrif;             // [g m-2] N from NH4 to NO3
        double N_Urea_Hydr;         // [g m-2] hydrolyzed urea to NH4
        double N_NO3_runoff;        // [g m-2] NO3 lost through surface & subsurface run off
        double N_NH4_runoff;        // [g m-2] NH4 lost through surface & subsurface run off

        // CARBON
        double C_litter_humus;      // [g m-2] C for litter to humus
        double C_litter_litter;     // [g m-2] recycled Nitrogen within litter
        double C_min_humus;         // [g m-2] C lost as CO2 by humus mineralization
        double C_min_litter;        // [g m-2] C lost as CO2 by litter mineralization
        double C_denitr_humus;      // [g m-2] C in humus lost as CO2 by means of denitrification
        double C_denitr_litter;     // [g m-2] C in litter lost as CO2 by means of denitrification
    };


    // fertilizer parameters
    class Crit3DFertilizerProperties
    {
    public:
        int ID_FERTILIZER;
        int ID_TYPE;
        double N_content;                   // TODO units
        double N_NO3_percentage;
        double N_NH4_percentage;
        double N_org_percentage;
        double C_N_ratio;
        double fertilizerDepth;
        double quantity;

        Crit3DFertilizerProperties();
    };


    // model parameters
    class Crit3DCarbonNitrogenSettings
    {
    public:
        double rate_C_litterMin;             // [d-1] litter mineralization rate
        double rate_C_humusMin;              // [d-1] humus mineralization rate
        double rate_N_NH4_volatilization;    // [d-1] ammonium volatilization rate
        double rate_urea_hydr;               // [d-1] urea hydrolisis rate
        double rate_N_nitrification;         // [d-1] nitrification rate
        double limRatio_nitr;                // [] limiting NO3/NH4 ratio in solution for nitrification
        double rate_N_denitrification;       // [d-1] denitrifition rate
        double max_afp_denitr;               // [] maximum air filled porosity fraction for denitrification onset
        double constant_sat_denitr;          // [mg l-1] semisaturation constant for denitrification
        double Kd_NH4;                       // [l kg-1] partition coefficient for ammonium
        double FE;                           // [] synthesis efficiency factor
        double FH;                           // [] humification factor
        double Q10;                          // [] temperature rate correction: increase factor every 10 °C
        double baseTemperature;
        double CN_RATIO_NOTHARVESTED;
        double LITTERINI_C_DEFAULT;
        double LITTERINI_N_DEFAULT;
        double LITTERINI_DEPTH_DEFAULT;
        double ratioHumusCN;
        double ratioLitterCN;
        double ratioBiomassCN;

        Crit3DFertilizerProperties fertilizerProperties;

        Crit3DCarbonNitrogenSettings();
    };


    double convertToGramsPerM3(double myQuantity, soil::Crit3DLayer& soilLayer);
    double convertToGramsPerLiter(double myQuantity, soil::Crit3DLayer &soilLayer);
    double convertToGramsPerKg(double myQuantity, soil::Crit3DLayer &soilLayer);
    double CNRatio(double C, double N, int flagOrganicMatter);


#endif // CARBON_NITROGEN_H
