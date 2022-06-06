#ifndef CARBON_H
#define CARBON_H

// Translation of the whole chemical reactions algorithms from LEACHM (2013)
// reviewed by Gabriele Antolini (2007-2009)
// This module computes the Nitrogen cycle
// The Nitrogen cycle is substituted through the LEACHN model (Hutson & Wagenet 1992)
// and some improvement added by Ducco

// COSTANTS---------------------------------------------------------------------------------------------------
#define KDW_NO3 0.00016416  // in water NO3 diffusion coefficient
                            // [m2 d-1] (Lide, 1997: 1.9 x 10-5 cm2 s-1)
#define FERTILIZER_UREA  9

class Crit3DCarbonNitrogenWholeProfile
{


    private:


    //rates ------------------------------------------------------------------------------------------------
    // tabulated values
    float rate_C_LitterMin;  //[d-1] litter mineralization rate
    float rate_C_HumusMin; //[d-1] humus mineralization rate
    float rate_N_NH4_Volatilization; //[d-1] ammonium volatilization rate
    float rate_Urea_Hydr; //[d-1] urea hydrolisis rate
    float rate_N_Nitrification; //[d-1] nitrification rate
    float limRatio_nitr; // [] limiting NO3/NH4 ratio in solution for nitrification
    float rate_N_Denitrification; //[d-1] denitrifition rate
    float max_afp_denitr; // [] maximum air filled porosity fraction for denitrification onset
    float Csat_denitr; // [mg l-1] semisaturation constant for denitrification
    float Kd_NH4;// [l kg-1] partition coefficient for ammonium
    float FE; // [] synthesis efficiency factor
    float FH; // [] humification factor
    float Q10; //[] temperature rate correction: increase factor every 10 °C
    float baseTemperature; // [°C] temperature rate correction: base temperature

    // values corrected for Temperature and RH

    float actualRate_C_HumusMin; //
    float actualRate_C_LitterToHumus; //
    float actualRate_C_LitterToCO2; //
    float actualRate_C_LitterToBiomass; //
    float actualRate_N_LitterMin; //           '[] rate of N mineralization in litter
    float actualRate_N_LitterImm; //          '[] rate of N immobilization in litter
    float actualRate_N_Nitrification; //
    float actualRate_N_Denitrification; //
    float actualRate_Urea_Hydr; //

    // fix variables --------------------------------------------------------------------------------------------
    float ratio_CN_humus;//[] rapporto C/N pool humus
    float ratio_CN_biomass; //[] rapporto C/N pool biomass


    public:
    float litterIniC; //[kg ha-1] initial litter carbon
    float LITTERINI_C_DEFAULT = 1200; //[kg ha-1] initial litter carbon (default)
    float litterIniN; //[kg ha-1] initial litter nitrogen
    float LITTERINI_N_DEFAULT = 40; //[kg ha-1] initial litter nitrogen (default)
    float litterIniProf ; //[cm] initial litter depth
    float LITTERINI_PROF_DEFAULT = 30; //[cm] initial litter depth (default)

    // flags -------------------------------------------------------------------------------------------------
    int flagSO; // 1: computes SO; 0: SO set at the default value
    int flagLocalOS; //1: Initializes the profile of SO without keeping that of soil
    bool flagWaterTableWashing; // if true: the solute is completely leached in groundwater
    bool flagWaterTableUpward; // if true: capillary rise is allowed

    // daily values---------------------------------------------------------------------------------
        // Nitrogen in soil
            // contents
    public:
    float N_humusGG; //[g m-2] azoto nell'humus
    float N_litterGG; //[g m-2] azoto nel litter
    float N_NH4_AdsorbedGG; //[g m-2] azoto ammoniacale adsorbito al giorno attuale
    float N_NH4_AdsorbedBeforeGG; //[g m-2] azoto ammoniacale adsorbito al giorno precedente
    double ProfiloNO3; //[g m-2] azoto nitrico totale nel profilo
    double ProfiloNH4; //[g m-2] azoto ammoniacale totale nel profilo
    double BilFinaleNO3; //[g m-2] azoto nitrico: errore di bilancio
    double BilFinaleNH4; //[g m-2] azoto ammoniacale: errore di bilancio
            'flussi
    float PrecN_NO3GG; //[g m-2] azoto nitrico apportato dalle precipitazioni
    float PrecN_NH4GG; //[g m-2] azoto ammoniacale apportato dalle precipitazioni
    float N_NO3_fertGG; //[g m-2] azoto nitrico apportato dalle fertilizzazioni
    float N_NH4_fertGG; //[g m-2] azoto ammoniacale apportato dalle fertilizzazioni
    float N_min_litterGG; //[g m-2] azoto mineralizzato dal litter
    private:
    float N_imm_l_NH4GG; //[g m-2] azoto ammoniacale immobilizzato nel litter
    float N_imm_l_NO3GG; //[g m-2] azoto nitrico immobilizzato nel litter
    public:
    float N_min_humusGG; //[g m-2] azoto mineralizzato dall'humus
    float N_litter_humusGG; //[g m-2] azoto da litter a humus
    float N_NH4_volGG; //[g m-2] azoto volatilizzato per l'intero profilo
    float N_nitrifGG; //[g m-2] azoto trasformato da ammonio a nitrato
    float N_Urea_HydrGG; //[g m-2] azoto ureico idrolizzato in ammonio
    float Flux_NO3GG; //[g m-2] lisciviazione di nitrato
    float Flux_NH4GG; //[g m-2] lisciviazione di ammonio
    float N_NO3_runoff0GG; //[g m-2] azoto nitrico perso nel ruscellamento superficiale
    float N_NH4_runoff0GG; //[g m-2] azoto ammoniacale perso nel ruscellamento superficiale
    float N_NO3_runoffGG; //[g m-2] azoto nitrico perso nel ruscellamento sottosuperficiale
    float N_NH4_runoffGG; //[g m-2] azoto ammoniacale perso nel ruscellamento sottosuperficiale
    // uptake
    float Date_N_EndCrop As Date                   '[date] data di fine coltura per N (raccolta o rottura prato)
    float Date_N_Plough As Date                    '[date] data di lavorazione per interramento residui N
    float N_Uptakable; //[g m-2] assorbimento massimo della coltura per ciclo colturale
    private:
    float maxRate_LAI_Ndemand; //[g m-2 d-1 LAI-1] maximum demand for unit LAI increment
    float CN_RATIO_NOTHARVESTED = 30; //[] C/N ratio in not harvested crop
    public:
    float N_CropToHarvest; //[g m-2] azoto assorbito destinato a raccolta
    float N_CropToResidues; //[g m-2] azoto assorbito destinato a residui sul campo
    float N_Roots; //[g m-2] azoto assorbito presente nelle radici
    private:
    float N_ratioHarvested; //[] ratio of harvested crop
    float N_ratioResidues; //[] ratio of residues not harvested left above the soil
    float N_ratioRoots; //[] ratio of living roots left at harvest
    public:
    float N_PotentialDemandCumulated; //[g m-2] azoto potenzialmente asportabile cumulato nel tempo in Attuale
    float N_DailyDemand; //[g m-2] azoto potenzialmente asportabile al giorno attuale
    float N_DailyDemandMaxCover; //[g m-2] azoto potenzialmente asportabile al giorno di massima copertura (LAI max)
    float N_UptakeMax; //[g m-2] azoto asportabile massimo
    float N_UptakeDeficit; //[g m-2] azoto non assorbito dalla pianta rispetto alla domanda
    float* N_deficit_daily; //[g m-2] array of deficit in the last days (defined by N_deficit_max_days)
    int N_deficit_max_days; //[d] giorni in cui è il deficit rimane disponibile
    float N_NH4_uptakeGG; //[g m-2] azoto ammoniacale assorbito dalla coltura
    float N_NO3_uptakeGG; //[g m-2] azoto nitrico assorbito dalla coltura
    float N_denitrGG; //[g m-2] azoto perso per denitrificazione
        //carbon
            //contents
    float C_humusGG; //[g m-2] carbonio nell'humus
    float C_litterGG; //[g m-2] carbonio nel litter
            //flussi
    float C_litter_humusGG; //[g m-2] carbonio traferito da litter a humus
    float C_litter_litterGG; //[g m-2] carbonio in riciclo interno litter
    float C_min_humusGG; //[g m-2] carbonio perso come CO2 nella mineralizzazione dell'humus
    float C_min_litterGG; //[g m-2] carbonio perso come CO2 nella mineralizzazione del litter







    Crit3DCarbonNitrogenWholeProfile();
};

class Crit3DCarbonNitrogenLayer
{
    private:
    // correction factors
    float  temperatureCorrectionFactor;// [] correction factor for soil temperature
    float waterCorrecctionFactor; //[] correction factor for soil water content
    float waterCorrecctionFactorDenitrification; //[] correction factor for soil water content (denitrification)

    // nitrogen
        // contents
    public:
    float N_NO3;//[g m-2] Nitrogen in form of Nitrates
    float N_NH4;//[g m-2] Nitrogen in form of Ammonium
    float N_NH4_Adsorbed;//[g m-2] Nitrogen in form of adsorbed Ammonium
    float N_NH4_Sol;//[g m-2] Nitrogen in form of dissolved Ammonium
    float N_urea;//[g m-2] Nitrogen in form of Urea
    float N_humus;//[g m-2] Nitrogen in humus
    float N_litter;//[g m-2] Nitrogen litter
        // fluxes
    private:

    float N_NO3_uptake;//[g m-2] azoto nitrico asportato dalla coltura
    float N_NH4_uptake;//[g m-2] azoto ammoniacale asportato dalla coltura
    float N_min_litter;//[g m-2] azoto mineralizzato dal litter
    float N_imm_l_NH4;//[g m-2] azoto ammoniacale immobilizzato nel litter
    float N_imm_l_NO3;//[g m-2] azoto nitrico immobilizzato nel litter
    float N_min_humus;//[g m-2] azoto mineralizzato dall'humus
    float N_litter_humus;//[g m-2] azoto da litter a humus
    float N_vol;//[g m-2] azoto ammoniacale volatilizzato
    float N_denitr;//[g m-2] azoto perso per denitrificazione
    float N_nitrif;//[g m-2] azoto trasformato da ammonio a nitrato
    float N_Urea_Hydr;//[g m-2] azoto ureico idrolizzato in ammonio
    float N_NO3_runoff;//[g m-2] azoto nitrico perso nel ruscellamento superficiale o sottosuperficiale
    float N_NH4_runoff;//[g m-2] azoto ammoniacale perso nel ruscellamento superficiale o sottosuperficiale
            //ratios
    public:
    float ration_CN_litter; //[-] rapporto C/N litter

    // carbon
        //contents
    public:
    float C_humus; //[g m-2] carbonio nell'humus
    float C_litter; //[g m-2] carbonio nel litter
            // fluxes
    private:
    float C_litter_humus; //[g m-2] carbonio trasferito da litter a humus
    float C_litter_litter; //[g m-2] carbonio in riciclo interno litter
    float C_min_humus; //[g m-2] carbonio perso come CO2 nella mineralizzazione dell'humus
    float C_min_litter; //[g m-2] carbonio perso come CO2 nella mineralizzazione del litter
    float C_denitr_humus; //[g m-2] carbonio humus perso come CO2 nella denitrificazione
    float C_denitr_litter; //[g m-2] carbonio litter perso come CO2 nella denitrificazione



    Crit3DCarbonNitrogenLayer();
};


#endif // CARBON_H










