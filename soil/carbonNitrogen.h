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

#include "crit3dDate.h"

class Crit3DCarbonNitrogen
{
    private:
    // correction factors
    float temperatureCorrectionFactor;   // [] correction factor for soil temperature
    float waterCorrecctionFactor;        //[] correction factor for soil water content
    float waterCorrecctionFactorDenitrification;     //[] correction factor for soil water content (denitrification)

    // nitrogen
        // contents
    public:
    float N_NO3;            //[g m-2] Nitrogen in form of Nitrates
    float N_NH4;            //[g m-2] Nitrogen in form of Ammonium
    float N_NH4_Adsorbed;   //[g m-2] Nitrogen in form of adsorbed Ammonium
    float N_NH4_Sol;        //[g m-2] Nitrogen in form of dissolved Ammonium
    float N_urea;           //[g m-2] Nitrogen in form of Urea
    float N_humus;//[g m-2] Nitrogen in humus
    float N_litter;//[g m-2] Nitrogen litter
        // fluxes
    private:

    float N_NO3_uptake;//[g m-2] NO3 crop uptake
    float N_NH4_uptake;//[g m-2] NH4 crop uptake
    float N_min_litter;//[g m-2] mineralized Nitrogen in litter
    float N_imm_l_NH4;//[g m-2] NH4 immobilized in litter
    float N_imm_l_NO3;//[g m-2] NO3 immobilized in litter
    float N_min_humus;//[g m-2] mineralized Nitrogen in humus
    float N_litter_humus;//[g m-2] N from litter to humus
    float N_vol;//[g m-2] volatilized NH4
    float N_denitr;//[g m-2] denitrified N
    float N_nitrif;//[g m-2] N from NH4 to NO3
    float N_Urea_Hydr;//[g m-2] hydrolyzed urea to NH4
    float N_NO3_runoff;//[g m-2] NO3 lost through surface & subsurface run off
    float N_NH4_runoff;//[g m-2] NH4 lost through surface & subsurface run off
            //ratios
    public:
    float ratio_CN_litter; //[-] ratio C/N in litter

    // carbon
        //contents
    public:
    float C_humus; //[g m-2] C in humus
    float C_litter; //[g m-2] C in litter
            // fluxes
    private:
    float C_litter_humus; //[g m-2] C for litter to humus
    float C_litter_litter; //[g m-2] recycled Nitrogen within litter
    float C_min_humus; //[g m-2] C lost as CO2 by humus mineralization
    float C_min_litter; //[g m-2] C lost as CO2 by litter mineralization
    float C_denitr_humus; //[g m-2] C in humus lost as CO2 by means of denitrification
    float C_denitr_litter; //[g m-2] C in litter lost as CO2 by means of denitrification



    Crit3DCarbonNitrogen();
};

class Crit3DCarbonNitrogenWholeProfile
{


    private:


    //rates ------------------------------------------------------------------------------------------------
    // tabulated values
    float rate_C_litterMin;             //[d-1] litter mineralization rate
    float rate_C_humusMin;              //[d-1] humus mineralization rate
    float rate_N_NH4_volatilization;    //[d-1] ammonium volatilization rate
    float rate_urea_hydr;               //[d-1] urea hydrolisis rate
    float rate_N_nitrification;         //[d-1] nitrification rate
    float limRatio_nitr;                // [] limiting NO3/NH4 ratio in solution for nitrification
    float rate_N_denitrification;       //[d-1] denitrifition rate
    float max_afp_denitr;               // [] maximum air filled porosity fraction for denitrification onset
    float constant_sat_denitr;                  // [mg l-1] semisaturation constant for denitrification
    float Kd_NH4;                       // [l kg-1] partition coefficient for ammonium
    float FE;                           // [] synthesis efficiency factor
    float FH;                           // [] humification factor
    float Q10;                          //[] temperature rate correction: increase factor every 10 °C
    float baseTemperature;              // [°C] temperature rate correction: base temperature

    // values corrected for Temperature and RH

    float actualRate_C_humusMin;        //
    float actualRate_C_litterToHumus;   //
    float actualRate_C_litterToCO2;     //
    float actualRate_C_litterToBiomass; //
    float actualRate_N_litterMin;       // [] rate of N mineralization in litter
    float actualRate_N_litterImm;       // [] rate of N immobilization in litter
    float actualRate_N_nitrification;   //
    float actualRate_N_denitrification; //
    float actualRateUreaHydr;         //

    // fix variables --------------------------------------------------------------------------------------------
    float ratio_CN_humus;               //[] rapporto C/N pool humus
    float ratio_CN_biomass;             //[] rapporto C/N pool biomass


    public:
    float litterIniC;                   //[kg ha-1] initial litter carbon
    float LITTERINI_C_DEFAULT = 1200;   //[kg ha-1] initial litter carbon (default)
    float litterIniN;                   //[kg ha-1] initial litter nitrogen
    float LITTERINI_N_DEFAULT = 40;     //[kg ha-1] initial litter nitrogen (default)
    float litterIniProf ;               //[cm] initial litter depth
    float LITTERINI_PROF_DEFAULT = 30;  //[cm] initial litter depth (default)

    // flags -------------------------------------------------------------------------------------------------
    int flagSO;                         // 1: computes SO; 0: SO set at the default value
    int flagLocalOS;                    //1: Initializes the profile of SO without keeping that of soil
    bool flagWaterTableWashing;         // if true: the solute is completely leached in groundwater
    bool flagWaterTableUpward;          // if true: capillary rise is allowed

    // daily values---------------------------------------------------------------------------------
        // Nitrogen in soil
            // contents
    public:
    float N_humusGG;                //[g m-2] Nitrogen within humus
    float N_litterGG;               //[g m-2] Nitrogen within litter
    float N_NH4_adsorbedGG;         //[g m-2] adsorbed Ammonium in the current day
    float N_NH4_adsorbedBeforeGG;   //[g m-2] adsorbed Ammonium in the previous day
    double profileNO3;              //[g m-2] N-NO3 in the whole profile
    double profileNH4;              //[g m-2] N-NH4 in the whole profile
    double balanceFinalNO3;            //[g m-2] N-NO3: budget error
    double balanceFinalNH4;            //[g m-2] N-NH4: budget error
            //fluxes
    float precN_NO3GG;              //[g m-2] NO3 from rainfall
    float precN_NH4GG;              //[g m-2] NH4 from rainfall
    float N_NO3_fertGG;             //[g m-2] NO3 from fertilization
    float N_NH4_fertGG;             //[g m-2] NH4 from fertilization
    float N_min_litterGG;           //[g m-2] mineralized Nitrogen from litter
    private:
    float N_imm_l_NH4GG;            //[g m-2] NH4 immobilized in litter
    float N_imm_l_NO3GG;            //[g m-2] NO3 immobilized in litter
    public:
    float N_min_humusGG;            //[g m-2] mineralized Nitrogen from humus
    float N_litter_humusGG;         //[g m-2] Nitrogen from litter to humus
    float N_NH4_volGG;              //[g m-2] Volatilized NH4 in the whole profile
    float N_nitrifGG;               //[g m-2] Nitrogen from NH4 to NO3
    float N_urea_hydrGG;            //[g m-2] Hydrolyzed urea urea to NH4
    float Flux_NO3GG;               //[g m-2] NO3 leaching flux
    float Flux_NH4GG;               //[g m-2] NH4 leaching flux
    float N_NO3_runoff0GG;          //[g m-2] NO3 lost through surface run off
    float N_NH4_runoff0GG;          //[g m-2] NH4 lost through surface run off
    float N_NO3_runoffGG;           //[g m-2] NO3 lost through subsurface run off
    float N_NH4_runoffGG;           //[g m-2] NH4 lost through subsurface run off
    // uptake
    Crit3DDate date_N_endCrop;
    Crit3DDate date_N_plough;
    //float Date_N_EndCrop As Date    //[date] data di fine coltura per N (raccolta o rottura prato)
    //float Date_N_Plough As Date     //[date] data di lavorazione per interramento residui N
    float N_uptakable;              //[g m-2] assorbimento massimo della coltura per ciclo colturale
    private:
    float maxRate_LAI_Ndemand;      //[g m-2 d-1 LAI-1] maximum demand for unit LAI increment
    float CN_RATIO_NOTHARVESTED=30; //[] C/N ratio in not harvested crop
    public:
    float N_cropToHarvest;          //[g m-2] Nitrogen absorbed in harvest
    float N_cropToResidues;         //[g m-2] Nitrogen absorbed in crop residues
    float N_roots;                  //[g m-2] Nitrogen absorbed in roots
    private:
    float N_ratioHarvested;         //[] ratio of harvested crop
    float N_ratioResidues;          //[] ratio of residues not harvested left above the soil
    float N_ratioRoots;             //[] ratio of living roots left at harvest
    public:
    float N_potentialDemandCum;     //[g m-2] cumulated potential Nitrogen at current date
    float N_dailyDemand;            //[g m-2] potential Nitrogen at current date
    float N_dailyDemandMaxCover;    //[g m-2] potential Nitrogen at max cover day (LAI_MAX)
    float N_uptakeMax;              //[g m-2] Max Nitrogen uptake
    float N_uptakeDeficit;          //[g m-2] Nitrogen deficit: not absorbed Nitrogen with respect to Nitrogen demand
    float* N_deficit_daily;         //[g m-2] array of deficit in the last days (defined by N_deficit_max_days)
    int N_deficit_max_days;         //[d] nr days with available deficit
    float N_NH4_uptakeGG;           //[g m-2] NH4 absorbed by the crop
    float N_NO3_uptakeGG;           //[g m-2] NO3 absorbed by the crop
    float N_denitrGG;               //[g m-2] Lost Nitrogen by denitrification
        //carbon
            //contents
    float C_humusGG;                //[g m-2] C in humus
    float C_litterGG;               //[g m-2] C in litter
            //flussi
    float C_litter_humusGG;         //[g m-2] C from litter to humus
    float C_litter_litterGG;        //[g m-2] C recycled within litter
    float C_min_humusGG;            //[g m-2] C lost as CO2 by humus mineralization
    float C_min_litterGG;           //[g m-2] C lost as CO2 by litter mineralization


    Crit3DCarbonNitrogen *arrayCarbonNitrogen;




    Crit3DCarbonNitrogenWholeProfile();
};




#endif // CARBON_H
/*
 *
void soluteFluxesPiston_old(float* mySolute, float* leached, float* CoeffPiston)
// 2008.10 FT GA
// calcolo dei flussi di nitrati gravitazionali (a 'pistone')

    int L;
    int minFluxPiston = 5;

    ReDim u_media(nrLayers) As Double
    ReDim F_Solute(nrLayers) As Double
    ReDim Solute_Macro(nrLayers) As Double
    ReDim CoeffPiston(nrLayers)

    //'initialize
    for (L = 0;L<nrLayers; L++)
    {
        u_media[L] = (U[L] + umid[L].BeforeInfiltration) / 2;
        //'u_media(L) = U(L)
        if (u_media[L] > suolo[L].CC)
            Solute_Macro[L] = (mySolute[L] / u_media[L]) * (u_media[L] - suolo[L].CC);
        else
            Solute_Macro[L] = 0;
    }

    f_Solute[0] = 0;

    for (L = 1; L< nrLayers; L++)
    {
        F_Solute[L] = 0;
        if (u_media[L] <= suolo[L].CC || Flux(L) <= 0)
            CoeffPiston[L] = 0;
        else
            CoeffPiston[L] = minValue(1, Flux[L] / (u_media[L] - suolo[L].CC)) * minValue(1, Flux[L] / minFluxPiston);

        //'azoto in entrata da nodo L-1
        Solute_Macro[L] += F_Solute[L - 1];
        mySolute[L] += F_Solute[L - 1];

        //'calcolo flussi convettivi
        if (CoeffPiston[L] > 0)
        {
            f_Solute[L] = Solute_Macro[L] * CoeffPiston[L];

            //'azoto in uscita da nodo L
            Solute_Macro[L] -= F_Solute[L];
            mySolute[L] -= F_Solute[L];
        }
    }

    //leaching
    *leached += F_Solute[nrLayers-1];

}


void soluteFluxes(float* mySolute(),bool flagRisalita, float pistonDepth,float* )

    //2008.10 GA eliminata parte dispersiva perché il meccanismo pseudo-numerico è già dispersivo di suo
    //2008.09 GA inserita componente dispersiva
    //2008.03 GA FT inserita componente diffusiva
    //2007.04 FT GA sistemato algoritmo pseudo-numerico a iterazione
    //04.03.02.FZ
    //-------------- NOTE -----------------------------------------------------
    //calcolo dei flussi di soluti con diluizione iterativa

    int L;                          //[-] contatore
    float* flux_Solute();           //[g m-2] flussi soluto
    int i;                          //[-] contatore
    int iterations;       //[-] numero di iterazioni per la diluizione
    double* f_Solute;
    double H2O_step_flux;
    double H2O_step_flux_L_1;
    double U_vol;
    int firstLayer;
    float myFreeSolute;
    float coeffMobile;

        if (pistonDepth >= suolo[nrLayers].prof)
            return;
        else
        {
            for (L=1; L<; nrLayers;L++)

                If suolo(L).prof >= PistonDepth Then Exit For
            Next L
            FirstLayer = L
            L=0;
            while(suolo[L].prof >= pistonDepth)
            {
                L++;
            }
            firstLayer = L;
            L=0;
        }
        flux_solute = (float *) calloc(nrLayers, sizeof(float));
        double *u_temp = (double *) calloc(nrLayers, sizeof(double));
        double *f_solute = (double *) calloc(nrLayers, sizeof(double));

        for (L = 0; L<nrLayers; L++)
        {
            flux_Solute[L] = 0;
            u_temp[L] = umid[L].BeforeInfiltration;
        }
        // ???????????????????
        For L = nrLayers To 1 Step -1

        Next L

        f_Solute[0] = 0;
        // ??????????????????????????

        //iterazioni = min(max(24, 0.1 * max(Flux(0), Abs(Flux(nrLayers))) * max(Flux(0), Abs(Flux(nrLayers)))), 1000)

        iterations = 1;
        for (i = 0; i<iterations; i++)
        {
            For (L = firstLayer; L<nrLayers; L++)
            {
                f_Solute[L] = 0;

                H2O_step_flux = (Flux[L] / iterations);
                H2O_step_flux_L_1 = (Flux[L - 1] / iterations);

                // acqua in entrata/uscita da nodo L
                u_temp[L] += H2O_step_flux_L_1 - H2O_step_flux;

                // calcolo flussi soluto
                if (Flux[L] > 0)
                {
                    CoeffMobile = 1;
                    myFreeSolute = mySolute[L] * CoeffMobile;
                    f_Solute[L] = minValue(mySolute[L], myFreeSolute / umid[L].BeforeInfiltration * H2O_step_flux);
                }
                else if (flagRisalita && (Flux[L] < 0) && (L < nrLayers))
                {
                    //myFreeSolute = mySolute[L + 1] * CoeffMobile;
                    myFreeSolute = mySolute[L + 1];
                    f_Solute[L] = min(mySolute[L + 1], myFreeSolute / umid[L + 1].BeforeInfiltration * H2O_step_flux)
                }

                //azoto in entrata/uscita da nodo L-1
                mySolute[L] += f_Solute[L - 1] - f_Solute[L];

                //flussi convettivi totali
                flux_Solute[L] += f_Solute[L];

            }
        }

        // leaching
        // FT GA 2007.12
        *leached += Flux_Solute[nrLayers-1];

}




// function develpoed by V. Marletto for watertable
void leachingWaterTable(float* mySolute, float* leached)
{
    int L;
    double mySolute_leach_edge;

    // dilavamento
    if ((waterTable != NODATA) && (waterTable > 0) && (flagWaterTable == 1) && (flagWaterTableCase == 1))
    {
        for (L = 0; L< Layers; L++)
        {
            if (suolo[L].prof > waterTable)
            {
                leached += mySolute[L]
                mySolute[L] = 0;
            }
            else if (suolo[L].prof >= waterTable - MAX_FRANGIA_CAPILLARE)
            {
                mySolute_leach_edge = (mySolute[L] / MAX_FRANGIA_CAPILLARE) * (MAX_FRANGIA_CAPILLARE - (waterTable - suolo[L].prof))
                mySolute[L] += - mySolute_leach_edge;
                leached += mySolute_leach_edge;
            }
        }
    }

}

void NH4_Balance()
{
    float profileNH4PreviousDay;

    profileNH4PreviousDay = profileNH4;
    // ProfiloNH4 = ProfileSum(N_NH4())
    profileNH4 = 0;
    for (int i=0;i<nrLayers;i++)
    {
        profileNH4 += N_NH4[i];
    }

    balanceFinalNH4 = profileNH4 - profileNH4PreviousDay - N_NH4_fertGG + N_imm_l_NH4GG;
    balanceFinalNH4 += - N_min_humusGG - N_min_litterGG;
    balanceFinalNH4 += N_NH4_volGG - N_Urea_HydrGG + N_nitrifGG;
    balanceFinalNH4 += N_NH4_uptakeGG;
    balanceFinalNH4 += N_NH4_runoff0GG + N_NH4_runoffGG + Flux_NH4GG - PrecN_NH4GG;

    //If BilFinaleNH4 > 0.01 Then Stop
    return;
}

void NO3_Balance()
{
    // 02.11.26.MVS translated by Antonio Volta 2022.07.29

    float profileNO3PreviousDay;

    profileNO3PreviousDay = profileNO3;
    //profileNO3 = ProfileSum(N_NO3());
    profileNO3 = 0;
    for (int i=0;i<nrLayers;i++)
    {
        profileNO3 += N_NO3[i];
    }
    balanceFinalNO3 = profileNO3 - profileNO3PreviousDay - N_NO3_fertGG + N_imm_l_NO3GG;
    balanceFinalNO3 += N_denitrGG - N_nitrifGG + N_NO3_uptakeGG;
    balanceFinalNO3 += N_NO3_runoff0GG + N_NO3_runoffGG - PrecN_NO3GG + Flux_NO3GG;
    return;
}

void N_initializeCrop(bool noReset)
{
    N_cropToHarvest = 0;
    N_cropToResidues = 0;

    if (!noReset)
        N_roots = 0;
    // da leggere da database
    N_uptakable = tbColture("Nasportabile") / 10;   //      da [kg ha-1] a [g m-2]
    N_uptakeDeficit = 0;
    N_uptakeMax = 0;
    N_potentialDemandCumulated = 0;
    ReDim N_deficit_daily(Nitrogen.N_deficit_max_days) // operazione da capire come gestire

    //Select Case TipoColtura
        if (TipoColtura == "arborea" || TipoColtura == "arborea_inerbita" || TipoColtura == "fruit_tree" || TipoColtura == "fruit_tree_with_grass")
        {
            // 2001 Rufat Dejong Fig. 4 e Tagliavini
            N_ratioHarvested = 0.4;      // fruits, pruning wood
            N_ratioResidues = 0.5;       // leaves
            N_ratioRoots = 0.1;           // roots, trunk, branches


        }
        else if (TipoColtura == "erbacea_poliennale" || TipoColtura == "herbaceous_perennial" || TipoColtura == "prativa" || TipoColtura == "grass" || TipoColtura == "incolto" || TipoColtura ==  "fallow" || TipoColtura == "prativa_primoanno" || TipoColtura == "grass_firstyear")
        {
            N_ratioHarvested = 0.9;
            N_ratioResidues = 0;
            N_ratioRoots = 0.1;
        }
        else
        {
            // colture annuali
            N_ratioHarvested = 0.9;
            N_ratioResidues = 0;
            N_ratioRoots = 0.1;
        }

    //in prima approssimazione calcolato da N massimo asportabile per ciclo
    //(parte asportabile e non asportabile) e LAIMAX
    //2013.10 GA
    //scambio mail con Ass.Agr.:
    //
    maxRate_LAI_Ndemand = (N_uptakable - N_roots) / LAIMAX ;

}


void N_harvest() // public function
{
        // 2013.06 GA translated in C++ by AV 2022.06
        // annual crops:roots are incorporated in litter at harvest
        // meadow and perennial herbaceous crops: half of N from roots is incorporated in litter
        // tree crops: half of N from roots is incorporated in litter
        // N of leaves is incorporated in litter through the upeer layer with a smoothly rate during the leaf fall

    int L;
    float N_toLitter;
    // !!! verificare USR PSR
    if (PSR == 0 && USR == 0)
        return;

    for (L = PSR; L <= USR; L++) // verificare i cicli for per cambio indici
    {
        //Select Case TipoColtura
            // annual crop
            if (TipoColtura == "erbacea" || TipoColtura == "herbaceous" || TipoColtura == "orticola", TipoColtura == "horticultural")
                N_toLitter = Radici.DensStrato(L) * N_roots; // !! prendere il dato da dove?

            // multiannual crop
            else if (TipoColtura == "erbacea_poliennale"|| TipoColtura == "herbaceous_perennial"|| TipoColtura ==  "prativa"|| TipoColtura ==  "grass"|| TipoColtura ==  "incolto"|| TipoColtura ==  "fallow"|| TipoColtura ==  "prativa_primoanno"|| TipoColtura ==  "grass_firstyear")
                N_toLitter = Radici.DensStrato(L) * N_roots / 2;

            // tree crops
            else if (TipoColtura ==  "arborea"|| TipoColtura == "fruit_tree"|| TipoColtura == "arborea_inerbita"|| TipoColtura == "fruit_tree_with_grass")
                N_toLitter = Radici.DensStrato(L) * N_roots / 2;



        N_litter(L) += N_toLitter
        C_litter(L) += CN_RATIO_NOTHARVESTED * N_toLitter
    }

    if (TipoColtura == "erbacea" || TipoColtura == "herbaceous" || TipoColtura == "orticola", TipoColtura == "horticultural")
    {
        // annual crops
        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots = 0;
    }
    else if (TipoColtura == "erbacea_poliennale"|| TipoColtura == "herbaceous_perennial"|| TipoColtura ==  "prativa"|| TipoColtura ==  "grass"|| TipoColtura ==  "incolto"|| TipoColtura ==  "fallow"|| TipoColtura ==  "prativa_primoanno"|| TipoColtura ==  "grass_firstyear")
    {
        //pluriennali
        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots *= 0.5;
    }
    else if (TipoColtura ==  "arborea"|| TipoColtura == "fruit_tree"|| TipoColtura == "arborea_inerbita"|| TipoColtura == "fruit_tree_with_grass")
    {
        //tree crops

            N_cropToHarvest = 0;
            N_Roots *= 0.5;
    }

    N_potentialDemandCumulated = 0;

}



void updateNCrop() // this function must be private
{
    if (coltura == Crops.CROP_ALFALFA || coltura == Crops.CROP_ALFALFA_FIRSTYEAR || coltura == Crops.CROP_SOYBEAN)
    {
            // the demand is satisfied by Nitrogen fixation
            // it prevails soil mineral uptake, if available
            N_cropToHarvest += N_dailyDemand * N_ratioHarvested;
            N_cropToResidues += N_dailyDemand * N_ratioResidues;
            N_Roots += N_dailyDemand * N_ratioRoots;
    }
    else
    {
            N_cropToHarvest += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioHarvested;
            N_cropToResidues += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioResidues;
            N_roots += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioRoots;
    }
    // pare che sia commentato chiedere a Gabri
    'N_UptakeDeficit = max(N_PotentialDemandCumulated - N_Crop, 0)
}

void N_plough() // this function must be public
{
    int L;
    float depthRatio;
    float N_toLitter; // sembra da togliere chiedere a Gabri
    float N_totLitter;
    float N_totHumus;
    float C_totLitter;
    float C_totHumus;
    float N_totNO3;
    float N_totNH4;
    int myLastLayer;
    float tmp;

        N_totLitter = N_cropToHarvest + N_cropToResidues + N_roots;
        C_totLitter = N_totLitter * CN_RATIO_NOTHARVESTED;
        N_totHumus = 0;
        C_totLitter = 0;
        C_totHumus = 0;
        N_totNO3 = 0;
        N_totNH4 = 0;

        L = 0;
        do{

            N_totLitter += N_litter[L];
            C_totLitter += C_litter[L];
            N_totHumus += N_humus[L];
            C_totHumus += C_humus[L];
            N_totNO3 += N_NO3[L];
            N_totNH4 += N_NH4[L];
            L++;
        } while (suolo(L).spess + suolo(L).prof <= N_Plough_Depth)

        if (L == 0)
            return;
        else
            myLastLayer = L - 1;

        tmp = 0;
        for (L=0;L<myLastLayer;L++) // verificare i cicli for per cambio indici
        {
            depthRatio = suolo(L).spess / (suolo(myLastLayer).spess + suolo(myLastLayer).prof)
            tmp += depthRatio;

            N_litter(L) = N_totLitter * depthRatio;
            C_litter(L) = C_totLitter * depthRatio;
            N_humus(L) = N_totHumus * depthRatio;
            C_humus(L) = C_totHumus * depthRatio;
            N_NO3(L) = N_totNO3 * depthRatio;
            N_NH4(L) = N_totNH4 * depthRatio;
        }
        Partitioning

        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots = 0;
}

void NFromCropSenescence(ByVal myDays As Single, ByVal coeffB As Single) // this function must be public
{
    //created in 2013.06 by GA, translated by AV 2022.06
    //new function for describing the release of Nitrogen from pluriannual crop residues
    // e.g. leaf fall
    //myDays  days past since beginning of senescence
    //coeffB  b coefficient in exponential senescence LAI curve


    float ratioSenescence;      //ratio of drop leaves

    ratioSenescence = exp(coeffB * myDays) * (1 - exp(-coeffB)) / (exp(coeffB * LENGTH_SENESCENCE) - 1);
    N_litter[0] = N_litter[0] + N_CropToResidues * ratioSenescence;

}
*/









