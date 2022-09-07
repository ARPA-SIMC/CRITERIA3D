#include <stdio.h>
#include <math.h>

#include "carbonNitrogen.h"
//#include "soil.h"








float Crit3DCarbonNitrogenWholeProfile::convertToGramsPerM3(int layerIndex, float myQuantity,std::vector<soil::Crit3DLayer> &soilLayers)
{
    // convert from g m-2 to g m-3 (= mg dm-3)

    return myQuantity / (soilLayers[layerIndex].thickness / 100);
}


float Crit3DCarbonNitrogenWholeProfile::convertToGramsPerLiter(int layerIndex,float myQuantity,std::vector<soil::Crit3DLayer> &soilLayers)
{
    //' convert from g m-2 to g l-1

    //' to g dm-3 and to g l-1
    return (convertToGramsPerM3(layerIndex, myQuantity, soilLayers) / 1000);// /WaterBalance.ConvertWCToVolumetric(suolo[layerIndex], U[layerIndex]);
}


float Crit3DCarbonNitrogenWholeProfile::convertToGramsPerKg(int layerIndex,float myQuantity, std::vector<soil::Crit3DLayer> &soilLayers,soil::Crit3DSoil* soil)
{
    //' convert from g m-2 to g kg-1

    //' to g dm-3 and then to g kg-1
    return (convertToGramsPerM3(layerIndex, myQuantity,soilLayers) / 1000) / soil->horizon[layerIndex].bulkDensity;
}
/*
 *
// da valutare
void N_InitializeLayers()
{
    Erase C_humus
    Erase C_litter
    Erase C_litter_humus
    Erase C_litter_litter
    Erase C_min_humus
    Erase C_min_litter
    Erase C_denitr_litter
    Erase C_denitr_humus
    Erase Wcorr_denitr
    Erase N_humus
    Erase N_denitr
    Erase N_litter
    Erase N_litter_humus
    Erase N_min_humus
    Erase N_min_litter
    Erase N_NH4
    Erase N_NH4_Adsorbed
    Erase N_NH4_Sol
    Erase N_nitrif
    Erase N_NO3
    Erase N_urea
    Erase N_Urea_Hydr
    Erase N_vol
    Erase N_NH4_uptake
    Erase N_NO3_uptake
    Erase CNratio_litter
    Erase SO
    Erase TCorr
    Erase WCorr

    ReDim C_humus(nrLayers)
    ReDim C_litter(nrLayers)
    ReDim C_litter_humus(nrLayers)
    ReDim C_litter_litter(nrLayers)
    ReDim C_min_humus(nrLayers)
    ReDim C_min_litter(nrLayers)
    ReDim C_denitr_litter(nrLayers)
    ReDim C_denitr_humus(nrLayers)
    ReDim Wcorr_denitr(nrLayers)
    ReDim N_humus(nrLayers)
    ReDim N_denitr(nrLayers)
    ReDim N_litter(nrLayers)
    ReDim N_litter_humus(nrLayers)
    ReDim N_min_humus(nrLayers)
    ReDim N_min_litter(nrLayers)
    ReDim N_NH4(nrLayers)
    ReDim N_NH4_Sol(nrLayers)
    ReDim N_NH4_Adsorbed(nrLayers)
    ReDim N_nitrif(nrLayers)
    ReDim N_NO3(nrLayers)
    ReDim N_urea(nrLayers)
    ReDim N_Urea_Hydr(nrLayers)
    ReDim N_vol(nrLayers)
    ReDim N_NH4_uptake(nrLayers)
    ReDim N_NO3_uptake(nrLayers)
    ReDim CNratio_litter(nrLayers)
    ReDim SO(nrLayers)
    ReDim TCorr(nrLayers)
    ReDim WCorr(nrLayers)

}


void humusIni()
{
    //'2008.09 GA
    //'GA 2007.12 perché C calcolato da (CN/CN+1) e non moltiplicando per 0.58 come solito?
    //'computes initial humus carbon and nitrogen for a layer L
    //'version 1.0, 2004.08.09.VM

    int L;
    // MVA                '[kg dm-3 = 10^6 g m-3]

    for (L = 1; L<nrLayers;L++)
    {
        C_humus[L] = suolo[L].MVA * 1000000 * (suolo[L].SostanzaO / 100) * 0.58 * suolo[L].spess / 100;
        N_humus[L] = C_humus[L] / CNratio_humus;
    }
}


void updateTotalOfPartitioned(float* mySoluteSum, float* mySoluteAds,float* mySoluteSol)
{
    int L;
    for (L = 0; L<nrLayers; L++)
    {
        mySoluteSum[L] = mySoluteAds[L] + mySoluteSol[L];
    }
}
*/
void Crit3DCarbonNitrogenWholeProfile::partitioning(float* theta,std::vector<soil::Crit3DLayer> &soilLayers,soil::Crit3DSoil* soil)
{
    //2013.06
    // partitioning of N (only NH4) between adsorbed and in solution

    // SE(M,I)=DSE(M,I)/(DLZ*(KD(M,I)*RHO(I)+W(I)))
    // ASE(M,I)=KD(M,I)*SE(M,I)
    // DSE / DLZ (mg dm-3)

    float N_NH4_g_dm3;               //[g dm-3] total ammonium
    float N_NH4_sol_g_l;             //[g l-1] ammonium in solution
    float N_NH4_ads_g_kg;            //[g kg-1] adsorbed ammonium
    float N_NH4_ads_g_m3;            //[g m-3] adsorbed ammonium
    float myTheta;
    int L;

    N_NH4_adsorbedGG = 0;

    for (L = 0; L<numberOfLayers; L++)
    {
        myTheta = theta[L];
        N_NH4_g_dm3 = convertToGramsPerM3(L,arrayCarbonNitrogen[L].N_NH4,soilLayers) / 1000;
        N_NH4_sol_g_l = N_NH4_g_dm3 / (Kd_NH4 * soil->horizon[L].bulkDensity + myTheta);

        arrayCarbonNitrogen[L].N_NH4_Sol = N_NH4_sol_g_l * (soilLayers[L].thickness / 100) * myTheta * 1000;

        N_NH4_ads_g_kg = Kd_NH4 * N_NH4_sol_g_l;
        N_NH4_ads_g_m3 = N_NH4_ads_g_kg * soil->horizon[L].bulkDensity * 1000;
        arrayCarbonNitrogen[L].N_NH4_Adsorbed = N_NH4_ads_g_m3 * soilLayers[L].thickness / 100;
        N_NH4_adsorbedGG += arrayCarbonNitrogen[L].N_NH4_Adsorbed;
    }
}

/*
void litterIni()
{
    //2008.10 GA inizializzazione indipendente da humus ma da input utente
    //computes initial litter carbon and nitrogen for a layer L
    //version 1.0, 2004.08.16.VM

    int L;
    float LayerRatio;
    float myDepth;

    myDepth = maxValue(litterIniProf, 6);

    for (L=0; L<nrLayers ;L++)
    {
        if (suolo[L].prof <= myDepth)
        {
            LayerRatio = (suolo[L].spess / myDepth);
            C_litter[L] = LitterIniC * LayerRatio / 10;          //from kg ha-1 to g m-2
            N_litter[L] = LitterIniN * LayerRatio / 10;          //from kg ha-1 to g m-2
        }
    }
}

void chemicalTransformations()
{
    // 2013.06 GA
    // revision from LEACHM
    // concentrations in g m-3 (= mg dm-3)

    float myN_NO3;                   //[g m-3] nitric nitrogen concentration
    float myN_NH4;                   //[g m-3] ammonium concentration
    float myN_NH4_sol;               //[g m-3] ammonium concentration in solution
    float NH4_ratio;                 //[] ratio of NH4 on total N
    float NO3_ratio;                 //[] ratio of NO3 on total N
    float myTotalC;                  //[g m-3] total carbon
    float myLitterC;                 //[g m-3] carbon concentration in litter
    float myHumusC;                  //[g m-3] carbon concentration in humus
    float myLitterN;                 //[g m-3] nitrogen concentration in litter
    float myHumusN;                  //[g m-3] nitrogen concentration in humus

    float CCDEN;                     //[g m-3] Amount of carbon equivalent to N removed (as CO2) from denitrification
    float CLIMM;                     //[g m-3] maximum N immobilization
    float CLIMX;                     //[g m-3] ratio of the maximum to the desired rate. Adjusted by 0.08 to extend immobilization and slow the rate
    float CCLH;                      //[g m-3] carbon from litter to humus
    float CLH;                       //[g m-3] nitrogen from litter to humus
    float CLI;                       //[g m-3] nitrogen internally recycled in litter
    float CCLI;                      //[g m-3] carbon internally recycled in litter
    float CCLCO2;                    //[g m-3] carbon from litter to CO2
    float CCHCO2;                    //[g m-3] carbon from humus to CO2
    float CCLDN;                     //[g m-3] carbonio rimosso dal litter per denitrificazione
    float CCHDN;                     //[g m-3] carbonio rimosso dall'humus per denitrificazione
    float CNHNO;                     //[g m-3] source di nitrato (da nitrificazione)
    float CNON;                      //[g m-3] sink di nitrato (da denitrificazione)
    float CURNH;                     //[g m-3] urea hydrolysis
    float CHNH;                      //[g m-3] N humus mineralization
    float CLNH;                      //[g m-3] N litter mineralization
    float CNHGAS;                    //[g m-3] NH4 volatilization
    float CNHL;                      //[g m-3] N NH4 litter immobilization
    float CNOL;                      //[g m-3] N NO3 litter immobilization
    float USENH4;                    //[g m-3] N NH4 uptake
    float USENO3;                    //[g m-3] N NO3 uptake

    float litterCSink;               //[g m-3] C litter sink
    float litterCSource;             //[g m-3] C litter source
    float litterCRecycle;            //[g m-3] C litter recycle
    float litterCNetSink;            //[g m-3] C litter net sink/source
    float humusCSink;                //[g m-3] C humus sink
    float humusCSource;              //[g m-3] C humus source
    float humusCNetSink;             //[g m-3] C net sink/source
    float litterNSink;               //[g m-3] N litter sink
    float litterNSource;             //[g m-3] N litter source
    float litterNRecycle;            //[g m-3] N litter recycle
    float litterNNetSink;            //[g m-3] N litter net sink/source
    float humusNSink;                //[g m-3] N humus sink
    float humusNSource;              //[g m-3] N humus source
    float humusNNetSink;             //[g m-3] N net sink/source
    float N_NH4_sink;
    float N_NH4_source;
    float N_NH4_netSink;
    float N_NO3_sink;
    float N_NO3_source;
    float N_NO3_netSink;

    float totalCO2;                   //[g m-3] source of CO2
    float def;
    float total;
    float factor;

    static float adjustFactor = 0.08; // factor to extend immobilization and slow the rate

    int L;

    for (L=0; L<nrLayers; L++)
    {
        // correction functions for soil temperature and humidity
        // inserire parametri in Options.mdb
        computeTemperatureCorrectionFactor(L);
        computeWaterCorrectionFactor(L);

        // compute layer transformation rates
        computeLayerRates(L);

        // convert to concentration
        myLitterC = convertToGramsPerM3(L, C_litter[L]);
        myHumusC = convertToGramsPerM3(L, C_humus[L]);
        myLitterN = convertToGramsPerM3(L, N_litter[L]);
        myHumusN = convertToGramsPerM3(L, N_humus[L]);

        myTotalC = (myLitterC + myHumusC);
        myN_NO3 = convertToGramsPerM3(L, N_NO3[L]);
        myN_NH4 = convertToGramsPerM3(L, N_NH4[L]);
        if ((myN_NH4 + myN_NO3) > 0)
        {
            NH4_ratio = myN_NH4 / (myN_NH4 + myN_NO3);
            NO3_ratio = 1 - NH4_ratio;
        }
        else
        {
            NH4_ratio = 0;
            NO3_ratio = 0;
        }

    // CARBON TRANSFORMATIONS

        // i) Associated with denitrification.  Amount of carbon
        //    equivalent to N removed.  (No more than a tenth of C present
        //    C  can be removed). Not in Johnsson's paper.
        //    si assume che tutto il nitrato sia in soluzione (mg l-1)
        //    CO2 prodotta associata a denitrificazione
        //    cinetica primo ordine della decomposizione della s.o. (rateo proporzionale alla concentrazione)
        CCDEN = maxValue(0, myN_NO3 * (1 - exp(-actualRate_N_Denitrification))) * 72. / 56.;
        // No more than a tenth of C present C  can be removed
            CCDEN = minValue(CCDEN,0.1 * myTotalC);


        // ii) litter transformation
        // litter C to humus C
        CCLH = maxValue(0, myLitterC * (1 - exp(-actualRate_C_LitterToHumus)));
        // litter C internal recycle
        CCLI = maxValue(0, myLitterC * (1 - exp(-actualRate_C_LitterToBiomass)));
        // litter C to CO2
        CCLCO2 = maxValue(0, myLitterC * (1 - exp(-actualRate_C_LitterToCO2)));
        // nitrogen immobilization
        CLIMX = 1;
        if (actualRate_N_LitterImm > 0)
        {
            CLIMM = actualRate_N_LitterImm * (CCLI + CCLH + CCLCO2);
            if (CLIMM > 0)
                CLIMX = minValue(1, (adjustFactor * (myN_NO3 + myN_NH4)) / CLIMM);
            // if immobilization limits mineralization then the effective rate of immobilization is reduced
            if (CLIMX < 1)
            {
                CCLH *= CLIMX;
                CCLI *= CLIMX;
                CCLCO2 *= CLIMX;
            }
            CNHL = CLIMM * CLIMX * NH4_ratio;
        }

        // Energy source for denitrification
        if (myTotalC > 0)
        {
            CCLDN = CCDEN * myLitterC / myTotalC;
        }
        else
        {
            CCLDN = 0;
        }
        litterCSink = CCLH + CCLCO2 + CCLDN;
        litterCSource = 0;
        litterCRecycle = CCLI;
        litterCNetSink = -litterCSink + litterCSource;

        // iii) humus transformations
        // humus to CO2
        CCHCO2 = maxValue(0, myHumusC * (1 - exp(-actualRate_C_HumusMin)));
        // energy source for denitrification
        if (myTotalC > 0)
            CCHDN = CCDEN * myHumusC / myTotalC;
        else
            CCHDN = 0;

        humusCSink = CCHDN + CCHCO2;
        humusCSource = CCLH;
        humusCNetSink = humusCSource - humusCSink;

        // iv) CO2
        totalCO2 = CCLCO2 + CCHCO2 + CCLDN + CCHDN;


    // NITROGEN TRANSFORMATIONS
        // i) urea hydrolysis
        CURNH = convertToGramsPerM3(L, N_urea[L]) * (1 - exp(-actualRate_Urea_Hydr));

        // ii) ammonium volatilization (only top 5 cm of soil) (in LEACHM 10 cm but layer thickness is 10 cm)
        if (suolo[L].prof + suolo[L].spess) < 5
        {
            myN_NH4_sol = convertToGramsPerM3(L, N_NH4_Sol[L]);
            CNHGAS = minValue(0.5 * myN_NH4_sol, myN_NH4_sol * (1 - exp(-rate_N_NH4_Volatilization)));
        }
        else
            CNHGAS = 0;

        // iii) nitrification
        CNHNO = (maxValue(0, myN_NH4 - myN_NO3 / limRatio_nitr)) * (1 - exp(-actualRate_N_Nitrification));

        // iv) mineralization
        CHNH = maxVale(myHumusN, 0) * actualRate_C_HumusMin;
        CLNH = actualRate_N_LitterMin * (CCLH + CCLCO2 + CCLI);
        if (actualRate_N_LitterImm > 0)
            CNHL = CLIMM * CLIMX * NH4_ratio;

        // NH4 sink/source
        USENH4 = convertToGramsPerM3(L, N_NH4_uptake[L]);
        N_NH4_sink = CNHNO + CNHL;
        N_NH4_source = CHNH + CLNH + CURNH;
        N_NH4_netSink = -N_NH4_sink + N_NH4_source - CNHGAS - USENH4;
        // adjustment
        def = myN_NH4 + N_NH4_netSink;
        total = CNHNO + CNHGAS + USENH4;
        if (def < 0 && total > 0)
        {
            factor = maxValue(0, 1. + def / total);
            CNHNO *= factor;
            CNHGAS *= factor;
            USENH4 *= factor;
            N_NH4_sink = CNHNO + CNHL;
            N_NH4_netSink = -N_NH4_sink + N_NH4_source - CNHGAS - USENH4;
        }

        // NO3 sink/source
        USENO3 = convertToGramsPerM3(L, N_NO3_uptake[L]);
        if (actualRate_N_LitterImm > 0)
            CNOL = CLIMM * CLIMX * NO3_ratio;
        CNON = CCDEN * 56. / 72.;
        N_NO3_source = CNHNO;
        N_NO3_sink = CNOL;
        N_NO3_netSink = -N_NO3_sink + N_NO3_source - USENO3 - CNON;
        // adjustment
        def = myN_NO3 + N_NO3_netSink;
        total = USENO3 + CNON;
        if (def < 0 && Total > 0)
        {
            factor = maxValue(0., 1. + def / total);
            CNON *= factor;
            USENO3 *= factor;
            N_NO3_sink = CNON + CNOL;
            N_NO3_netSink = -N_NO3_sink + N_NO3_source - USENO3;
        }

        // litter N sink/source
        CLH = CCLH / CNratio_humus;
        CLI = CCLI / CNratio_humus;
        litterNSink = CLH + CLNH;
        litterNSource = CNHL + CNOL;
        litterNRecycle = CLI;
        litterNNetSink = -litterNSink + litterNSource;

        // humus sink/source
        humusNSink = CHNH;
        humusNSource = CLH;
        humusNNetSink = -humusNSink + humusNSource;

    // ---------------------------------------------------------------------------------

        // convert back to g m-2
        N_NO3_uptake[L] = USENO3 * (suolo[L].spess / 100.);
        N_NH4_uptake[L] = USENH4 * (suolo[L].spess / 100.);
        N_min_litter[L] = CLNH * (suolo[L].spess / 100.);
        N_imm_l_NH4[L] = CNHL * (suolo[L].spess / 100.);
        N_imm_l_NO3[L] = CNOL * (suolo[L].spess / 100.);
        N_min_humus[L] = CHNH * (suolo[L].spess / 100.);
        N_litter_humus[L] = CLH * (suolo[L].spess / 100.);
        N_vol[L] = CNHGAS * (suolo[L].spess / 100.);
        N_denitr[L] = CNON * (suolo[L].spess / 100.);
        N_nitrif[L] = CNHNO * (suolo[L].spess / 100.);
        N_Urea_Hydr[L] = CURNH * (suolo[L].spess / 100.);

        C_litter_humus[L] = CCLH * (suolo[L].spess / 100.);
        C_litter_litter[L] = CCLI * (suolo[L].spess / 100.);
        C_min_litter[L] = CCLCO2 * (suolo[L].spess / 100.);
        C_min_humus[L] = CCHCO2 * (suolo[L].spess / 100.);
        C_denitr_humus[L] = CCHDN * (suolo[L].spess / 100.);
        C_denitr_litter[L] = CCLDN * (suolo[L].spess / 100.);

    // -----------------------------------------------------------------------------------
    // mass balancing

        N_NH4[L] += N_NH4_netSink * (suolo[L].spess / 100.);
        N_NO3[L] += N_NO3_netSink * (suolo[L].spess / 100.);
        C_litter[L] += litterCNetSink * (suolo[L].spess / 100.);
        C_humus[L] += humusCNetSink * (suolo[L].spess / 100.);
        N_litter[L] += litterNNetSink * (suolo[L].spess / 100.);
        N_humus[L] += humusNNetSink * (suolo[L].spess / 100.);
        N_urea[L] -= N_Urea_Hydr[L];

        //If N_NH4(L) < 0 Then Stop
        //If N_NO3(L) < 0 Then Stop
        //If C_litter(L) < 0 Then Stop
        //If C_humus(L) < 0 Then Stop
        //If N_litter(L) < 0 Then Stop
        //If N_humus(L) < 0 Then Stop

        N_NH4[L] = maxValue(0, N_NH4[L]);
        N_NO3[L] = maxValue(0, N_NO3[L]);
        C_litter[L] = maxValue(0, C_litter[L]);
        C_humus[L] = maxValue(0, C_humus[L]);
        N_litter[L] = maxValue(0, N_litter[L]);
        N_humus[L] = maxValue(0, N_humus[L]);
        N_urea[L] = maxValue(0, N_urea[L]);

        // profile totals
        N_humusGG += N_humus[L];
        N_litterGG += N_litter[L];
        N_litter_humusGG += N_litter_humus[L];
        N_min_humusGG += N_min_humus[L];
        N_min_litterGG += N_min_litter[L];
        N_imm_l_NH4GG += N_imm_l_NH4[L];
        N_imm_l_NO3GG += N_imm_l_NO3[L];
        C_humusGG += C_humus[L];
        C_min_humusGG += C_min_humus[L];
        C_litter_humusGG += C_litter_humus[L];
        C_litter_litterGG += C_litter_litter[L];
        C_min_litterGG += C_min_litter[L];
        C_litterGG += C_litter[L];
        N_NO3_uptakeGG += N_NO3_uptake[L];
        N_NH4_uptakeGG += N_NH4_uptake[L];
        N_NH4_volGG += N_vol[L];
        N_nitrifGG += N_nitrif[L];
        N_Urea_HydrGG += N_Urea_Hydr[L];
        N_denitrGG += N_denitr[L];
    }

    updateNCrop();

}


// da valutare
void N_Initialize()
'****************************************************************
'Scopo: lettura da tbAzoto di coefficienti relativi al ciclo di N
'****************************************************************
'04.08.04.VM eliminazione dei vettori di costanti
'00.07.19.VM.MVS nuova sub chiamata da SUB_INI_Profilo per le variabili dell'azoto
'****************************************************************

    tbAzoto.MoveFirst

    Rate_C_HumusMin = tbAzoto("miner_rate_humus")
    Rate_C_LitterMin = tbAzoto("miner_rate_litter")
    Rate_N_NH4_Volatilization = tbAzoto("volatNH4_rate")
    Rate_N_Denitrification = tbAzoto("denitr_rate")
    Max_afp_denitr = tbAzoto("denitr_max_AFPF")
    Csat_denitr = tbAzoto("denitr_Csat")
    Rate_Urea_Hydr = tbAzoto("hydr_urea")
    Rate_N_Nitrification = tbAzoto("nitr_rate")
    limRatio_nitr = tbAzoto("nitr_limRatio")
    FE = tbAzoto("Fe")
    FH = tbAzoto("Fh")
    CNratio_humus = tbAzoto("CNh")
    Kd_NH4 = tbAzoto("Kd_NH4")
    Q10 = tbAzoto("Q10")
    Tbase = tbAzoto("TBase")

    CNratio_biomass = CNratio_humus
    FlagSO = 1

    '2008.02 GA da inserire in database
    Nitrogen.N_deficit_max_days = 3
    ReDim Nitrogen.N_deficit_daily(0)
    N_CropToHarvest = 0
    N_CropToResidues = 0
    N_Roots = 0

End Sub


void N_Fertilization()
{
    //07.12.17 GA cambiata dichiarazione da Integer a Single per concentrazioni (per valori frazionari)
    //02.11.27.MVS aggiunta concime organico
    //02.11.26.MVS aggiunta N_NO3_fertGG e N_NH4_fertGG per il bilancio
    //02.03.10.GD
    //01.01.10.GD
    //00.06.16.GD.MVS Questa sub è attivata nel momento della concimazione.
    //-------------- NOTE -----------------------------------------------------
    //Legge i dati dalla story e alimenta il suolo con le forme azotate appropiate.
    //-------------- Input Variables ------------------------------------------
    //N_NH4()                    [g m-2] azoto sotto forma ammoniacale
    //N_NO3()                    [g m-2] azoto sotto forma nitrica
    //nrLayers                    [-] numero di strati del profilo simulato
    //ProfConcimeN               [cm] profondità della concimazione azotata
    //QuantitàConcimeTot               [kg ha -1] quantità totale di concime di azoto
    //Suolo().Spess              [cm] spessore dello strato
    //-------------- Output Variables -----------------------------------------
    //N_NH4()                    [g m-2] azoto sotto forma ammoniacale
    //N_NO3()                    [g m-2] azoto sotto forma nitrica
    //-------------- Internal Variables ---------------------------------------
    //
    //-------------- Input Parameters -----------------------------------------
    //
    //-------------- Internal Parameters --------------------------------------
    float* quantityN = (float *) calloc(nrLayers, sizeof(float));
    int L;//            'contatore
    int LL;//        'contatore
    float quantityNcm; // As Single   'quantità per cm
    float percNO3; // As Single       'percentuale di nitrato nel concime
    float percNH4; // As Single      'percentuale di ione ammonio nel concime
    float percNorg; // As Single       'percentuale di sostanza organica nel concime

    int ID_Fertilizer;
    string ID_TipoConcime // As String (valutare come trattarlo)
    float QuantityNtot;
    float titoloN;
    float C_N_organic;
    string str; // valutare As String
    float* N_Norg_fert = (float *) calloc(nrLayers, sizeof(float)); // As Single
    float* N_NO3_fert = (float *) calloc(nrLayers, sizeof(float)); // As Single
    float* N_NH4_fert = (float *) calloc(nrLayers, sizeof(float)); // As Single

    // ReDim N_Norg_fert(nrLayers)
    // ReDim N_NO3_fert(nrLayers)
    // ReDim N_NH4_fert(nrLayers)
    // ReDim QuantitàN(nrLayers)

    str = "ID_FERTILIZER = " & TipoConcime & ""

    // 'calcolo quantità N nella concimazione N/P/K
    tbConcimi.FindFirst str
    If Not tbConcimi.NoMatch Then
        ID_Fertilizer = tbConcimi("ID_FERTILIZER")
        ID_TipoConcime = tbConcimi("ID_tipo")
        TitoloN = tbConcimi("TitoloN")
        PercNO3 = tbConcimi("N-NO3")
        PercNH4 = tbConcimi("N-NH4")
        PercNorg = tbConcimi("N-Norg")
        C_N_organico = tbConcimi("C_N_organico")
    Else
        StampaErrore ("type of fertilizer missing")
    End If


    if (ID_TipoConcime == "organico")
    {
        //letame e liquame è espresso in tonnellate
        QuantitàConcimeTot = QuantitàConcimeTot *= 1000;  //da tonnellate a kg
    }

    QuantitàNtot = QuantitàConcimeTot / 100 * TitoloN;

    //perdita immediata per volatilizzazione dell'urea
    if (ID_Fertilizer == FERTILIZER_UREA)
        QuantityNtot *= 0.8;

    //2007.04 GA se profondità = 0 (tutto nei primi 10 cm)
    if (ProfConcime == 0)
        ProfConcime = 10;

    // divido la quantità per cm
    quantityNcm = QuantityNtot / ProfConcime;
    for (L = 0; L < nrLayers; L++)
    {
        if (quantityNtot == 0)
        {
            LL = L - 1;
            break;
        }
        quantityN[L] = quantityNcm * suolo[L].spess;
        if (quantityN[L] > quantityNtot)
            QuantityN[L] = QuantityNtot; //controllo per non superare la quantità di concime nell'ultimo strato concimato
        quantityNtot -= quantityN[L];
        quantityN[L] /= 10 // da kg ha-1 a g m-2
    }

    for (L=0; L<LL; L++)
    {
        //'2007.12 GA inserita concimazione ureica (prima N_urea era sempre 0!)
        //'perdita del 30% per volatilizzazione immediata
        if (ID_Fertilizer == FERTILIZER_UREA)
            N_urea[L] = quantityN[L];

        N_NO3_fert[L] = PercNO3 * QuantitàN[L] / 100;
        N_NH4_fert[L] = PercNH4 * QuantitàN[L] / 100;
        N_Norg_fert[L] = PercNorg * QuantitàN[L] / 100;

        //per il bilancio...
        N_NO3_fertGG += N_NO3_fert[L];
        N_NH4_fertGG += N_NH4_fert[L];

        N_NO3[L] += N_NO3_fert[L]; //aggiornamento N_NO3
        N_NH4[L] += N_NH4_fert[L]; //aggiornamento N_NH4
        N_litter[L] += N_Norg_fert[L]; //aggiornamento N_litter
        C_litter[L] += N_Norg_fert[L] * C_N_organic; //aggiornamento C_litter
    }

}

// da valutare
void N_InitializeVariables()
'2004.08.16.VM introduzione di FUN_CNhumus_INI e LitterIni
'2004.08.05.VM eliminato da tbLog la costante CNratio_humus
'2004.06.25.VM forzato la sostanza organica del suolo
'2002.03.15.GD correzione calcolo N_humus
'2000.11.20.MVS nuovo codice sulla lettura di tbIniProfilo e tblog
'1999.12.02.MVS cambiamento tbUscite in tbUscite_azoto...
'1999.05.20.GD
'1999.03.15.GD
'-------------- NOTE -----------------------------------------------------
'Questa routine carica i valori iniziali nelle variabili
'utilizzando due metodi alternativi:
'1) nel caso la simulazione sia stata interrotta legge
'   lo stato delle variabili dal database delle uscite
'2) nel caso la simulazione sia nuova pone a zero oppure a valori
'   iniziali le stesse variabili
'-------------- Input Variables ------------------------------------------
'nrLayers           [-] numero di strati
'DataIniziale      [-] data di inizio della simulazione
'-------------- Output Variables -----------------------------------------

'-------------- Internal Variables ---------------------------------------
Dim L As Integer   '[-] numero dello strato
Dim Nome$          '[-] nome del campo umidità
'-------------- Input Parameters -----------------------------------------
'
'-------------- Internal Parameters --------------------------------------
'
'-------------------------------------------------------------------------
    Dim dataformattata As String
    Dim i As Integer
    Dim variable As String
    Dim fldname As String

    dataformattata = "Data=#" & format(Month(Attuale), "00") & "/" & format(Day(Attuale), "00") & "/" & format(Year(Attuale), "0000") & "#"    ' - 1900, "00") & "#" '****cambio formato

    tbUscite_Azoto.FindFirst dataformattata

    If Envi = "GEO" Then
        If Not tbLog.EOF Then

            While Not tbLog.EOF

                    variable = tbLog("Varname")

                    If variable = "C_humus" Or _
                       variable = "C_litter" Or _
                       variable = "N_Humus" Or _
                       variable = "N_litter" Or _
                       variable = "N_NH4" Or _
                       variable = "N_NH4_Adsorbed" Or _
                       variable = "N_NO3" Or _
                       variable = "N_urea" Or _
                       variable = "CNratio_litter" Then

                        For L = 1 To nrLayers
                            fldname = "Str" & CStr(L)
                            If variable = "C_humus" Then C_humus(L) = CSng(tbLog(fldname))
                            If variable = "C_litter" Then C_litter(L) = CSng(tbLog(fldname))
                            If variable = "N_Humus" Then N_humus(L) = CSng(tbLog(fldname))
                            If variable = "N_litter" Then N_litter(L) = CSng(tbLog(fldname))
                            If variable = "N_NH4" Then N_NH4(L) = CSng(tbLog(fldname))
                            If variable = "N_NH4_Adsorbed" Then N_NH4_Adsorbed(L) = CSng(tbLog(fldname))
                            If variable = "N_NO3" Then N_NO3(L) = CSng(tbLog(fldname))
                            If variable = "N_urea" Then N_urea(L) = CSng(tbLog(fldname))
                            If variable = "CNratio_litter" Then CNratio_litter(L) = CSng(tbLog(fldname))
                        Next L
                    End If
                tbLog.MoveNext
            Wend
            tbLog.MoveFirst

        Else
            HumusIni
            LitterIni
            Partitioning
        End If

    Else
        HumusIni
        LitterIni
        Partitioning
    End If

    'azzeramento delle variabili non rilette

    ProfiloNO3 = ProfileSum(N_NO3())
    ProfiloNH4 = ProfileSum(N_NH4())

End Sub

// da valutare come replicare se fare riferimento ad un database
void ApriTabellaUsciteAzoto(tbname_azoto As String)


Dim L As Integer

    If Not TableExists(dbNitrogen, tbname_azoto) Then

        Dim td As TableDef

        Set td = dbWater.CreateTableDef(tbname_azoto)

        Dim data As field
        Set data = td.CreateField("DATA", dbDate)
        td.fields.Append data

        Dim ind As Index
        Set ind = td.CreateIndex("Primario")
        Set data = ind.CreateField("Data")
        ind.Primary = True
        ind.Required = True
        ind.IgnoreNulls = False
        ind.Unique = True

        ind.fields.Append data
        td.Indexes.Append ind

        If FlVarOutN_NO3strato = True Then

            Dim fldN_NO3() As field
            ReDim fldN_NO3(NOutputLayer)
            For L = 1 To NOutputLayer
                Set fldN_NO3(L) = td.CreateField("N_NO3" + CStr(L), dbSingle)
                td.fields.Append fldN_NO3(L)
            Next L

        End If

        If FlVarOutN_NH4strato = True Then
            Dim fldN_NH4() As field
            ReDim fldN_NH4(NOutputLayer)
            For L = 1 To NOutputLayer
                Set fldN_NH4(L) = td.CreateField("N_NH4" + CStr(L), dbSingle)
                td.fields.Append fldN_NH4(L)
            Next L
        End If

        If FlVarOutProfiloNH4 = True Then
            Dim fldProfiloNH4 As field 'GG
            Set fldProfiloNH4 = td.CreateField("ProfiloNH4", dbSingle)
            td.fields.Append fldProfiloNH4
        End If

        If FlVarOutN_NH4_fertGG = True Then
            Dim fldN_NH4_fertGG As field 'GG
            Set fldN_NH4_fertGG = td.CreateField("N_NH4_fertGG", dbSingle)
            td.fields.Append fldN_NH4_fertGG
        End If

        If FlVarOutN_min_humusGG = True Then
            Dim fldN_min_humusGG As field
            Set fldN_min_humusGG = td.CreateField("N_min_humusGG", dbSingle)
            td.fields.Append fldN_min_humusGG
        End If

        If FlVarOutN_min_litterGG = True Then
            Dim fldN_min_litterGG As field 'NEW!
            Set fldN_min_litterGG = td.CreateField("N_min_litterGG", dbSingle)
            td.fields.Append fldN_min_litterGG
        End If

        If FlVarOutN_idr_ureaGG = True Then
            Dim fldN_idr_ureaGG As field 'GG
            Set fldN_idr_ureaGG = td.CreateField("N_idr_ureaGG", dbSingle)
            td.fields.Append fldN_idr_ureaGG
        End If

        If FlVarOutN_NH4adsorbGG = True Then
            Dim fldN_NH4adsorbGG As field 'NEW!
            Set fldN_NH4adsorbGG = td.CreateField("N_NH4adsorbGG", dbSingle)
            td.fields.Append fldN_NH4adsorbGG
        End If

        If FlVarOutN_imm_l_NH4GG = True Then
            Dim fldN_imm_l_NH4GG As field 'NEW!
            Set fldN_imm_l_NH4GG = td.CreateField("N_imm_l_NH4GG", dbSingle)
            td.fields.Append fldN_imm_l_NH4GG
        End If

        If FlVarOutN_volGG = True Then
            Dim fldN_volGG  As field 'GG
            Set fldN_volGG = td.CreateField("N_volGG", dbSingle)
            td.fields.Append fldN_volGG
        End If

        If FlVarOutN_nitrifGGNH4 = True Then
            Dim fldN_nitrifGGnh4 As field 'GG
            Set fldN_nitrifGGnh4 = td.CreateField("N_nitrifGGnh4", dbSingle)
            td.fields.Append fldN_nitrifGGnh4
        End If

        If FlVarOutN_NH4_uptakeGG = True Then
            Dim fldN_NH4_uptakeGG As field 'GG
            Set fldN_NH4_uptakeGG = td.CreateField("NH4_uptakeGG", dbSingle)
            td.fields.Append fldN_NH4_uptakeGG
        End If

        If FlVarOutN_NH4_runoff0GG = True Then
            Dim fldN_NH4_runoff0GG As field
            Set fldN_NH4_runoff0GG = td.CreateField("N_NH4_runoff0GG", dbSingle)
            td.fields.Append fldN_NH4_runoff0GG
        End If

        If FlVarOutN_NH4_runoffGG = True Then
            Dim fldN_NH4_runoffGG As field
            Set fldN_NH4_runoffGG = td.CreateField("N_NH4_runoffGG", dbSingle)
            td.fields.Append fldN_NH4_runoffGG
        End If

        If FlVarOutFlux_NH4GG = True Then
            Dim fldFlux_NH4GG As field
            Set fldFlux_NH4GG = td.CreateField("Flux_NH4GG", dbSingle)
            td.fields.Append fldFlux_NH4GG
        End If

        If FlVarOutBilFinaleNH4 = True Then
            Dim fldBilFinaleNH4 As field
            Set fldBilFinaleNH4 = td.CreateField("BilFinaleNH4", dbSingle)
            td.fields.Append fldBilFinaleNH4
        End If

        'BILANCIO NO3
        If FlVarOutProfiloNO3 = True Then
            Dim fldProfiloNO3 As field 'GG
            Set fldProfiloNO3 = td.CreateField("ProfiloNO3", dbSingle)
            td.fields.Append fldProfiloNO3
        End If

        If FlVarOutN_NO3_fertGG = True Then
            Dim fldN_NO3_fertGG As field 'GG
            Set fldN_NO3_fertGG = td.CreateField("N_NO3_fertGG", dbSingle)
            td.fields.Append fldN_NO3_fertGG
        End If

        If FlVarOutPrecN_NO3GG = True Then
            Dim fldPrecN_NO3GG As field 'GG
            Set fldPrecN_NO3GG = td.CreateField("PrecN_NO3GG", dbSingle)
            td.fields.Append fldPrecN_NO3GG
        End If

        If FlVarOutN_nitrifGGNO3 = True Then
            Dim fldN_nitrifGGno3 As field 'sommato a NO3 e sottratto a NH4!
            Set fldN_nitrifGGno3 = td.CreateField("N_nitrifGGno3", dbSingle)
            td.fields.Append fldN_nitrifGGno3
        End If

        If FlVarOutN_imm_l_NO3GG = True Then
            Dim fldN_imm_l_NO3GG As field 'NEW!
            Set fldN_imm_l_NO3GG = td.CreateField("N_imm_l_NO3GG", dbSingle)
            td.fields.Append fldN_imm_l_NO3GG
        End If

        If FlVarOutN_denitrGG = True Then
            Dim fldN_denitrGG As field 'GG
            Set fldN_denitrGG = td.CreateField("N_denitrGG", dbSingle)
            td.fields.Append fldN_denitrGG
        End If

        If FlVarOutN_NO3_uptakeGG = True Then
            Dim fldN_NO3_uptakeGG As field 'GG
            Set fldN_NO3_uptakeGG = td.CreateField("NO3_uptakeGG", dbSingle)
            td.fields.Append fldN_NO3_uptakeGG
        End If

        If FlVarOutN_NO3_runoff0GG = True Then
            Dim fldN_NO3_runoff0GG As field
            Set fldN_NO3_runoff0GG = td.CreateField("N_NO3_runoff0GG", dbSingle)
            td.fields.Append fldN_NO3_runoff0GG
        End If

        If FlVarOutN_NO3_runoffGG = True Then
            Dim fldN_NO3_runoffGG As field
            Set fldN_NO3_runoffGG = td.CreateField("N_NO3_runoffGG", dbSingle)
            td.fields.Append fldN_NO3_runoffGG
        End If

        If FlVarOutFlux_NO3GG = True Then
            Dim fldFlux_NO3GG As field
            Set fldFlux_NO3GG = td.CreateField("Flux_NO3GG", dbSingle)
            td.fields.Append fldFlux_NO3GG
        End If

        If FlVarOutBilFinaleNO3 = True Then
            Dim fldBilFinaleNO3 As field
            Set fldBilFinaleNO3 = td.CreateField("BilFinaleNO3", dbSingle)
            td.fields.Append fldBilFinaleNO3
        End If

        'GENERALE
        If FlVarOutC_litter_humusGG = True Then
            Dim fldC_litter_humusGG As field 'GG
            Set fldC_litter_humusGG = td.CreateField("C_litter_humusGG", dbSingle)
            td.fields.Append fldC_litter_humusGG
        End If

        If FlVarOutC_litter_litterGG = True Then
            Dim fldC_litter_litterGG As field 'GG
            Set fldC_litter_litterGG = td.CreateField("C_litter_litterGG", dbSingle)
            td.fields.Append fldC_litter_litterGG
        End If

        If FlVarOutC_min_litterGG = True Then
            Dim fldC_min_litterGG As field 'GG
            Set fldC_min_litterGG = td.CreateField("C_min_litterGG", dbSingle)
            td.fields.Append fldC_min_litterGG
        End If

        If FlVarOutC_min_humusGG = True Then
            Dim fldC_min_humusGG As field 'GG
            Set fldC_min_humusGG = td.CreateField("C_min_humusGG", dbSingle)
            td.fields.Append fldC_min_humusGG
        End If

        If FlVarOutC_humusGG = True Then
            Dim fldC_humusGG As field 'GG
            Set fldC_humusGG = td.CreateField("C_humusGG", dbSingle)
            td.fields.Append fldC_humusGG
        End If

        If FlVarOutC_litterGG = True Then
            Dim fldC_litterGG As field 'GG
            Set fldC_litterGG = td.CreateField("C_litterGG", dbSingle)
            td.fields.Append fldC_litterGG
        End If

        If FlVarOutN_humusGG = True Then
            Dim fldN_humus_totGG As field
            Set fldN_humus_totGG = td.CreateField("N_humusGG", dbSingle)
            td.fields.Append fldN_humus_totGG
        End If

        If FlVarOutN_litterGG = True Then
            Dim fldN_litter_totGG As field
            Set fldN_litter_totGG = td.CreateField("N_litterGG", dbSingle)
            td.fields.Append fldN_litter_totGG
        End If

        If FlVarOutN_uptakePOTGG = True Then
            Dim fldN_uptakePOTGG As field 'GG
            Set fldN_uptakePOTGG = td.CreateField("N_uptakePOTGG", dbSingle)
            td.fields.Append fldN_uptakePOTGG
        End If

        dbNitrogen.TableDefs.Append td

    End If

    Set tbUscite_Azoto = dbNitrogen.OpenRecordset(tbname_azoto, dbOpenDynaset)

End Sub

// da valutare come riscrivere gli output
void N_Output()

Dim L As Integer

    If FlVarOutN_NO3strato Then SUB_UTIL_AggregaOutput N_NO3(), tbUscite_Azoto, "N_NO3", 10  'da [g m-2] a [kg ha-1]

    '2007.12 GA
    If FlVarOutN_NH4strato Then SUB_UTIL_AggregaOutput N_NH4(), tbUscite_Azoto, "N_NH4", 10
    If FlVarOutProfiloNH4 Then tbUscite_Azoto("ProfiloNH4") = max(tbUscite_Azoto("ProfiloNH4"), ProfiloNH4 * 10)  'da g m-2 a kg ha-1
    If FlVarOutN_NH4_fertGG Then tbUscite_Azoto("N_NH4_fertGG") = tbUscite_Azoto("N_NH4_fertGG") + N_NH4_fertGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_min_humusGG Then tbUscite_Azoto("N_min_humusGG") = tbUscite_Azoto("N_min_humusGG") + N_min_humusGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_min_litterGG Then tbUscite_Azoto("N_min_litterGG") = tbUscite_Azoto("N_min_litterGG") + N_min_litterGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_idr_ureaGG Then tbUscite_Azoto("N_idr_ureaGG") = tbUscite_Azoto("N_idr_ureaGG") + N_Urea_HydrGG * 10
    If FlVarOutN_NH4adsorbGG Then tbUscite_Azoto("N_NH4adsorbGG") = tbUscite_Azoto("N_NH4adsorbGG") + N_NH4_AdsorbedGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_imm_l_NH4GG Then tbUscite_Azoto("N_imm_l_NH4GG") = tbUscite_Azoto("N_imm_l_NH4GG") + N_imm_l_NH4GG * 10
    If FlVarOutN_volGG Then tbUscite_Azoto("N_volGG") = tbUscite_Azoto("N_volGG") + N_NH4_volGG * 10
    If FlVarOutN_nitrifGGNH4 Then tbUscite_Azoto("N_nitrifGGnh4") = tbUscite_Azoto("N_nitrifGGnh4") + N_nitrifGG * 10
    If FlVarOutN_NH4_uptakeGG Then tbUscite_Azoto("NH4_uptakeGG") = tbUscite_Azoto("NH4_uptakeGG") + N_NH4_uptakeGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_NH4_runoff0GG Then tbUscite_Azoto("N_NH4_runoff0GG") = tbUscite_Azoto("N_NH4_runoff0GG") + N_NH4_runoff0GG * 10
    If FlVarOutN_NH4_runoffGG Then tbUscite_Azoto("N_NH4_runoffGG") = tbUscite_Azoto("N_NH4_runoffGG") + N_NH4_runoffGG * 10
    If FlVarOutFlux_NH4GG Then tbUscite_Azoto("Flux_NH4GG") = tbUscite_Azoto("Flux_NH4GG") + Flux_NH4GG * 10
    If FlVarOutBilFinaleNH4 Then tbUscite_Azoto("BilFinaleNH4") = tbUscite_Azoto("BilFinaleNH4") + BilFinaleNH4 * 10

    If FlVarOutProfiloNO3 Then tbUscite_Azoto("ProfiloNO3") = max(tbUscite_Azoto("ProfiloNO3"), ProfiloNO3 * 10)  'da g m-2 a kg ha-1
    If FlVarOutN_NO3_fertGG Then tbUscite_Azoto("N_NO3_fertGG") = tbUscite_Azoto("N_NO3_fertGG") + N_NO3_fertGG * 10 'da g m-2 a kg ha-1
    If FlVarOutPrecN_NO3GG Then tbUscite_Azoto("PrecN_NO3GG") = tbUscite_Azoto("PrecN_NO3GG") + PrecN_NO3GG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_nitrifGGNO3 Then tbUscite_Azoto("N_nitrifGGno3") = tbUscite_Azoto("N_nitrifGGno3") + N_nitrifGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_imm_l_NO3GG Then tbUscite_Azoto("N_imm_l_NO3GG") = tbUscite_Azoto("N_imm_l_NO3GG") + N_imm_l_NO3GG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_denitrGG Then tbUscite_Azoto("N_denitrGG") = tbUscite_Azoto("N_denitrGG") + N_denitrGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_NO3_uptakeGG Then tbUscite_Azoto("NO3_uptakeGG") = tbUscite_Azoto("NO3_uptakeGG") + N_NO3_uptakeGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_NO3_runoff0GG Then tbUscite_Azoto("N_NO3_runoff0GG") = tbUscite_Azoto("N_NO3_runoff0GG") + N_NO3_runoff0GG * 10
    If FlVarOutN_NO3_runoffGG Then tbUscite_Azoto("N_NO3_runoffGG") = tbUscite_Azoto("N_NO3_runoffGG") + N_NO3_runoffGG * 10
    If FlVarOutFlux_NO3GG Then tbUscite_Azoto("Flux_NO3GG") = tbUscite_Azoto("Flux_NO3GG") + Flux_NO3GG * 10
    If FlVarOutBilFinaleNO3 Then tbUscite_Azoto("BilFinaleNO3") = tbUscite_Azoto("BilFinaleNO3") + BilFinaleNO3 * 10

'GENERALE
    If FlVarOutC_litter_humusGG Then tbUscite_Azoto("C_litter_humusGG") = tbUscite_Azoto("C_litter_humusGG") + C_litter_humusGG * 10 'da g m-2 a kg ha-1
    If FlVarOutC_litter_litterGG Then tbUscite_Azoto("C_litter_litterGG") = tbUscite_Azoto("C_litter_litterGG") + C_litter_litterGG * 10 'da g m-2 a kg ha-1
    If FlVarOutC_min_litterGG Then tbUscite_Azoto("C_min_litterGG") = tbUscite_Azoto("C_min_litterGG") + C_min_litterGG * 10 'da g m-2 a kg ha-1
    If FlVarOutC_min_humusGG Then tbUscite_Azoto("C_min_humusGG") = tbUscite_Azoto("C_min_humusGG") + C_min_humusGG * 10 'da g m-2 a kg ha-1

    If FlVarOutC_humusGG Then tbUscite_Azoto("C_humusGG") = tbUscite_Azoto("C_humusGG") + C_humusGG * 10 'da g m-2 a kg ha-1
    If FlVarOutC_litterGG Then tbUscite_Azoto("C_litterGG") = tbUscite_Azoto("C_litterGG") + C_litterGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_humusGG Then tbUscite_Azoto("N_humusGG") = tbUscite_Azoto("N_humusGG") + N_humusGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_litterGG Then tbUscite_Azoto("N_litterGG") = tbUscite_Azoto("N_litterGG") + N_litterGG * 10 'da g m-2 a kg ha-1
    If FlVarOutN_uptakePOTGG Then tbUscite_Azoto("N_uptakePOTGG") = tbUscite_Azoto("N_uptakePOTGG") + N_DailyDemand * 10

End Sub
*/
void Crit3DCarbonNitrogenWholeProfile::N_main(float precGG,int nrLayers,float* theta,std::vector<soil::Crit3DLayer> &soilLayers,soil::Crit3DSoil* soil)
{
    //++++++++++ MAIN NITROGEN ROUTINE +++++++++++++++++++++++++++++++++++
    //2008.09 GA
    //04.08.05.VM Version 2004.4
    //04.02.20.VM VERSION 2004.3
    //04.01.28.FZ REVISIONE 2004.2
    //04.01.09.FZ REVISIONE 2004.1
    //02.11.26.MVS riscritto il vecchio SUB_SOIL_N

    //inputs from precipitation
        //da dati deposizioni umide ARPA
        //pianura ravennate-bolognese media areale di 400 eqN ha-1 y-1 (1 eqN = 14 g)
        //6 kg ha-1 y-1
        //considerando una precipitazione annua di 400 mm
        //0.015 kg ha-1 N mm-1
        //0.0015 g m-2 N mm-1
        //suddividendo equamente tra NO3 e NH4
        //0.00075 g m-2 N nitrico e ammoniacale mm-1
        //controllare dati di concentrazione ARPA piogge
    numberOfLayers = nrLayers;
    arrayCarbonNitrogen = (Crit3DCarbonNitrogen*) calloc(numberOfLayers,sizeof(Crit3DCarbonNitrogen));
    // (float *) calloc(nrLayers, sizeof(float));
    if (precGG > 0)
    {
        precN_NO3GG = 0.00075 * precGG;
        precN_NH4GG = 0.00075 * precGG;
        arrayCarbonNitrogen[0].N_NO3 += precN_NO3GG;
        arrayCarbonNitrogen[0].N_NH4 += precN_NH4GG;
        partitioning(theta,soilLayers,soil);
    }
    /*
    N_Uptake();
    // definire attuale
    if (Attuale == Date_N_EndCrop)
        N_Harvest();

    if (Attuale == Date_N_Plough)
        N_Plough();

    //ciclo della sostanza organica e le trasformazioni dell'azoto
    chemicalTransformations();
    partitioning();

    //flussi di azoto nel suolo
    float myPistonDepth;
        //myPistonDepth = FindPistonDepth
        //SoluteFluxesPiston N_NO3, myPistonDepth, Flux_NO3GG
        //SoluteFluxesPiston N_NH4, myPistonDepth, Flux_NH4GG

    soluteFluxes(N_NO3, FlagWaterTableUpward, myPistonDepth, flux_NO3GG);

    soluteFluxes (N_NH4_Sol, FlagWaterTableUpward, myPistonDepth, Flux_NH4GG);
    updateTotalOfPartitioned N_NH4, N_NH4_Adsorbed, N_NH4_Sol);
    partitioning();

    if (FlagWaterTableWashing)
        leachingWaterTable(N_NO3, flux_NO3GG);

    if (FlagWaterTableWashing)
        leachingWaterTable(N_NH4_Sol, Flux_NH4GG);
    updateTotalOfPartitioned(N_NH4, N_NH4_Adsorbed, N_NH4_Sol);
    partitioning();

    // perdita superficiale
    if (FlagRunoff == 1)
        N_SurfaceRunoff();
    partitioning();

    // perdita per ruscellamento ipodermico
    If (FlagSSRunoff == 1 && FlagInfiltration != infiltration_1d)
        N_SubSurfaceRunoff();
    Partitioning

    //bilanci
    NH4_Balance();
    NO3_Balance();
    */
}
/*
float CNRatio(float c,float n)
{
    // 2004.02.20.VM
    // computes the C/N ratio
    if (FlagSO != 1)
        return 20.;
    if (n > 0.000001)
        return maxValue(0.001, c/n);
    else
        return 100.;
}

void computeWaterCorrectionFactor(int L)
{
    // LEACHM

    static double AOPT = 0.08;           // High end of optimum water content range, air-filled porosity
    static double SCORR = 0.6;           // Relative transformation rate at saturation (except denitrification)
    static int RM = 1;
    float wHigh;
    float wMin;
    float wLow;
    float myTheta, myThetaPA, myThetaCC, myThetaSAT;

    myTheta = WaterBalance.ConvertWCToVolumetric(suolo[L], U[L]);
    myThetaPA = WaterBalance.ConvertWCToVolumetric(suolo[L], suolo[L].PA);
    myThetaCC = WaterBalance.ConvertWCToVolumetric(suolo[L], suolo[L].CC);
    myThetaSAT = WaterBalance.ConvertWCToVolumetric(suolo[L], suolo[L].SAT);

    wMin = myThetaPA;
    wLow = myThetaCC;
    wHigh = myThetaSAT - AOPT;

    if (myTheta > wHigh Then)
        waterCorrectionFactor[L] = pow(SCORR + (1 - SCORR) * ((myThetaSAT - myTheta) / (myThetaSAT - wHigh)),RM);
    else if (myTheta <= wHigh && myTheta >= wLow)
        waterCorrectionFactor[L] = 1;
    else if (myTheta < wLow)
        waterCorrectionFactor[L] = pow(((maxValue(myTheta, wMin) - wMin) / (wLow - wMin)),RM);
}

void computeTemperatureCorrectionFactor(int L)
{
    //2008.10 GA
    //2004.02.20.VM
    //computes the temperature correction factor
    //----- Inputs --------------------
    //T [°C] temperature
    //Q10 [-] rate increase every 10 °C
    //Tbase [°C] base temperature

    if (flagHeat)
        temperatureCorrectionFactor[L] = pow(Q10, ((soilTemperature[L] - Tbase) / 10));
    else
        temperatureCorrectionFactor[L] = 1;
}

void computeLayerRates(int L)
{
    float totalCorrectionFactor;
    float wCorr_Denitrification;
    float theta;
    float thetaSAT;
    float conc_N_NO3;

    // update C/N ratio (fixed for humus and biomass)
    CNratio_litter[L] = CNRatio(C_litter[L], N_litter[L]);

    totalCorrectionFactor = temperatureCorrectionFactor[L] * waterCorrectionFactor[L];

    // carbon

    // humus mineralization
    actualRate_C_HumusMin = rate_C_HumusMin * totalCorrectionFactor;

    // litter to humus
    actualRate_C_LitterToHumus = rate_C_LitterMin * totalCorrectionFactor * (FE * FH);

    // litter to CO2
    actualRate_C_LitterToCO2 = rate_C_LitterMin * totalCorrectionFactor * (1 - FE);

    // litter to biomass
    actualRate_C_LitterToBiomass = rate_C_LitterMin * TotalCorrectionFactor * FE * (1 - FH);

    //nitrogen

    // litter mineralization/immobilization
    actualRate_N_LitterMin = maxValue(0, 1 / CNratio_litter[L] - FE / CNratio_biomass);
    if (N_litter[L] > 0)
        actualRate_N_LitterImm = -minValue(0, 1 / CNratio_litter[L] - FE / CNratio_biomass);
    else
        actualRate_N_LitterImm = 0;

    //nitrification
    actualRate_N_Nitrification = Rate_N_Nitrification * totalCorrectionFactor;

    // denitrification
    thetaSAT = orizzonti(suolo[L].Orizzonte).thetaS;
    theta = WaterBalance.ConvertWCToVolumetric(suolo[L], U[L]);
    wCorr_Denitrification = pow(maxValue(0, (theta - (1 - Max_afp_denitr) * thetaSAT)) / (thetaSAT - (1 - Max_afp_denitr) * ThetaSAT)), 2);
    conc_N_NO3 = convertToGramsPerLiter(L, N_NO3[L]) * 1000 'mg l-1;
    actualRate_N_Denitrification = Rate_N_Denitrification * temperatureCorrectionFactor[L] * wCorr_Denitrification
        * conc_N_NO3 / (conc_N_NO3 + Csat_denitr);

    // urea hydrolysis
    actualRate_Urea_Hydr = rate_Urea_Hydr * totalCorrectionFactor;

}
*/
void Crit3DCarbonNitrogenWholeProfile::N_Uptake()
{
    // 2008.09 GA ristrutturazione in base a LEACHM
    //           + nuovo calcolo potenziale uptake giornaliero (eliminato FGS)
    // 04.03.02.FZ modifica percentuali in NradiciCum
    // 02.11.26.MVS
    // 01.01.10.GD

    /*float N_max_transp;          // potential N uptake in transpiration stream
    float* N_NO3_up_max = (float *) calloc(numberOfLayers, sizeof(float));
    float* N_NH4_up_max = (float *) calloc(numberOfLayers, sizeof(float));
    int L;

    if (LAI <= 0)
    {
        return;
    }

    // uptake da germinazione a raccolta
    if (GGAttuale <= GGGermination)
    {
        return;
    }

    // controlla se ho esaurito il totale assimilabile
    if (N_PotentialDemandCumulated >= N_Uptakable)
    {
        return;
    }

    // uptake potenziale (dipendente da LAI)
    N_Uptake_Potential();

    if (N_DailyDemand == 0)
    {
        return
    }

    for(L=0;l>nrLayers;L++)
    {
        N_NO3_up_max[L] = 0;
        N_NH4_up_max[L] = 0;
    }

    //2008.09 GA niente residuo
    //aggiungo eventuale residuo
    //N_Uptake_Max

    N_UptakeMax = N_DailyDemand;

    if ((TR == 0) || (TM == 0))
    {
        return;
    }

    // calcolo massimi uptake per specie
    N_max_transp = 0;
    for (L = PSR; L< USR; L++)
    {
        if (TReale[L] > 0)
        {
            N_NO3_up_max[L] = N_NO3[L] / umid[L].BeforeTranspiration * TReale[L];
            N_NH4_up_max[L] = N_NH4_Sol[L] / umid[L].BeforeTranspiration * TReale[L];
        }
        else
        {
            N_NO3_up_max[L] = 0;
            N_NH4_up_max[L] = 0;
        }

        N_max_transp += N_NO3_up_max[L] + N_NH4_up_max[L];
    }

    if (N_max_transp > 0)
    {
        for (L = PSR;L<USR;L++)
        {
            N_NO3_uptake[L] = minValue(N_NO3[L], (N_NO3_up_max[L] / N_max_transp) * N_UptakeMax);
            //GA2017 dialogo con Ceotto (mais San Prospero)
            N_NH4_uptake[L] = 0 'min(N_NH4_Sol[L], (N_NH4_up_max[L] / N_max_transp) * N_UptakeMax)
        }
    }
*/
}
/*
void N_SurfaceRunoff()
{
    //-----------------------------------------
    //02.11.19.MVS Surface separato da Subsurface
    //-------------- NOTE -----------------------------------------------------
    //sub la stima del N asportato tramite l'acqua di ruscellamento superficiale

    if (supRunoffGG > 0)
    {
        // calcolo dell'azoto perso nel ruscellamento superficiale
        // seguendo i calcoli tratti da EPIC per il fosforo
        N_NO3_runoff0GG = minValue(N_NO3[0], N_NO3[0] / umid[0].BeforeRunoff * supRunoffGG);
        N_NH4_runoff0GG = minValue(N_NH4_Sol[0], N_NH4_Sol[0] / umid[0].BeforeRunoff * supRunoffGG);

        N_NO3[1] -= N_NO3_runoff0GG;
        N_NH4[1] -= N_NH4_runoff0GG;

    }

}


void N_SubSurfaceRunoff()
{
    //02.11.19.MVS Surface separato da Subsurface
    //02.03.14.GD
    //02.03.05.GD.MVS ruscellamento superficiale
    //02.03.04.GD
    //-------------- NOTE -----------------------------------------------------
    //sub la stima del N asportato tramite l'acqua di ruscellamento ipodermico

    int L;

    if (hypRunoffGG > 0)
    {
        // ReDim N_NH4_conc(nrLayers) // capire cosa sono questi 2
        // ReDim N_NO3_conc(nrLayers) // capire cosa sono questi 2

        for (L = 0; L<nrLayers;L++)
        {
            if (suolo[L].prof + suolo[L].spess) > PScol)
            {
                break;
            }

            if (umid[L].BeforeSubrunoff > 0)
            {
                // calcolo dell'azoto perso nel ruscellamento ipodermico
                N_NO3_runoff[L] = minValue(N_NO3[L], N_NO3[L] / umid[L].BeforeSubrunoff * runOff[L]);
                N_NH4_runoff[L] = minValue(N_NH4_Sol[L], N_NH4_Sol(L) / umid[L].BeforeSubrunoff * runOff[L]);

                N_NO3_runoffGG += N_NO3_runoff[L];
                N_NH4_runoffGG += N_NH4_runoff[L];

                N_NO3[L] -= N_NO3_runoff[L];
                N_NH4[L] -= N_NH4_runoff[L];
            }
        }

    }

}


void N_Uptake_Potential()
{
    //2008.09 GA nuova routine per calcolo di domanda di azoto
    //2008.04 GA
    //2002.11.26.MVS nuova routine a partire dal calcolo del lai

    N_DailyDemand = 0;

    //per evitare salti bruschi appena il LAI parte
    if (LAI_previous == 0)
    {
        return;
    }

    //solo in fase di crescita
    if (GGAttuale > (GGCrescita + GGEmergence))
    {
        return;
    }
    N_DailyDemand = minValue(maxValue(0, LAI - LAI_previous) * MaxRate_LAI_Ndemand, MaxRate_LAI_Ndemand);
    N_PotentialDemandCumulated += N_DailyDemand;

}

void N_Uptake_Max()
{
    //'2008.02 GA revisione (da manuale LEACHM)
    //'2002.11.19.MVS

    int L; //contatore
    //-------------------------------------------------------------------------
    int myDays;
    int i;
    float previousDeficit;

    // per medica non c'è deficit
    if ((coltura == Crops.CROP_ALFALFA) || (coltura == Crops.CROP_ALFALFA_FIRSTYEAR) || (coltura == Crops.CROP_SOYBEAN))
    {
        N_UptakeDeficit = 0;
        return;
    }

    // aggiorno deficit degli ultimi giorni
    previousDeficit = 0;
    myDays = UBound(Nitrogen.N_deficit_daily); //!! da modificare tutta la struttura in C usando array dinamici oppure inserendo qualche metodo alternativo di conteggio dei giorni
    if (myDays < nitrogen.N_deficit_max_days)
    {
        ReDim Preserve nitrogen.N_deficit_daily(myDays + 1);
    }
    for (i = 0;i<UBound(nitrogen.N_deficit_daily) - 1;i++)
    {
        nitrogen.N_deficit_daily(i) = nitrogen.N_deficit_daily(i + 1);
        previousDeficit += nitrogen.N_deficit_daily(i);
    }
    nitrogen.N_deficit_daily[UBound(nitrogen.N_deficit_daily)] = N_UptakeDeficit;
    N_UptakeDeficit = N_UptakeDeficit + previousDeficit;

    //'2008.02 GA verso la fine del ciclo la pianta il deficit non può essere totalmente compensato
    //'(LeachM)
    if (GGAttuale > (GGCrescita + GGEmergence))
    {
        N_UptakeDeficit = 0;
    }
    N_UptakeMax = N_DailyDemand + N_UptakeDeficit;
    N_UptakeDeficit = 0;
}


void N_Reset()
{
    //'02.11.26.MVS
    //'02.10.22.GD

    //'azzeramento giornaliero
    // credo che venga fatto così semplicemente per riazzerare piuttosto che cambiare dimensione
    ReDim N_imm_l_NH4(nrLayers)
    ReDim N_imm_l_NO3(nrLayers)

    ReDim C_litter_humus(nrLayers)
    ReDim C_litter_litter(nrLayers)
    ReDim C_min_humus(nrLayers)
    ReDim C_min_litter(nrLayers)
    ReDim C_denitr_litter(nrLayers)
    ReDim C_denitr_humus(nrLayers)

    ReDim N_NO3_uptake(nrLayers)
    ReDim N_NH4_uptake(nrLayers)

    ReDim N_min_humus(nrLayers)
    ReDim N_min_litter(nrLayers)
    ReDim N_litter_humus(nrLayers)
    ReDim N_nitrif(nrLayers)
    ReDim N_Urea_Hydr(nrLayers)
    ReDim N_vol(nrLayers)

    ReDim CNratio_litter(nrLayers)

    ReDim N_NO3_runoff(nrLayers)
    ReDim N_NH4_runoff(nrLayers)

    ReDim N_denitr(nrLayers)

    //'azzero tutte le variabili giornaliere
    //'bil NO3
    N_NO3_fertGG = 0;
    N_imm_l_NO3GG = 0;
    N_denitrGG = 0;//   'Denitrification non viene piu' chiamata
    N_NO3_uptakeGG = 0;
    N_NO3_runoff0GG = 0;
    N_NO3_runoffGG = 0;
    flux_NO3GG = 0;
    precN_NO3GG = 0;
    precN_NH4GG = 0;
    N_nitrifGG = 0;
    N_NH4_fertGG = 0;
    N_NH4_AdsorbedGG = 0;
    N_NH4_AdsorbedBeforeGG = 0;
    N_imm_l_NH4GG = 0;
    N_min_humusGG = 0;
    N_min_litterGG = 0;
    N_NH4_volGG = 0;
    N_Urea_HydrGG = 0;
    N_NH4_uptakeGG = 0;
    flux_NH4GG = 0;
    N_NH4_runoff0GG = 0;
    N_NH4_runoffGG = 0;
    N_humusGG = 0;
    N_litterGG = 0;
    C_humusGG = 0;
    C_litterGG = 0;
    C_min_humusGG = 0;
    C_min_litterGG = 0;
    C_litter_humusGG = 0;
    C_litter_litterGG = 0;
}


float findPistonDepth()
{
    int L;
    for (L = 0;L<nrLayers;L++)
    {
        if (umid[L].BeforeInfiltration > suolo[L].CC)
        {
            if (Flux[L] < (umid[L].BeforeInfiltration - suolo[L].CC))
            {
                break;
            }
        }
        else
        {
            break;
        }
    }
    if (L > nrLayers)
    {
        return MaxSoilDepth;
    }
    else
    {
        return suolo[L].prof;
    }
}



//calcolo dei flussi di soluti gravitazionali (a 'pistone')
void soluteFluxesPiston(float* mySolute, float PistonDepth,
    float* leached)
{
    int L;
    float myFreeSolute;
    double* f_Solute;
    f_Solute = (double *) calloc(nrLayers, sizeof(double));
    f_Solute[0] = 0;

    for (L = 0; L < nrLayers; L++)
    {
        f_Solute(L) = 0;

        // azoto in entrata da nodo L-1
        mySolute[L] += f_Solute[L - 1];

        if (suolo[L].prof >= PistonDepth)
        {
            break;
        }

//'        If umid(L).BeforeInfiltration > suolo(L).CC Then
//'            myFreeSolute = mySolute(L) * (umid(L).BeforeInfiltration - suolo(L).CC) / umid(L).BeforeInfiltration
//'        Else
//'            myFreeSolute = 0
//'        End If

        f_Solute[L] = minValue(mySolute[L], myFreeSolute / (umid[L].BeforeInfiltration) * Flux[L]);

        // azoto in uscita da nodo L
        mySolute[L] -= f_Solute[L];
    }

    //leaching
    *leached += f_Solute[nrLayers];
    free (f_Solute);
}


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
        f_solute = (double *) calloc(nrLayers, sizeof(double));

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
        *leached += flux_Solute[nrLayers-1];
        free (flux_Solute);
        free (u_temp);
        free (f_solute);

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

void NFromCropSenescence(float myDays,float coeffB) // this function must be public
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
