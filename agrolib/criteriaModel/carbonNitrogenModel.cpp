#include "commonConstants.h"
#include "carbonNitrogenModel.h"
#include <math.h>

Crit3DCarbonNitrogenSettings::Crit3DCarbonNitrogenSettings()
{
    rate_C_humusMin = 0.000005;
    rate_C_litterMin = 0.01;
    rate_N_NH4_volatilization = 0.4;
    rate_N_denitrification = 0.001;
    max_afp_denitr = 0.1;
    constant_sat_denitr = 10;
    rate_urea_hydr = 0.43;
    rate_N_nitrification = 0.0018;
    limRatio_nitr = 8;
    FE = 0.5;
    FH = 0.2;
    Q10 = 2.3;
    baseTemperature = 20;
    Kd_NH4 = 4;
    CN_RATIO_NOTHARVESTED = 30;
    LITTERINI_C_DEFAULT = 1200;   //[kg ha-1] initial litter carbon (default)
    LITTERINI_N_DEFAULT = 40;     //[kg ha-1] initial litter nitrogen (default)
    LITTERINI_DEPTH_DEFAULT = 30;  //[cm] initial litter depth (default)
    ratioBiomassCN = 7;
    ratioHumusCN = 7;
    ratioLitterCN = 7;
}


Crit3DCarbonNitrogenProfile::Crit3DCarbonNitrogenProfile()
{
    flagSOM = 1;

    // TODO initialize

}

double Crit3DCarbonNitrogenProfile::convertToGramsPerM3(double myQuantity, soil::Crit3DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] = [mg dm-3]
    return myQuantity / soilLayer.thickness;
}


double Crit3DCarbonNitrogenProfile::convertToGramsPerLiter(double myQuantity, soil::Crit3DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] -> [g l-1]
    return (convertToGramsPerM3(myQuantity, soilLayer) / 1000);
}


double Crit3DCarbonNitrogenProfile::convertToGramsPerKg(double myQuantity, soil::Crit3DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] -> [g kg-1]
    return (convertToGramsPerM3(myQuantity, soilLayer) / 1000) / soilLayer.horizon->bulkDensity;
}



void Crit3DCarbonNitrogenProfile::humusIni(Crit1DCase &myCase)
{
    //'2008.09 GA
    //'GA 2007.12 perché C calcolato da (CN/CN+1) e non moltiplicando per 0.58 come solito?
    //'computes initial humus carbon and nitrogen for a layer L
    //'version 1.0, 2004.08.09.VM


    // MVA                '[kg dm-3 = 10^6 g m-3]

    for (unsigned int l = 0; l<myCase.soilLayers.size();l++)
    {
        myCase.carbonNitrogenLayers[l].C_humus = myCase.soilLayers[l].horizon->bulkDensity * 1000000 * (myCase.soilLayers[l].horizon->organicMatter / 100) * 0.58 * myCase.soilLayers[l].thickness; // tolto il 100 perche gia in metri
        myCase.carbonNitrogenLayers[l].N_humus = myCase.carbonNitrogenLayers[l].C_humus / carbonNitrogenParameter.ratioHumusCN;
    }
}

double Crit3DCarbonNitrogenProfile::updateTotalOfPartitioned(double mySoluteAds,double mySoluteSol)
{
    return (mySoluteAds + mySoluteSol);
}

void Crit3DCarbonNitrogenProfile::partitioning(Crit1DCase &myCase)
{
    // partitioning of N (only NH4) between adsorbed and solute


    double N_NH4_g_dm3;               // [g dm-3] total ammonium
    double N_NH4_sol_g_l;             // [g l-1] ammonium in solution
    double N_NH4_ads_g_kg;            // [g kg-1] adsorbed ammonium
    double N_NH4_ads_g_m3;            // [g m-3] adsorbed ammonium
    double myTheta;

    N_NH4_adsorbedGG = 0;

    for (unsigned int l = 0; l < myCase.soilLayers.size(); l++)
    {
        myTheta = myCase.soilLayers[l].waterContent / (myCase.soilLayers[l].thickness * 1000);
        N_NH4_g_dm3 = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NH4, myCase.soilLayers[l]) / 1000;
        N_NH4_sol_g_l = N_NH4_g_dm3 / (carbonNitrogenParameter.Kd_NH4 * myCase.soilLayers[l].horizon->bulkDensity + myTheta);

        myCase.carbonNitrogenLayers[l].N_NH4_Sol = N_NH4_sol_g_l * myCase.soilLayers[l].thickness * myTheta * 1000;

        N_NH4_ads_g_kg = carbonNitrogenParameter.Kd_NH4 * N_NH4_sol_g_l;
        N_NH4_ads_g_m3 = N_NH4_ads_g_kg * myCase.soilLayers[l].horizon->bulkDensity * 1000;
        myCase.carbonNitrogenLayers[l].N_NH4_Adsorbed = N_NH4_ads_g_m3 * myCase.soilLayers[l].thickness;
        N_NH4_adsorbedGG += myCase.carbonNitrogenLayers[l].N_NH4_Adsorbed;
    }
}


void Crit3DCarbonNitrogenProfile::litterIni(Crit1DCase &myCase)
{
    //2008.10 GA inizializzazione indipendente da humus ma da input utente
    //computes initial litter carbon and nitrogen for a layer L
    //version 1.0, 2004.08.16.VM

    double layerRatio;
    double myDepth;
    if (fabs(litterIniC-NODATA) < EPSILON)
        litterIniC = carbonNitrogenParameter.LITTERINI_C_DEFAULT;
    if (fabs(litterIniN-NODATA) < EPSILON)
        litterIniN = carbonNitrogenParameter.LITTERINI_N_DEFAULT;
    if (fabs(litterIniDepth-NODATA) < EPSILON)
        litterIniDepth = carbonNitrogenParameter.LITTERINI_DEPTH_DEFAULT;

    myDepth = MAXVALUE(litterIniDepth, 6);
    myDepth /= 100; // transform depth from cm to m
    for (unsigned int l=0; l<myCase.soilLayers.size() ;l++)
    {
        if (myCase.soilLayers[l].depth <= myDepth)
        {
            layerRatio = (myCase.soilLayers[l].thickness / myDepth);
            myCase.carbonNitrogenLayers[l].C_litter = litterIniC * layerRatio / 10;          //from kg ha-1 to g m-2
            myCase.carbonNitrogenLayers[l].N_litter = litterIniN * layerRatio / 10;          //from kg ha-1 to g m-2
        }
    }
}

void Crit3DCarbonNitrogenProfile::chemicalTransformations(Crit1DCase &myCase)
{
    // 2013.06 GA
    // revision from LEACHM
    // concentrations in g m-3 (= mg dm-3)

    double myN_NO3;                   //[g m-3] nitric nitrogen concentration
    double myN_NH4;                   //[g m-3] ammonium concentration
    double myN_NH4_sol;               //[g m-3] ammonium concentration in solution
    double NH4_ratio;                 //[] ratio of NH4 on total N
    double NO3_ratio;                 //[] ratio of NO3 on total N
    double myTotalC;                  //[g m-3] total carbon
    double myLitterC;                 //[g m-3] carbon concentration in litter
    double myHumusC;                  //[g m-3] carbon concentration in humus
    double myLitterN;                 //[g m-3] nitrogen concentration in litter
    double myHumusN;                  //[g m-3] nitrogen concentration in humus

    double CCDEN;                     //[g m-3] Amount of carbon equivalent to N removed (as CO2) from denitrification
    double CLIMM;                     //[g m-3] maximum N immobilization
    double CLIMX;                     //[g m-3] ratio of the maximum to the desired rate. Adjusted by 0.08 to extend immobilization and slow the rate
    double CCLH;                      //[g m-3] carbon from litter to humus
    double CLH;                       //[g m-3] nitrogen from litter to humus
    double CLI;                       //[g m-3] nitrogen internally recycled in litter
    double CCLI;                      //[g m-3] carbon internally recycled in litter
    double CCLCO2;                    //[g m-3] carbon from litter to CO2
    double CCHCO2;                    //[g m-3] carbon from humus to CO2
    double CCLDN;                     //[g m-3] carbonio rimosso dal litter per denitrificazione
    double CCHDN;                     //[g m-3] carbonio rimosso dall'humus per denitrificazione
    double CNHNO;                     //[g m-3] source di nitrato (da nitrificazione)
    double CNON;                      //[g m-3] sink di nitrato (da denitrificazione)
    double CURNH;                     //[g m-3] urea hydrolysis
    double CHNH;                      //[g m-3] N humus mineralization
    double CLNH;                      //[g m-3] N litter mineralization
    double CNHGAS;                    //[g m-3] NH4 volatilization
    double CNHL;                      //[g m-3] N NH4 litter immobilization
    double CNOL;                      //[g m-3] N NO3 litter immobilization
    double USENH4;                    //[g m-3] N NH4 uptake
    double USENO3;                    //[g m-3] N NO3 uptake

    double litterCSink;               //[g m-3] C litter sink
    double litterCSource;             //[g m-3] C litter source
    double litterCRecycle;            //[g m-3] C litter recycle
    double litterCNetSink;            //[g m-3] C litter net sink/source
    double humusCSink;                //[g m-3] C humus sink
    double humusCSource;              //[g m-3] C humus source
    double humusCNetSink;             //[g m-3] C net sink/source
    double litterNSink;               //[g m-3] N litter sink
    double litterNSource;             //[g m-3] N litter source
    double litterNRecycle;            //[g m-3] N litter recycle
    double litterNNetSink;            //[g m-3] N litter net sink/source
    double humusNSink;                //[g m-3] N humus sink
    double humusNSource;              //[g m-3] N humus source
    double humusNNetSink;             //[g m-3] N net sink/source
    double N_NH4_sink;
    double N_NH4_source;
    double N_NH4_netSink;
    double N_NO3_sink;
    double N_NO3_source;
    double N_NO3_netSink;

    double totalCO2;                   //[g m-3] source of CO2
    double def;
    double total;
    double factor;

    static double adjustFactor = 0.08; // factor to extend immobilization and slow the rate

    for (unsigned l=0; l<myCase.soilLayers.size(); l++)
    {
        // correction functions for soil temperature and humidity
        // inserire parametri in Options.mdb
        bool flagHeat = false;
        double soilTemperature=25; // soil temperature should be computed
        double baseTemperature = 10;
        myCase.carbonNitrogenLayers[l].temperatureCorrectionFactor = computeTemperatureCorrectionFactor(flagHeat,soilTemperature,baseTemperature);
        myCase.carbonNitrogenLayers[l].waterCorrecctionFactor = computeWaterCorrectionFactor(l, myCase);

        // compute layer transformation rates
        computeLayerRates(l, myCase);

        // convert to concentration
        myLitterC = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].C_litter,myCase.soilLayers[l]);
        myHumusC = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].C_humus,myCase.soilLayers[l]);
        myLitterN = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_litter,myCase.soilLayers[l]);
        myHumusN = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_humus,myCase.soilLayers[l]);

        myTotalC = myLitterC + myHumusC;
        myN_NO3 = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NO3,myCase.soilLayers[l]);
        myN_NH4 = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NH4,myCase.soilLayers[l]);

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
        CCDEN = MAXVALUE(0, myN_NO3 * (1 - exp(-actualRate_N_denitrification))) * 72. / 56.;
        // No more than a tenth of C present C  can be removed
            CCDEN = MINVALUE(CCDEN,0.1 * myTotalC);


        // ii) litter transformation
        // litter C to humus C    
        CCLH = MAXVALUE(0, myLitterC * (1 - exp(-actualRate_C_litterToHumus)));
        // litter C internal recycle
        CCLI = MAXVALUE(0, myLitterC * (1 - exp(-actualRate_C_litterToHumus)));
        // litter C to CO2
        CCLCO2 = MAXVALUE(0, myLitterC * (1 - exp(-actualRate_C_litterToHumus)));
        // nitrogen immobilization
        CLIMX = 1;

        if (actualRate_N_litterImm > 0)
        {
            CLIMM = actualRate_N_litterImm * (CCLI + CCLH + CCLCO2);
            if (CLIMM > 0)
                CLIMX = MINVALUE(1, (adjustFactor * (myN_NO3 + myN_NH4)) / CLIMM);
            // if immobilization limits mineralization then the effective rate of immobilization is reduced
            if (CLIMX < 1)
            {
                CCLH *= CLIMX;
                CCLI *= CLIMX;
                CCLCO2 *= CLIMX;
            }
            CNHL = CLIMM * CLIMX * NH4_ratio;
        }
        else
        {
            CNHL = 0;
            CLIMM = 0;
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
        litterCRecycle = CCLI; // verified also in the original this term was not used
        litterCNetSink = -litterCSink + litterCSource;

        // iii) humus transformations
        // humus to CO2
        CCHCO2 = MAXVALUE(0, myHumusC * (1 - exp(-actualRate_C_humusMin)));
        // energy source for denitrification
        if (myTotalC > 0)
            CCHDN = CCDEN * myHumusC / myTotalC;
        else
            CCHDN = 0;

        humusCSink = CCHDN + CCHCO2;
        humusCSource = CCLH;
        humusCNetSink = humusCSource - humusCSink;

        // iv) CO2
        // TODO controllare i casi come questo (is never read)
        totalCO2 = CCLCO2 + CCHCO2 + CCLDN + CCHDN; // verified not used in the code


    // NITROGEN TRANSFORMATIONS
        // i) urea hydrolysis
        CURNH = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_urea, myCase.soilLayers[l]) * (1 - exp(-actualRateUreaHydr));

        // ii) ammonium volatilization (only top 5 cm of soil) (in LEACHM 10 cm but layer thickness is 10 cm)
        if ((myCase.soilLayers[l].depth + myCase.soilLayers[l].thickness) < 0.05)
        {
            myN_NH4_sol = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NH4_Sol,myCase.soilLayers[l]);
            CNHGAS =myN_NH4_sol * (MINVALUE(0.5,  (1 - exp(-carbonNitrogenParameter.rate_N_NH4_volatilization))));
        }
        else
            CNHGAS = 0;

        // iii) nitrification
        CNHNO = (MAXVALUE(0, myN_NH4 - myN_NO3 / carbonNitrogenParameter.limRatio_nitr)) * (1 - exp(-actualRate_N_nitrification));

        // iv) mineralization
        CHNH = MAXVALUE(myHumusN, 0) * actualRate_C_humusMin;
        CLNH = actualRate_N_litterMin * (CCLH + CCLCO2 + CCLI);
        //if (actualRate_N_litterImm > 0)
            //CNHL = CLIMM * CLIMX * NH4_ratio; // already computed above

        // NH4 sink/source
        USENH4 = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NH4_uptake,myCase.soilLayers[l]);
        N_NH4_sink = CNHNO + CNHL;
        N_NH4_source = CHNH + CLNH + CURNH;
        N_NH4_netSink = -N_NH4_sink + N_NH4_source - CNHGAS - USENH4;
        // adjustment
        def = myN_NH4 + N_NH4_netSink;
        total = CNHNO + CNHGAS + USENH4;
        if (def < 0 && total > 0)
        {
            factor = MAXVALUE(0, 1. + def / total);
            CNHNO *= factor;
            CNHGAS *= factor;
            USENH4 *= factor;
            N_NH4_sink = CNHNO + CNHL;
            N_NH4_netSink = -N_NH4_sink + N_NH4_source - CNHGAS - USENH4;
        }

        // NO3 sink/source
        USENO3 = convertToGramsPerM3(myCase.carbonNitrogenLayers[l].N_NO3_uptake,myCase.soilLayers[l]);
        if (actualRate_N_litterImm > 0)
            CNOL = CLIMM * CLIMX * NO3_ratio;
        else
            CNOL = 0;
        CNON = CCDEN * 56. / 72.;
        N_NO3_source = CNHNO;
        N_NO3_sink = CNOL;
        N_NO3_netSink = -N_NO3_sink + N_NO3_source - USENO3 - CNON;
        // adjustment
        def = myN_NO3 + N_NO3_netSink;
        total = USENO3 + CNON;

        if (def < 0 && total > 0)
        {
            factor = MAXVALUE(0., 1. + def / total);
            CNON *= factor;
            USENO3 *= factor;
            N_NO3_sink = CNON + CNOL;
            N_NO3_netSink = -N_NO3_sink + N_NO3_source - USENO3;
        }

        // litter N sink/source
        CLH = CCLH / ratio_CN_humus;
        CLI = CCLI / ratio_CN_humus;
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
        myCase.carbonNitrogenLayers[l].N_NO3_uptake = USENO3 * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_NH4_uptake = USENH4 * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_min_litter = CLNH * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_imm_l_NH4 = CNHL * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_imm_l_NO3 = CNOL * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_min_humus = CHNH * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_litter_humus = CLH * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_vol = CNHGAS * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_denitr = CNON * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_nitrif = CNHNO * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_Urea_Hydr = CURNH * myCase.soilLayers[l].thickness;

        myCase.carbonNitrogenLayers[l].C_litter_humus = CCLH * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_litter_litter = CCLI * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_min_litter = CCLCO2 * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_min_humus = CCHCO2 *  myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_denitr_humus = CCHDN * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_denitr_litter = CCLDN * myCase.soilLayers[l].thickness;

    // -----------------------------------------------------------------------------------
    // mass balancing

        myCase.carbonNitrogenLayers[l].N_NH4 += N_NH4_netSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_NO3 += N_NO3_netSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_litter += litterCNetSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].C_humus += humusCNetSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_litter += litterNNetSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_humus += humusNNetSink * myCase.soilLayers[l].thickness;
        myCase.carbonNitrogenLayers[l].N_urea -= myCase.carbonNitrogenLayers[l].N_Urea_Hydr;


        //If N_NH4(L) < 0 Then Stop
        //If N_NO3(L) < 0 Then Stop
        //If C_litter(L) < 0 Then Stop
        //If C_humus(L) < 0 Then Stop
        //If N_litter(L) < 0 Then Stop
        //If N_humus(L) < 0 Then Stop

        myCase.carbonNitrogenLayers[l].N_NH4 = MAXVALUE(0, myCase.carbonNitrogenLayers[l].N_NH4);
        myCase.carbonNitrogenLayers[l].N_NO3 = MAXVALUE(0, myCase.carbonNitrogenLayers[l].N_NO3);
        myCase.carbonNitrogenLayers[l].C_litter = MAXVALUE(0, myCase.carbonNitrogenLayers[l].C_litter);
        myCase.carbonNitrogenLayers[l].C_humus = MAXVALUE(0, myCase.carbonNitrogenLayers[l].C_humus);
        myCase.carbonNitrogenLayers[l].N_litter = MAXVALUE(0, myCase.carbonNitrogenLayers[l].N_litter);
        myCase.carbonNitrogenLayers[l].N_humus = MAXVALUE(0, myCase.carbonNitrogenLayers[l].N_humus);
        myCase.carbonNitrogenLayers[l].N_urea = MAXVALUE(0, myCase.carbonNitrogenLayers[l].N_urea);

        // profile totals
        N_humusGG += myCase.carbonNitrogenLayers[l].N_humus;
        N_litterGG += myCase.carbonNitrogenLayers[l].N_litter;
        N_litter_humusGG += myCase.carbonNitrogenLayers[l].N_litter_humus;
        N_min_humusGG += myCase.carbonNitrogenLayers[l].N_min_humus;
        N_min_litterGG += myCase.carbonNitrogenLayers[l].N_min_litter;
        N_imm_l_NH4GG += myCase.carbonNitrogenLayers[l].N_imm_l_NH4;
        N_imm_l_NO3GG += myCase.carbonNitrogenLayers[l].N_imm_l_NO3;
        C_humusGG += myCase.carbonNitrogenLayers[l].C_humus;
        C_min_humusGG += myCase.carbonNitrogenLayers[l].C_min_humus;
        C_litter_humusGG += myCase.carbonNitrogenLayers[l].C_litter_humus;
        C_litter_litterGG += myCase.carbonNitrogenLayers[l].C_litter_litter;
        C_min_litterGG += myCase.carbonNitrogenLayers[l].C_min_litter;
        C_litterGG += myCase.carbonNitrogenLayers[l].C_litter;
        N_NO3_uptakeGG += myCase.carbonNitrogenLayers[l].N_NO3_uptake;
        N_NH4_uptakeGG += myCase.carbonNitrogenLayers[l].N_NH4_uptake;
        N_NH4_volGG += myCase.carbonNitrogenLayers[l].N_vol;
        N_nitrifGG += myCase.carbonNitrogenLayers[l].N_nitrif;
        N_urea_hydrGG += myCase.carbonNitrogenLayers[l].N_Urea_Hydr;
        N_denitrGG += myCase.carbonNitrogenLayers[l].N_denitr;

    }

    updateNCrop(myCase.crop);

}


// da valutare
void Crit3DCarbonNitrogenProfile::N_Initialize()
{
    //****************************************************************
    //Scopo: lettura da tbAzoto di coefficienti relativi al ciclo di N
    //****************************************************************
    //04.08.04.VM eliminazione dei vettori di costanti
    //00.07.19.VM.MVS nuova sub chiamata da SUB_INI_Profilo per le variabili dell'azoto
    //****************************************************************

        // 2008.02 GA da inserire in database
        /*Nitrogen.N_deficit_max_days = 3
        ReDim Nitrogen.N_deficit_daily(0)
        */

        ratio_CN_humus = ratio_CN_biomass = 12; // !! to be read somewhere
        flagSOM = 1; // !! to be read somewhere
        flagWaterTableUpward = true;  // !! to be read somewhere
        flagWaterTableWashing = true;  // !! to be read somewhere
        litterIniC = NODATA; // !! to be read somewhere
        litterIniN = NODATA; // !! to be read somewhere
        litterIniDepth = NODATA; // !! to be read somewhere

        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots = 0;
}

void Crit3DCarbonNitrogenProfile::N_Fertilization(Crit1DCase &myCase,TfertilizerProperties fertilizerProperties)
{
    //07.12.17 GA cambiata dichiarazione da Integer a Single per concentrazioni (per valori frazionari)
    //02.11.27.MVS aggiunta concime organico
    //02.11.26.MVS aggiunta N_NO3_fertGG e N_NH4_fertGG per il bilancio
    //02.03.10.GD
    //01.01.10.GD
    //00.06.16.GD.MVS Questa sub è attivata nel momento della concimazione.

    double quantityNcm;     // quantity per cm
    double N_totalQuantity;
    std::vector<double> quantityN;
    quantityN.resize(myCase.soilLayers.size());
    std::vector<double> N_Norg_fert;
    N_Norg_fert.resize(myCase.soilLayers.size());
    std::vector<double> N_NO3_fert;
    N_NO3_fert.resize(myCase.soilLayers.size());
    std::vector<double> N_NH4_fert;
    N_NH4_fert.resize(myCase.soilLayers.size());

    /* farei questa trasformazione fuori non dentro la funzione
    if (ID_TipoConcime == "organico")
    {
        //letame e liquame è espresso in tonnellate
        QuantitàConcimeTot = QuantitàConcimeTot *= 1000;  //da tonnellate a kg
    }*/

    N_totalQuantity = fertilizerProperties.quantity / 100 * fertilizerProperties.N_content;

    //perdita immediata per volatilizzazione dell'urea
    if (fertilizerProperties.ID_FERTILIZER == FERTILIZER_UREA)
        N_totalQuantity *= 0.8; // assuming 20% becomes NH3 by volatilization

    //2007.04 GA se profondità = 0 (tutto nei primi 10 cm)
    if (fertilizerProperties.fertilizerDepth < 10)
        fertilizerProperties.fertilizerDepth = 10;

    // divido la quantità per cm
    quantityNcm = N_totalQuantity / fertilizerProperties.fertilizerDepth;
    unsigned int ll=1;
    for (unsigned int l=0; l<myCase.soilLayers.size(); l++)
    {
        if (N_totalQuantity <= 0)
        {
            ll = l - 1;
            break;
        }
        quantityN[l] = quantityNcm * myCase.soilLayers[l].thickness*100;
        if (quantityN[l] > N_totalQuantity)
            quantityN[l] = N_totalQuantity; //check to not overpass the fertilizer quantity in the last layer
        N_totalQuantity -= quantityN[l];
        quantityN[l] /= 10; // from kg ha-1 to g m-2
    }

    for (unsigned int l=0; l<ll; l++)
    {
        //'2007.12 GA inserita concimazione ureica (prima N_urea era sempre 0!)
        //'perdita del 30% per volatilizzazione immediata
        if (fertilizerProperties.ID_FERTILIZER == FERTILIZER_UREA)
            myCase.carbonNitrogenLayers[l].N_urea = quantityN[l];

        N_NO3_fert[l] = quantityN[l]*fertilizerProperties.N_NO3_percentage / 100;
        N_NH4_fert[l] = quantityN[l]*fertilizerProperties.N_NH4_percentage/ 100;
        N_Norg_fert[l] = quantityN[l]*fertilizerProperties.N_org_percentage/ 100;

        //per il bilancio...
        N_NO3_fertGG += N_NO3_fert[l];
        N_NH4_fertGG += N_NH4_fert[l];

        myCase.carbonNitrogenLayers[l].N_NO3 += N_NO3_fert[l];                                      // updating N_NO3
        myCase.carbonNitrogenLayers[l].N_NH4 += N_NH4_fert[l];                                      // updating N_NH4
        myCase.carbonNitrogenLayers[l].N_litter += N_Norg_fert[l];                                  // updating N_litter
        myCase.carbonNitrogenLayers[l].C_litter += N_Norg_fert[l] * fertilizerProperties.C_N_ratio; // updating C_litter
    }

}

// da valutare
void Crit3DCarbonNitrogenProfile::N_InitializeVariables(Crit1DCase &myCase)
{
    //2004.08.16.VM introduzione di FUN_CNhumus_INI e LitterIni
    //2004.08.05.VM eliminato da tbLog la costante CNratio_humus
    //2004.06.25.VM forzato la sostanza organica del suolo
    //2002.03.15.GD correzione calcolo N_humus
    //2000.11.20.MVS nuovo codice sulla lettura di tbIniProfilo e tblog
    //1999.12.02.MVS cambiamento tbUscite in tbUscite_azoto...
    //1999.05.20.GD
    //1999.03.15.GD
    //-------------- NOTE -----------------------------------------------------
    //Questa routine carica i valori iniziali nelle variabili
    //utilizzando due metodi alternativi:
    //1) nel caso la simulazione sia stata interrotta legge
    //   lo stato delle variabili dal database delle uscite
    //2) nel caso la simulazione sia nuova pone a zero oppure a valori
    //   iniziali le stesse variabili
    //-------------- Input Variables ------------------------------------------
    //nrLayers           [-] numero di strati
    //DataIniziale      [-] data di inizio della simulazione
    //-------------- Output Variables -----------------------------------------

    //-------------- Internal Variables ---------------------------------------

    /*Dim Nome$          '[-] nome del campo umidità


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

    */
    humusIni(myCase);
    litterIni(myCase);
    partitioning(myCase);

    double nitrogenPerCm = 0.05; // [g m-2] valore da rivedere
    // azzeramento delle variabili non rilette
    for(unsigned int l=0; l<myCase.soilLayers.size();l++)
    {
        myCase.carbonNitrogenLayers[l].N_NO3 = 100*myCase.soilLayers[l].thickness*nitrogenPerCm;
        myCase.carbonNitrogenLayers[l].N_NH4 = 100*myCase.soilLayers[l].thickness*nitrogenPerCm;
        myCase.carbonNitrogenLayers[l].N_NH4_Adsorbed = 0.1*(myCase.carbonNitrogenLayers[l].N_NH4);
        myCase.carbonNitrogenLayers[l].N_urea = 0;

    }
    profileNH4 = profileNO3 = 0;
    for(unsigned int l=0; l<myCase.soilLayers.size();l++)
    {
        profileNO3 += myCase.carbonNitrogenLayers[l].N_NO3;
        profileNH4 += myCase.carbonNitrogenLayers[l].N_NH4;
    }

    // End Sub
}
/*
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
void Crit3DCarbonNitrogenProfile::N_main(double precGG, Crit1DCase &myCase,Crit3DDate &myDate)
{
    //++++++++++ MAIN NITROGEN ROUTINE +++++++++++++++++++++++++++++++++++
    //2008.09 GA
    //04.08.05.VM Version 2004.4
    //04.02.20.VM VERSION 2004.3
    //04.01.28.FZ REVISIONE 2004.2
    //04.01.09.FZ REVISIONE 2004.1
    //02.11.26.MVS riscritto il vecchio SUB_SOIL_N

    //inputs from precipitation
        //wet deposition data (ARPAE)
        //values for the Bologna-Ravenna plain area about 400 eqN ha-1 y-1 (1 eqN = 14 g)
        //6 kg ha-1 y-1
        //considering average annual rainfall of 400 mm
        //0.015 kg ha-1 N mm-1
        //0.0015 g m-2 N mm-1
        //dividing equally between N-NO3 and N-NH4
        //0.00075 g m-2 N-NO3 and N-NH4 mm-1

    if (precGG > 0)
    {
        precN_NO3GG = 0.00075 * precGG;
        precN_NH4GG = 0.00075 * precGG;
        myCase.carbonNitrogenLayers[0].N_NO3 += precN_NO3GG;
        myCase.carbonNitrogenLayers[0].N_NH4 += precN_NH4GG;
        partitioning(myCase);
    }

    N_Uptake(myCase);
    int currentDOY = getDoyFromDate(myDate);

    // definire attuale
    if (currentDOY == getDoyFromDate(date_N_endCrop))
        N_harvest(myCase);

    if (currentDOY == getDoyFromDate(date_N_plough))
        N_plough(myCase);

    //ciclo della sostanza organica e le trasformazioni dell'azoto
    chemicalTransformations(myCase);
    partitioning(myCase);

    //flussi di azoto nel suolo
    double myPistonDepth;
    myPistonDepth = findPistonDepth(myCase);
        //SoluteFluxesPiston N_NO3, myPistonDepth, Flux_NO3GG
        //SoluteFluxesPiston N_NH4, myPistonDepth, Flux_NH4GG


    std::vector<double> mySolute;
    mySolute.resize(myCase.soilLayers.size());

    for(unsigned int l=0;l<myCase.soilLayers.size();l++)
    {
        mySolute[l] = myCase.carbonNitrogenLayers[l].N_NO3;
    }
    soluteFluxes(mySolute, flagWaterTableUpward , myPistonDepth, &flux_NO3GG, myCase);
    for(unsigned int l=0;l<myCase.soilLayers.size();l++)
    {
        mySolute[l] = myCase.carbonNitrogenLayers[l].N_NH4_Sol;
    }
    soluteFluxes(mySolute, flagWaterTableUpward, myPistonDepth, &flux_NH4GG, myCase);

    for(unsigned int l=0;l<myCase.soilLayers.size();l++)
    {
        myCase.carbonNitrogenLayers[l].N_NH4  = updateTotalOfPartitioned(myCase.carbonNitrogenLayers[l].N_NH4_Adsorbed, myCase.carbonNitrogenLayers[l].N_NH4_Sol);
    }
    partitioning(myCase);

    if (flagWaterTableWashing)
    {
        for(unsigned int l=0;l<myCase.soilLayers.size();l++)
        {
            mySolute[l] = myCase.carbonNitrogenLayers[l].N_NO3;
        }
        leachingWaterTable(mySolute, &flux_NO3GG,myCase);
        for(unsigned int l=0;l<myCase.soilLayers.size();l++)
        {
            mySolute[l] = myCase.carbonNitrogenLayers[l].N_NH4_Sol;
        }
        leachingWaterTable(mySolute, &flux_NH4GG,myCase);
    }
    mySolute.clear();
    for(unsigned int l=0;l<myCase.soilLayers.size();l++)
    {
        myCase.carbonNitrogenLayers[l].N_NH4  = updateTotalOfPartitioned(myCase.carbonNitrogenLayers[l].N_NH4_Adsorbed, myCase.carbonNitrogenLayers[l].N_NH4_Sol);
    }
    partitioning(myCase);

    // loss due to surface runoff
    bool flagRunoff = true;
    if (flagRunoff == true)
        N_SurfaceRunoff(myCase); // da modificare la funzione
    partitioning(myCase);
    /* the next if cycle should not be present in Criteria1D because there is no lateral movement
    // loss due to subsurface runoff
    If (FlagSSRunoff == 1 && FlagInfiltration != infiltration_1d)
        N_SubSurfaceRunoff();
    partitioning(myCase);
    */

    //nitrogen budgets
    NH4_Balance(myCase);
    NO3_Balance(myCase);
}


double Crit3DCarbonNitrogenProfile::CNRatio(double C,double N,int flagOrganicMatter)
{
    // 2004.02.20.VM
    // computes the C/N ratio
    if (flagOrganicMatter != 1)
        return 20.;
    if (N > 0.000001)
        return MAXVALUE(0.001, C/N);
    else
        return 100.;
}


double Crit3DCarbonNitrogenProfile::computeWaterCorrectionFactor(int l,Crit1DCase &myCase)
{
    // LEACHM

    static double AOPT = 0.08;           // High end of optimum water content range, air-filled porosity
    static double SCORR = 0.6;           // Relative transformation rate at saturation (except denitrification)
    static int RM = 1;
    double wHigh;
    double wMin;
    double wLow;
    double myTheta, myThetaWP, myThetaFC, myThetaSAT;

    myTheta = myCase.soilLayers[l].waterContent / (myCase.soilLayers[l].thickness * 1000);
    myThetaWP = myCase.soilLayers[l].WP / (myCase.soilLayers[l].thickness * 1000);
    myThetaFC = myCase.soilLayers[l].FC / (myCase.soilLayers[l].thickness * 1000);
    myThetaSAT = myCase.soilLayers[l].SAT / (myCase.soilLayers[l].thickness * 1000);

    wMin = myThetaWP;
    wLow = myThetaFC;
    wHigh = myThetaSAT - AOPT;

    if (myTheta > wHigh)
        return pow(SCORR + (1 - SCORR) * ((myThetaSAT - myTheta) / (myThetaSAT - wHigh)),RM);
    else if (myTheta < wLow)
        return pow(((MAXVALUE(myTheta, wMin) - wMin) / (wLow - wMin)),RM);
    else
        return 1;
}


double Crit3DCarbonNitrogenProfile::computeTemperatureCorrectionFactor(bool flagHeat, double layerSoilTemperature, double baseTemperature)
{
    //2008.10 GA
    //2004.02.20.VM
    //computes the temperature correction factor
    //----- Inputs --------------------
    //T [°C] temperature
    //Q10 [-] rate increase every 10 °C
    //Tbase [°C] base temperature

    if (flagHeat)
        return pow(carbonNitrogenParameter.Q10, ((layerSoilTemperature - baseTemperature) / 10.));
    else
        return 1;
}

void Crit3DCarbonNitrogenProfile::computeLayerRates(unsigned l, Crit1DCase &myCase)
{
    double totalCorrectionFactor;
    double wCorr_Denitrification;
    double theta;
    double thetaSAT;
    double conc_N_NO3;

    // update C/N ratio (fixed for humus and biomass)
    myCase.carbonNitrogenLayers[l].ratio_CN_litter = CNRatio(myCase.carbonNitrogenLayers[l].C_litter, myCase.carbonNitrogenLayers[l].N_litter,flagSOM);
    totalCorrectionFactor = myCase.carbonNitrogenLayers[l].temperatureCorrectionFactor * myCase.carbonNitrogenLayers[l].waterCorrecctionFactor;

    // carbon

    // humus mineralization
    actualRate_C_humusMin = carbonNitrogenParameter.rate_C_humusMin * totalCorrectionFactor;

    // litter to humus
    actualRate_C_litterToHumus = carbonNitrogenParameter.rate_C_litterMin * totalCorrectionFactor * (carbonNitrogenParameter.FE * carbonNitrogenParameter.FH);

    // litter to CO2
    actualRate_C_litterToCO2 = carbonNitrogenParameter.rate_C_litterMin * totalCorrectionFactor * (1 - carbonNitrogenParameter.FE);

    // litter to biomass
    actualRate_C_litterToBiomass = carbonNitrogenParameter.rate_C_litterMin * totalCorrectionFactor * carbonNitrogenParameter.FE * (1 - carbonNitrogenParameter.FH);

    //nitrogen

    // litter mineralization/immobilization
    actualRate_N_litterMin = MAXVALUE(0, 1. / myCase.carbonNitrogenLayers[l].ratio_CN_litter - carbonNitrogenParameter.FE / ratio_CN_biomass);
    if (myCase.carbonNitrogenLayers[l].N_litter > 0)
        actualRate_N_litterImm = -MINVALUE(0, 1. / myCase.carbonNitrogenLayers[l].ratio_CN_litter - carbonNitrogenParameter.FE / ratio_CN_biomass);
    else
        actualRate_N_litterImm = 0;

    //nitrification
    actualRate_N_nitrification = carbonNitrogenParameter.rate_N_nitrification * totalCorrectionFactor;

    // denitrification
    thetaSAT = myCase.soilLayers[l].SAT  / (myCase.soilLayers[l].thickness * 1000);
    theta = myCase.soilLayers[l].waterContent / (myCase.soilLayers[l].thickness * 1000);
    wCorr_Denitrification = pow(MAXVALUE(0, (theta - (1 - carbonNitrogenParameter.max_afp_denitr) * thetaSAT)) / (thetaSAT - (1 - carbonNitrogenParameter.max_afp_denitr) * thetaSAT), 2);
    conc_N_NO3 =convertToGramsPerLiter(myCase.carbonNitrogenLayers[l].N_NO3,myCase.soilLayers[l]) * 1000;

    actualRate_N_denitrification = carbonNitrogenParameter.rate_N_denitrification * myCase.carbonNitrogenLayers[l].temperatureCorrectionFactor * wCorr_Denitrification
        * conc_N_NO3 / (conc_N_NO3 + carbonNitrogenParameter.constant_sat_denitr);

    // urea hydrolysis

    actualRateUreaHydr = carbonNitrogenParameter.rate_urea_hydr * totalCorrectionFactor;

}

void Crit3DCarbonNitrogenProfile::N_Uptake(Crit1DCase &myCase)
{
    // 2008.09 GA written based on LEACHM
    //           + new computation for the daily uptake (quit FGS)
    // 04.03.02.FZ modifica percentuali in NradiciCum
    // 02.11.26.MVS
    // 01.01.10.GD
    double N_max_transp;          // potential N uptake in transpiration stream

    if (myCase.crop.LAI <= 0)
    {
        return;
    }
    if(myCase.crop.degreeDays <= myCase.crop.degreeDaysEmergence)
    {
        return;
    }

    // uptake during the crop cycle

    if (myCase.crop.degreeDays <= myCase.crop.degreeDaysEmergence)
    {
        return;
    }

    // check if total assimilable Nitrogen has already been taken up

    if (N_potentialDemandCum >= N_uptakable)
    {
        return;
    }

    // potential N uptake (dependent on LAI)
    N_Uptake_Potential(myCase);

    if (N_dailyDemand == 0)
    {
        return;
    }


    //2008.09 GA no residues
    //Add residues

    N_uptakeMax = N_dailyDemand;

    if ((myCase.output.dailyTranspiration == 0) || (myCase.crop.getMaxTranspiration(myCase.output.dailyEt0) == 0))
    {
        //N_NO3_up_max.clear();
        //N_NH4_up_max.clear();
        return;
    }
    std::vector<double> N_NO3_up_max;
    std::vector<double> N_NH4_up_max;
    N_NO3_up_max.resize(myCase.soilLayers.size());
    N_NH4_up_max.resize(myCase.soilLayers.size());
    for(unsigned int l=0;l<myCase.soilLayers.size();l++)
    {
        N_NO3_up_max[l] = 0;
        N_NH4_up_max[l] = 0;
    }
    N_max_transp = 0;
    for (int l = myCase.crop.roots.firstRootLayer; l< myCase.crop.roots.lastRootLayer; l++)
    {

        if (myCase.crop.layerTranspiration[l] > 0)
        {
            N_NO3_up_max[l] = myCase.carbonNitrogenLayers[l].N_NO3 / myCase.prevWaterContent[l] * myCase.crop.layerTranspiration[l];
            N_NH4_up_max[l] = myCase.carbonNitrogenLayers[l].N_NH4_Sol / myCase.prevWaterContent[l] * myCase.crop.layerTranspiration[l];
        }
        //else
        //{
            //N_NO3_up_max[l] = 0;
            //N_NH4_up_max[l] = 0;
        //}
        // max uptake per species
        N_max_transp += N_NO3_up_max[l] + N_NH4_up_max[l];
    }

    if (N_max_transp > 0)
    {
        for (int l = myCase.crop.roots.firstRootLayer;l<myCase.crop.roots.lastRootLayer;l++)
        {
            myCase.carbonNitrogenLayers[l].N_NO3_uptake = MINVALUE(myCase.carbonNitrogenLayers[l].N_NO3, (N_NO3_up_max[l] / N_max_transp) * N_uptakeMax);
            //GA2017 dialogo con Ceotto (mais San Prospero)
            myCase.carbonNitrogenLayers[l].N_NH4_uptake = 0; //'min(N_NH4_Sol[L], (N_NH4_up_max[L] / N_max_transp) * N_UptakeMax)
        }
    }
    N_NO3_up_max.clear();
    N_NH4_up_max.clear();
}

void Crit3DCarbonNitrogenProfile::N_SurfaceRunoff(Crit1DCase &myCase)
{
    //-----------------------------------------
    //02.11.19.MVS Surface separato da Subsurface
    //-------------- NOTE -----------------------------------------------------
    //sub la stima del N asportato tramite l'acqua di ruscellamento superficiale

    if (myCase.output.dailySurfaceRunoff > 0)
    {
        // calcolo dell'azoto perso nel ruscellamento superficiale
        // seguendo i calcoli tratti da EPIC per il fosforo
        N_NO3_runoff0GG = MINVALUE(myCase.carbonNitrogenLayers[0].N_NO3, myCase.carbonNitrogenLayers[0].N_NO3 / myCase.prevWaterContent[0] * myCase.output.dailySurfaceRunoff);
        N_NH4_runoff0GG = MINVALUE(myCase.carbonNitrogenLayers[0].N_NH4_Sol, myCase.carbonNitrogenLayers[0].N_NH4_Sol / myCase.prevWaterContent[0] * myCase.output.dailySurfaceRunoff);

        myCase.carbonNitrogenLayers[0].N_NO3 -= N_NO3_runoff0GG;
        myCase.carbonNitrogenLayers[0].N_NH4_Sol -= N_NH4_runoff0GG;

    }

}

/*
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
*/

void Crit3DCarbonNitrogenProfile::N_Uptake_Potential(Crit1DCase &myCase)
{
    //2008.09 GA new routine to compute Nitrogen demand
    //2008.04 GA
    //2002.11.26.MVS this computation depends on LAI

    N_dailyDemand = 0;

    // to avoid strong changes as far as LAI>0
    if (myCase.crop.LAIpreviousDay == 0)
    {
        return;
    }

    //only in early growing phase
    if (myCase.crop.degreeDays > (myCase.crop.degreeDaysIncrease + myCase.crop.degreeDaysEmergence))
    {
        return;
    }

    N_dailyDemand = MINVALUE(MAXVALUE(0, myCase.crop.LAI - myCase.crop.LAIpreviousDay) * maxRate_LAI_Ndemand, maxRate_LAI_Ndemand);
    N_potentialDemandCum += N_dailyDemand;
}
/*
void N_Uptake_Max()
{
    //'2008.02 GA revisione (da manuale LEACHM)
    //'2002.11.19.MVS

    int L; //contatore
    //-------------------------------------------------------------------------
    int myDays;
    int i;
    double previousDeficit;

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
*/

void Crit3DCarbonNitrogenProfile::N_Reset()
{
    //'02.11.26.MVS
    //'02.10.22.GD

    //'azzeramento giornaliero
    // credo che venga fatto così semplicemente per riazzerare piuttosto che cambiare dimensione
    /*
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
    */
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
    N_NH4_adsorbedGG = 0;
    N_NH4_adsorbedBeforeGG = 0;
    N_imm_l_NH4GG = 0;
    N_min_humusGG = 0;
    N_min_litterGG = 0;
    N_NH4_volGG = 0;
    N_urea_hydrGG = 0;
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


double Crit3DCarbonNitrogenProfile::findPistonDepth(Crit1DCase &myCase)
{
    unsigned int l;
    for (l = 0;l<myCase.soilLayers.size();l++)
    {
        if (myCase.prevWaterContent[l] > myCase.soilLayers[l].FC)
        {
            if (myCase.soilLayers[l].flux < (myCase.prevWaterContent[l] - myCase.soilLayers[l].FC))
            {
                break;
            }
        }
        else
        {
            break;
        }
    }
    if (l > myCase.soilLayers.size())
    {
        return myCase.mySoil.totalDepth;
    }
    else
    {
        return myCase.soilLayers[l].depth;
    }
}

/*

//calcolo dei flussi di soluti gravitazionali (a 'pistone')
void soluteFluxesPiston(double* mySolute, double PistonDepth,
    double* leached)
{
    int L;
    double myFreeSolute;
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


void soluteFluxesPiston_old(double* mySolute, double* leached, double* CoeffPiston)
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

*/
void Crit3DCarbonNitrogenProfile::soluteFluxes(std::vector<double> &mySolute,bool upwardFlag, double pistonDepth,double* leached,Crit1DCase &myCase)
{
    //2022.12 AV code translation from VB to C/C++
    //2008.10 GA quit the dispersive algorithm because the pseudonumerical procedure is already self dispersive
    //2008.09 GA added dispersive component
    //2008.03 GA FT added diffusive component
    //2007.04 FT GA debugged the iteration pseudo-numerical algorithm
    //04.03.02.FZ first implementation
    //-------------- NOTES -----------------------------------------------------
    //computation of solute fluxes through iterative dilution

    unsigned int iterations;                   //[-] needed iterations for dilution
    std::vector<double> fluxSolute;           //[g m-2] vector used to compute the fluxes computation
    std::vector<double> fSolute;
    std::vector<double> u_temp;
    double H2O_step_flux;
    double H2O_step_flux_L_1;
    //double U_vol;
    int firstLayer=0;
    double myFreeSolute;
    double coeffMobile;

        if (pistonDepth >= myCase.soilLayers[myCase.soilLayers.size()-1].depth)
            return;
        else
        {
            unsigned int l=0;
            while(myCase.soilLayers[l].depth >= pistonDepth)
            {
                l++;
                firstLayer = l;
            }
        }

        fluxSolute.resize(myCase.soilLayers.size());
        fSolute.resize(myCase.soilLayers.size());
        u_temp.resize(myCase.soilLayers.size());
        for (unsigned int l = 0; l<myCase.soilLayers.size(); l++)
        {
            fluxSolute[l] = 0;
            u_temp[l] = myCase.prevWaterContent[l];
        }
        // ???????????????????
        //For L = nrLayers To 1 Step -1

        //Next L

        //fSolute[0] = 0;
        // ??????????????????????????

        //iterazioni = min(max(24, 0.1 * max(Flux(0), Abs(Flux(nrLayers))) * max(Flux(0), Abs(Flux(nrLayers)))), 1000)

        iterations = 1;
        for (unsigned int i = 0; i<iterations; i++)
        {
            for (unsigned int l = firstLayer; l<myCase.soilLayers.size(); l++)
            {
                fSolute[l] = 0;

                H2O_step_flux = (myCase.soilLayers[l].flux / iterations);
                H2O_step_flux_L_1 = (myCase.soilLayers[l-1].flux / iterations);

                // water balance from node l
                u_temp[l] += H2O_step_flux_L_1 - H2O_step_flux;

                // computation of solute fluxes
                if (myCase.soilLayers[l].flux > 0)
                {
                    coeffMobile = 1;
                    myFreeSolute = mySolute[l] * coeffMobile;
                    fSolute[l] = MINVALUE(mySolute[l], myFreeSolute / myCase.prevWaterContent[l] * H2O_step_flux);
                }
                else if (upwardFlag && (myCase.soilLayers[l].flux < 0) && (l < myCase.soilLayers.size()))
                {
                    //myFreeSolute = mySolute[L + 1] * CoeffMobile;
                    myFreeSolute = mySolute[l+1];
                    fSolute[l] = MINVALUE(mySolute[l+1], myFreeSolute / myCase.prevWaterContent[l+1] * H2O_step_flux);
                }

                //azoto in entrata/uscita da nodo L-1
                mySolute[l] += fSolute[l-1] - fSolute[l];

                //flussi convettivi totali
                fluxSolute[l] += fSolute[l];

            }
        }

        // leaching
        // FT GA 2007.12
        *leached += fluxSolute[myCase.soilLayers.size()-1];
        fluxSolute.clear();
        u_temp.clear();
        fSolute.clear();
}

// function developed by V. Marletto for watertable

void Crit3DCarbonNitrogenProfile::leachingWaterTable(std::vector<double> &mySolute, double* leached,Crit1DCase &myCase)
{
    double mySoluteLeachEdge;

    // leaching

    if ((myCase.output.dailyWaterTable != NODATA) && (myCase.output.dailyWaterTable > 0) && (myCase.unit.useWaterTableData == 1))// da chiarire && (flagWaterTableCase == 1))
    {
        for (unsigned int l = 0; l< myCase.soilLayers.size(); l++)
        {
            double maxCapillaryRise = myCase.soilLayers[l].critical - myCase.soilLayers[l].waterContent;
            if (myCase.soilLayers[l].depth > myCase.output.dailyWaterTable)
            {
                *leached += mySolute[l];
                mySolute[l] = 0;
            }            
            else if (myCase.soilLayers[l].depth >= myCase.output.dailyWaterTable - maxCapillaryRise)
            {
                mySoluteLeachEdge = (mySolute[l] / maxCapillaryRise) * (maxCapillaryRise - (myCase.output.dailyWaterTable - myCase.soilLayers[l].depth));
                mySolute[l] += - mySoluteLeachEdge;
                *leached += mySoluteLeachEdge;
            }
        }
    }

}

void Crit3DCarbonNitrogenProfile::NH4_Balance(Crit1DCase &myCase)
{
    double profileNH4PreviousDay;

    profileNH4PreviousDay = profileNH4;
    // ProfiloNH4 = ProfileSum(N_NH4())
    profileNH4 = 0;
    for (unsigned int i=0;i<myCase.soilLayers.size();i++)
    {
        profileNH4 += myCase.carbonNitrogenLayers[0].N_NH4;
    }

    balanceFinalNH4 = profileNH4 - profileNH4PreviousDay - N_NH4_fertGG + N_imm_l_NH4GG;
    balanceFinalNH4 += - N_min_humusGG - N_min_litterGG;
    balanceFinalNH4 += N_NH4_volGG - N_urea_hydrGG + N_nitrifGG;
    balanceFinalNH4 += N_NH4_uptakeGG;
    balanceFinalNH4 += N_NH4_runoff0GG + N_NH4_runoffGG + flux_NH4GG - precN_NH4GG;

    //If BilFinaleNH4 > 0.01 Then Stop
}

void Crit3DCarbonNitrogenProfile::NO3_Balance(Crit1DCase &myCase)
{
    // 02.11.26.MVS translated by Antonio Volta 2022.07.29

    double profileNO3PreviousDay;

    profileNO3PreviousDay = profileNO3;
    //profileNO3 = ProfileSum(N_NO3());
    profileNO3 = 0;
    for (unsigned int i=0;i<myCase.soilLayers.size();i++)
    {
        profileNO3 += myCase.carbonNitrogenLayers[i].N_NO3;
    }
    balanceFinalNO3 = profileNO3 - profileNO3PreviousDay - N_NO3_fertGG + N_imm_l_NO3GG;
    balanceFinalNO3 += N_denitrGG - N_nitrifGG + N_NO3_uptakeGG;
    balanceFinalNO3 += N_NO3_runoff0GG + N_NO3_runoffGG - precN_NO3GG + flux_NO3GG;
}

void Crit3DCarbonNitrogenProfile::N_initializeCrop(bool noReset,Crit1DCase &myCase)
{
    N_cropToHarvest = 0;
    N_cropToResidues = 0;

    if (!noReset)
        N_roots = 0;
    // da leggere da database
    //N_uptakable = tbColture("Nasportabile") / 10;   //      da [kg ha-1] a [g m-2]
    N_uptakable = 20; // TODO da fare lettura chiedere a Fausto

    N_uptakeDeficit = 0;
    N_uptakeMax = 0;
    N_potentialDemandCum = 0;
    // ReDim N_deficit_daily(Nitrogen.N_deficit_max_days) // TODO operazione da capire come gestire

    //Select Case TipoColtura

        if (myCase.crop.type == FRUIT_TREE)
        {
            // 2001 Rufat Dejong Fig. 4 e Tagliavini
            N_ratioHarvested = 0.4;      // fruits, pruning wood
            N_ratioResidues = 0.5;       // leaves
            N_ratioRoots = 0.1;           // roots, trunk, branches


        }
        else if (myCase.crop.type == HERBACEOUS_PERENNIAL || myCase.crop.type ==  GRASS || myCase.crop.type ==  FALLOW)
        {
            N_ratioHarvested = 0.9;
            N_ratioResidues = 0;
            N_ratioRoots = 0.1;
        }
        else
        {
            // annual crops
            N_ratioHarvested = 0.9;
            N_ratioResidues = 0;
            N_ratioRoots = 0.1;
        }

    //in prima approssimazione calcolato da N massimo asportabile per ciclo
    //(parte asportabile e non asportabile) e LAIMAX
    //2013.10 GA
    //scambio mail con Ass.Agr.:
    //
    maxRate_LAI_Ndemand = (N_uptakable - N_roots) / myCase.crop.LAImax ;

}


void Crit3DCarbonNitrogenProfile::N_harvest(Crit1DCase &myCase) // public function
{
        // 2013.06 GA translated in C++ by AV 2022.06
        // annual crops:roots are incorporated in litter at harvest
        // meadow and perennial herbaceous crops: half of N from roots is incorporated in litter
        // tree crops: half of N from roots is incorporated in litter
        // N of leaves is incorporated in litter through the upeer layer with a smoothly rate during the leaf fall

    double N_toLitter = 0;
    // !!! verificare USR PSR
    if (myCase.crop.roots.firstRootLayer == 0 && myCase.crop.roots.lastRootLayer == 0)
        return;

    for (int l = myCase.crop.roots.firstRootLayer; l <= myCase.crop.roots.lastRootLayer; l++) // verificare i cicli for per cambio indici
    {
        //Select Case TipoColtura
        // annual crop
        if (myCase.crop.type == HERBACEOUS_ANNUAL || myCase.crop.type == HORTICULTURAL)
            N_toLitter = myCase.crop.roots.rootDensity[l] * N_roots; // !! prendere il dato da dove?
        // multiannual crop
        else if (myCase.crop.type == HERBACEOUS_PERENNIAL|| myCase.crop.type == GRASS|| myCase.crop.type ==  FALLOW)
            N_toLitter = myCase.crop.roots.rootDensity[l] * N_roots / 2;
        // tree crops
        else if (myCase.crop.type == FRUIT_TREE)
            N_toLitter = myCase.crop.roots.rootDensity[l] * N_roots / 2;

        myCase.carbonNitrogenLayers[l].N_litter += N_toLitter;
        myCase.carbonNitrogenLayers[l].C_litter += carbonNitrogenParameter.CN_RATIO_NOTHARVESTED * N_toLitter;
    }

    if (myCase.crop.type == HERBACEOUS_ANNUAL || myCase.crop.type == HORTICULTURAL)
    {
        // annual crops
        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots = 0;
    }
    else if (myCase.crop.type == HERBACEOUS_PERENNIAL|| myCase.crop.type == GRASS|| myCase.crop.type ==  FALLOW)
    {
        //pluriennali
        N_cropToHarvest = 0;
        N_cropToResidues = 0;
        N_roots *= 0.5;
    }
    else if (myCase.crop.type == FRUIT_TREE)
    {
        //tree crops
            N_cropToHarvest = 0;
            N_roots *= 0.5;
    }
    N_potentialDemandCum = 0;
}



void Crit3DCarbonNitrogenProfile::updateNCrop(Crit3DCrop crop) // this function must be private
{
    // if leguminous
    if (crop.name == "SOYBEAN" || crop.name == "ALFALFA1Y" || crop.name == "ALFALFA")
    {
            // the demand is satisfied by Nitrogen fixation
            // it prevails soil mineral uptake, if available
            N_cropToHarvest += N_dailyDemand * N_ratioHarvested;
            N_cropToResidues += N_dailyDemand * N_ratioResidues;
            N_roots += N_dailyDemand * N_ratioRoots;
    }
    else
    {
            N_cropToHarvest += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioHarvested;
            N_cropToResidues += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioResidues;
            N_roots += (N_NH4_uptakeGG + N_NO3_uptakeGG) * N_ratioRoots;
    }
    // pare che sia commentato chiedere a Gabri
    //'N_UptakeDeficit = max(N_PotentialDemandCumulated - N_Crop, 0)
}

void Crit3DCarbonNitrogenProfile::N_plough(Crit1DCase &myCase) // this function must be public
{
    unsigned int l=0;
    double depthRatio;
    double N_totLitter;
    double N_totHumus = 0;
    double C_totLitter;
    double C_totHumus = 0;
    double N_totNO3 = 0;
    double N_totNH4 = 0;
    unsigned int myLastLayer;

    N_totLitter = N_cropToHarvest + N_cropToResidues + N_roots;
    C_totLitter = N_totLitter * carbonNitrogenParameter.CN_RATIO_NOTHARVESTED;

    do{

        N_totLitter += myCase.carbonNitrogenLayers[l].N_litter;
        C_totLitter += myCase.carbonNitrogenLayers[l].C_litter;
        N_totHumus += myCase.carbonNitrogenLayers[l].N_humus;
        C_totHumus += myCase.carbonNitrogenLayers[l].C_humus;
        N_totNO3 += myCase.carbonNitrogenLayers[l].N_NO3;
        N_totNH4 += myCase.carbonNitrogenLayers[l].N_NH4;
        l++;
    } while (myCase.soilLayers[l].thickness + myCase.soilLayers[l].depth <= myCase.ploughedSoilDepth);

    if (l == 0)
        return;
    else
        myLastLayer = l - 1;

    // tmp = 0;
    for (l=0;l<myLastLayer;l++) // verificare i cicli for per cambio indici
    {
        depthRatio = myCase.soilLayers[l].thickness / (myCase.soilLayers[l].thickness + myCase.soilLayers[l].depth);
        //tmp += depthRatio;
        myCase.carbonNitrogenLayers[l].N_litter = N_totLitter * depthRatio;
        myCase.carbonNitrogenLayers[l].C_litter = C_totLitter * depthRatio;
        myCase.carbonNitrogenLayers[l].N_humus = N_totHumus * depthRatio;
        myCase.carbonNitrogenLayers[l].C_humus = C_totHumus * depthRatio;
        myCase.carbonNitrogenLayers[l].N_NO3 = N_totNO3 * depthRatio;
        myCase.carbonNitrogenLayers[l].N_NH4 = N_totNH4 * depthRatio;
    }
    partitioning(myCase);
    N_cropToHarvest = 0;
    N_cropToResidues = 0;
    N_roots = 0;
}

void Crit3DCarbonNitrogenProfile::NFromCropSenescence(double myDays,double coeffB,Crit1DCase &myCase) // this function must be public
{
    //created in 2013.06 by GA, translated by AV 2022.06
    //new function for describing the release of Nitrogen from pluriannual crop residues
    // e.g. leaf fall
    //myDays  days past since beginning of senescence
    //coeffB  b coefficient in exponential senescence LAI curve

    double LENGTH_SENESCENCE = 20;
    double ratioSenescence;      //ratio of drop leaves
    ratioSenescence = exp(coeffB * myDays) * (1 - exp(-coeffB)) / (exp(coeffB * LENGTH_SENESCENCE) - 1);    
    myCase.carbonNitrogenLayers[0].N_litter += N_cropToResidues * ratioSenescence;

}

