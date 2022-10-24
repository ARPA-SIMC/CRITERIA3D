#ifndef CARBON_H
#define CARBON_H

// Translation of the whole chemical reactions algorithms from LEACHM (2013)
// reviewed by Gabriele Antolini (2007-2009)
// This module computes the Nitrogen cycle
// The Nitrogen cycle is substituted through the LEACHN model (Hutson & Wagenet 1992)
// and some improvement added by Ducco

#include "crit3dDate.h"
#include "soil.h"

// in water NO3 diffusion coefficient
// [m2 d-1] (Lide, 1997: 1.9 x 10-5 cm2 s-1)
#define KDW_NO3 0.00016416

#define FERTILIZER_UREA  9

class Crit3DCarbonNitrogenLayer
{
public:

    Crit3DCarbonNitrogenLayer();

    // NITROGEN
    double N_NO3;            //[g m-2] Nitrogen in form of Nitrates
    double N_NH4;            //[g m-2] Nitrogen in form of Ammonium
    double N_NH4_Adsorbed;   //[g m-2] Nitrogen in form of adsorbed Ammonium
    double N_NH4_Sol;        //[g m-2] Nitrogen in form of dissolved Ammonium
    double N_urea;           //[g m-2] Nitrogen in form of Urea
    double N_humus;          //[g m-2] Nitrogen in humus
    double N_litter;         //[g m-2] Nitrogen litter
    double N_NO3_uptake;     //[g m-2] NO3 crop uptake
    double N_NH4_uptake;     //[g m-2] NH4 crop uptake

    // CARBON
    double C_humus;          //[g m-2] C in humus
    double C_litter;         //[g m-2] C in litter

    // ratios
    double ratio_CN_litter; //[-] ratio C/N in litter

    // correction factors
    double temperatureCorrectionFactor;   // [] correction factor for soil temperature
    double waterCorrecctionFactor;        // [] correction factor for soil water content
    double waterCorrecctionFactorDenitrification;     // [] correction factor for soil water content (denitrification)

private:
    // NITROGEN

    double N_min_litter;     //[g m-2] mineralized Nitrogen in litter
    double N_imm_l_NH4;      //[g m-2] NH4 immobilized in litter
    double N_imm_l_NO3;      //[g m-2] NO3 immobilized in litter
    double N_min_humus;      //[g m-2] mineralized Nitrogen in humus
    double N_litter_humus;   //[g m-2] N from litter to humus
    double N_vol;            //[g m-2] volatilized NH4
    double N_denitr;         //[g m-2] denitrified N
    float N_nitrif;         //[g m-2] N from NH4 to NO3
    double N_Urea_Hydr;      //[g m-2] hydrolyzed urea to NH4
    double N_NO3_runoff;     //[g m-2] NO3 lost through surface & subsurface run off
    double N_NH4_runoff;     //[g m-2] NH4 lost through surface & subsurface run off

    // CARBON
    double C_litter_humus;   //[g m-2] C for litter to humus
    double C_litter_litter;  //[g m-2] recycled Nitrogen within litter
    double C_min_humus;      //[g m-2] C lost as CO2 by humus mineralization
    double C_min_litter;     //[g m-2] C lost as CO2 by litter mineralization
    double C_denitr_humus;   //[g m-2] C in humus lost as CO2 by means of denitrification
    double C_denitr_litter;  //[g m-2] C in litter lost as CO2 by means of denitrification



};


#endif // CARBON_H










