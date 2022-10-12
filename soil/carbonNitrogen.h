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
    float N_NO3;            //[g m-2] Nitrogen in form of Nitrates
    float N_NH4;            //[g m-2] Nitrogen in form of Ammonium
    float N_NH4_Adsorbed;   //[g m-2] Nitrogen in form of adsorbed Ammonium
    float N_NH4_Sol;        //[g m-2] Nitrogen in form of dissolved Ammonium
    float N_urea;           //[g m-2] Nitrogen in form of Urea
    float N_humus;          //[g m-2] Nitrogen in humus
    float N_litter;         //[g m-2] Nitrogen litter

    // ratios
    float ratio_CN_litter; //[-] ratio C/N in litter

    // CARBON
    float C_humus;          //[g m-2] C in humus
    float C_litter;         //[g m-2] C in litter

private:
    // NITROGEN
    float N_NO3_uptake;     //[g m-2] NO3 crop uptake
    float N_NH4_uptake;     //[g m-2] NH4 crop uptake
    float N_min_litter;     //[g m-2] mineralized Nitrogen in litter
    float N_imm_l_NH4;      //[g m-2] NH4 immobilized in litter
    float N_imm_l_NO3;      //[g m-2] NO3 immobilized in litter
    float N_min_humus;      //[g m-2] mineralized Nitrogen in humus
    float N_litter_humus;   //[g m-2] N from litter to humus
    float N_vol;            //[g m-2] volatilized NH4
    float N_denitr;         //[g m-2] denitrified N
    float N_nitrif;         //[g m-2] N from NH4 to NO3
    float N_Urea_Hydr;      //[g m-2] hydrolyzed urea to NH4
    float N_NO3_runoff;     //[g m-2] NO3 lost through surface & subsurface run off
    float N_NH4_runoff;     //[g m-2] NH4 lost through surface & subsurface run off

    // correction factors
    float temperatureCorrectionFactor;   // [] correction factor for soil temperature
    float waterCorrecctionFactor;        // [] correction factor for soil water content
    float waterCorrecctionFactorDenitrification;     // [] correction factor for soil water content (denitrification)

    // CARBON
    float C_litter_humus;   //[g m-2] C for litter to humus
    float C_litter_litter;  //[g m-2] recycled Nitrogen within litter
    float C_min_humus;      //[g m-2] C lost as CO2 by humus mineralization
    float C_min_litter;     //[g m-2] C lost as CO2 by litter mineralization
    float C_denitr_humus;   //[g m-2] C in humus lost as CO2 by means of denitrification
    float C_denitr_litter;  //[g m-2] C in litter lost as CO2 by means of denitrification
};


#endif // CARBON_H










