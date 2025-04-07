/*######################################################################################################################
#
#  RothC C++ version
#
#  This C++ version was translated from the Python code by Caterina Toscano, Antonio Volta and Enrico Balugani 03/03/2025
#
#  The Rothamsted Carbon Model: RothC
#  Developed by David Jenkinson and Kevin Coleman
#
#  INPUTS:
#
#  clay:  clay content of the soil (units: %)
#  depth: depth of topsoil (units: cm)
#  IOM: inert organic matter (t C /ha)
#  nsteps: number of timesteps
#
#  year:    year
#  month:   month (1-12)
#  modern:   %modern
#  TMP:      Air temperature (C)
#  Rain:     Rainfall (mm)
#  Evap:     open pan evaporation (mm)
#  C_inp:    carbon input to the soil each month (units: t C /ha)
#  FYM:      Farmyard manure input to the soil each month (units: t C /ha)
#  PC:       Plant cover (0 = no cover, 1 = covered by a crop)
#  DPM/RPM:  Ratio of DPM to RPM for carbon additions to the soil (units: none)
#
#  OUTPUTS:
#
#  All pools are carbon and not organic matter
#
#  DPM:   Decomposable Plant Material (units: t C /ha)
#  RPM:   Resistant Plant Material    (units: t C /ha)
#  Bio:   Microbial Biomass           (units: t C /ha)
#  Hum:   Humified Organic Matter     (units: t C /ha)
#  IOM:   Inert Organic Matter        (units: t C /ha)
#  SOC:   Soil Organic Matter / Total organic Matter (units: t C / ha)
#
#  DPM_Rage:   radiocarbon age of DPM
#  RPM_Rage:   radiocarbon age of RPM
#  Bio_Rage:   radiocarbon age of Bio
#  HUM_Rage:   radiocarbon age of Hum
#  Total_Rage: radiocarbon age of SOC (/ TOC)
#
#  SWC:       soil moisture deficit (mm per soil depth)
#  RM_TMP:    rate modifying fator for temperature (0.0 - ~5.0)
#  RM_Moist:  rate modifying fator for moisture (0.0 - 1.0)
#  RM_PC:     rate modifying fator for plant retainment (0.6 or 1.0)

######################################################################################################################*/

#ifndef ROTHCPLUSPLUS_H
#define ROTHCPLUSPLUS_H

/*
#ifndef COMMONCONSTANTS_H
#include "commonConstants.h"
#endif
#ifndef CRIT3DDATE_H
#include "crit3dDate.h"
#endif
*/
#ifndef GIS_H
#include "gis.h"
#endif

#define CONR  0.0001244876401867718 // equivalent to std::log(2.0)/5568.0;
#define EXP_DECAY(FACTOR, EXT_COEF,RAGE) ((FACTOR) * std::exp(-(EXT_COEF) * (RAGE)))
#define LOG_RAGE(PLANT_MATERIAL, PLAT_MATERIAL_ACT) (std::log((PLANT_MATERIAL)/(PLAT_MATERIAL_ACT)) ) / CONR

class Crit3D_RothCplusplusMaps
{
private:

public:
    //
    gis::Crit3DRasterGrid* decomposablePlantMaterial; //[tC/ha]
    gis::Crit3DRasterGrid* resistantPlantMaterial; //[tC/ha]
    gis::Crit3DRasterGrid* microbialBiomass; //[tC/ha]
    gis::Crit3DRasterGrid* humifiedOrganicMatter; //[tC/ha]
    gis::Crit3DRasterGrid* inertOrganicMatter; //[tC/ha]
    gis::Crit3DRasterGrid* soilOrganicMatter; //[tC/ha]



    Crit3D_RothCplusplusMaps();
    ~Crit3D_RothCplusplusMaps();

    void initialize(const gis::Crit3DRasterGrid& DEM);
};
class Crit3D_RothCplusplus{
public:

    //Crit3D_RothCplusplus();
    //~Crit3D_RothCplusplus();


};


double RMF_PC(bool PC);
double RMF_Moist(double RAIN, double PEVAP, double clay, double depth, bool PC, double &SWC);
double RMF_Moist(double monthlyBIC, double clay, double depth, bool PC, double &SWC);
double RMF_Tmp(double TEMP);
void decomp(int timeFact, double &DPM, double &RPM, double &BIO, double &HUM, double &IOM,
            double &SOC, double &DPM_Rage, double &RPM_Rage, double &BIO_Rage, double &HUM_Rage,
            double &IOM_Rage, double &Total_Rage, double &modernC, double &RateM, double &clay,
            double &C_Inp, double &FYM_Inp, double &DPM_RPM);
void RothC(int timeFact, double &DPM, double &RPM, double &BIO, double &HUM, double &IOM, double &SOC,
           double &DPM_Rage, double &RPM_Rage, double &BIO_Rage, double &HUM_Rage, double &IOM_Rage,
           double &Total_Rage, double &modernC, double &clay, double &depth, double &TEMP, double &RAIN,
           double &WATERLOSS, bool isET0, bool &PC, double &DPM_RPM, double C_Inp, double FYM_Inp, double &SWC);












#endif // ROTHCPLUSPLUS_H
