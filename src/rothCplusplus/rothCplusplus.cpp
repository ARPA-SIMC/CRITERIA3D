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


//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <string>

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "basicMath.h"
#include "commonConstants.h"
#include "rothCplusplus.h"

using namespace std;

void Crit3DRothCMaps::initialize(const gis::Crit3DRasterGrid& DEM)
{
    clear();

    decomposablePlantMaterial = new gis::Crit3DRasterGrid;
    resistantPlantMaterial  = new gis::Crit3DRasterGrid;
    microbialBiomass = new gis::Crit3DRasterGrid;
    humifiedOrganicMatter = new gis::Crit3DRasterGrid;
    inertOrganicMatter = new gis::Crit3DRasterGrid;
    soilOrganicMatter = new gis::Crit3DRasterGrid;

    clayMap = new gis::Crit3DRasterGrid;

    avgBICMap.resize(12);
    avgTempMap.resize(12);

    for (unsigned int i = 0; i < 12; i++)
        avgBICMap[i] = new gis::Crit3DRasterGrid;

    for (unsigned int i = 0; i < 12; i++)
        avgTempMap[i] = new gis::Crit3DRasterGrid;

    decomposablePlantMaterial->initializeGrid(DEM);
    resistantPlantMaterial->initializeGrid(DEM);
    microbialBiomass->initializeGrid(DEM);
    humifiedOrganicMatter->initializeGrid(DEM);
    inertOrganicMatter->initializeGrid(DEM);
    soilOrganicMatter->initializeGrid(DEM);

    clayMap->initializeGrid(DEM);

    for (unsigned int i = 0; i < 12; i++)
        avgBICMap[i]->initializeGrid(DEM);

    for (unsigned int i = 0; i < 12; i++)
        avgTempMap[i]->initializeGrid(DEM);
}


void Crit3DRothCMaps::clear()
{
    delete decomposablePlantMaterial;
    decomposablePlantMaterial = nullptr;

    delete resistantPlantMaterial;
    resistantPlantMaterial = nullptr;

    delete microbialBiomass;
    microbialBiomass = nullptr;

    delete humifiedOrganicMatter;
    humifiedOrganicMatter = nullptr;

    delete inertOrganicMatter;
    inertOrganicMatter = nullptr;

    delete soilOrganicMatter;
    soilOrganicMatter = nullptr;

    delete clayMap;
    clayMap = nullptr;

    for (auto* map : avgBICMap)
        delete map;

    avgBICMap.clear();

    for (auto* map : avgTempMap)
        delete map;

    avgTempMap.clear();
}


double Crit3DRothCMaps::getAvgBIC(int row, int col, int month) const
{
    int monthIndex = month - 1;

    if (monthIndex >= 0 && monthIndex < int(avgBICMap.size()))
        return avgBICMap[monthIndex]->value[row][col];
    else
        return NODATA;
}

double Crit3DRothCMaps::getAvgTemp(int row, int col, int month) const
{
    int monthIndex = month - 1;

    if (monthIndex >= 0 && monthIndex < int(avgTempMap.size()))
        return avgTempMap[monthIndex]->value[row][col];
    else
        return NODATA;
}

std::vector<double> Crit3DRothCMaps::getAvgTempVector(int row, int col) const
{
    std::vector<double> v(12, NODATA);

    for (unsigned int i = 0; i < 12; i++)
        v[i] = avgTempMap[i]->getValueFromRowCol(row, col);

    return v;
}

std::vector<double> Crit3DRothCMaps::getAvgBICVector(int row, int col) const
{
    std::vector<double> v(12, NODATA);

    for (unsigned int i = 0; i < 12; i++)
        v[i] = avgBICMap[i]->getValueFromRowCol(row, col);

    return v;
}


Crit3DRothC::Crit3DRothC()
{
    initialize();
}

void Crit3DRothC::initialize()
{
    // Inizializzato su python, copiato i pool all'equilibrio
    decomposablePlantMatter = 0.14547;
    resistantPlantMatter = 5.67812;
    microbialBiomass = 0.74059;
    humifiedOrganicMatter = 27.64277;
    inorganicMatter = 3.0041;

    soilOrganicCarbon = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter + inorganicMatter;

    decomposablePMResistantPMRatio = 1.44;  // [-]

    _inputC = 0;                // [t C ha-1]

    _inputFYM = 0.4;            // [kg C day-1 ha-1]
    _inputFYM *= 0.03;          // [t C month-1 ha-1]

    _plantCover = 1.0;          // [-]  TODO fare mappa?

    _clay = NODATA;             // [%]
    _fractionAW = NODATA;       // [-]


    tableYield = {
        {"LARCH", 1.0},
        {"PICEA_ABIES", 4.2},
        {"ABIES_ALBA",2.9},
        {"PINUS_SYLVESTRIS_SCOTCH_PINE",1.6},
        {"PINUS_NIGRA", 2.1},
        {"PINUS_PINEA", 0.9},
        {"CONIFER", 2.1},
        {"BEECH",   2.2},
        {"QUERCUS_PETREA_ROBUR_PUBESCENS", 1.1},
        {"QUERCUS_CERRIS_FRAINETTO_VALLONEA", 1.3},
        {"CASTINEA_SATIVA", 1.5},
        {"CARPINUS_BETULUS_OTRYA_OXYCARPA", 1.4},
        {"HYGROPHILOUS_FOREST", 1.5},
        {"BROADLEAF", 1.5},
        {"QUERCUS_ILEX", 2.7},
        {"QUERCUS_SUBER", 0.6},
        {"MEDITERRANEAN_EVERGREEN_TREE",0.7},
        {"POPULUS_ARTIFICIAL", 1.5},
        {"BROADLEAF_ARTIFICIAL", 1},
        {"CONIFERS_ARTIFICIAL", 2.9},
        {"SHRUB_SUBALPINE", 1},
        {"SHRUB_TEMPERATE", 1},
        {"SHRUB_MEDITERRANEAN", 0.6}
        //These values are expressed as MgC ha-1. It indicates the carbon stock annual increment in aboveground tree biomass
        // we consider that the same quantity is stocked in roots, foliage and shoots
    };
}


void Crit3DRothC::setStateVariables(int row, int col)
{
    decomposablePlantMatter = maps.decomposablePlantMaterial->value[row][col];
    resistantPlantMatter = maps.resistantPlantMaterial->value[row][col];
    microbialBiomass = maps.microbialBiomass->value[row][col];
    humifiedOrganicMatter = maps.humifiedOrganicMatter->value[row][col];
    inorganicMatter =  maps.inertOrganicMatter->value[row][col];
    soilOrganicCarbon = maps.soilOrganicMatter->value[row][col];
}


void Crit3DRothC::storeStateVariables(int row, int col)
{
    maps.setDPMRowCol(decomposablePlantMatter, row, col);
    maps.setRPMRowCol(resistantPlantMatter, row, col);
    maps.setBIORowCol(microbialBiomass, row, col);
    maps.setHUMRowCol(humifiedOrganicMatter, row, col);
    maps.setIOMRowCol(inorganicMatter, row, col);
    maps.setSOCRowCol(soilOrganicCarbon, row, col);
}


bool Crit3DRothC::checkCellPools()
{
    return (isEqual(decomposablePlantMatter, NODATA) ||
            isEqual(resistantPlantMatter, NODATA) ||
            isEqual(microbialBiomass, NODATA) ||
            isEqual(humifiedOrganicMatter, NODATA) ||
            isEqual(inorganicMatter, NODATA) ||
            isEqual(soilOrganicCarbon, NODATA));
}


// plant rate modifying factor (boolean)
double Crit3DRothC::RMF_plantCover(bool plantCover)
{
    if (!plantCover)
        return 1.0;
    else
        return 0.6;
}


// plant rate modifying factor (linear)
double Crit3DRothC::RMF_plantCover(double plantCover)
{
    // with 0 < plantCover < 1
    return -0.4 * plantCover + 1;
}


/*!
 * \brief RMF_Moist_FractionAW
 * \param fractionAW: fraction of available water [0-1]
 * 0.0: wilting point
 * 1.0: field capacity
 *
 * \return soil moisture rate modifying factor [-]
 * 0.2: very dry soil
 * 1.0: no water stress
 */
double Crit3DRothC::RMF_Moist_FractionAW(double fractionAW)
{
    if (isEqual(fractionAW, NODATA))
        return NODATA;

    // TODO gestire casi FAW > 1 (suolo sopra capacità di campo)
    const double cast_FAW = std::max(0.0, std::min(1.0, fractionAW));

    return 0.2 + cast_FAW * 0.8;
}


/*!
 * \brief RMF_Moist_BIC
 * \param sixMonthBIC: 6-month hydroclimatic balance [mm]
 * assumption:
 * 6-month BIC >= 100 mm: no water stress
 * 6-month BIC <= -DEFICIT_MAX: maximum water stress
 *
 * \return soil moisture rate modifying factor [-]
 * 0.2: very dry soil
 * 1.0: no water stress
 */
double Crit3DRothC::RMF_Moist_BIC(double sixMonthBIC)
{
    if (isEqual(sixMonthBIC, NODATA))
        return NODATA;

    const double DEFICIT_MAX = 360.0;                           // [mm]

    const double deficit = std::max(0.0, -sixMonthBIC);        // [mm]

    const double deficitRatio = std::min(1.0, deficit / DEFICIT_MAX);   // [-]

    return 1.0 - 0.8 * deficitRatio;
}


// Calculates the rate modifying factor for temperature (RMF_Tmp)
double Crit3DRothC::RMF_Tmp(double avgT)
{
    if (avgT < -5.0)
        return 0.0;

    return 47.91 / (std::exp(106.06 / (avgT + 18.27)) + 1.0);
}


void Crit3DRothC::decomp(int timeFact, double modifyingRate)
{
    const double oldSOC = decomposablePlantMatter + resistantPlantMatter + microbialBiomass
                          + humifiedOrganicMatter + inorganicMatter;

    const double decomposablePlantMatter_k = 10.0;
    const double resistantPlantMatter_k = 0.3;
    const double microbialBiomass_k = 0.66;
    const double humifiedOrganicMatter_k = 0.02;

    // const double conr = 0.0001244876401867718; // equivalent to std::log(2.0)/5568.0;
    double tstep = 1.0/timeFact; //monthly 1/12 or daily 1/365
    double exc = std::exp(-CONR*tstep);

    // decomposition
    double decomposablePlantMatter1 = decomposablePlantMatter*std::exp(-modifyingRate*decomposablePlantMatter_k*tstep);
    double resistantPlantMatter1 = resistantPlantMatter*std::exp(-modifyingRate*resistantPlantMatter_k*tstep);
    double microbialBiomass1 = microbialBiomass*std::exp(-modifyingRate*microbialBiomass_k*tstep);
    double humifiedOrganicMatter1 = humifiedOrganicMatter*std::exp(-modifyingRate*humifiedOrganicMatter_k*tstep);

    double decomposablePlantMatterDelta = decomposablePlantMatter - decomposablePlantMatter1;
    double resistantPlantMatterDelta = resistantPlantMatter - resistantPlantMatter1;
    double microbialBiomassDelta = microbialBiomass - microbialBiomass1;
    double humifiedOrganicMatterDelta = humifiedOrganicMatter - humifiedOrganicMatter1;

    //calculating redistribution of carbon into each pool
    double x = 1.67*(1.85+1.60*std::exp(-0.0786 * _clay));

    const double ratioCO2 = x / (x + 1.0);
    const double ratioBIO = 0.46 / (x + 1.0);
    const double ratioHUM = 0.54 / (x + 1.0);

    //proportion C from each pool into CO2, microbialBiomass and humifiedOrganicMatter
    double decomposablePlantMatterToCo2 = decomposablePlantMatterDelta * ratioCO2;
    double decomposablePlantMatterToMicrobialBiomass = decomposablePlantMatterDelta * ratioBIO;
    double decomposablePlantMatterToHumifiedOrganicMatter = decomposablePlantMatterDelta * ratioHUM;

    double resistantPlantMatterToCo2 = resistantPlantMatterDelta * ratioCO2;
    double resistantPlantMatterToMicrobialBiomass = resistantPlantMatterDelta * ratioBIO;
    double resistantPlantMatterToHumifiedOrganicMatter = resistantPlantMatterDelta * ratioHUM;

    double microbialBiomassToCo2 = microbialBiomassDelta * ratioCO2;
    double microbialBiomassToMicrobialBiomass = microbialBiomassDelta * ratioBIO;
    double microbialBiomassToHumifiedOrganicMatter = microbialBiomassDelta * ratioHUM;

    double humifiedOrganicMatter_co2 = humifiedOrganicMatterDelta * ratioCO2;
    double humifiedOrganicMatter_microbialBiomass = humifiedOrganicMatterDelta * ratioBIO;
    double humifiedOrganicMatter_humifiedOrganicMatter = humifiedOrganicMatterDelta * ratioHUM;

    //update C pools
    decomposablePlantMatter = decomposablePlantMatter1;
    resistantPlantMatter = resistantPlantMatter1;
    microbialBiomass = microbialBiomass1 + decomposablePlantMatterToMicrobialBiomass + resistantPlantMatterToMicrobialBiomass + microbialBiomassToMicrobialBiomass + humifiedOrganicMatter_microbialBiomass;
    humifiedOrganicMatter = humifiedOrganicMatter1 + decomposablePlantMatterToHumifiedOrganicMatter + resistantPlantMatterToHumifiedOrganicMatter + microbialBiomassToHumifiedOrganicMatter + humifiedOrganicMatter_humifiedOrganicMatter;

    //split plant C to decomposablePlantMatter and resistantPlantMatter
    double PI_C_decomposablePlantMatter = decomposablePMResistantPMRatio / (decomposablePMResistantPMRatio + 1.0) * _inputC;
    double PI_C_resistantPlantMatter = 1.0 / (decomposablePMResistantPMRatio + 1.0) * _inputC;

    //split FYM C to decomposablePlantMatter, resistantPlantMatter and humifiedOrganicMatter
    double FYM_C_decomposablePlantMatter = 0.49 * _inputFYM;
    double FYM_C_resistantPlantMatter = 0.49 * _inputFYM;
    double FYM_C_humifiedOrganicMatter = 0.02 * _inputFYM;

    //add plant C and FYM_C to decomposablePlantMatter, resistantPlantMatter and humifiedOrganicMatter
    decomposablePlantMatter = decomposablePlantMatter + PI_C_decomposablePlantMatter + FYM_C_decomposablePlantMatter;
    resistantPlantMatter = resistantPlantMatter + PI_C_resistantPlantMatter + FYM_C_resistantPlantMatter;
    humifiedOrganicMatter = humifiedOrganicMatter + FYM_C_humifiedOrganicMatter;

    // update soil organic carbon
    soilOrganicCarbon = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter + inorganicMatter;

    if (radioCarbon.isActive)
    {
        // calc new ract of each pool
        double decomposablePlantMatter_Ract = decomposablePlantMatter1 * std::exp(-CONR*radioCarbon.decomposablePlantMatter_age);
        double resistantPlantMatter_Ract = resistantPlantMatter1 * std::exp(-CONR*radioCarbon.resistantPlantMatter_age);

        double microbialBiomass_Ract = microbialBiomass1 * std::exp(-CONR*radioCarbon.microbialBiomass_age);
        double decomposablePlantMatter_microbialBiomass_Ract = decomposablePlantMatterToMicrobialBiomass * std::exp(-CONR*radioCarbon.decomposablePlantMatter_age);
        double resistantPlantMatter_microbialBiomass_Ract = resistantPlantMatterToMicrobialBiomass * std::exp(-CONR*radioCarbon.resistantPlantMatter_age);
        double microbialBiomass_microbialBiomass_Ract = microbialBiomassToMicrobialBiomass * std::exp(-CONR*radioCarbon.microbialBiomass_age);
        double humifiedOrganicMatter_microbialBiomass_Ract = humifiedOrganicMatter_microbialBiomass * std::exp(-CONR*radioCarbon.humifiedOrganicMatter_age);

        double humifiedOrganicMatter_Ract = humifiedOrganicMatter1 *std::exp(-CONR*radioCarbon.humifiedOrganicMatter_age);
        double decomposablePlantMatter_humifiedOrganicMatter_Ract = decomposablePlantMatterToHumifiedOrganicMatter * std::exp(-CONR*radioCarbon.decomposablePlantMatter_age);
        double resistantPlantMatter_humifiedOrganicMatter_Ract = resistantPlantMatterToHumifiedOrganicMatter * std::exp(-CONR*radioCarbon.resistantPlantMatter_age);
        double microbialBiomass_humifiedOrganicMatter_Ract = microbialBiomassToHumifiedOrganicMatter * std::exp(-CONR*radioCarbon.microbialBiomass_age);
        double humifiedOrganicMatter_humifiedOrganicMatter_Ract = humifiedOrganicMatter_humifiedOrganicMatter * std::exp(-CONR*radioCarbon.humifiedOrganicMatter_age);

        // todo check (*exc)?
        double IOM_Ract = inorganicMatter * std::exp(-CONR*radioCarbon.IOM_age);

        // assign new C from plant and FYM the correct age
        double PI_decomposablePlantMatter_Ract = radioCarbon.modernC * PI_C_decomposablePlantMatter;
        double PI_resistantPlantMatter_Ract = radioCarbon.modernC * PI_C_resistantPlantMatter;

        double FYM_decomposablePlantMatter_Ract = radioCarbon.modernC * FYM_C_decomposablePlantMatter;
        double FYM_resistantPlantMatter_Ract = radioCarbon.modernC * FYM_C_resistantPlantMatter;
        double FYM_humifiedOrganicMatter_Ract = radioCarbon.modernC * FYM_C_humifiedOrganicMatter;

        // update ract for each pool
        double decomposablePlantMatter_Ract_new = FYM_decomposablePlantMatter_Ract + PI_decomposablePlantMatter_Ract + decomposablePlantMatter_Ract*exc;
        double resistantPlantMatter_Ract_new = FYM_resistantPlantMatter_Ract + PI_resistantPlantMatter_Ract + resistantPlantMatter_Ract*exc;

        double microbialBiomass_Ract_new = (microbialBiomass_Ract + decomposablePlantMatter_microbialBiomass_Ract + resistantPlantMatter_microbialBiomass_Ract + microbialBiomass_microbialBiomass_Ract + humifiedOrganicMatter_microbialBiomass_Ract )*exc;

        double humifiedOrganicMatter_Ract_new = FYM_humifiedOrganicMatter_Ract + (humifiedOrganicMatter_Ract + decomposablePlantMatter_humifiedOrganicMatter_Ract + resistantPlantMatter_humifiedOrganicMatter_Ract + microbialBiomass_humifiedOrganicMatter_Ract + humifiedOrganicMatter_humifiedOrganicMatter_Ract)*exc;

        double Total_Ract = decomposablePlantMatter_Ract_new + resistantPlantMatter_Ract_new + microbialBiomass_Ract_new + humifiedOrganicMatter_Ract_new + IOM_Ract;

        // calculate rage of each pool
        if (decomposablePlantMatter <= EPSILON)
            radioCarbon.decomposablePlantMatter_age = 0;
        else
            radioCarbon.decomposablePlantMatter_age = (std::log(decomposablePlantMatter/decomposablePlantMatter_Ract_new) ) / CONR;

        if(resistantPlantMatter <= EPSILON)
            radioCarbon.resistantPlantMatter_age = 0;
        else
            radioCarbon.resistantPlantMatter_age = (std::log(resistantPlantMatter/resistantPlantMatter_Ract_new) ) / CONR;

        if(microbialBiomass <= EPSILON)
            radioCarbon.microbialBiomass_age = 0;
        else
            radioCarbon.microbialBiomass_age = (std::log(microbialBiomass/microbialBiomass_Ract_new) ) / CONR;

        if(humifiedOrganicMatter <= EPSILON)
            radioCarbon.humifiedOrganicMatter_age = 0;
        else
            radioCarbon.humifiedOrganicMatter_age = (std::log(humifiedOrganicMatter/humifiedOrganicMatter_Ract_new) ) / CONR;

        if(soilOrganicCarbon <= EPSILON)
            totalRage = 0;
        else
            totalRage = (std::log(soilOrganicCarbon/Total_Ract) ) / CONR;
    }

    // carbon balance
    const double carbonLoss = decomposablePlantMatterToCo2 + resistantPlantMatterToCo2 + microbialBiomassToCo2 + humifiedOrganicMatter_co2;
    const double carbonInput = _inputC + _inputFYM;
    const double expectedSOC = oldSOC + carbonInput - carbonLoss;
    const double balanceError = soilOrganicCarbon - expectedSOC;

    // debug stops if false
    assert(std::abs(balanceError) < 1e-10);
}


// The Rothamsted Carbon Model: RothC
bool Crit3DRothC::RothC()
{
    soilOrganicCarbon = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter + inorganicMatter;

    // Calculate RMFs
    double RM_TMP = RMF_Tmp(_avgTemp);
    const double defaultMoisture = 0.7;
    double RM_Moist = defaultMoisture;

    if (! isEqual(_fractionAW, NODATA))
    {
        // RM_Moist based on fraction of available water (from water balance)
        const double RM_Moist_FAW = RMF_Moist_FractionAW(_fractionAW);

        if (! isEqual(RM_Moist_FAW, NODATA))
            RM_Moist = RM_Moist_FAW;
    }
    else
    {
        // RM_Moist based on BIC (climate)
        const double RM_Moist_BIC = RMF_Moist_BIC(_sixMonthBIC);

        if (! isEqual(RM_Moist_BIC, NODATA))
            RM_Moist = RM_Moist_BIC;
    }

    double RM_PC = RMF_plantCover(_plantCover);

    // Combine RMF's into one.
    double modifyingRate = RM_TMP * RM_Moist * RM_PC;

    int timeFact = 12;
    decomp(timeFact, modifyingRate);

    return true;
}


bool Crit3DRothC::initializeFromClimate(const std::vector<double>& monthlyAvgT,
                                        const std::vector<double>& monthlyBIC)
{
    // check meteo data
    if (monthlyAvgT.size() != 12 || monthlyBIC.size() != 12)
        return false;

    for (int m = 0; m < 12; ++m)
    {
        if (isEqual(monthlyAvgT[m], NODATA) || isEqual(monthlyBIC[m], NODATA))
            return false;
    }

    // compute previous 6-month hydroclimatic balance.
    std::vector<double> sixMonthBIC(12, 0);
    for (int m = 0; m < 12; ++m)
    {
        for (int i = 0; i < 6; i++)
        {
            const int monthIndex = (m - i + 12) % 12;
            sixMonthBIC[m] += monthlyBIC[monthIndex];
        }
    }

    // soils are not considered to be in equilibrium.
    // run model for 75 years using climatological data
    const int INITIALIZATION_YEARS = 75;
    for (int count = 0; count < (INITIALIZATION_YEARS * 12); ++count)
    {
        const int monthIndex = count % 12;

        setAvgTemperature(monthlyAvgT[monthIndex]);
        setSixMonthBIC(sixMonthBIC[monthIndex]);

        RothC();
    }

    return true;
}


/*
// Calculates the rate modifying factor for moisture (RMF_Moist)
double Crit3DRothC::RMF_Moist(double RAIN, double PEVAP, bool PC) {
    const double RMFMax = 1.0;
    const double RMFMin = 0.2;

    //calc soil water functions properties
    double SMDMax = -(20 + 1.3 * clay - 0.01 * (clay * clay));
    double SMDMaxAdj = SMDMax * depth / 23.0;
    double SMD1bar = 0.444 * SMDMaxAdj;
    double SMDBare = 0.556 * SMDMaxAdj;

    double DF = RAIN - 0.75 * PEVAP;

    double minSWCDF = std::min(0.0, SWC + DF);
    double minSMDBareSWC = std::min(SMDBare, SWC);
    if (PC) {
        SWC = std::max(SMDMaxAdj, minSWCDF);
    } else {
        SWC = std::max(minSMDBareSWC, minSWCDF);
    }

    double RM_Moist;
    if (SWC > SMD1bar) {
        RM_Moist = 1.0;
    } else {
        RM_Moist = RMFMin + (RMFMax - RMFMin) * (SMDMaxAdj - SWC) / (SMDMaxAdj - SMD1bar);
    }

    return RM_Moist;
}
*/

/*
double Crit3DRothC::RMF_Moist(double monthlyBIC, bool PC) {
    const double RMFMax = 1.0;
    const double RMFMin = 0.2;

    //calc soil water functions properties
    double SMDMax = -(20 + 1.3 * clay - 0.01 * (clay * clay));
    double SMDMaxAdj = SMDMax * depth / 23.0;
    double SMD1bar = 0.444 * SMDMaxAdj;
    double SMDBare = 0.556 * SMDMaxAdj;

    double DF = monthlyBIC;

    double minSWCDF = std::min(0.0, SWC + DF);
    double minSMDBareSWC = std::min(SMDBare, SWC);
    if (PC) {
        SWC = std::max(SMDMaxAdj, minSWCDF);
    } else {
        SWC = std::max(minSMDBareSWC, minSWCDF);
    }
    double RM_Moist;
    if (SWC > SMD1bar) {
        RM_Moist = 1.0;
    } else {
        RM_Moist = RMFMin + (RMFMax - RMFMin) * (SMDMaxAdj - SWC) / (SMDMaxAdj - SMD1bar);
    }
    return RM_Moist;
}
*/

/*
std::vector<std::vector<double>> leggi_csv(const std::string& nome_file) {
    std::vector<std::vector<double>> dati;
    std::ifstream file(nome_file);

    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file: " << nome_file << std::endl;
        return dati;
    }

    std::string linea;
    std::getline(file, linea);

    while (std::getline(file, linea)) {
        std::vector<double> riga;
        std::istringstream ss(linea);
        std::string token;

        for (int i = 0; i < 10; ++i) {
            std::getline(ss, token, ',');
            try {
                riga.push_back(std::stod(token));
            } catch (const std::invalid_argument&) {
                std::cerr << "Errore nella conversione di un valore in double: " << token << std::endl;
            }
        }
        if (!riga.empty())
            dati.push_back(riga);
    }
    file.close();
    return dati;
}


void Crit3DRothC::scrivi_csv(const std::string& nome_file, const std::vector<std::vector<double>>& dati) {
    std::ofstream file(nome_file);

    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file: " << nome_file << std::endl;
        return;
    }

    //file << "index,Year,Month,DPM_t_C_ha,RPM_t_C_ha,BIO_t_C_ha,HUM_t_C_ha,IOM_t_C_ha,SOC_t_C_ha,deltaC" << std::endl;

    for (const auto& riga : dati) {
        std::stringstream ss;
        for (double valore : riga) {
            ss << valore << ",";
        }
        ss.seekp(-1, std::ios::end);
        ss << std::endl;
        file << ss.str();
    }

    file.close();
}
*/


/*
void Crit3DRothC::computeRothCPoint()
{
    // set initial soil water content (deficit)

    soilOrganicCarbon = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter + inorganicMatter;

    std::cout << "," << decomposablePlantMatter << ","<< resistantPlantMatter << ","<< microbialBiomass << ","
              << humifiedOrganicMatter << ","<< inorganicMatter << ","<< soilOrganicCarbon << "\n";

    double modernC = 100;

    if (radioCarbon.isActive)
       double totalDelta = (std::exp(-totalRage/8035.0) - 1) * 1000;

    inputFYM = 0.4;     // kg C day-1 ha-1
    inputFYM *= 0.03;   // t C month-1 ha-1

    RothC_main(rothCplantCover);

    if (radioCarbon.isActive)
        double totalDelta = (std::exp(-totalRage/8035.0) - 1.0) * 1000;

    std::cout << "," << decomposablePlantMatter << ","<< resistantPlantMatter << ","<< microbialBiomass << ","
              << humifiedOrganicMatter << ","<< inorganicMatter << ","<< soilOrganicCarbon << "\n";

    //todo: remove
    if (false)
    {
        std::ofstream myFile;
        myFile.open("RothC.csv", std::ios_base::app);
        myFile << decomposablePlantMatter<< ","<< resistantPlantMatter<<","<< microbialBiomass<<","<< humifiedOrganicMatter<<","<<
            inorganicMatter<<","<< soilOrganicCarbon << "\n";
        myFile.close();
    }
}*/


/*
int Crit3DRothC::old_main()
{
    //set initial pool values
    double DPM_Rage = 0.0;
    double RPM_Rage = 0.0;
    double BIO_Rage = 0.0;
    double HUM_Rage = 0.0;
    double IOM_Rage = 50000.0;

    //set initial soil water content (deficit)
    double SWC = 0;
    double TOC1 = 0;

    //TODO: read in RothC input data file
    //double clay = 13.0;     //[%]
    //double depth = 25.0;    //[cm]
    //double IOM = 3.0041;    //[t C/ha]
    //int nsteps = 840;       //[-]

    //std::vector<std::vector<double>> data = createDataMatrix();
    std::vector<std::vector<double>> data = leggi_csv("C:/Github/rothCStandAlone/data_input.csv");

    int k = -1;
    int j = -1;

    //SOC = DPM + RPM + BIO + HUM + IOM;
    soilOrganicCarbon = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter + inorganicMatter;

    std::cout << j << "," << decomposablePlantMatter << ","<< resistantPlantMatter << ","<< microbialBiomass << ","
              << humifiedOrganicMatter << ","<< inorganicMatter << ","<< soilOrganicCarbon << "\n";

    int timeFact = 12;

    double test = 100;
    while (test > 0.000001)
    {
        k = k+1;
        j = j+1;

        if (k == timeFact) k = 0;

        meteoVariable.setTemperature(data[k][3]);
        meteoVariable.setPrecipitation(data[k][4]);
        meteoVariable.setWaterLoss(data[k][5]);
        //PC = bool(data[k][8]);
        //DPM_RPM = data[k][9];
        inputC = data[k][6];
        inputFYM = data[k][7];
        //modernC = data[k][2]/100;

        totalRage = 0;

        RothC_main(timeFact, rothCplantCover);

        if (((k+1)%timeFact) == 0)
        {
            double TOC0 = TOC1;
            TOC1 = decomposablePlantMatter + resistantPlantMatter + microbialBiomass + humifiedOrganicMatter;
            test = fabs(TOC1-TOC0);
        }
    }

    double totalDelta = (std::exp(-totalRage/8035.0) - 1) * 1000;

    std::cout << j << "," << decomposablePlantMatter << ","<< resistantPlantMatter << ","<< microbialBiomass << ","
              << humifiedOrganicMatter << ","<< inorganicMatter << ","<< soilOrganicCarbon << "\n";

    std::vector<std::vector<double>> yearList;
//    std::vector<std::vector<double>> yearList = {{double(1), double(j+1), DPM, RPM, BIO, HUM, IOM, SOC, totalDelta}};


    std::vector<std::vector<double>> monthList;
    int timeFactIndex;

    for (int i = timeFact; i < data.size(); i++)
    {
        meteoVariable.setTemperature(data[i][3]);
        meteoVariable.setPrecipitation(data[i][4]);
        meteoVariable.setWaterLoss(data[i][5]);
        //PC = bool(data[i][8]);
        //DPM_RPM = data[i][9];
        inputC = data[i][6];
        inputFYM = data[i][7];
        //modernC = data[i][2]/100;

        RothC_main(timeFact, rothCplantCover);

        totalDelta = (std::exp(-totalRage/8035.0) - 1.0) * 1000;

        //std::cout << C_Inp << "," << FYM_Inp << "," << TEMP << "," << RAIN << "," << PEVAP << "," << SWC << ","
                  //<< PC << "," << DPM <<"," << RPM <<"," << BIO <<"," << HUM <<"," << IOM <<"," << SOC << "\n";

        monthList.push_back({double(i-timeFact), double(data[i][0]), double(data[i][1]), decomposablePlantMatter,
                             resistantPlantMatter, microbialBiomass, humifiedOrganicMatter, inorganicMatter, soilOrganicCarbon, totalDelta});

        if (int(data[i][1]) == timeFact)
        {
            timeFactIndex = int(i/timeFact);
            yearList.push_back({double(timeFactIndex), data[i][0], data[i][1], decomposablePlantMatter,
                                resistantPlantMatter, microbialBiomass, humifiedOrganicMatter, inorganicMatter, soilOrganicCarbon, totalDelta});
            //std::cout << i << "," << DPM << "," << RPM << "," << BIO << "," << HUM << "," << IOM << "," << SOC << "," << totalDelta << "\n";
        }
    }

    scrivi_csv("C:/Github/rothCStandAlone/CMonthResults.csv", monthList);
    scrivi_csv("C:/Github/rothCStandAlone/CYearResults.csv", yearList);

    return 0;
}*/
