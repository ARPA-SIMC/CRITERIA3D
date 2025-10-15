/*!
    \name hydrall.cpp
    \brief
    \authors Antonio Volta, Caterina Toscano
    \email avolta@arpae.it ctoscano@arpae.it

*/


//#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include "crit3dDate.h"
#include "commonConstants.h"
#include "hydrall.h"
#include "furtherMathFunctions.h"
#include "basicMath.h"
#include "physics.h"
#include "statistics.h"


Crit3DHydrallLightExtinctionCoefficient::Crit3DHydrallLightExtinctionCoefficient()
{
    global = NODATA;
    par = NODATA;
    nir = NODATA;
}

Crit3DHydrallState::Crit3DHydrallState()
{
    standBiomass = NODATA;
    rootBiomass = NODATA;
}

Crit3DHydrallStatePlant::Crit3DHydrallStatePlant()
{
    treeNetPrimaryProduction = 0;
    treeBiomassFoliage = 0.1; //[kgDM m-2]
    treeBiomassRoot = 0.05; //[kgDM m-2]
    treeBiomassSapwood = 0.2; //[kgDM m-2]
    understoreyNetPrimaryProduction = 0;
    understoreyBiomassFoliage = 0.08; //[kgDM m-2]
    understoreyBiomassRoot = 0.02; //[kgDM m-2]
}

Crit3DHydrallWeatherDerivedVariable::Crit3DHydrallWeatherDerivedVariable()
{
    airVapourPressure = NODATA;
    emissivitySky = NODATA;
    longWaveIrradiance = NODATA;
    slopeSatVapPressureVSTemp = NODATA;
    myDirectIrradiance = NODATA;
    myDiffuseIrradiance = NODATA;
    myEmissivitySky = NODATA;
    myLongWaveIrradiance = NODATA;
    psychrometricConstant = NODATA;
    et0 = NODATA;
}

Crit3DHydrallWeatherVariable::Crit3DHydrallWeatherVariable()
{
    myInstantTemp = NODATA;
    prec = NODATA;
    irradiance = NODATA;
    relativeHumidity = NODATA;
    windSpeed = NODATA;
    atmosphericPressure = NODATA;
    //meanDailyTemperature;
    vaporPressureDeficit = NODATA;
    last30DaysTAvg = NODATA;
    meanDailyTemp = NODATA;

    yearlyET0 = NODATA;
    yearlyPrec = NODATA;
}

Crit3DHydrallEnvironmentalVariable::Crit3DHydrallEnvironmentalVariable()
{
    CO2 = NODATA;
    sineSolarElevation = NODATA;
}

Crit3DHydrallPlant::Crit3DHydrallPlant()
{
    myChlorophyllContent = NODATA;
    height = NODATA; // in cm
    myLeafWidth = NODATA;
    isAmphystomatic = false;
    foliageLongevity = 1; //[yr]
    sapwoodLongevity = 35; //[yr]
    fineRootLongevity = 1; //[yr]
    foliageDensity = 0.1; //[kgDM m-3]
    woodDensity = RHOS;
    specificLeafArea = NODATA; //[m2 kg-1]
    psiLeaf = NODATA;
    psiSoilCritical = NODATA; //could be removed
    psiLeafMinimum = NODATA; //[MPa]
    transpirationPerUnitFoliageAreaCritical = NODATA;
    leafAreaIndexCanopy = NODATA;
    leafAreaIndexCanopyMax = NODATA;
    standVolume = NODATA; // maps referred to stand volume MUST be initialized
    currentIncrementalVolume = EPSILON;
    transpirationCritical = NODATA; //(mol m-2 s-1)
    rootShootRatioRef = 0.33;
    mBallBerry = NODATA;
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

    tableEcophysiologicalParameters = {
        {"LARCH",                            35.0, 6.0, false, 0.29, 0.8},
        {"PICEA_ABIES",                       35.0, 6.0, false, 0.29, 0.8},
        {"ABIES_ALBA",                      30.0, 6.0, false, 0.28, 0.8},
        {"PINUS_SYLVESTRIS_SCOTCH_PINE",          30.0, 6.0, false, 0.29, 0.8},
        {"PINUS_NIGRA",                         30.0, 6.0, false, 0.29, 0.8},
        {"PINUS_PINEA",                 40.0, 7.0, true, 0.29, 0.8},
        {"CONIFER",      30.0, 6.0, false, 0.29, 0.8},
        {"BEECH",                                     50.0, 8.0, false, 0.2, 0.4},
        {"QUERCUS_PETREA_ROBUR_PUBESCENS",       50.0, 8.0, false, 0.2, 0.4},
        {"QUERCUS_CERRIS_FRAINETTO_VALLONEA", 50.0, 8.0, false, 0.2, 0.4},
        {"CASTINEA_SATIVA",                                  50.0, 8.0, false, 0.28, 0.4},
        {"CARPINUS_BETULUS_OTRYA_OXYCARPA",                         50.0, 8.0, false, 0.26, 0.4},
        {"HYGROPHILOUS_FOREST",                             60.0, 9.0, false, 0.22, 0.4},
        {"BROADLEAF",                    50.0, 8.0, false, 0.22, 0.4},
        {"QUERCUS_ILEX",                                     40.0, 7.0, true, 0.2, 0.4},
        {"QUERCUS_SUBER",                                   40.0, 7.0, true, 0.2, 0.4},
        {"MEDITERRANEAN_EVERGREEN_TREE",      40.0, 7.0, true, 0.22, 1},
        {"POPULUS_ARTIFICIAL",                        70.0, 9.0, false, 0.21, 0.4},
        {"BROADLEAF_ARTIFICIAL",             60.0, 8.0, false, 0.24, 0.4},
        {"CONIFERS_ARTIFICIAL",                     40.0, 6.0, false, 0.29, 0.4},
        {"SHRUB_SUBALPINE",                         40.0, 7.0, false, 0.33, 1},
        {"SHRUB_TEMPERATE",                40.0, 7.0, false, 0.33, 1},
        {"SHRUB_MEDITERRANEAN",             40.0, 8.0, true, 0.33, 1} //TODO: check some of these values
    };


    rangeLAI = {
        {"LARCH", 0.1, 4.0},
        {"PICEA_ABIES", 1.5, 6.0},
        {"ABIES_ALBA", 1.5, 6.0},
        {"PINUS_SYLVESTRIS_SCOTCH_PINE", 1.0, 4.0},
        {"PINUS_NIGRA", 1.0, 4.0},
        {"PINUS_PINEA", 1.0, 4.0},
        {"CONIFER_FOREST_OTHERS", 1.0, 5.0},
        {"BEECH", 0.1, 6.0},
        {"QUERCUS_PETREA_ROBUR_PUBESCENS", 0.1, 5.0},
        {"QUERCUS_CERRIS_FRAINETTO_VALLONEA", 0.1, 5},
        {"CASTINEA_SATIVA", 0.1, 5.0},
        {"CARPINUS_BETULUS_OTRYA_OXYCARPA", 0.1, 5.0},
        {"HYGROPHILOUS_FOREST", 0.1, 6.0},
        {"BROADLEAF_FOREST_OTHERS", 0.1, 5.0},
        {"QUERCUS_ILEX", 1.5, 4.0},
        {"QUERCUS_SUBER", 1.5, 4.0},
        {"MEDITERRANEAN_EVERGREEN_TREE", 1.5, 4.0},
        {"POPULUS_ARTIFICIAL", 0.1, 6.0},
        {"BROADLEAF_ARTIFICIAL", 0.1, 6.0},
        {"CONIFERS_ARTIFICIAL", 1.0, 5.0},
        {"SHRUB_SUBALPINE", 0.1, 2.0},
        {"SHRUB_TEMPERATE", 0.1, 3.0},
        {"SHRUB_MEDITERRANEAN", 1.0, 3.0}
    }; // da aggiungere il sottobosco

    phenologyLAI = {
        {"LARCH", 250, 1000, 2250},              // Intermedia
        {"PICEA_ABIES", 300, 1100, 2500},         // Tardiva
        {"ABIES_ALBA", 300, 1100, 2500},        // Tardiva
        {"PINUS_SYLVESTRIS_SCOTCH_PINE", 250, 1000, 2250}, // Intermedia
        {"PINUS_NIGRA", 250, 1000, 2250},           // Intermedia
        {"PINUS_PINEA", 200, 900, 2000},    // Precoce
        {"CONIFER_FOREST_OTHERS", 250, 1000, 2250}, // Intermedia
        {"BEECH", 300, 1100, 2500},                       // Tardiva
        {"QUERCUS_PETREA_ROBUR_PUBESCENS", 250, 1000, 2250}, // Intermedia
        {"QUERCUS_CERRIS_FRAINETTO_VALLONEAa", 300, 1100, 2500}, // Tardiva
        {"CASTINEA_SATIVA", 250, 1000, 2250},                    // Intermedia
        {"CARPINUS_BETULUS_OTRYA_OXYCARPA", 250, 1000, 2250},           // Intermedia
        {"HYGROPHILOUS_FOREST", 250, 1000, 2250},               // Intermedia
        {"BROADLEAF_FOREST_OTHERS", 250, 1000, 2250},      // Intermedia
        {"QUERCUS_ILEX", 200, 900, 2000},                        // Precoce
        {"QUERCUS_SUBER", 200, 900, 2000},                      // Precoce
        {"MEDITERRANEAN_EVERGREEN_TREE", 200, 900, 2000}, // Precoce
        {"POPULUS_ARTIFICIAL", 200, 900, 2000},           // Precoce
        {"BROADLEAF_ARTIFICIAL", 250, 1000, 2250}, // Intermedia
        {"CONIFERS_ARTIFICIAL", 250, 1000, 2250},       // Intermedia
        {"SHRUB_SUBALPINE", 250, 1000, 2250},           // Intermedia
        {"SHRUB_TEMPERATE", 250, 1000, 2250},  // Intermedia
        {"SHRUB_MEDITERRANEAN", 200, 900, 2000} // Precoce
    };


}

Crit3DHydrallSoil::Crit3DHydrallSoil()
{
    layersNr = NODATA;
    totalDepth = NODATA;
    temperature = NODATA;
    rootDensity.clear();
    stressCoefficient.clear();
    waterContent.clear();
    wiltingPoint.clear();
    fieldCapacity.clear();
    saturation.clear();
    hydraulicConductivity.clear();
    satHydraulicConductivity.clear();
    nodeThickness.clear();
    clay.clear();
    sand.clear();
    silt.clear();
    bulkDensity.clear();
    waterPotential.clear();
}

Crit3DHydrallBigLeaf::Crit3DHydrallBigLeaf()
{
    absorbedPAR = NODATA;
    isothermalNetRadiation = NODATA;
    leafAreaIndex = NODATA;
    totalConductanceHeatExchange = NODATA;
    aerodynamicConductanceHeatExchange = NODATA;
    aerodynamicConductanceCO2Exchange = NODATA;
    leafTemperature = NODATA;
    darkRespiration = NODATA;
    minimalStomatalConductance = NODATA;
    maximalCarboxylationRate = NODATA;
    maximalElectronTrasportRate = NODATA;
    carbonMichaelisMentenConstant = NODATA;
    oxygenMichaelisMentenConstant = NODATA;
    compensationPoint = NODATA;
    convexityFactorNonRectangularHyperbola = NODATA;
    quantumYieldPS2 = NODATA;
    assimilation = NODATA;
    transpiration = NODATA;
    stomatalConductance = NODATA;
}

Crit3DHydrallParameterWangLeuning::Crit3DHydrallParameterWangLeuning()
{
    optimalTemperatureForPhotosynthesis = 298.15; // K
    stomatalConductanceMin = 0.01; // [Pa Pa-1]
    sensitivityToVapourPressureDeficit = 1300;
    alpha = 340000; //1100000; // this parameter must be multiplied by 10^-6 in order to be compliant with literature
    psiLeaf = 1800;                 // kPa
    waterStressThreshold = NODATA;
    maxCarboxRate = 150;           // Vcmo at optimal temperature (25°C) umol m-2 s-1
}

Crit3DHydrallDeltaTimeOutputs::Crit3DHydrallDeltaTimeOutputs()
{
    netAssimilation = NODATA;
    grossAssimilation = NODATA;
    transpiration = NODATA;
    interceptedWater = NODATA;
    netDryMatter = NODATA;
    absorbedPAR = NODATA;
    respiration = NODATA;
    transpirationGrass = NODATA;
    transpirationNoStress = NODATA;
    evaporation = NODATA;
    evapoTranspiration = NODATA;
    understoreyNetAssimilation = NODATA;
}

Crit3DHydrallNitrogen::Crit3DHydrallNitrogen()
{
    interceptLeaf = NODATA;
    slopeLeaf = NODATA;
    leaf = 0.024;  //[kg kgDM-1]
    stem = 0.0078; //[kg kgDM-1]
    root = 0.0021; //[kg kgDM-1]
}

Crit3DHydrallBiomass::Crit3DHydrallBiomass()
{
    leaf = 0.1; //[kgDM m-2]
    sapwood = 0.2; //[kgDM m-2]
    fineRoot = 0.05; //[kgDM m-2]
    total = leaf + sapwood + fineRoot; //[kgDM m-2]
}

Crit3DHydrallAllocationCoefficient::Crit3DHydrallAllocationCoefficient()
{
    toFoliage = NODATA;
    toFineRoots = NODATA;
    toSapwood = NODATA;
}

Crit3DHydrallMaps::Crit3DHydrallMaps()
{
    treeNetPrimaryProduction = new gis::Crit3DRasterGrid; //SAVE
    treeBiomassFoliage = new gis::Crit3DRasterGrid; //SAVE
    treeBiomassRoot = new gis::Crit3DRasterGrid; //SAVE
    treeBiomassSapwood = new gis::Crit3DRasterGrid; //SAVE
    understoreyNetPrimaryProduction = new gis::Crit3DRasterGrid; //SAVE
    understoreyBiomassFoliage = new gis::Crit3DRasterGrid; //SAVE
    understoreyBiomassRoot = new gis::Crit3DRasterGrid; //SAVE

    outputC = new gis::Crit3DRasterGrid;

    criticalSoilWaterPotential = new gis::Crit3DRasterGrid;
    criticalTranspiration = new gis::Crit3DRasterGrid;
    minLeafWaterPotential = new gis::Crit3DRasterGrid;

    yearlyET0 = new gis::Crit3DRasterGrid;
    yearlyPrec = new gis::Crit3DRasterGrid;
}


void Crit3DHydrallMaps::initialize(const gis::Crit3DRasterGrid& DEM)
{
    treeSpeciesMap.initializeGrid(DEM);
    plantHeight.initializeGrid(DEM); //TODO
    criticalSoilWaterPotential->initializeGrid(DEM);
    criticalTranspiration->initializeGrid(DEM);
    minLeafWaterPotential->initializeGrid(DEM);

    treeNetPrimaryProduction->initializeGrid(DEM); //TODO: initial maps must be loaded
    treeBiomassFoliage->initializeGrid(DEM); //SAVE
    treeBiomassRoot->initializeGrid(DEM); //SAVE
    treeBiomassSapwood->initializeGrid(DEM); //SAVE
    understoreyNetPrimaryProduction->initializeGrid(DEM); //SAVE
    understoreyBiomassFoliage->initializeGrid(DEM); //SAVE
    understoreyBiomassRoot->initializeGrid(DEM);

    outputC->initializeGrid(DEM);
    yearlyPrec->initializeGrid(DEM);
    yearlyET0->initializeGrid(DEM);

    for (int i = 0; i < DEM.header->nrRows; i++)
    {
        for (int j = 0; j < DEM.header->nrCols; j++)
        {
            if (! isEqual(DEM.value[i][j], DEM.header->flag))
            {
                //TODO: change when initial biomass maps will be available
                treeNetPrimaryProduction->value[i][j] = 0;
                treeBiomassFoliage->value[i][j] = 0.1f;
                treeBiomassRoot->value[i][j] = 0.05f;
                treeBiomassSapwood->value[i][j] = 0.2f;
                understoreyNetPrimaryProduction->value[i][j] = 0;
                understoreyBiomassFoliage->value[i][j] = 0;
                understoreyBiomassRoot->value[i][j] = 0;
                outputC->value[i][j] = 0;
                yearlyPrec->value[i][j] = 0;
                yearlyET0->value[i][j] = 0;

            }
        }
    }
}

Crit3DHydrallMaps::~Crit3DHydrallMaps()
{

}

bool Crit3DHydrall::computeHydrallPoint()
{

    //plant.leafAreaIndexCanopyMax = statePlant.treecumulatedBiomassFoliage *  plant.specificLeafArea / cover;
    //plant.getLAICanopy() = MAXVALUE(4,plant.leafAreaIndexCanopyMax * computeLAI(myDate));
    //understoreyLeafAreaIndexMax = statePlant.understoreycumulatedBiomassFoliage * plant.specificLeafArea;
    //understorey.leafAreaIndex = MAXVALUE(LAIMIN,understoreyLeafAreaIndexMax* computeLAI(myDate));

    plant.setLAICanopy(plant.getLAICanopy() - plant.getLAICanopyMin());
    plant.setLAICanopy(MAXVALUE(0, plant.getLAICanopy()));
    understorey.leafAreaIndex = 1;
    //plant.setLAICanopy(5); //DEBUG
    //plant.setLAICanopyMax(6); //DEBUG
    plant.specificLeafArea = plant.getLAICanopyMax() / statePlant.treeBiomassFoliage;

    Crit3DHydrall::photosynthesisAndTranspiration();

    /* necessaria per ogni specie:
     *  il contenuto di clorofilla (g cm-2) il default è 500
     *  lo spessore della foglia 0.2 cm default
     *  un booleano che indichi se la specie è anfistomatica oppure no
     *  parametro alpha del modello di Leuning
     *
    */
    // la temperatura del mese precedente arriva da fuori

    return true;
}


double Crit3DHydrall::getCO2(Crit3DDate myDate)
{
    double atmCO2 = 400 ; //https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5/download.table
    double year[24] = {1750,1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100};
    double valueCO2[24] = {278,283,285,296,300,303,307,310,311,317,325,339,354,369,389,413,443,473,503,530,550,565,570,575};

    atmCO2 = interpolation::linearInterpolation(double(myDate.year), year, valueCO2, 24);


    atmCO2 += 3*cos(2*PI*getDoyFromDate(myDate)/365.0);		     // to consider the seasonal effects
    return atmCO2 * weatherVariable.atmosphericPressure/1000000;   // [Pa] in +- ppm/10 formula changed from the original Hydrall
}
/*
double Crit3DHydrall::getPressureFromElevation(double myTemperature, double myElevation)
{
    return P0 * exp((- GRAVITY * M_AIR * myElevation) / (R_GAS * myTemperature));
}
*/
double Crit3DHydrall::computeLAI(Crit3DDate myDate)
{
    // TODO

    if (getDoyFromDate(myDate) < 100)
        return LAIMIN;
    else if (getDoyFromDate(myDate) > 300)
        return LAIMIN;
    else if (getDoyFromDate(myDate) >= 100 && getDoyFromDate(myDate) <= 200)
        return LAIMIN+(LAIMAX-LAIMIN)/100*(getDoyFromDate(myDate)-100);
    else
        return LAIMAX;
}


void Crit3DHydrall::nullPhotosynthesis()
{
    treeAssimilationRate = 0 ;
    if (soil.layersNr != NODATA)
    {
        treeTranspirationRate.resize(soil.layersNr);
        for (int i=0; i < soil.layersNr; i++)
            treeTranspirationRate[i] = 0;
    }
}


double Crit3DHydrall::photosynthesisAndTranspiration()
{
    //if LAI is too small or if it's nighttime, canopy photosynthesis is null and only understorey is computed

    const double minLaiToComputePhotosynthesis = 0.1;
    if (plant.getLAICanopy() > minLaiToComputePhotosynthesis && environmentalVariable.sineSolarElevation > 0.001)
    {
        Crit3DHydrall::radiationAbsorption();
        Crit3DHydrall::photosynthesisAndTranspirationUnderstorey();
        Crit3DHydrall::aerodynamicalCoupling();
        Crit3DHydrall::upscale();
        Crit3DHydrall::carbonWaterFluxesProfile();
    }
    else
    {
        Crit3DHydrall::nullPhotosynthesis();
        Crit3DHydrall::photosynthesisAndTranspirationUnderstorey();
    }

    Crit3DHydrall::cumulatedResults();

    return 0;
}


double Crit3DHydrall::photosynthesisAndTranspirationUnderstorey()
{
    understoreyTranspirationRate.resize(1);
    understoreyTranspirationRate[0] = 0;

    if (understorey.absorbedPAR > EPSILON && environmentalVariable.sineSolarElevation > 0.001)
    {
        const double rootEfficiencyInWaterExtraction = 1.25e-3;  //[kgH2O kgDM-1 s-1]
        const double understoreyLightUtilization = 1.77e-9;      //[kgC J-1]
        double cumulatedUnderstoreyTranspirationRate = 0;
        double waterUseEfficiency;                               //[molC molH2O-1]

        waterUseEfficiency = environmentalVariable.CO2 * 0.1875 / weatherVariable.vaporPressureDeficit;

        double lightLimitedUnderstoreyAssimilation;          //[molC m-2 s-1]
        double waterLimitedUnderstoreyAssimilation;          //[molC m-2 s-1]

        lightLimitedUnderstoreyAssimilation = understoreyLightUtilization * understorey.absorbedPAR / MC; //convert units from kgC m-2 s-1 into molC m-2 s-1
        double density=1.;
        if (soil.layersNr != NODATA)
        {
            if (soil.layersNr > 1)
                density = 1./(soil.layersNr-1);

            for (int i = 1; i < soil.layersNr; i++)
            {
                understoreyTranspirationRate.push_back(rootEfficiencyInWaterExtraction * understoreyBiomass.fineRoot * soil.stressCoefficient[i]*density);
                cumulatedUnderstoreyTranspirationRate += understoreyTranspirationRate[i];
            }
        }

        cumulatedUnderstoreyTranspirationRate /= MH2O;  //convert units from kgH2O m-2 s-1 into molH2O m-2 s-1
        waterLimitedUnderstoreyAssimilation = cumulatedUnderstoreyTranspirationRate * waterUseEfficiency;

        if (lightLimitedUnderstoreyAssimilation > waterLimitedUnderstoreyAssimilation)
        {
            understoreyAssimilationRate = waterLimitedUnderstoreyAssimilation;
        }
        else
        {
            understoreyAssimilationRate = lightLimitedUnderstoreyAssimilation;
            double lightWaterRatio = lightLimitedUnderstoreyAssimilation/waterLimitedUnderstoreyAssimilation;
            for (int j = 1; j < soil.layersNr; j++)
            {
                understoreyTranspirationRate[j] *= lightWaterRatio;
            }
        }
    }
    else
    {
        for (int i = 1; i < soil.layersNr; i++)
            understoreyTranspirationRate.push_back(0);
        understoreyAssimilationRate = 0;
    }
    return 0;
}


Crit3DHydrall::Crit3DHydrall()
{
    initialize();
}


void Crit3DHydrall::initialize()
{
    plant.myChlorophyllContent = NODATA;
    elevation = NODATA;
    isFirstYearSimulation = true;
    totalTranspirationRate = 0;

    carbonStock = 0;

    // .. TODO
}

void Crit3DHydrall::setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, Crit3DDate currentDate, double sunElevation,double meanTemp30Days,double et0)
{
    setWeatherVariables(temp, irradiance, prec, relativeHumidity, windSpeed, directIrradiance, diffuseIrradiance, cloudIndex, atmosphericPressure,meanTemp30Days,et0);
    environmentalVariable.CO2 = getCO2(currentDate);
    environmentalVariable.sineSolarElevation = MAXVALUE(0.0001,sin(sunElevation*DEG_TO_RAD));
}

bool Crit3DHydrall::setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, double meanTemp30Days, double et0)
{

    bool isReadingOK = false ;
    weatherVariable.irradiance = irradiance ;
    weatherVariable.myInstantTemp = temp ;
    weatherVariable.prec = prec ;
    weatherVariable.relativeHumidity = relativeHumidity ;
    weatherVariable.windSpeed = windSpeed ;
    //weatherVariable.meanDailyTemperature = meanDailyTemp;
    double deltaRelHum = MAXVALUE(100.0 - weatherVariable.relativeHumidity, 0.01);
    weatherVariable.vaporPressureDeficit = 0.01 * deltaRelHum * 613.75 * exp(17.502 * weatherVariable.myInstantTemp / (240.97 + weatherVariable.myInstantTemp));
    weatherVariable.atmosphericPressure = atmosphericPressure;
    weatherVariable.last30DaysTAvg = meanTemp30Days;

    setDerivedWeatherVariables(directIrradiance, diffuseIrradiance, cloudIndex,et0);

    if ((int(prec) != NODATA) && (int(temp) != NODATA) && (int(windSpeed) != NODATA)
        && (int(irradiance) != NODATA) && (int(relativeHumidity) != NODATA)
        && (int(atmosphericPressure) != NODATA))
        isReadingOK = true;

    return isReadingOK;
}

void Crit3DHydrall::setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex,double et0)
{
    weatherVariable.derived.airVapourPressure = saturationVaporPressure(weatherVariable.myInstantTemp)*weatherVariable.relativeHumidity/100.;
    weatherVariable.derived.slopeSatVapPressureVSTemp = 2588464.2 / POWER2(240.97 + weatherVariable.myInstantTemp) * exp(17.502 * weatherVariable.myInstantTemp / (240.97 + weatherVariable.myInstantTemp)) ;
    weatherVariable.derived.myDirectIrradiance = directIrradiance;
    weatherVariable.derived.myDiffuseIrradiance = diffuseIrradiance;
    double myCloudiness = BOUNDFUNCTION(0,1,cloudIndex);
    weatherVariable.derived.myEmissivitySky = 1.24 * pow((weatherVariable.derived.airVapourPressure/100.0) / (weatherVariable.myInstantTemp+ZEROCELSIUS),(1.0/7.0))*(1 - 0.84*myCloudiness)+ 0.84*myCloudiness;
    weatherVariable.derived.myLongWaveIrradiance = POWER4(weatherVariable.myInstantTemp+ZEROCELSIUS) * weatherVariable.derived.myEmissivitySky * STEFAN_BOLTZMANN ;
    weatherVariable.derived.psychrometricConstant = psychro(weatherVariable.atmosphericPressure,weatherVariable.myInstantTemp);
    weatherVariable.derived.et0 = et0;
    return;
}

bool Crit3DHydrall::setPlantVariables(int forestIndex, double chlorophyllContent, double height, double psiMinimum)
{
    plant.myChlorophyllContent = chlorophyllContent;
    plant.height = height;
    plant.psiLeafMinimum = psiMinimum;
    //plant.psiSoilCritical = psiCritical;

    if (forestIndex >= conversionTableVector.size())
        return false;

    parameterWangLeuning.maxCarboxRate = plant.tableEcophysiologicalParameters[conversionTableVector[forestIndex]].Vcmo;
    plant.isAmphystomatic = plant.tableEcophysiologicalParameters[conversionTableVector[forestIndex]].isAmphystomatic;
    plant.rootShootRatioRef = plant.tableEcophysiologicalParameters[conversionTableVector[forestIndex]].rootShootRatio;
    plant.mBallBerry = plant.tableEcophysiologicalParameters[conversionTableVector[forestIndex]].mBallBerry;
    plant.wildfireDamage = plant.tableEcophysiologicalParameters[conversionTableVector[forestIndex]].wildfireDamage;

    return true;
}

void Crit3DHydrall::setStateVariables(const Crit3DHydrallMaps &stateMap, int row, int col)
{
    statePlant.treeNetPrimaryProduction = stateMap.treeNetPrimaryProduction->value[row][col];
    statePlant.understoreyNetPrimaryProduction = stateMap.understoreyNetPrimaryProduction->value[row][col];

    statePlant.treeBiomassFoliage = stateMap.treeBiomassFoliage->value[row][col];
    statePlant.understoreyBiomassFoliage = stateMap.understoreyBiomassFoliage->value[row][col];

    statePlant.treeBiomassRoot = stateMap.treeBiomassRoot->value[row][col];
    statePlant.understoreyBiomassRoot = stateMap.understoreyBiomassRoot->value[row][col];

    statePlant.treeBiomassSapwood = stateMap.treeBiomassSapwood->value[row][col];

    outputC = stateMap.outputC->value[row][col];
}

void Crit3DHydrall::setSoilVariables(int iLayer, int currentNode,float checkFlag, double waterContent, double waterContentFC, double waterContentWP,double clay, double sand,double thickness,double bulkDensity,double waterContentSat, double kSat, double waterPotential)
{
    if (iLayer == 0)
    {
        soil.layersNr = 0;
    }
    (soil.layersNr)++;

    soil.waterContent.resize(soil.layersNr);
    soil.stressCoefficient.resize(soil.layersNr);
    soil.clay.resize(soil.layersNr);
    soil.sand.resize(soil.layersNr);
    soil.silt.resize(soil.layersNr);
    soil.nodeThickness.resize(soil.layersNr);
    soil.bulkDensity.resize(soil.layersNr);
    soil.saturation.resize(soil.layersNr);
    soil.fieldCapacity.resize(soil.layersNr);
    soil.wiltingPoint.resize(soil.layersNr);
    soil.satHydraulicConductivity.resize(soil.layersNr);
    soil.waterPotential.resize(soil.layersNr);


    if (currentNode != checkFlag)
    {
        soil.waterContent[iLayer] = waterContent;
        soil.stressCoefficient[iLayer] = BOUNDFUNCTION(0.01,1, MINVALUE((10*(waterContent-waterContentWP))/(3*(waterContentFC-waterContentWP)),  log(waterContentSat/waterContent)/log(2*waterContentSat/(waterContentFC+waterContentSat))));
        //soil.stressCoefficient[iLayer] = BOUNDFUNCTION(0,MINVALUE(1.0,(waterContent - 2* waterContentFC)/(waterContentSat-2 * waterContentFC)), (10*(soil.waterContent[iLayer]-waterContentWP))/(3*(waterContentFC-waterContentWP)));
        //soil.stressCoefficient[iLayer] = 0.01;
        soil.clay[iLayer] = clay/100.;
        soil.sand[iLayer] = sand/100.;
        soil.silt[iLayer] = 1 - soil.sand[iLayer] - soil.clay[iLayer];
        soil.nodeThickness[iLayer] = thickness;
        soil.bulkDensity[iLayer] = bulkDensity;
        soil.fieldCapacity[iLayer] = waterContentFC;
        soil.wiltingPoint[iLayer] = waterContentWP;
        soil.saturation[iLayer] = waterContentSat;
        soil.satHydraulicConductivity[iLayer] = kSat;
        soil.waterPotential[iLayer] = waterPotential;
    }

    //soil.clayAverage = statistics::weighedMean(soil.nodeThickness,soil.clay);
    //soil.clayAverage = statistics::weighedMean(soil.nodeThickness,soil.sand);
    //soil.siltAverage = 1 - soil.clayAverage - soil.sandAverage;
    //soil.bulkDensityAverage = statistics::weighedMean(soil.nodeThickness,soil.bulkDensity);
}


void Crit3DHydrall::saveStateVariables(Crit3DHydrallMaps &stateMap, int row, int col)
{
    stateMap.treeNetPrimaryProduction->value[row][col] = (float)statePlant.treeNetPrimaryProduction;
    stateMap.understoreyNetPrimaryProduction->value[row][col] = (float)statePlant.understoreyNetPrimaryProduction;

    stateMap.treeBiomassFoliage->value[row][col] = (float)statePlant.treeBiomassFoliage;
    stateMap.understoreyBiomassFoliage->value[row][col] = (float)statePlant.understoreyBiomassFoliage;

    stateMap.treeBiomassRoot->value[row][col] = (float)statePlant.treeBiomassRoot;
    stateMap.understoreyBiomassRoot->value[row][col] = (float)statePlant.understoreyBiomassRoot;

    stateMap.treeBiomassSapwood->value[row][col] = (float)statePlant.treeBiomassSapwood;

    stateMap.outputC->value[row][col] = (float)outputC;
}

/*
void Crit3DHydrall::getPlantAndSoilVariables(Crit3DHydrallMaps &map, int row, int col)
{
    map.criticalSoilWaterPotential->value[row][col] = plant.psiSoilCritical;
    map.minLeafWaterPotential->value[row][col] = plant.psiLeafMinimum;
    map.criticalTranspiration->value[row][col] = plant.transpirationCritical;
}*/

void Crit3DHydrall::radiationAbsorption()
{
    // taken from Hydrall Model, Magnani UNIBO

    // TODO chiedere a Magnani questi parametri
    static double   leafAbsorbanceNIR= 0.2;
    static double   clumpingParameter = 1.0 ; // from 0 to 1 <1 for needles
    double  diffuseLightSector1K = 0.5;
    double diffuseLightSector2K = 0.5;
    double diffuseLightSector3K = 0.5;
    double scatteringCoefPAR, scatteringCoefNIR ;
    std::vector<double> dum(17, NODATA);
    double  sunlitAbsorbedNIR ,  shadedAbsorbedNIR, sunlitAbsorbedLW , shadedAbsorbedLW;
    double directIncomingPAR, directIncomingNIR , diffuseIncomingPAR , diffuseIncomingNIR,leafAbsorbancePAR;
    double directReflectionCoefficientPAR , directReflectionCoefficientNIR , diffuseReflectionCoefficientPAR , diffuseReflectionCoefficientNIR;
    double canopyLAI = plant.getLAICanopy();
    //projection of the unit leaf area in the direction of the sun's beam, following Sellers 1985 (in Wang & Leuning 1998)

    //directLightExtinctionCoefficient.global = MINVALUE(50,(0.5 - hemisphericalIsotropyParameter*(0.633-1.11*environmentalVariable.sineSolarElevation) - POWER2(hemisphericalIsotropyParameter)*(0.33-0.579*environmentalVariable.sineSolarElevation))/ environmentalVariable.sineSolarElevation);

    directLightExtinctionCoefficient.global = MINVALUE(50,0.5/environmentalVariable.sineSolarElevation);
    /*Extinction coeff for canopy of black leaves, diffuse radiation
    The average extinctio coefficient is computed considering three sky sectors,
    assuming SOC conditions (Goudriaan & van Laar 1994, p 98-99)*/
    //diffuseLightSector1K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.259) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.259))/0.259;  //projection of unit leaf for first sky sector (0-30 elevation)
    //diffuseLightSector2K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.707) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.707))/0.707 ; //second sky sector (30-60 elevation)
    //diffuseLightSector3K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.966) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.966))/ 0.966 ; // third sky sector (60-90 elevation)

    //if (plant.getLAICanopy() > EPSILON)
    //{
        diffuseLightExtinctionCoefficient.global =- 1.0/canopyLAI * log(0.178 * exp(-diffuseLightSector1K*canopyLAI) + 0.514 * exp(-diffuseLightSector2K*canopyLAI)
                                                                                          + 0.308 * exp(-diffuseLightSector3K*canopyLAI));  //approximation based on relative radiance from 3 sky sectors
    //}
    //else
    //{
        //diffuseLightExtinctionCoefficient.global = (diffuseLightSector1K+diffuseLightSector2K+diffuseLightSector3K)/3;
    //}
        //Include effects of leaf clumping (see Goudriaan & van Laar 1994, p 110)
    directLightExtinctionCoefficient.global  *= clumpingParameter ;//direct light
    diffuseLightExtinctionCoefficient.global *= clumpingParameter ;//diffuse light
    //Based on approximation by Goudriaan 1977 (in Goudriaan & van Laar 1994)
    double exponent= -pow(10,0.28 + 0.63*log10(plant.myChlorophyllContent*0.85/1000));
    leafAbsorbancePAR = 1 - pow(10,exponent);//from Agusti et al (1994), Eq. 1, assuming Chl a = 0.85 Chl (a+b)
    scatteringCoefPAR = 1.0 - leafAbsorbancePAR ; //scattering coefficient for PAR
    scatteringCoefNIR = 1.0 - leafAbsorbanceNIR ; //scattering coefficient for NIR
    diffuseLightExtinctionCoefficient.par = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR) ;//extinction coeff of PAR, direct light
    diffuseLightExtinctionCoefficient.nir = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, direct light
    directLightExtinctionCoefficient.par  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR); //extinction coeff of PAR, diffuse light
    directLightExtinctionCoefficient.nir  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, diffuse light

    if (environmentalVariable.sineSolarElevation > 0.001)
    {
        //Leaf area index of sunlit (1) and shaded (2) big-leaf
        sunlit.leafAreaIndex = UPSCALINGFUNC(directLightExtinctionCoefficient.global,canopyLAI);
        shaded.leafAreaIndex = canopyLAI - sunlit.leafAreaIndex ;
        //understorey.leafAreaIndex = 0.2;
        //Extinction coefficients for direct and diffuse PAR and NIR radiation, scattering leaves
        //Based on approximation by Goudriaan 1977 (in Goudriaan & van Laar 1994)
        /*double exponent= -pow(10,0.28 + 0.63*log10(plant.myChlorophyllContent*0.85/1000));
        leafAbsorbancePAR = 1 - pow(10,exponent);//from Agusti et al (1994), Eq. 1, assuming Chl a = 0.85 Chl (a+b)
        scatteringCoefPAR = 1.0 - leafAbsorbancePAR ; //scattering coefficient for PAR
        scatteringCoefNIR = 1.0 - leafAbsorbanceNIR ; //scattering coefficient for NIR
        diffuseLightExtinctionCoefficient.par = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR) ;//extinction coeff of PAR, direct light
        diffuseLightExtinctionCoefficient.nir = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, direct light
        directLightExtinctionCoefficient.par  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR); //extinction coeff of PAR, diffuse light
        directLightExtinctionCoefficient.nir  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, diffuse light
        */
        //Canopy+soil reflection coefficients for direct, diffuse PAR, NIR radiation
        dum[2]= (1-sqrt(1-scatteringCoefPAR)) / (1+sqrt(1-scatteringCoefPAR));
        dum[3]= (1-sqrt(1-scatteringCoefNIR)) / (1+sqrt(1-scatteringCoefNIR));
        dum[4]= 2.0 * directLightExtinctionCoefficient.global / (directLightExtinctionCoefficient.global + diffuseLightExtinctionCoefficient.global) ;
        directReflectionCoefficientNIR = directReflectionCoefficientPAR = dum[4] * dum[2];
        diffuseReflectionCoefficientNIR = diffuseReflectionCoefficientPAR = dum[4] * dum[3];
        //Incoming direct PAR and NIR (W m-2)
        directIncomingNIR = directIncomingPAR = weatherVariable.derived.myDirectIrradiance * 0.5 ;
        //Incoming diffuse PAR and NIR (W m-2)
        diffuseIncomingNIR = diffuseIncomingPAR = weatherVariable.derived.myDiffuseIrradiance * 0.5 ;
        //Preliminary computations

        preliminaryComputations(diffuseIncomingPAR, diffuseReflectionCoefficientPAR, directIncomingPAR, directReflectionCoefficientPAR,
                                diffuseIncomingNIR, diffuseReflectionCoefficientNIR, directIncomingNIR, directReflectionCoefficientNIR, scatteringCoefPAR, scatteringCoefNIR, dum);

        // PAR absorbed by sunlit (1) and shaded (2) big-leaf (W m-2) from Wang & Leuning 1998
        sunlit.absorbedPAR = dum[5] * dum[11] + dum[6] * dum[12] + dum[7] * dum[15] ;
        shaded.absorbedPAR = dum[5]*(UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,canopyLAI)- dum[11])
                             + dum[6]*(UPSCALINGFUNC(directLightExtinctionCoefficient.par,canopyLAI)- dum[12])
                             - dum[7] * dum[15];
        // NIR absorbed by sunlit (1) and shaded (2) big-leaf (W m-2) fromWang & Leuning 1998
        sunlitAbsorbedNIR = dum[8]*dum[13]+dum[9]*dum[14]+dum[10]*dum[15];
        shadedAbsorbedNIR = dum[8]*(UPSCALINGFUNC(diffuseLightExtinctionCoefficient.nir,canopyLAI)-dum[13])+dum[9]*(UPSCALINGFUNC(directLightExtinctionCoefficient.nir,canopyLAI)- dum[14]) - dum[10] * dum[15];

        double emissivityLeaf = 0.96 ; // supposed constant because variation is very small
        double emissivitySoil= 0.94 ;   // supposed constant because variation is very small
        sunlitAbsorbedLW = (dum[16] * UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.global),canopyLAI))*emissivityLeaf+(1.0-emissivitySoil)*(emissivityLeaf-weatherVariable.derived.myEmissivitySky)* UPSCALINGFUNC((2*diffuseLightExtinctionCoefficient.global),canopyLAI)* UPSCALINGFUNC((directLightExtinctionCoefficient.global-diffuseLightExtinctionCoefficient.global),canopyLAI);
        shadedAbsorbedLW = dum[16] * UPSCALINGFUNC(diffuseLightExtinctionCoefficient.global,canopyLAI) - sunlitAbsorbedLW ;
        // Isothermal net radiation for sunlit (1) and shaded (2) big-leaf
        sunlit.isothermalNetRadiation= sunlit.absorbedPAR + sunlitAbsorbedNIR + sunlitAbsorbedLW ;
        shaded.isothermalNetRadiation = shaded.absorbedPAR + shadedAbsorbedNIR + shadedAbsorbedLW ;


        understorey.absorbedPAR = (directIncomingPAR + diffuseIncomingPAR - sunlit.absorbedPAR - shaded.absorbedPAR) * cover + (directIncomingPAR + diffuseIncomingPAR) * (1 - cover);
        understorey.absorbedPAR *= (1-std::exp(-0.8*understorey.leafAreaIndex));
    }
    else
    {
        sunlit.leafAreaIndex =  0.0 ;
        sunlit.absorbedPAR = 0.0 ;

        // TODO: non servono?
        sunlitAbsorbedNIR = 0.0 ;
        sunlitAbsorbedLW = 0.0 ;
        sunlit.isothermalNetRadiation =  0.0 ;

        understorey.absorbedPAR = 0.0;
        understorey.leafAreaIndex = 0.0;

        shaded.leafAreaIndex = canopyLAI;
        shaded.absorbedPAR = 0.0 ;
        shadedAbsorbedNIR = 0.0 ;
        dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*pow(weatherVariable.myInstantTemp + ZEROCELSIUS,4) ;
        dum[16] *= diffuseLightExtinctionCoefficient.global ;
        shadedAbsorbedLW= dum[16] * (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.global,canopyLAI) - UPSCALINGFUNC(directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.global,canopyLAI)) ;
        shaded.isothermalNetRadiation = shaded.absorbedPAR + shadedAbsorbedNIR + shadedAbsorbedLW ;
    }

    // Convert absorbed PAR into units of mol m-2 s-1
    sunlit.absorbedPAR *= 4.57E-6 ;
    shaded.absorbedPAR *= 4.57E-6 ;
}

void Crit3DHydrall::preliminaryComputations(double diffuseIncomingPAR, double diffuseReflectionCoefficientPAR, double directIncomingPAR, double directReflectionCoefficientPAR,
                                             double diffuseIncomingNIR, double diffuseReflectionCoefficientNIR, double directIncomingNIR, double directReflectionCoefficientNIR,
                                             double scatteringCoefPAR, double scatteringCoefNIR, std::vector<double> &dum)
{
    dum[5]= diffuseIncomingPAR * (1.0-diffuseReflectionCoefficientPAR) * diffuseLightExtinctionCoefficient.par ;
    dum[6]= directIncomingPAR * (1.0-directReflectionCoefficientPAR) * directLightExtinctionCoefficient.par ;
    dum[7]= directIncomingPAR * (1.0-scatteringCoefPAR) * directLightExtinctionCoefficient.global ;
    dum[8]=  diffuseIncomingNIR * (1.0-diffuseReflectionCoefficientNIR) * diffuseLightExtinctionCoefficient.nir;
    dum[9]= directIncomingNIR * (1.0-directReflectionCoefficientNIR) * directLightExtinctionCoefficient.nir;
    dum[10]= directIncomingNIR * (1.0-scatteringCoefNIR) * directLightExtinctionCoefficient.global ;
    dum[11]= UPSCALINGFUNC((diffuseLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.getLAICanopy());
    dum[12]= UPSCALINGFUNC((directLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.getLAICanopy());
    dum[13]= UPSCALINGFUNC((diffuseLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.getLAICanopy());
    dum[14]= UPSCALINGFUNC((directLightExtinctionCoefficient.nir+directLightExtinctionCoefficient.global),plant.getLAICanopy());
    dum[15]= UPSCALINGFUNC(directLightExtinctionCoefficient.global,plant.getLAICanopy()) - UPSCALINGFUNC((2.0*directLightExtinctionCoefficient.global),plant.getLAICanopy()) ;

    // Long-wave radiation balance by sunlit (1) and shaded (2) big-leaf (W m-2) from Wang & Leuning 1998
    dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*POWER4(weatherVariable.myInstantTemp+ZEROCELSIUS); //negativo
    dum[16] *= diffuseLightExtinctionCoefficient.global ;
}

void Crit3DHydrall::leafTemperature()
{
    if (environmentalVariable.sineSolarElevation > 0.001)
    {
        double sunlitGlobalRadiation,shadedGlobalRadiation;

        //double shadedIrradiance = myDiffuseIrradiance * shaded.leafAreaIndex / statePlant.stateGrowth.leafAreaIndex;
        shadedGlobalRadiation = weatherVariable.derived.myDiffuseIrradiance * HOUR_SECONDS ;
        shaded.leafTemperature = weatherVariable.myInstantTemp + 1.67*1.0e-6 * shadedGlobalRadiation - 0.25 * weatherVariable.vaporPressureDeficit / weatherVariable.derived.psychrometricConstant; // by Stanghellini 1987 phd thesis

        // sunlitIrradiance = myDiffuseIrradiance * sunlit.leafAreaIndex/ statePlant.stateGrowth.leafAreaIndex;
        //sunlitIrradiance = myDirectIrradiance * sunlit.leafAreaIndex/ statePlant.stateGrowth.leafAreaIndex;
        sunlitGlobalRadiation = (weatherVariable.derived.myDiffuseIrradiance + weatherVariable.derived.myDirectIrradiance) * HOUR_SECONDS ;
        sunlit.leafTemperature = weatherVariable.myInstantTemp + 1.67*1.0e-6 * sunlitGlobalRadiation - 0.25 * weatherVariable.vaporPressureDeficit / weatherVariable.derived.psychrometricConstant; // by Stanghellini 1987 phd thesis
    }
    else
    {
        sunlit.leafTemperature = shaded.leafTemperature = weatherVariable.myInstantTemp;
    }
    sunlit.leafTemperature += ZEROCELSIUS;
    shaded.leafTemperature += ZEROCELSIUS;
}

void Crit3DHydrall::aerodynamicalCoupling()
{
    double laiMinToComputeAerodynamicalCoupling = 0.05;
    if (plant.getLAICanopy() > laiMinToComputeAerodynamicalCoupling)
    {
        // taken from Hydrall Model, Magnani UNIBO
        static double A = 0.0067;
        static double BETA = 3.0;
        static double KARM = 0.41;
        double heightReference , roughnessLength,zeroPlaneDisplacement, sensibleHeat, frictionVelocity, windSpeedTopCanopy;
        double canopyAerodynamicConductanceToMomentum, aerodynamicConductanceToCO2, dummy, sunlitDeltaTemp,shadedDeltaTemp;
        double leafBoundaryLayerConductance;
        double aerodynamicConductanceForHeat=0;
        double windSpeed;
        canopyAerodynamicConductanceToMomentum = aerodynamicConductanceToCO2 = dummy = sunlitDeltaTemp = shadedDeltaTemp = NODATA;
        heightReference = plant.height + 5 ; // [m]
        windSpeed = weatherVariable.windSpeed * pow((heightReference/10.),0.14);
        windSpeed = MAXVALUE(3,weatherVariable.windSpeed);
        dummy = 0.2 * plant.getLAICanopy() ;
        zeroPlaneDisplacement = MINVALUE(plant.height * (log(1+pow(dummy,0.166)) + 0.03*log(1+powerIntegerExponent(dummy,6))), 0.99*plant.height) ;
        if (dummy < 0.2) roughnessLength = 0.01 + 0.28*sqrt(dummy) * plant.height ;
        else roughnessLength = 0.3 * plant.height * (1.0 - zeroPlaneDisplacement/plant.height);

        // Canopy energy balance.
        // Compute iteratively:
        // - leaf temperature (different for sunlit and shaded foliage)
        // - aerodynamic conductance (non-neutral conditions)

        // Initialize sensible heat flux and friction velocity
        sensibleHeat = sunlit.isothermalNetRadiation + shaded.isothermalNetRadiation ;
        frictionVelocity = MAXVALUE(1.0e-4,KARM*windSpeed/log((heightReference-zeroPlaneDisplacement)/roughnessLength));

        short i = 0 ;
        double sensibleHeatOld = NODATA;
        double threshold = fabs(sensibleHeat/10000.0);
        const double coefficientFromBeta = ((2.0/BETA)*(1-exp(-BETA/2.0)));
        while( (20 > i++) && (fabs(sensibleHeat - sensibleHeatOld)> threshold))
        {
            //first occurence of the loop
            if (isEqual(sunlit.aerodynamicConductanceCO2Exchange, 0))
            {
                sunlit.aerodynamicConductanceCO2Exchange = 1.05 * sunlit.leafAreaIndex/plant.getLAICanopy();
                shaded.aerodynamicConductanceCO2Exchange = 1.05 * shaded.leafAreaIndex/plant.getLAICanopy();
            }

            // Monin-Obukhov length (m) and nondimensional height
            // Note: imposed a limit to non-dimensional height under stable
            // conditions, corresponding to a value of 0.2 for the generalized
            // stability factor F (=1/FIM/FIH)
            sensibleHeatOld = sensibleHeat ;
            double moninObukhovLength,zeta,deviationFunctionForMomentum,deviationFunctionForHeat,radiativeConductance,totalConductanceToHeatExchange,stomatalConductanceWater ;

            moninObukhovLength = -(POWER3(frictionVelocity))*HEAT_CAPACITY_AIR_MOLAR*weatherVariable.atmosphericPressure;
            moninObukhovLength /= (R_GAS*(KARM*9.8*sensibleHeat));

            zeta = MINVALUE((heightReference-zeroPlaneDisplacement)/moninObukhovLength,0.25) ;

            if (zeta < 0)
            {
                //Stability function for momentum and heat (-)
                double x,y,stabilityFunctionForMomentum;
                stabilityFunctionForMomentum = pow((1.0-16.0*zeta),-0.25);
                x= 1.0/stabilityFunctionForMomentum;
                y= 1.0/POWER2((stabilityFunctionForMomentum));
                //Deviation function for momentum and heat (-)
                deviationFunctionForMomentum = 2.0*log((1+x)/2.) + log((1+x*x)/2.0)- 2.0*atan(x) + PI/2.0 ;
                deviationFunctionForHeat = 2*log((1+y)/2) ;
            }
            else
            {
                // Stable conditions
                //stabilityFunctionForMomentum = (1+5*zeta);
                // Deviation function for momentum and heat (-)
                deviationFunctionForMomentum = deviationFunctionForHeat = - 5*zeta ;
            }
            //friction velocity
            frictionVelocity = KARM*windSpeed/(log((heightReference-zeroPlaneDisplacement)/roughnessLength) - deviationFunctionForMomentum);
            frictionVelocity = MAXVALUE(frictionVelocity,1.0e-4);

            // Wind speed at canopy top	(m s-1)
            windSpeedTopCanopy = (frictionVelocity/KARM) * log((plant.height - zeroPlaneDisplacement)/roughnessLength);
            windSpeedTopCanopy = MAXVALUE(windSpeedTopCanopy,1.0e-4);

            // Average leaf boundary-layer conductance cumulated over the canopy (m s-1)
            leafBoundaryLayerConductance = A*sqrt(windSpeedTopCanopy/(leafWidth()))* coefficientFromBeta * plant.getLAICanopy();
            //       Total canopy aerodynamic conductance for momentum exchange (s m-1)
            canopyAerodynamicConductanceToMomentum= frictionVelocity / (windSpeed/frictionVelocity + (deviationFunctionForMomentum-deviationFunctionForHeat)/KARM);
            // Aerodynamic conductance for heat exchange (mol m-2 s-1)
            dummy =	(weatherVariable.atmosphericPressure/R_GAS)/(weatherVariable.myInstantTemp + ZEROCELSIUS);// conversion factor m s-1 into mol m-2 s-1
            aerodynamicConductanceForHeat =  ((canopyAerodynamicConductanceToMomentum*leafBoundaryLayerConductance)/(canopyAerodynamicConductanceToMomentum + leafBoundaryLayerConductance)) * dummy ; //whole canopy
            sunlit.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat * sunlit.leafAreaIndex/plant.getLAICanopy() ;//sunlit big-leaf
            shaded.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat - sunlit.aerodynamicConductanceHeatExchange ; //  shaded big-leaf
            // Canopy radiative conductance (mol m-2 s-1)
            radiativeConductance= 4*(weatherVariable.derived.slopeSatVapPressureVSTemp/weatherVariable.derived.psychrometricConstant)*(STEFAN_BOLTZMANN/HEAT_CAPACITY_AIR_MOLAR)*POWER3((weatherVariable.myInstantTemp + ZEROCELSIUS));
            // Total conductance to heat exchange (mol m-2 s-1)
            totalConductanceToHeatExchange =  aerodynamicConductanceForHeat + radiativeConductance; //whole canopy
            sunlit.totalConductanceHeatExchange = totalConductanceToHeatExchange * sunlit.leafAreaIndex/plant.getLAICanopy();	//sunlit big-leaf
            shaded.totalConductanceHeatExchange = totalConductanceToHeatExchange - sunlit.totalConductanceHeatExchange;  //shaded big-leaf

            // Temperature of big-leaf (approx. expression)
            stomatalConductanceWater = 10.0/shaded.leafAreaIndex ; //dummy stom res for shaded big-leaf
            //if (shaded.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/shaded.isothermalNetRadiation,0.5);
            shadedDeltaTemp = ((stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)*weatherVariable.derived.psychrometricConstant*shaded.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                               - weatherVariable.vaporPressureDeficit)/shaded.totalConductanceHeatExchange
                              /(weatherVariable.derived.psychrometricConstant*(stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)
                                 + weatherVariable.derived.slopeSatVapPressureVSTemp/shaded.totalConductanceHeatExchange);
            //shadedDeltaTemp = 0.0;
            shaded.leafTemperature = weatherVariable.myInstantTemp + shadedDeltaTemp + ZEROCELSIUS;  //shaded big-leaf

            if (sunlit.leafAreaIndex > EPSILON)
            {
                stomatalConductanceWater= (10.0/sunlit.leafAreaIndex); //dummy stom res for sunlit big-leaf
                //if (sunlit.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/sunlit.isothermalNetRadiation,0.5);
                sunlitDeltaTemp = ((stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceHeatExchange)
                                  *weatherVariable.derived.psychrometricConstant*sunlit.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                                  - weatherVariable.vaporPressureDeficit)
                                  /sunlit.totalConductanceHeatExchange/(weatherVariable.derived.psychrometricConstant
                                  *(stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceCO2Exchange)
                                  +weatherVariable.derived.slopeSatVapPressureVSTemp/sunlit.totalConductanceHeatExchange);
            }
            else
            {
                sunlitDeltaTemp = shadedDeltaTemp; // in night-time both temperatures must be equal
            }


            sunlit.leafTemperature = weatherVariable.myInstantTemp + sunlitDeltaTemp	+ ZEROCELSIUS ; //sunlit big-leaf


            // Sensible heat flux from the whole canopy
            sensibleHeat = HEAT_CAPACITY_AIR_MOLAR * (sunlit.aerodynamicConductanceHeatExchange*sunlitDeltaTemp + shaded.aerodynamicConductanceHeatExchange*shadedDeltaTemp);
        }

        if (plant.isAmphystomatic) aerodynamicConductanceToCO2 = 0.78 * aerodynamicConductanceForHeat; //amphystomatous species. Ratio of diffusivities from Wang & Leuning 1998
        else aerodynamicConductanceToCO2 = 0.78 * (canopyAerodynamicConductanceToMomentum * leafBoundaryLayerConductance)/(leafBoundaryLayerConductance + 2.0*canopyAerodynamicConductanceToMomentum) * dummy; //hypostomatous species

        sunlit.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * sunlit.leafAreaIndex/plant.getLAICanopy() ; //sunlit big-leaf
        shaded.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * shaded.leafAreaIndex/plant.getLAICanopy() ;  //shaded big-leaf

    }
    else
    {
        shaded.aerodynamicConductanceCO2Exchange = sunlit.aerodynamicConductanceCO2Exchange = 0;

        shaded.leafTemperature = sunlit.leafTemperature = weatherVariable.myInstantTemp + ZEROCELSIUS;
    }
}

double  Crit3DHydrall::leafWidth()
{
    // la funzione deve essere scritta secondo regole che possono fr variare lo spessore in base alla fenologia
    // come per la vite?
    //TODO leaf width
    plant.myLeafWidth = 0.02;
    return plant.myLeafWidth;
}

void Crit3DHydrall::upscale()
{
    //cultivar->parameterWangLeuning.maxCarbonRate era input, ora da prendere da classe e leggere da tipo di pianta. TODO
    double maxCarboxRate = 150; // umol CO2 m-2 s-1

    // taken from Hydrall Model, Magnani UNIBO
    static double BETA = 0.5 ;

    double dum[10],darkRespirationT0;
    double optimalCarboxylationRate,optimalElectronTransportRate ;
    double leafConvexityFactor ;
    //     Preliminary computations
    dum[0]= R_GAS/1000.0 * sunlit.leafTemperature ; //[kJ mol-1]
    dum[1]= R_GAS/1000.0 * shaded.leafTemperature ;
    dum[2]= sunlit.leafTemperature - ZEROCELSIUS ; // [oC]
    dum[3]= shaded.leafTemperature - ZEROCELSIUS ;



    // optimalCarboxylationRate = (nitrogenContent.interceptLeaf + nitrogenContent.slopeLeaf * nitrogenContent.leafNitrogen/specificLeafArea*1000)*1e-6; // carboxylation rate based on nitrogen leaf
    optimalCarboxylationRate = maxCarboxRate * 1.0e-6; // [mol m-2 s-1] from Greer et al. 2011
    darkRespirationT0 = 0.0089 * optimalCarboxylationRate ;
    //   Adjust unit dark respiration rate for temperature (mol m-2 s-1)
    sunlit.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[0])* UPSCALINGFUNC((directLightExtinctionCoefficient.global + diffuseLightExtinctionCoefficient.par),plant.getLAICanopy()); //sunlit big-leaf
    shaded.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[1]); //shaded big-leaf
    shaded.darkRespiration *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.getLAICanopy()) - UPSCALINGFUNC((directLightExtinctionCoefficient.global + diffuseLightExtinctionCoefficient.par),plant.getLAICanopy()));
    double entropicFactorElectronTransporRate = (-0.75*(weatherVariable.last30DaysTAvg)+660);  // entropy term for J (kJ mol-1 oC-1)
    double entropicFactorCarboxyliation = (-1.07*(weatherVariable.last30DaysTAvg)+668); // entropy term for VCmax (kJ mol-1 oC-1)
    if (environmentalVariable.sineSolarElevation > 1.0e-3)
    {
        //Stomatal conductance to CO2 in darkness (molCO2 m-2 s-1)
        sunlit.minimalStomatalConductance = parameterWangLeuning.stomatalConductanceMin  * UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy())	;
        shaded.minimalStomatalConductance = parameterWangLeuning.stomatalConductanceMin  * (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.getLAICanopy()) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy()));
        // Carboxylation rate
        //sunlit.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[0]); //sunlit big leaf
        //shaded.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[1]); //shaded big leaf
        sunlit.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,sunlit.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuning.optimalTemperatureForPhotosynthesis); //sunlit big leaf
        shaded.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,shaded.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuning.optimalTemperatureForPhotosynthesis); //shaded big leaf
        // Scale-up maximum carboxylation rate (mol m-2 s-1)
        sunlit.maximalCarboxylationRate *= UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy());
        shaded.maximalCarboxylationRate *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.getLAICanopy()) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy()));
        //CO2 compensation point in dark
        sunlit.carbonMichaelisMentenConstant = exp(CKC - HAKC/dum[0]) * 1.0e-6 * weatherVariable.atmosphericPressure ;
        shaded.carbonMichaelisMentenConstant = exp(CKC - HAKC/dum[1]) * 1.0E-6 * weatherVariable.atmosphericPressure ;
        // Adjust Michaelis constant of oxygenation for temp (Pa)
        sunlit.oxygenMichaelisMentenConstant = exp(CKO - HAKO/dum[0])* 1.0e-3 * weatherVariable.atmosphericPressure ;
        shaded.oxygenMichaelisMentenConstant = exp(CKO - HAKO/dum[1])* 1.0E-3 * weatherVariable.atmosphericPressure ;
        // CO2 compensation point with no dark respiration (Pa)
        sunlit.compensationPoint = exp(CGSTAR - HAGSTAR/dum[0]) * 1.0e-6 * weatherVariable.atmosphericPressure ;
        shaded.compensationPoint = exp(CGSTAR - HAGSTAR/dum[1]) * 1.0e-6 * weatherVariable.atmosphericPressure ;
        // Electron transport
        // Compute potential e- transport at ref temp (mol e m-2 s-1) from correlation with Vcmax
        optimalElectronTransportRate = 1.5 * optimalCarboxylationRate ; //general correlation based on Leuning (1997)
        // check value and compare with 2.5 value in Medlyn et al (1999) and 1.67 value in Medlyn et al (2002) Based on greer Weedon 2011
        // Adjust maximum potential electron transport for temperature (mol m-2 s-1)
        //sunlit.maximalElectronTrasportRate = optimalElectronTransportRate * exp(CJM - HAJM/dum[0]);
        //shaded.maximalElectronTrasportRate = optimalElectronTransportRate * exp(CJM - HAJM/dum[1]);
        sunlit.maximalElectronTrasportRate = optimalElectronTransportRate * acclimationFunction(HAJM*1000,HDEACTIVATION*1000,sunlit.leafTemperature,entropicFactorElectronTransporRate,parameterWangLeuning.optimalTemperatureForPhotosynthesis);
        shaded.maximalElectronTrasportRate = optimalElectronTransportRate * acclimationFunction(HAJM*1000,HDEACTIVATION*1000,shaded.leafTemperature,entropicFactorElectronTransporRate,parameterWangLeuning.optimalTemperatureForPhotosynthesis);

        // Compute maximum PSII quantum yield, light-acclimated (mol e- mol-1 quanta absorbed)
        sunlit.quantumYieldPS2 = 0.352 + 0.022*dum[2] - 3.4E-4*POWER2((dum[2]));      //sunlit big-leaf
        shaded.quantumYieldPS2 = 0.352 + 0.022*dum[3] - 3.4E-4*POWER2((dum[3]));      //shaded big-leaf
        // Compute convexity factor of light response curve (-)
        // The value derived from leaf Chl content is modified for temperature effects, according to Bernacchi et al. (2003)
        leafConvexityFactor = 1 - plant.myChlorophyllContent * 6.93E-4 ;    //from Pons & Anten (2004), fig. 3b
        sunlit.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[2] - 3.7E-4*POWER2((dum[2])));  //sunlit big-leaf
        shaded.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[3] - 3.7E-4*POWER2((dum[3])));  //shaded big-leaf
        // Scale-up potential electron transport of sunlit big-leaf (mol m-2 s-1)
        sunlit.maximalElectronTrasportRate *= UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy());
        // Adjust electr transp of sunlit big-leaf for PAR effects (mol e- m-2 s-1)
        dum[4]= sunlit.absorbedPAR * sunlit.quantumYieldPS2 * BETA ; //  potential PSII e- transport of sunlit big-leaf (mol m-2 s-1)
        dum[5]= dum[4] + sunlit.maximalElectronTrasportRate ;
        dum[6]= dum[4] * sunlit.maximalElectronTrasportRate ;
        sunlit.maximalElectronTrasportRate = (dum[5] - sqrt(POWER2((dum[5])) - 4.0*sunlit.convexityFactorNonRectangularHyperbola*dum[6])) / (2.0*sunlit.convexityFactorNonRectangularHyperbola);
        // Scale-up potential electron transport of shaded big-leaf (mol m-2 s-1)
        // The simplified formulation proposed by de Pury & Farquhar (1999) is applied
        shaded.maximalElectronTrasportRate *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.getLAICanopy()) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.getLAICanopy()));
        // Adjust electr transp of shaded big-leaf for PAR effects (mol e- m-2 s-1)
        dum[4]= shaded.absorbedPAR * shaded.quantumYieldPS2 * BETA ; // potential PSII e- transport of sunlit big-leaf (mol m-2 s-1)
        dum[5]= dum[4] + shaded.maximalElectronTrasportRate ;
        dum[6]= dum[4] * shaded.maximalElectronTrasportRate ;
        shaded.maximalElectronTrasportRate = (dum[5] - sqrt(POWER2((dum[5])) - 4.0*shaded.convexityFactorNonRectangularHyperbola*dum[6])) / (2.0*shaded.convexityFactorNonRectangularHyperbola);
    }
    else
    {  //night-time computations
        sunlit.maximalElectronTrasportRate = 0.0;
        shaded.maximalElectronTrasportRate = 0.0;
        sunlit.darkRespiration = 0.0;
        sunlit.maximalCarboxylationRate = 0.0;
    }
}

inline double Crit3DHydrall::acclimationFunction(double Ha , double Hd, double leafTemp,
                                             double entropicTerm,double optimumTemp)
{
    // taken from Hydrall Model, Magnani UNIBO
    return exp(Ha*(leafTemp - optimumTemp)/(optimumTemp*R_GAS*leafTemp))
           *(1+exp((optimumTemp*entropicTerm-Hd)/(optimumTemp*R_GAS)))
           /(1+exp((leafTemp*entropicTerm-Hd)/(leafTemp*R_GAS)));
}


void Crit3DHydrall::carbonWaterFluxesProfile()
{
    // taken from Hydrall Model, Magnani UNIBO
    treeAssimilationRate = 0 ;

    //ball-berry constants. TODO
    double mi = 9.31;

    treeTranspirationRate.resize(soil.layersNr);

    //double totalStomatalConductance = 0;
    maxIterationNumber = 0;
    for (int i=0; i < soil.layersNr; i++)
    {
        treeTranspirationRate[i] = 0;
        sunlit.assimilation = 0.0;
        sunlit.stomatalConductance = 0.0;
        sunlit.transpiration = 0.0;
        shaded.assimilation = 0.0;
        shaded.transpiration = 0.0;
        if (i > 0)
        {
            if(sunlit.leafAreaIndex > 0)
            {
                Crit3DHydrall::photosynthesisKernel(sunlit.compensationPoint, sunlit.aerodynamicConductanceCO2Exchange, sunlit.aerodynamicConductanceHeatExchange, sunlit.minimalStomatalConductance,
                                                                 sunlit.maximalElectronTrasportRate, sunlit.carbonMichaelisMentenConstant,
                                                                 sunlit.oxygenMichaelisMentenConstant,sunlit.darkRespiration, sunlit.isothermalNetRadiation,
                                                                 mi * soil.stressCoefficient[i], sunlit.maximalCarboxylationRate,
                                                                 &(sunlit.assimilation), &(sunlit.stomatalConductance),
                                                                 &(sunlit.transpiration));
            }

            //treeAssimilationRate += sunlit.assimilation * soil.rootDensity[i] ;

            // shaded big leaf
            Crit3DHydrall::photosynthesisKernel(shaded.compensationPoint, shaded.aerodynamicConductanceCO2Exchange,shaded.aerodynamicConductanceHeatExchange, shaded.minimalStomatalConductance,
                                                             shaded.maximalElectronTrasportRate, shaded.carbonMichaelisMentenConstant,
                                                             shaded.oxygenMichaelisMentenConstant,shaded.darkRespiration, shaded.isothermalNetRadiation,
                                                             mi * soil.stressCoefficient[i], shaded.maximalCarboxylationRate,
                                                             &(shaded.assimilation), &(shaded.stomatalConductance),
                                                             &(shaded.transpiration));
            treeAssimilationRate += ( shaded.assimilation + sunlit.assimilation) * soil.getRootDensity()[i] ; //canopy gross assimilation (mol m-2 s-1)
        }

        treeTranspirationRate[i] += (shaded.transpiration + sunlit.transpiration) * soil.getRootDensity()[i] ;
    }
    maxIterationNumber /= soil.layersNr;
}


void Crit3DHydrall::photosynthesisKernel(double COMP,double GAC,double GHR,double GSCD,double J,double KC,double KO
                                            ,double RD,double RNI,double STOMWL,double VCmax,double *ASS,double *GSC,double *TR)
{
// taken from Hydrall Model, Magnani UNIBO
// daily time computation
// STOMWL in set in Pascal in order to compute the recursive algorithm in molCO2 m-2 s-1
#define NODATA_TOLERANCE 9999
#define OSS              21176       /*!< oxygen part pressure in the atmosphere, Pa  */

    double myStromalCarbonDioxide, VPDS, WC,WJ,VC,DUM1,CS; //, myPreviousVPDS;
    double ASSOLD, deltaAssimilation, myTolerance; //, myPreviousDelta;
    int I,Imax ;
    double myStromalCarbonDioxideOld;

    //leaf surface relative humidity conversion factor from VPD
    double RHFactor = (613.75 * exp(17.502 * weatherVariable.myInstantTemp / (240.97 + weatherVariable.myInstantTemp)));
    double RH;

    Imax = 10000 ;
    myTolerance = 1e-7;
    deltaAssimilation = NODATA_TOLERANCE;
    double CSmolFraction = NODATA;
    double COMPmolFraction = NODATA;
    //myPreviousDelta = deltaAssimilation;
    if (J >= 1.0e-7)
    {
        // Initialize variables
        myStromalCarbonDioxide = 0.7 * environmentalVariable.CO2 ;
        VPDS = weatherVariable.vaporPressureDeficit;
        //myPreviousVPDS = VPDS;
        RH = 1 - VPDS / RHFactor;
        ASSOLD = NODATA;
        DUM1 = 1.6 * weatherVariable.derived.slopeSatVapPressureVSTemp/weatherVariable.derived.psychrometricConstant + GHR/GAC;
        double dampingPar = 0.01;
        for (I=0; (I<Imax) && (deltaAssimilation > myTolerance); I++)
        {
            //Assimilation
            WC = VCmax * myStromalCarbonDioxide / (myStromalCarbonDioxide + KC * (1.0 + OSS / KO));  //RuBP-limited carboxylation (mol m-2 s-1)
            WJ = J * myStromalCarbonDioxide / (4.5 * myStromalCarbonDioxide + 10.5 * COMP);  //electr transp-limited carboxyl (mol m-2 s-1)
            VC = MINVALUE(WC,WJ);  //carboxylation rate (mol m-2 s-1)

            *ASS = MAXVALUE(1e-8, VC * (1.0 - COMP / myStromalCarbonDioxide));  //gross assimilation (mol m-2 s-1)
            CS = environmentalVariable.CO2 - weatherVariable.atmosphericPressure * (*ASS - RD) / GAC;	//CO2 concentration at leaf surface (Pa)
            CSmolFraction = CS/weatherVariable.atmosphericPressure*1e6;
            COMPmolFraction= COMP/weatherVariable.atmosphericPressure*1e6;
            CS = MAXVALUE(1e-4,CS);
            CSmolFraction = MAXVALUE(1e-3, CSmolFraction);
            //Stomatal conductance
            double temp = (CS-COMP)*weatherVariable.atmosphericPressure/1e6;
            double temp2 = (*ASS-RD) / ((CSmolFraction-COMP)*weatherVariable.atmosphericPressure) * RH;
            //*GSC = GSCD + STOMWL * (*ASS-RD) / (CS-COMP) * RH; //stom conduct to CO2 (mol m-2 s-1)
            *GSC = GSCD + STOMWL * (*ASS-RD)*1e6/ (CSmolFraction-COMPmolFraction) * RH; //stom conduct to CO2 (mol m-2 s-1)
            *GSC = MAXVALUE(*GSC,1.0e-5);
            // Stromal CO2 concentration
            myStromalCarbonDioxideOld = myStromalCarbonDioxide;
            myStromalCarbonDioxide = CS - weatherVariable.atmosphericPressure * (*ASS - RD) / (*GSC);	 //CO2 concentr at carboxyl sites (Pa)
            myStromalCarbonDioxide = BOUNDFUNCTION(0.01,environmentalVariable.CO2,myStromalCarbonDioxide);
            myStromalCarbonDioxide = dampingPar*myStromalCarbonDioxide + (1-dampingPar)*myStromalCarbonDioxideOld;
            myStromalCarbonDioxide = BOUNDFUNCTION(0.01,environmentalVariable.CO2,myStromalCarbonDioxide);
            //Vapour pressure deficit at leaf surface
            VPDS = (weatherVariable.derived.slopeSatVapPressureVSTemp / HEAT_CAPACITY_AIR_MOLAR*RNI + weatherVariable.vaporPressureDeficit * GHR) / (GHR+(*GSC)*DUM1);  //VPD at the leaf surface (Pa)
            RH = 1 - VPDS / RHFactor;
            deltaAssimilation = fabs((*ASS) - ASSOLD);

            if (I>0)
            {
                double ratioAssimilation = BOUNDFUNCTION(0.1,10,*ASS/ASSOLD); // RD(new):RD(old)=ASS(new):ASS(old)
                RD *= ratioAssimilation;
            }
            ASSOLD = *ASS;
            maxIterationNumber++;
        }
    }
    else //night time computation
    {
        *ASS= 0.0;
        *GSC= GSCD;
        VPDS= weatherVariable.vaporPressureDeficit ;
    }
    //  Transpiration rate
    /*
     one has to convert the diffusivity computed for CO2 to H20. The ratio is set to 0.64
    */
    *TR = MAXVALUE(1.0E-8,(*GSC / 0.64) * VPDS/weatherVariable.atmosphericPressure) ;  //Transpiration rate (mol m-2 s-1). Ratio of diffusivities from Wang & Leuning 1998
    //*TR = MAXVALUE(1.0E-8,*TR);
}


void Crit3DHydrall::cumulatedResults()
{
    // taken from Hydrall Model, Magnani UNIBO
    // Cumulate hourly values of gas exchange

    //deltaTime.absorbedPAR = HOUR_SECONDS*(sunlit.absorbedPAR+shaded.absorbedPAR);  //absorbed PAR (mol m-2)
    deltaTime.grossAssimilation = HOUR_SECONDS * treeAssimilationRate ; // canopy gross assimilation (mol m-2)
    deltaTime.respiration = HOUR_SECONDS * Crit3DHydrall::plantRespiration() ;
    deltaTime.netAssimilation = deltaTime.grossAssimilation - deltaTime.respiration ;
    if (printHourlyRecords)
    {
        //DEBUG
        //std::cout << deltaTime.grossAssimilation * 10e6 << ", " << deltaTime.respiration * 10e6 << ", " << deltaTime.netAssimilation *10e6 << std::endl;
        std::ofstream myFile;
        myFile.open("outputLAIetc.csv", std::ios_base::app);
        //myFile << deltaTime.grossAssimilation/HOUR_SECONDS*1e6 <<","<<deltaTime.respiration/HOUR_SECONDS*1e6<<","<<deltaTime.netAssimilation/HOUR_SECONDS*1e6<<","<< plant.getLAICanopy()<< "," << maxIterationNumber <<"\n";
        myFile << plant.getLAICanopy()<< "," <<deltaTime.grossAssimilation/HOUR_SECONDS*1e6 <<","<<deltaTime.respiration/HOUR_SECONDS*1e6<<","<<deltaTime.netAssimilation/HOUR_SECONDS*1e6<<","<< weatherVariable.myInstantTemp << "," << weatherVariable.prec <<"\n";
        myFile.close();
    }
    deltaTime.netAssimilation = deltaTime.netAssimilation*12/1000.0; // [KgC m-2] TODO da motiplicare dopo per CARBONFACTOR DA METTERE dopo convert to kg DM m-2
    deltaTime.understoreyNetAssimilation = HOUR_SECONDS * MH2O * understoreyAssimilationRate - MH2O*understoreyRespiration();
    statePlant.treeNetPrimaryProduction += deltaTime.netAssimilation; // state plant considers the biomass stored during the current year
    statePlant.understoreyNetPrimaryProduction += deltaTime.understoreyNetAssimilation; // [KgC m-2]
    //understorey


    deltaTime.transpiration = 0.;
    totalTranspirationRate = 0;
    for (int i=1; i < soil.layersNr; i++)
    {
        totalTranspirationRate += treeTranspirationRate[i];
        treeTranspirationRate[i] *= (HOUR_SECONDS * MH2O); // [mm]
        understoreyTranspirationRate[i] *= (HOUR_SECONDS * MH2O); // [mm]
        deltaTime.transpiration += (treeTranspirationRate[i] + understoreyTranspirationRate[i]);
    }

    //updateCriticalPsi();

    return;

    //evaporation
    //deltaTime.evaporation = computeEvaporation(); // TODO chiedere a Fausto come gestire l'evaporazione sui layer.

}

void Crit3DHydrall::updateCriticalPsi()
{
    double averageSoilWaterPotential = statistics::weighedMean(soil.nodeThickness,soil.waterPotential) * 0.009804139432; // Converted to MPa
    cavitationConditions();
    //double lastTerm = (totalTranspirationRate/plant.getLAICanopy() * MH2O/1000. * plant.hydraulicResistancePerFoliageArea);
    plant.psiLeaf = averageSoilWaterPotential - (plant.height * 0.009804139432 ) - (totalTranspirationRate/plant.getLAICanopy() * MH2O/1000. * plant.hydraulicResistancePerFoliageArea);

    if (plant.psiLeaf < plant.psiLeafMinimum || isEqual(plant.psiLeafMinimum,NODATA))
    {
        plant.psiLeafMinimum = plant.psiLeaf;
        plant.psiSoilCritical = averageSoilWaterPotential;
        plant.transpirationCritical = totalTranspirationRate / plant.getLAICanopy();
    }

}

double Crit3DHydrall::cavitationConditions()
{
    std::vector <std::vector <double>> conductivityWeights(2, std::vector<double>(soil.layersNr, NODATA));
    for (int i=0; i<soil.layersNr; i++)
    {
        // it is ok to start with 0 because the weights of the first layer will be anyhow 0
        conductivityWeights[0][i] = soil.getRootDensity()[i];
        conductivityWeights[1][i] = soil.nodeThickness[i];
    }


    double ksl = statistics::weighedMeanMultifactor(logarithmic10Values,conductivityWeights,soil.satHydraulicConductivity);
    ksl /= (1.625*RHOS*RADRT*RADRT);
    double soilRootsSpecificConductivity = 1/(1/KR + 1/ksl);
    soilRootsSpecificConductivity *= 0.5151 + MAXVALUE(0,0.0242*soil.temperature);
    //new sapwood specific conductivity
    double sapwoodSpecificConductivity = KSMAX * (1-std::exp(-0.69315*plant.height/H50)); //adjust for height effects
    sapwoodSpecificConductivity *= 0.5151 + MAXVALUE(0,0.0242*weatherVariable.meanDailyTemp);

    /*double a = 1./(statePlant.treeBiomassRoot*soilRootsSpecificConductivity);
    double b = (plant.height*plant.height*plant.woodDensity)/(statePlant.treeBiomassSapwood*sapwoodSpecificConductivity);
    double c = statePlant.treeBiomassFoliage*plant.specificLeafArea;*/
    //resulting leaf specific resistance (MPa s m2 m-3)
    plant.hydraulicResistancePerFoliageArea = (1./(statePlant.treeBiomassRoot*soilRootsSpecificConductivity)
                                               + (plant.height*plant.height*plant.woodDensity)/(statePlant.treeBiomassSapwood*sapwoodSpecificConductivity))
                                              * (statePlant.treeBiomassFoliage*plant.specificLeafArea);

    //return coefficient only when called in rootfind
    return std::sqrt(soilRootsSpecificConductivity/sapwoodSpecificConductivity*plant.sapwoodLongevity/plant.fineRootLongevity*plant.woodDensity);
}

double Crit3DHydrall::computeEvaporation()
{
    return weatherVariable.derived.et0 * MAXVALUE(0.2, 1 - 0.8*((understorey.leafAreaIndex + plant.getLAICanopy())/4)); //-0.8 / LAIMAX *ETP;
}

double Crit3DHydrall::understoreyRespiration()
{
    double understorey10DegRespirationFoliage;
    double understorey10DegRespirationFineroot;
    double correctionFactorFoliage;
    double correctionFactorFineroot;
    std::vector<double> correctionFactorFoliageVector;
    std::vector<double> correctionFactorFinerootVector;

    /*if(firstMonthVegetativeSeason) //maca inizializzazione di questo
    {
        understorey10DegRespirationFoliage = 0.0106/2. * (understoreyBiomass.leaf * nitrogenContent.leaf);
        understorey10DegRespirationFineroot= 0.0106/2. * (understoreyBiomass.fineRoot * nitrogenContent.root);
    }*/

    understorey10DegRespirationFoliage = 0.0106/2. * (understoreyBiomass.leaf * nitrogenContent.leaf);
    understorey10DegRespirationFineroot= 0.0106/2. * (understoreyBiomass.fineRoot * nitrogenContent.root);

    //double understoreyRespirationFoliage,understoreyRespirationFineroot;
    //understoreyRespirationFoliage = understorey10DegRespirationFoliage*;
    //const double PSIS0 = -2;// MPa * 101.972; //(m)
    //const double K_VW= 1.5;
    const double A_Q10= 0.503;
    const double B_Q10= 1.619;
    //double PENTRY,BSL,RVW_0,RVW_50,SIGMAG,Q10;
    double Q10;
    double VWCORR;
    double RVWSL;
    std::vector<double> nodeThicknessRealSoil(soil.layersNr-1);
    for (int iLayer = 1; iLayer<soil.layersNr; iLayer++)
    {
        nodeThicknessRealSoil[iLayer-1] = soil.nodeThickness[iLayer];
        //PENTRY = std::sqrt(std::exp(soil.clay[iLayer]*std::log(0.001) + soil.silt[iLayer]*std::log(0.026) + soil.sand[iLayer]*std::log(1.025)));
        //PENTRY = -0.5 / PENTRY / 1000;
        //SIGMAG =std::exp(std::sqrt(soil.clay[iLayer]*POWER2(std::log(0.001)) + soil.silt[iLayer]*POWER2(std::log(0.026)) + soil.sand[iLayer]*POWER2(std::log(1.025))));
        //BSL = -2*PENTRY*1000 + 0.2*SIGMAG;
        //PENTRY *= (std::pow(soil.bulkDensity[iLayer]/1.3,0.67*BSL));
        //RVW_0= std::pow((PSIS0/PENTRY),(-1/BSL)); // soil water content for null respiration
        //RVW_50= RVW_0 + (1.-RVW_0)/K_VW; //
        RVWSL= soil.waterContent[iLayer]/ soil.saturation[iLayer];//relative soil water content (as a fraction of value at saturation)
        VWCORR = moistureCorrectionFactor(iLayer);
        Q10= A_Q10 + B_Q10 * RVWSL; // effects of soil humidity on sensitivity to temperature
        correctionFactorFoliageVector.push_back(VWCORR * std::pow(Q10,((weatherVariable.myInstantTemp-25)/10.))); //temperature dependence of respiration, based on Q10 approach
        correctionFactorFinerootVector.push_back(VWCORR * std::pow(Q10,((soil.temperature-25)/10.)));
    }

    correctionFactorFoliage = statistics::weighedMean(nodeThicknessRealSoil,correctionFactorFoliageVector);
    correctionFactorFineroot = statistics::weighedMean(soil.nodeThickness,correctionFactorFinerootVector);
    return (understorey10DegRespirationFoliage * correctionFactorFoliage + understorey10DegRespirationFineroot * correctionFactorFineroot);
}

double Crit3DHydrall::plantRespiration()
{
    // taken from Hydrall Model, Magnani UNIBO
    double leafRespiration,rootRespiration,sapwoodRespiration;
    double totalRespiration;
    nitrogenContent.leaf = 0.02;    //[kg kgDM-1]
    nitrogenContent.root = 0.0078;  //[kg kgDM-1]
    nitrogenContent.stem = 0.0021;  //[kg kgDM-1]

    // Compute stand respiration rate at 10 oC (mol m-2 s-1)
    leafRespiration = RESPIRATION_PARAMETER * (statePlant.treeBiomassFoliage * nitrogenContent.leaf/0.014);
    sapwoodRespiration = RESPIRATION_PARAMETER * (statePlant.treeBiomassSapwood * nitrogenContent.stem/0.014);
    rootRespiration = RESPIRATION_PARAMETER * (statePlant.treeBiomassRoot * nitrogenContent.root/0.014);

    //calcolo temperatureMoistureFactor che deve passare per media del moisture ?
    double temperatureFactor = Crit3DHydrall::temperatureFunction(weatherVariable.myInstantTemp + ZEROCELSIUS);

    std::vector<double> moistureFactorVector(soil.layersNr-1);
    std::vector<double> nodeThicknessRealSoil(soil.layersNr-1);
    for (int i = 1; i < soil.layersNr; i++)
    {
        moistureFactorVector[i-1] = moistureCorrectionFactor(i);
        nodeThicknessRealSoil[i-1] = soil.nodeThickness[i];
    }
    double moistureFactor = statistics::weighedMean(soil.nodeThickness, moistureFactorVector);


    // Adjust for temperature effects
    leafRespiration *= BOUNDFUNCTION(0,1,temperatureFactor*moistureFactor) ;
    sapwoodRespiration *= BOUNDFUNCTION(0,1,temperatureFactor*moistureFactor);
    //shootRespiration *= MAXVALUE(0,MINVALUE(1,Vine3D_Grapevine::temperatureMoistureFunction(myInstantTemp + ZEROCELSIUS))) ;
    std::vector<std::vector<double>> weights;
    weights.push_back(soil.nodeThickness);
    weights.push_back(soil.getRootDensity());

    soil.temperature = Crit3DHydrall::soilTemperatureModel();
    moistureFactor = statistics::weighedMeanMultifactor(linearValues, weights, moistureFactorVector);
    //rootRespiration *= MAXVALUE(0,MINVALUE(1,Crit3DHydrall::temperatureMoistureFunction(soil.temperature + ZEROCELSIUS))) ;
    rootRespiration *= BOUNDFUNCTION(0,1,temperatureFactor*moistureFactor);
    // canopy respiration (sapwood+fine roots)
    totalRespiration =(leafRespiration + sapwoodRespiration + rootRespiration);


    //TODO understorey respiration

    return totalRespiration;
}

inline double Crit3DHydrall::soilTemperatureModel()
{
    // taken from Hydrall Model, Magnani UNIBO
    return 0.8 * weatherVariable.last30DaysTAvg + 0.2 * weatherVariable.myInstantTemp;
}

double Crit3DHydrall::moistureCorrectionFactorOld(int index)
{
    double correctionSoilMoisture = 1;
    double stressThreshold = 0.5*(soil.saturation[index]+soil.fieldCapacity[index]);
    if (soil.waterContent[index] <= soil.fieldCapacity[index])
    {
        correctionSoilMoisture = log(soil.wiltingPoint[index]/soil.waterContent[index]) / log(soil.wiltingPoint[index]/soil.fieldCapacity[index]);
    }
    else if(soil.waterContent[index] > stressThreshold)
    {
        correctionSoilMoisture = log(soil.saturation[index]/soil.waterContent[index]) / log(soil.saturation[index]/stressThreshold);
    }
    return correctionSoilMoisture;
}

double Crit3DHydrall::moistureCorrectionFactor(int index)
{
    /*if (soil.waterContent[index] < soil.fieldCapacity[index])
    {
        return MINVALUE(1, 10./3. * (soil.waterContent[index]-soil.wiltingPoint[index])/(soil.fieldCapacity[index]- soil.wiltingPoint[index]));
    }
    else
    {
        return MINVALUE(1, (soil.waterContent[index] - 2* soil.fieldCapacity[index])/(soil.saturation[index]-2 * soil.fieldCapacity[index]));
    }*/

    return BOUNDFUNCTION(0.01,1, MINVALUE((10*(soil.waterContent[index]-soil.wiltingPoint[index]))/(3*(soil.fieldCapacity[index]-soil.wiltingPoint[index])),
                                           log(soil.saturation[index]/soil.waterContent[index])/log(2*soil.saturation[index]/(soil.fieldCapacity[index]+soil.saturation[index]))));;
}

double Crit3DHydrall::temperatureFunction(double temperature)
{
    // taken from Hydrall Model, Magnani UNIBO
    double temperatureMoistureFactor = 1;
    // TODO
    int MODEL = 2;
    //1. AP_H model
    if (MODEL == 1)
        temperatureMoistureFactor = pow(2.0,((temperature - parameterWangLeuning.optimalTemperatureForPhotosynthesis)/10.0)); // temperature dependence of respiration, based on Q10 approach
    else if (MODEL == 2)
        temperatureMoistureFactor= exp(308.56 * (1.0/(parameterWangLeuning.optimalTemperatureForPhotosynthesis + 46.02) - 1.0/(temperature+46.02)));  //temperature dependence of respiration, based on Lloyd & Taylor (1994)


    /*int   MODEL;
    double temperatureMoistureFactor,correctionSoilMoisture; //K_VW
    //K_VW= 1.5;
    temperatureMoistureFactor = 1. ;
    MODEL = 2;
    //T = climate.instantT;
    //1. AP_H model
    if (MODEL == 1) {

        if(psiSoilAverage >= psiFieldCapacityAverage)
        {
            correctionSoilMoisture = 1.0; //effects of soil water potential
        }
        else if (psiSoilAverage <= wiltingPoint)
        {
            correctionSoilMoisture = 0.0;
        }
        else
        {
            correctionSoilMoisture = log(wiltingPoint/psiSoilAverage) / log(wiltingPoint/psiFieldCapacityAverage);
        }
        temperatureMoistureFactor = pow(2.0,((temperature - parameterWangLeuningFix.optimalTemperatureForPhotosynthesis)/10.0)); // temperature dependence of respiration, based on Q10 approach
        temperatureMoistureFactor *= correctionSoilMoisture;
    }
    // 2. AP_LT model
    else if (MODEL == 2){
        //effects of soil water potential
        if (psiSoilAverage >= psiFieldCapacityAverage)
        {
            correctionSoilMoisture = 1.0;
        }
        else if (psiSoilAverage <= wiltingPoint)
        {
            correctionSoilMoisture = 0.0;
        }
        else
        {
            correctionSoilMoisture = log(wiltingPoint/psiSoilAverage) / log(wiltingPoint/psiFieldCapacityAverage);
        }
        temperatureMoistureFactor= exp(308.56 * (1.0/(parameterWangLeuningFix.optimalTemperatureForPhotosynthesis + 46.02) - 1.0/(temperature+46.02)));  //temperature dependence of respiration, based on Lloyd & Taylor (1994)
        temperatureMoistureFactor *= correctionSoilMoisture;
    }*/
    return temperatureMoistureFactor;
}

bool Crit3DHydrall::simplifiedGrowthStand()
{
    const double understoreyAllocationCoefficientToRoot = 0.5;
    // understorey update TODO IMPORTANTE: SERVE CARBONFACTOR ANCHE QUI?
    statePlant.understoreyBiomassFoliage = statePlant.understoreyNetPrimaryProduction * (1.-understoreyAllocationCoefficientToRoot) / CARBONFACTOR;    //understorey growth: foliage...
    statePlant.understoreyBiomassRoot = statePlant.understoreyNetPrimaryProduction * understoreyAllocationCoefficientToRoot / CARBONFACTOR;         //...and roots

    //outputC calculation for RothC model. necessario [t C/ha] ora in kgDM m-2
    //natural death
    outputC = statePlant.treeBiomassFoliage/plant.foliageLongevity + statePlant.treeBiomassSapwood/plant.sapwoodLongevity +
              statePlant.treeBiomassRoot/plant.fineRootLongevity * CARBONFACTOR * 10;

    statePlant.treeBiomassFoliage -= (statePlant.treeBiomassFoliage/plant.foliageLongevity);
    statePlant.treeBiomassSapwood -= (statePlant.treeBiomassSapwood/plant.sapwoodLongevity);
    statePlant.treeBiomassRoot -= (statePlant.treeBiomassRoot/plant.fineRootLongevity);

    //distributed wildfire loss
    double distributedWildfireLoss = getFirewoodLostSurfacePercentage(0.02, year); //TODO: this parameter must be able to vary based on what if scenario
    statePlant.treeBiomassFoliage -= statePlant.treeBiomassFoliage * distributedWildfireLoss * 1; //foliage is completely lost in the event of a wildfire
    outputC += statePlant.treeBiomassRoot * distributedWildfireLoss * 1; //roots are preserved but dead and become input for carbon model
    statePlant.treeBiomassRoot -= statePlant.treeBiomassRoot * distributedWildfireLoss * 1;
    outputC += statePlant.treeBiomassSapwood * distributedWildfireLoss * plant.wildfireDamage; //40% or 80% of sapwood is lost based on species
    statePlant.treeBiomassSapwood -= statePlant.treeBiomassSapwood * distributedWildfireLoss * plant.wildfireDamage;

    //woodland management
    plant.management = 1; //DEBUG //0 is non managed, 1 is coppice, 2 is high forest, 3 is plantation, 4 is urban forest
    double woodExtraction = 0;

    if (plant.management == 1) //coppice management produces mostly burning wood
    {
        woodExtraction = 1./30;
        outputC += statePlant.treeBiomassFoliage * woodExtraction; //foliage is left in the forest
    }
    else if (plant.management == 2) //high forest management produces wood that stocks carbon for 35 years (IPCC) and is saved as a regional value
    {
        woodExtraction = 1./100;
        carbonStock += statePlant.treeBiomassSapwood * woodExtraction * 1./35;
        outputC += statePlant.treeBiomassFoliage * woodExtraction; //foliage is left in the forest

    }
    else if (plant.management == 3)
    {
        //plantation
    }
    else if (plant.management == 4)
    {
        woodExtraction = 0.5;
    }

    //if plant.management == 0, woodExtraction = 0
    statePlant.treeBiomassSapwood -= statePlant.treeBiomassSapwood * woodExtraction;
    statePlant.treeBiomassFoliage -= statePlant.treeBiomassFoliage * woodExtraction;
    outputC += statePlant.treeBiomassRoot * woodExtraction; //dead roots become input for carbon model
    statePlant.treeBiomassRoot -= statePlant.treeBiomassRoot * woodExtraction;


    // TODO to understand what's internalCarbonStorage (STORE), afterwards the uninitialized value is used
    //annual stand growth
    if (isFirstYearSimulation)
    {
        annualGrossStandGrowth = statePlant.treeNetPrimaryProduction / CARBONFACTOR; //conversion to kg DM m-2
        internalCarbonStorage = 0;
    }
    else
    {
        annualGrossStandGrowth = (internalCarbonStorage + statePlant.treeNetPrimaryProduction) / 2 / CARBONFACTOR;
        internalCarbonStorage = (internalCarbonStorage + statePlant.treeNetPrimaryProduction) / 2;
    }

    //if (isFirstYearSimulation) è necessario?

    //computing root/shoot ratio based on values found in [Vitullo et al. 2007] and modified to account for water scarcity
    double alpha = 0.7;
    double rootShootRatio = MAXVALUE(MINVALUE(plant.rootShootRatioRef*(alpha*0.5 + 1), plant.rootShootRatioRef*(alpha*(1-weatherVariable.getYearlyPrec()/weatherVariable.getYearlyET0())+1)), plant.rootShootRatioRef);

    allocationCoefficient.toFineRoots = rootShootRatio / (1 + rootShootRatio);
    allocationCoefficient.toFoliage = ( 1 - allocationCoefficient.toFineRoots ) * 0.05;
    allocationCoefficient.toSapwood = 1 - allocationCoefficient.toFineRoots - allocationCoefficient.toFoliage;

    /*if (printHourlyRecords)
    {
        std::ofstream myFile;
        myFile.open("outputAlloc.csv", std::ios_base::app);
        myFile << allocationCoefficient.toFoliage <<","<< allocationCoefficient.toFineRoots <<","<<allocationCoefficient.toSapwood <<","
               << rootShootRatio <<"," << weatherVariable.getYearlyET0() << "," << weatherVariable.getYearlyPrec() <<"\n";
        myFile.close();
    }*/

    if (annualGrossStandGrowth * allocationCoefficient.toFoliage > statePlant.treeBiomassFoliage/(plant.foliageLongevity - 1))
    {
        statePlant.treeBiomassFoliage = MAXVALUE(statePlant.treeBiomassFoliage + annualGrossStandGrowth * allocationCoefficient.toFoliage, EPSILON);
        statePlant.treeBiomassRoot = MAXVALUE(statePlant.treeBiomassRoot + annualGrossStandGrowth * allocationCoefficient.toFineRoots, EPSILON);
        statePlant.treeBiomassSapwood = MAXVALUE(statePlant.treeBiomassSapwood + annualGrossStandGrowth * allocationCoefficient.toSapwood, EPSILON);
    }
    // TODO manca il computo del volume sia generale che incrementale vedi funzione grstand.for


    isFirstYearSimulation = false;
    return true;
}

bool Crit3DHydrall::growthStand()
{
    const double understoreyAllocationCoefficientToRoot = 0.5;
    // understorey update TODO IMPORTANTE: SERVE CARBONFACTOR ANCHE QUI?
    statePlant.understoreyBiomassFoliage = statePlant.understoreyNetPrimaryProduction * (1.-understoreyAllocationCoefficientToRoot);    //understorey growth: foliage...
    statePlant.understoreyBiomassRoot = statePlant.understoreyNetPrimaryProduction * understoreyAllocationCoefficientToRoot;         //...and roots

    //outputC calculation for RothC model. necessario [t C/ha] ora in kgDM m-2
    //MANCA OUTPUT DA TAGLIO
    outputC = statePlant.treeBiomassFoliage/plant.foliageLongevity + statePlant.treeBiomassSapwood/plant.sapwoodLongevity +
              statePlant.treeBiomassRoot/plant.fineRootLongevity /CARBONFACTOR * 1e5;

    // canopy update
    statePlant.treeBiomassFoliage -= (statePlant.treeBiomassFoliage/plant.foliageLongevity);
    statePlant.treeBiomassSapwood -= (statePlant.treeBiomassSapwood/plant.sapwoodLongevity);
    statePlant.treeBiomassRoot -= (statePlant.treeBiomassRoot/plant.fineRootLongevity);


    // TODO to understand what's internalCarbonStorage (STORE), afterwards the uninitialized value is used
    //annual stand growth
    if (isFirstYearSimulation)
    {
        annualGrossStandGrowth = statePlant.treeNetPrimaryProduction / CARBONFACTOR; //conversion to kg DM m-2
        internalCarbonStorage = 0;
    }
    else
    {
        annualGrossStandGrowth = (internalCarbonStorage + statePlant.treeNetPrimaryProduction) / 2 / CARBONFACTOR;
        internalCarbonStorage = (internalCarbonStorage + statePlant.treeNetPrimaryProduction) / 2;
    }

    if (isFirstYearSimulation)
    {
        optimal();
    }
    else
    {
        double allocationCoeffientFoliageOld = allocationCoefficient.toFoliage;
        double allocationCoeffientFineRootsOld = allocationCoefficient.toFineRoots;
        double allocationCoeffientSapwoodOld = allocationCoefficient.toSapwood;

        optimal();

        allocationCoefficient.toFoliage = (allocationCoeffientFoliageOld + allocationCoefficient.toFoliage) / 2;
        allocationCoefficient.toFineRoots = (allocationCoeffientFineRootsOld + allocationCoefficient.toFineRoots) / 2;
        allocationCoefficient.toSapwood = (allocationCoeffientSapwoodOld + allocationCoefficient.toSapwood) / 2;
    }

    std::ofstream myFile;
    myFile.open("outputAlloc.csv", std::ios_base::app);
    myFile << allocationCoefficient.toFoliage <<","<< allocationCoefficient.toFineRoots <<","<<allocationCoefficient.toSapwood <<"\n";
    myFile.close();

    if (annualGrossStandGrowth * allocationCoefficient.toFoliage > statePlant.treeBiomassFoliage/(plant.foliageLongevity - 1))
    {
        statePlant.treeBiomassFoliage = MAXVALUE(statePlant.treeBiomassFoliage + annualGrossStandGrowth * allocationCoefficient.toFoliage, EPSILON);
        statePlant.treeBiomassRoot = MAXVALUE(statePlant.treeBiomassRoot + annualGrossStandGrowth * allocationCoefficient.toFineRoots, EPSILON);
        statePlant.treeBiomassSapwood = MAXVALUE(statePlant.treeBiomassSapwood + annualGrossStandGrowth * allocationCoefficient.toSapwood, EPSILON);
    }
    // TODO manca il computo del volume sia generale che incrementale vedi funzione grstand.for


    isFirstYearSimulation = false;
    return true;
}


void Crit3DHydrall::resetStandVariables()
{
    statePlant.treeNetPrimaryProduction = 0;
}

void Crit3DHydrall::optimal()
{
    double allocationCoefficientFoliageOld;
    double increment;
    double incrementStart = 5e-2;
    bool sol = false;
    double allocationCoefficientFoliage0;
    double bisectionMethodIntervalALLF;
    int jmax = 40;
    double accuracy = 1e-3;

    for (int j = 0; j < 3; j++)
    {
        allocationCoefficientFoliageOld = 1;
        increment = incrementStart / std::pow(10, j);

        for (allocationCoefficient.toFoliage = 1; allocationCoefficient.toFoliage > 0; allocationCoefficient.toFoliage -= increment)
        {
            rootfind(allocationCoefficient.toFoliage, allocationCoefficient.toFineRoots, allocationCoefficient.toSapwood, sol);

            if (sol)
                break;

            allocationCoefficientFoliageOld = allocationCoefficient.toFoliage;
        }
        if (sol)
            break;
    }

    if (sol)
    {
        //find optimal allocation coefficients by bisection technique

        double allocationCoefficientFoliageMid,  allocationCoefficientFineRootsMid, allocationCoefficientSapwoodMid;
        bool solmid = 0;

        //set starting point and range
        allocationCoefficientFoliage0 = allocationCoefficient.toFoliage;
        bisectionMethodIntervalALLF = allocationCoefficientFoliageOld - allocationCoefficient.toFoliage;

        //bisection loop
        int j = 0;
        while (std::abs(bisectionMethodIntervalALLF) > accuracy && j < jmax)
        {
            bisectionMethodIntervalALLF /= 2;
            allocationCoefficientFoliageMid = allocationCoefficientFoliage0 + bisectionMethodIntervalALLF;

            rootfind(allocationCoefficientFoliageMid, allocationCoefficientFineRootsMid, allocationCoefficientSapwoodMid, solmid);

            if (solmid)
            {
                allocationCoefficientFoliage0 = allocationCoefficientFoliageMid;
                allocationCoefficient.toFoliage = allocationCoefficientFoliageMid;
                allocationCoefficient.toFineRoots = allocationCoefficientFineRootsMid;
                allocationCoefficient.toSapwood = allocationCoefficientSapwoodMid;
            }

            j++;
        }
    }
}

void Crit3DHydrall::rootfind(double &allf, double &allr, double &alls, bool &sol)
{
    //search for a solution to hydraulic constraint

    //new foliage biomass of tree after growth
    allf = MAXVALUE(0,allf);
    //if (allf < 0) allf = 0;
    // TODO verificare le unità di misura di rootfind con la routine originale di hydrall c'è un fattore 1000
    statePlant.treeBiomassFoliage += (allf*annualGrossStandGrowth);

    //new tree-height after growth
    if (allf*annualGrossStandGrowth > statePlant.treeBiomassFoliage/(plant.foliageLongevity-1)) {
        plant.height += (allf*annualGrossStandGrowth-statePlant.treeBiomassFoliage/(plant.foliageLongevity-1))/plant.foliageDensity;
    }

    //soil hydraulic conductivity
    //optimal coefficient of allocation to fine roots and sapwood for set allocation to foliage
    //cavitationConditions does other calculations and returns quadraticEqCoefficient
    double quadraticEqCoefficient = cavitationConditions();
    allr = (statePlant.treeBiomassSapwood - quadraticEqCoefficient*plant.height*statePlant.treeBiomassRoot +
            annualGrossStandGrowth*(1-allf))/annualGrossStandGrowth/(1+quadraticEqCoefficient*plant.height);

    //allr = MAXVALUE(EPSILON,allr); //bracket ALLR between (1-ALLF) and a small value
    //allr = MINVALUE(MAXVALUE(EPSILON,allr),1-allf);
    allr = BOUNDFUNCTION(EPSILON,1-allf,allr); // TODO to be checked
    //alls = 1 - allf - allr;
    // alls = MAXVALUE(alls,EPSILON); //bracket ALLS between 1 and a small value
    //alls = MINVALUE(1,MAXVALUE(alls,EPSILON));
    alls = BOUNDFUNCTION(EPSILON,1,1-allf-allr); // TODO to be checked
    //resulting fine root and sapwood biomass
    statePlant.treeBiomassRoot += allr * annualGrossStandGrowth;
    statePlant.treeBiomassRoot = MAXVALUE(EPSILON,statePlant.treeBiomassRoot);
    statePlant.treeBiomassSapwood += alls * annualGrossStandGrowth;
    statePlant.treeBiomassSapwood = MAXVALUE(EPSILON,statePlant.treeBiomassSapwood);

    //resulting minimum leaf water potential

    //plant.psiSoilCritical = PSITHR - 0.5;
    plant.psiLeafMinimum = plant.psiSoilCritical - (plant.height)* 0.009804139432 -(plant.transpirationCritical * MH2O/1000. * plant.hydraulicResistancePerFoliageArea);
    //plant.psiLeafMinimum = plant.psiSoilCritical - (0.01 * plant.height * 9.806650e-5)-(plant.transpirationCritical * MH2O/1000. * plant.hydraulicResistancePerFoliageArea);
    //check if given value of ALLF satisfies optimality constraint
    if(plant.psiLeafMinimum >= PSITHR)
        sol = true;
    else
    {
        sol = false;
        allr = statePlant.treeBiomassSapwood
                +  annualGrossStandGrowth -statePlant.treeBiomassRoot*quadraticEqCoefficient*plant.height;
        allr /= (annualGrossStandGrowth * (1.+quadraticEqCoefficient*plant.height));
        allr = BOUNDFUNCTION(EPSILON,1,allr); // TODO to be checked
        alls = 1.-allr;
        //allf = 0; // TODO verify its value
    }

}

double Crit3DHydrall::getFirewoodLostSurfacePercentage(double percentageSurfaceLostByFirewoodAtReferenceYear, int simulationYear)
{
    //TODO check if percentage or ratio
    // for Emilia-Romagna region set percentageSurfaceLostByFirewoodAtReferenceYear=0.002
    // the function is based on the ISPRA projection
    double hazard;
    int dimTable=3;
    double *firstColumn = (double *) calloc(dimTable, sizeof(double));
    double *secondColumn = (double *) calloc(dimTable, sizeof(double));
    firstColumn[0] = 2020.;
    firstColumn[1] = 2040.;
    firstColumn[2] = 2050.;
    secondColumn[0] = 1.;
    secondColumn[1] = 22./19.;
    secondColumn[2] = 28./19.;
    hazard = interpolation::linearInterpolation(double(simulationYear),firstColumn,secondColumn,dimTable);
    free(firstColumn);
    free(secondColumn);
    return hazard*percentageSurfaceLostByFirewoodAtReferenceYear;
}
