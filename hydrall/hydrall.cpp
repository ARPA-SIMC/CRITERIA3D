/*!
    \name hydrall.cpp
    \brief
    \authors Antonio Volta, Caterina Toscano
    \email avolta@arpae.it ctoscano@arpae.it

*/


//#include <stdio.h>
#include <math.h>
#include "crit3dDate.h"
#include "commonConstants.h"
#include "hydrall.h"
#include "furtherMathFunctions.h"
#include "basicMath.h"
#include "physics.h"
#include "statistics.h"

Crit3DHydrallMaps::Crit3DHydrallMaps()
{
    mapLAI = new gis::Crit3DRasterGrid;
    standBiomassMap = new gis::Crit3DRasterGrid;
    rootBiomassMap = new gis::Crit3DRasterGrid;
    mapLast30DaysTavg = new gis::Crit3DRasterGrid;
}

void Crit3DHydrallMaps::initialize(const gis::Crit3DRasterGrid& DEM)
{
    mapLAI->initializeGrid(DEM);
    standBiomassMap->initializeGrid(DEM);
    rootBiomassMap->initializeGrid(DEM);
    mapLast30DaysTavg->initializeGrid(DEM);
    treeSpeciesMap.initializeGrid(DEM);
    plantHeight.initializeGrid(DEM);
}

Crit3DHydrallMaps::~Crit3DHydrallMaps()
{
    mapLAI->clear();
    standBiomassMap->clear();
    rootBiomassMap->clear();
    mapLast30DaysTavg->clear();
}

bool Crit3D_Hydrall::computeHydrallPoint(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep)
{
    //getCO2(myDate, myTemperature, myElevation);


    // da qui in poi bisogna fare un ciclo su tutte le righe e le colonne
    plant.leafAreaIndexCanopyMax = statePlant.treecumulatedBiomassFoliage *  plant.specificLeafArea / cover;
    plant.leafAreaIndexCanopy = MAXVALUE(5,plant.leafAreaIndexCanopyMax * computeLAI(myDate));
    understoreyLeafAreaIndexMax = statePlant.understoreycumulatedBiomassFoliage * plant.specificLeafArea;
    understorey.leafAreaIndex = MAXVALUE(LAIMIN,understoreyLeafAreaIndexMax* computeLAI(myDate));

    Crit3D_Hydrall::photosynthesisAndTranspiration();

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

double Crit3D_Hydrall::getCO2(Crit3DDate myDate, double myTemperature)
{
    double atmCO2 = 400 ; //https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5/download.table
    double year[24] = {1750,1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100};
    double valueCO2[24] = {278,283,285,296,300,303,307,310,311,317,325,339,354,369,389,413,443,473,503,530,550,565,570,575};

    atmCO2 = interpolation::linearInterpolation(double(myDate.year), year, valueCO2, 24);


    atmCO2 += 3*cos(2*PI*getDoyFromDate(myDate)/365.0);		     // to consider the seasonal effects
    return atmCO2 * weatherVariable.atmosphericPressure/1000000;   // [Pa] in +- ppm/10 formula changed from the original Hydrall
}
/*
double Crit3D_Hydrall::getPressureFromElevation(double myTemperature, double myElevation)
{
    return P0 * exp((- GRAVITY * M_AIR * myElevation) / (R_GAS * myTemperature));
}
*/
double Crit3D_Hydrall::computeLAI(Crit3DDate myDate)
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

double Crit3D_Hydrall::photosynthesisAndTranspiration()
{
    TweatherDerivedVariable weatherDerivedVariable;

    Crit3D_Hydrall::radiationAbsorption();
    Crit3D_Hydrall::aerodynamicalCoupling();
    //Crit3D_Hydrall::upscale();

    //Crit3D_Hydrall::carbonWaterFluxesProfile();
    //Crit3D_Hydrall::cumulatedResults();

    return 0;
}

double Crit3D_Hydrall::photosynthesisAndTranspirationUnderstorey()
{
    const double rootEfficiencyInWaterExtraction = 1.25e-3;  //[kgH2O kgDM-1 s-1]
    const double understoreyLightUtilization = 1.77e-9;      //[kgC J-1]
    double cumulatedUnderstoreyTranspirationRate = 0;
    double waterUseEfficiency;                               //[molC molH2O-1]

    if (understorey.absorbedPAR > EPSILON)
    {
        waterUseEfficiency = environmentalVariable.CO2 * 0.1875 / weatherVariable.vaporPressureDeficit;

        double lightLimitedUnderstoreyAssimilation;          //[molC m-2 s-1]
        double waterLimitedUnderstoreyAssimilation;          //[molC m-2 s-1]

        lightLimitedUnderstoreyAssimilation = understoreyLightUtilization * understorey.absorbedPAR / MC; //convert units from kgC m-2 s-1 into molC m-2 s-1

        for (int i = 0; i < soil.layersNr; i++)
        {
            understoreyTranspirationRate[i] = rootEfficiencyInWaterExtraction * understoreyBiomass.fineRoot * soil.stressCoefficient[i];
            cumulatedUnderstoreyTranspirationRate += understoreyTranspirationRate[i];
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
            for (int j = 0; j < soil.layersNr; j++)
            {
                understoreyTranspirationRate[j] *= lightWaterRatio;
            }
        }
    }
    else
    {
        for (int i = 0; i < understoreyTranspirationRate.size(); i++)
            understoreyTranspirationRate[i] = 0;
        understoreyAssimilationRate = 0;
    }
    return 0;
}


void Crit3D_Hydrall::initialize()
{
    plant.myChlorophyllContent = NODATA;
    elevation = NODATA;
}

void Crit3D_Hydrall::setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure, double CO2, double sunElevation)
{
    setWeatherVariables(temp, irradiance, prec, relativeHumidity, windSpeed, directIrradiance, diffuseIrradiance, cloudIndex, atmosphericPressure);
    environmentalVariable.CO2 = CO2;
    environmentalVariable.sineSolarElevation = MAXVALUE(0.0001,sin(sunElevation*DEG_TO_RAD));
}

bool Crit3D_Hydrall::setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex, double atmosphericPressure)
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


    setDerivedWeatherVariables(directIrradiance, diffuseIrradiance, cloudIndex);

    if ((int(prec) != NODATA) && (int(temp) != NODATA) && (int(windSpeed) != NODATA)
        && (int(irradiance) != NODATA) && (int(relativeHumidity) != NODATA)
        && (int(atmosphericPressure) != NODATA))
        isReadingOK = true;

    return isReadingOK;
}

void Crit3D_Hydrall::setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex)
{
    weatherVariable.derived.airVapourPressure = saturationVaporPressure(weatherVariable.myInstantTemp)*weatherVariable.relativeHumidity/100.;
    weatherVariable.derived.slopeSatVapPressureVSTemp = 2588464.2 / POWER2(240.97 + weatherVariable.myInstantTemp) * exp(17.502 * weatherVariable.myInstantTemp / (240.97 + weatherVariable.myInstantTemp)) ;
    weatherVariable.derived.myDirectIrradiance = directIrradiance;
    weatherVariable.derived.myDiffuseIrradiance = diffuseIrradiance;
    double myCloudiness = BOUNDFUNCTION(0,1,cloudIndex);
    weatherVariable.derived.myEmissivitySky = 1.24 * pow((weatherVariable.derived.airVapourPressure/100.0) / (weatherVariable.myInstantTemp+ZEROCELSIUS),(1.0/7.0))*(1 - 0.84*myCloudiness)+ 0.84*myCloudiness;
    weatherVariable.derived.myLongWaveIrradiance = POWER4(weatherVariable.myInstantTemp+ZEROCELSIUS) * weatherVariable.derived.myEmissivitySky * STEFAN_BOLTZMANN ;
    weatherVariable.derived.psychrometricConstant = psychro(weatherVariable.atmosphericPressure,weatherVariable.myInstantTemp);
    return;
}

void Crit3D_Hydrall::setPlantVariables(double chlorophyllContent, double height)
{
    plant.myChlorophyllContent = chlorophyllContent;
    plant.height = height;
}

void Crit3D_Hydrall::setStateVariables(Crit3DHydrallMaps &stateMap, int row, int col)
{
    stateVariable.standBiomass = stateMap.standBiomassMap->value[row][col];
    stateVariable.rootBiomass = stateMap.rootBiomassMap->value[row][col];
}

void Crit3D_Hydrall::setSoilVariables(int iLayer, int currentNode,float checkFlag, int horizonIndex, double waterContent, double waterContentFC, double waterContentWP, int firstRootLayer, int lastRootLayer, double rootDensity)
{
    if (iLayer == 0)
    {
        soil.layersNr = 1;
    }
    (soil.layersNr)++;
    soil.waterContent.resize(soil.layersNr);
    soil.stressCoefficient.resize(soil.layersNr);
    soil.rootDensity.resize(soil.layersNr);

    if (currentNode != checkFlag)
    {
        soil.waterContent[iLayer] = waterContent;
        soil.stressCoefficient[iLayer] = MINVALUE(1.0, (10*(soil.waterContent[iLayer]-waterContentWP))/(3*(waterContentFC-waterContentWP)));
    }
    soil.rootDensity[iLayer] = LOGICAL_IO((iLayer >= firstRootLayer && iLayer <= lastRootLayer),rootDensity,0);

}

void Crit3D_Hydrall::getStateVariables(Crit3DHydrallMaps &stateMap, int row, int col)
{
    stateMap.standBiomassMap->value[row][col] = stateVariable.standBiomass;
    stateMap.rootBiomassMap->value[row][col] = stateVariable.rootBiomass;
}

void Crit3D_Hydrall::radiationAbsorption()
{
    // taken from Hydrall Model, Magnani UNIBO

    // TODO chiedere a Magnani questi parametri
    static double   leafAbsorbanceNIR= 0.2;
    static double   hemisphericalIsotropyParameter = 0. ; // in order to change the hemispherical isotropy from -0.4 to 0.6 Wang & Leuning 1998
    static double   clumpingParameter = 1.0 ; // from 0 to 1 <1 for needles
    double  diffuseLightSector1K,diffuseLightSector2K,diffuseLightSector3K ;
    double scatteringCoefPAR, scatteringCoefNIR ;
    std::vector<double> dum(17, NODATA);
    double  sunlitAbsorbedNIR ,  shadedAbsorbedNIR, sunlitAbsorbedLW , shadedAbsorbedLW;
    double directIncomingPAR, directIncomingNIR , diffuseIncomingPAR , diffuseIncomingNIR,leafAbsorbancePAR;
    double directReflectionCoefficientPAR , directReflectionCoefficientNIR , diffuseReflectionCoefficientPAR , diffuseReflectionCoefficientNIR;
    //projection of the unit leaf area in the direction of the sun's beam, following Sellers 1985 (in Wang & Leuning 1998)

    directLightExtinctionCoefficient.global = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*environmentalVariable.sineSolarElevation) - POWER2(hemisphericalIsotropyParameter)*(0.33-0.579*environmentalVariable.sineSolarElevation))/ environmentalVariable.sineSolarElevation;

    /*Extinction coeff for canopy of black leaves, diffuse radiation
    The average extinctio coefficient is computed considering three sky sectors,
    assuming SOC conditions (Goudriaan & van Laar 1994, p 98-99)*/
    diffuseLightSector1K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.259) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.259))/0.259;  //projection of unit leaf for first sky sector (0-30 elevation)
    diffuseLightSector2K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.707) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.707))/0.707 ; //second sky sector (30-60 elevation)
    diffuseLightSector3K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.966) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.966))/ 0.966 ; // third sky sector (60-90 elevation)
    diffuseLightExtinctionCoefficient.global =- 1.0/plant.leafAreaIndexCanopy * log(0.178 * exp(-diffuseLightSector1K*plant.leafAreaIndexCanopy) + 0.514 * exp(-diffuseLightSector2K*plant.leafAreaIndexCanopy)
                                                                      + 0.308 * exp(-diffuseLightSector3K*plant.leafAreaIndexCanopy));  //approximation based on relative radiance from 3 sky sectors
    //Include effects of leaf clumping (see Goudriaan & van Laar 1994, p 110)
    directLightExtinctionCoefficient.global  *= clumpingParameter ;//direct light
    diffuseLightExtinctionCoefficient.global *= clumpingParameter ;//diffuse light
    if (environmentalVariable.sineSolarElevation > 0.001)
    {
        //Leaf area index of sunlit (1) and shaded (2) big-leaf
        sunlit.leafAreaIndex = UPSCALINGFUNC(directLightExtinctionCoefficient.global,plant.leafAreaIndexCanopy);
        shaded.leafAreaIndex = plant.leafAreaIndexCanopy - sunlit.leafAreaIndex ;
        understorey.leafAreaIndex = 0.2;
        //Extinction coefficients for direct and diffuse PAR and NIR radiation, scattering leaves
        //Based on approximation by Goudriaan 1977 (in Goudriaan & van Laar 1994)
        double exponent= -pow(10,0.28 + 0.63*log10(plant.myChlorophyllContent*0.85/1000));
        leafAbsorbancePAR = 1 - pow(10,exponent);//from Agusti et al (1994), Eq. 1, assuming Chl a = 0.85 Chl (a+b)
        scatteringCoefPAR = 1.0 - leafAbsorbancePAR ; //scattering coefficient for PAR
        scatteringCoefNIR = 1.0 - leafAbsorbanceNIR ; //scattering coefficient for NIR
        diffuseLightExtinctionCoefficient.par = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR) ;//extinction coeff of PAR, direct light
        diffuseLightExtinctionCoefficient.nir = diffuseLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, direct light
        directLightExtinctionCoefficient.par  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefPAR); //extinction coeff of PAR, diffuse light
        directLightExtinctionCoefficient.nir  = directLightExtinctionCoefficient.global * sqrt(1-scatteringCoefNIR); //extinction coeff of NIR radiation, diffuse light
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
        shaded.absorbedPAR = dum[5]*(UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy)- dum[11])
                             + dum[6]*(UPSCALINGFUNC(directLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy)- dum[12])
                             - dum[7] * dum[15];
        // NIR absorbed by sunlit (1) and shaded (2) big-leaf (W m-2) fromWang & Leuning 1998
        sunlitAbsorbedNIR = dum[8]*dum[13]+dum[9]*dum[14]+dum[10]*dum[15];
        shadedAbsorbedNIR = dum[8]*(UPSCALINGFUNC(diffuseLightExtinctionCoefficient.nir,plant.leafAreaIndexCanopy)-dum[13])+dum[9]*(UPSCALINGFUNC(directLightExtinctionCoefficient.nir,plant.leafAreaIndexCanopy)- dum[14]) - dum[10] * dum[15];

        double emissivityLeaf = 0.96 ; // supposed constant because variation is very small
        double emissivitySoil= 0.94 ;   // supposed constant because variation is very small
        sunlitAbsorbedLW = (dum[16] * UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy))*emissivityLeaf+(1.0-emissivitySoil)*(emissivityLeaf-weatherVariable.derived.myEmissivitySky)* UPSCALINGFUNC((2*diffuseLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy)* UPSCALINGFUNC((directLightExtinctionCoefficient.global-diffuseLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy);
        shadedAbsorbedLW = dum[16] * UPSCALINGFUNC(diffuseLightExtinctionCoefficient.global,plant.leafAreaIndexCanopy) - sunlitAbsorbedLW ;
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

        shaded.leafAreaIndex = plant.leafAreaIndexCanopy;
        shaded.absorbedPAR = 0.0 ;
        shadedAbsorbedNIR = 0.0 ;
        dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*pow(weatherVariable.myInstantTemp + ZEROCELSIUS,4) ;
        dum[16] *= diffuseLightExtinctionCoefficient.global ;
        shadedAbsorbedLW= dum[16] * (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.global,plant.leafAreaIndexCanopy) - UPSCALINGFUNC(directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.global,plant.leafAreaIndexCanopy)) ;
        shaded.isothermalNetRadiation = shaded.absorbedPAR + shadedAbsorbedNIR + shadedAbsorbedLW ;
    }

    // Convert absorbed PAR into units of mol m-2 s-1
    sunlit.absorbedPAR *= 4.57E-6 ;
    shaded.absorbedPAR *= 4.57E-6 ;
}

void Crit3D_Hydrall::preliminaryComputations(double diffuseIncomingPAR, double diffuseReflectionCoefficientPAR, double directIncomingPAR, double directReflectionCoefficientPAR,
                                             double diffuseIncomingNIR, double diffuseReflectionCoefficientNIR, double directIncomingNIR, double directReflectionCoefficientNIR,
                                             double scatteringCoefPAR, double scatteringCoefNIR, std::vector<double> &dum)
{
    dum[5]= diffuseIncomingPAR * (1.0-diffuseReflectionCoefficientPAR) * diffuseLightExtinctionCoefficient.par ;
    dum[6]= directIncomingPAR * (1.0-directReflectionCoefficientPAR) * directLightExtinctionCoefficient.par ;
    dum[7]= directIncomingPAR * (1.0-scatteringCoefPAR) * directLightExtinctionCoefficient.global ;
    dum[8]=  diffuseIncomingNIR * (1.0-diffuseReflectionCoefficientNIR) * diffuseLightExtinctionCoefficient.nir;
    dum[9]= directIncomingNIR * (1.0-directReflectionCoefficientNIR) * directLightExtinctionCoefficient.nir;
    dum[10]= directIncomingNIR * (1.0-scatteringCoefNIR) * directLightExtinctionCoefficient.global ;
    dum[11]= UPSCALINGFUNC((diffuseLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy);
    dum[12]= UPSCALINGFUNC((directLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy);
    dum[13]= UPSCALINGFUNC((diffuseLightExtinctionCoefficient.par+directLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy);
    dum[14]= UPSCALINGFUNC((directLightExtinctionCoefficient.nir+directLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy);
    dum[15]= UPSCALINGFUNC(directLightExtinctionCoefficient.global,plant.leafAreaIndexCanopy) - UPSCALINGFUNC((2.0*directLightExtinctionCoefficient.global),plant.leafAreaIndexCanopy) ;

    // Long-wave radiation balance by sunlit (1) and shaded (2) big-leaf (W m-2) from Wang & Leuning 1998
    dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*POWER4(weatherVariable.myInstantTemp+ZEROCELSIUS); //negativo
    dum[16] *= diffuseLightExtinctionCoefficient.global ;
}

void Crit3D_Hydrall::leafTemperature()
{
    if (environmentalVariable.sineSolarElevation > 0.001)
    {
        double sunlitGlobalRadiation,shadedGlobalRadiation;

        //shadedIrradiance = myDiffuseIrradiance * shaded.leafAreaIndex / statePlant.stateGrowth.leafAreaIndex;
        shadedGlobalRadiation = weatherVariable.derived.myDiffuseIrradiance * simulationStepInSeconds ;
        shaded.leafTemperature = weatherVariable.myInstantTemp + 1.67*1.0e-6 * shadedGlobalRadiation - 0.25 * weatherVariable.vaporPressureDeficit / weatherVariable.derived.psychrometricConstant; // by Stanghellini 1987 phd thesis

        // sunlitIrradiance = myDiffuseIrradiance * sunlit.leafAreaIndex/ statePlant.stateGrowth.leafAreaIndex;
        //sunlitIrradiance = myDirectIrradiance * sunlit.leafAreaIndex/ statePlant.stateGrowth.leafAreaIndex;
        sunlitGlobalRadiation = weatherVariable.myInstantTemp * simulationStepInSeconds ;
        sunlit.leafTemperature = weatherVariable.myInstantTemp + 1.67*1.0e-6 * sunlitGlobalRadiation - 0.25 * weatherVariable.vaporPressureDeficit / weatherVariable.derived.psychrometricConstant; // by Stanghellini 1987 phd thesis
    }
    else
    {
        sunlit.leafTemperature = shaded.leafTemperature = weatherVariable.myInstantTemp;
    }
    sunlit.leafTemperature += ZEROCELSIUS;
    shaded.leafTemperature += ZEROCELSIUS;
}

void Crit3D_Hydrall::aerodynamicalCoupling()
{
    // taken from Hydrall Model, Magnani UNIBO
    static double A = 0.0067;
    static double BETA = 3.0;
    static double KARM = 0.41;
    double heightReference , roughnessLength,zeroPlaneDisplacement, sensibleHeat, frictionVelocity, windSpeedTopCanopy;
    double canopyAerodynamicConductanceToMomentum, aerodynamicConductanceToCO2, dummy, sunlitDeltaTemp,shadedDeltaTemp;
    double leafBoundaryLayerConductance;
    double aerodynamicConductanceForHeat;
    double windSpeed;

    windSpeed = MAXVALUE(5,weatherVariable.windSpeed);
    heightReference = plant.height + 5 ; // [m]
    dummy = 0.2 * plant.leafAreaIndexCanopy ;
    zeroPlaneDisplacement = MINVALUE(plant.height * (log(1+pow(dummy,0.166)) + 0.03*log(1+pow(dummy,6))), 0.99*plant.height) ;
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
    double sensibleHeatOld = -9999;
    double threshold = fabs(sensibleHeat/10000.0);

    while( (20 > i++) && (fabs(sensibleHeat - sensibleHeatOld)> threshold))
    {
        //first occurence of the loop
        if (isEqual(sunlit.aerodynamicConductanceCO2Exchange, 0))
        {
            sunlit.aerodynamicConductanceCO2Exchange = 1.05 * sunlit.leafAreaIndex/plant.leafAreaIndexCanopy;
            shaded.aerodynamicConductanceCO2Exchange = 1.05 * shaded.leafAreaIndex/plant.leafAreaIndexCanopy;
        }

        // Monin-Obukhov length (m) and nondimensional height
        // Note: imposed a limit to non-dimensional height under stable
        // conditions, corresponding to a value of 0.2 for the generalized
        // stability factor F (=1/FIM/FIH)
        sensibleHeatOld = sensibleHeat ;
        double moninObukhovLength,zeta,deviationFunctionForMomentum,deviationFunctionForHeat,radiativeConductance,totalConductanceToHeatExchange,stomatalConductanceWater ;

        moninObukhovLength = -(pow(frictionVelocity,3))*HEAT_CAPACITY_AIR_MOLAR*weatherVariable.atmosphericPressure;
        moninObukhovLength /= (R_GAS*(KARM*9.8*sensibleHeat));

        zeta = MINVALUE((heightReference-zeroPlaneDisplacement)/moninObukhovLength,0.25) ;

        if (zeta < 0)
        {
            //Stability function for momentum and heat (-)
            double x,y,stabilityFunctionForMomentum;
            stabilityFunctionForMomentum = pow((1.0-16.0*zeta),-0.25);
            x= 1.0/stabilityFunctionForMomentum;
            y= 1.0/pow(stabilityFunctionForMomentum,2);
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
        leafBoundaryLayerConductance = A*sqrt(windSpeedTopCanopy/(leafWidth()))*((2.0/BETA)*(1-exp(-BETA/2.0))) * plant.leafAreaIndexCanopy;
        //       Total canopy aerodynamic conductance for momentum exchange (s m-1)
        canopyAerodynamicConductanceToMomentum= frictionVelocity / (windSpeed/frictionVelocity + (deviationFunctionForMomentum-deviationFunctionForHeat)/KARM);
        // Aerodynamic conductance for heat exchange (mol m-2 s-1)
        dummy =	(weatherVariable.atmosphericPressure/R_GAS)/(weatherVariable.myInstantTemp + ZEROCELSIUS);// conversion factor m s-1 into mol m-2 s-1
        aerodynamicConductanceForHeat =  ((canopyAerodynamicConductanceToMomentum*leafBoundaryLayerConductance)/(canopyAerodynamicConductanceToMomentum + leafBoundaryLayerConductance)) * dummy ; //whole canopy
        sunlit.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat * sunlit.leafAreaIndex/plant.leafAreaIndexCanopy ;//sunlit big-leaf
        shaded.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat - sunlit.aerodynamicConductanceHeatExchange ; //  shaded big-leaf
        // Canopy radiative conductance (mol m-2 s-1)
        radiativeConductance= 4*(weatherVariable.derived.slopeSatVapPressureVSTemp/weatherVariable.derived.psychrometricConstant)*(STEFAN_BOLTZMANN/HEAT_CAPACITY_AIR_MOLAR)*pow((weatherVariable.myInstantTemp + ZEROCELSIUS),3);
        // Total conductance to heat exchange (mol m-2 s-1)
        totalConductanceToHeatExchange =  aerodynamicConductanceForHeat + radiativeConductance; //whole canopy
        sunlit.totalConductanceHeatExchange = totalConductanceToHeatExchange * sunlit.leafAreaIndex/plant.leafAreaIndexCanopy;	//sunlit big-leaf
        shaded.totalConductanceHeatExchange = totalConductanceToHeatExchange - sunlit.totalConductanceHeatExchange;  //shaded big-leaf

        // Temperature of big-leaf (approx. expression)
        if (sunlit.leafAreaIndex > EPSILON)
        {
            stomatalConductanceWater= (10.0/sunlit.leafAreaIndex); //dummy stom res for sunlit big-leaf
            //if (sunlit.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/sunlit.isothermalNetRadiation,0.5);
            sunlitDeltaTemp = ((stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceHeatExchange)
                              *weatherVariable.derived.psychrometricConstant*sunlit.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                              - weatherVariable.vaporPressureDeficit*(1+0.001*sunlit.isothermalNetRadiation))
                              /sunlit.totalConductanceHeatExchange/(weatherVariable.derived.psychrometricConstant
                              *(stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceCO2Exchange)
                              +weatherVariable.derived.slopeSatVapPressureVSTemp/sunlit.totalConductanceHeatExchange);
        } //(1+0.001*sunlit.isothermalNetRadiation) sospetto TODO
        else
        {
            sunlitDeltaTemp = 0.0;
        }
        //sunlitDeltaTemp = 0.0;  // TODO: check

        sunlit.leafTemperature = weatherVariable.myInstantTemp + sunlitDeltaTemp	+ ZEROCELSIUS ; //sunlit big-leaf
        stomatalConductanceWater = 10.0/shaded.leafAreaIndex ; //dummy stom res for shaded big-leaf
        //if (shaded.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/shaded.isothermalNetRadiation,0.5);
        shadedDeltaTemp = ((stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)*weatherVariable.derived.psychrometricConstant*shaded.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                           - weatherVariable.vaporPressureDeficit*(1+0.001*shaded.isothermalNetRadiation))/shaded.totalConductanceHeatExchange
                           /(weatherVariable.derived.psychrometricConstant*(stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)
                           + weatherVariable.derived.slopeSatVapPressureVSTemp/shaded.totalConductanceHeatExchange);
        //shadedDeltaTemp = 0.0;
        shaded.leafTemperature = weatherVariable.myInstantTemp + shadedDeltaTemp + ZEROCELSIUS;  //shaded big-leaf
        // Sensible heat flux from the whole canopy
        sensibleHeat = HEAT_CAPACITY_AIR_MOLAR * (sunlit.aerodynamicConductanceHeatExchange*sunlitDeltaTemp + shaded.aerodynamicConductanceHeatExchange*shadedDeltaTemp);
    }

    if (plant.isAmphystomatic) aerodynamicConductanceToCO2 = 0.78 * aerodynamicConductanceForHeat; //amphystomatous species. Ratio of diffusivities from Wang & Leuning 1998
    else aerodynamicConductanceToCO2 = 0.78 * (canopyAerodynamicConductanceToMomentum * leafBoundaryLayerConductance)/(leafBoundaryLayerConductance + 2.0*canopyAerodynamicConductanceToMomentum) * dummy; //hypostomatous species

    sunlit.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * sunlit.leafAreaIndex/plant.leafAreaIndexCanopy ; //sunlit big-leaf
    shaded.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * shaded.leafAreaIndex/plant.leafAreaIndexCanopy ;  //shaded big-leaf
}

double  Crit3D_Hydrall::leafWidth()
{
    // la funzione deve essere scritta secondo regole che possono fr variare lo spessore in base alla fenologia
    // come per la vite?
    //TODO leaf width
    plant.myLeafWidth = 0.1;
    return plant.myLeafWidth;
}

void Crit3D_Hydrall::upscale()
{
    //cultivar->parameterWangLeuning.maxCarbonRate era input, ora da prendere da classe e leggere da tipo di pianta
    double maxCarboxRate = 0.0;

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
    sunlit.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[0])* UPSCALINGFUNC((directLightExtinctionCoefficient.global + diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy); //sunlit big-leaf
    shaded.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[1]); //shaded big-leaf
    shaded.darkRespiration *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy) - UPSCALINGFUNC((directLightExtinctionCoefficient.global + diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy));
    double entropicFactorElectronTransporRate = (-0.75*(weatherVariable.last30DaysTAvg)+660);  // entropy term for J (kJ mol-1 oC-1)
    double entropicFactorCarboxyliation = (-1.07*(weatherVariable.last30DaysTAvg)+668); // entropy term for VCmax (kJ mol-1 oC-1)
    if (environmentalVariable.sineSolarElevation > 1.0e-3)
    {
        //Stomatal conductance to CO2 in darkness (molCO2 m-2 s-1)
        sunlit.minimalStomatalConductance = parameterWangLeuning.stomatalConductanceMin  * UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy)	;
        shaded.minimalStomatalConductance = parameterWangLeuning.stomatalConductanceMin  * (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy));
        // Carboxylation rate
        //sunlit.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[0]); //sunlit big leaf
        //shaded.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[1]); //shaded big leaf
        sunlit.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,sunlit.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuning.optimalTemperatureForPhotosynthesis); //sunlit big leaf
        shaded.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,shaded.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuning.optimalTemperatureForPhotosynthesis); //shaded big leaf
        // Scale-up maximum carboxylation rate (mol m-2 s-1)
        sunlit.maximalCarboxylationRate *= UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy);
        shaded.maximalCarboxylationRate *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy));
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
        sunlit.quantumYieldPS2 = 0.352 + 0.022*dum[2] - 3.4E-4*pow(dum[2],2);      //sunlit big-leaf
        shaded.quantumYieldPS2 = 0.352 + 0.022*dum[3] - 3.4E-4*pow(dum[3],2);      //shaded big-leaf
        // Compute convexity factor of light response curve (-)
        // The value derived from leaf Chl content is modified for temperature effects, according to Bernacchi et al. (2003)
        leafConvexityFactor = 1 - plant.myChlorophyllContent * 6.93E-4 ;    //from Pons & Anten (2004), fig. 3b
        sunlit.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[2] - 3.7E-4*pow(dum[2],2));  //sunlit big-leaf
        shaded.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[3] - 3.7E-4*pow(dum[3],2));  //shaded big-leaf
        // Scale-up potential electron transport of sunlit big-leaf (mol m-2 s-1)
        sunlit.maximalElectronTrasportRate *= UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy);
        // Adjust electr transp of sunlit big-leaf for PAR effects (mol e- m-2 s-1)
        dum[4]= sunlit.absorbedPAR * sunlit.quantumYieldPS2 * BETA ; //  potential PSII e- transport of sunlit big-leaf (mol m-2 s-1)
        dum[5]= dum[4] + sunlit.maximalElectronTrasportRate ;
        dum[6]= dum[4] * sunlit.maximalElectronTrasportRate ;
        sunlit.maximalElectronTrasportRate = (dum[5] - sqrt(pow(dum[5],2) - 4.0*sunlit.convexityFactorNonRectangularHyperbola*dum[6])) / (2.0*sunlit.convexityFactorNonRectangularHyperbola);
        // Scale-up potential electron transport of shaded big-leaf (mol m-2 s-1)
        // The simplified formulation proposed by de Pury & Farquhar (1999) is applied
        shaded.maximalElectronTrasportRate *= (UPSCALINGFUNC(diffuseLightExtinctionCoefficient.par,plant.leafAreaIndexCanopy) - UPSCALINGFUNC((directLightExtinctionCoefficient.global+diffuseLightExtinctionCoefficient.par),plant.leafAreaIndexCanopy));
        // Adjust electr transp of shaded big-leaf for PAR effects (mol e- m-2 s-1)
        dum[4]= shaded.absorbedPAR * shaded.quantumYieldPS2 * BETA ; // potential PSII e- transport of sunlit big-leaf (mol m-2 s-1)
        dum[5]= dum[4] + shaded.maximalElectronTrasportRate ;
        dum[6]= dum[4] * shaded.maximalElectronTrasportRate ;
        shaded.maximalElectronTrasportRate = (dum[5] - sqrt(pow(dum[5],2) - 4.0*shaded.convexityFactorNonRectangularHyperbola*dum[6])) / (2.0*shaded.convexityFactorNonRectangularHyperbola);
    }
    else
    {  //night-time computations
        sunlit.maximalElectronTrasportRate = 0.0;
        shaded.maximalElectronTrasportRate = 0.0;
        sunlit.darkRespiration = 0.0;
        sunlit.maximalCarboxylationRate = 0.0;
    }
}

inline double Crit3D_Hydrall::acclimationFunction(double Ha , double Hd, double leafTemp,
                                             double entropicTerm,double optimumTemp)
{
    // taken from Hydrall Model, Magnani UNIBO
    return exp(Ha*(leafTemp - optimumTemp)/(optimumTemp*R_GAS*leafTemp))
           *(1+exp((optimumTemp*entropicTerm-Hd)/(optimumTemp*R_GAS)))
           /(1+exp((leafTemp*entropicTerm-Hd)/(leafTemp*R_GAS))) ;
}


void Crit3D_Hydrall::carbonWaterFluxesProfile()
{
    // taken from Hydrall Model, Magnani UNIBO
    treeAssimilationRate = 0 ;

    treeTranspirationRate.resize(soil.layersNr);

    double totalStomatalConductance = 0 ;
    for (int i=0; i < soil.layersNr; i++)
    {
        treeTranspirationRate[i] = 0 ;

        if(sunlit.leafAreaIndex > 0)
        {
            Crit3D_Hydrall::photosynthesisKernel(sunlit.compensationPoint, sunlit.aerodynamicConductanceCO2Exchange, sunlit.aerodynamicConductanceHeatExchange, sunlit.minimalStomatalConductance,
                                                             sunlit.maximalElectronTrasportRate, sunlit.carbonMichaelisMentenConstant,
                                                             sunlit.oxygenMichaelisMentenConstant,sunlit.darkRespiration, sunlit.isothermalNetRadiation,
                                                             parameterWangLeuning.alpha * soil.stressCoefficient[i], sunlit.maximalCarboxylationRate,
                                                             &(sunlit.assimilation), &(sunlit.stomatalConductance),
                                                             &(sunlit.transpiration));
        }
        else
        {
            sunlit.assimilation = 0.0;
            sunlit.stomatalConductance = 0.0;
            sunlit.transpiration = 0.0;
        }

        treeAssimilationRate += sunlit.assimilation * soil.rootDensity[i] ;
        treeTranspirationRate[i] += sunlit.transpiration * soil.rootDensity[i] ;

        // shaded big leaf
        Crit3D_Hydrall::photosynthesisKernel(shaded.compensationPoint, shaded.aerodynamicConductanceCO2Exchange,shaded.aerodynamicConductanceHeatExchange, shaded.minimalStomatalConductance,
                                                         shaded.maximalElectronTrasportRate, shaded.carbonMichaelisMentenConstant,
                                                         shaded.oxygenMichaelisMentenConstant,shaded.darkRespiration, shaded.isothermalNetRadiation,
                                                         parameterWangLeuning.alpha * soil.stressCoefficient[i], shaded.maximalCarboxylationRate,
                                                         &(shaded.assimilation), &(shaded.stomatalConductance),
                                                         &(shaded.transpiration));
        treeAssimilationRate += shaded.assimilation * soil.rootDensity[i] ; //canopy gross assimilation (mol m-2 s-1)
        treeTranspirationRate[i] += shaded.transpiration * soil.rootDensity[i] ; //canopy transpiration (mol m-2 s-1)

    }
}


void Crit3D_Hydrall::photosynthesisKernel(double COMP,double GAC,double GHR,double GSCD,double J,double KC,double KO
                                            ,double RD,double RNI,double STOMWL,double VCmax,double *ASS,double *GSC,double *TR)
{
// taken from Hydrall Model, Magnani UNIBO
// daily time computation
#define NODATA_TOLERANCE 9999
#define OSS              21176       /*!< oxygen part pressure in the atmosphere, Pa  */

    double myStromalCarbonDioxide, VPDS, WC,WJ,VC,DUM1,CS; //, myPreviousVPDS;
    double ASSOLD, deltaAssimilation, myTolerance; //, myPreviousDelta;
    int I,Imax ;


    Imax = 1000 ;
    myTolerance = 1e-7;
    deltaAssimilation = NODATA_TOLERANCE;
    //myPreviousDelta = deltaAssimilation;
    if (J >= 1.0e-7)
    {
        // Initialize variables
        myStromalCarbonDioxide = 0.7 * environmentalVariable.CO2 ;
        VPDS = weatherVariable.vaporPressureDeficit;
        //myPreviousVPDS = VPDS;
        ASSOLD = NODATA ;
        DUM1 = 1.6*weatherVariable.derived.slopeSatVapPressureVSTemp/weatherVariable.derived.psychrometricConstant+ GHR/GAC;
        I = 0 ; // initialize the cycle variable
        while ((I++ < Imax) && (deltaAssimilation > myTolerance))
        {
            //Assimilation
            WC = VCmax * myStromalCarbonDioxide / (myStromalCarbonDioxide + KC * (1.0 + OSS / KO));  //RuBP-limited carboxylation (mol m-2 s-1)
            WJ = J * myStromalCarbonDioxide / (4.5 * myStromalCarbonDioxide + 10.5 * COMP);  //electr transp-limited carboxyl (mol m-2 s-1)
            VC = MINVALUE(WC,WJ);  //carboxylation rate (mol m-2 s-1)

            *ASS = MAXVALUE(0.0, VC * (1.0 - COMP / myStromalCarbonDioxide));  //gross assimilation (mol m-2 s-1)
            CS = environmentalVariable.CO2 - weatherVariable.atmosphericPressure * (*ASS - RD) / GAC;	//CO2 concentration at leaf surface (Pa)
            CS = MAXVALUE(1e-4,CS);
            //Stomatal conductance
            *GSC = GSCD + STOMWL * (*ASS-RD) / (CS-COMP) * parameterWangLeuning.sensitivityToVapourPressureDeficit / (parameterWangLeuning.sensitivityToVapourPressureDeficit +VPDS); //stom conduct to CO2 (mol m-2 s-1)
            *GSC = MAXVALUE(*GSC,1.0e-5);
            // Stromal CO2 concentration
            myStromalCarbonDioxide = CS - weatherVariable.atmosphericPressure * (*ASS - RD) / (*GSC);	 //CO2 concentr at carboxyl sites (Pa)
            myStromalCarbonDioxide = MAXVALUE(1.0e-2,myStromalCarbonDioxide) ;
            //Vapour pressure deficit at leaf surface
            VPDS = (weatherVariable.derived.slopeSatVapPressureVSTemp / HEAT_CAPACITY_AIR_MOLAR*RNI + weatherVariable.vaporPressureDeficit * GHR) / (GHR+(*GSC)*DUM1);  //VPD at the leaf surface (Pa)
            deltaAssimilation = fabs((*ASS) - ASSOLD);
            ASSOLD = *ASS ;
        }
    }
    else //night time computation
    {
        *ASS= 0.0;
        *GSC= GSCD;
        VPDS= weatherVariable.vaporPressureDeficit ;
    }
    //  Transpiration rate
    *TR = (*GSC / 0.64) * VPDS/weatherVariable.atmosphericPressure ;  //Transpiration rate (mol m-2 s-1). Ratio of diffusivities from Wang & Leuning 1998
    *TR = MAXVALUE(1.0E-8,*TR) ;
}


void Crit3D_Hydrall::cumulatedResults()
{
    // taken from Hydrall Model, Magnani UNIBO
    // Cumulate hourly values of gas exchange
    deltaTime.absorbedPAR = simulationStepInSeconds*(sunlit.absorbedPAR+shaded.absorbedPAR);  //absorbed PAR (mol m-2)
    deltaTime.grossAssimilation = simulationStepInSeconds * treeAssimilationRate ; // canopy gross assimilation (mol m-2)
    deltaTime.respiration = simulationStepInSeconds * Crit3D_Hydrall::plantRespiration() ;
    deltaTime.netAssimilation = deltaTime.grossAssimilation- deltaTime.respiration ;
    deltaTime.netAssimilation = deltaTime.netAssimilation*12/1000.0 ; //CARBONFACTOR DA METTERE dopo convert to kg DM m-2

    statePlant.treeNetPrimaryProduction += deltaTime.netAssimilation ;

    //understorey


    deltaTime.transpiration = 0.;

    for (int i=0; i < soil.layersNr; i++)
    {
        treeTranspirationRate[i] = simulationStepInSeconds * MH2O * treeTranspirationRate[i]; // [mm]
        deltaTime.transpiration += treeTranspirationRate[i];
    }
    deltaTime.transpiration += understorey.transpiration;

    //evaporation
    deltaTime.evaporation = computeEvaporation();

}

double Crit3D_Hydrall::computeEvaporation()
{
    double ETP = 0.5;
    double totalLAI = understorey.leafAreaIndex + plant.leafAreaIndexCanopy;

    if (totalLAI == LAIMAX)
        return 0.2 * ETP;
    else
        return -0.8 / LAIMAX *ETP;
}

double Crit3D_Hydrall::plantRespiration()
{
    // taken from Hydrall Model, Magnani UNIBO
    double leafRespiration,rootRespiration,sapwoodRespiration;
    double totalRespiration;
    nitrogenContent.leaf = 0.02;    //[kg kgDM-1] //0.02 * 10^3 [g kgDM-1]
    nitrogenContent.root = 0.0078;  //[kg kgDM-1]
    nitrogenContent.stem = 0.0021;  //[kg kgDM-1]

    // Compute hourly stand respiration at 10 oC (mol m-2 h-1)
    leafRespiration = 0.0106/2.0 * (treeBiomass.leaf * nitrogenContent.leaf/0.014) / 3600 ; //TODO: CHECK se veramente erano output orari da trasformare in respirazione al secondo
    sapwoodRespiration = 0.0106/2.0 * (treeBiomass.sapwood * nitrogenContent.stem/0.014) / 3600 ;
    rootRespiration = 0.0106/2.0 * (treeBiomass.fineRoot * nitrogenContent.root/0.014) / 3600 ;

    // Adjust for temperature effects
    //leafRespiration *= MAXVALUE(0,MINVALUE(1,Vine3D_Grapevine::temperatureMoistureFunction(myInstantTemp + ZEROCELSIUS))) ;
    //sapwoodRespiration *= MAXVALUE(0,MINVALUE(1,Vine3D_Grapevine::temperatureMoistureFunction(myInstantTemp + ZEROCELSIUS))) ;
    //shootRespiration *= MAXVALUE(0,MINVALUE(1,Vine3D_Grapevine::temperatureMoistureFunction(myInstantTemp + ZEROCELSIUS))) ;
    soil.temperature = Crit3D_Hydrall::soilTemperatureModel();
    //rootRespiration *= MAXVALUE(0,MINVALUE(1,Crit3D_Hydrall::temperatureMoistureFunction(soil.temperature + ZEROCELSIUS))) ;
    rootRespiration *= BOUNDFUNCTION(0,1,Crit3D_Hydrall::temperatureMoistureFunction(soil.temperature + ZEROCELSIUS));
    // hourly canopy respiration (sapwood+fine roots)
    totalRespiration =(leafRespiration + sapwoodRespiration + rootRespiration);
    // per second respiration
    totalRespiration /= double(HOUR_SECONDS);

    //TODO understorey respiration

    return totalRespiration;
}

double Crit3D_Hydrall::soilTemperatureModel()
{
    // taken from Hydrall Model, Magnani UNIBO
    double temp;
    temp = 0.8 * weatherVariable.last30DaysTAvg + 0.2 * weatherVariable.myInstantTemp;
    return temp;
}

double Crit3D_Hydrall::temperatureMoistureFunction(double temperature)
{
    double temperatureMoistureFactor;
    // TODO
    /*// taken from Hydrall Model, Magnani UNIBO
    int   MODEL;
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

bool Crit3D_Hydrall::growthStand()
{
    const double understoreyAllocationCoefficientToRoot = 0.5;
    // understorey update
    statePlant.understoreycumulatedBiomassFoliage = statePlant.understoreycumulatedBiomass * (1.-understoreyAllocationCoefficientToRoot);    //understorey growth: foliage...
    statePlant.understoreycumulatedBiomassRoot = statePlant.understoreycumulatedBiomass * understoreyAllocationCoefficientToRoot;         //...and roots

    // canopy update
    statePlant.treecumulatedBiomassFoliage -= (statePlant.treecumulatedBiomassFoliage/plant.foliageLongevity);
    statePlant.treecumulatedBiomassSapwood -= (statePlant.treecumulatedBiomassSapwood/plant.sapwoodLongevity);
    statePlant.treecumulatedBiomassRoot -= (statePlant.treecumulatedBiomassRoot/plant.fineRootLongevity);


    double store; // TODO to understand what's that, afterwards the uninitialized value is used
    //annual stand growth
    if (isFirstYearSimulation)
    {
        annualGrossStandGrowth = statePlant.treeNetPrimaryProduction / CARBONFACTOR; //kg DM m-2
        store = 0;
    }
    else
    {
        annualGrossStandGrowth = (store + statePlant.treeNetPrimaryProduction) / 2 / CARBONFACTOR;
        store = (store + statePlant.treeNetPrimaryProduction) / 2;
    }

    if (isFirstYearSimulation)
    {
        //optimal
    }
    else
    {
        double allocationCoeffientFoliageOld = allocationCoefficient.toFoliage;
        double allocationCoeffientFineRootsOld = allocationCoefficient.toFineRoots;
        double allocationCoeffientSapwoodOld = allocationCoefficient.toSapwood;

        //optimal

        allocationCoefficient.toFoliage = (allocationCoeffientFoliageOld + allocationCoefficient.toFoliage) / 2;
        allocationCoefficient.toFineRoots = (allocationCoeffientFineRootsOld + allocationCoefficient.toFineRoots) / 2;
        allocationCoefficient.toSapwood = (allocationCoeffientSapwoodOld + allocationCoefficient.toSapwood) / 2;
    }

    if (annualGrossStandGrowth * allocationCoefficient.toFoliage > statePlant.treecumulatedBiomassFoliage/(plant.foliageLongevity - 1))

    isFirstYearSimulation = false;
    return true;
}


void Crit3D_Hydrall::resetStandVariables()
{
    statePlant.treeNetPrimaryProduction = 0;
}

void Crit3D_Hydrall::optimal()
{
    double allocationCoefficientFoliageOld;
    double increment;
    double incrementStart = 5e-2;
    bool sol = 0;
    double allocationCoefficientFoliage0;
    double bisectionMethodIntervalALLF;
    int jmax = 40;
    double accuracy = 1e-3;

    for (int j = 0; j < 3; j++)
    {
        allocationCoefficientFoliageOld = 1;
        increment = incrementStart / std::pow(10, j);
        allocationCoefficient.toFoliage = 1;

        while(! sol && allocationCoefficient.toFoliage > -EPSILON)
        {
            rootfind(allocationCoefficient.toFoliage, allocationCoefficient.toFineRoots, allocationCoefficient.toSapwood, sol);

            if (sol)
                break;

            allocationCoefficientFoliageOld = allocationCoefficient.toFoliage;
            allocationCoefficient.toFoliage -= increment;

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

void Crit3D_Hydrall::rootfind(double &allf, double &allr, double &alls, bool &sol)
{
    //search for a solution to hydraulic constraint

    //new foliage biomass of tree after growth
    allf = MAXVALUE(0,allf);
    //if (allf < 0) allf = 0;

    statePlant.treecumulatedBiomassFoliage += (allf*annualGrossStandGrowth);

    //new tree height after growth
    if (allf*annualGrossStandGrowth > statePlant.treecumulatedBiomassFoliage/(plant.foliageLongevity-1)) {
        plant.height += (allf*annualGrossStandGrowth-statePlant.treecumulatedBiomassFoliage/(plant.foliageLongevity-1)/plant.foliageDensity);
    }

    //soil hydraulic conductivity
    double ksl;
    std::vector <std::vector <double>> conductivityWeights(2, std::vector<double>(soil.layersNr, NODATA));
    for (int i=0; i<soil.layersNr; i++)
    {
        conductivityWeights[0][i] = soil.rootDensity[i];
        conductivityWeights[1][i] = soil.nodeThickness[i];
    }
    ksl = statistics::weighedMeanMultifactor(logarithmic10Values,conductivityWeights,soil.satHydraulicConductivity);
    //specific hydraulic conductivity of soil+roots
    double soilRootsSpecificConductivity = 1/(1/KR + 1/ksl);
    //double dum = 0.5151 + MAXVALUE(0,0.0242*soil.temperature);
    //if (dum < 0.5151) dum = 0.5151;
    //soilRootsSpecificConductivity *= dum; //adjust for temp effects on water viscosity
    soilRootsSpecificConductivity *= 0.5151 + MAXVALUE(0,0.0242*soil.temperature);
    //new sapwood specific conductivity
    double sapwoodSpecificConductivity = KSMAX * (1-std::exp(-0.69315*plant.height/H50)); //adjust for height effects
    //dum = 0.5151 + MAXVALUE(0,0.0242*weatherVariable.meanDailyTemp);
    //dum = MAXVALUE(0.5151,dum);
    //if (dum < 0.5151) dum = 0.5151;
    //sapwoodSpecificConductivity *= dum;     //adjust for temp effects on water viscosity
    sapwoodSpecificConductivity *= 0.5151 + MAXVALUE(0,0.0242*weatherVariable.meanDailyTemp);

    //optimal coefficient of allocation to fine roots and sapwood for set allocation to foliage
    double quadraticEqCoefficient = std::sqrt(soilRootsSpecificConductivity/sapwoodSpecificConductivity*plant.sapwoodLongevity/plant.fineRootLongevity*plant.woodDensity);
    allr = (statePlant.treecumulatedBiomassSapwood - quadraticEqCoefficient*plant.height*statePlant.treecumulatedBiomassRoot +
            annualGrossStandGrowth*(1-allf))/annualGrossStandGrowth/(1+quadraticEqCoefficient*plant.height);

    //allr = MAXVALUE(EPSILON,allr); //bracket ALLR between (1-ALLF) and a small value
    //allr = MINVALUE(MAXVALUE(EPSILON,allr),1-allf);
    allr = BOUNDFUNCTION(EPSILON,1-allf,allr); // TODO to be checked
    //alls = 1 - allf - allr;
    // alls = MAXVALUE(alls,EPSILON); //bracket ALLS between 1 and a small value
    //alls = MINVALUE(1,MAXVALUE(alls,EPSILON));
    alls = BOUNDFUNCTION(EPSILON,1,1-allf-allr); // TODO to be checked
    //resulting fine root and sapwood biomass
    statePlant.treecumulatedBiomassRoot += allr * annualGrossStandGrowth;
    statePlant.treecumulatedBiomassRoot = MAXVALUE(EPSILON,statePlant.treecumulatedBiomassRoot);
    statePlant.treecumulatedBiomassSapwood += alls * annualGrossStandGrowth;
    statePlant.treecumulatedBiomassSapwood = MAXVALUE(EPSILON,statePlant.treecumulatedBiomassSapwood);

    //resulting leaf specific resistance (MPa s m2 m-3)
    double hydraulicResistancePerFoliageArea;
    hydraulicResistancePerFoliageArea = (1./(statePlant.treecumulatedBiomassRoot*soilRootsSpecificConductivity)
        + (plant.height*plant.height*plant.woodDensity)/(statePlant.treecumulatedBiomassSapwood*sapwoodSpecificConductivity))
        * (statePlant.treecumulatedBiomassFoliage*plant.specificLeafArea);
    //resulting minimum leaf water potential

    plant.psiLeafMinimum = plant.psiLeafCritical - (0.01 * plant.height)-(plant.transpirationPerUnitFoliageAreaCritical * 0.018/1000. * hydraulicResistancePerFoliageArea);
    //check if given value of ALLF satisfies optimality constraint
    if(plant.psiLeafMinimum >= PSITHR)
        sol = true;
    else
    {
        sol = false;
        allr = statePlant.treecumulatedBiomassSapwood
                +  annualGrossStandGrowth -statePlant.treecumulatedBiomassRoot*quadraticEqCoefficient*plant.height;
        allr /= (annualGrossStandGrowth * (1.+quadraticEqCoefficient*plant.height));
        //allr = MINVALUE(1,MAXVALUE(allr,EPSILON));
        allr = BOUNDFUNCTION(EPSILON,1,allr); // TODO to be checked
        alls = 1.-allr;
        allf = 0; // TODO verify its value
    }

}
