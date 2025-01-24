/*!
    \name hydrall.cpp
    \brief 
    \authors Antonio Volta, Caterina Toscano
	
*/


//#include <stdio.h>
#include <math.h>
#include "crit3dDate.h"
#include "commonConstants.h"
#include "hydrall.h"
#include "furtherMathFunctions.h"
#include "physics.h"

Crit3DHydrallMaps::Crit3DHydrallMaps(const gis::Crit3DRasterGrid& DEM)
{
    mapLAI = new gis::Crit3DRasterGrid;
    mapLast30DaysTavg = new gis::Crit3DRasterGrid;

    mapLAI->initializeGrid(DEM);
    mapLast30DaysTavg->initializeGrid(DEM);
}


bool computeHydrallPoint(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep)
{
    getCO2(myDate, myTemperature, myElevation);

    // da qui in poi bisogna fare un ciclo su tutte le righe e le colonne

    double actualLAI = getLAI();
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

double getCO2(Crit3DDate myDate, double myTemperature, double myElevation)
{
    double atmCO2 ; //https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5/download.table
    double year[24] = {1750,1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100};
    double valueCO2[24] = {278,283,285,296,300,303,307,310,311,317,325,339,354,369,389,413,443,473,503,530,550,565,570,575};

    // exponential fitting Mauna Loa
    if (myDate.year < 1990)
    {
        atmCO2= 280 * exp(0.0014876*(myDate.year -1840));//exponential change in CO2 concentration (ppm)
    }
    else
    {
        atmCO2= 353 * exp(0.00630*(myDate.year - 1990));
    }
    atmCO2 += 3*cos(2*PI*getDoyFromDate(myDate)/365.0);		     // to consider the seasonal effects
    return atmCO2*getPressureFromElevation(myTemperature, myElevation)/1000000  ;   // [Pa] in +- ppm/10
}

double getPressureFromElevation(double myTemperature, double myElevation)
{
    return SEA_LEVEL_PRESSURE * exp((- GRAVITY * M_AIR * myElevation) / (R_GAS * myTemperature));
}

double getLAI()
{
    // TODO
    return 4;
}

double photosynthesisAndTranspiration()
{
    TweatherDerivedVariable weatherDerivedVariable;

    return 0;
}

void weatherVariables(double myInstantTemp,double myRelativeHumidity,double myCloudiness,TweatherDerivedVariable weatherDerivedVariable)
{
    // taken from Hydrall Model, Magnani UNIBO
    weatherDerivedVariable.airVapourPressure = saturationVaporPressure(myInstantTemp)*myRelativeHumidity/100.;
    weatherDerivedVariable.emissivitySky = 1.24 * pow((weatherDerivedVariable.airVapourPressure/100.0) / (myInstantTemp+ZEROCELSIUS),(1.0/7.0))*(1 - 0.84*myCloudiness)+ 0.84*myCloudiness;
    weatherDerivedVariable.longWaveIrradiance = pow(myInstantTemp+ZEROCELSIUS,4) * weatherDerivedVariable.emissivitySky * STEFAN_BOLTZMANN ;
    weatherDerivedVariable.slopeSatVapPressureVSTemp = 2588464.2 / pow(240.97 + myInstantTemp, 2) * exp(17.502 * myInstantTemp / (240.97 + myInstantTemp)) ;
}
/*
double meanLastMonthTemperature(double previousLastMonthTemp, double simulationStepInSeconds, double myInstantTemp)
{
    double newTemperature;
    double monthFraction;
    monthFraction = simulationStepInSeconds/(2592000.0); // seconds of 30 days
    newTemperature = previousLastMonthTemp * (1 - monthFraction) + myInstantTemp * monthFraction ;
    return newTemperature;
}*/
