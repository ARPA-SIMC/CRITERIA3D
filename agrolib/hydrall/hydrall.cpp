/*!
    \name hydrall.cpp
    \brief 
    \authors Antonio Volta, Caterina Toscano
	
*/


#include <stdio.h>
#include <math.h>
#include "crit3dDate.h"
#include "commonConstants.h"
#include "hydrall.h"


bool computeHydrall(Crit3DDate myDate, double myTemperature, double myElevation)
{
    getCO2(myDate, myTemperature, myElevation);
    return true;
}

double getCO2(Crit3DDate myDate, double myTemperature, double myElevation)
{
    double atmCO2 ; // fitting from data of Mauna Loa,Hawaii
    if (myDate.year < 1990)
    {
        atmCO2= 280 * exp(0.0014876*(myDate.year -1840));//exponential change in CO2 concentration (ppm)
    }
    else
    {
        atmCO2= 350 * exp(0.00630*(myDate.year - 1990));
    }
    atmCO2 += 3*cos(2*PI*getDoyFromDate(myDate)/365.0);		     // to consider the seasonal effects
    return atmCO2*getPressureFromElevation(myTemperature, myElevation)/1000000  ;   // [Pa] in +- ppm/10
}

double getPressureFromElevation(double myTemperature, double myElevation)
{
    return SEA_LEVEL_PRESSURE * exp((- GRAVITY * M_AIR * myElevation) / (R_GAS * myTemperature));
}
