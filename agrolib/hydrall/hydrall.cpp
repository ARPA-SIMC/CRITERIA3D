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

Crit3DHydrallMaps::Crit3DHydrallMaps()
{
    mapLAI = new gis::Crit3DRasterGrid;
    mapLast30DaysTavg = new gis::Crit3DRasterGrid;
}

void Crit3DHydrallMaps::initialize(const gis::Crit3DRasterGrid& DEM)
{
    mapLAI->initializeGrid(DEM);
    mapLast30DaysTavg->initializeGrid(DEM);
}

Crit3DHydrallMaps::~Crit3DHydrallMaps()
{
    mapLAI->clear();
    mapLast30DaysTavg->clear();
}

bool Crit3D_Hydrall::computeHydrallPoint(Crit3DDate myDate, double myTemperature, double myElevation, int secondPerStep, double &AGBiomass, double &rootBiomass)
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

double Crit3D_Hydrall::getCO2(Crit3DDate myDate, double myTemperature, double myElevation)
{
    double atmCO2 = 400 ; //https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5/download.table
    double year[24] = {1750,1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100};
    double valueCO2[24] = {278,283,285,296,300,303,307,310,311,317,325,339,354,369,389,413,443,473,503,530,550,565,570,575};

    atmCO2 = interpolation::linearInterpolation(double(myDate.year), year, valueCO2, 24);

    // exponential fitting Mauna Loa
    /*if (myDate.year < 1990)
    {
        atmCO2= 280 * exp(0.0014876*(myDate.year -1840));//exponential change in CO2 concentration (ppm)
    }
    else
    {
        atmCO2= 353 * exp(0.00630*(myDate.year - 1990));
    }
*/
    atmCO2 += 3*cos(2*PI*getDoyFromDate(myDate)/365.0);		     // to consider the seasonal effects
    //return atmCO2*getPressureFromElevation(myTemperature, myElevation)/1000000  ;   // [Pa] in +- ppm/10 formula changed from the original Hydrall

    return atmCO2 * weatherVariable.atmosphericPressure/1000000;
}
/*
double Crit3D_Hydrall::getPressureFromElevation(double myTemperature, double myElevation)
{
    return P0 * exp((- GRAVITY * M_AIR * myElevation) / (R_GAS * myTemperature));
}
*/
double Crit3D_Hydrall::getLAI()
{
    // TODO
    return 4;
}

double Crit3D_Hydrall::photosynthesisAndTranspiration()
{
    TweatherDerivedVariable weatherDerivedVariable;

    return 0;
}

void Crit3D_Hydrall::initialize()
{
    myChlorophyllContent = NODATA;
    elevation = NODATA;
}

void Crit3D_Hydrall::setHourlyVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex)
{
    setWeatherVariables(temp, irradiance, prec, relativeHumidity, windSpeed, directIrradiance, diffuseIrradiance, cloudIndex);
}

bool Crit3D_Hydrall::setWeatherVariables(double temp, double irradiance , double prec , double relativeHumidity , double windSpeed, double directIrradiance, double diffuseIrradiance, double cloudIndex)
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

    setDerivedWeatherVariables(directIrradiance, diffuseIrradiance, cloudIndex);

    if ((int(prec) != NODATA) && (int(temp) != NODATA) && (int(windSpeed) != NODATA)
        && (int(irradiance) != NODATA) && (int(relativeHumidity) != NODATA))
        isReadingOK = true;

    return isReadingOK;
}

void Crit3D_Hydrall::setDerivedWeatherVariables(double directIrradiance, double diffuseIrradiance, double cloudIndex)
{
    weatherVariable.derived.airVapourPressure = saturationVaporPressure(weatherVariable.myInstantTemp)*weatherVariable.relativeHumidity/100.;
    weatherVariable.derived.slopeSatVapPressureVSTemp = 2588464.2 / pow(240.97 + weatherVariable.myInstantTemp, 2) * exp(17.502 * weatherVariable.myInstantTemp / (240.97 + weatherVariable.myInstantTemp)) ;
    weatherVariable.derived.myDirectIrradiance = directIrradiance;
    weatherVariable.derived.myDiffuseIrradiance = diffuseIrradiance;
    double myCloudiness = MINVALUE(1,MAXVALUE(0,cloudIndex));
    weatherVariable.derived.myEmissivitySky = 1.24 * pow((weatherVariable.derived.airVapourPressure/100.0) / (weatherVariable.myInstantTemp+ZEROCELSIUS),(1.0/7.0))*(1 - 0.84*myCloudiness)+ 0.84*myCloudiness;
    weatherVariable.derived.myLongWaveIrradiance = pow(weatherVariable.myInstantTemp+ZEROCELSIUS,4) * weatherVariable.derived.myEmissivitySky * STEFAN_BOLTZMANN ;
    weatherVariable.derived.psychrometricConstant = psychro(weatherVariable.atmosphericPressure,weatherVariable.myInstantTemp);
    return;
}

void Crit3D_Hydrall::setPlantVariables(double chlorophyllContent)
{
    myChlorophyllContent = chlorophyllContent;
}

void Crit3D_Hydrall::radiationAbsorption(double mySunElevation)
{
    // taken from Hydrall Model, Magnani UNIBO

    // TODO chiedere a Magnani questi parametri
    static double   leafAbsorbanceNIR= 0.2;
    static double   hemisphericalIsotropyParameter = 0. ; // in order to change the hemispherical isotropy from -0.4 to 0.6 Wang & Leuning 1998
    static double   clumpingParameter = 1.0 ; // from 0 to 1 <1 for needles
    double  diffuseLightSector1K,diffuseLightSector2K,diffuseLightSector3K ;
    double scatteringCoefPAR, scatteringCoefNIR ;
    double dum[17];
    double  sunlitAbsorbedNIR ,  shadedAbsorbedNIR, sunlitAbsorbedLW , shadedAbsorbedLW;
    double directIncomingPAR, directIncomingNIR , diffuseIncomingPAR , diffuseIncomingNIR,leafAbsorbancePAR;
    double directReflectionCoefficientPAR , directReflectionCoefficientNIR , diffuseReflectionCoefficientPAR , diffuseReflectionCoefficientNIR;
    //projection of the unit leaf area in the direction of the sun's beam, following Sellers 1985 (in Wang & Leuning 1998)
    double diffuseLightK;

    sineSolarElevation = MAXVALUE(0.0001,sin(mySunElevation*DEG_TO_RAD));
    directLightK = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*sineSolarElevation) - pow(hemisphericalIsotropyParameter,2)*(0.33-0.579*sineSolarElevation))/ sineSolarElevation;

    /*Extinction coeff for canopy of black leaves, diffuse radiation
    The average extinctio coefficient is computed considering three sky sectors,
    assuming SOC conditions (Goudriaan & van Laar 1994, p 98-99)*/
    diffuseLightSector1K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.259) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.259))/0.259;  //projection of unit leaf for first sky sector (0-30 elevation)
    diffuseLightSector2K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.707) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.707))/0.707 ; //second sky sector (30-60 elevation)
    diffuseLightSector3K = (0.5 - hemisphericalIsotropyParameter*(0.633-1.11*0.966) - hemisphericalIsotropyParameter*hemisphericalIsotropyParameter*(0.33-0.579*0.966))/ 0.966 ; // third sky sector (60-90 elevation)
    diffuseLightK =- 1.0/leafAreaIndex * log(0.178 * exp(-diffuseLightSector1K*leafAreaIndex) + 0.514 * exp(-diffuseLightSector2K*leafAreaIndex)
                                                                      + 0.308 * exp(-diffuseLightSector3K*leafAreaIndex));  //approximation based on relative radiance from 3 sky sectors
    //Include effects of leaf clumping (see Goudriaan & van Laar 1994, p 110)
    directLightK  *= clumpingParameter ;//direct light
    diffuseLightK *= clumpingParameter ;//diffuse light
    if (sineSolarElevation > 0.001)
    {
        //Leaf area index of sunlit (1) and shaded (2) big-leaf
        sunlit.leafAreaIndex = UPSCALINGFUNC(directLightK,leafAreaIndex);
        shaded.leafAreaIndex = leafAreaIndex - sunlit.leafAreaIndex ;
        //Extinction coefficients for direct and diffuse PAR and NIR radiation, scattering leaves
        //Based on approximation by Goudriaan 1977 (in Goudriaan & van Laar 1994)
        double exponent= -pow(10,0.28 + 0.63*log10(myChlorophyllContent*0.85/1000));
        leafAbsorbancePAR = 1 - pow(10,exponent);//from Agusti et al (1994), Eq. 1, assuming Chl a = 0.85 Chl (a+b)
        scatteringCoefPAR = 1.0 - leafAbsorbancePAR ; //scattering coefficient for PAR
        scatteringCoefNIR = 1.0 - leafAbsorbanceNIR ; //scattering coefficient for NIR
        dum[0]= sqrt(1-scatteringCoefPAR);
        dum[1]= sqrt(1-scatteringCoefNIR);
        diffuseLightKPAR = diffuseLightK * dum[0] ;//extinction coeff of PAR, direct light
        diffuseLightKNIR = diffuseLightK * dum[1]; //extinction coeff of NIR radiation, direct light
        directLightKPAR  = directLightK * dum[0]; //extinction coeff of PAR, diffuse light
        directLightKNIR  = directLightK * dum[1]; //extinction coeff of NIR radiation, diffuse light
        //Canopy+soil reflection coefficients for direct, diffuse PAR, NIR radiation
        dum[2]= (1-dum[0]) / (1+dum[0]);
        dum[3]= (1-dum[1]) / (1+dum[1]);
        dum[4]= 2.0 * directLightK / (directLightK + diffuseLightK) ;
        directReflectionCoefficientNIR = directReflectionCoefficientPAR = dum[4] * dum[2];
        diffuseReflectionCoefficientNIR = diffuseReflectionCoefficientPAR = dum[4] * dum[3];
        //Incoming direct PAR and NIR (W m-2)
        directIncomingNIR = directIncomingPAR = weatherVariable.derived.myDirectIrradiance * 0.5 ;
        //Incoming diffuse PAR and NIR (W m-2)
        diffuseIncomingNIR = diffuseIncomingPAR = weatherVariable.derived.myDiffuseIrradiance * 0.5 ;
        //Preliminary computations
        dum[5]= diffuseIncomingPAR * (1.0-diffuseReflectionCoefficientPAR) * diffuseLightKPAR ;
        dum[6]= directIncomingPAR * (1.0-directReflectionCoefficientPAR) * directLightKPAR ;
        dum[7]= directIncomingPAR * (1.0-scatteringCoefPAR) * directLightK ;
        dum[8]=  diffuseIncomingNIR * (1.0-diffuseReflectionCoefficientNIR) * diffuseLightKNIR;
        dum[9]= directIncomingNIR * (1.0-directReflectionCoefficientNIR) * directLightKNIR;
        dum[10]= directIncomingNIR * (1.0-scatteringCoefNIR) * directLightKNIR ;
        dum[11]= UPSCALINGFUNC((diffuseLightKPAR+directLightK),leafAreaIndex);
        dum[12]= UPSCALINGFUNC((directLightKPAR+directLightK),leafAreaIndex);
        dum[13]= UPSCALINGFUNC((diffuseLightKPAR+directLightK),leafAreaIndex);
        dum[14]= UPSCALINGFUNC((directLightKNIR+directLightK),leafAreaIndex);
        dum[15]= UPSCALINGFUNC(directLightK,leafAreaIndex) - UPSCALINGFUNC((2.0*directLightK),leafAreaIndex) ;
        // PAR absorbed by sunlit (1) and shaded (2) big-leaf (W m-2) from Wang & Leuning 1998
        sunlit.absorbedPAR = dum[5] * dum[11] + dum[6] * dum[12] + dum[7] * dum[15] ;
        shaded.absorbedPAR = dum[5]*(UPSCALINGFUNC(diffuseLightKPAR,leafAreaIndex)- dum[11]) + dum[6]*(UPSCALINGFUNC(directLightKPAR,leafAreaIndex)- dum[12]) - dum[7] * dum[15];
        // NIR absorbed by sunlit (1) and shaded (2) big-leaf (W m-2) fromWang & Leuning 1998
        sunlitAbsorbedNIR = dum[8]*dum[13]+dum[9]*dum[14]+dum[10]*dum[15];
        shadedAbsorbedNIR = dum[8]*(UPSCALINGFUNC(diffuseLightKNIR,leafAreaIndex)-dum[13])+dum[9]*(UPSCALINGFUNC(directLightKNIR,leafAreaIndex)- dum[14]) - dum[10] * dum[15];
        // Long-wave radiation balance by sunlit (1) and shaded (2) big-leaf (W m-2) from Wang & Leuning 1998
        dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*pow(weatherVariable.myInstantTemp+ZEROCELSIUS,4); //negativo
        dum[16] *= diffuseLightK ;
        double emissivityLeaf, emissivitySoil;
        emissivityLeaf = 0.96 ; // supposed constant because variation is very small
        emissivitySoil= 0.94 ;   // supposed constant because variation is very small
        sunlitAbsorbedLW = (dum[16] * UPSCALINGFUNC((directLightK+diffuseLightK),leafAreaIndex))*emissivityLeaf+(1.0-emissivitySoil)*(emissivityLeaf-weatherVariable.derived.myEmissivitySky)* UPSCALINGFUNC((2*diffuseLightK),leafAreaIndex)* UPSCALINGFUNC((directLightK-diffuseLightK),leafAreaIndex);
        shadedAbsorbedLW = dum[16] * UPSCALINGFUNC(diffuseLightK,leafAreaIndex) - sunlitAbsorbedLW ;
        // Isothermal net radiation for sunlit (1) and shaded (2) big-leaf
        sunlit.isothermalNetRadiation= sunlit.absorbedPAR + sunlitAbsorbedNIR + sunlitAbsorbedLW ;
        shaded.isothermalNetRadiation = shaded.absorbedPAR + shadedAbsorbedNIR + shadedAbsorbedLW ;
    }
    else
    {
        sunlit.leafAreaIndex =  0.0 ;
        sunlit.absorbedPAR = 0.0 ;

        // TODO: non servono?
        sunlitAbsorbedNIR = 0.0 ;
        sunlitAbsorbedLW = 0.0 ;
        sunlit.isothermalNetRadiation =  0.0 ;

        shaded.leafAreaIndex = leafAreaIndex;
        shaded.absorbedPAR = 0.0 ;
        shadedAbsorbedNIR = 0.0 ;
        dum[16]= weatherVariable.derived.myLongWaveIrradiance -STEFAN_BOLTZMANN*pow(weatherVariable.myInstantTemp + ZEROCELSIUS,4) ;
        dum[16] *= diffuseLightK ;
        shadedAbsorbedLW= dum[16] * (UPSCALINGFUNC(diffuseLightK,leafAreaIndex) - UPSCALINGFUNC(directLightK+diffuseLightK,leafAreaIndex)) ;
        shaded.isothermalNetRadiation = shaded.absorbedPAR + shadedAbsorbedNIR + shadedAbsorbedLW ;
    }

    // Convert absorbed PAR into units of mol m-2 s-1
    sunlit.absorbedPAR *= 4.57E-6 ;
    shaded.absorbedPAR *= 4.57E-6 ;
}

void Crit3D_Hydrall::leafTemperature()
{
    if (sineSolarElevation > 0.001)
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
    heightReference = plantHeight + 5 ; // [m]
    dummy = 0.2 * leafAreaIndex ;
    zeroPlaneDisplacement = MINVALUE(plantHeight * (log(1+pow(dummy,0.166)) + 0.03*log(1+pow(dummy,6))), 0.99*plantHeight) ;
    if (dummy < 0.2) roughnessLength = 0.01 + 0.28*sqrt(dummy) * plantHeight ;
    else roughnessLength = 0.3 * plantHeight * (1.0 - zeroPlaneDisplacement/plantHeight);

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

    while( (20 > i++) && (fabs(sensibleHeat - sensibleHeatOld)> threshold)) {
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
        windSpeedTopCanopy = (frictionVelocity/KARM) * log((plantHeight - zeroPlaneDisplacement)/roughnessLength);
        windSpeedTopCanopy = MAXVALUE(windSpeedTopCanopy,1.0e-4);

        // Average leaf boundary-layer conductance cumulated over the canopy (m s-1)
        leafBoundaryLayerConductance = A*sqrt(windSpeedTopCanopy/(leafWidth()))*((2.0/BETA)*(1-exp(-BETA/2.0))) * leafAreaIndex;
        //       Total canopy aerodynamic conductance for momentum exchange (s m-1)
        canopyAerodynamicConductanceToMomentum= frictionVelocity / (windSpeed/frictionVelocity + (deviationFunctionForMomentum-deviationFunctionForHeat)/KARM);
        // Aerodynamic conductance for heat exchange (mol m-2 s-1)
        dummy =	(weatherVariable.atmosphericPressure/R_GAS)/(weatherVariable.myInstantTemp + ZEROCELSIUS);// conversion factor m s-1 into mol m-2 s-1
        aerodynamicConductanceForHeat =  ((canopyAerodynamicConductanceToMomentum*leafBoundaryLayerConductance)/(canopyAerodynamicConductanceToMomentum + leafBoundaryLayerConductance)) * dummy ; //whole canopy
        sunlit.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat * sunlit.leafAreaIndex/leafAreaIndex ;//sunlit big-leaf
        shaded.aerodynamicConductanceHeatExchange = aerodynamicConductanceForHeat - sunlit.aerodynamicConductanceHeatExchange ; //  shaded big-leaf
        // Canopy radiative conductance (mol m-2 s-1)
        radiativeConductance= 4*(weatherVariable.derived.slopeSatVapPressureVSTemp/weatherVariable.derived.psychrometricConstant)*(STEFAN_BOLTZMANN/HEAT_CAPACITY_AIR_MOLAR)*pow((weatherVariable.myInstantTemp + ZEROCELSIUS),3);
        // Total conductance to heat exchange (mol m-2 s-1)
        totalConductanceToHeatExchange =  aerodynamicConductanceForHeat + radiativeConductance; //whole canopy
        sunlit.totalConductanceHeatExchange = totalConductanceToHeatExchange * sunlit.leafAreaIndex/leafAreaIndex;	//sunlit big-leaf
        shaded.totalConductanceHeatExchange = totalConductanceToHeatExchange - sunlit.totalConductanceHeatExchange;  //shaded big-leaf

        // Temperature of big-leaf (approx. expression)
        if (sunlit.leafAreaIndex > 1.0E-6)
        {
            stomatalConductanceWater= (10.0/sunlit.leafAreaIndex); //dummy stom res for sunlit big-leaf
            //if (sunlit.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/sunlit.isothermalNetRadiation,0.5);
            sunlitDeltaTemp = ((stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceHeatExchange)
                              *weatherVariable.derived.psychrometricConstant*sunlit.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                              - weatherVariable.vaporPressureDeficit*(1+0.001*sunlit.isothermalNetRadiation))
                              /sunlit.totalConductanceHeatExchange/(weatherVariable.derived.psychrometricConstant
                              *(stomatalConductanceWater+1.0/sunlit.aerodynamicConductanceCO2Exchange)
                              +weatherVariable.derived.slopeSatVapPressureVSTemp/sunlit.totalConductanceHeatExchange);
        }
        else
        {
            sunlitDeltaTemp = 0.0;
        }
        sunlitDeltaTemp = 0.0;  // TODO: check

        sunlit.leafTemperature = weatherVariable.myInstantTemp + sunlitDeltaTemp	+ ZEROCELSIUS ; //sunlit big-leaf
        stomatalConductanceWater = 10.0/shaded.leafAreaIndex ; //dummy stom res for shaded big-leaf
        //if (shaded.isothermalNetRadiation > 100) stomatalConductanceWater *= pow(100/shaded.isothermalNetRadiation,0.5);
        shadedDeltaTemp = ((stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)*weatherVariable.derived.psychrometricConstant*shaded.isothermalNetRadiation/HEAT_CAPACITY_AIR_MOLAR
                           - weatherVariable.vaporPressureDeficit*(1+0.001*shaded.isothermalNetRadiation))/shaded.totalConductanceHeatExchange
                           /(weatherVariable.derived.psychrometricConstant*(stomatalConductanceWater + 1.0/shaded.aerodynamicConductanceHeatExchange)
                           + weatherVariable.derived.slopeSatVapPressureVSTemp/shaded.totalConductanceHeatExchange);
        shadedDeltaTemp = 0.0;
        shaded.leafTemperature = weatherVariable.myInstantTemp + shadedDeltaTemp + ZEROCELSIUS;  //shaded big-leaf
        // Sensible heat flux from the whole canopy
        sensibleHeat = HEAT_CAPACITY_AIR_MOLAR * (sunlit.aerodynamicConductanceHeatExchange*sunlitDeltaTemp + shaded.aerodynamicConductanceHeatExchange*shadedDeltaTemp);
    }

    if (isAmphystomatic) aerodynamicConductanceToCO2 = 0.78 * aerodynamicConductanceForHeat; //amphystomatous species. Ratio of diffusivities from Wang & Leuning 1998
    else aerodynamicConductanceToCO2 = 0.78 * (canopyAerodynamicConductanceToMomentum * leafBoundaryLayerConductance)/(leafBoundaryLayerConductance + 2.0*canopyAerodynamicConductanceToMomentum) * dummy; //hypostomatous species

    sunlit.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * sunlit.leafAreaIndex/leafAreaIndex ; //sunlit big-leaf
    shaded.aerodynamicConductanceCO2Exchange = aerodynamicConductanceToCO2 * shaded.leafAreaIndex/leafAreaIndex ;  //shaded big-leaf
}

double  Crit3D_Hydrall::leafWidth()
{
    // la funzione deve essere scritta secondo regole che possono fr variare lo spessore in base alla fenologia
    // come per la vite?
    return myLeafWidth;
}

void Crit3D_Hydrall::upscale()
{
    //cultivar->parameterWangLeuning.maxCarbonRate era input, ora da prendere da classe e leggere da tipo di pianta
    double maxCarboxRate = 0.0;

    // taken from Hydrall Model, Magnani UNIBO
    static double BETA = 0.5 ;

    //CONSTANTS (only used here)
    const double CRD = 18.7;             /*!< scaling factor in RD0 response to temperature (-)  */
    const double HARD = 46.39;           /*!< activation energy of RD0 (kJ mol-1)   */
    const double CKC = 38.05;            /*!< scaling factor in KCT0 response to temperature (-)  */
    const double CKO = 20.30;            /*!< scaling factor in KOT0 response to temperature (-)  */
    const double HAKO = 36.38;           /*!< activation energy of KOT0 (kJ mol-1)  */
    const double HAGSTAR = 37.83;        /*!< activation energy of Gamma_star (kJ mol-1)  */
    const double HAKC = 79.43;           /*!< activation energy of KCT0 (kJ mol-1)  */
    const double CGSTAR = 19.02;         /*!< scaling factor in Gamma_star response to temperature (-)  */
    const double HAJM = 43.9;            /*!< activation energy of JMOP (kJ mol-1 e-)  */
    const double HDEACTIVATION = 200;    /*!< deactivation energy from Kattge & Knorr 2007 (kJ mol-1)  */
    const double HAVCM = 65.33;          /*!< activation energy of VCMOP (kJ mol-1)  */


    double dum[10],darkRespirationT0;
    double optimalCarboxylationRate,optimalElectronTransportRate ;
    double leafConvexityFactor ;
    //     Preliminary computations
    dum[0]= R_GAS/1000.0 * sunlit.leafTemperature ; //[kJ mol-1]
    dum[1]= R_GAS/1000.0 * shaded.leafTemperature ;
    dum[2]= sunlit.leafTemperature - ZEROCELSIUS ; // [oC]
    dum[3]= shaded.leafTemperature - ZEROCELSIUS ;



    // optimalCarboxylationRate = (nitrogen.interceptLeaf + nitrogen.slopeLeaf * nitrogen.leafNitrogen/specificLeafArea*1000)*1e-6; // carboxylation rate based on nitrogen leaf
    optimalCarboxylationRate = maxCarboxRate * 1.0e-6; // [mol m-2 s-1] from Greer et al. 2011
    darkRespirationT0 = 0.0089 * optimalCarboxylationRate ;
    //   Adjust unit dark respiration rate for temperature (mol m-2 s-1)
    sunlit.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[0])* UPSCALINGFUNC((directLightK + diffuseLightKPAR),leafAreaIndex); //sunlit big-leaf
    shaded.darkRespiration = darkRespirationT0 * exp(CRD - HARD/dum[1]); //shaded big-leaf
    shaded.darkRespiration *= (UPSCALINGFUNC(diffuseLightKPAR,leafAreaIndex) - UPSCALINGFUNC((directLightK + diffuseLightKPAR),leafAreaIndex)); //originariamente statePlant.stateGrowth.leafAreaIndex, è giusto quel leafAreaIndex?
    double entropicFactorElectronTransporRate = (-0.75*(weatherVariable.last30DaysTAvg)+660);  // entropy term for J (kJ mol-1 oC-1)
    double entropicFactorCarboxyliation = (-1.07*(weatherVariable.last30DaysTAvg)+668); // entropy term for VCmax (kJ mol-1 oC-1)
    if (sineSolarElevation > 1.0e-3)
    {
        //Stomatal conductance to CO2 in darkness (molCO2 m-2 s-1)
        sunlit.minimalStomatalConductance = parameterWangLeuningFix.stomatalConductanceMin  * UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex)	;
        shaded.minimalStomatalConductance = parameterWangLeuningFix.stomatalConductanceMin  * (UPSCALINGFUNC(diffuseLightKPAR,leafAreaIndex) - UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex));
        // Carboxylation rate
        //sunlit.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[0]); //sunlit big leaf
        //shaded.maximalCarboxylationRate = optimalCarboxylationRate * exp(CVCM - HAVCM/dum[1]); //shaded big leaf
        sunlit.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,sunlit.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuningFix.optimalTemperatureForPhotosynthesis); //sunlit big leaf
        shaded.maximalCarboxylationRate = optimalCarboxylationRate * acclimationFunction(HAVCM*1000,HDEACTIVATION*1000,shaded.leafTemperature,entropicFactorCarboxyliation,parameterWangLeuningFix.optimalTemperatureForPhotosynthesis); //shaded big leaf
        // Scale-up maximum carboxylation rate (mol m-2 s-1)
        sunlit.maximalCarboxylationRate *= UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex);
        shaded.maximalCarboxylationRate *= (UPSCALINGFUNC(diffuseLightKPAR,leafAreaIndex) - UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex));
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
        sunlit.maximalElectronTrasportRate = optimalElectronTransportRate * acclimationFunction(HAJM*1000,HDEACTIVATION*1000,sunlit.leafTemperature,entropicFactorElectronTransporRate,parameterWangLeuningFix.optimalTemperatureForPhotosynthesis);
        shaded.maximalElectronTrasportRate = optimalElectronTransportRate * acclimationFunction(HAJM*1000,HDEACTIVATION*1000,shaded.leafTemperature,entropicFactorElectronTransporRate,parameterWangLeuningFix.optimalTemperatureForPhotosynthesis);

        // Compute maximum PSII quantum yield, light-acclimated (mol e- mol-1 quanta absorbed)
        sunlit.quantumYieldPS2 = 0.352 + 0.022*dum[2] - 3.4E-4*pow(dum[2],2);      //sunlit big-leaf
        shaded.quantumYieldPS2 = 0.352 + 0.022*dum[3] - 3.4E-4*pow(dum[3],2);      //shaded big-leaf
        // Compute convexity factor of light response curve (-)
        // The value derived from leaf Chl content is modified for temperature effects, according to Bernacchi et al. (2003)
        leafConvexityFactor = 1 - myChlorophyllContent * 6.93E-4 ;    //from Pons & Anten (2004), fig. 3b
        sunlit.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[2] - 3.7E-4*pow(dum[2],2));  //sunlit big-leaf
        shaded.convexityFactorNonRectangularHyperbola = leafConvexityFactor/0.98 * (0.76 + 0.018*dum[3] - 3.7E-4*pow(dum[3],2));  //shaded big-leaf
        // Scale-up potential electron transport of sunlit big-leaf (mol m-2 s-1)
        sunlit.maximalElectronTrasportRate *= UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex);
        // Adjust electr transp of sunlit big-leaf for PAR effects (mol e- m-2 s-1)
        dum[4]= sunlit.absorbedPAR * sunlit.quantumYieldPS2 * BETA ; //  potential PSII e- transport of sunlit big-leaf (mol m-2 s-1)
        dum[5]= dum[4] + sunlit.maximalElectronTrasportRate ;
        dum[6]= dum[4] * sunlit.maximalElectronTrasportRate ;
        sunlit.maximalElectronTrasportRate = (dum[5] - sqrt(pow(dum[5],2) - 4.0*sunlit.convexityFactorNonRectangularHyperbola*dum[6])) / (2.0*sunlit.convexityFactorNonRectangularHyperbola);
        // Scale-up potential electron transport of shaded big-leaf (mol m-2 s-1)
        // The simplified formulation proposed by de Pury & Farquhar (1999) is applied
        shaded.maximalElectronTrasportRate *= (UPSCALINGFUNC(diffuseLightKPAR,leafAreaIndex) - UPSCALINGFUNC((directLightK+diffuseLightKPAR),leafAreaIndex));
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

double Crit3D_Hydrall::acclimationFunction(double Ha , double Hd, double leafTemp,
                                             double entropicTerm,double optimumTemp)
{
    // taken from Hydrall Model, Magnani UNIBO
    return exp(Ha*(leafTemp - optimumTemp)/(optimumTemp*R_GAS*leafTemp))
           *(1+exp((optimumTemp*entropicTerm-Hd)/(optimumTemp*R_GAS)))
           /(1+exp((leafTemp*entropicTerm-Hd)/(leafTemp*R_GAS))) ;
}
