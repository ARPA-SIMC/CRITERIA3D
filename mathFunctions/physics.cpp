/*-----------------------------------------------------------------------------------

    CRITERIA 3D
    Copyright (C) 2011 Fausto Tomei, Gabriele Antolini, Alberto Pistocchi,
    Antonio Volta, Giulia Villani, Marco Bittelli

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpa.emr.it
    fausto.tomei@gmail.com
    gantolini@arpa.emr.it
    alberto.pistocchi@gecosistema.it
    marco.bittelli@unibo.it
-----------------------------------------------------------------------------------*/

#include <math.h>
#include "physics.h"
#include "basicMath.h"
#include "commonConstants.h"


/*!
 * \brief pressure [Pa]
 * \param altitude in meters above the sea level [m]
 * \return atmospheric pressure (Pa)
 */
double pressureFromAltitude(double height)
{	/* from Allen et al., 1994.
    An update for the calculation of reference evapotranspiration.
    ICID Bulletin 43: 65
    */

    double pressure = P0 * pow(1 + height * LAPSE_RATE_MOIST_AIR / TP0, - GRAVITY / (LAPSE_RATE_MOIST_AIR * R_DRY_AIR));
    return pressure;
}


/*!
 * \brief Boyle-Charles law
 * \param myPressure (Pa)
 * \param myT (K)
 * \return air molar density [mol m-3]
 */
double airMolarDensity(double myPressure, double myT)
{
    return 44.65 * (myPressure / P0) * (ZEROCELSIUS / myT);
}


double volumetricLatentHeatVaporization(double myPressure, double myT)
// [J m-3] latent heat of vaporization
{
    double rhoAir = airMolarDensity(myPressure, myT); // [mol m-3] molar density of air
    return (rhoAir * (45144. - 48. * (myT - ZEROCELSIUS)));	// Campbell 1994
}


/*!
 * \brief Vapour Pressure Deficit (VPD)
 * \param air temperature (C)
 * \param relative humidity (%)
 * \return Vapour pressure deficit [hPa]
 */
double vapourPressureDeficit(double tAir, double relativeHumidity)
{
    // check relative humidity
    if (relativeHumidity < 1) relativeHumidity = 1.0;
    if (relativeHumidity > 100) relativeHumidity = 100.0;

    return (1.0 - relativeHumidity / 100.0) * saturationVaporPressure(tAir) / 100.0;
}


double vaporPressureFromConcentration(double myConcentration, double myT)
// [Pa] convert vapor partial pressure from concentration in kg m-3
{
    return (myConcentration * R_GAS * myT / MH2O);
}


double vaporConcentrationFromPressure(double myPressure, double myT)
// [kg m-3] compute vapor concentration from pressure (Pa) and temperature (K)
{
    return (myPressure * MH2O / (R_GAS * myT));
}


double airVolumetricSpecificHeat(double myPressure, double myT)
{ // (J m-3 K-1) volumetric specific heat of air

    double myMolarDensity = airMolarDensity(myPressure, myT); // mol m-3
    double mySpHeat = (HEAT_CAPACITY_AIR_MOLAR * myMolarDensity);
    return (mySpHeat);
}


/*!
 * \brief [Pa] saturation vapor pressure
 * \param myTCelsius [degC]
 * \return result
 */
double saturationVaporPressure(double myTCelsius)
{
    return 611 * exp(17.502 * myTCelsius / (myTCelsius + 240.97));
}


/*!
 * \brief [kPa degC-1] slope of saturation vapor pressure curve
 * \param airTCelsius (degC)
 * \param satVapPressure (kPa)
 * \return result
 */
double saturationSlope(double airTCelsius, double satVapPressure)
{
    return (4098. * satVapPressure / ((237.3 + airTCelsius) * (237.3 + airTCelsius)));
}


double getAirVaporDeficit(double myT, double myVapor)
{
    double myVaporPressure = saturationVaporPressure(myT - ZEROCELSIUS);
    double mySatVapor = vaporConcentrationFromPressure(myVaporPressure, myT);
    return (mySatVapor - myVapor);
}


/*!
 * \brief [J kg-1] latent heat of vaporization
 * \param myTCelsius
 * \return result
 */
double latentHeatVaporization(double myTCelsius)
{
    return (2501000. - 2369.2 * myTCelsius);
}


/*!
 * \brief [kPa °C-1] psychrometric instrument constant
 * \param myPressure [kPa]
 * \param myTemp [°C]
 * \return result
 */
double psychro(double myPressure, double myTemp)
{
    return CP * myPressure / (RATIO_WATER_VD * latentHeatVaporization(myTemp));
}


double AirDensity(double myTemperature, double myRelativeHumidity)
{
    double totalPressure;       // air total pressure (Pa)
    double vaporPressure;		// vapor partial pressure (Pa)
    double satVaporPressure;    // saturation vapor partial pressure (Pa)
    double myDensity;			// air density (kg m-3)

    satVaporPressure = saturationVaporPressure(myTemperature - ZEROCELSIUS);
    vaporPressure = (satVaporPressure * myRelativeHumidity);
    totalPressure = 101300;

    myDensity = (totalPressure / (R_DRY_AIR * myTemperature)) * (1 - (0.378 * vaporPressure / totalPressure));

    return (myDensity);
}


/*!
* \brief computes aerodynamic conductance
* \param heightTemperature: reference height for temperature and humidity measurement (m)
* \param heightWind: reference height for wind measurement (m)
* \param mySoilSurfaceTemperature: soil temperature (K)
* \param myRoughnessHeight: roughness height (m)
* \param myAirTemperature: air temperature (K)
* \param myWindSpeed: wind speed (m s-1)
* \return aerodynamic conductance for heat and vapor [m s-1]
* from Campbell Norman 1998
*/
double aerodynamicConductance(double heightTemperature,
                              double heightWind,
                              double soilSurfaceTemperature,
                              double roughnessHeight,
                              double airTemperature,
                              double windSpeed)
{
    double K = NODATA;				    // (m s-1) aerodynamic conductance
    double psiM, psiH;					// () diabatic correction factors for momentum and for heat
    double uStar;						// (m s-1) friction velocity
    double zeroPlane;					// (m) zero place displacement
    double roughnessMomentum;           // () surface roughness parameter for momentum
    double roughnessHeat;				// () surface roughness parameter for heat
    double Sp;                          // () stability parameter
    double H;                           // (W m-2) sensible heat flux
    double Ch;                          // (J m-3 K-1) volumetric specific heat of air

    windSpeed = MAXVALUE(windSpeed, 0.1);

    zeroPlane = 0.77 * roughnessHeight;
    roughnessMomentum = 0.13 * roughnessHeight;
    roughnessHeat = 0.2 * roughnessMomentum;

    psiM = 0.;
    psiH = 0.;
    Ch = airVolumetricSpecificHeat(pressureFromAltitude(heightWind), airTemperature);

    for (short i = 1; i <= 3; i++)
    {
        uStar = VON_KARMAN_CONST * windSpeed / (log((heightWind - zeroPlane + roughnessMomentum) / roughnessMomentum) + psiM);
        K = VON_KARMAN_CONST * uStar / (log((heightTemperature - zeroPlane + roughnessHeat) / roughnessHeat) + psiH);
        H = K * Ch * (soilSurfaceTemperature - airTemperature);
        Sp = -VON_KARMAN_CONST * heightWind * GRAVITY * H / (Ch * airTemperature * (pow(uStar, 3)));
        if (Sp > 0)
        {// stability
            psiH = 6 * log(1 + Sp);
            psiM = psiH;
        }
        else
        {// unstability
            psiH = -2 * log((1 + sqrt(1 - 16 * Sp)) / 2);
            psiM = 0.6 * psiH;
        }
    }

    return (K);

}


/*!
* \brief computes aerodynamic conductance for an open water surface
* \param myHeight: reference height (m)
* \param myWaterBodySurface: surface of water body (m2)
* \param myAirTemperature: air temperature (m)
* \param myWindSpeed: wind speed (m s-1)
* \return aerodynamic conductance for heat and vapor [m s-1]
* McJannet et al 2008
*/
double aerodynamicConductanceOpenwater(double myHeight, double myWaterBodySurface, double myAirTemperature, double myWindSpeed10)
{
    double myPressure;		// Pa
    double myT;				// K
    double myVolSpecHeat;	// J m-3 K-1
    double myPsycro;		// kPa K-1
    double windFunction;	// (MJ m-2 d-1 kPa-1) wind function (Sweers 1976)

    myPressure = pressureFromAltitude(myHeight);
    myT = myAirTemperature;
    myVolSpecHeat = airVolumetricSpecificHeat(myPressure, myT);
    myPsycro = psychro(myPressure / 1000, myT);

    windFunction = pow((5. / (myWaterBodySurface * 1000000)), 0.05) * (3.8 + 1.57 * myWindSpeed10);
    windFunction *= 1000000. / DAY_SECONDS; //to J m-2 s-1 kPa

    return (1. / (myVolSpecHeat / (myPsycro * windFunction)));
}


float erosivityFactor(std::vector<float> values, int nValues)
{

    double erosivityFactor = NODATA;

    for (int i = 0; i < nValues; i++)
    {
        if ( values[i] != NODATA)
        {
            if (erosivityFactor == NODATA)
            {
                erosivityFactor = 0;
            }
            if ( (values[i] > 0) && (values[i] != NODATA))
            {
                erosivityFactor += 0.11 * pow(values[i], 1.82);
            }
        }
    }

    return float(erosivityFactor);
}


float rainIntensity(std::vector<float> values, int nValues, float rainfallThreshold)
{

    if (nValues == 0)
        return NODATA;

    float rainySum = 0;
    int rainyDays = 0;

    for (int i = 0; i < nValues; i++)
    {
        if (values[i] != NODATA)
        {
            if (values[i] > rainfallThreshold)
            {
                rainyDays = rainyDays + 1;
                rainySum = rainySum + values[i];
            }
        }
    }

    if (rainyDays == 0)
        return 0;
    else
        return rainySum / rainyDays;

}


int windPrevailingDir(std::vector<float> intensity, std::vector<float> dir, int nValues, bool useIntensity)
{
    float windInt = NODATA;
    float windDir;
    float delta = 45.f;
    unsigned long quadr;
    unsigned long nrClass = 8;
    unsigned long prevailingDir = 0;
    bool condition;

    std::vector<float> intQuadr(nrClass);
    std::vector<int> dirQuadr(nrClass);

    for (int i = 0; i < nValues; i++)
    {
        if (useIntensity)
        {
            windInt = intensity[i];
        }
        windDir = dir[i];
        if (useIntensity)
        {
            if ( windDir != NODATA && windInt != NODATA && windDir >= 0 && windInt > 0 )
            {
                quadr = (unsigned long) round(windDir / delta);
                if (quadr == 0)
                {
                    quadr = nrClass;
                }
                dirQuadr[quadr] = dirQuadr[quadr] + 1;
                intQuadr[quadr] = intQuadr[quadr] + windInt;
            }
        }
        else
        {
            if (windDir != NODATA && windDir >= 0)
            {
                quadr = int(round(windDir / delta));
                if (quadr == 0)
                {
                    quadr = nrClass;
                }
                dirQuadr[quadr] = dirQuadr[quadr] + 1;
            }
        }
    }

    for (quadr = 0; quadr < nrClass; quadr++ )
    {
        if (dirQuadr[quadr] > 0)
        {
            if (useIntensity)
            {
                condition = (dirQuadr[quadr] > dirQuadr[prevailingDir]) || ((dirQuadr[quadr] = dirQuadr[prevailingDir]) && (intQuadr[quadr] > intQuadr[prevailingDir]));
            }
            else
            {
                if (dirQuadr[quadr] > dirQuadr[prevailingDir])
                {
                    condition = true;
                }
                else
                {
                    condition = false;
                }

            }
            if (prevailingDir == 0)
            {
                prevailingDir = quadr;
            }
            else if (condition)
            {
                prevailingDir = quadr;
            }
        }
    }

    if (prevailingDir == 0)
    {
        return int(NODATA);
    }
    else
    {
        return int(prevailingDir * delta);
    }

}

float timeIntegrationFunction(std::vector<float> values, float timeStep)
{

    if (values.size() == 0)
        return NODATA;

    float sum = NODATA;

    for (unsigned i = 0; i < values.size(); i++)
        if (! isEqual(values[i], NODATA))
            sum = (isEqual(sum, NODATA) ? values[i] * timeStep : sum + values[i] * timeStep);

    return sum;

}
