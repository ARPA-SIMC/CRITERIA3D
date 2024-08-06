/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/

#include <math.h>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "quality.h"
#include "physics.h"
#include "meteo.h"
#include "color.h"

Crit3DMeteoSettings::Crit3DMeteoSettings()
{
    initialize();
}

void Crit3DMeteoSettings::initialize()
{
    minimumPercentage = DEFAULT_MIN_PERCENTAGE;
    rainfallThreshold = DEFAULT_RAINFALL_THRESHOLD;
    thomThreshold = DEFAULT_THOM_THRESHOLD;
    temperatureThreshold = DEFAULT_TEMPERATURE_THRESHOLD;
    transSamaniCoefficient = DEFAULT_TRANSMISSIVITY_SAMANI;
    windIntensityDefault = DEFAULT_WIND_INTENSITY;
    hourlyIntervals = DEFAULT_HOURLY_INTERVALS;
    automaticTavg = DEFAULT_AUTOMATIC_TMED;
    automaticET0HS = DEFAULT_AUTOMATIC_ET0HS;
}

float Crit3DMeteoSettings::getMinimumPercentage() const
{
    return minimumPercentage;
}

void Crit3DMeteoSettings::setMinimumPercentage(float value)
{
    minimumPercentage = value;
}

float Crit3DMeteoSettings::getRainfallThreshold() const
{
    return rainfallThreshold;
}

void Crit3DMeteoSettings::setRainfallThreshold(float value)
{
    rainfallThreshold = value;
}

float Crit3DMeteoSettings::getTransSamaniCoefficient() const
{
    return transSamaniCoefficient;
}

void Crit3DMeteoSettings::setTransSamaniCoefficient(float value)
{
    transSamaniCoefficient = value;
}

int Crit3DMeteoSettings::getHourlyIntervals() const
{
    return hourlyIntervals;
}

void Crit3DMeteoSettings::setHourlyIntervals(int value)
{
    hourlyIntervals = value;
}

float Crit3DMeteoSettings::getWindIntensityDefault() const
{
    return windIntensityDefault;
}

void Crit3DMeteoSettings::setWindIntensityDefault(float value)
{
    windIntensityDefault = value;
}

float Crit3DMeteoSettings::getThomThreshold() const
{
    return thomThreshold;
}

float Crit3DMeteoSettings::getTemperatureThreshold() const
{
    return temperatureThreshold;
}

void Crit3DMeteoSettings::setThomThreshold(float value)
{
    thomThreshold = value;
}

void Crit3DMeteoSettings::setTemperatureThreshold(float value)
{
    temperatureThreshold = value;
}

bool Crit3DMeteoSettings::getAutomaticTavg() const
{
    return automaticTavg;
}

void Crit3DMeteoSettings::setAutomaticTavg(bool value)
{
    automaticTavg = value;
}

bool Crit3DMeteoSettings::getAutomaticET0HS() const
{
    return automaticET0HS;
}

void Crit3DMeteoSettings::setAutomaticET0HS(bool value)
{
    automaticET0HS = value;
}

Crit3DClimateParameters::Crit3DClimateParameters()
{
    tmin.resize(12);
    tmax.resize(12);
    tdmin.resize(12);
    tdmax.resize(12);
    tminLapseRate.resize(12);
    tmaxLapseRate.resize(12);
    tdMinLapseRate.resize(12);
    tdMaxLapseRate.resize(12);

    for (unsigned i=0; i<12; i++)
    {
        tmin[i] = NODATA;
        tmax[i] = NODATA;
        tdmin[i] = NODATA;
        tdmax[i] = NODATA;
        tminLapseRate[i] = NODATA;
        tmaxLapseRate[i] = NODATA;
        tdMinLapseRate[i] = NODATA;
        tdMaxLapseRate[i] = NODATA;
    }
}


float computeTminHourlyWeight(int myHour)
{
    if (myHour >= 6 && myHour <= 14)
        return (1 - (myHour - 6) / 8.f);
    else if (myHour > 14)
        return (1 - MINVALUE(24 - myHour + 6, 12) / 12.f);
    else
        return (1 - (6 - myHour) / 12.f);
}


float Crit3DClimateParameters::getClimateLapseRate(meteoVariable myVar, Crit3DTime myTime)
{
    Crit3DDate myDate = myTime.date;
    int myHour = myTime.getNearestHour();

    // TODO improve!
    if (myDate.isNullDate() || myHour == NODATA)
        return -0.006f;

    unsigned int indexMonth = unsigned(myDate.month - 1);

    if (myVar == dailyAirTemperatureMin)
        return tminLapseRate[indexMonth];
    else if (myVar == dailyAirTemperatureMax)
        return tmaxLapseRate[indexMonth];
    else if (myVar == dailyAirTemperatureAvg)
        return (tmaxLapseRate[indexMonth] + tminLapseRate[indexMonth]) / 2;
    else
    {
        float lapseTmin, lapseTmax;
        if (myVar == airTemperature)
        {
            lapseTmin = tminLapseRate[indexMonth];
            lapseTmax = tmaxLapseRate[indexMonth];
        }
        else if (myVar == airDewTemperature)
        {
            lapseTmin = tdMinLapseRate[indexMonth];
            lapseTmax = tdMaxLapseRate[indexMonth];
        }
        else
            return NODATA;

        float tminWeight = computeTminHourlyWeight(myHour);
        return (lapseTmin * tminWeight + lapseTmax * (1 - tminWeight));
    }
}

float Crit3DClimateParameters::getClimateLapseRate(meteoVariable myVar, int month)
{
    // TODO improve!
    if (month == NODATA)
        return -0.006f;

    unsigned int indexMonth = unsigned(month - 1);

    if (myVar == dailyAirTemperatureMin)
        return tminLapseRate[indexMonth];
    else if (myVar == dailyAirTemperatureMax)
        return tmaxLapseRate[indexMonth];
    else if (myVar == dailyAirTemperatureAvg)
        return (tmaxLapseRate[indexMonth] + tminLapseRate[indexMonth]) / 2;
    else
        return NODATA;
}


float Crit3DClimateParameters::getClimateVar(meteoVariable myVar, int month, float height, float refHeight)
{
    unsigned int indexMonth = unsigned(month - 1);
    float climateVar = NODATA;

    switch(myVar)
    {
    case dailyAirTemperatureMin:
        climateVar = tmin[indexMonth];
        break;
    case dailyAirTemperatureMax:
        climateVar = tmax[indexMonth];
        break;
    case dailyAirRelHumidityMin:
        climateVar = tdmin[indexMonth];
        break;
    case dailyAirRelHumidityMax:
        climateVar = tdmax[indexMonth];
        break;
    default:
        return NODATA;
    }

    if (climateVar != NODATA && height != NODATA)
    {
        climateVar += getClimateLapseRate(myVar, month) * (height - refHeight);
    }

    return climateVar;
}


float tDewFromRelHum(float RH, float T)
{
    if (isEqual(RH, NODATA) || isEqual(T, NODATA) || RH == 0)
        return NODATA;

    RH = MINVALUE(100, RH);

    double mySaturatedVaporPres = exp((16.78 * double(T) - 116.9) / (double(T) + 237.3));
    double actualVaporPres = double(RH) / 100. * mySaturatedVaporPres;
    return float((log(actualVaporPres) * 237.3 + 116.9) / (16.78 - log(actualVaporPres)));
}


double tDewFromRelHum(double RH, double T)
{
    if (isEqual(RH, NODATA) || isEqual(T, NODATA) || RH == 0)
        return NODATA;

    RH = MINVALUE(100, RH);

    double mySaturatedVaporPres = exp((16.78 * T - 116.9) / (T + 237.3));
    double actualVaporPres = RH / 100. * mySaturatedVaporPres;
    return (log(actualVaporPres) * 237.3 + 116.9) / (16.78 - log(actualVaporPres));
}


float relHumFromTdew(float Td, float T)
{
    if (isEqual(Td, NODATA) || isEqual(T, NODATA))
        return NODATA;

    double d = 237.3;
    double c = 17.2693882;
    double esp = 1 / (double(T) + d);
    double rh = pow(exp((c * double(Td)) - ((c * double(T) / (double(T) + d))) * (double(Td) + d)), esp);
    rh *= 100;

    if (rh > 100) return 100;

    return std::max(1.f, float(rh));
}


double relHumFromTdew(double Td, double T)
{
    if (isEqual(Td, NODATA) || isEqual(T, NODATA))
        return NODATA;

    double d = 237.3;
    double c = 17.2693882;
    double esp = 1 / (double(T) + d);
    double rh = pow(exp((c * Td) - (c * T / (T + d)) * (Td + d)), esp);
    rh *= 100;

    if (rh > 100) return 100;

    return std::max(1., rh);
}


double dailyExtrRadiation(double myLat, int myDoy)
{
    /*!
    2011 GA
    da quaderno FAO
    MJ m-2 d-1
    */

    double OmegaS;                               /*!< [rad] sunset hour angle */
    double Phi;                                  /*!< [rad] latitude in radiants */
    double delta;                                /*!< [rad] solar declination */
    double dr;                                   /*!< [-] inverse Earth-Sun relative distance */
    double doy = double(myDoy);

    Phi = PI / 180. * myLat;
    delta = 0.4093 * sin((2. * PI / 365.) * doy - 1.39);
    dr = 1. + 0.033 * cos(2. * PI * doy / 365.);
    OmegaS = acos(-tan(Phi) * tan(delta));

    return SOLAR_CONSTANT * DAY_SECONDS / 1000000. * dr / PI * (OmegaS * sin(Phi) * sin(delta) + cos(Phi) * cos(delta) * sin(OmegaS));
}

float computeDailyBIC(float prec, float etp)
{

    Crit3DQuality qualityCheck;

    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    quality::qualityType qualityPrec = qualityCheck.syntacticQualitySingleValue(dailyPrecipitation, prec);
    quality::qualityType qualityETP = qualityCheck.syntacticQualitySingleValue(dailyReferenceEvapotranspirationHS, etp);
    if (qualityPrec == quality::accepted && qualityETP == quality::accepted)
    {
            return (prec - etp);
    }
    else
        return NODATA;

}

float dailyThermalRange(float Tmin, float Tmax)
{

    Crit3DQuality qualityCheck;

    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, Tmin);
    quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, Tmax);
    if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
        return (Tmax - Tmin);
    else
        return NODATA;

}

float dailyAverageT(float Tmin, float Tmax)
{
        Crit3DQuality qualityCheck;

        // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
        quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, Tmin);
        quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, Tmax);
        if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
            return ( (Tmin + Tmax) / 2) ;
        else
            return NODATA;
}


float dailyEtpHargreaves(float Tmin, float Tmax, Crit3DDate date, double latitude, Crit3DMeteoSettings* meteoSettings)
{
    Crit3DQuality qualityCheck;

    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, Tmin);
    quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, Tmax);
    int dayOfYear = getDoyFromDate(date);
    if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
            return float(ET0_Hargreaves(meteoSettings->getTransSamaniCoefficient(), latitude, dayOfYear, Tmax, Tmin));
    else
        return NODATA;
}


float dewPoint(float relHumAir, float tempAir)
{
    if (relHumAir == NODATA || relHumAir == 0 || tempAir == NODATA)
        return NODATA;

    relHumAir = MINVALUE(100, relHumAir);

    double saturatedVaporPres = exp((16.78 * tempAir - 116.9) / (tempAir + 237.3));
    double actualVaporPres = relHumAir / 100 * saturatedVaporPres;
    return float((log(actualVaporPres) * 237.3 + 116.9) / (16.78 - log(actualVaporPres)));
}


/*!
 * \brief [] net surface emissivity
 * \param myVP
 * \return result
 */
double emissivityFromVaporPressure(double myVP)
{
    return 0.34 - 0.14 * sqrt(myVP);
}

/*!
 * \brief hourly leaf wetness (0/1)
 * \param prec [mm] total precipitation
 * \param relHumidity [%] air relative humidity
 * \return leafW
 */
bool computeLeafWetness(double prec, double relHumidity, short* leafW)
{
    *leafW = 0;

    if (isEqual(relHumidity, NODATA) || isEqual(prec, NODATA)) return false;

    if (prec > 0 || relHumidity > DEFAULT_LEAFWETNESS_RH_THRESHOLD)
        *leafW = 1;

    return true;
}

/*!
 * \brief 2016 GA. comments: G is currently ignored (if heat flux is active, should be added)
 * \param myDOY [] day of year
 * \param myElevation
 * \param myLatitude [°] latitude in decimal degrees
 * \param myTmin [°C] daily minimum temperature
 * \param myTmax [°C] daily maximum temperature
 * \param myTminDayAfter
 * \param myUmed [%] daily average relative humidity
 * \param myVmed10 [m s-1] daily average wind intensity
 * \param mySWGlobRad [MJ m-2 d-1] daily global short wave radiation
 * \return result
 */
double ET0_Penman_daily(int myDOY, double myElevation, double myLatitude,
                        double myTmin, double myTmax, double myTminDayAfter,
                        double myUmed, double myVmed10, double mySWGlobRad)
{
        double MAXTRANSMISSIVITY = 0.75;

        double myPressure;                   /*!<  [kPa] atmospheric pressure */
        double myDailySB;                    /*!<  [MJ m-2 d-1 K-4] Stefan Boltzmann constant */
        double myPsychro;                    /*!<  [kPa °C-1] psychrometric instrument constant */
        double myTmed;                       /*!<  [°C] daily average temperature */
        double myTransmissivity;             /*!<  [] global atmospheric trasmissivity */
        double myVapPress;                   /*!<  [kPa] actual vapor pressure */
        double mySatVapPress;                /*!<  [kPa] actual average vapor pressure */
        double myExtraRad;                   /*!<  [MJ m-2 d-1] extraterrestrial radiation */
        double mySWNetRad;                   /*!<  [MJ m-2 d-1] net short wave radiation */
        double myLWNetRad;                   /*!<  [MJ m-2 d-1] net long wave emitted radiation */
        double myNetRad;                     /*!<  [MJ m-2 d-1] net surface radiation */
        double delta;                        /*!<  [kPa °C-1] slope of vapour pressure curve */
        double vmed2;                        /*!<  [m s-1] average wind speed estimated at 2 meters */
        double EvapDemand;                   /*!<  [mm d-1] evaporative demand of atmosphere */
        double myEmissivity;                 /*!<  [] surface emissivity */
        double myLambda;                     /*!<  [MJ kg-1] latent heat of vaporization */


        if (myTmin == NODATA || myTmax == NODATA || myVmed10 == NODATA || myUmed == NODATA || myUmed < 0 || myUmed > 100 || myTminDayAfter == NODATA)
            return NODATA;

        myTmed = 0.5 * (myTmin + myTmax);

        myExtraRad = dailyExtrRadiation(myLatitude, myDOY);
        if (myExtraRad > 0)
            myTransmissivity = MINVALUE(MAXTRANSMISSIVITY, mySWGlobRad / myExtraRad);
        else
            myTransmissivity = 0;

        myPressure = 101.3 * pow(((293 - 0.0065 * myElevation) / 293), 5.26);

        myPsychro = psychro(myPressure, myTmed);

        /*!
        \brief
        differs from the one presented in the FAO Irrigation and Drainage Paper N° 56.
        Analysis with several climatic data sets proved that more accurate estimates of ea can be
        obtained using es(Tmed) than with the equation reported in the FAO paper if only mean
        relative humidity is available (G. Van Halsema and G. Muñoz, Personal communication).
        */

        /*! Monteith and Unsworth (2008) */
        mySatVapPress = 0.61078 * exp(17.27 * myTmed / (myTmed + 237.3));
        myVapPress = mySatVapPress * myUmed / 100;
        delta = saturationSlope(myTmed, mySatVapPress);

        myDailySB = STEFAN_BOLTZMANN * DAY_SECONDS / 1000000;       /*!<   to MJ */
        myEmissivity = emissivityFromVaporPressure(myVapPress);
        myLWNetRad = myDailySB * (pow(myTmax + 273, 4) + pow(myTmin + 273, 4) / 2) * myEmissivity * (1.35 * (myTransmissivity / MAXTRANSMISSIVITY) - 0.35);

        mySWNetRad = mySWGlobRad * (1 - ALBEDO_CROP_REFERENCE);
        myNetRad = (mySWNetRad - myLWNetRad);

        myLambda = latentHeatVaporization(myTmed) / 1000000; /*!<  to MJ */

        vmed2 = myVmed10 * 0.748;

        EvapDemand = 900 / (myTmed + 273) * vmed2 * (mySatVapPress - myVapPress);

        return (delta * myNetRad + myPsychro * EvapDemand / myLambda) / (delta + myPsychro * (1 + 0.34 * vmed2));

}



/*!
 * \brief ET0_Penman_hourly http://www.cimis.water.ca.gov/cimis/infoEtoPmEquation.jsp
 * \param heigth elevation above mean sea level (meters)
 * \param normalizedTransmissivity normalized tramissivity [0-1] ()
 * \param globalIrradiance global surface downwelling irradiance (W m-2)
 * \param airTemp air temperature (C)
 * \param airHum relative humidity (%)
 * \param windSpeed10 wind speed at 10 meters (m s-1)
 * \return result
 */
double ET0_Penman_hourly(double heigth, double normalizedTransmissivity, double globalIrradiance,
                double airTemp, double airHum, double windSpeed10)
{
    double mySigma;                              /*!<  Steffan-Boltzman constant J m-2 h-1 K-4 */
    double es;                                   /*!<  saturation vapor pressure (kPa) at the mean hourly air temperature in C */
    double ea;                                   /*!<  actual vapor pressure (kPa) at the mean hourly air temperature in C */
    double emissivity;                           /*!<  net emissivity of the surface */
    double cloudFactor;                          /*!<  cloudiness factor for long wave radiation */
    double netRadiation;                         /*!<  net radiation (J m-2) */
    double netLWRadiation;                       /*!<  net longwave radiation (J m-2) */
    double netSWRadiation;                       /*!<  net shortwave radiation (J m-2) */
    double g;                                    /*!<  soil heat flux density (J m-2) */
    double Cd;                                   /*!<  bulk surface resistance and aerodynamic resistance coefficient */
    double tAirK;                                /*!<  air temperature (Kelvin) */
    double windSpeed2;                           /*!<  wind speed at 2 meters (m s-1) */
    double delta;                                /*!<  slope of saturation vapor pressure curve (kPa C-1) at mean air temperature */
    double pressure;                             /*!<  barometric pressure (kPa) */
    double lambda;                               /*!<  latent heat of vaporization in (J kg-1) */
    double gamma;                                /*!<  psychrometric constant (kPa C-1) */
    double firstTerm, secondTerm, denominator;


    es = saturationVaporPressure(airTemp) / 1000.;
    ea = airHum * es / 100.0;
    emissivity = emissivityFromVaporPressure(ea);
    tAirK = airTemp + ZEROCELSIUS;
    mySigma = STEFAN_BOLTZMANN * HOUR_SECONDS;
    cloudFactor = MAXVALUE(0, 1.35 * MINVALUE(normalizedTransmissivity, 1) - 0.35);
    netLWRadiation = cloudFactor * emissivity * mySigma * (pow(tAirK, 4));

    /*!   from [W m-2] to [J h-1 m-2] */
    netSWRadiation = (3600 * globalIrradiance);
    netRadiation = (1 - ALBEDO_CROP_REFERENCE) * netSWRadiation - netLWRadiation;

    /*!   values for grass */
    if (netRadiation > 0)
    {   g = 0.1 * netRadiation;
        Cd = 0.24;
    }
    else
    {
        g = 0.5 * netRadiation;
        Cd = 0.96;
    }

    delta = saturationSlope(airTemp, es);

    pressure = pressureFromAltitude(heigth) / 1000.;

    gamma = psychro(pressure, airTemp);
    lambda = latentHeatVaporization(airTemp);

    windSpeed2 = windSpeed10 * 0.748;

    denominator = delta + gamma * (1 + Cd * windSpeed2);
    firstTerm = delta * (netRadiation - g) / (lambda * denominator);
    secondTerm = (gamma * (37 / tAirK) * windSpeed2 * (es - ea)) / denominator;

    return MAXVALUE(firstTerm + secondTerm, 0);
}

/*!
 * \brief ET0_Penman_hourly_net_rad http://www.cimis.water.ca.gov/cimis/infoEtoPmEquation.jsp
 * \param heigth elevation above mean sea level (meters)
 * \param netIrradiance net surface irradiance (W m-2)
 * \param airTemp air temperature (C)
 * \param airHum relative humidity (%)
 * \param windSpeed10 wind speed at 10 meters (m s-1)
 * \return result
 */
double ET0_Penman_hourly_net_rad(double heigth, double netIrradiance, double airTemp, double airHum, double windSpeed10)
{
    double netRadiation;                         /*!<  net radiation (J m-2) */
    double es;                                   /*!<  saturation vapor pressure (kPa) at the mean hourly air temperature in C */
    double ea;                                   /*!<  actual vapor pressure (kPa) at the mean hourly air temperature in C */
    double g;                                    /*!<  soil heat flux density (J m-2 h-1) */
    double Cd;                                   /*!<  bulk surface resistance and aerodynamic resistance coefficient */
    double tAirK;                                /*!<  air temperature (Kelvin) */
    double windSpeed2;                           /*!<  wind speed at 2 meters (m s-1) */
    double delta;                                /*!<  slope of saturation vapor pressure curve (kPa C-1) at mean air temperature */
    double pressure;                             /*!<  barometric pressure (kPa) */
    double lambda;                               /*!<  latent heat of vaporization in (J kg-1) */
    double gamma;                                /*!<  psychrometric constant (kPa C-1) */
    double firstTerm, secondTerm, denominator;

    netRadiation = 3600 * netIrradiance;

    es = saturationVaporPressure(airTemp) / 1000.;
    ea = airHum * es / 100.0;

    tAirK = airTemp + ZEROCELSIUS;

    /*!   values for grass */
    if (netRadiation > 0)
    {   g = 0.1 * netRadiation;
        Cd = 0.24;
    }
    else
    {
        g = 0.5 * netRadiation;
        Cd = 0.96;
    }

    delta = saturationSlope(airTemp, es);

    pressure = pressureFromAltitude(heigth) / 1000.;

    gamma = psychro(pressure, airTemp);
    lambda = latentHeatVaporization(airTemp);

    windSpeed2 = windSpeed10 * 0.748;

    denominator = delta + gamma * (1 + Cd * windSpeed2);
    firstTerm = delta * (netRadiation - g) / (lambda * denominator);
    secondTerm = (gamma * (37 / tAirK) * windSpeed2 * (es - ea)) / denominator;

    return MAXVALUE(firstTerm + secondTerm, 0);
}


/*!
 * \brief computes [mm d-1] potential evapotranspiration. Trange minimum: 0.25°C  equivalent to 8.5% transimissivity
 * \param KT [-] Samani empirical coefficient
 * \param myLat [degrees] Latitude
 * \param myDoy [-] Day number (Jan 1st = 1)
 * \param tmax [°C] daily maximum air temperature
 * \param tmin [°C] daily minimum air temperature
 * \return result
 */
double ET0_Hargreaves(double KT, double myLat, int myDoy, double tmax, double tmin)
{
    double tavg, deltaT, extraTerrRadiation;

    if (tmax == NODATA || tmin == NODATA || KT == NODATA || myLat == NODATA || myDoy == NODATA)
        return NODATA;

    extraTerrRadiation = dailyExtrRadiation(myLat, myDoy);
    deltaT = MAXVALUE(fabs(tmax - tmin), 0.25);

    tavg = (tmax + tmin) * 0.5;

    return 0.0135 * (tavg + 17.78) * KT * (extraTerrRadiation / 2.456) * sqrt(deltaT);
    
    // 2.456 MJ kg-1 latent heat of vaporization
}


// Thom Discomfort Index (physiological thermal stress indicator for people based on dry-bulb and wet-bulb temperature)
float computeThomIndex(float temp, float relHum)
{
    if (int(temp) != int(NODATA) && int(relHum) != int(NODATA))
    {
        float zT = temp;
        float zUR = relHum;
        float es = float(0.611 * exp(17.27 * zT / (zT + ZEROCELSIUS - 36)));
        float zTwb = zT;
        float zTwbPrec = -999.f;

        while (fabs(zTwb - zTwbPrec) > 0.1f)
        {
            zTwbPrec = zTwb;
            float zT1 = (zT + zTwb) / 2;
            float es1 = float(0.611 * exp(17.27 * zT1 / (zT1 + ZEROCELSIUS - 36)));
            float delta = float(es1 / (zT1 + ZEROCELSIUS) * log(207700000 / es1));
            zTwb = zT - es * (1.f - zUR / 100.f) / (delta + 0.06667f);
        }

        return 0.4f * (zT + zTwb) + 4.8f;
    }
    else
        return NODATA;
}

bool computeWindCartesian(float intensity, float direction, float* u, float* v)
{
    if (isEqual(intensity, NODATA) || isEqual(direction, NODATA))
        return false;

    float angle;
    angle = 90 - direction;
    if (angle < 0) angle = angle + 360;

    *u = -intensity * getCosDecimalDegree(angle);
    *v = -intensity * getSinDecimalDegree(angle);

    return true;
}

bool computeWindPolar(float u, float v, float* intensity, float* direction)
{
    *intensity = NODATA;
    *direction = NODATA;

    if (isEqual(u, NODATA) || isEqual(v, NODATA)) return false;

    *intensity = sqrtf(u * u + v * v);

    if (isEqual(u, 0))
    {
        if (v < 0)
            *direction = 360;
        else
            *direction = 180;

        return true;
    }

    *direction = 90 - float(atan(double(v) / double(u)) * RAD_TO_DEG);

    if (*direction < 0) *direction += 360;

    if (u > 0) *direction += 180;

    return true;
}

bool setColorScale(meteoVariable variable, Crit3DColorScale *colorScale)
{
    if (colorScale == nullptr) return false;

    switch(variable)
    {
        case airTemperature: case dailyAirTemperatureAvg: case dailyAirTemperatureMax:
        case dailyAirTemperatureMin: case dailyAirTemperatureRange:
        case airDewTemperature:
        case snowSurfaceTemperature:
        case dailyHeatingDegreeDays:
            setTemperatureScale(colorScale);
            break;
        case elaboration:
            setDefaultScale(colorScale);
            break;
        case airRelHumidity: case dailyAirRelHumidityAvg: case dailyAirRelHumidityMax:
        case dailyAirRelHumidityMin: case leafWetness: case dailyLeafWetness:
        case thom: case dailyThomMax: case dailyThomAvg: case dailyThomHoursAbove: case dailyThomDaytime: case dailyThomNighttime:
            setRelativeHumidityScale(colorScale);
            break;
        case precipitation: case dailyPrecipitation: case referenceEvapotranspiration:
        case dailyReferenceEvapotranspirationHS: case dailyReferenceEvapotranspirationPM: case actualEvaporation:
        case snowFall: case snowWaterEquivalent: case snowLiquidWaterContent: case snowMelt:
        case dailyWaterTableDepth:
            setPrecipitationScale(colorScale);
            if (variable == snowFall || variable == snowWaterEquivalent
                || variable == snowLiquidWaterContent || variable == snowMelt)
            {
                colorScale->setHideOutliers(true);
                colorScale->setTransparent(true);
            }
            break;  
        case snowAge:
            setGrayScale(colorScale);
            reverseColorScale(colorScale);
            break;
        case dailyBIC:
            setCenteredScale(colorScale);
            break;
        case globalIrradiance: case directIrradiance: case diffuseIrradiance: case reflectedIrradiance:
        case netIrradiance: case dailyGlobalRadiation: case atmTransmissivity:
        case snowInternalEnergy: case snowSurfaceEnergy:
        case sensibleHeat: case latentHeat:
            setRadiationScale(colorScale);
            break;
        case windVectorIntensity: case windScalarIntensity: case windVectorX: case windVectorY: case dailyWindVectorIntensityAvg: case dailyWindVectorIntensityMax: case dailyWindScalarIntensityAvg: case dailyWindScalarIntensityMax:
        case atmPressure:
            setWindIntensityScale(colorScale);
            break;
        case leafAreaIndex:
            setLAIScale(colorScale);
            break;
        case anomaly:
            setAnomalyScale(colorScale);
            break;
        case noMeteoTerrain:
            setDTMScale(colorScale);
            break;

        default:
            setDefaultScale(colorScale);
    }

    return true;
}


std::string getVariableString(meteoVariable myVar)
{
    if (myVar == airTemperature || myVar == dailyAirTemperatureAvg || myVar == monthlyAirTemperatureAvg)
        return "Air temperature (°C)";
    else if (myVar == dailyAirTemperatureMax || myVar == monthlyAirTemperatureMax)
        return "Maximum air temperature (°C)";
    else if (myVar == dailyAirTemperatureMin || myVar == monthlyAirTemperatureMin)
        return "Minimum air temperature (°C)";
    else if (myVar == dailyAirTemperatureRange)
        return "Air temperature range (°C)";
    else if (myVar == airRelHumidity || myVar == dailyAirRelHumidityAvg)
        return "Air relative humidity (%)";
    else if (myVar == dailyAirRelHumidityMax)
        return "Maximum relative humidity (%)";
    else if (myVar == dailyAirRelHumidityMin)
        return "Minimum relative humidity (%)";
    else if (myVar == airDewTemperature)
        return "Air dew temperature (°C)";
    else if (myVar == thom || myVar == dailyThomAvg)
        return "Thom index ()";
    else if (myVar == dailyThomDaytime)
        return "Day Thom index ()";
    else if (myVar == dailyThomNighttime)
        return "Night Thom index ()";
    else if (myVar == dailyThomHoursAbove)
        return "Hours with Thom index above (h)";
    else if ((myVar == dailyPrecipitation ||  myVar == precipitation || myVar == monthlyPrecipitation))
        return "Precipitation (mm)";
    else if (myVar == dailyGlobalRadiation || myVar == monthlyGlobalRadiation)
        return "Solar radiation (MJ m-2)";
    else if (myVar == globalIrradiance)
        return "Global solar irradiance (W m-2)";
    else if (myVar == directIrradiance)
        return "Direct solar irradiance (W m-2)";
    else if (myVar == diffuseIrradiance)
        return "Diffuse solar irradiance (W m-2)";
    else if (myVar == reflectedIrradiance)
        return "Reflected solar irradiance (W m-2)";
    else if (myVar == netIrradiance)
        return "Solar net irradiance (W m-2)";
    else if (myVar == atmTransmissivity)
        return "Atmospheric transmissivity [-]";
    else if (myVar == windVectorIntensity)
        return "Wind vector intensity (m s-1)";
    else if (myVar == windVectorDirection)
        return "Wind vector direction (deg)";
    else if (myVar == windVectorX)
        return "Wind vector component X (m s-1)";
    else if (myVar == windVectorY)
        return "Wind vector component Y (m s-1)";
    else if (myVar == windScalarIntensity)
        return "Wind scalar intensity (m s-1)";
    else if (myVar == dailyWindVectorIntensityAvg)
        return "Average wind vector intensity (m s-1)";
    else if (myVar == dailyWindVectorIntensityMax)
        return "Maximum wind vector intensity (m s-1)";
    else if (myVar == dailyWindVectorDirectionPrevailing)
        return "Prevailing wind direction (deg)";
    else if (myVar == dailyWindScalarIntensityAvg)
        return "Average wind scalar intensity (m s-1)";
    else if (myVar == dailyWindScalarIntensityMax)
        return "Maximum wind scalar intensity (m s-1)";
    else if (myVar == referenceEvapotranspiration ||
             myVar == dailyReferenceEvapotranspirationHS ||
             myVar == monthlyReferenceEvapotranspirationHS ||
             myVar == dailyReferenceEvapotranspirationPM ||
             myVar == actualEvaporation)
        return "Reference evapotranspiration (mm)";
    else if (myVar == leafWetness || myVar == dailyLeafWetness)
        return "Leaf wetness (h)";
    else if (myVar == dailyBIC || myVar == monthlyBIC)
        return "Hydroclimatic balance (mm)";
    else if (myVar == dailyWaterTableDepth)
        return "Water table depth (mm)";
    else if (myVar == snowWaterEquivalent)
        return "Snow water equivalent (mm)";
    else if (myVar == snowFall)
        return "Snow fall (mm)";
    else if (myVar == snowMelt)
        return "Snowmelt (mm)";
    else if (myVar == snowLiquidWaterContent)
        return "Snow liquid water content (mm)";
    else if (myVar == snowAge)
        return "Snow age (days)";
    else if (myVar == snowSurfaceTemperature)
        return "Surface temperature (°C)";
    else if (myVar == snowInternalEnergy)
        return "Energy content (kJ m-2)";
    else if (myVar == snowSurfaceEnergy)
        return "Energy content surface layer (kJ m-2)";
    else if (myVar == sensibleHeat)
        return "Sensible heat (kJ m-2)";
    else if (myVar == latentHeat)
        return "Latent heat (kJ m-2)";
    else if (myVar == dailyHeatingDegreeDays)
        return "Heating degree days (°D)";
    else if (myVar == leafAreaIndex)
            return "Leaf area index (m2 m-2)";

    else if (myVar == elaboration)
        return "Elaboration";
    else if (myVar == anomaly)
        return "Anomaly";
    else if (myVar == noMeteoTerrain)
        return "Elevation (m)";
    else
        return "No variable";
}

std::string getKeyStringMeteoMap(std::map<std::string, meteoVariable> map, meteoVariable value)
{

    std::map<std::string, meteoVariable>::const_iterator it;
    std::string key = "";

    for (it = map.begin(); it != map.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}


std::string getUnitFromVariable(meteoVariable var)
{
    std::string unit = "";
    std::map<std::vector<meteoVariable>, std::string>::const_iterator it;
    std::vector<meteoVariable> key;

    for (it = MapVarUnit.begin(); it != MapVarUnit.end(); ++it)
    {
        key = it->first;
        if(std::find(key.begin(), key.end(), var) != key.end())
        {
            unit = it->second;
            break;
        }
        key.clear();
    }

    return unit;
}


meteoVariable getKeyMeteoVarMeteoMap(std::map<meteoVariable,std::string> map, const std::string& value)
{
    std::map<meteoVariable, std::string>::const_iterator it;
    meteoVariable key = noMeteoVar;

    for (it = map.begin(); it != map.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

meteoVariable getKeyMeteoVarMeteoMapWithoutUnderscore(std::map<meteoVariable,std::string> map, const std::string& value)
{
    std::map<meteoVariable, std::string>::const_iterator it;
    meteoVariable key = noMeteoVar;

    for (it = map.begin(); it != map.end(); ++it)
    {
        std::string str = it->second;
        str.erase(std::remove(str.begin(), str.end(), '_'), str.end());
        if (str == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

frequencyType getVarFrequency(meteoVariable myVar)
{
    if (MapDailyMeteoVarToString.find(myVar) != MapDailyMeteoVarToString.end())
        return daily;
    else if (MapHourlyMeteoVarToString.find(myVar) != MapHourlyMeteoVarToString.end())
        return hourly;
    else if (MapMonthlyMeteoVarToString.find(myVar) != MapMonthlyMeteoVarToString.end())
        return monthly;
    else
        return noFrequency;
}


meteoVariable getMeteoVar(std::string varString)
{
    auto search = MapDailyMeteoVar.find(varString);

    if (search != MapDailyMeteoVar.end())
        return search->second;
    else
    {
        search = MapHourlyMeteoVar.find(varString);
        if (search != MapHourlyMeteoVar.end())
        {
            return search->second;
        }
        else
        {
            search = MapMonthlyMeteoVar.find(varString);
            if (search != MapMonthlyMeteoVar.end())
            {
                return search->second;
            }
        }
    }

    return noMeteoVar;
}


std::string getMeteoVarName(meteoVariable var)
{
    auto search = MapDailyMeteoVarToString.find(var);

    if (search != MapDailyMeteoVarToString.end())
        return search->second;
    else
    {
        search = MapHourlyMeteoVarToString.find(var);
        if (search != MapHourlyMeteoVarToString.end())
            return search->second;
        else
        {
            search = MapMonthlyMeteoVarToString.find(var);
            if (search != MapMonthlyMeteoVarToString.end())
                return search->second;
        }
    }

    return "";
}


std::string getCriteria3DVarName(criteria3DVariable var)
{
    auto search = MapCriteria3DVarToString.find(var);

    if (search != MapCriteria3DVarToString.end())
    {
        return search->second;
    }
    else
    {
        return "";
    }
}


std::string getLapseRateCodeName(lapseRateCodeType code)
{
    auto search = MapLapseRateCodeToString.find(code);
    if (search != MapLapseRateCodeToString.end())
        return search->second;

    return "";
}

meteoVariable getHourlyMeteoVar(std::string varString)
{
    auto search = MapHourlyMeteoVar.find(varString);

    if (search != MapHourlyMeteoVar.end())
        return search->second;
    else
        return noMeteoVar;
}


bool checkLapseRateCode(lapseRateCodeType myType, bool useLapseRateCode, bool useSupplemental)
{
    if (useSupplemental)
        return (! useLapseRateCode || myType == primary || myType == supplemental);
    else
        return (! useLapseRateCode || myType == primary || myType == secondary);
}


meteoVariable getDailyMeteoVarFromHourly(meteoVariable myVar, aggregationMethod myAggregation)
{
    if (myVar == airTemperature)
    {
        if (myAggregation == aggrMin)
            return dailyAirTemperatureMin;
        else if (myAggregation == aggrMax)
            return dailyAirTemperatureMax;
        else if (myAggregation == aggrAverage)
            return dailyAirTemperatureAvg;
    }
    else if (myVar == airRelHumidity)
    {
        if (myAggregation == aggrMin)
            return dailyAirRelHumidityMin;
        else if (myAggregation == aggrMax)
            return dailyAirRelHumidityMax;
        else if (myAggregation == aggrAverage)
            return dailyAirRelHumidityAvg;
    }
    else if (myVar == windVectorIntensity)
    {
        if (myAggregation == aggrAverage)
            return dailyWindVectorIntensityAvg;
        else if (myAggregation == aggrMax)
            return dailyWindVectorIntensityMax;
    }
    else if (myVar == windVectorDirection)
    {
        if (myAggregation == aggrPrevailing)
            return dailyWindVectorDirectionPrevailing;
    }
    else if (myVar == windScalarIntensity)
    {
        if (myAggregation == aggrAverage)
            return dailyWindScalarIntensityAvg;
        else if (myAggregation == aggrMax)
            return dailyWindScalarIntensityMax;
    }
    else if (myVar == precipitation)
    {
        if (myAggregation == aggrSum)
            return dailyPrecipitation;
    }
    else if (myVar == referenceEvapotranspiration)
    {
        if (myAggregation == aggrSum)
            return dailyReferenceEvapotranspirationPM;
    }
    else if (myVar == globalIrradiance)
    {
        if (myAggregation == aggrIntegral)
            return dailyGlobalRadiation;
    }
    else if (myVar == directIrradiance)
    {
        if (myAggregation == aggrIntegral)
            return dailyDirectRadiation;
    }
    else if (myVar == diffuseIrradiance)
    {
        if (myAggregation == aggrIntegral)
            return dailyDiffuseRadiation;
    }
    else if (myVar == reflectedIrradiance)
    {
        if (myAggregation == aggrIntegral)
            return dailyReflectedRadiation;
    }
    else if (myVar == leafWetness)
    {
        if (myAggregation == aggrSum)
            return dailyLeafWetness;
    }

    return noMeteoVar;
}

meteoVariable updateMeteoVariable(meteoVariable myVar, frequencyType myFreq)
{
    if (myFreq == daily)
    {
        //check
        if (myVar == airTemperature || myVar == monthlyAirTemperatureAvg)
            return dailyAirTemperatureAvg;

        else if (myVar == monthlyAirTemperatureMin)
            return dailyAirTemperatureMin;

        else if (myVar == monthlyAirTemperatureMax)
            return dailyAirTemperatureMax;

        else if (myVar == precipitation || myVar == monthlyPrecipitation)
            return dailyPrecipitation;

        else if (myVar == globalIrradiance || myVar == monthlyGlobalRadiation)
            return dailyGlobalRadiation;

        else if (myVar == airRelHumidity)
            return dailyAirRelHumidityAvg;

        else if (myVar == thom)
            return dailyThomAvg;

        else if (myVar == windScalarIntensity)
            return dailyWindScalarIntensityAvg;

        else if (myVar== windVectorIntensity || myVar == windVectorX || myVar == windVectorY)
            return dailyWindVectorIntensityAvg;

        else if (myVar == windVectorDirection)
            return dailyWindVectorDirectionPrevailing;

        else if (myVar == leafWetness)
            return dailyLeafWetness;

        else if (myVar == referenceEvapotranspiration || myVar == monthlyReferenceEvapotranspirationHS)
            return dailyReferenceEvapotranspirationHS;

        else if (myVar == monthlyBIC)
            return dailyBIC;

        else
            return noMeteoVar;
    }

    if (myFreq == hourly)
    {
        //check
        if (myVar == dailyAirTemperatureAvg || myVar == dailyAirTemperatureMax || myVar == dailyAirTemperatureMin || myVar == dailyAirTemperatureRange
                || myVar == monthlyAirTemperatureAvg || myVar == monthlyAirTemperatureMax || myVar == monthlyAirTemperatureMin)
            return airTemperature;

        else if (myVar == dailyAirRelHumidityAvg || myVar == dailyAirRelHumidityMax || myVar == dailyAirRelHumidityMin)
            return airRelHumidity;

        else if (myVar == dailyPrecipitation || myVar == monthlyPrecipitation)
            return precipitation;

        else if (myVar == dailyGlobalRadiation || myVar == monthlyGlobalRadiation)
            return globalIrradiance;

        else if (myVar == dailyDirectRadiation)
            return directIrradiance;

        else if (myVar == dailyDiffuseRadiation)
            return diffuseIrradiance;

        else if (myVar == dailyReflectedRadiation)
            return reflectedIrradiance;

        else if (myVar == dailyThomAvg || myVar == dailyThomMax || myVar == dailyThomHoursAbove || myVar == dailyThomDaytime || myVar == dailyThomNighttime)
            return thom;

        else if (myVar == dailyWindScalarIntensityAvg || myVar == dailyWindScalarIntensityMax)
            return windScalarIntensity;

        else if (myVar == dailyWindVectorIntensityAvg || myVar == dailyWindVectorIntensityMax)
            return windVectorIntensity;

        else if (myVar == dailyWindVectorDirectionPrevailing)
            return windVectorDirection;

        else if (myVar == dailyLeafWetness)
            return leafWetness;

        else if (myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM || myVar == monthlyReferenceEvapotranspirationHS)
            return referenceEvapotranspiration;
        else
            return noMeteoVar;
    }

    if (myFreq == monthly)
    {
        if (myVar == dailyAirTemperatureMin)
            return monthlyAirTemperatureMin;
        else if (myVar == dailyAirTemperatureMax)
            return monthlyAirTemperatureMax;
        else if (myVar == dailyAirTemperatureAvg || myVar == airTemperature)
            return monthlyAirTemperatureAvg;
        else if (myVar == dailyPrecipitation || myVar == precipitation)
            return monthlyPrecipitation;
        else if (myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM || myVar == referenceEvapotranspiration)
            return monthlyReferenceEvapotranspirationHS;
        else if (myVar == globalIrradiance || myVar == dailyGlobalRadiation)
            return monthlyGlobalRadiation;
        else if (myVar == dailyBIC)
            return monthlyBIC;
        else
            return noMeteoVar;
    }

    return noMeteoVar;
}
