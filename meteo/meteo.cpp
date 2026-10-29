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
#include <unordered_map>

#include "commonConstants.h"
#include "basicMath.h"
#include "crit3dDate.h"
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


float Crit3DClimateParameters::getClimateLapseRate(meteoVariable myVar, const Crit3DTime &myTime) const
{
    Crit3DDate myDate = myTime.date;
    int myHour = myTime.getNearestHour();

    if (myDate.isNullDate() || myHour == NODATA)
        return NODATA;

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
            return DEFAULT_LAPSERATE;

        float tminWeight = computeTminHourlyWeight(myHour);
        return (lapseTmin * tminWeight + lapseTmax * (1 - tminWeight));
    }
}

float Crit3DClimateParameters::getClimateLapseRate(meteoVariable myVar, int month) const
{
    if (month > 12 || month < 1 || month == NODATA)
        return NODATA;

    const unsigned idx = unsigned(month - 1);

    if (myVar == dailyAirTemperatureMin)
        return tminLapseRate[idx];
    else if (myVar == dailyAirTemperatureMax)
        return tmaxLapseRate[idx];
    else if (myVar == dailyAirTemperatureAvg)
        return (tmaxLapseRate[idx] + tminLapseRate[idx]) / 2;
    else
        return NODATA;
}


float Crit3DClimateParameters::getClimateVar(meteoVariable myVar, int month, float height, float refHeight) const
{
    if (month > 12 || month < 1)
        return NODATA;

    const unsigned idx = unsigned(month - 1);
    float climateVar, lapseRate;

    switch (myVar)
    {
    case dailyAirTemperatureMin:    climateVar = tmin[idx];  lapseRate = tminLapseRate[idx];   break;
    case dailyAirTemperatureMax:    climateVar = tmax[idx];  lapseRate = tmaxLapseRate[idx];   break;
    case dailyAirRelHumidityMin:    climateVar = tdmin[idx]; lapseRate = tdMinLapseRate[idx];  break;
    case dailyAirRelHumidityMax:    climateVar = tdmax[idx]; lapseRate = tdMaxLapseRate[idx];  break;
    default:                        return NODATA;
    }

    if (isEqual(climateVar, NODATA) || isEqual(height, NODATA))
        return climateVar;

    if (isEqual(lapseRate, NODATA))
        return NODATA;              // cannot correct for height

    return climateVar + lapseRate * (height - refHeight);
}


double tDewFromRelHum(double RH, double T)
{
    if (isEqual(RH, NODATA) || isEqual(T, NODATA) || RH <= 0)
        return NODATA;

    RH = std::min(RH, 100.);
    const double g = std::log(RH / 100.) + MAGNUS_B * T / (MAGNUS_C + T);
    return MAGNUS_C * g / (MAGNUS_B - g);
}

float tDewFromRelHum(float RH, float T)
{
    return float(tDewFromRelHum(double(RH), double(T)));
}

double relHumFromTdew(double Td, double T)
{
    if (isEqual(Td, NODATA) || isEqual(T, NODATA))
        return NODATA;

    const double rh = 100. * std::exp(MAGNUS_B * Td / (MAGNUS_C + Td) - MAGNUS_B * T / (MAGNUS_C + T));
    return std::min(100., std::max(1., rh));
}

float relHumFromTdew(float Td, float T)
{
    return float(relHumFromTdew(double(Td), double(T)));
}

float dewPoint(float relHumAir, float tempAir)
{
    return tDewFromRelHum(relHumAir, tempAir);
}


/*!
    2011 GA
    da quaderno FAO
    MJ m-2 d-1
    */
double dailyExtrRadiation(double myLat, int myDoy)
{
    const double doy = double(myDoy);

    const double phi = PI / 180. * myLat;                               /*!< [rad] latitude in radiants */
    const double delta = 0.4093 * sin((2. * PI / 365.) * doy - 1.39);   /*!< [rad] solar declination */

    const double dr = 1. + 0.033 * cos(2. * PI * doy / 365.);           /*!< [-] inverse Earth-Sun relative distance */

    // avoid NaN at the poles
    double tanTerm = -tan(phi) * tan(delta);
    tanTerm = std::max(-1.0, std::min(1.0, tanTerm));

    const double omegaS = acos(tanTerm);                                /*!< [rad] sunset hour angle */


    return SOLAR_CONSTANT * DAY_SECONDS / 1000000. * dr / PI * (omegaS * sin(phi) * sin(delta)
                                                                + cos(phi) * cos(delta) * sin(omegaS));
}


float computeDailyBIC(float prec, float etp)
{
    Crit3DQuality qualityCheck;

    quality::qualityType qualityPrec = qualityCheck.syntacticQualitySingleValue(dailyPrecipitation, prec);
    quality::qualityType qualityETP = qualityCheck.syntacticQualitySingleValue(dailyReferenceEvapotranspirationHS, etp);

    if (qualityPrec == quality::accepted && qualityETP == quality::accepted)
        return (prec - etp);
    else
        return NODATA;
}


float dailyThermalRange(float tMin, float tMax)
{
    Crit3DQuality qualityCheck;

    quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, tMin);
    quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, tMax);

    if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
        return (tMax - tMin);
    else
        return NODATA;
}


float dailyAverageT(float tMin, float tMax)
{
        Crit3DQuality qualityCheck;

        quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, tMin);
        quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, tMax);

        if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
            return (tMin + tMax) * 0.5f;
        else
            return NODATA;
}


float dailyEtpHargreaves(float Tmin, float Tmax, const Crit3DDate &date, double latitude, Crit3DMeteoSettings* meteoSettings)
{
    if (meteoSettings == nullptr)
        return NODATA;

    Crit3DQuality qualityCheck;

    quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, Tmin);
    quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, Tmax);

    if (qualityTmin != quality::accepted || qualityTmax != quality::accepted)
        return NODATA;

    const int dayOfYear = getDoyFromDate(date);
    const double et0 = ET0_Hargreaves(meteoSettings->getTransSamaniCoefficient(), latitude, dayOfYear, Tmax, Tmin);

    return float(et0);
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
bool computeLeafWetness(double prec, double relHumidity, short& leafW)
{
    leafW = 0;

    if (isEqual(relHumidity, NODATA) || isEqual(prec, NODATA))
        return false;

    if (prec > 0.0 || relHumidity > DEFAULT_LEAFWETNESS_RH_THRESHOLD)
        leafW = 1;

    return true;
}

/*!
 * \brief 2016 GA. comments: G is currently ignored (if heat flux is active, should be added)
 * \param myDOY [-] day of year
 * \param myElevation [m]
 * \param myLatitude [°] latitude in decimal degrees
 * \param myTmin [°C] daily minimum temperature
 * \param myTmax [°C] daily maximum temperature
 * \param myUmed [%] daily average relative humidity
 * \param myVmed10 [m s-1] daily average wind intensity
 * \param mySWGlobRad [MJ m-2 d-1] daily global short wave radiation
 * \return ET0 daily [mm]
 */
double ET0_Penman_daily(int myDOY, double myElevation, double myLatitude, double myTmin, double myTmax,
                        double myUmed, double myVmed10, double mySWGlobRad)
{
    double MAXTRANSMISSIVITY = 0.75;

    double myPressure;                   /*!<  [kPa] atmospheric pressure */
    double myDailySB;                    /*!<  [MJ m-2 d-1 K-4] Stefan Boltzmann constant */
    double myPsychro;                    /*!<  [kPa °C-1] psychrometric instrument constant */
    double myTmed;                       /*!<  [°C] daily average temperature */
    double myTransmissivity;             /*!<  [-] global atmospheric trasmissivity */
    double myVapPress;                   /*!<  [kPa] actual vapor pressure */
    double mySatVapPress;                /*!<  [kPa] actual average vapor pressure */
    double myExtraRad;                   /*!<  [MJ m-2 d-1] extraterrestrial radiation */
    double mySWNetRad;                   /*!<  [MJ m-2 d-1] net short wave radiation */
    double myLWNetRad;                   /*!<  [MJ m-2 d-1] net long wave emitted radiation */
    double myNetRad;                     /*!<  [MJ m-2 d-1] net surface radiation */
    double delta;                        /*!<  [kPa °C-1] slope of vapour pressure curve */
    double windSpeed2;                   /*!<  [m s-1] average wind speed estimated at 2 meters */
    double evapDemand;                   /*!<  [mm d-1] evaporative demand of atmosphere */
    double emissivity;                   /*!<  [-] surface emissivity */
    double lambda;                       /*!<  [MJ kg-1] latent heat of vaporization */


    if (myTmin == NODATA || myTmax == NODATA || myVmed10 == NODATA
        || myUmed == NODATA || myUmed > 100 || myUmed < 0)
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

    emissivity = emissivityFromVaporPressure(myVapPress);

    myLWNetRad = myDailySB * (pow(myTmax + ZEROCELSIUS, 4) + pow(myTmin + ZEROCELSIUS, 4)) / 2 * emissivity
                 * (1.35 * (myTransmissivity / MAXTRANSMISSIVITY) - 0.35);

    mySWNetRad = mySWGlobRad * (1 - ALBEDO_CROP_REFERENCE);
    myNetRad = (mySWNetRad - myLWNetRad);

    lambda = latentHeatVaporization(myTmed) / 1000000;        /*!<  to MJ */

    constexpr double WIND_SPEED_10M_TO_2M = 0.748;
    windSpeed2 = myVmed10 * WIND_SPEED_10M_TO_2M;

    evapDemand = 900 / (myTmed + ZEROCELSIUS) * windSpeed2 * (mySatVapPress - myVapPress);

    const double denominator = delta + myPsychro * (1 + 0.34 * windSpeed2);
    const double firstTerm = (delta * myNetRad) / (lambda * denominator);
    const double secondTerm = (myPsychro * evapDemand) / denominator;

    return std::max(firstTerm + secondTerm, 0.0);
}


/*!
 * \brief ET0_Penman_hourly http://www.cimis.water.ca.gov/cimis/infoEtoPmEquation.jsp
 * \param height elevation above mean sea level (meters)
 * \param normalizedTransmissivity normalized tramissivity [0-1] ()
 * \param globalIrradiance global surface downwelling irradiance (W m-2)
 * \param airTemp air temperature (C)
 * \param airHum relative humidity (%)
 * \param windSpeed10 wind speed at 10 meters (m s-1)
 * \return ET0 hourly (mm)
 */
double ET0_Penman_hourly(double height, double normalizedTransmissivity, double globalIrradiance,
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

    es = saturationVaporPressure(airTemp) / 1000.;
    ea = airHum * es / 100.0;
    emissivity = emissivityFromVaporPressure(ea);
    tAirK = airTemp + ZEROCELSIUS;
    mySigma = STEFAN_BOLTZMANN * HOUR_SECONDS;

    cloudFactor = std::max(0.0, 1.35 * std::min(normalizedTransmissivity, 1.0) - 0.35);

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

    pressure = pressureFromAltitude(height) / 1000.;

    gamma = psychro(pressure, airTemp);
    lambda = latentHeatVaporization(airTemp);

    constexpr double WIND_SPEED_10M_TO_2M = 0.748;
    windSpeed2 = windSpeed10 * WIND_SPEED_10M_TO_2M;

    const double denominator = delta + gamma * (1 + Cd * windSpeed2);
    const double firstTerm = delta * (netRadiation - g) / (lambda * denominator);
    const double secondTerm = (gamma * (37 / tAirK) * windSpeed2 * (es - ea)) / denominator;

    return std::max(firstTerm + secondTerm, 0.0);
}


/*!
 * \brief ET0_Penman_hourly_net_rad http://www.cimis.water.ca.gov/cimis/infoEtoPmEquation.jsp
 * \param heigth elevation above mean sea level (meters)
 * \param netIrradiance net surface irradiance (W m-2)
 * \param airTemp air temperature (C)
 * \param airHum relative humidity (%)
 * \param windSpeed10 wind speed at 10 meters (m s-1)
 * \return hourly ET0 (mm)
 */
double ET0_Penman_hourly_net_rad(double height, double netIrradiance, double airTemp, double airHum, double windSpeed10)
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

    if (height == NODATA || netIrradiance == NODATA || airTemp == NODATA || airHum == NODATA || windSpeed10 == NODATA)
        return NODATA;

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

    pressure = pressureFromAltitude(height) / 1000.;

    gamma = psychro(pressure, airTemp);
    lambda = latentHeatVaporization(airTemp);

    constexpr double WIND_SPEED_10M_TO_2M = 0.748;
    windSpeed2 = windSpeed10 * WIND_SPEED_10M_TO_2M;

    const double denominator = delta + gamma * (1 + Cd * windSpeed2);
    const double firstTerm = delta * (netRadiation - g) / (lambda * denominator);
    const double secondTerm = (gamma * (37 / tAirK) * windSpeed2 * (es - ea)) / denominator;

    return std::max(firstTerm + secondTerm, 0.0);
}


/*!
 * \brief computes [mm d-1] potential evapotranspiration.
 * Trange minimum: 0.25°C  equivalent to 8.5% transimissivity
 * \param KT [-] Samani empirical coefficient
 * \param myLat [degrees] Latitude
 * \param myDoy [-] Day number (Jan 1st = 1)
 * \param tmax [°C] daily maximum air temperature
 * \param tmin [°C] daily minimum air temperature
 * \return daily ETO [mm]
 */
double ET0_Hargreaves(double KT, double myLat, int myDoy, double tmax, double tmin)
{
    if (tmax == NODATA || tmin == NODATA || KT == NODATA || myLat == NODATA || myDoy == NODATA)
        return NODATA;

    const double extraTerrRadiation = dailyExtrRadiation(myLat, myDoy);

    const double deltaT = std::max(tmax - tmin, 0.25);
    const double tAvg = (tmax + tmin) * 0.5;

    const double latentHeatVap = latentHeatVaporization(tAvg) * 1E-6;      // [MJ kg-1]
    const double et0 = 0.0135 * (tAvg + 17.78) * KT * (extraTerrRadiation / latentHeatVap) * sqrt(deltaT);

    return std::max(et0, 0.0);
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
    *u = NODATA;
    *v = NODATA;

    if (isEqual(intensity, NODATA) || isEqual(direction, NODATA))
        return false;

    float angle = 90 - direction;
    if (angle < 0)
        angle += 360;

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
        case airTemperature: case dailyAirTemperatureAvg:
        case dailyAirTemperatureMax: case dailyAirTemperatureMin: case dailyAirTemperatureRange:
        case monthlyAirTemperatureAvg: case monthlyAirTemperatureMax: case monthlyAirTemperatureMin:
        case airDewTemperature:
        case snowSurfaceTemperature:
        case dailyHeatingDegreeDays:
        case atmPressure:
            setTemperatureScale(colorScale);
            break;
        case elaborationVar:
            setDefaultScale(colorScale);
            break;
        case airRelHumidity: case dailyAirRelHumidityAvg: case dailyAirRelHumidityMax:
        case dailyAirRelHumidityMin: case leafWetness: case dailyLeafWetness:
        case thom: case dailyThomMax: case dailyThomAvg: case dailyThomHoursAbove: case dailyThomDaytime: case dailyThomNighttime:
            setRelativeHumidityScale(colorScale);
            break;
        case snowWaterEquivalent: case snowLiquidWaterContent: case snowMelt: case snowFall:
        case precipitation: case dailyPrecipitation: case monthlyPrecipitation:
            setPrecipitationScale(colorScale);
            colorScale->setHideZero(true);
            break;
        case referenceEvapotranspiration: case dailyReferenceEvapotranspirationHS:
        case dailyReferenceEvapotranspirationPM: case actualEvaporation:
        case dailyWaterTableDepth:
            setPrecipitationScale(colorScale);
            break;
        case snowAge:
            setGrayScale(colorScale);
            reverseColorScale(colorScale);
            break;
        case dailyBIC: case monthlyBIC:
            setPrecipitationScale(colorScale);
            reverseColorScale(colorScale);
            break;
        case globalIrradiance: case directIrradiance: case diffuseIrradiance: case reflectedIrradiance:
        case netIrradiance: case dailyGlobalRadiation: case atmTransmissivity:
        case snowInternalEnergy: case snowSurfaceEnergy:
        case sensibleHeat: case latentHeat:
            setRadiationScale(colorScale);
            break;
        case windVectorIntensity: case windScalarIntensity: case windVectorX: case windVectorY:
        case dailyWindVectorIntensityAvg: case dailyWindVectorIntensityMax:
        case dailyWindScalarIntensityAvg: case dailyWindScalarIntensityMax:
            setWindIntensityScale(colorScale);
            break;
        case leafAreaIndex:
            setLAIScale(colorScale);
            break;
        case snowVariation:
            setAnomalyScale(colorScale);
            reverseColorScale(colorScale);
            colorScale->setHideZero(true);
            break;
        case anomalyVar:
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


static const std::unordered_map<meteoVariable, std::string> variableStringMap =
    {
        {airTemperature, "Air temperature (°C)"},
        {dailyAirTemperatureAvg, "Air temperature (°C)"},
        {monthlyAirTemperatureAvg, "Monthly air temperature (°C)"},

        {dailyAirTemperatureMax, "Maximum air temperature (°C)"},
        {monthlyAirTemperatureMax, "Monthly maximum air temperature (°C)"},

        {dailyAirTemperatureMin, "Minimum air temperature (°C)"},
        {monthlyAirTemperatureMin, "Monthly minimum air temperature (°C)"},

        {dailyAirTemperatureRange, "Air temperature range (°C)"},

        {airRelHumidity, "Air relative humidity (%)"},
        {dailyAirRelHumidityAvg, "Air relative humidity (%)"},

        {dailyAirRelHumidityMax, "Maximum relative humidity (%)"},
        {dailyAirRelHumidityMin, "Minimum relative humidity (%)"},

        {airDewTemperature, "Air dew temperature (°C)"},

        {thom, "Thom index ()"},
        {dailyThomAvg, "Thom index ()"},
        {dailyThomDaytime, "Day Thom index ()"},
        {dailyThomNighttime, "Night Thom index ()"},
        {dailyThomHoursAbove, "Hours with Thom index above (h)"},

        {dailyPrecipitation, "Precipitation (mm)"},
        {precipitation, "Precipitation (mm)"},
        {monthlyPrecipitation, "Monthly precipitation (mm)"},

        {dailyGlobalRadiation, "Solar radiation (MJ m-2)"},
        {monthlyGlobalRadiation, "Monthly solar radiation (MJ m-2)"},

        {globalIrradiance, "Global solar irradiance (W m-2)"},
        {directIrradiance, "Direct solar irradiance (W m-2)"},
        {diffuseIrradiance, "Diffuse solar irradiance (W m-2)"},
        {reflectedIrradiance, "Reflected solar irradiance (W m-2)"},
        {netIrradiance, "Solar net irradiance (W m-2)"},

        {atmTransmissivity, "Atmospheric transmissivity [-]"},
        {atmPressure, "Atmospheric pressure [hPa]"},

        {windVectorIntensity, "Wind vector intensity (m s-1)"},
        {windVectorDirection, "Wind vector direction (deg)"},
        {windVectorX, "Wind vector component X (m s-1)"},
        {windVectorY, "Wind vector component Y (m s-1)"},
        {windScalarIntensity, "Wind scalar intensity (m s-1)"},

        {dailyWindVectorIntensityAvg, "Average wind vector intensity (m s-1)"},
        {dailyWindVectorIntensityMax, "Maximum wind vector intensity (m s-1)"},
        {dailyWindVectorDirectionPrevailing, "Prevailing wind direction (deg)"},

        {dailyWindScalarIntensityAvg, "Average wind scalar intensity (m s-1)"},
        {dailyWindScalarIntensityMax, "Maximum wind scalar intensity (m s-1)"},

        {referenceEvapotranspiration, "Reference evapotranspiration (mm)"},
        {dailyReferenceEvapotranspirationHS, "Reference evapotranspiration HS (mm)"},
        {dailyReferenceEvapotranspirationPM, "Reference evapotranspiration PM (mm)"},
        {monthlyReferenceEvapotranspirationHS, "Monthly reference evapotranspiration HS (mm)"},

        {actualEvaporation, "Actual evaporation (mm)"},

        {leafWetness, "Leaf wetness (h)"},
        {dailyLeafWetness, "Leaf wetness (h)"},

        {dailyBIC, "Hydroclimatic balance (mm)"},
        {monthlyBIC, "Monthly hydroclimatic balance (mm)"},

        {dailyWaterTableDepth, "Water table depth (mm)"},

        {snowWaterEquivalent, "Snow water equivalent (mm)"},
        {snowFall, "Snow fall (mm)"},
        {snowMelt, "Snowmelt (mm)"},
        {snowVariation, "Variation in SWE (mm)"},
        {snowLiquidWaterContent, "Snow liquid water content (mm)"},
        {snowAge, "Snow age (days)"},

        {snowSurfaceTemperature, "Surface temperature (°C)"},
        {snowInternalEnergy, "Energy content (kJ m-2)"},
        {snowSurfaceEnergy, "Energy content surface layer (kJ m-2)"},

        {sensibleHeat, "Sensible heat (kJ m-2)"},
        {latentHeat, "Latent heat (kJ m-2)"},

        {dailyHeatingDegreeDays, "Heating degree days (°D)"},

        {leafAreaIndex, "Leaf area index (m2 m-2)"},

        {elaborationVar, "Elaboration"},
        {anomalyVar, "Anomaly"},

        {noMeteoTerrain, "Elevation (m)"}
};


std::string getVariableString(meteoVariable myVar)
{
    auto it = variableStringMap.find(myVar);

    if (it != variableStringMap.end())
        return it->second;

    return "No variable.";
}


std::string getKeyStringMeteoMap(const std::map<std::string, meteoVariable> &map, meteoVariable value)
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
    static const std::unordered_map<meteoVariable, std::string> unitMap = []
    {
        std::unordered_map<meteoVariable, std::string> m;
        for (const auto& item : MapVarUnit)
            for (meteoVariable v : item.first)
                m.emplace(v, item.second);
        return m;
    }();

    auto it = unitMap.find(var);
    return (it != unitMap.end()) ? it->second : "";
}

std::string getUnitFromVariable_old(meteoVariable var)
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


meteoVariable getKeyMeteoVarMeteoMap(const std::map<meteoVariable, std::string> &map, const std::string& value)
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


meteoVariable getKeyMeteoVarMeteoMapWithoutUnderscore(const std::map<meteoVariable, std::string> &map, const std::string& value)
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


meteoVariable getMeteoVar(const std::string& varString)
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
        return search->second;

    return "";
}


std::string getLapseRateCodeName(lapseRateCodeType code)
{
    auto search = MapLapseRateCodeToString.find(code);
    if (search != MapLapseRateCodeToString.end())
        return search->second;

    return "";
}

meteoVariable getHourlyMeteoVar(const std::string& varString)
{
    auto search = MapHourlyMeteoVar.find(varString);
    if (search != MapHourlyMeteoVar.end())
        return search->second;

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
