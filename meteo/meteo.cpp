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

#include "commonConstants.h"
#include "basicMath.h"
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
    transSamaniCoefficient = DEFAULT_TRANSMISSIVITY_SAMANI;
    windIntensityDefault = DEFAULT_WIND_INTENSITY;
    hourlyIntervals = DEFAULT_HOURLY_INTERVALS;
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

void Crit3DMeteoSettings::setThomThreshold(float value)
{
    thomThreshold = value;
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
    int myHour = myTime.getHour();

    // TODO improve!
    if (myDate == getNullDate() || myHour == NODATA)
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


float Crit3DClimateParameters::getClimateVar(meteoVariable myVar, Crit3DDate myDate, int myHour)
{
    unsigned int indexMonth = unsigned(myDate.month - 1);

    if (myVar == dailyAirTemperatureMin)
        return tmin[indexMonth];
    else if (myVar == dailyAirTemperatureMax)
        return tmax[indexMonth];
    else if (myVar == dailyAirTemperatureAvg)
        return (tmax[indexMonth] + tmin[indexMonth]) / 2;
    else
    {
        float myTmin, myTmax;
        if (myVar == airTemperature)
        {
            myTmin = tmin[indexMonth];
            myTmax = tmax[indexMonth];
        }
        else if (myVar == airDewTemperature)
        {
            myTmin = tdmin[indexMonth];
            myTmax = tdmax[indexMonth];
        }
        else
            return NODATA;

        float tminWeight = computeTminHourlyWeight(myHour);
        return (myTmin * tminWeight + myTmax * (1 - tminWeight));
    }
}


float tDewFromRelHum(float rhAir, float airT)
{
    if (int(rhAir) == int(NODATA) || int(airT) == int(NODATA))
        return NODATA;

    rhAir = MINVALUE(100, rhAir);

    double mySaturatedVaporPres = exp((16.78 * double(airT) - 116.9) / (double(airT) + 237.3));
    double actualVaporPres = double(rhAir) / 100. * mySaturatedVaporPres;
    return float((log(actualVaporPres) * 237.3 + 116.9) / (16.78 - log(actualVaporPres)));
}


float relHumFromTdew(float dewT, float airT)
{
    if (int(dewT) == int(NODATA) || int(airT) == int(NODATA))
        return NODATA;

    double d = 237.3;
    double c = 17.2693882;
    double esp = 1 / (double(airT) + d);
    double myValue = pow((exp((c * double(dewT)) - ((c * double(airT) / (double(airT) + d))) * (double(dewT) + d))), esp);
    myValue *= 100.;

    if (myValue > 100.)
        return 100;
    else if (myValue <= 0.)
        return 1.;
    else
        return float(myValue);
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

    Phi = PI / 180 * myLat;
    delta = 0.4093 * sin((2 * PI / 365) * myDoy - 1.39);
    dr = 1 + 0.033 * cos(2 * PI * myDoy / 365);
    OmegaS = acos(-tan(Phi) * tan(delta));

    return SOLAR_CONSTANT * DAY_SECONDS / 1000000 * dr / PI * (OmegaS * sin(Phi) * sin(delta) + cos(Phi) * cos(delta) * sin(OmegaS));
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
 * \brief 2016 GA. comments: G is ignored for now (if heat is active, should be added)
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

        myPsychro = Psychro(myPressure, myTmed);

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
        delta = SaturationSlope(myTmed, mySatVapPress);

        myDailySB = STEFAN_BOLTZMANN * DAY_SECONDS / 1000000;       /*!<   to MJ */
        myEmissivity = emissivityFromVaporPressure(myVapPress);
        myLWNetRad = myDailySB * (pow(myTmax + 273, 4) + pow(myTmin + 273, 4) / 2) * myEmissivity * (1.35 * (myTransmissivity / MAXTRANSMISSIVITY) - 0.35);

        mySWNetRad = mySWGlobRad * (1 - ALBEDO_CROP_REFERENCE);
        myNetRad = (mySWNetRad - myLWNetRad);

        myLambda = LatentHeatVaporization(myTmed) / 1000000; /*!<  to MJ */

        vmed2 = myVmed10 * 0.748;

        EvapDemand = 900 / (myTmed + 273) * vmed2 * (mySatVapPress - myVapPress);

        return (delta * myNetRad + myPsychro * EvapDemand / myLambda) / (delta + myPsychro * (1 + 0.34 * vmed2));

}



/*!
 * \brief ET0_Penman_hourly http://www.cimis.water.ca.gov/cimis/infoEtoPmEquation.jsp
 * \param heigth elevation above mean sea level (meters)
 * \param normalizedTransmissivity normalized tramissivity [0-1] ()
 * \param globalSWRadiation net Short Wave radiation (W m-2)
 * \param airTemp air temperature (C)
 * \param airHum relative humidity (%)
 * \param windSpeed10 wind speed at 10 meters (m s-1)
 * \return result
 */
double ET0_Penman_hourly(double heigth, double normalizedTransmissivity, double globalSWRadiation,
                double airTemp, double airHum, double windSpeed10)
{
    double mySigma;                              /*!<  Steffan-Boltzman constant J m-2 h-1 K-4 */
    double es;                                   /*!<  saturation vapor pressure (kPa) at the mean hourly air temperature in C */
    double ea;                                   /*!<  actual vapor pressure (kPa) at the mean hourly air temperature in C */
    double emissivity;                           /*!<  net emissivity of the surface */
    double cloudFactor;                          /*!<  cloudiness factor for long wave radiation */
    double netRadiation;                         /*!<  net radiation (J m-2 h-1) */
    double netLWRadiation;                       /*!<  net longwave radiation (J m-2 h-1) */
    double netSWRadiation;                       /*!<  net shortwave radiation (J m-2 h-1) */
    double g;                                    /*!<  soil heat flux density (J m-2 h-1) */
    double Cd;                                   /*!<  bulk surface resistance and aerodynamic resistance coefficient */
    double tAirK;                                /*!<  air temperature (Kelvin) */
    double windSpeed2;                           /*!<  wind speed at 2 meters (m s-1) */
    double delta;                                /*!<  slope of saturation vapor pressure curve (kPa C-1) at mean air temperature */
    double pressure;                             /*!<  barometric pressure (kPa) */
    double lambda;                               /*!<  latent heat of vaporization in (J kg-1) */
    double gamma;                                /*!<  psychrometric constant (kPa C-1) */
    double firstTerm, secondTerm, denominator;


    es = SaturationVaporPressure(airTemp) / 1000.;
    ea = airHum * es / 100.0;
    emissivity = emissivityFromVaporPressure(ea);
    tAirK = airTemp + ZEROCELSIUS;
    mySigma = STEFAN_BOLTZMANN * HOUR_SECONDS;
    cloudFactor = MAXVALUE(0, 1.35 * MINVALUE(normalizedTransmissivity, 1) - 0.35);
    netLWRadiation = cloudFactor * emissivity * mySigma * (pow(tAirK, 4));

    /*!   from [W m-2] to [J h-1 m-2] */
    netSWRadiation = (3600 * globalSWRadiation);
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

    delta = SaturationSlope(airTemp, es);

    pressure = PressureFromAltitude(heigth) / 1000.;

    gamma = Psychro(pressure, airTemp);
    lambda = LatentHeatVaporization(airTemp);

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
        float es = 0.611f * exp(17.27f * zT / (zT + float(ZEROCELSIUS) - 36.f));
        float zTwb = zT;
        float zTwbPrec = -999.f;

        while (abs(zTwb - zTwbPrec) > 0.1f)
        {
            zTwbPrec = zTwb;
            float zT1 = (zT + zTwb) / 2;
            float es1 = 0.611f * exp(17.27f * zT1 / (zT1 + float(ZEROCELSIUS) - 36.f));
            float delta = es1 / (zT1 + float(ZEROCELSIUS)) * log(207700000 / es1);
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

    *intensity = sqrt(u * u + v * v);

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
        case airDewTemperature: case dailyAirDewTemperatureAvg:
            setTemperatureScale(colorScale);
            break;
        case airRelHumidity: case dailyAirRelHumidityAvg: case dailyAirRelHumidityMax:
        case dailyAirRelHumidityMin: case leafWetness: case dailyLeafWetness:
            setRelativeHumidityScale(colorScale);
            break;
        case precipitation: case dailyPrecipitation: case referenceEvapotranspiration:
        case dailyReferenceEvapotranspirationHS: case dailyReferenceEvapotranspirationPM:
            setPrecipitationScale(colorScale);
            break;
        case dailyBIC:
            setZeroCenteredScale(colorScale);
            break;
        case globalIrradiance: case dailyGlobalRadiation: case atmTransmissivity:
            setRadiationScale(colorScale);
            break;
        case windVectorIntensity: case windScalarIntensity: case dailyWindVectorIntensityAvg: case dailyWindVectorIntensityMax: case dailyWindScalarIntensityAvg: case dailyWindScalarIntensityMax:
            setWindIntensityScale(colorScale);
            break;
        case anomaly:
            setAnomalyScale(colorScale);
            break;
        case noMeteoTerrain:
            setDefaultDEMScale(colorScale);
            break;

        default:
            setDefaultDEMScale(colorScale);
    }

    return true;
}


std::string getVariableString(meteoVariable myVar)
{
    if (myVar == airTemperature || myVar == dailyAirTemperatureAvg)
        return "Air temperature (°C)";
    else if (myVar == dailyAirTemperatureMax)
        return "Maximum air temperature (°C)";
    else if (myVar == dailyAirTemperatureMin)
        return "Minimum air temperature (°C)";
    else if (myVar == dailyAirTemperatureRange)
        return "Air temperature range (°C)";
    else if (myVar == airRelHumidity || myVar == dailyAirRelHumidityAvg)
        return "Air relative humidity (%)";
    else if (myVar == dailyAirRelHumidityMax)
        return "Maximum relative humidity (%)";
    else if (myVar == dailyAirRelHumidityMin)
        return "Minimum relative humidity (%)";
    else if (myVar == airDewTemperature || myVar == dailyAirDewTemperatureAvg)
        return "Air dew temperature (°C)";
    else if ((myVar == dailyPrecipitation ||  myVar == precipitation))
        return "Precipitation (mm)";
    else if (myVar == dailyGlobalRadiation)
        return "Solar radiation (MJ m-2)";
    else if (myVar == globalIrradiance)
        return "Solar irradiance (W m-2)";
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
             myVar == dailyReferenceEvapotranspirationPM)
        return "Reference evapotranspiration (mm)";
    else if (myVar == leafWetness || myVar == dailyLeafWetness)
        return "Leaf wetness (h)";
    else if (myVar == dailyBIC)
        return "Hydroclimatic balance (mm)";
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

frequencyType getVarFrequency(meteoVariable myVar)
{
    // todo: create maps of hourly and monthly variables
    if (MapDailyMeteoVarToString.find(myVar) != MapDailyMeteoVarToString.end())
        return daily;
    else
        return hourly;
}

meteoVariable getMeteoVar(std::string varString)
{
    meteoVariable meteoVar;

    try {
      meteoVar = MapDailyMeteoVar.at(varString);
    }
    catch (const std::out_of_range& ) {
        try {
            meteoVar = MapHourlyMeteoVar.at(varString);
        }
        catch (const std::out_of_range& ) {
            meteoVar = noMeteoVar;
        }
    }

    return (meteoVar);
}


meteoVariable getHourlyMeteoVar(std::string varString)
{
    meteoVariable meteoVar;

    try {
        meteoVar = MapHourlyMeteoVar.at(varString);
    }
    catch (const std::out_of_range& ) {
        meteoVar = noMeteoVar;
    }

    return (meteoVar);
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
