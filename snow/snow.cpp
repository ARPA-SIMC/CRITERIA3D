/*!
    \copyright Fausto Tomei, Gabriele Antolini,
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
    ftomei@arpae.it
*/

#include <algorithm>
#include <math.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "meteo.h"
#include "snow.h"


Crit3DSnowParameters::Crit3DSnowParameters()
{
    initialize();
}


void Crit3DSnowParameters::initialize()
{
    // default values
    skinThickness = 0.02;               /*!<  [m]   */ // VERSIONI DIVERSE IN BROOKS: 3mm (nel testo), 2-3cm (nel codice)
    soilAlbedo = 0.2;                   /*!<  [-] bare soil */
    snowVegetationHeight = 1;           /*!<  [m]   */
    snowWaterHoldingCapacity = 0.05;    /*!<  [-]   */
    tempMaxWithSnow = 2;                /*!<  [°C]  */
    tempMinWithRain = -0.5;             /*!<  [°C]  */
    snowSurfaceDampingDepth = 0.05;     /*!<  [m]   */ // VERSIONI DIVERSE IN BROOKS: 0.05 - 0.15
}


Crit3DSnow::Crit3DSnow()
{
    initialize();
}


void Crit3DSnow::initialize()
{
    snowParameters.initialize();

    // input
    _airT = NODATA;
    _prec = NODATA;
    _airRH = NODATA;
    _windInt = NODATA;
    _globalRadiation = NODATA;
    _beamRadiation = NODATA;
    _transmissivity = NODATA;
    _clearSkyTransmissivity = NODATA;
    _surfaceWaterContent = NODATA;

    // output
    _precSnow = NODATA;
    _precRain = NODATA;
    _snowMelt = NODATA;
    _sensibleHeat = NODATA;
    _latentHeat = NODATA;
    _evaporation = NODATA;

    _snowWaterEquivalent = NODATA;
    _iceContent = NODATA;
    _liquidWaterContent = NODATA;
    _internalEnergy = NODATA;
    _surfaceEnergy = NODATA;
    _surfaceTemp = NODATA;
    _ageOfSnow = NODATA;
}


void Crit3DSnow::setInputData(double temp, double prec, double relHum, double windInt, double globalRad,
                              double beamRad, double transmissivity, double clearSkyTransmissivity, double waterContent)
{
    _airT = temp;
    _prec = prec;
    _airRH = relHum;
    _windInt = windInt;
    _globalRadiation = globalRad;
    _beamRadiation = beamRad;
    _transmissivity = transmissivity;
    _clearSkyTransmissivity = clearSkyTransmissivity;
    _surfaceWaterContent = std::max(waterContent, 0.0);
}


bool Crit3DSnow::checkValidPoint()
{
    if ( int(_airT) == int(NODATA)
        || int(_prec) == int(NODATA)
        || int(_globalRadiation) == int(NODATA)
        || int(_beamRadiation) == int(NODATA)
        || int(_snowWaterEquivalent) == int(NODATA)
        || int(_surfaceTemp) == int(NODATA) )
    {
        return false;
    }

    return true;
}


void Crit3DSnow::computeSnowFall()
{
    double liquidWater = _prec;

    if (liquidWater > 0)
    {
        if (_airT <= snowParameters.tempMinWithRain)
        {
            liquidWater = 0;
        }
         else if (_airT < snowParameters.tempMaxWithSnow)
        {
            liquidWater *= (_airT - snowParameters.tempMinWithRain) / (snowParameters.tempMaxWithSnow - snowParameters.tempMinWithRain);
        }
    }

    _precSnow = MAXVALUE(_prec - liquidWater, 0);
    _precRain = liquidWater;
}


void Crit3DSnow::computeSnowBrooksModel()
{
    double solarRadTot;
    double cloudCover;                           /*!<   [-]        */
    double currentRatio;

    double airActualVapDensity;                  /*!<   [kg m-3]   */
    double waterActualVapDensity;                /*!<   [kg m-3]   */
    double longWaveAtmEmissivity;                /*!<   [-]        */
    double albedo;                               /*!<   [-]        */

    double QSolar;                               /*!<   [kJ/m^2]   integrale della radiazione solare */
    double QPrecip;                              /*!<   [kJ/m^2]   avvezione (trasferimento di calore dalla precipitazione) */
    double QPrecipW;                             /*!<   [kJ/m^2]   */
    double QPrecipS;                             /*!<   [kJ/m^2]   */
    double QLongWave;                            /*!<   [kJ/m^2]   emissione di radiazione onda lunga (eq. Stefan Boltzmann) */
    double QTempGradient;                        /*!<   [kJ/m^2]   scambio di calore sensibile (dovuto al gradiente di temperatura) */
    double QVaporGradient;                       /*!<   [kJ/m^2]   scambio di calore latente (vapore) */
    double QTotal;                               /*!<   [kJ/m^2]   */
    double QWaterHeat;                           /*!<   [kJ/m^2]   */
    double QWaterKinetic;                        /*!<   [kJ/m^2]   */

    double sublimation;                          /*!<   [mm]       */
    double aerodynamicResistance;                /*!<   [s m-1]    */

    double const ONE_HOUR = 1./24.;

    // free water
    bool isWater = false;
    if (_surfaceWaterContent > 100. )     /*!<  [mm]  acqua libera (fiumi - torrenti) */
            isWater = true;

    if (isWater || (! checkValidPoint()))
    {
        _snowMelt = NODATA;
        _iceContent = NODATA;
        _liquidWaterContent = NODATA;
        _snowWaterEquivalent = NODATA;
        _surfaceEnergy = NODATA;
        _surfaceTemp = NODATA;
        _ageOfSnow = NODATA;

        _precSnow = NODATA;
        _precRain = NODATA;
        _snowMelt = NODATA;
        _sensibleHeat = NODATA;
        _latentHeat = NODATA;
        _evaporation = NODATA;
        return;
    }

    computeSnowFall();

    double dewPoint = double(tDewFromRelHum(_airRH, _airT));     /*!< [°C] */

    if (! isEqual(_transmissivity, NODATA))
        cloudCover = 1 - std::min(double(_transmissivity) / _clearSkyTransmissivity, 1.);
    else
        cloudCover = 0.1;

    // vegetation shadowing (4 m sopra manto nevoso: ombreggiamento completo)
    // TODO: use LAI when available
    double maxSnowDensity = 10;          // 1 mm snow = 1 cm water
    double maxVegetationHeight = 4;      // [m]
    double vegetationShadowing;          // [-]
    double maxSnowHeight = _snowWaterEquivalent * maxSnowDensity / 1000;                 // [m]
    double heightVegetation = snowParameters.snowVegetationHeight - maxSnowHeight;       // [m]
    vegetationShadowing = std::max(std::min(heightVegetation / maxVegetationHeight, 1.), 0.);
    solarRadTot = _globalRadiation - _beamRadiation * vegetationShadowing;

    double previousSWE = _snowWaterEquivalent;
    double prevInternalEnergy = _internalEnergy;
    double prevSurfaceEnergy = _surfaceEnergy;
    double prevSurfaceTemp = _surfaceTemp;
    double prevIceContent = _iceContent;
    double prevLWaterContent = _liquidWaterContent;

    if (previousSWE > 0)
    {
        /*--------------------------------------------------------------------
        // controlli di coerenza per eventuali modifiche manuali su mappa SWE
        // -------------------------------------------------------------------*/
        if (prevIceContent <= 0 && prevLWaterContent <= 0)
        {
            prevIceContent = previousSWE;
            prevLWaterContent = previousSWE * snowParameters.snowWaterHoldingCapacity / (1 - snowParameters.snowWaterHoldingCapacity);

            // Pag. 53 formula 3.23
            prevInternalEnergy = -previousSWE * 0.001 * LATENT_HEAT_FUSION * WATER_DENSITY;

            prevSurfaceTemp = std::min(prevSurfaceTemp, 0.);
            prevSurfaceEnergy = computeSurfaceEnergySnow(prevSurfaceTemp, std::min(previousSWE, snowParameters.skinThickness));

            _ageOfSnow = 1;
        }

        /*! check on sum */
        currentRatio = previousSWE / (prevIceContent + prevLWaterContent);
        if (! isEqual(currentRatio, 1) )
        {
            prevIceContent = prevIceContent * currentRatio;
            prevLWaterContent = prevLWaterContent * currentRatio;
        }
    }
    else
    {
        prevIceContent = 0;
        prevLWaterContent = 0;
        _ageOfSnow = NODATA;
    }

    /*! \brief check on soil internal energy - added by ftomei  */

    if ( previousSWE < EPSILON )
    {
        // no snow condition: all soil
        double estInternalEnergy = computeInternalEnergySoil(prevSurfaceTemp, DEFAULT_BULK_DENSITY);    // [kJ m-2]
        double absDifference = fabs(estInternalEnergy - prevInternalEnergy);                            // [kJ m-2]

        // the difference is very high
        if (absDifference > 1000)
        {
            // check to avoid division by zero
            if ( isEqual(estInternalEnergy, 0) )
            {
                estInternalEnergy = EPSILON;
            }
            double ratio = prevInternalEnergy / estInternalEnergy;                                      // [-]
            if ( (ratio < 0.5) || (ratio > 2))
            {
                prevInternalEnergy = (prevInternalEnergy + estInternalEnergy) * 0.5;
            }
        }
    }

    /*! \brief compute Roughness and Vapor Density  */

    // brooks originale
    if ( previousSWE > SNOW_MINIMUM_HEIGHT)
        aerodynamicResistance = aerodynamicResistanceCampbell77(true, 10, _windInt, snowParameters.snowVegetationHeight);
    else
        aerodynamicResistance = aerodynamicResistanceCampbell77(false, 10, _windInt, snowParameters.snowVegetationHeight);

    // pag. 52 (3.20)
    // source: Jensen et al. (1990) and Tetens (1930)
    // saturated vapor density
    airActualVapDensity = double(exp((16.78 * dewPoint - 116.9) / (dewPoint + 237.3))
                              / ((ZEROCELSIUS + dewPoint) * THERMO_WATER_VAPOR) );

    // over snow
    waterActualVapDensity = double( exp((16.78 * prevSurfaceTemp - 116.9) / (prevSurfaceTemp + 237.3))
                                / ((ZEROCELSIUS + prevSurfaceTemp) * THERMO_WATER_VAPOR) );

    // over ice
    // LC: non trovo riferimenti a questa formula
    /*
    if (previousSWE > EPSILON)
    {
        waterActualVapDensity *= exp(MH2O * LATENT_HEAT_FUSION * prevSurfaceTemp
                                 / (R_GAS * 0.001 * pow(prevSurfaceTemp + ZEROCELSIUS, 2)) );
    }
    */

    /*!
    * \brief Atmospheric Emissivity Calculations for Longwave Radiation
    *-----------------------------------------------------------
    * Unsworth, M.H. and L.J. Monteith. 1975. Long-wave radiation a the ground. I. Angular distribution of incoming radiation. Quarterly Journal of the Royal Meteorological Society 101(427):13-24.
    */

    longWaveAtmEmissivity = (0.72 + 0.005 * _airT) * (1.0 - 0.84 * cloudCover) + 0.84 * cloudCover;

    /*! albedo */
    if (! isEqual(_ageOfSnow, NODATA))
        /*! O'NEILL, A.D.J. GRAY D.M.1973. Spatial and temporal variations of the albedo of prairie snowpacks. The Role of Snow and Ice in Hydrology: Proceedings of the Banff Syn~posia, 1972. Unesc - WMO -IAHS, Geneva -Budapest-Paris, Vol. 1,  pp. 176-186
        * arrotondato da U.S. Army Corps
        */
        albedo = std::min(0.9, 0.74 * pow(_ageOfSnow , -0.191));
    else
        albedo = snowParameters.soilAlbedo;

    /*! \brief Incoming Energy Fluxes */

    // pag. 52 (3.22) considerando i 2 contributi invece che solo uno
    QPrecipW = (HEAT_CAPACITY_WATER / 1000.) * (_precRain / 1000.) * (std::max(0., _airT) - prevSurfaceTemp);
    QPrecipS = (HEAT_CAPACITY_SNOW / 1000.) * (_precSnow / 1000.) * (std::min(0., _airT) - prevSurfaceTemp);
    QPrecip = QPrecipW + QPrecipS;

    // temperatura dell'acqua: almeno 1 grado
    QWaterHeat = (HEAT_CAPACITY_WATER / 1000.) * (_surfaceWaterContent / 1000.)
                 * (std::max(1., (prevSurfaceTemp + _airT) / 2.) - prevSurfaceTemp);

    // energia acqua libera
    QWaterKinetic = 0;

    // TODO free water flux
    /*
    double freeWaterFlux = 0;
    nodeIndex = GIS.GetValueFromXY(criteria3DModule.Crit3DIndexMap(0), x, y)
    If nodeIndex <> criteria3DModule.Crit3DIndexMap(0).header.flag
    {
        //[m3/h]
        freeWaterFlux = criteria3DModule.getLateralFlow(nodeIndex, False)
        //[m2]
        avgExchangeArea = Crit3DIndexMap(0).header.cellSize * (surfaceWaterContent / 1000.0)
        //[m/s]
        freeWaterFlux = (freeWaterFlux / avgExchangeArea) / 3600.0

        if (freeWaterFlux > 0.01)
        {
            avgMass = (surfaceWaterContent / 1000.0) * WATER_DENSITY; //[kg/m2]
            QWaterKinetic = 0.5 * avgMass * (freeWaterFlux * freeWaterFlux) / 1000.0; //[kJ/m2]
        }
    }
    */

    // pag. 50 (3.14)
    QSolar = (1. - albedo) * (solarRadTot * 3600.) / 1000.;

    double surfaceEmissivity;                          //  [-]
    if (previousSWE > SNOW_MINIMUM_HEIGHT)
        surfaceEmissivity = double(SNOW_EMISSIVITY);
    else
        surfaceEmissivity = double(SOIL_EMISSIVITY);

    // pag. 50 (3.15)
    QLongWave = double(STEFAN_BOLTZMANN * 3.6 * (longWaveAtmEmissivity * pow(_airT + ZEROCELSIUS, 4.0)
              - surfaceEmissivity * pow (prevSurfaceTemp + ZEROCELSIUS, 4.0)));

    // sensible heat pag. 50 (3.17)
    QTempGradient = 3600. * (HEAT_CAPACITY_AIR / 1000.) * (_airT - prevSurfaceTemp) / aerodynamicResistance;

    // latent heat pag. 51 (3.19)
    // FT tolta WATER_DENSITY dall'eq. (non corrispondevano le unità di misura)
    QVaporGradient = 3600. * (LATENT_HEAT_VAPORIZATION + LATENT_HEAT_FUSION)
            * (airActualVapDensity - waterActualVapDensity) / aerodynamicResistance;

    // TODO serve formula diversa quando non c'è neve
    if (previousSWE < EPSILON)
    {
        QVaporGradient *= 0.4;
    }

    /*! Energy Balance */
    QTotal = QSolar + QPrecip + QLongWave + QTempGradient + QVaporGradient + QWaterHeat + QWaterKinetic;
    _sensibleHeat = QTempGradient;
    _latentHeat = QVaporGradient;

    /*! Condensation (positive) or evaporation (negative) */
    sublimation = 0;                        // [mm]
    _evaporation = 0;                       // [mm]
    if (previousSWE > EPSILON)
    {
        // pag. 51 (3.21) [mm]
        sublimation = QVaporGradient / (LATENT_HEAT_FUSION + LATENT_HEAT_VAPORIZATION);

        if (sublimation < 0)
        {
            /*! Evaporation [mm]
             *  controllo aggiunto (può evaporare solo la neve presente)
             */
            sublimation = -std::min(fabs(sublimation), previousSWE + _precSnow);
            _evaporation = -sublimation;
        }
    }

    /*! refreeze or melt */
    double freeze_melt = 0;      // [mm] freeze (positive) or melt (negative)

    /*! net amount of liquid water that freezes (heat added to the snow pack)
     *  and ice that melts (heat removed from the snow pack)
     *  pag.53 (3.25)
    */
    double w = (prevInternalEnergy + QTotal) / (LATENT_HEAT_FUSION * WATER_DENSITY);    // [m]
    if (w < 0)
    {
        /*! freeze
         *  add check on surface temperatures
         */
        if (prevSurfaceTemp <= 0)
        {
            freeze_melt = std::min(prevLWaterContent + _precRain, -w * 1000.);          // [mm]
        }
    }
    else if (w > 0)
    {
        /*! melt */
        freeze_melt = -std::min(prevIceContent + _precSnow + sublimation, w * 1000.);   // [mm]
    }

    /*! Snowmelt or refreeze [mm] - source/sink for Criteria3D */
    _snowMelt = -freeze_melt;

    /*! latent heat exchange in the snow pack [kJ m-2]
     *  pag.53 (3.23) modificata (errore nel denominatore)
     */
    double Qr = (freeze_melt / 1000.) * LATENT_HEAT_FUSION * WATER_DENSITY;

    /*! Internal energy [kJ m-2] */
    _internalEnergy = prevInternalEnergy + QTotal + Qr;

    /*! Snow Pack Mass */

    /*! Ice content */
    if (_internalEnergy > EPSILON)
    {
        _iceContent = 0;
    }
    else
    {
        _iceContent = prevIceContent + _precSnow + sublimation + freeze_melt;
        _iceContent = std::max(_iceContent, 0.);
    }

    double waterHoldingCapacity = snowParameters.snowWaterHoldingCapacity
                                  / (1 - snowParameters.snowWaterHoldingCapacity);      // [%]

    /*! Liquid water content */
    if (_internalEnergy > EPSILON)
    {
        _liquidWaterContent = 0;
    }
    else
    {
        _liquidWaterContent = prevLWaterContent + _precRain + _surfaceWaterContent - freeze_melt;    // [mm]
        _liquidWaterContent = std::min(std::max(_liquidWaterContent, 0.), _iceContent * waterHoldingCapacity);
    }

    /*! Snow water equivalent */
    _snowWaterEquivalent = _iceContent + _liquidWaterContent;

    /*! surface energy [kJ m-2] and surface temperature [°C] */
    double surfaceEnergySnow;
    // snow
    if (_snowWaterEquivalent > 0 && fabs(_internalEnergy) < EPSILON)
    {
        surfaceEnergySnow = 0.;
    }
    else
    {
        double snowRatio = std::min(snowWaterEquivalent * 0.001, snowParameters.skinThickness) / snowParameters.snowSurfaceDampingDepth;
        surfaceEnergySnow = std::min(0., prevSurfaceEnergy + (QTotal + Qr) * snowRatio);
    }
    double surfaceTempSnow = surfaceEnergySnow / (WATER_DENSITY * SNOW_SPECIFIC_HEAT * snowParameters.skinThickness);

    // all soil
    double surfaceEnergySoil = prevSurfaceEnergy + (QTotal + Qr) * (snowParameters.skinThickness / SOIL_DAMPING_DEPTH);
    double surfaceTempSoil = surfaceEnergySoil / (DEFAULT_BULK_DENSITY * SOIL_SPECIFIC_HEAT * snowParameters.skinThickness);

    double snowDepthRatio = 4.;
    double snowFraction = std::min(_snowWaterEquivalent * snowDepthRatio / 1000., snowParameters.skinThickness) / snowParameters.skinThickness;

    _surfaceEnergy = (surfaceEnergySnow * snowFraction) + surfaceEnergySoil * (1 - snowFraction);
    _surfaceTemp = (surfaceTempSnow * snowFraction) + surfaceTempSoil * (1 - snowFraction);

    /*! _ageOfSnow [days] */
    if (_snowWaterEquivalent < EPSILON)
    {
        _ageOfSnow = NODATA;
    }
    else
    {
        if (_ageOfSnow == NODATA || _precSnow > EPSILON)
            _ageOfSnow = 0;
        else
            _ageOfSnow += ONE_HOUR;
    }
}


double Crit3DSnow::getSnowFall()
{
    return _precSnow;
}

double Crit3DSnow::getSnowMelt()
{
    return MAXVALUE(_snowMelt, 0);
}


double Crit3DSnow::getSensibleHeat()
{
    return _sensibleHeat;
}

double Crit3DSnow::getLatentHeat()
{
    return _latentHeat;
}

double Crit3DSnow::getSnowWaterEquivalent()
{
    return _snowWaterEquivalent;
}
void Crit3DSnow::setSnowWaterEquivalent(float value)
{
    _snowWaterEquivalent = double(value);
}

double Crit3DSnow::getIceContent()
{
    return _iceContent;
}
void Crit3DSnow::setIceContent(float value)
{
    _iceContent = double(value);
}

double Crit3DSnow::getLiquidWaterContent()
{
    return _liquidWaterContent;
}

void Crit3DSnow::setLiquidWaterContent(float value)
{
    _liquidWaterContent = double(value);
}

double Crit3DSnow::getInternalEnergy()
{
    return _internalEnergy;
}

void Crit3DSnow::setInternalEnergy(float value)
{
    _internalEnergy = double(value);
}

double Crit3DSnow::getSurfaceEnergy()
{
    return _surfaceEnergy;
}

void Crit3DSnow::setSurfaceEnergy(float value)
{
    _surfaceEnergy = double(value);
}

double Crit3DSnow::getSnowSurfaceTemp()
{
    return _surfaceTemp;
}

void Crit3DSnow::setSnowSurfaceTemp(float value)
{
    _surfaceTemp = double(value);
}

double Crit3DSnow::getAgeOfSnow()
{
    return _ageOfSnow;
}

void Crit3DSnow::setAgeOfSnow(float value)
{
    _ageOfSnow = double(value);
}


/*!
 * \brief Computes aerodynamic Resistance (resistance to heat transfer)
 * \param zRefWind          [m] heights of windspeed measurements
 * \param windSpeed         [m s-1] wind speed
 * \param vegetativeHeight  [m] height of the vegetation
 * \ref Brooks pag.51 (3.18)
 * \return aerodynamic Resistance [s m-1]
 */
double aerodynamicResistanceCampbell77(bool isSnow , double zRefWind, double windSpeed, double vegetativeHeight)
{
    double zeroPlane;            /*!  [m] zero-plane displacement (snow = 0m, vegetative cover d = 0.64 times the height of the vegetative) */
    double momentumRoughness;    /*!  [m] momentum roughness parameter (for snow = 0.001m, for vegetative cover zm = 0.13 times the height of the vegetation) */
    double zRefTemp = 2;         /*!  [m] heights of temperature measurements */

    /*! check on wind speed [m s-1] */
    windSpeed = std::max(windSpeed, 0.05);
    windSpeed = std::min(windSpeed, 10.);

    /*! check on vegetativeHeight  [m] */
    vegetativeHeight = std::max(vegetativeHeight, 0.01);

    if (isSnow)
    {
        zeroPlane = 0;
        momentumRoughness = 0.001;
    }
    else
    {
        zeroPlane = 0.64 * vegetativeHeight;
        momentumRoughness = 0.13 * vegetativeHeight;
    }

    double log1 = log((MAXVALUE(zRefWind - zeroPlane, 1.0) + momentumRoughness) / momentumRoughness);

    double heatVaporRoughness = 0.2 * momentumRoughness;
    double log2 = log((MAXVALUE(zRefTemp - zeroPlane, 1.0) + heatVaporRoughness) / heatVaporRoughness);

    return log1 * log2 / (VON_KARMAN_CONST * VON_KARMAN_CONST * windSpeed);
}


// pag. 54 (3.29)
double computeInternalEnergy(double soilTemperature, int bulkDensity, double swe)
{
    return soilTemperature * ( WATER_DENSITY * SNOW_SPECIFIC_HEAT * swe * 0.001
                                  + bulkDensity * SOIL_SPECIFIC_HEAT * SOIL_DAMPING_DEPTH );
}


double computeInternalEnergySoil(double soilTemperature, int bulkDensity)
{
    return soilTemperature * bulkDensity * SOIL_SPECIFIC_HEAT * SOIL_DAMPING_DEPTH;
}


double computeSurfaceEnergySnow(double surfaceTemperature, double skinThickness)
{
    return surfaceTemperature * WATER_DENSITY * SNOW_SPECIFIC_HEAT * skinThickness;
}


double computeSurfaceEnergySoil(double surfaceTemperature, double skinThickness)
{
    return surfaceTemperature * DEFAULT_BULK_DENSITY * SOIL_SPECIFIC_HEAT * skinThickness;
}

