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

#include "basicMath.h"
#include "meteo.h"
#include "snow.h"


Crit3DSnowParameters::Crit3DSnowParameters()
{
    initializeSnowParameters();
}


void Crit3DSnowParameters::initializeSnowParameters()
{
    // default values
    skinThickness = 0.02;               /*!<  [m]   */  // versioni diverse in Brooks: 2-3 cm
    soilAlbedo = 0.2;                   /*!<  [-]   */  // bare soil
    snowVegetationHeight = 1;           /*!<  [m]   */
    snowWaterHoldingCapacity = 0.05;    /*!<  [-]   */
    tempMaxWithSnow = 2;                /*!<  [°C]  */
    tempMinWithRain = -0.5;             /*!<  [°C]  */
    snowSurfaceDampingDepth = 0.05;     /*!<  [m]   */  // versioni diverse in Brooks: 0.05 - 0.15
}


Crit3DSnow::Crit3DSnow()
{
    initializeSnow();
}


void Crit3DSnow::initializeSnow()
{
    snowParameters.initializeSnowParameters();

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
    _deltaSWE = NODATA;
    _waterOutflow = NODATA;
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
    _snowDensity = NODATA;
}


void Crit3DSnow::setSnowInputData(double temp, double prec, double relHum, double windInt, double globalRad,
                              double beamRad, double transmissivity, double clearSkyTransmissivity, double surfaceWaterContent)
{
    _airT = temp;
    _prec = prec;
    _airRH = relHum;
    _windInt = windInt;
    _globalRadiation = globalRad;
    _beamRadiation = beamRad;
    _transmissivity = transmissivity;
    _clearSkyTransmissivity = clearSkyTransmissivity;
    _surfaceWaterContent = std::max(surfaceWaterContent, 0.0);     // [mm]
}


bool Crit3DSnow::isSnowPointValid()
{
    if (   isEqual(_airT, NODATA)
        || isEqual(_prec, NODATA)
        || isEqual(_airRH, NODATA)
        || isEqual(_windInt, NODATA)
        || isEqual(_globalRadiation, NODATA)
        || isEqual(_beamRadiation, NODATA)
        || isEqual(_snowWaterEquivalent, NODATA)
        || isEqual(_surfaceTemp, NODATA) )
        return false;
    else
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
            const double delta = snowParameters.tempMaxWithSnow - snowParameters.tempMinWithRain;
            if (isEqual(delta, 0))
                liquidWater = 0;
            else
                liquidWater *= (_airT - snowParameters.tempMinWithRain) / delta;
        }
    }

    _precSnow = MAXVALUE(_prec - liquidWater, 0);
    _precRain = liquidWater;
}


void Crit3DSnow::computeSnowDensity()
{
    if (_snowWaterEquivalent <= 0.)
    {
        _snowDensity = NODATA;
        return;
    }

    double oldDensity = _snowDensity;
    if (isEqual(oldDensity, NODATA))
        oldDensity = SNOW_DENSITY_DEFAULT;

    oldDensity = std::max(SNOW_DENSITY_FRESH, std::min(oldDensity, SNOW_DENSITY_MAX_WET));

    // wetness [0-1]: liquid water fraction with respect to the holding capacity
    double wetness = 0;
    if (snowParameters.snowWaterHoldingCapacity > 0)
    {
        wetness = std::min((_liquidWaterContent / _snowWaterEquivalent) / snowParameters.snowWaterHoldingCapacity, 1.);
        wetness = std::max(wetness, 0.);
    }
    double maxDensity = SNOW_DENSITY_MAX_DRY + (SNOW_DENSITY_MAX_WET - SNOW_DENSITY_MAX_DRY) * wetness;

    // settling: the density tends to maxDensity (it never decreases)
    if (oldDensity < maxDensity)
    {
        double timeStep = HOUR_SECONDS / DAY_SECONDS;       // [days]
        oldDensity = maxDensity - (maxDensity - oldDensity) * exp(-timeStep / SNOW_DENSITY_TIMESCALE);
    }

    // new snow: mass-weighted mean of the volumes
    double newSnow = std::min(std::max(_precSnow, 0.), _snowWaterEquivalent);
    double oldSnow = _snowWaterEquivalent - newSnow;

    _snowDensity = _snowWaterEquivalent / (oldSnow / oldDensity + newSnow / SNOW_DENSITY_FRESH);
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

    double QSolar;                               /*!<   [kJ m-2]   integrale della radiazione solare */
    double QPrecip;                              /*!<   [kJ m-2]   avvezione (trasferimento di calore dalla precipitazione) */
    double QPrecipW;                             /*!<   [kJ m-2]   */
    double QPrecipS;                             /*!<   [kJ m-2]   */
    double QLongWave;                            /*!<   [kJ m-2]   emissione di radiazione onda lunga (eq. Stefan Boltzmann) */
    double QTempGradient;                        /*!<   [kJ m-2]   scambio di calore sensibile (dovuto al gradiente di temperatura) */
    double QVaporGradient;                       /*!<   [kJ m-2]   scambio di calore latente (vapore) */
    double QTotal;                               /*!<   [kJ m-2]   */
    double QWaterHeat;                           /*!<   [kJ m-2]   */
    double QWaterKinetic;                        /*!<   [kJ m-2]   */

    double sublimation;                          /*!<   [mm]       */
    double aerodynamicResistance;                /*!<   [s m-1]    */

    bool isFreeWater = false;
    if (_surfaceWaterContent > 100. )            /*!<  [mm]  acqua libera (fiumi - torrenti) */
            isFreeWater = true;

    bool isStateValid = (! isEqual(_snowWaterEquivalent, NODATA) && ! isEqual(_surfaceTemp, NODATA));

    if (! isFreeWater && isStateValid && ! isSnowPointValid())
    {
        // missing meteo data: the state variables are preserved, only the outputs are not available
        _precSnow = NODATA;
        _precRain = NODATA;
        _deltaSWE = NODATA;
        _waterOutflow = NODATA;
        _snowMelt = NODATA;

        _sensibleHeat = NODATA;
        _latentHeat = NODATA;
        _evaporation = NODATA;

        return;
    }

    if (isFreeWater || ! isStateValid)
    {
        _iceContent = NODATA;
        _liquidWaterContent = NODATA;
        _snowWaterEquivalent = NODATA;
        _internalEnergy = NODATA;
        _surfaceEnergy = NODATA;
        _surfaceTemp = NODATA;
        _ageOfSnow = NODATA;
        _snowDensity = NODATA;

        _precSnow = NODATA;
        _precRain = NODATA;
        _deltaSWE = NODATA;
        _waterOutflow = NODATA;
        _snowMelt = NODATA;

        _sensibleHeat = NODATA;
        _latentHeat = NODATA;
        _evaporation = NODATA;

        return;
    }

    computeSnowFall();

    double dewPoint = double(tDewFromRelHum(_airRH, _airT));     /*!< [°C] */

    if (! isEqual(_transmissivity, NODATA) && ! isEqual(_clearSkyTransmissivity, NODATA) && _clearSkyTransmissivity > 0)
        cloudCover = 1 - std::min(double(_transmissivity) / _clearSkyTransmissivity, 1.);
    else
        cloudCover = 0.1;

    // vegetation shadowing (4 m sopra manto nevoso: ombreggiamento completo)
    // TODO: use LAI when available
    double maxVegetationHeight = 4;         // [m]
    double vegetationShadowing;             // [-]

    double oldSnowDepth = 0.;               // [m]
    if (! isEqual(_snowDensity, NODATA))
        oldSnowDepth = _snowWaterEquivalent / _snowDensity;
    else if (_snowWaterEquivalent > 0)
        oldSnowDepth = _snowWaterEquivalent / SNOW_DENSITY_DEFAULT;

    double heightVegetation = snowParameters.snowVegetationHeight - oldSnowDepth;               // [m]
    vegetationShadowing = std::max(std::min(heightVegetation / maxVegetationHeight, 1.), 0.);
    solarRadTot = _globalRadiation - _beamRadiation * vegetationShadowing;

    // fraction of the surface layer occupied by snow [-]
    double oldSnowFraction = std::min(oldSnowDepth, snowParameters.skinThickness) / snowParameters.skinThickness;

    // missing energy: it is estimated from the surface temperature
    if (isEqual(_surfaceEnergy, NODATA))
    {
        if (_snowWaterEquivalent > 0)
            _surfaceEnergy = computeSurfaceEnergySnow(std::min(_surfaceTemp, 0.), snowParameters.skinThickness);
        else
            _surfaceEnergy = computeSurfaceEnergySoil(_surfaceTemp, snowParameters.skinThickness);
    }
    if (isEqual(_internalEnergy, NODATA))
    {
        if (_snowWaterEquivalent > 0)
            _internalEnergy = computeInternalEnergy(std::min(_surfaceTemp, 0.), DEFAULT_BULK_DENSITY, _snowWaterEquivalent);
        else
            _internalEnergy = computeInternalEnergySoil(_surfaceTemp, DEFAULT_BULK_DENSITY);
    }

    double previousSWE = _snowWaterEquivalent;
    double prevInternalEnergy = _internalEnergy;
    double prevSurfaceEnergy = _surfaceEnergy;
    double prevSurfaceTemp = _surfaceTemp;
    double prevIceContent = _iceContent;
    double prevLWaterContent = _liquidWaterContent;

    if (previousSWE > 0)
    {
        /*--------------------------------------------------------------------
        // aggiorna le variabili dopo modifiche manuali su mappa SWE
        // -------------------------------------------------------------------*/
        if (prevIceContent <= 0 && prevLWaterContent <= 0)
        {
            prevIceContent = previousSWE;
            prevLWaterContent = 0;

            // snow surface cannot be above 0 °C
            prevSurfaceTemp = std::min(prevSurfaceTemp, 0.);
            prevSurfaceEnergy = computeSurfaceEnergySnow(prevSurfaceTemp, snowParameters.skinThickness);

            // pack temperature equal to the surface temperature
            prevInternalEnergy = computeInternalEnergy(prevSurfaceTemp, DEFAULT_BULK_DENSITY, previousSWE);

            _ageOfSnow = 1;
        }

        // snow without age (first step after initialization)
        if (isEqual(_ageOfSnow, NODATA))
            _ageOfSnow = 1;

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
        _snowDensity = NODATA;
    }

    /*! \brief check on soil internal energy - added by ftomei  */
    if ( previousSWE < EPSILON )
    {
        // no snow condition: all soil
        double estInternalEnergy = computeInternalEnergySoil(prevSurfaceTemp, DEFAULT_BULK_DENSITY);    // [kJ m-2]
        const double absDifference = fabs(estInternalEnergy - prevInternalEnergy);                      // [kJ m-2]

        // the difference is very high
        if (absDifference > 1000.)
        {
            // check to avoid division by zero
            if (isEqual(estInternalEnergy, 0))
                estInternalEnergy = EPSILON;

            const double ratio = prevInternalEnergy / estInternalEnergy;       // [-]
            if ((ratio < 0.5) || (ratio > 2))
            {
                prevInternalEnergy = (prevInternalEnergy + estInternalEnergy) * 0.5;
            }
        }
    }

    /*! \brief compute Roughness and Vapor Density  */

    // brooks originale, with a gradual transition between soil and snow
    double resistanceSnow = aerodynamicResistanceCampbell77(true, 10, _windInt, snowParameters.snowVegetationHeight);
    double resistanceSoil = aerodynamicResistanceCampbell77(false, 10, _windInt, snowParameters.snowVegetationHeight);
    aerodynamicResistance = resistanceSnow * oldSnowFraction + resistanceSoil * (1 - oldSnowFraction);

    // pag. 52 (3.20)
    // source: Jensen et al. (1990) and Tetens (1930)
    // saturated vapor density
    // actual vapor pressure (saturated at dew point) at air temperature
    airActualVapDensity = double(exp((16.78 * dewPoint - 116.9) / (dewPoint + 237.3))
                              / ((ZEROCELSIUS + _airT) * THERMO_WATER_VAPOR) );

    // saturated vapor pressure at the surface [kPa]
    double surfaceVapPressure;
    if (previousSWE > EPSILON && prevSurfaceTemp < 0)
    {
        // over ice: Murray (1967)
        surfaceVapPressure = 0.6108 * exp(21.875 * prevSurfaceTemp / (prevSurfaceTemp + 265.5));
    }
    else
    {
        // over water
        surfaceVapPressure = exp((16.78 * prevSurfaceTemp - 116.9) / (prevSurfaceTemp + 237.3));
    }
    waterActualVapDensity = surfaceVapPressure / ((ZEROCELSIUS + prevSurfaceTemp) * THERMO_WATER_VAPOR);

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

    /*!
    *  \brief albedo
    *---------
    * O'NEILL, A.D.J. GRAY D.M.1973. Spatial and temporal variations of the albedo of prairie snowpacks. The Role of Snow and Ice in Hydrology: Proceedings of the Banff Syn~posia, 1972. Unesc - WMO -IAHS, Geneva -Budapest-Paris, Vol. 1,  pp. 176-186
    * arrotondato da U.S. Army Corps
    */
    if (! isEqual(_ageOfSnow, NODATA))
    {
        double snowAlbedo = std::min(0.9, 0.74 * pow(_ageOfSnow , -0.191));
        snowAlbedo = std::max(snowAlbedo, SNOW_MIN_ALBEDO);

        // thin snow: weighted with soil albedo
        albedo = snowAlbedo * oldSnowFraction + snowParameters.soilAlbedo * (1 - oldSnowFraction);
    }
    else
        albedo = snowParameters.soilAlbedo;

    /*! \brief Incoming Energy Fluxes */

    // pag. 52 (3.22) considerando i 2 contributi invece che solo uno
    // the internal energy is referred to 0 °C:
    // rain on snow is retained or leaves the pack at 0 °C, rain on bare soil leaves at surface temperature
    double rainRefTemp = (previousSWE > EPSILON) ? 0. : prevSurfaceTemp;
    QPrecipW = (HEAT_CAPACITY_WATER / 1000.) * (_precRain / 1000.) * (std::max(0., _airT) - rainRefTemp);
    // snowfall remains in the pack
    QPrecipS = (HEAT_CAPACITY_SNOW / 1000.) * (_precSnow / 1000.) * std::min(0., _airT);
    QPrecip = QPrecipW + QPrecipS;

    // temperatura dell'acqua superficiale: almeno 1 grado
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
        freeWaterFlux = (freeWaterFlux / avgExchangeArea) / HOUR_SECONDS

        if (freeWaterFlux > 0.01)
        {
            avgMass = (surfaceWaterContent / 1000.0) * WATER_DENSITY; //[kg/m2]
            QWaterKinetic = 0.5 * avgMass * (freeWaterFlux * freeWaterFlux) / 1000.0; //[kJ/m2]
        }
    }
    */

    // pag. 50 (3.14)
    QSolar = (1. - albedo) * (solarRadTot * HOUR_SECONDS) / 1000.;

    double surfaceEmissivity = SNOW_EMISSIVITY * oldSnowFraction + SOIL_EMISSIVITY * (1 - oldSnowFraction);    //  [-]

    // pag. 50 (3.15)
    // absorbed atmospheric radiation (absorptivity = emissivity) - emitted radiation
    QLongWave = double(STEFAN_BOLTZMANN * 3.6 * surfaceEmissivity * (longWaveAtmEmissivity * pow(_airT + ZEROCELSIUS, 4.0)
              - pow (prevSurfaceTemp + ZEROCELSIUS, 4.0)));

    // sensible heat pag. 50 (3.17)
    QTempGradient = HOUR_SECONDS * (HEAT_CAPACITY_AIR / 1000.) * (_airT - prevSurfaceTemp) / aerodynamicResistance;

    // latent heat pag. 51 (3.19)
    // FT tolta WATER_DENSITY dall'eq. (non corrispondevano le unità di misura)
    // snow: sublimation (vaporization + fusion), soil: vaporization
    double latentHeatVapor = LATENT_HEAT_VAPORIZATION_KJ;                   // [kJ kg-1]
    if (previousSWE > EPSILON)
        latentHeatVapor += LATENT_HEAT_FUSION_KJ;

    QVaporGradient = HOUR_SECONDS * latentHeatVapor * (airActualVapDensity - waterActualVapDensity) / aerodynamicResistance;

    // TODO serve formula diversa quando non c'è neve
    if (previousSWE < EPSILON)
    {
        QVaporGradient *= 0.4;
    }

    /*! Condensation (positive) or evaporation (negative) */
    sublimation = 0;                        // [mm]
    _evaporation = 0;                       // [mm]
    if (previousSWE > EPSILON)
    {
        // pag. 51 (3.21) [mm]
        sublimation = QVaporGradient / latentHeatVapor;

        if (sublimation < 0)
        {
            /*! Evaporation [mm]
             *  controllo aggiunto (può evaporare solo la neve presente)
             */
            double maxEvaporation = previousSWE + _precSnow;
            if (fabs(sublimation) > maxEvaporation)
            {
                sublimation = -maxEvaporation;
                // the energy flux is consistent with the mass
                QVaporGradient = sublimation * latentHeatVapor;
            }
            _evaporation = -sublimation;
        }
    }

    /*! Energy Balance */
    QTotal = QSolar + QPrecip + QLongWave + QTempGradient + QVaporGradient + QWaterHeat + QWaterKinetic;
    _sensibleHeat = QTempGradient;
    _latentHeat = QVaporGradient;

    /*! refreeze or melt */
    double freeze_melt = 0;      // [mm] freeze (positive) or melt (negative)

    /*! net amount of liquid water that freezes (heat added to the snow pack)
     *  and ice that melts (heat removed from the snow pack)
     *  pag.53 (3.25)
    */
    double w = (prevInternalEnergy + QTotal) / (LATENT_HEAT_FUSION_KJ * WATER_DENSITY);    // [m]
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

    /*! latent heat exchange in the snow pack [kJ m-2]
     *  pag.53 (3.23) modificata (errore nel denominatore)
     */
    double Qr = (freeze_melt / 1000.) * LATENT_HEAT_FUSION_KJ * WATER_DENSITY;

    /*! Internal energy [kJ m-2] */
    _internalEnergy = prevInternalEnergy + QTotal + Qr;

    /*! Snow Pack Mass */

    /*! Ice content mass balance: independent from the energy threshold */
    _iceContent = prevIceContent + _precSnow + sublimation + freeze_melt;
    _iceContent = std::max(_iceContent, 0.);

    double waterHoldingCapacity = snowParameters.snowWaterHoldingCapacity
                                  / (1 - snowParameters.snowWaterHoldingCapacity);      // [-]

    /*! Liquid water content */
    if (_iceContent < EPSILON)
    {
        // no ice left: the pack cannot retain liquid water
        _iceContent = 0;
        _liquidWaterContent = 0;
    }
    else
    {
        _liquidWaterContent = prevLWaterContent + _precRain + _surfaceWaterContent - freeze_melt;    // [mm]
        _liquidWaterContent = std::max(_liquidWaterContent, 0.);
        _liquidWaterContent = std::min(_liquidWaterContent, _iceContent * waterHoldingCapacity);
    }

    /*! Snow water equivalent */
    _snowWaterEquivalent = _iceContent + _liquidWaterContent;
    _deltaSWE = _snowWaterEquivalent - previousSWE;

    /*! snow density and snow depth */
    computeSnowDensity();

    double snowDepth = 0;
    if (! isEqual(_snowDensity, NODATA))
        snowDepth = _snowWaterEquivalent / _snowDensity;        // [m] (1 mm SWE = 1 kg m-2)

    /*! water not retained by the pack [mm] (closes the mass balance) */
    _waterOutflow = previousSWE + _precSnow + _precRain + _surfaceWaterContent + sublimation - _snowWaterEquivalent;
    _waterOutflow = std::max(_waterOutflow, 0.);

    _snowMelt = std::max(_waterOutflow - _surfaceWaterContent - _precRain, 0.);

    /*! surface energy [kJ m-2] and surface temperature [°C] */
    double surfaceEnergySnow;
    // snow
    if (_snowWaterEquivalent > 0 && fabs(_internalEnergy) < EPSILON)
    {
        surfaceEnergySnow = 0.;
    }
    else
    {
        double snowRatio = std::min(snowDepth, snowParameters.skinThickness) / snowParameters.snowSurfaceDampingDepth;
        surfaceEnergySnow = std::min(0., prevSurfaceEnergy + (QTotal + Qr) * snowRatio);
    }
    double surfaceTempSnow = surfaceEnergySnow / (WATER_DENSITY * SNOW_SPECIFIC_HEAT * snowParameters.skinThickness);

    // warning: corretto da claude
    double heatCapacity = DEFAULT_BULK_DENSITY * SOIL_SPECIFIC_HEAT * SOIL_DAMPING_DEPTH
                          + WATER_DENSITY * SNOW_SPECIFIC_HEAT * _snowWaterEquivalent * 0.001;      // [kJ m-2 °C-1]
    double surfaceTempSoil = _internalEnergy / heatCapacity;
    double surfaceEnergySoil = computeSurfaceEnergySoil(surfaceTempSoil, snowParameters.skinThickness);

    /* old code
    double surfaceEnergySoil = prevSurfaceEnergy + (QTotal + Qr) * (snowParameters.skinThickness / SOIL_DAMPING_DEPTH);
    double surfaceTempSoil = surfaceEnergySoil / (DEFAULT_BULK_DENSITY * SOIL_SPECIFIC_HEAT * snowParameters.skinThickness);
    */

    double snowFraction = std::min(snowDepth, snowParameters.skinThickness) / snowParameters.skinThickness;

    _surfaceEnergy = (surfaceEnergySnow * snowFraction) + surfaceEnergySoil * (1 - snowFraction);
    _surfaceTemp = (surfaceTempSnow * snowFraction) + surfaceTempSoil * (1 - snowFraction);

    /*! _ageOfSnow [days] */
    if (_snowWaterEquivalent > EPSILON)
    {
        if (_ageOfSnow == NODATA || _precSnow > 0.1)
        {
            // new snowfall
            _ageOfSnow = 0;
        }
        else
        {
            // no snowfall, increase age
            double oneHour = HOUR_SECONDS / DAY_SECONDS;       // day fraction
            _ageOfSnow += oneHour;
        }
    }
    else
    {
        _ageOfSnow = NODATA;
    }
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


/*!
 * \brief computeInternalEnergy
 * \param swe [mm]
 * \param bulkDensity [kg m-3]
 */
double computeInternalEnergy(double soilTemperature, int bulkDensity, double swe)
{
    return soilTemperature * ( WATER_DENSITY * SNOW_SPECIFIC_HEAT * (swe * 0.001)
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

