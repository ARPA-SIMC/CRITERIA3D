/*!
    \copyright 2010-2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/

#include <algorithm>
#include <math.h>

#include "commonConstants.h"
#include "meteoPoint.h"
#include "snow.h"


// TODO ora le costanti in commonConstant sono in [J m-3 K-1]
// controllare se tutte hanno divisione per 1000 per trasformare in kJ

Crit3DSnowParameters::Crit3DSnowParameters()
{
    initialize();
}


void Crit3DSnowParameters::initialize()
{
    // default values
    snowSkinThickness = 0.02;            /*!<  [m] */ // LC: VARIE VERSIONI IN BROOKS: 3mm (nel testo), 2-3cm (nel codice)
    soilAlbedo = 0.2;                    /*!<  [-] bare soil - 20% */
    snowVegetationHeight = 1;
    snowWaterHoldingCapacity = 0.05;
    snowMaxWaterContent = 0.1;
    tempMaxWithSnow = 2;
    tempMinWithRain = -0.5;
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
    _snowFall = NODATA;
    _snowMelt = NODATA;
    _snowWaterEquivalent = NODATA;
    _iceContent = NODATA;
    _liquidWaterContent = NODATA;
    _internalEnergy = NODATA;
    _surfaceInternalEnergy = NODATA;
    _snowSurfaceTemp = NODATA;
    _ageOfSnow = NODATA;
    _evaporation = NODATA;
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
        || int(_snowSurfaceTemp) == int(NODATA) )
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
        if (_airT < snowParameters.tempMinWithRain)
        {
            liquidWater = 0;
        }
         else if (_airT < snowParameters.tempMaxWithSnow)
        {
            liquidWater *= (_airT - snowParameters.tempMinWithRain) / (snowParameters.tempMaxWithSnow - snowParameters.tempMinWithRain);
        }
     }

    _snowFall = _prec - liquidWater;
    _prec = liquidWater;
}


void Crit3DSnow::computeSnowBrooksModel()
{
    double solarRadTot;
    double cloudCover;                           /*!<   [-]        */
    double prevIceContent, prevLWaterContent;    /*!<   [mm]       */
    double currentRatio;

    double AirActualVapDensity;                  /*!<   [kg/m^3]   */
    double WaterActualVapDensity;                /*!<   [kg/m^3]   */
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

    double bulk_density;                         /*!<   [kg/m^3]   */
    double EvapCond;                             /*!<   [mm]       */
    double refreeze;                             /*!<   [mm]       */

    double aerodynamicResistance;
    double myEmissivity;

    // gestione specchi d'acqua
    bool isWater = false;
    if ((_surfaceWaterContent / 1000) > snowParameters.snowMaxWaterContent )     /*!<  [m]  */
            isWater = true;

    if (isWater || (! checkValidPoint()))
    {
        _snowMelt = NODATA;
        _iceContent = NODATA;
        _liquidWaterContent = NODATA;
        _snowWaterEquivalent = NODATA;
        _surfaceInternalEnergy = NODATA;
        _snowSurfaceTemp = NODATA;
        _ageOfSnow = NODATA;
        return;
    }

    computeSnowFall();

    double dewPoint = tDewFromRelHum(_airRH, _airT);     /*!< [°C] */

    if (int(_transmissivity) != int(NODATA))
        cloudCover = 1 - std::min(double(_transmissivity) / _clearSkyTransmissivity, 1.);
    else
        cloudCover = 0.1;

    // ombreggiamento per vegetazione (4m sopra manto nevoso: ombreggiamento completo)
    // TODO migliorare - aggiungere LAI se disponibile
    double maxSnowDensity = 10;          // 1 mm snow = 1 cm water
    double maxVegetationHeight = 4;      // [m]
    double vegetationShadowing;          // [-]
    double maxSnowHeight = _snowWaterEquivalent * maxSnowDensity / 1000;                 // [m]
    double heightVegetation = snowParameters.snowVegetationHeight - maxSnowHeight;       // [m]
    vegetationShadowing = std::max(std::min(heightVegetation / maxVegetationHeight, 1.), 0.);
    solarRadTot = _globalRadiation - _beamRadiation * vegetationShadowing;

    double prevSurfacetemp = _snowSurfaceTemp;
    double previousSWE = _snowWaterEquivalent;
    double prevInternalEnergy = _internalEnergy;
    double prevSurfaceIntEnergy = _surfaceInternalEnergy;

    /*--------------------------------------------------------------------
    // COERENZA
    // controlli di coerenza per eventuali modifiche manuale su mappa SWE
    // -------------------------------------------------------------------*/
    if (previousSWE > 0)
    {
        prevIceContent = _iceContent;
        prevLWaterContent = _liquidWaterContent;

        if ( (prevIceContent <= 0) && (prevLWaterContent <= 0) )
        {
            // neve aggiunta
            prevIceContent = previousSWE;

            // Pag. 53 formula 3.23
            prevInternalEnergy = -(previousSWE / 1000) * LATENT_HEAT_FUSION * WATER_DENSITY;

            // stato: neve recente prossima alla fusione, con una settimana di età
            _ageOfSnow = 7;
            prevSurfacetemp = std::min(prevSurfacetemp, -0.1);
            prevSurfaceIntEnergy = std::min(prevSurfaceIntEnergy, -0.1);
        }

        /*! check on sum */
        currentRatio = previousSWE / (prevIceContent + prevLWaterContent);
        if (fabs(currentRatio - 1) > 0.001)
        {
            prevIceContent = prevIceContent * currentRatio;
            prevLWaterContent = prevLWaterContent * currentRatio;
        }
    }
    else
    {
        prevIceContent = 0;
        prevLWaterContent = 0;
        _ageOfSnow = 0;
    }

    /*! \brief Vapor Density and Roughness Calculations */

    // brooks originale
    if ( previousSWE > SNOW_MINIMUM_HEIGHT)
        aerodynamicResistance = aerodynamicResistanceCampbell77(true, 10, _windInt, snowParameters.snowVegetationHeight);
    else
        aerodynamicResistance = aerodynamicResistanceCampbell77(false, 10, _windInt, snowParameters.snowVegetationHeight);

    // ok pag.52 (3.20)
    // source: Jensen et al. (1990) and Tetens (1930)
    // saturated vapor density
    AirActualVapDensity = double(exp((16.78 * dewPoint - 116.9) / (dewPoint + 237.3))
                              / ((ZEROCELSIUS + dewPoint) * THERMO_WATER_VAPOR) );

    // over water
    WaterActualVapDensity = double( exp((16.78 * prevSurfacetemp - 116.9) / (prevSurfacetemp + 237.3))
                                / ((ZEROCELSIUS + prevSurfacetemp) * THERMO_WATER_VAPOR) );

    // over ice
    // LC: controllare
    // non trovo riferimenti a questa formula
    /*if (prevInternalEnergy <= 0)
    {
        WaterActualVapDensity *= exp(MH2O * LATENT_HEAT_FUSION * prevSurfacetemp * 1000
                                 / (R_GAS * pow(prevSurfacetemp + ZEROCELSIUS, 2)));
    }*/

    /*!
    * \brief Atmospheric Emissivity Calculations for Longwave Radiation
    *-----------------------------------------------------------
    * Unsworth, M.H. and L.J. Monteith. 1975. Long-wave radiation a the ground. I. Angular distribution of incoming radiation. Quarterly Journal of the Royal Meteorological Society 101(427):13-24.
    */

    longWaveAtmEmissivity = (0.72 + 0.005 * _airT) * (1.0 - 0.84 * cloudCover) + 0.84 * cloudCover;

    /*! Age of snow & albedo */

    if ( _snowFall > 0 && _prec <= 0)
        _ageOfSnow = 1. / 24.;
    else if (previousSWE > 0)
        _ageOfSnow = _ageOfSnow + 1. / 24.;
    else
        _ageOfSnow = 0;

    if ( (previousSWE > 0) || (_snowFall > 0 && _prec <= 0))
        /*! O'NEILL, A.D.J. GRAY D.M.1973. Spatial and temporal variations of the albedo of prairie snowpacks. The Role of Snow and Ice in Hydrology: Proceedings of the Banff Syn~posia, 1972. Unesc - WMO -IAHS, Geneva -Budapest-Paris, Vol. 1,  pp. 176-186
        * arrotondato da U.S. Army Corps
        */
        albedo = std::min(0.9, 0.74 * pow ( _ageOfSnow , -0.19));
    else
        albedo = snowParameters.soilAlbedo;

    /*! \brief Incoming Energy Fluxes */

    // pag. 52 (3.22) considerando i 2 contributi invece che solo uno
    QPrecipW = (HEAT_CAPACITY_WATER / 1000.) * (_prec / 1000.) * (std::max(0., _airT) - prevSurfacetemp);
    QPrecipS = (HEAT_CAPACITY_SNOW / 1000.) * (_snowFall / 1000.) * (std::min(0., _airT) - prevSurfacetemp);
    QPrecip = QPrecipW + QPrecipS;

    // temperatura dell'acqua: almeno 1 grado
    QWaterHeat = (HEAT_CAPACITY_WATER / 1000.) * (_surfaceWaterContent / 1000.) * (std::max(1., (prevSurfacetemp + _airT) / 2.) - prevSurfacetemp);

    // energia acqua libera
    QWaterKinetic = 0;

    // TO DO free water flux
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

    if (previousSWE > SNOW_MINIMUM_HEIGHT)
        myEmissivity = double(SNOW_EMISSIVITY);
    else
        myEmissivity = double(SOIL_EMISSIVITY);

    // pag. 50 (3.15)
    QLongWave = double(STEFAN_BOLTZMANN * 3.6 * (longWaveAtmEmissivity * pow((_airT + ZEROCELSIUS), 4.0)
              - myEmissivity * pow ((prevSurfacetemp + ZEROCELSIUS), 4.0)));

    // pag. 50 (3.17)
    QTempGradient = (3600. * (HEAT_CAPACITY_AIR / 1000.) * (_airT - prevSurfacetemp)) / aerodynamicResistance;

    // FT calcolare solo se c'e' manto nevoso
    if (previousSWE > SNOW_MINIMUM_HEIGHT)
    {
        // LC: pag. 51 (eq. 3.19)
        // FT: tolta WATER_DENSITY dall'eq. (non corrispondevano le unità di misura)
        QVaporGradient = 3600. * (LATENT_HEAT_VAPORIZATION + LATENT_HEAT_FUSION) * (AirActualVapDensity - WaterActualVapDensity) / aerodynamicResistance;
    }
    else
    {
        QVaporGradient = 0;
    }

    /*! \brief Energy Balance */
    QTotal = QSolar + QPrecip + QLongWave + QTempGradient + QVaporGradient + QWaterHeat + QWaterKinetic;

    /*! \brief Evaporation/Condensation */
    if (previousSWE > SNOW_MINIMUM_HEIGHT)
    {
        // pag 51 (3.21)
        EvapCond = QVaporGradient / ((LATENT_HEAT_FUSION + LATENT_HEAT_VAPORIZATION) * WATER_DENSITY) * 1000.;
        if (EvapCond < 0)
        {
            //FT: controllo aggiunto: può evaporare al massimo la neve presente
            EvapCond = -std::min(previousSWE + _snowFall, -EvapCond);
        }
    }
    else
    {
        EvapCond = 0;
    }

    /*! sign of evaporation is negative */
    if (EvapCond < 0)
        _evaporation = -EvapCond;
    else
        _evaporation = 0;

    /*! \brief refreeze */
    if (previousSWE > SNOW_MINIMUM_HEIGHT)
    {
        // pag. 53 (3.25) (3.26) (3.24)
        double wFreeze = std::min((_prec + prevLWaterContent), std::max(0., -1000. / (LATENT_HEAT_FUSION * WATER_DENSITY) * (prevInternalEnergy + QTotal)));
        double wThaw =  std::min((_snowFall + prevIceContent + EvapCond), std::max(0., 1000. / (LATENT_HEAT_FUSION * WATER_DENSITY) * (prevInternalEnergy + QTotal)));
        refreeze = wFreeze - wThaw;
    }
    else
    {
        refreeze = 0;
    }

    /*! Internal energy */
    _internalEnergy = prevInternalEnergy + QTotal + (refreeze / 1000.) * LATENT_HEAT_FUSION * WATER_DENSITY;
    // FT: aggiunto per stabilità numerica
    if (abs(_internalEnergy) < 0.001)
    {
        _internalEnergy = 0.;
    }

    /*! \brief Snow Pack Mass */

    double waterHoldingCapacity = snowParameters.snowWaterHoldingCapacity / (1 - snowParameters.snowWaterHoldingCapacity); // [%]

    /*! Ice content */
    if (_internalEnergy > 0.)
    {
        _iceContent = 0;
    }
    else
    {
        _iceContent = std::max(prevIceContent + _snowFall + refreeze + EvapCond, 0.);
    }

    /*! Liquid water content */
    if (_internalEnergy == 0.)
    {
        _liquidWaterContent = std::min(waterHoldingCapacity * _iceContent, prevLWaterContent + _prec + _surfaceWaterContent - refreeze);
        _liquidWaterContent = std::max(0., _liquidWaterContent);
    }
    else
    {
        _liquidWaterContent = 0;
    }

    /*! Snow water equivalent */
    _snowWaterEquivalent = _iceContent + _liquidWaterContent;

    /*! Snowmelt (or refreeze) - source/sink for Criteria3D */
    _snowMelt = previousSWE + _snowFall + EvapCond - _snowWaterEquivalent;

    /*! Snow surface energy */
    if (_internalEnergy == 0.)
    {
        _surfaceInternalEnergy = 0.;
    }
    else
    {
        if (_snowWaterEquivalent > 0.)
            _surfaceInternalEnergy = std::min(0., prevSurfaceIntEnergy + (QTotal + (refreeze / 1000.) * LATENT_HEAT_FUSION * WATER_DENSITY)
                                    * (std::min(_snowWaterEquivalent / 1000., snowParameters.snowSkinThickness) / SNOW_DAMPING_DEPTH
                                       + std::max(snowParameters.snowSkinThickness - (_snowWaterEquivalent / 1000.), 0.) / SOIL_DAMPING_DEPTH));
        else
            _surfaceInternalEnergy = prevSurfaceIntEnergy + (QTotal + (refreeze / 1000.) * LATENT_HEAT_FUSION * WATER_DENSITY) * (snowParameters.snowSkinThickness / SOIL_DAMPING_DEPTH);
    }

    // TODO passare bulk density
    bulk_density = DEFAULT_BULK_DENSITY;

    _snowSurfaceTemp = _surfaceInternalEnergy / ((HEAT_CAPACITY_SNOW / 1000.) * std::min(_snowWaterEquivalent / 1000., snowParameters.snowSkinThickness)
                   + SOIL_SPECIFIC_HEAT * std::max(0., snowParameters.snowSkinThickness - _snowWaterEquivalent / 1000.) * bulk_density);

}


double Crit3DSnow::getSnowFall()
{
    return _snowFall;
}

double Crit3DSnow::getSnowMelt()
{
    return _snowMelt;
}

double Crit3DSnow::getSnowWaterEquivalent()
{
    return _snowWaterEquivalent;
}
void Crit3DSnow::setSnowWaterEquivalent(float value)
{
    _snowWaterEquivalent = value;
}

double Crit3DSnow::getIceContent()
{
    return _iceContent;
}
void Crit3DSnow::setIceContent(float value)
{
    _iceContent = value;
}

double Crit3DSnow::getLiquidWaterContent()
{
    return _liquidWaterContent;
}

void Crit3DSnow::setLiquidWaterContent(float value)
{
    _liquidWaterContent = value;
}

double Crit3DSnow::getInternalEnergy()
{
    return _internalEnergy;
}

void Crit3DSnow::setInternalEnergy(float value)
{
    _internalEnergy = value;
}

double Crit3DSnow::getSurfaceInternalEnergy()
{
    return _surfaceInternalEnergy;
}

void Crit3DSnow::setSurfaceInternalEnergy(float value)
{
    _surfaceInternalEnergy = value;
}

double Crit3DSnow::getSnowSurfaceTemp()
{
    return _snowSurfaceTemp;
}

void Crit3DSnow::setSnowSurfaceTemp(float value)
{
    _snowSurfaceTemp = value;
}

double Crit3DSnow::getAgeOfSnow()
{
    return _ageOfSnow;
}

void Crit3DSnow::setAgeOfSnow(float value)
{
    _ageOfSnow = value;
}



/*!
 * \brief Computes aerodynamic Resistance
 * \param isSnow
 * \param zRefWind [m] measurement height for wind
 * \param myWindSpeed [m * s-1] wind speed measured at reference height
 * \param vegetativeHeight [m] height of the vegetative
 * \return result
 */
double aerodynamicResistanceCampbell77(bool isSnow , double zRefWind, double myWindSpeed, double vegetativeHeight)
{

    double zeroPlane;            /*!  [m] zero-plane displacement (snow = 0m, vegetative cover d = 0.64 times the height of the vegetative) */
    double momentumRoughness;    /*!  [m] momentum roughness parameter (for snow = 0.001m, for vegetative cover zm = 0.13 times the height of the vegetation) */
    double log2;                 // equivalente a vegetativeHeight = 1; log2 = (Zt - d + Zh)/Zh , Zt: measurement height for temperature  sempre pari a 2m (i sensori di temperatura sono piazzati a 2metri) */
    double log1;

    /*! check on wind speed [m/s] */
    myWindSpeed = std::max(myWindSpeed, 0.2);

    if (isSnow)
    {
        zeroPlane = 0;
        momentumRoughness = 0.001;
        log2 = 9.2;                 // equivalent to vegetativeHeight = 1
    }
    else
    {
        /*! check on vegetativeHeight  [m] */
        vegetativeHeight = std::max(vegetativeHeight, 0.1);

        //pag 51: the height of the zero-plane displacement
        zeroPlane = 0.64 * vegetativeHeight;

        momentumRoughness = 0.13 * vegetativeHeight;
        log2 = 4;                   // equivalent to vegetativeHeight = 1
    }

    if (zeroPlane > zRefWind)
        zeroPlane = zRefWind;

    // formula 3.18 pag 51
    log1 = log((zRefWind - zeroPlane + momentumRoughness) / momentumRoughness);

    return log1 * log2 / (VON_KARMAN_CONST * VON_KARMAN_CONST * myWindSpeed);
}


// LC: InternalEnergyMap pag. 54 formula 3.29  initSoilPackTemp sarebbe da chiamare initSnowPackTemp ????
double computeInternalEnergy(double initSoilPackTemp,int bulkDensity, double initSWE)
{
    return initSoilPackTemp * (HEAT_CAPACITY_SNOW / 1000. * initSWE + bulkDensity * SNOW_DAMPING_DEPTH * SOIL_SPECIFIC_HEAT);
}


// LC: è la formula 3.27 a pag. 54 in cui ha diviso la surface come la somma dei contributi della parte "water" e di quella "soil"
double computeSurfaceInternalEnergy(double initSnowSurfaceTemp,int bulkDensity, double initSWE, double snowSkinThickness)
{
    return initSnowSurfaceTemp * (HEAT_CAPACITY_SNOW / 1000. * std::min(initSWE, snowSkinThickness)
                                  + SOIL_SPECIFIC_HEAT * std::max(0., snowSkinThickness - initSWE) * bulkDensity);
}

