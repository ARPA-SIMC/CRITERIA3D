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

#include "commonConstants.h"
#include "snowPoint.h"
#include "meteoPoint.h"
#include "math.h"


// TODO ora le costanti in commonConstant sono in [J m-3 K-1] (controllare se tutte hanno divisione per 1000 per trasformare in kJ)

Crit3DSnowPoint::Crit3DSnowPoint(struct TradPoint* radpoint, float temp, float prec, float relHum, float windInt, float clearSkyTransmissivity)
{
    _snowFall = NODATA;
    _snowMelt = NODATA;
    _snowWaterEquivalent = NODATA;
    _iceContent = NODATA;
    _lWContent = NODATA;
    _internalEnergy = NODATA;
    _surfaceInternalEnergy = NODATA;
    _snowSurfaceTemp = NODATA;
    _ageOfSnow = NODATA;

    _waterContent = NODATA;
    _evaporation = NODATA;

    _radpoint = radpoint;
    _clearSkyTransmissivity = clearSkyTransmissivity;
    _airT = temp;
    _prec = prec;
    _airRH = relHum;
    _windInt = windInt;

    _parameters->snowSkinThickness = 0.02f;            /*!<  [m] */ // LC: VARIE VERSIONI IN BROOKS: 3mm (nel testo), 2-3cm (nel codice)
    _parameters->soilAlbedo = 0.2f;                    /*!<  [-] bare soil - 20% */
    _parameters->snowVegetationHeight = 1;
    _parameters->snowWaterHoldingCapacity = 0.05f;
    _parameters->snowMaxWaterContent = 0.1f;
    _parameters->tempMaxWithSnow = 2;                  //  Valore originale (tesi Brooks): 0.5 gradi due anni, 3 gradi un anno (problema inv. termica?)
    _parameters->tempMinWithRain = -0.5f;

}


bool Crit3DSnowPoint::checkValidPoint()
{
    if (int(_radpoint->global) != int(NODATA) && int(_radpoint->beam) != int(NODATA)
            && int(_snowFall) != int(NODATA) && int(_snowWaterEquivalent) != int(NODATA)
            && int(_snowSurfaceTemp) != int(NODATA))
        return true;
    else
        return false;
}


void Crit3DSnowPoint::computeSnowFall()
{
    // mettere nella funzione che la chiama prima il controllo che i 2 valori siano diversi da NODATA, se uno dei 2 è un NODATA, allora SnowFallMap.Value(row, col) = SnowFallMap.header.flag
    float liquidWater = _prec;
    if (liquidWater > 0)
    {
        if (_airT < _parameters->tempMinWithRain)
            liquidWater = 0;
         else if (_airT < _parameters->tempMaxWithSnow)
            liquidWater *= (_airT - _parameters->tempMinWithRain) / (_parameters->tempMaxWithSnow - _parameters->tempMinWithRain);
     }

    _snowFall = _prec - liquidWater;
    _prec = liquidWater;
}


void Crit3DSnowPoint::computeSnowBrooksModel()
{
    float globalRadiation, beamRadiation, solarRadTot;
    float cloudCover;                           /*!<   [-]        */
    float prevIceContent, prevLWaterContent;    /*!<   [mm]       */
    float currentRatio;

    float AirActualVapDensity;                  /*!<   [kg/m^3]   */
    float WaterActualVapDensity;                /*!<   [kg/m^3]   */
    float longWaveAtmEmissivity;                /*!<   [-]        */
    float albedo;                               /*!<   [-]        */

    float QSolar;                               /*!<   [kJ/m^2]   integrale della radiazione solare */
    float QPrecip;                              /*!<   [kJ/m^2]   avvezione (trasferimento di calore dalla precipitazione) */
    float QPrecipW;                             /*!<   [kJ/m^2]   */
    float QPrecipS;                             /*!<   [kJ/m^2]   */
    float QLongWave;                            /*!<   [kJ/m^2]   emissione di radiazione onda lunga (eq. Stefan Boltzmann) */
    float QTempGradient;                        /*!<   [kJ/m^2]   scambio di calore sensibile (dovuto al gradiente di temperatura) */
    float QVaporGradient;                       /*!<   [kJ/m^2]   scambio di calore latente (vapore) */
    float QTotal;                               /*!<   [kJ/m^2]   */
    float QWaterHeat;                           /*!<   [kJ/m^2]   */
    float QWaterKinetic;                        /*!<   [kJ/m^2]   */

    float bulk_density;                         /*!<   [kg/m^3]   */
    float EvapCond;                             /*!<   [mm]       */
    float refreeze;                             /*!<   [mm]       */

    float aerodynamicResistance;
    float myEmissivity;

    // gestione specchi d'acqua
     bool isWater = false;

     //if criteria3DModule.flgCriteria3D_Ready Then
     float surfaceWaterContent = MAXVALUE( _waterContent, 0.0f );                                   /*!<  [mm] */
     if ( (surfaceWaterContent / 1000) > _parameters->snowMaxWaterContent )                         /*!<  [m]  */
        isWater = true;


     if ((!isWater) && (checkValidPoint()))
     {
        float dewPoint = tDewFromRelHum(_airRH, _airT);     /*!< [°C] */

        if (int(_radpoint->transmissivity) != int(NODATA))
            cloudCover = 1 - MINVALUE(float(_radpoint->transmissivity) / _clearSkyTransmissivity, 1.0f);
        else
            cloudCover = 0.1f;

        globalRadiation = float(_radpoint->global);
        beamRadiation = float(_radpoint->beam);

        // ombreggiamento per vegetazione (4m sopra manto nevoso: ombreggiamento completo)
        // TODO migliorare - aggiungere LAI se disponibile
        float maxSnowDensity = 10;          // 1 mm snow = 1 cm water
        float maxVegetationHeight = 4;      // [m]
        float vegetationShadowing;          // [-]
        float maxSnowHeight = _snowWaterEquivalent * maxSnowDensity / 1000;                 // [m]
        float heightVegetation = _parameters->snowVegetationHeight - maxSnowHeight;         // [m]
        vegetationShadowing = MAXVALUE(MINVALUE(heightVegetation / maxVegetationHeight, 1.0f), 0.0f);
        solarRadTot = globalRadiation - beamRadiation * vegetationShadowing;

        float prevSurfacetemp = _snowSurfaceTemp;
        float previousSWE = _snowWaterEquivalent;
        float prevInternalEnergy = _internalEnergy;
        float prevSurfaceIntEnergy = _surfaceInternalEnergy;

        /*--------------------------------------------------------------------
        // COERENZA
        // controlli di coerenza per eventuali modifiche manuale su mappa SWE
        // -------------------------------------------------------------------*/
        if (prevSurfacetemp < -30)
            prevSurfacetemp = -30;

        if (previousSWE <= 0)
        {
            prevIceContent = 0;
            prevLWaterContent = 0;
            _ageOfSnow = 0;
        }
        else
        {
            prevIceContent = _iceContent;
            prevLWaterContent = _lWContent;

            if ( (prevIceContent <= 0) && (prevLWaterContent <= 0) )
            {
                // neve aggiunta
                prevIceContent = previousSWE;

                // Pag. 53 formula 3.23
                prevInternalEnergy = -(previousSWE / 1000) * LATENT_HEAT_FUSION * WATER_DENSITY;

                // stato: neve recente prossima alla fusione, con una settimana di età
                _ageOfSnow = 7;
                prevSurfacetemp = MINVALUE(prevSurfacetemp, -0.1f);
                prevSurfaceIntEnergy = MINVALUE(prevSurfaceIntEnergy, -0.1f);
            }

            /*! check on sum */
            currentRatio = previousSWE / (prevIceContent + prevLWaterContent);
            if (fabs(currentRatio - 1) > 0.001f)
            {
                prevIceContent = prevIceContent * currentRatio;
                prevLWaterContent = prevLWaterContent * currentRatio;
            }

         }

         /*! \brief Vapor Density and Roughness Calculations */

         // brooks originale
         if ( previousSWE > SNOW_MINIMUM_HEIGHT)
            aerodynamicResistance = Crit3DSnowPoint::aerodynamicResistanceCampbell77(true, 10, _windInt, _parameters->snowVegetationHeight);
         else
            aerodynamicResistance = Crit3DSnowPoint::aerodynamicResistanceCampbell77(false, 10, _windInt, _parameters->snowVegetationHeight);

          //ok pag.52 (3.20)
          // source: Jensen et al. (1990) and Tetens (1930)
          // saturated vapor density
          AirActualVapDensity = float( exp((16.78f * dewPoint - 116.9f) / (dewPoint + 237.3f))
                                      / ((ZEROCELSIUS + dewPoint) * THERMO_WATER_VAPOR) );

          //ok
          //over water ( over snow?)
          WaterActualVapDensity = float( exp((16.78f * prevSurfacetemp - 116.9f) / (prevSurfacetemp + 237.3f))
                                        / ((ZEROCELSIUS + prevSurfacetemp) * THERMO_WATER_VAPOR) );

          //over ice
          // LC: controllare
          // non trovo riferimenti a questa formula
          // perchè amlcune sono define ed altre no?
          // 0.018 is [kg] vapor molar mass???
          if (prevInternalEnergy <= 0)
            WaterActualVapDensity = float(WaterActualVapDensity * exp(0.018f * LATENT_HEAT_FUSION * prevSurfacetemp * 1000.0f
                                                                / (R_GAS * pow((prevSurfacetemp + ZEROCELSIUS) , 2.0f))));


          /*!
          * \brief Atmospheric Emissivity Calculations for Longwave Radiation
          *-----------------------------------------------------------
          * Unsworth, M.H. and L.J. Monteith. 1975. Long-wave radiation a the ground. I. Angular distribution of incoming radiation. Quarterly Journal of the Royal Meteorological Society 101(427):13-24.
          */

          longWaveAtmEmissivity = (0.72f + 0.005f * _airT) * (1.0f - 0.84f * cloudCover) + 0.84f * cloudCover;

          /*! Age of snow & albedo */

          if ( _snowFall > 0 && _prec <= 0)
            _ageOfSnow = 1 / 24;
          else if (previousSWE > 0)
             _ageOfSnow = _ageOfSnow + 1 / 24;
          else
            _ageOfSnow = 0;

          if ( (previousSWE > 0) || (_snowFall > 0 && _prec <= 0))
                /*! O'NEILL, A.D.J. GRAY D.M.1973. Spatial and temporal variations of the albedo of prairie snowpacks. The Role of Snow and Ice in Hydrology: Proceedings of the Banff Syn~posia, 1972. Unesc - WMO -IAHS, Geneva -Budapest-Paris, Vol. 1,  pp. 176-186
                * arrotondato da U.S. Army Corps
                */
                albedo = MINVALUE(0.9f, 0.74f * pow ( _ageOfSnow , -0.19f));
          else
                albedo = _parameters->soilAlbedo;


          /*! \brief Incoming Energy Fluxes */
          // pag. 52 (3.22) considerando i 2 contributi invece che solo uno
          QPrecipW = (HEAT_CAPACITY_WATER / 1000.0f) * (_prec / 1000.0f) * (MAXVALUE(0.0f, _airT) - prevSurfacetemp);
          QPrecipS = (HEAT_CAPACITY_SNOW / 1000.0f) * (_snowFall / 1000.0f) * (MINVALUE(0.0f, _airT) - prevSurfacetemp);
          QPrecip = QPrecipW + QPrecipS;

          // energia acqua libera (TROY site test)
          QWaterHeat = 0;
          QWaterKinetic = 0;
          if (surfaceWaterContent > 0.1f)
          {
                //temperatura dell 'acqua: almeno 1 grado
                QWaterHeat = (HEAT_CAPACITY_WATER / 1000.0f) * (surfaceWaterContent / 1000.0f) * (std::max(1.0f, (prevSurfacetemp + _airT) / 2.0f) - prevSurfacetemp);

                //////////////////////////////////////////////////
                // TO DO
                /*
                float freeWaterFlux = 0;
                nodeIndex = GIS.GetValueFromXY(criteria3DModule.Crit3DIndexMap(0), x, y)
                        If nodeIndex <> criteria3DModule.Crit3DIndexMap(0).header.flag Then
                            //[m3/h]
                            freeWaterFlux = criteria3DModule.getLateralFlow(nodeIndex, False)
                            //[m2]
                            avgExchangeArea = Crit3DIndexMap(0).header.cellSize * (surfaceWaterContent / 1000.0)
                            //[m/s]
                            freeWaterFlux = (freeWaterFlux / avgExchangeArea) / 3600.0
                        End If

                        if (freeWaterFlux > 0.01)
                        {
                            avgMass = (surfaceWaterContent / 1000.0) * WATER_DENSITY; //[kg/m2]
                            QWaterKinetic = 0.5 * avgMass * (freeWaterFlux * freeWaterFlux) / 1000.0; //[kJ/m2]
                        }
                }
                */

                // pag. 50 (3.14)
                QSolar = (1 - albedo) * (solarRadTot * 3600) / 1000;

                if (previousSWE > SNOW_MINIMUM_HEIGHT)
                    myEmissivity = float(SNOW_EMISSIVITY);
                else
                    myEmissivity = float(SOIL_EMISSIVITY);

                // pag. 50 (3.15)
                QLongWave = float(STEFAN_BOLTZMANN * 3.6 * (longWaveAtmEmissivity * pow((_airT + ZEROCELSIUS), 4.0)
                          - myEmissivity * pow ((prevSurfacetemp + ZEROCELSIUS), 4.0)));

                // pag. 50 (3.17)
                QTempGradient = HEAT_CAPACITY_AIR / 1000 * (_airT - prevSurfacetemp) / (aerodynamicResistance / 3600);

                // FT calcolare solo se c'e' manto nevoso
                if (previousSWE > SNOW_MINIMUM_HEIGHT)
                {
                    // LC: pag. 51 (3.19)
                    // manca il fattore WATER_DENSITY ???
                    // dovrebbe essere:
                    // QVaporGradient = (LATENT_HEAT_VAPORIZATION + LATENT_HEAT_FUSION) * WATER_DENSITY *(AirActualVapDensity - WaterActualVapDensity) / (aerodynamicResistance / 3600);
                    // coerente però con il codice in appendice, inoltre le unità di misura tornano  NON mettendo il fattore WATER_DENSITY
                    QVaporGradient = (LATENT_HEAT_VAPORIZATION + LATENT_HEAT_FUSION) * (AirActualVapDensity - WaterActualVapDensity) / (aerodynamicResistance / 3600);
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
                    EvapCond = QVaporGradient / ((LATENT_HEAT_FUSION + LATENT_HEAT_VAPORIZATION) * WATER_DENSITY) * 1000;
                    if (EvapCond < 0)
                    {
                        //controllo aggiunto: può evaporare al massimo la neve presente
                        EvapCond = - std::min(previousSWE + _snowFall, -EvapCond);
                    }
                }
                else
                    EvapCond = 0;


                /*! sign of evaporation is negative */
                if (EvapCond < 0)
                    _evaporation = -EvapCond;
                else
                    _evaporation = 0;

                /*! refreeze */
                if (previousSWE > SNOW_MINIMUM_HEIGHT)
                {
                    // pag. 53 (3.25) (3.26) (3.24)
                    float wFreeze = std::min((_prec + prevLWaterContent), std::max(0.0f, -1000 / (LATENT_HEAT_FUSION * WATER_DENSITY) * (prevInternalEnergy + QTotal)));
                    float wThaw =  std::min((_snowFall + prevIceContent + EvapCond), std::max(0.0f, 1000 / (LATENT_HEAT_FUSION * WATER_DENSITY) * (prevInternalEnergy + QTotal)));
                    refreeze = wFreeze - wThaw;
                }
                else
                    refreeze = 0;


                /*! Internal energy */
                _internalEnergy = prevInternalEnergy + QTotal + (refreeze / 1000) * LATENT_HEAT_FUSION * WATER_DENSITY;

                /*! \brief Snow Pack Mass */

                /*! Ice content */
                if (_internalEnergy > 0.001f)
                    _iceContent = 0;
                else
                    _iceContent = std::max(prevIceContent + _snowFall + refreeze + EvapCond, 0.0f);


                float waterHoldingCapacity = _parameters->snowWaterHoldingCapacity / (1 - _parameters->snowWaterHoldingCapacity); //[%]

                /*! Liquid water content */
                if (fabs(_internalEnergy) < 0.001f)
                    _lWContent = std::min(waterHoldingCapacity * _iceContent, prevLWaterContent + _prec + surfaceWaterContent - refreeze);
                else
                    _lWContent = 0;

                if (_lWContent < 0)
                    _lWContent = 0;

                /*! Snow water equivalent */
                _snowWaterEquivalent = _iceContent + _lWContent;

                /*! Snowmelt (or refreeze) - source/sink for Criteria3D */
                _snowMelt = previousSWE + _snowFall + EvapCond - _snowWaterEquivalent;

                /*! Snow surface energy */
                if (fabs(_internalEnergy) < 0.001f)
                {
                    _surfaceInternalEnergy = 0;
                }
                else
                {
                    if (_snowWaterEquivalent > 0)
                        _surfaceInternalEnergy = MINVALUE(0, prevSurfaceIntEnergy + (QTotal + (refreeze / 1000) * LATENT_HEAT_FUSION * WATER_DENSITY) * (std::min(_snowWaterEquivalent / 1000, _parameters->snowSkinThickness) / SNOW_DAMPING_DEPTH + std::max(_parameters->snowSkinThickness - (_snowWaterEquivalent / 1000), 0.0f) / SOIL_DAMPING_DEPTH));
                    else
                        _surfaceInternalEnergy = prevSurfaceIntEnergy + (QTotal + (refreeze / 1000) * LATENT_HEAT_FUSION * WATER_DENSITY) * (_parameters->snowSkinThickness / SOIL_DAMPING_DEPTH);
                }


                bulk_density = DEFAULT_BULK_DENSITY;
                /*
                    If MSoil.SoilMap.isLoaded Then
                        'bulk density superficiale (1 cm)
                        'da reperire dato reale di bulk_density, ora messo a valore di default
                         bulk_density = DEFAULT_BULK_DENSITY
                '                        If MSoil.readPointSoilHorizon(x, y, 1, soilIndex, nrHorizon, myHorizon) Then
                '                             If myHorizon.bulkDensity <> Definitions.NO_DATA Then
                '                                 bulk_density = myHorizon.bulkDensity * 1000         '[kg/m^3]
                '                             End If
                '                        End If
                    End If
                 */
                _snowSurfaceTemp = _surfaceInternalEnergy / ((HEAT_CAPACITY_SNOW / 1000) * MINVALUE(_snowWaterEquivalent / 1000, _parameters->snowSkinThickness)
                               + SOIL_SPECIFIC_HEAT * MAXVALUE(0.0f, _parameters->snowSkinThickness - _snowWaterEquivalent / 1000) * bulk_density);
           }
           else
           {
                //snowfall diventa snowmelt negli specchi d'acqua
                _snowMelt = _snowFall;
                _iceContent = NODATA;
                _lWContent = NODATA;
                _snowWaterEquivalent = NODATA;
                _surfaceInternalEnergy = NODATA;
                _snowSurfaceTemp = NODATA;
            }
        }

}


float Crit3DSnowPoint::getSnowFall()
{
    return _snowFall;
}

float Crit3DSnowPoint::getSnowMelt()
{
    return _snowMelt;
}

float Crit3DSnowPoint::getSnowWaterEquivalent()
{
    return _snowWaterEquivalent;
}

float Crit3DSnowPoint::getIceContent()
{
    return _iceContent;
}

float Crit3DSnowPoint::getLWContent()
{
    return _lWContent;
}

float Crit3DSnowPoint::getInternalEnergy()
{
    return _internalEnergy;
}

float Crit3DSnowPoint::getSurfaceInternalEnergy()
{
    return _surfaceInternalEnergy;
}

float Crit3DSnowPoint::getSnowSurfaceTemp()
{
    return _snowSurfaceTemp;
}

float Crit3DSnowPoint::getAgeOfSnow()
{
    return _ageOfSnow;
}

float Crit3DSnowPoint::getSnowSkinThickness()
{
    return _parameters->snowSkinThickness;
}

float Crit3DSnowPoint::getSoilAlbedo()
{
    return _parameters->soilAlbedo;
}

float Crit3DSnowPoint::getSnowVegetationHeight()
{
    return _parameters->snowVegetationHeight;
}

float Crit3DSnowPoint::getSnowWaterHoldingCapacity()
{
    return _parameters->snowWaterHoldingCapacity;
}

float Crit3DSnowPoint::getSnowMaxWaterContent()
{
    return _parameters->snowMaxWaterContent;
}

float Crit3DSnowPoint::getTempMaxWithSnow()
{
    return _parameters->tempMaxWithSnow;
}

float Crit3DSnowPoint::getTempMinWithRain()
{
    return _parameters->tempMinWithRain;
}


/*!
 * \brief Computes aerodynamic Resistance
 * \param isSnow
 * \param zRefWind [m] measurement height for wind
 * \param myWindSpeed [m * s-1] wind speed measured at reference height
 * \param vegetativeHeight [m] height of the vegetative
 * \return result
 */
float Crit3DSnowPoint::aerodynamicResistanceCampbell77(bool isSnow , float zRefWind, float myWindSpeed, float vegetativeHeight)
{

    float zeroPlane;            /*!  [m] zero-plane displacement (snow = 0m, vegetative cover d = 0.64 times the height of the vegetative) */
    float momentumRoughness;    /*!  [m] momentum roughness parameter (for snow = 0.001m, for vegetative cover zm = 0.13 times the height of the vegetation) */
    float log2;                 // equivalente a vegetativeHeight = 1; log2 = (Zt - d + Zh)/Zh , Zt: measurement height for temperature  sempre pari a 2m (i sensori di temperatura sono piazzati a 2metri) */
    float log1;

    /*! check on wind speed [m/s] */
    myWindSpeed = std::max(myWindSpeed, 0.2f);

    if (isSnow)
    {
        zeroPlane = 0;
        momentumRoughness = 0.001f;
        log2 = 9.2f;
    }
    else
    {
        /*! check on vegetativeHeight  [m] */
        vegetativeHeight = std::max(vegetativeHeight, 0.1f);

        //pag 51: the height of the zero-plane displacement
        zeroPlane = 0.64f * vegetativeHeight;

        momentumRoughness = 0.13f * vegetativeHeight;
        log2 = 4;
    }

    if (zeroPlane > zRefWind)
        zeroPlane = zRefWind;

    //formula 3.18 pag 51
    log1 = log((zRefWind - zeroPlane + momentumRoughness) / momentumRoughness);

    return log1 * log2 / (float(VON_KARMAN_CONST * VON_KARMAN_CONST) * myWindSpeed);
}
