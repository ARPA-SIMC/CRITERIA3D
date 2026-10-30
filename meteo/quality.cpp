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

#include "commonConstants.h"
#include "basicMath.h"
#include "quality.h"
#include "meteoPoint.h"
#include "meteo.h"


Crit3DQuality::Crit3DQuality()
    : qualityHourlyT(-60, 60),
    qualityHourlyTd(-60, 50),
    qualityHourlyP(0, 300),             // [mm]
    qualityHourlyRH(1, 104),
    qualityHourlyWInt(0, 150),
    qualityHourlyWDir(0, 360),
    qualityHourlyGIrr(-20, 1353),
    qualityTransmissivity(0, 1),
    qualityHourlyET0(0, 5),             // [mm]
    qualityHourlyleafWetness(0, 1),
    qualityDailyT(-60, 60),
    qualityDailyP(0, 999),              // [mm]
    qualityDailyRH(1, 104),
    qualityDailyWInt(0, 150),
    qualityDailyWDir(0, 360),
    qualityDailyGRad(-20, 120),
    qualityDailyET0(0, 20),             // [mm]
    qualityDailyBIC(-20, 999),          // [mm]
    qualityMissing(NODATA, NODATA)
{
    referenceHeight = DEF_VALUE_REF_HEIGHT;
    relHumTolerance = DEF_VALUE_REL_HUM_TOLERANCE;
    waterTableMaximumDepth = DEF_VALUE_WATERTABLE_MAX_DEPTH;
}


const quality::Range &Crit3DQuality::getQualityRange(meteoVariable myVar)
{
    // hourly
    if (myVar == airTemperature)
        return qualityHourlyT;
    else if (myVar == precipitation)
        return qualityHourlyP;
    else if (myVar == globalIrradiance)
        return qualityHourlyGIrr;
    else if (myVar == atmTransmissivity)
        return qualityTransmissivity;
    else if (myVar == airRelHumidity)
        return qualityHourlyRH;
    else if (myVar == windScalarIntensity || myVar == windVectorIntensity)
        return qualityHourlyWInt;
    else if (myVar == windVectorDirection || myVar == dailyWindVectorDirectionPrevailing)
        return qualityHourlyWDir;
    else if (myVar == airDewTemperature)
        return qualityHourlyTd;
    else if (myVar == leafWetness)
        return qualityHourlyleafWetness;
    else if (myVar == referenceEvapotranspiration);

    // daily
    else if (myVar == dailyAirTemperatureMax
          || myVar == dailyAirTemperatureMin
          || myVar == dailyAirTemperatureAvg)
        return qualityDailyT;

    else if (myVar == dailyPrecipitation)
        return qualityDailyP;

    else if (myVar == dailyAirRelHumidityMax
          || myVar == dailyAirRelHumidityMin
          || myVar == dailyAirRelHumidityAvg)
        return qualityDailyRH;

    else if (myVar == dailyGlobalRadiation)
        return qualityDailyGRad;

    else if (myVar == dailyWindScalarIntensityAvg || myVar == dailyWindScalarIntensityMax || myVar == dailyWindVectorIntensityAvg || myVar == dailyWindVectorIntensityMax)
        return qualityDailyWInt;

    else if (myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM)
        return qualityDailyET0;

    else if (myVar == dailyBIC)
        return qualityDailyBIC;

    return qualityMissing;
}


void Crit3DQuality::syntacticQualityControl(meteoVariable myVar, std::vector<Crit3DMeteoPoint> &meteoPoints)
{
    float qualityMin = NODATA;
    float qualityMax = NODATA;

    bool hasRange = false;
    const quality::Range& myRange = getQualityRange(myVar);

    if (! isEqual(myRange.getMin(), NODATA) && ! isEqual(myRange.getMax(), NODATA))
    {
        hasRange = true;
        qualityMin = myRange.getMin();
        qualityMax = myRange.getMax();
    }

    for (size_t i = 0; i < meteoPoints.size(); ++i)
    {
        if (isEqual(meteoPoints[i].currentValue, NODATA))
        {
            meteoPoints[i].quality = quality::missing_data;
        }
        else
        {
            if (! hasRange)
            {
                meteoPoints[i].quality = quality::accepted;
            }
            else
            {
                if (meteoPoints[i].currentValue < qualityMin || meteoPoints[i].currentValue > qualityMax)
                {
                    meteoPoints[i].quality = quality::wrong_syntactic;
                }
                else
                {
                    meteoPoints[i].quality = quality::accepted;
                }
            }
        }
    }
}


quality::qualityType Crit3DQuality::syntacticQualitySingleValue(meteoVariable myVar, float myValue)
{
    float qualityMin = NODATA;
    float qualityMax = NODATA;

    bool hasRange = false;
    const quality::Range& myRange = getQualityRange(myVar);

    if (! isEqual(myRange.getMin(), NODATA) && ! isEqual(myRange.getMax(), NODATA))
    {
        hasRange = true;
        qualityMin = myRange.getMin();
        qualityMax = myRange.getMax();
    }

    if (isEqual(myValue, NODATA))
        return quality::missing_data;
    else
    {
        if (! hasRange)
            return quality::accepted;

        else if (myValue < qualityMin || myValue > qualityMax)
            return quality::wrong_syntactic;

        else
            return quality::accepted;
    }
}


quality::qualityType Crit3DQuality::checkFastValueDaily_SingleValue(meteoVariable myVar, const Crit3DClimateParameters &climateParam,
                                                                    float myValue, int month, float height)
{
    if (isEqual(myValue, NODATA))
        return quality::missing_data;
    else if (wrongValueDaily_SingleValue(myVar, climateParam, myValue, month, height))
        return quality::wrong_spatial;
    else
        return quality::accepted;
}


bool Crit3DQuality::wrongValueDaily_SingleValue(meteoVariable myVar, const Crit3DClimateParameters &climateParam,
                                                float myValue, int month, float height)
{
    if (myVar == dailyAirTemperatureMin || myVar == dailyAirTemperatureMax || myVar == dailyAirTemperatureAvg)
    {
        float tminClima = climateParam.getClimateVar(dailyAirTemperatureMin, month, height, getReferenceHeight());
        float tmaxClima = climateParam.getClimateVar(dailyAirTemperatureMax, month, height, getReferenceHeight());

        if (isEqual(tminClima, NODATA) || isEqual(tmaxClima, NODATA))
        {
            return (myValue < qualityDailyT.getMin()
                    || myValue > qualityDailyT.getMax());
        }

        if (myVar == dailyAirTemperatureMin)
        {
            if (myValue < qualityDailyT.getMin() + tminClima
                || myValue > qualityDailyT.getMax() + tminClima)
                return true;
        }
        else if (myVar == dailyAirTemperatureAvg)
        {
            if (myValue < qualityDailyT.getMin() + tminClima
                || myValue > qualityDailyT.getMax() + tmaxClima)
                return true;
        }
        else if (myVar == dailyAirTemperatureMax)
        {
            if (myValue < qualityDailyT.getMin() + tmaxClima
                || myValue > qualityDailyT.getMax() + tmaxClima)
                return true;
        }
    }
    else
    {
        const quality::Range& myRange = getQualityRange(myVar);

        if (! isEqual(myRange.getMin(), NODATA) && ! isEqual(myRange.getMax(), NODATA))
        {
            if (myValue < myRange.getMin() || myValue > myRange.getMax())
                return true;
        }
    }

    return false;
}


quality::qualityType Crit3DQuality::checkFastValueHourly_SingleValue(meteoVariable myVar, const Crit3DClimateParameters &climateParam,
                                                                     float myValue, int month, float height)
{
    if (isEqual(myValue, NODATA))
        return quality::missing_data;
    else if (wrongValueHourly_SingleValue(myVar, climateParam, myValue, month, height))
        return quality::wrong_spatial;
    else
        return quality::accepted;
}


bool Crit3DQuality::wrongValueHourly_SingleValue(meteoVariable myVar, const Crit3DClimateParameters &climateParam,
                                                 float myValue, int month, float height)
{
    switch (myVar)
    {
    case airTemperature:
    {
        float tminClima = climateParam.getClimateVar(dailyAirTemperatureMin, month, height, getReferenceHeight());
        float tmaxClima = climateParam.getClimateVar(dailyAirTemperatureMax, month, height, getReferenceHeight());
        if (isEqual(tminClima, NODATA) || isEqual(tmaxClima, NODATA))
            return false;

        if (myValue < qualityHourlyT.getMin() + tminClima ||
            myValue > qualityHourlyT.getMax() + tmaxClima)
            return true;
        break;
    }

    case airDewTemperature:
    {
        float tdminClima = climateParam.getClimateVar(dailyAirRelHumidityMin, month, height, getReferenceHeight());
        float tdmaxClima = climateParam.getClimateVar(dailyAirRelHumidityMax, month, height, getReferenceHeight());
        if (isEqual(tdminClima, NODATA) || isEqual(tdmaxClima, NODATA))
            return false;

        if (myValue < qualityHourlyTd.getMin() + tdminClima ||
            myValue > qualityHourlyTd.getMax() + tdmaxClima)
            return true;
        break;
    }

    default:
        const quality::Range& myRange = getQualityRange(myVar);

        if (! isEqual(myRange.getMin(), NODATA) && ! isEqual(myRange.getMax(), NODATA))
        {
            if (myValue < myRange.getMin() || myValue > myRange.getMax())
                return true;
        }
    }

    return false;
}
