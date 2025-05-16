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


void Crit3DQuality::initialize()
{
    referenceHeight = DEF_VALUE_REF_HEIGHT;
    deltaTSuspect = DEF_VALUE_DELTA_T_SUSP;
    deltaTWrong = DEF_VALUE_DELTA_T_WRONG;
    relHumTolerance = DEF_VALUE_REL_HUM_TOLERANCE;
    waterTableMaximumDepth = DEF_VALUE_WATERTABLE_MAX_DEPTH;
}


Crit3DQuality::Crit3DQuality()
{
    qualityHourlyT = new quality::Range(-60, 60);
    qualityHourlyTd = new quality::Range(-60, 50);
    qualityHourlyP = new quality::Range(0, 300);
    qualityHourlyRH = new quality::Range(1, 104);
    qualityHourlyWInt = new quality::Range(0, 150);
    qualityHourlyWDir = new quality::Range(0, 360);
    qualityHourlyGIrr = new quality::Range(-20, 1353);
    qualityHourlyET0 = new quality::Range(0, 5);
    qualityHourlyleafWetness = new quality::Range(0, 1);

    qualityTransmissivity = new quality::Range(0, 1);

    qualityDailyT = new quality::Range(-60, 60);
    qualityDailyP = new quality::Range(0, 1000);        // [mm]
    qualityDailyRH = new quality::Range(1, 104);
    qualityDailyWInt = new quality::Range(0, 150);
    qualityDailyWDir = new quality::Range(0, 360);
    qualityDailyGRad = new quality::Range(-20, 120);
    qualityDailyET0 = new quality::Range(0, 20);        // [mm]
    qualityDailyBIC = new quality::Range(-20, 1000);    // [mm]

    initialize();
}


Crit3DQuality::~Crit3DQuality()
{
    delete qualityHourlyT;
    delete qualityHourlyTd;
    delete qualityHourlyP;
    delete qualityHourlyRH;
    delete qualityHourlyWInt;
    delete qualityHourlyWDir;
    delete qualityHourlyGIrr;
    delete qualityHourlyET0;
    delete qualityHourlyleafWetness;

    delete qualityTransmissivity;

    delete qualityDailyT;
    delete qualityDailyP;
    delete qualityDailyRH;
    delete qualityDailyWInt;
    delete qualityDailyWDir;
    delete qualityDailyGRad;
    delete qualityDailyET0;
    delete qualityDailyBIC;
}


quality::Range* Crit3DQuality::getQualityRange(meteoVariable myVar)
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

    return nullptr;
}


void Crit3DQuality::syntacticQualityControl(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints)
{
    float qualityMin = NODATA;
    float qualityMax = NODATA;

    quality::Range* myRange = this->getQualityRange(myVar);
    if (myRange != nullptr)
    {
        qualityMin = myRange->getMin();
        qualityMax = myRange->getMax();
    }

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (isEqual(meteoPoints[i].currentValue, NODATA))
            meteoPoints[i].quality = quality::missing_data;
        else
        {
            if (myRange == nullptr)
            {
                meteoPoints[i].quality = quality::accepted;
            }
            else
            {
                if (meteoPoints[i].currentValue < qualityMin || meteoPoints[i].currentValue > qualityMax)
                    meteoPoints[i].quality = quality::wrong_syntactic;
                else
                    meteoPoints[i].quality = quality::accepted;
            }
        }
    }
}


quality::qualityType Crit3DQuality::syntacticQualitySingleValue(meteoVariable myVar, float myValue)
{
    float qualityMin = NODATA;
    float qualityMax = NODATA;

    quality::Range* myRange = this->getQualityRange(myVar);
    if (myRange != nullptr)
    {
        qualityMin = myRange->getMin();
        qualityMax = myRange->getMax();
    }

    if (int(myValue) == int(NODATA))
        return quality::missing_data;
    else
    {
        if (myRange == nullptr)
            return quality::accepted;

        else if (myValue < qualityMin || myValue > qualityMax)
            return quality::wrong_syntactic;

        else
            return quality::accepted;
    }
}


quality::qualityType Crit3DQuality::checkFastValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam,
                                                                    float myValue, int month, float height)
{
    if (isEqual(myValue, NODATA))
        return quality::missing_data;
    else if (wrongValueDaily_SingleValue(myVar, climateParam, myValue, month, height))
        return quality::wrong_spatial;
    else
        return quality::accepted;
}


bool Crit3DQuality::wrongValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam,
                                                float myValue, int month, float height)
{
    if (myVar == dailyAirTemperatureMin || myVar == dailyAirTemperatureMax || myVar == dailyAirTemperatureAvg)
    {
        float tminClima = climateParam->getClimateVar(dailyAirTemperatureMin, month, height, getReferenceHeight());
        float tmaxClima = climateParam->getClimateVar(dailyAirTemperatureMax, month, height, getReferenceHeight());

        if (isEqual(tminClima, NODATA) || isEqual(tmaxClima, NODATA))
            return false;

        if (myVar == dailyAirTemperatureMin)
        {
            if (myValue < tminClima - getDeltaTWrong() ||
                myValue > tminClima + getDeltaTWrong()) return true;
        }
        else if (myVar == dailyAirTemperatureAvg)
        {
            if (myValue < tminClima - getDeltaTWrong() ||
                myValue > tmaxClima + getDeltaTWrong()) return true;
        }
        else if (myVar == dailyAirTemperatureMax)
        {
            if (myValue < tmaxClima - getDeltaTWrong() ||
                myValue > tmaxClima + getDeltaTWrong()) return true;
        }
    }
    else
    {
        quality::Range* qualityRange = getQualityRange(myVar);
        if (qualityRange != nullptr)
        {
            if (myValue < qualityRange->getMin() || myValue > qualityRange->getMax())
                return true;
        }
    }

    return false;
}


quality::qualityType Crit3DQuality::checkFastValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam, float myValue, int month, float height)
{
    if (isEqual(myValue, NODATA))
        return quality::missing_data;
    else if (wrongValueHourly_SingleValue(myVar, climateParam, myValue, month, height))
        return quality::wrong_spatial;
    else
        return quality::accepted;
}


bool Crit3DQuality::wrongValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam,
                                                 float myValue, int month, float height)
{
    float tminClima = NODATA;
    float tmaxClima = NODATA;
    float tdminClima = NODATA;
    float tdmaxClima = NODATA;

    if (myVar == airTemperature)
    {
        tminClima = climateParam->getClimateVar(dailyAirTemperatureMin, month, height, getReferenceHeight());
        tmaxClima = climateParam->getClimateVar(dailyAirTemperatureMax, month, height, getReferenceHeight());
        if (isEqual(tminClima, NODATA) || isEqual(tmaxClima, NODATA))
            return false;
    }
    if (myVar == airRelHumidity || myVar == airDewTemperature)
    {
        tdminClima = climateParam->getClimateVar(dailyAirRelHumidityMin, month, height, getReferenceHeight());
        tdmaxClima = climateParam->getClimateVar(dailyAirRelHumidityMax, month, height, getReferenceHeight());
        if ( isEqual(tdminClima, NODATA) || isEqual(tdmaxClima, NODATA))
            return false;
    }

    if (myVar == airTemperature)
    {
        if (myValue < tminClima + qualityHourlyT->getMin() ||
            myValue > tmaxClima + qualityHourlyT->getMax()) return true;
    }
    else if (myVar == precipitation)
    {
        if (myValue < qualityHourlyP->getMin() ||
            myValue > qualityHourlyP->getMax()) return true;
    }
    else if (myVar == airRelHumidity)
    {
        if (myValue < qualityHourlyRH->getMin() ||
            myValue > qualityHourlyRH->getMax()) return true;
    }
    else if (myVar == airDewTemperature)
    {
        if ( (! isEqual(myValue, NODATA)) &&
            (myValue < (tdminClima + qualityHourlyTd->getMin()) || myValue > (tdmaxClima + qualityHourlyTd->getMax())) )
            return true;
    }
    else if (myVar == windVectorIntensity || myVar == windScalarIntensity || myVar == windVectorX || myVar == windVectorY)
    {
        if (myValue < qualityDailyWInt->getMin() ||
            myValue > qualityDailyWInt->getMax()) return true;
    }
    else if (myVar == windVectorDirection)
    {
        if (myValue < qualityHourlyWDir->getMin() ||
            myValue > qualityHourlyWDir->getMax()) return true;
    }
    else if (myVar == globalIrradiance)
    {
        if (myValue < qualityHourlyRH->getMin() ||
            myValue > qualityHourlyRH->getMax()) return true;
    }
    else if (myVar == leafWetness)
    {
        if (myValue < qualityHourlyleafWetness->getMin() ||
            myValue > qualityHourlyleafWetness->getMax()) return true;
    }
    else
    {
        quality::Range* qualityRange = getQualityRange(myVar);
        if (qualityRange != nullptr)
        {
            if (myValue < qualityRange->getMin() || myValue > qualityRange->getMax())
                return true;
        }
    }

    return false;
}
