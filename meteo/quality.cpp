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
#include "quality.h"
#include "meteoPoint.h"

namespace quality
{
    Range::Range()
    {
        min = NODATA;
        max = NODATA;
    }

    Range::Range(float myMin, float myMax)
    {
        min = myMin;
        max = myMax;
    }

    float Range::getMin() { return min; }

    float Range::getMax() { return max; }
}


float Crit3DQuality::getReferenceHeight() const
{
    return referenceHeight;
}

void Crit3DQuality::setReferenceHeight(float value)
{
    referenceHeight = value;
}

float Crit3DQuality::getDeltaTSuspect() const
{
    return deltaTSuspect;
}

void Crit3DQuality::setDeltaTSuspect(float value)
{
    deltaTSuspect = value;
}

float Crit3DQuality::getDeltaTWrong() const
{
    return deltaTWrong;
}

void Crit3DQuality::setDeltaTWrong(float value)
{
    deltaTWrong = value;
}

float Crit3DQuality::getRelHumTolerance() const
{
    return relHumTolerance;
}

void Crit3DQuality::setRelHumTolerance(float value)
{
    relHumTolerance = value;
}

void Crit3DQuality::initialize()
{
    referenceHeight = DEF_VALUE_REF_HEIGHT;
    deltaTSuspect = DEF_VALUE_DELTA_T_SUSP;
    deltaTWrong = DEF_VALUE_DELTA_T_WRONG;
    relHumTolerance = DEF_VALUE_REL_HUM_TOLERANCE;
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
    qualityTransmissivity = new quality::Range(0, 1);

    qualityDailyT = new quality::Range(-60, 60);
    qualityDailyP = new quality::Range(0, 800);
    qualityDailyRH = new quality::Range(1, 104);
    qualityDailyWInt = new quality::Range(0, 150);
    qualityDailyWDir = new quality::Range(0, 360);
    qualityDailyGRad = new quality::Range(-20, 120);

    initialize();
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

    else
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
        if (int(meteoPoints[i].currentValue) == int(NODATA))
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

quality::qualityType Crit3DQuality::checkFastValueDaily_SingleValue(meteoVariable myVar, float myValue, int month, int height)
{

    if (int(myValue) == int(NODATA))
        return quality::missing_data;
    else if (wrongValueDaily_SingleValue(myVar, myValue, month, height))
    {
        return quality::wrong_spatial; // LC in vb è Quality.qualityWrongData
    }
    else
    {
        return quality::accepted; // LC in vb Quality.qualityGoodData
    }

}

bool Crit3DQuality::wrongValueDaily_SingleValue(meteoVariable myVar, float myValue, int month, int height)
{
    Crit3DClimateParameters climateParam;
    Crit3DDate myDate(1,month,2020); // LC serve solo il mese, non ha piu' senso che la getClimateVar prenda solo il mese?
    int myHour = 0; // LC quale è il valore di default?

    float tMinClima = climateParam.getClimateVar(dailyAirTemperatureMin, myDate, myHour);
    float tMaxClima = climateParam.getClimateVar(dailyAirTemperatureMax, myDate, myHour);
    if (myVar == dailyAirTemperatureMin || myVar == dailyAirTemperatureMax || myVar == dailyAirTemperatureAvg)
    {
        if ( tMinClima == NODATA || tMaxClima == NODATA)
        {
            return false;
        }
    }
    if (myVar == dailyAirTemperatureMin)
    {
        /*
        If (dato < TMINclima + quality_range.tmin_range.red.min) Or _
                        (dato > TMAXclima + quality_range.tmax_range.red.max) Then
                        WrongValueDaily_SingleValue = True
                        qualityError = qualityError + InfoMsg.OutOfRange
                    End If
                    */
        // LC quality_range.tmax_range.red.max ?? red?
        if (myValue < tMinClima + getQualityRange(myVar)->getMin() || myValue > tMinClima + getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyAirTemperatureAvg)
    {
        if (myValue < tMinClima + getQualityRange(dailyAirTemperatureMin)->getMin() || myValue > tMaxClima + getQualityRange(dailyAirTemperatureMax)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyAirTemperatureMax)
    {
        if (myValue < tMaxClima + getQualityRange(myVar)->getMin() || myValue > tMaxClima + getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyPrecipitation)
    {
        if (myValue < 0 || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM)
    {
        // LC non c'è la referenceEvap in getQualituRange
        if (myValue < getQualityRange(myVar)->getMin() || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyAirRelHumidityMin || myVar == dailyAirRelHumidityMax || dailyAirRelHumidityAvg)
    {
        if (myValue < getQualityRange(myVar)->getMin() || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyWindScalarIntensityAvg)
    {
        // Case Definitions.DAILY_VMED_INT, Definitions.DAILY_VMED_X, Definitions.DAILY_VMED_Y a quali variabili corrispondono ??
        if (myValue < getQualityRange(myVar)->getMin() || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyWindVectorDirectionPrevailing)
    {
        if (myValue < 0 || myValue > 360)
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyGlobalRadiation)
    {
        if (myValue < getQualityRange(myVar)->getMin() || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    else if (myVar == dailyLeafWetness)
    {
        // LC la getQualityRange non ha questa var
        if (myValue < getQualityRange(myVar)->getMin() || myValue > getQualityRange(myVar)->getMax())
        {
            // qualityError = qualityError + InfoMsg.OutOfRange  LC dobbiamo salvare questa info?
            return true;
        }
    }
    return false;

}
