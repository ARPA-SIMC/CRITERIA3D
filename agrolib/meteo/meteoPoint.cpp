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
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "meteoPoint.h"
#include "quality.h"
#include "crit3dDate.h"
#include "meteo.h"


namespace {

    // values stored as they are (no derived computation)
    float storedDailyValue(const TObsDataD& dailydata, meteoVariable myVar)
    {
        switch (myVar)
        {
        case dailyAirTemperatureMax:                return dailydata.tMax;
        case dailyAirTemperatureMin:                return dailydata.tMin;
        case dailyAirTemperatureAvg:                return dailydata.tAvg;
        case dailyPrecipitation:                    return dailydata.prec;
        case dailyAirRelHumidityMax:                return dailydata.rhMax;
        case dailyAirRelHumidityMin:                return dailydata.rhMin;
        case dailyAirRelHumidityAvg:                return dailydata.rhAvg;
        case dailyGlobalRadiation:                  return dailydata.globRad;
        case dailyReferenceEvapotranspirationHS:    return dailydata.et0_hs;
        case dailyReferenceEvapotranspirationPM:    return dailydata.et0_pm;
        case dailyBIC:                              return dailydata.bic;
        case dailyHeatingDegreeDays:                return dailydata.dd_heating;
        case dailyCoolingDegreeDays:                return dailydata.dd_cooling;
        case dailyWindScalarIntensityAvg:           return dailydata.windScalIntAvg;
        case dailyWindScalarIntensityMax:           return dailydata.windScalIntMax;
        case dailyWindVectorIntensityAvg:           return dailydata.windVecIntAvg;
        case dailyWindVectorIntensityMax:           return dailydata.windVecIntMax;
        case dailyWindVectorDirectionPrevailing:    return dailydata.windVecDirPrev;
        case dailyLeafWetness:                      return dailydata.leafW;
        case dailyWaterTableDepth:                  return dailydata.waterTable;

        default:                                    return NODATA;
        }
    }
}


Crit3DMeteoPoint::Crit3DMeteoPoint()
{
    this->topographicDistanceMap = nullptr;
    this->clear();
}

void Crit3DMeteoPoint::clear()
{
    this->dataset = "";
    this->municipality = "";
    this->state = "";
    this->region = "";
    this->province = "";

    this->name = "";
    this->id = "";
    this->isUTC = true;
    this->isForecast = false;

    this->aggregationPointsMaxNr = 0;

    this->latitude = NODATA;
    this->longitude = NODATA;
    this->area = NODATA;
    this->latInt = NODATA;
    this->lonInt = NODATA;
    this->isInsideDem = false;

    this->hourlyFraction = 1;

    this->currentValue = NODATA;
    this->residual = NODATA;

    this->elaboration = NODATA;
    this->anomaly = NODATA;
    this->anomalyPercentage = NODATA;
    this->climate = NODATA;

    this->active = false;
    this->selected = false;
    this->marked = false;

    this->lapseRateCode = primary;

    this->cleanAllData();
}


void Crit3DMeteoPoint::cleanAllData()
{
    obsDataH.clear();
    obsDataD.clear();
    obsDataM.clear();

    proxyValues.clear();
    glocalWeights.clear();
    aggregationPoints.clear();

    if (topographicDistanceMap != nullptr)
    {
        topographicDistanceMap->clear();
        delete topographicDistanceMap;
        topographicDistanceMap = nullptr;
    }

    quality = quality::missing_data;
}


void Crit3DMeteoPoint::setLapseRateCode(const std::string &lapseRateCodeStr)
{
    if (lapseRateCodeStr == "primary")
    {
        lapseRateCode = primary;
    }
    else if (lapseRateCodeStr == "secondary")
    {
        lapseRateCode = secondary;
    }
    else if (lapseRateCodeStr == "supplemental")
    {
        lapseRateCode = supplemental;
    }
}


bool Crit3DMeteoPoint::initializeObsDataH(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate)
{
    obsDataH.clear();

    if (numberOfDays <= 0)
        return false;

    if (myHourlyFraction <= 0 || 60 % myHourlyFraction != 0)
        return false;

    obsDataH.resize(numberOfDays);
    hourlyFraction = myHourlyFraction;
    const unsigned int nrDailyValues = unsigned(hourlyFraction * 24);

    quality = quality::missing_data;
    residual = NODATA;

    Crit3DDate myDate = firstDate;
    for (int i = 0; i < numberOfDays; i++)
    {
        obsDataH[i].date = myDate;

        obsDataH[i].tAir.assign(nrDailyValues, NODATA);
        obsDataH[i].prec.assign(nrDailyValues, NODATA);
        obsDataH[i].rhAir.assign(nrDailyValues, NODATA);
        obsDataH[i].tDew.assign(nrDailyValues, NODATA);
        obsDataH[i].irradiance.assign(nrDailyValues, NODATA);
        obsDataH[i].netIrradiance.assign(nrDailyValues, NODATA);
        obsDataH[i].et0.assign(nrDailyValues, NODATA);
        obsDataH[i].windVecX.assign(nrDailyValues, NODATA);
        obsDataH[i].windVecY.assign(nrDailyValues, NODATA);
        obsDataH[i].windVecInt.assign(nrDailyValues, NODATA);
        obsDataH[i].windVecDir.assign(nrDailyValues, NODATA);
        obsDataH[i].windScalInt.assign(nrDailyValues, NODATA);
        obsDataH[i].leafW.assign(nrDailyValues, int(NODATA));
        obsDataH[i].transmissivity.assign(nrDailyValues, NODATA);
        obsDataH[i].pressure.assign(nrDailyValues, NODATA);

        ++myDate;
    }

    return true;
}


void Crit3DMeteoPoint::initializeObsDataHFromMp(int myHourlyFraction, int numberOfDays,
                                                const Crit3DDate& firstDate, const Crit3DMeteoPoint& meteoPoint)
{
    if (&meteoPoint == this || obsDataH.empty() || meteoPoint.obsDataH.empty())
        return;
    if (myHourlyFraction != hourlyFraction || meteoPoint.hourlyFraction != hourlyFraction)
        return;

    const int nrDays = std::min({numberOfDays, int(obsDataH.size()), int(meteoPoint.obsDataH.size())});

    Crit3DDate myDate = firstDate;
    for (int i = 0; i < nrDays; i++, ++myDate)
    {
        obsDataH[size_t(i)] = meteoPoint.obsDataH[size_t(i)];   // whole-struct copy: safe if members are added
        obsDataH[size_t(i)].date = myDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataD(unsigned int numberOfDays, const Crit3DDate& firstDate)
{
    obsDataD.clear();
    obsDataD.resize(numberOfDays);

    quality = quality::missing_data;
    residual = NODATA;

    Crit3DDate myDate = firstDate;
    for (unsigned int i = 0; i < numberOfDays; i++)
    {
        obsDataD[i].date = myDate;
        obsDataD[i].tMax = NODATA;
        obsDataD[i].tMin = NODATA;
        obsDataD[i].tAvg = NODATA;
        obsDataD[i].prec = NODATA;
        obsDataD[i].rhMax = NODATA;
        obsDataD[i].rhMin = NODATA;
        obsDataD[i].rhAvg = NODATA;
        obsDataD[i].globRad = NODATA;
        obsDataD[i].et0_hs = NODATA;
        obsDataD[i].et0_pm = NODATA;
        obsDataD[i].bic = NODATA;
        obsDataD[i].dd_heating = NODATA;
        obsDataD[i].dd_cooling = NODATA;
        obsDataD[i].windVecIntAvg = NODATA;
        obsDataD[i].windVecIntMax = NODATA;
        obsDataD[i].windVecDirPrev = NODATA;
        obsDataD[i].windScalIntAvg = NODATA;
        obsDataD[i].windScalIntMax = NODATA;
        obsDataD[i].leafW = NODATA;
        obsDataD[i].waterTable = NODATA;
        ++myDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataDFromMp(unsigned int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint &refMeteoPoint)
{
    initializeObsDataD(numberOfDays, firstDate);

    Crit3DDate currentDate = firstDate;
    const size_t nrRows = std::min(size_t(numberOfDays), refMeteoPoint.obsDataD.size());

    for (size_t i = 0; i < nrRows; i++)
    {
        obsDataD[i] = refMeteoPoint.obsDataD[i];
        obsDataD[i].date = currentDate;
        ++currentDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataM(unsigned int numberOfMonths, unsigned int month, int year)
{
    obsDataM.clear();
    obsDataM.resize(numberOfMonths);

    quality = quality::missing_data;
    residual = NODATA;
    int addYear = 0;

    for (unsigned int i = month; i < month+numberOfMonths; i++)
    {
        if (i <= 12)
        {
            obsDataM[i-month]._month = i;   // obsDataM start from 0
            obsDataM[i-month]._year = year;
        }
        else
        {
            if (i%12 == 0)
            {
                obsDataM[i-month]._month = 12;
            }
            else
            {
                obsDataM[i-month]._month = i%12;
            }
            if (obsDataM[i-month]._month == 1)
            {
                // new year
                addYear++;
            }
            obsDataM[i-month]._year = year + addYear;
        }

        obsDataM[i-month].tMax = NODATA;
        obsDataM[i-month].tMin = NODATA;
        obsDataM[i-month].tAvg = NODATA;
        obsDataM[i-month].prec = NODATA;
        obsDataM[i-month].et0_hs = NODATA;
        obsDataM[i-month].globRad = NODATA;
        obsDataM[i-month].bic = NODATA;
    }
}


void Crit3DMeteoPoint::emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& myDate)
{
    if (! isDateLoadedH(myDate)) return;

    int nrDayValues = hourlyFraction * 24;
    int i = obsDataH[0].date.daysTo(myDate);
    residual = NODATA;

    if (i >= 0 && i < obsDataH.size())
        if (obsDataH[i].date == myDate)
            for (int j = 0; j < nrDayValues; j++)
            {
                if (myVar == airTemperature)
                    obsDataH[i].tAir[j] = NODATA;
                else if (myVar == precipitation)
                    obsDataH[i].prec[j] = NODATA;
                else if (myVar == airRelHumidity)
                    obsDataH[i].rhAir[j] = NODATA;
                else if (myVar == airDewTemperature)
                    obsDataH[i].tDew[j] = NODATA;
                else if (myVar == globalIrradiance)
                    obsDataH[i].irradiance[j] = NODATA;
                else if (myVar == netIrradiance)
                    obsDataH[i].netIrradiance[j] = NODATA;
                else if (myVar == windScalarIntensity)
                    obsDataH[i].windScalInt[j] = NODATA;
                else if (myVar == windVectorX)
                    obsDataH[i].windVecX[j] = NODATA;
                else if (myVar == windVectorY)
                    obsDataH[i].windVecY[j] = NODATA;
                else if (myVar == windVectorIntensity)
                    obsDataH[i].windVecInt[j] = NODATA;
                else if (myVar == windVectorDirection)
                    obsDataH[i].windVecDir[j] = NODATA;
                else if (myVar == leafWetness)
                    obsDataH[i].leafW[j] = NODATA;
                else if (myVar == atmTransmissivity)
                    obsDataH[i].transmissivity[j] = NODATA;
				else if (myVar == referenceEvapotranspiration)
                    obsDataH[i].et0[j] = NODATA;
                else if (myVar == atmPressure)
                    obsDataH[i].pressure[j] = NODATA;
			}
}


void Crit3DMeteoPoint::emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int nrDayValues = hourlyFraction * 24;
    int indexIni = obsDataH[0].date.daysTo(date1);
    int indexFin = obsDataH[0].date.daysTo(date2);
    residual = NODATA;

    for (int i = indexIni; i <= indexFin; i++)
        for (int j = 0; j < nrDayValues; j++)
        {
            if (myVar == airTemperature)
                obsDataH[i].tAir[j] = NODATA;
            else if (myVar == precipitation)
                obsDataH[i].prec[j] = NODATA;
            else if (myVar == airRelHumidity)
                obsDataH[i].rhAir[j] = NODATA;
            else if (myVar == airDewTemperature)
                obsDataH[i].tDew[j] = NODATA;
            else if (myVar == globalIrradiance)
                obsDataH[i].irradiance[j] = NODATA;
            else if (myVar == netIrradiance)
                obsDataH[i].netIrradiance[j] = NODATA;
            else if (myVar == windScalarIntensity)
                obsDataH[i].windScalInt[j] = NODATA;
            else if (myVar == windVectorX)
                obsDataH[i].windVecX[j] = NODATA;
            else if (myVar == windVectorY)
                obsDataH[i].windVecY[j] = NODATA;
            else if (myVar == windVectorIntensity)
                obsDataH[i].windVecInt[j] = NODATA;
            else if (myVar == windVectorDirection)
                obsDataH[i].windVecDir[j] = NODATA;
            else if (myVar == leafWetness)
                obsDataH[i].leafW[j] = NODATA;
            else if (myVar == atmTransmissivity)
                obsDataH[i].transmissivity[j] = NODATA;
			else if (myVar == referenceEvapotranspiration)
                obsDataH[i].et0[j] = NODATA;
            else if (myVar == atmPressure)
                obsDataH[i].pressure[j] = NODATA;
        }
}


bool Crit3DMeteoPoint::emptyObsDataH(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2))
        return false;

    const int indexIni = obsDataH[0].date.daysTo(date1);
    const int indexFin = obsDataH[0].date.daysTo(date2);

    for (int i = indexIni; i <= indexFin; ++i)
    {
        auto& obs = obsDataH[i];
        std::fill(obs.tAir.begin(), obs.tAir.end(), NODATA);
        std::fill(obs.prec.begin(), obs.prec.end(), NODATA);
        std::fill(obs.rhAir.begin(), obs.rhAir.end(), NODATA);
        std::fill(obs.tDew.begin(), obs.tDew.end(), NODATA);
        std::fill(obs.irradiance.begin(), obs.irradiance.end(), NODATA);
        std::fill(obs.netIrradiance.begin(), obs.netIrradiance.end(), NODATA);
        std::fill(obs.transmissivity.begin(), obs.transmissivity.end(), NODATA);
        std::fill(obs.leafW.begin(), obs.leafW.end(), int(NODATA));
        std::fill(obs.et0.begin(), obs.et0.end(), NODATA);
        std::fill(obs.pressure.begin(), obs.pressure.end(), NODATA);
        std::fill(obs.windScalInt.begin(), obs.windScalInt.end(), NODATA);
        std::fill(obs.windVecX.begin(), obs.windVecX.end(), NODATA);
        std::fill(obs.windVecY.begin(), obs.windVecY.end(), NODATA);
        std::fill(obs.windVecInt.begin(), obs.windVecInt.end(), NODATA);
        std::fill(obs.windVecDir.begin(), obs.windVecDir.end(), NODATA);
    }

    return true;
}


bool Crit3DMeteoPoint::emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedD(date1, date2))
        return false;

    // select the member once, outside the loop
    float TObsDataD::* member = nullptr;
    switch (myVar)
    {
    case dailyAirTemperatureMax:                member = &TObsDataD::tMax; break;
    case dailyAirTemperatureMin:                member = &TObsDataD::tMin; break;
    case dailyAirTemperatureAvg:                member = &TObsDataD::tAvg; break;
    case dailyPrecipitation:                    member = &TObsDataD::prec; break;
    case dailyAirRelHumidityMax:                member = &TObsDataD::rhMax; break;
    case dailyAirRelHumidityMin:                member = &TObsDataD::rhMin; break;
    case dailyAirRelHumidityAvg:                member = &TObsDataD::rhAvg; break;
    case dailyGlobalRadiation:                  member = &TObsDataD::globRad; break;
    case dailyWindScalarIntensityAvg:           member = &TObsDataD::windScalIntAvg; break;
    case dailyWindScalarIntensityMax:           member = &TObsDataD::windScalIntMax; break;
    case dailyWindVectorIntensityAvg:           member = &TObsDataD::windVecIntAvg; break;
    case dailyWindVectorIntensityMax:           member = &TObsDataD::windVecIntMax; break;
    case dailyWindVectorDirectionPrevailing:    member = &TObsDataD::windVecDirPrev; break;
    case dailyReferenceEvapotranspirationHS:    member = &TObsDataD::et0_hs; break;
    case dailyReferenceEvapotranspirationPM:    member = &TObsDataD::et0_pm; break;
    case dailyBIC:                              member = &TObsDataD::bic; break;
    case dailyLeafWetness:                      member = &TObsDataD::leafW; break;
    case dailyHeatingDegreeDays:                member = &TObsDataD::dd_heating; break;
    case dailyCoolingDegreeDays:                member = &TObsDataD::dd_cooling; break;
    case dailyWaterTableDepth:                  member = &TObsDataD::waterTable; break;
    default:                                    return false;
    }

    const int indexIni = obsDataD[0].date.daysTo(date1);
    const int indexFin = obsDataD[0].date.daysTo(date2);

    for (int i = indexIni; i <= indexFin; i++)
        obsDataD[unsigned(i)].*member = NODATA;

    residual = NODATA;

    return true;
}


bool Crit3DMeteoPoint::emptyObsDataD(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedD(date1, date2))
        return false;

    const int indexIni = obsDataD[0].date.daysTo(date1);
    const int indexFin = obsDataD[0].date.daysTo(date2);

    for (unsigned int i = indexIni; i <= unsigned(indexFin); i++)
    {
        obsDataD[i].tMax = NODATA;
        obsDataD[i].tMin = NODATA;
        obsDataD[i].tAvg = NODATA;
        obsDataD[i].prec = NODATA;
        obsDataD[i].rhMax = NODATA;
        obsDataD[i].rhMin = NODATA;
        obsDataD[i].rhAvg = NODATA;
        obsDataD[i].globRad = NODATA;
        obsDataD[i].windScalIntAvg = NODATA;
        obsDataD[i].windScalIntMax = NODATA;
        obsDataD[i].windVecIntAvg = NODATA;
        obsDataD[i].windVecIntMax = NODATA;
        obsDataD[i].windVecDirPrev = NODATA;
        obsDataD[i].et0_hs = NODATA;
        obsDataD[i].et0_pm = NODATA;
        obsDataD[i].bic = NODATA;
        obsDataD[i].dd_heating = NODATA;
        obsDataD[i].dd_cooling = NODATA;
        obsDataD[i].leafW = NODATA;
        obsDataD[i].waterTable = NODATA;
    }

    return true;
}


bool Crit3DMeteoPoint::emptyObsDataM(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedM(date1, date2))
        return false;

    const int indexIni = (date1.year - obsDataM[0]._year)*12 + date1.month-obsDataM[0]._month;
    const int indexFin = (date2.year - obsDataM[0]._year)*12 + date2.month-obsDataM[0]._month;

    for (unsigned int i = indexIni; i <= unsigned(indexFin); i++)
    {
        obsDataM[i].tMax = NODATA;
        obsDataM[i].tMin = NODATA;
        obsDataM[i].tAvg = NODATA;
        obsDataM[i].prec = NODATA;
        obsDataM[i].et0_hs = NODATA;
        obsDataM[i].globRad = NODATA;
        obsDataM[i].bic = NODATA;
    }

    return true;
}


bool Crit3DMeteoPoint::isDateLoadedH(const Crit3DDate& myDate) const
{
    if (obsDataH.empty())
        return false;
    if (myDate < obsDataH[0].date || myDate > obsDataH[obsDataH.size() - 1].date)
        return false;

    return true;
}


bool Crit3DMeteoPoint::isDateTimeLoadedH(const Crit3DTime& myDateTime) const
{
    if (obsDataH.empty())
        return false;
    if (myDateTime < Crit3DTime(obsDataH[0].date,1) || myDateTime >= Crit3DTime(obsDataH.back().date,1))
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedH(const Crit3DDate& date1, const Crit3DDate& date2) const
{
    if (obsDataH.empty())
        return false;
    if (date1 > date2)
        return false;
    if (date1 < obsDataH[0].date || date2 > obsDataH.back().date)
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateLoadedD(const Crit3DDate& myDate) const
{
    if (obsDataD.empty())
        return false;
    if (myDate < obsDataD[0].date || myDate > obsDataD.back().date)
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateLoadedM(const Crit3DDate& myDate) const
{
    if (obsDataM.empty())
        return false;
    if ((myDate.year < obsDataM[0]._year ) || (myDate.year > obsDataM.back()._year))
        return false;
    if (myDate.year == obsDataM[0]._year && myDate.month < obsDataM[0]._month)
        return false;
    if (myDate.year == obsDataM.back()._year && myDate.month > obsDataM.back()._month)
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2) const
{
    if (obsDataD.empty())
        return false;
    if (date1 > date2)
        return false;
    if (date1 < obsDataD[0].date || date2 > obsDataD.back().date)
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedM(const Crit3DDate& date1, const Crit3DDate& date2) const
{
    if (obsDataM.empty())
        return false;
    if (date1 > date2)
        return false;
    if (date1.year < obsDataM[0]._year  || date2.year > obsDataM.back()._year)
        return false;
    if (date1.year == obsDataM[0]._year && date1.month < obsDataM[0]._month)
        return false;
    if (date2.year == obsDataM.back()._year && date2.month > obsDataM.back()._month)
        return false;

    return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedH(const Crit3DTime& timeIni, const Crit3DTime& timeFin) const
{
    if (obsDataH.empty())
        return false;
    if (timeIni > timeFin)
        return false;
    if (timeIni.date < obsDataH[0].date || timeFin.date > obsDataH.back().date)
        return false;

    return true;
}


float Crit3DMeteoPoint::obsDataConsistencyH(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin) const
{
    if (obsDataH.empty())
        return 0.0;
    if (timeIni > timeFin)
        return 0.0;
    if (timeFin.date < obsDataH[0].date || timeIni.date > obsDataH.back().date)
        return 0.0;

    const int deltaSeconds = 3600 / hourlyFraction;

    int counter = 0;
    int counterAll = 0;
    Crit3DTime myTime = timeIni;

    while (myTime <= timeFin)
    {
        const float myValue = getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar);
        if (! isEqual(myValue, NODATA) )
            counter++;

        counterAll++;
        myTime = myTime.addSeconds(deltaSeconds);
    }

    const float percentage = float(counter) / float(counterAll);
    return percentage;
}


bool Crit3DMeteoPoint::setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue)
{
    if (obsDataH.empty() || myVar == noMeteoVar)
        return false;

    // day index
    int iDay = obsDataH[0].date.daysTo(myDate);

    // out of range (accept +1 day exceed)
    if (iDay < 0 || size_t(iDay) > obsDataH.size())
        return false;

    // sub-hourly index: integer ceil(myMinutes / minutesPerStep)
    const int minutesPerStep = 60 / hourlyFraction;
    const int subH = (myMinutes + minutesPerStep - 1) / minutesPerStep;

    // if +1 day exceed, accept only 00:00
    if (iDay == obsDataH.size() && (myHour != 0 || subH != 0))
        return false;

    // hour 00:00 becomes hour 24 of the previous day
    if (myHour == 0 && subH == 0)
    {
        myHour = 24;
        if (--iDay < 0)
            return false;
    }

    // (sub)hour index
    const int j = hourlyFraction * myHour + subH - 1;
    if (j < 0 || j >= hourlyFraction * 24)
        return false;

    TObsDataH& obs = obsDataH[iDay];

    switch (myVar)
    {
    case airTemperature:                obs.tAir[j] = myValue;          break;
    case precipitation:                 obs.prec[j] = myValue;          break;
    case airRelHumidity:                obs.rhAir[j] = myValue;         break;
    case airDewTemperature:             obs.tDew[j] = myValue;          break;
    case globalIrradiance:              obs.irradiance[j] = myValue;    break;
    case netIrradiance:                 obs.netIrradiance[j] = myValue; break;
    case referenceEvapotranspiration:   obs.et0[j] = myValue;           break;
    case windScalarIntensity:           obs.windScalInt[j] = myValue;   break;
    case leafWetness:                   obs.leafW[j] = int(myValue);    break;
    case atmTransmissivity:             obs.transmissivity[j] = myValue; break;
    case atmPressure:                   obs.pressure[j] = myValue;      break;

    case windVectorX:
    case windVectorY:
    {
        // update the component first, then recompute the polar form
        if (myVar == windVectorX)
            obs.windVecX[j] = myValue;
        else
            obs.windVecY[j] = myValue;

        computeWindPolar(obs.windVecX[j], obs.windVecY[j], &obs.windVecInt[j], &obs.windVecDir[j]);
        break;
    }

    case windVectorIntensity:
    case windVectorDirection:
    {
        // update the polar component first, then recompute the cartesian form
        if (myVar == windVectorIntensity)
            obs.windVecInt[j] = myValue;
        else
            obs.windVecDir[j] = myValue;

        computeWindCartesian(obs.windVecInt[j], obs.windVecDir[j], &obs.windVecX[j], &obs.windVecY[j]);
        break;
    }

    default:
        return false;
    }

    return true;
}


bool Crit3DMeteoPoint::setMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, float myValue)
{
    if (obsDataD.empty() || myVar == noMeteoVar)
        return false;

    int index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= obsDataD.size()))
        return false;

    const unsigned i = unsigned(index);

    switch(myVar)
    {
    case dailyAirTemperatureMax:
        obsDataD[i].tMax = myValue; break;
    case dailyAirTemperatureMin:
        obsDataD[i].tMin = myValue; break;
    case dailyAirTemperatureAvg:
        obsDataD[i].tAvg = myValue; break;
    case dailyPrecipitation:
        obsDataD[i].prec = myValue; break;
    case dailyAirRelHumidityMax:
        obsDataD[i].rhMax = myValue; break;
    case dailyAirRelHumidityMin:
        obsDataD[i].rhMin = myValue; break;
    case dailyAirRelHumidityAvg:
        obsDataD[i].rhAvg = myValue; break;
    case dailyGlobalRadiation:
        obsDataD[i].globRad = myValue; break;
    case dailyReferenceEvapotranspirationHS:
        obsDataD[i].et0_hs = myValue; break;
    case dailyReferenceEvapotranspirationPM:
        obsDataD[i].et0_pm = myValue; break;
    case dailyBIC:
        obsDataD[i].bic = myValue; break;
    case dailyHeatingDegreeDays:
        obsDataD[i].dd_heating = myValue; break;
    case dailyCoolingDegreeDays:
        obsDataD[i].dd_cooling = myValue; break;
    case dailyWindScalarIntensityAvg:
        obsDataD[i].windScalIntAvg = myValue; break;
    case dailyWindScalarIntensityMax:
        obsDataD[i].windScalIntMax = myValue; break;
    case dailyWindVectorIntensityAvg:
        obsDataD[i].windVecIntAvg = myValue; break;
    case dailyWindVectorIntensityMax:
        obsDataD[i].windVecIntMax = myValue; break;
    case dailyWindVectorDirectionPrevailing:
        obsDataD[i].windVecDirPrev = myValue; break;
    case dailyLeafWetness:
        obsDataD[i].leafW = myValue; break;
    case dailyWaterTableDepth:
        obsDataD[i].waterTable = myValue; break;
    default:
        return false;
    }

    return true;
}


bool Crit3DMeteoPoint::setMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar, float myValue)
{
    if (obsDataM.empty() || myVar == noMeteoVar)
        return false;

    const int index = (myDate.year - obsDataM[0]._year) * 12 + (myDate.month - obsDataM[0]._month);

    if ((index < 0) || (index >= obsDataM.size()))
        return false;

    const unsigned i = unsigned(index);

    if (myVar == monthlyAirTemperatureMax)
        obsDataM[i].tMax = myValue;
    else if (myVar == monthlyAirTemperatureMin)
        obsDataM[i].tMin = myValue;
    else if (myVar == monthlyAirTemperatureAvg)
        obsDataM[i].tAvg = myValue;
    else if (myVar == monthlyPrecipitation)
        obsDataM[i].prec = myValue;
    else if (myVar == monthlyReferenceEvapotranspirationHS)
        obsDataM[i].et0_hs = myValue;
    else if (myVar == monthlyGlobalRadiation)
        obsDataM[i].globRad = myValue;
    else if (myVar == monthlyBIC)
        obsDataM[i].bic = myValue;
    else
        return false;

    return true;
}


float Crit3DMeteoPoint::getMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar) const
{
    if (obsDataH.empty() || myVar == noMeteoVar)
        return NODATA;

    // day index
    int iDay = obsDataH[0].date.daysTo(myDate);

    // out of range (accept +1 day exceed)
    if (iDay < 0 || size_t(iDay) > obsDataH.size())
        return NODATA;

    // sub-hourly index: integer ceil(myMinutes / minutesPerStep)
    const int minutesPerStep = 60 / hourlyFraction;
    const int subH = (myMinutes + minutesPerStep - 1) / minutesPerStep;

    // if +1 day exceed, accept only 00:00
    if (iDay == obsDataH.size() && (myHour != 0 || subH != 0))
        return NODATA;

    // hour 00:00 becomes hour 24 of the previous day
    if (myHour == 0 && subH == 0)
    {
        myHour = 24;
        if (--iDay < 0)
            return NODATA;
    }

    // (sub)hour index
    const int j = hourlyFraction * myHour + subH - 1;
    if (j < 0 || j >= hourlyFraction * 24)
        return NODATA;

    const TObsDataH& obs = obsDataH[iDay];

    switch (myVar)
    {
    case airTemperature:                return obs.tAir[j];
    case precipitation:                 return obs.prec[j];
    case airRelHumidity:                return obs.rhAir[j];
    case airDewTemperature:
    {
        const float tDew = obs.tDew[j];
        return isEqual(tDew, NODATA) ? tDewFromRelHum(obs.rhAir[j], obs.tAir[j]) : tDew;
    }
    case globalIrradiance:              return obs.irradiance[j];
    case netIrradiance:                 return obs.netIrradiance[j];
    case referenceEvapotranspiration:   return obs.et0[j];
    case windScalarIntensity:           return obs.windScalInt[j];
    case windVectorX:                   return obs.windVecX[j];
    case windVectorY:                   return obs.windVecY[j];
    case windVectorIntensity:           return obs.windVecInt[j];
    case windVectorDirection:           return obs.windVecDir[j];
    case leafWetness:                   return float(obs.leafW[j]);
    case atmTransmissivity:             return obs.transmissivity[j];
    case atmPressure:                   return obs.pressure[j];
    default:                            return NODATA;
    }
}


Crit3DDate Crit3DMeteoPoint::getMeteoPointHourlyValuesDate(int index) const
{
    if (obsDataH.empty() || index < 0 || index >= obsDataH.size())
        return NO_DATE;

    return obsDataH[index].date;
}


bool Crit3DMeteoPoint::getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH* &hourlyValues)
{
    if (obsDataH.empty())
        return false;

    int d = obsDataH[0].date.daysTo(myDate);
    if (d < 0 || d >= obsDataH.size())
        return false;

    hourlyValues = &(obsDataH[d]);

    return true;
}


bool Crit3DMeteoPoint::existDailyData(const Crit3DDate& myDate) const
{
    if (obsDataD.empty())
        return false;

    const int index = obsDataD[0].date.daysTo(myDate);

    if (index < 0 || index >= obsDataD.size() )
        return false;

    return true;
}


Crit3DDate Crit3DMeteoPoint::getLastDailyData() const
{
    if (obsDataD.empty())
        return NO_DATE;

    return obsDataD.back().date;
}


Crit3DDate Crit3DMeteoPoint::getFirstDailyData() const
{
    if (obsDataD.empty())
        return NO_DATE;

    return obsDataD[0].date;
}


float Crit3DMeteoPoint::getDailyET0_HS(const TObsDataD &dailyData, const Crit3DDate &myDate, Crit3DMeteoSettings* meteoSettings) const
{
    if (! isEqual(dailyData.et0_hs, NODATA))
        return dailyData.et0_hs;

    if (meteoSettings == nullptr || ! meteoSettings->getAutomaticET0HS())
        return NODATA;

    if (isEqual(dailyData.tMin, NODATA) || isEqual(dailyData.tMax, NODATA))
        return NODATA;

    return float(ET0_Hargreaves(meteoSettings->getTransSamaniCoefficient(),
                                latitude, getDoyFromDate(myDate),
                                dailyData.tMax, dailyData.tMin));
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar) const
{
    if (obsDataD.empty() || myVar == noMeteoVar)
        return NODATA;

    const int index = obsDataD[0].date.daysTo(myDate);
    if (index < 0 || size_t(index) >= obsDataD.size())
        return NODATA;

    const TObsDataD& obs = obsDataD[index];
    return storedDailyValue(obs, myVar);
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings) const
{
    if (obsDataD.empty() || myVar == noMeteoVar)
        return NODATA;

    const int index = obsDataD[0].date.daysTo(myDate);
    if (index < 0 || size_t(index) >= obsDataD.size())
        return NODATA;

    const TObsDataD& obs = obsDataD[index];

    switch (myVar)
    {
    case dailyAirTemperatureAvg:
    {
        if (! isEqual(obs.tAvg, NODATA))
            return obs.tAvg;

        if (meteoSettings != nullptr && meteoSettings->getAutomaticTavg()
            && ! isEqual(obs.tMin, NODATA) && ! isEqual(obs.tMax, NODATA))
            return (obs.tMin + obs.tMax) * 0.5f;

        return NODATA;
    }

    case dailyReferenceEvapotranspirationHS:
        return getDailyET0_HS(obs, myDate, meteoSettings);

    case dailyBIC:
    {
        if (! isEqual(obs.bic, NODATA))
            return obs.bic;

        if (isEqual(obs.prec, NODATA))
            return NODATA;

        const float et0 = getDailyET0_HS(obs, myDate, meteoSettings);
        return isEqual(et0, NODATA) ? NODATA : computeDailyBIC(obs.prec, et0);
    }

    default:
        return storedDailyValue(obs, myVar);
    }
}


float Crit3DMeteoPoint::getMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar) const
{
    if (obsDataM.empty() || myVar == noMeteoVar)
        return NODATA;

    const int index = (myDate.year - obsDataM[0]._year) * 12 + (myDate.month - obsDataM[0]._month);

    if (index < 0 || index >= obsDataM.size())
        return NODATA;

    const unsigned i = unsigned(index);

    if (myVar == monthlyAirTemperatureMax)
        return (obsDataM[i].tMax);
    else if (myVar == monthlyAirTemperatureMin)
        return (obsDataM[i].tMin);
    else if (myVar == monthlyAirTemperatureAvg)
        return obsDataM[i].tAvg;
    else if (myVar == monthlyPrecipitation)
        return (obsDataM[i].prec);
    else if (myVar == monthlyReferenceEvapotranspirationHS)
        return (obsDataM[i].et0_hs);
    else if (myVar == monthlyGlobalRadiation)
        return (obsDataM[i].globRad);
    else if (myVar == monthlyBIC)
        return (obsDataM[i].bic);
    else
        return NODATA;
}


float Crit3DMeteoPoint::getMeteoPointValue(const Crit3DTime& myTime, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings)
{
    frequencyType frequency = getVarFrequency(myVar);
    if (frequency == hourly)
        return getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar);
    else if (frequency == daily)
        return getMeteoPointValueD(myTime.date, myVar, meteoSettings);
    else
        return NODATA;
}

float Crit3DMeteoPoint::getProxyValue(unsigned pos)
{
    if (pos < proxyValues.size())
        return proxyValues[pos];
    else
        return NODATA;
}

std::vector <double> Crit3DMeteoPoint::getProxyValues()
{
    std::vector <double> myValues;
    for (unsigned int i=0; i < proxyValues.size(); i++)
        myValues.push_back(getProxyValue(i));

    return myValues;
}


bool Crit3DMeteoPoint::computeHourlyDerivedVar(const Crit3DTime &dateTime, meteoVariable myVar, bool useNetRad)
{
    const Crit3DDate myDate = dateTime.date;
    const int myHour = dateTime.getHour();
    float value = NODATA;

    if (myVar == leafWetness)
    {
        const float prec = getMeteoPointValueH(myDate, myHour, 0, precipitation);
        const float rh = getMeteoPointValueH(myDate, myHour, 0, airRelHumidity);

        short valueShort;
        if (! computeLeafWetness(prec, rh, valueShort))
            return false;

        value = float(valueShort);
    }
    else if (myVar == referenceEvapotranspiration)
    {
        const float tAir = getMeteoPointValueH(myDate, myHour, 0, airTemperature);
        const float rh = getMeteoPointValueH(myDate, myHour, 0, airRelHumidity);
        const float wind = getMeteoPointValueH(myDate, myHour, 0, windScalarIntensity);

        if (isEqual(tAir, NODATA) || isEqual(rh, NODATA) || isEqual(wind, NODATA))
            return false;

        if (useNetRad)
        {
            const float netRad = getMeteoPointValueH(myDate, myHour, 0, netIrradiance);
            if (isEqual(netRad, NODATA))
                return false;

            value = float(ET0_Penman_hourly_net_rad(double(point.z), double(netRad),
                                                    double(tAir), double(rh), double(wind)));
        }
        else
        {
            const float trans = getMeteoPointValueH(myDate, myHour, 0, atmTransmissivity);
            const float irr = getMeteoPointValueH(myDate, myHour, 0, globalIrradiance);

            if (isEqual(trans, NODATA) || isEqual(irr, NODATA))
                return false;

            value = float(ET0_Penman_hourly(double(point.z), double(trans / 0.75f), double(irr),
                                            double(tAir), double(rh), double(wind)));
        }
    }
    else
        return false;

    return setMeteoPointValueH(myDate, myHour, 0, myVar, value);
}


bool Crit3DMeteoPoint::computeDailyDerivedVar(const Crit3DDate &date, meteoVariable myVar, Crit3DMeteoSettings& meteoSettings)
{
    float value = NODATA;

    if (myVar == dailyReferenceEvapotranspirationHS)
    {
        value = dailyEtpHargreaves(getMeteoPointValueD(date, dailyAirTemperatureMin),
                               getMeteoPointValueD(date, dailyAirTemperatureMax),
                               date, latitude, &meteoSettings);

        setMeteoPointValueD(date, dailyReferenceEvapotranspirationHS, value);
    }

    if (myVar == dailyBIC)
    {
        const float et0 = dailyEtpHargreaves(getMeteoPointValueD(date, dailyAirTemperatureMin),
                                   getMeteoPointValueD(date, dailyAirTemperatureMax),
                                   date, latitude, &meteoSettings);

        if (! isEqual(et0, NODATA))
        {
            value = computeDailyBIC(getMeteoPointValueD(date, dailyPrecipitation), et0);
            setMeteoPointValueD(date, dailyBIC, value);
        }
    }

    const bool isComputed = !isEqual(value, NODATA);

    return isComputed;
}


bool Crit3DMeteoPoint::computeMonthlyAggregate(const Crit3DDate &firstDate, const Crit3DDate &lastDate, meteoVariable dailyMeteoVar,
                                               Crit3DMeteoSettings* meteoSettings, Crit3DQuality* qualityCheck,
                                               const Crit3DClimateParameters &climateParam)
{
    if (meteoSettings == nullptr || qualityCheck == nullptr)
        return false;

    const meteoVariable monthlyVar = updateMeteoVariable(dailyMeteoVar, monthly);
    if (monthlyVar == noMeteoVar)
        return false;

    const bool isSum = (dailyMeteoVar == dailyPrecipitation || dailyMeteoVar == dailyReferenceEvapotranspirationHS
                        || dailyMeteoVar == dailyReferenceEvapotranspirationPM || dailyMeteoVar == dailyGlobalRadiation
                        || dailyMeteoVar == dailyBIC);

    int currentMonth = firstDate.month;
    int nrDays = getDaysInMonth(currentMonth, firstDate.year);

    double sum = 0;
    int nrValid = 0;
    bool aggregated = false;

    for (Crit3DDate actualDate = firstDate; actualDate <= lastDate; ++actualDate)
    {
        const float dailyValue = getMeteoPointValueD(actualDate, dailyMeteoVar, meteoSettings);
        if (qualityCheck->checkFastValueDaily_SingleValue(dailyMeteoVar, climateParam, dailyValue,
                                                          currentMonth, float(point.z)) == quality::accepted)
        {
            sum += dailyValue;
            nrValid++;
        }

        if (actualDate.day == nrDays || actualDate == lastDate)
        {
            const float percentage = 100.0f * float(nrValid) / float(nrDays);

            float monthlyValue = NODATA;
            if (nrValid > 0 && percentage >= meteoSettings->getMinimumPercentage())
            {
                monthlyValue = isSum ? float(sum) : float(sum / nrValid);
                aggregated = true;
            }
            setMeteoPointValueM(actualDate, monthlyVar, monthlyValue);

            sum = 0;
            nrValid = 0;

            const Crit3DDate nextDate = actualDate.addDays(1);
            currentMonth = nextDate.month;
            nrDays = getDaysInMonth(currentMonth, nextDate.year);
        }
    }

    return aggregated;
}


bool Crit3DMeteoPoint::getDailyDataCsv_TPrec(std::string &outStr)
{
    if (obsDataD.empty())
        return false;

    outStr = "Date, Tmin (C), Tmax (C), Tavg (C), Prec (mm)\n";

    auto appendValue = [&outStr](float val)
    {
        if (! isEqual(val, NODATA))
        {
            std::ostringstream valueStream;
            valueStream << std::fixed << std::setprecision(1) << val;
            outStr += valueStream.str();
        }
    };

    for (const TObsDataD &obs : obsDataD)
    {
        outStr += obs.date.toISOString() + ",";

        appendValue(obs.tMin);  outStr += ",";
        appendValue(obs.tMax);  outStr += ",";
        appendValue(obs.tAvg);  outStr += ",";
        appendValue(obs.prec);
        outStr += "\n";
    }

    return true;
}


float Crit3DMeteoPoint::getPercValueVariable(const Crit3DDate &firstDate, const Crit3DDate &lastDate, meteoVariable dailyMeteoVar)
{
    Crit3DQuality qualityCheck;
    int nrValidValues = 0;
    int nrTotValues = 0;

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        nrTotValues++;

        const float value = getMeteoPointValueD(myDate, dailyMeteoVar);
        quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(dailyMeteoVar, value);

        if (qualityT == quality::accepted)
            nrValidValues++;
    }

    if (nrTotValues == 0)
        return 0.f;

    return float(nrValidValues) / float(nrTotValues);
}


// ---- end class


bool isSelectionPointsActive(const std::vector<Crit3DMeteoPoint> &meteoPoints)
{
    for (size_t i = 0; i < meteoPoints.size(); ++i)
    {
        if (meteoPoints[i].selected)
            return true;
    }

    return false;
}

