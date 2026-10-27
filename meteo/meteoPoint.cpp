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


Crit3DMeteoPoint::Crit3DMeteoPoint()
{
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

    this->cleanAllData();

    this->currentValue = NODATA;
    this->residual = NODATA;

    this->elaboration = NODATA;
    this->anomaly = NODATA;
    this->anomalyPercentage = NODATA;
    this->climate = NODATA;

    this->active = false;
    this->selected = false;
    this->marked = false;

    proxyValues.clear();
    lapseRateCode = primary;
    topographicDistance = nullptr;
    glocalWeights.clear();
}


void Crit3DMeteoPoint::setLapseRateCode(const std::string &lapseRateCode)
{
    if (lapseRateCode == "primary")
    {
        this->lapseRateCode = primary;
    }
    else if (lapseRateCode == "secondary")
    {
        this->lapseRateCode = secondary;
    }
    else if (lapseRateCode == "supplemental")
    {
        this->lapseRateCode = supplemental;
    }
}


void Crit3DMeteoPoint::initializeObsDataH(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate)
{
    cleanObsDataH();

    if (numberOfDays <= 0)
        return;

    nrObsDataDaysH = numberOfDays;
    hourlyFraction = myHourlyFraction;
    quality = quality::missing_data;
    residual = NODATA;

    unsigned int nrDailyValues = unsigned(hourlyFraction * 24);
    _obsDataH = new TObsDataH[unsigned(numberOfDays)];

    Crit3DDate myDate = firstDate;
    for (unsigned int i = 0; i < unsigned(numberOfDays); i++)
    {
        _obsDataH[i].date = myDate;
        _obsDataH[i].tAir = new float[nrDailyValues];
        _obsDataH[i].prec = new float[nrDailyValues];
        _obsDataH[i].rhAir = new float[nrDailyValues];
        _obsDataH[i].tDew = new float[nrDailyValues];
        _obsDataH[i].irradiance = new float[nrDailyValues];
        _obsDataH[i].netIrradiance = new float[nrDailyValues];
        _obsDataH[i].et0 = new float[nrDailyValues];
        _obsDataH[i].windVecX = new float[nrDailyValues];
        _obsDataH[i].windVecY = new float[nrDailyValues];
        _obsDataH[i].windVecInt = new float[nrDailyValues];
        _obsDataH[i].windVecDir = new float[nrDailyValues];
        _obsDataH[i].windScalInt = new float[nrDailyValues];
        _obsDataH[i].leafW = new int[nrDailyValues];
        _obsDataH[i].transmissivity = new float[nrDailyValues];
        _obsDataH[i].pressure = new float[nrDailyValues];

        for (unsigned int j = 0; j < nrDailyValues; j++)
        {
            _obsDataH[i].tAir[j] = NODATA;
            _obsDataH[i].prec[j] = NODATA;
            _obsDataH[i].rhAir[j] = NODATA;
            _obsDataH[i].tDew[j] = NODATA;
            _obsDataH[i].irradiance[j] = NODATA;
            _obsDataH[i].netIrradiance[j] = NODATA;
            _obsDataH[i].et0[j] = NODATA;
            _obsDataH[i].windVecX[j] = NODATA;
            _obsDataH[i].windVecY[j] = NODATA;
            _obsDataH[i].windVecInt[j] = NODATA;
            _obsDataH[i].windVecDir[j] = NODATA;
            _obsDataH[i].windScalInt[j] = NODATA;
            _obsDataH[i].leafW[j] = NODATA;
            _obsDataH[i].transmissivity[j] = NODATA;
            _obsDataH[i].pressure[j] = NODATA;
        }
        ++myDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataHFromMp(int myHourlyFraction, int numberOfDays,
                                                const Crit3DDate& firstDate, const Crit3DMeteoPoint& meteoPoint)
{
    // this must be already allocated with the same time resolution:
    // different sizes would cause buffer overflows
    if (_obsDataH == nullptr || meteoPoint._obsDataH == nullptr)
        return;
    if (myHourlyFraction != hourlyFraction || meteoPoint.hourlyFraction != this->hourlyFraction)
        return;

    int nrDays = std::min({numberOfDays, nrObsDataDaysH, meteoPoint.nrObsDataDaysH});

    const size_t nrDailyValues = size_t(hourlyFraction) * 24;
    const bool isSelf = (&meteoPoint == this);

    Crit3DDate myDate = firstDate;
    for (int i = 0; i < nrDays; i++, ++myDate)
    {
        TObsDataH& dst = _obsDataH[i];
        dst.date = myDate;

        if (isSelf)
            continue;   // only the dates change, copy onto itself is undefined

        const TObsDataH& src = meteoPoint._obsDataH[i];

        std::copy_n(src.tAir, nrDailyValues, dst.tAir);
        std::copy_n(src.prec, nrDailyValues, dst.prec);
        std::copy_n(src.rhAir, nrDailyValues, dst.rhAir);
        std::copy_n(src.tDew, nrDailyValues, dst.tDew);
        std::copy_n(src.irradiance, nrDailyValues, dst.irradiance);
        std::copy_n(src.netIrradiance, nrDailyValues, dst.netIrradiance);
        std::copy_n(src.et0, nrDailyValues, dst.et0);
        std::copy_n(src.windVecX, nrDailyValues, dst.windVecX);
        std::copy_n(src.windVecY, nrDailyValues, dst.windVecY);
        std::copy_n(src.windVecInt, nrDailyValues, dst.windVecInt);
        std::copy_n(src.windVecDir, nrDailyValues, dst.windVecDir);
        std::copy_n(src.windScalInt, nrDailyValues, dst.windScalInt);
        std::copy_n(src.leafW, nrDailyValues, dst.leafW);
        std::copy_n(src.transmissivity, nrDailyValues, dst.transmissivity);
        std::copy_n(src.pressure, nrDailyValues, dst.pressure);
    }
}


void Crit3DMeteoPoint::initializeObsDataD(unsigned int numberOfDays, const Crit3DDate& firstDate)
{
    obsDataD.clear();
    obsDataD.resize(numberOfDays);
    nrObsDataDaysD = int(numberOfDays);

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


void Crit3DMeteoPoint::initializeObsDataDFromMp(unsigned int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint &meteoPoint)
{
    initializeObsDataD(numberOfDays, firstDate);

    Crit3DDate currentDate = firstDate;
    const size_t nrRows = std::min(size_t(numberOfDays), meteoPoint.obsDataD.size());

    for (size_t i = 0; i < nrRows; i++)
    {
        obsDataD[i] = meteoPoint.obsDataD[i];
        obsDataD[i].date = currentDate;
        ++currentDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataM(unsigned int numberOfMonths, unsigned int month, int year)
{
    obsDataM.clear();
    obsDataM.resize(numberOfMonths);
    nrObsDataDaysM = numberOfMonths;

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
    int i = _obsDataH[0].date.daysTo(myDate);
    residual = NODATA;

    if (i >= 0 && i < nrObsDataDaysH)
        if (_obsDataH[i].date == myDate)
            for (int j = 0; j < nrDayValues; j++)
            {
                if (myVar == airTemperature)
                    _obsDataH[i].tAir[j] = NODATA;
                else if (myVar == precipitation)
                    _obsDataH[i].prec[j] = NODATA;
                else if (myVar == airRelHumidity)
                    _obsDataH[i].rhAir[j] = NODATA;
                else if (myVar == airDewTemperature)
                    _obsDataH[i].tDew[j] = NODATA;
                else if (myVar == globalIrradiance)
                    _obsDataH[i].irradiance[j] = NODATA;
                else if (myVar == netIrradiance)
                    _obsDataH[i].netIrradiance[j] = NODATA;
                else if (myVar == windScalarIntensity)
                    _obsDataH[i].windScalInt[j] = NODATA;
                else if (myVar == windVectorX)
                    _obsDataH[i].windVecX[j] = NODATA;
                else if (myVar == windVectorY)
                    _obsDataH[i].windVecY[j] = NODATA;
                else if (myVar == windVectorIntensity)
                    _obsDataH[i].windVecInt[j] = NODATA;
                else if (myVar == windVectorDirection)
                    _obsDataH[i].windVecDir[j] = NODATA;
                else if (myVar == leafWetness)
                    _obsDataH[i].leafW[j] = NODATA;
                else if (myVar == atmTransmissivity)
                    _obsDataH[i].transmissivity[j] = NODATA;
				else if (myVar == referenceEvapotranspiration)
                    _obsDataH[i].et0[j] = NODATA;
                else if (myVar == atmPressure)
                    _obsDataH[i].pressure[j] = NODATA;
			}
}

void Crit3DMeteoPoint::emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int nrDayValues = hourlyFraction * 24;
    int indexIni = _obsDataH[0].date.daysTo(date1);
    int indexFin = _obsDataH[0].date.daysTo(date2);
    residual = NODATA;

    for (int i = indexIni; i <= indexFin; i++)
        for (int j = 0; j < nrDayValues; j++)
        {
            if (myVar == airTemperature)
                _obsDataH[i].tAir[j] = NODATA;
            else if (myVar == precipitation)
                _obsDataH[i].prec[j] = NODATA;
            else if (myVar == airRelHumidity)
                _obsDataH[i].rhAir[j] = NODATA;
            else if (myVar == airDewTemperature)
                _obsDataH[i].tDew[j] = NODATA;
            else if (myVar == globalIrradiance)
                _obsDataH[i].irradiance[j] = NODATA;
            else if (myVar == netIrradiance)
                _obsDataH[i].netIrradiance[j] = NODATA;
            else if (myVar == windScalarIntensity)
                _obsDataH[i].windScalInt[j] = NODATA;
            else if (myVar == windVectorX)
                _obsDataH[i].windVecX[j] = NODATA;
            else if (myVar == windVectorY)
                _obsDataH[i].windVecY[j] = NODATA;
            else if (myVar == windVectorIntensity)
                _obsDataH[i].windVecInt[j] = NODATA;
            else if (myVar == windVectorDirection)
                _obsDataH[i].windVecDir[j] = NODATA;
            else if (myVar == leafWetness)
                _obsDataH[i].leafW[j] = NODATA;
            else if (myVar == atmTransmissivity)
                _obsDataH[i].transmissivity[j] = NODATA;
			else if (myVar == referenceEvapotranspiration)
                _obsDataH[i].et0[j] = NODATA;
            else if (myVar == atmPressure)
                _obsDataH[i].pressure[j] = NODATA;
        }
}

void Crit3DMeteoPoint::emptyObsDataH(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int nrDayValues = hourlyFraction * 24;
    int indexIni = _obsDataH[0].date.daysTo(date1);
    int indexFin = _obsDataH[0].date.daysTo(date2);

    for (int i = indexIni; i <= indexFin; i++)
        for (int j = 0; j < nrDayValues; j++)
        {
            _obsDataH[i].tAir[j] = NODATA;
            _obsDataH[i].prec[j] = NODATA;
            _obsDataH[i].rhAir[j] = NODATA;
            _obsDataH[i].tDew[j] = NODATA;
            _obsDataH[i].irradiance[j] = NODATA;
            _obsDataH[i].netIrradiance[j] = NODATA;
            _obsDataH[i].windScalInt[j] = NODATA;
            _obsDataH[i].windVecX[j] = NODATA;
            _obsDataH[i].windVecY[j] = NODATA;
            _obsDataH[i].windVecInt[j] = NODATA;
            _obsDataH[i].windVecDir[j] = NODATA;
            _obsDataH[i].leafW[j] = NODATA;
            _obsDataH[i].transmissivity[j] = NODATA;
            _obsDataH[i].et0[j] = NODATA;
            _obsDataH[i].pressure[j] = NODATA;
        }
}


void Crit3DMeteoPoint::emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedD(date1, date2))
        return;

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
    default:                                    return;
    }

    const int indexIni = obsDataD[0].date.daysTo(date1);
    const int indexFin = obsDataD[0].date.daysTo(date2);

    for (int i = indexIni; i <= indexFin; i++)
        obsDataD[unsigned(i)].*member = NODATA;

    residual = NODATA;
}


void Crit3DMeteoPoint::emptyObsDataD(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedD(date1, date2))
        return;

    int indexIni = obsDataD[0].date.daysTo(date1);
    int indexFin = obsDataD[0].date.daysTo(date2);

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
}


void Crit3DMeteoPoint::emptyObsDataM(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedM(date1, date2))
        return;

    int indexIni = (date1.year - obsDataM[0]._year)*12 + date1.month-obsDataM[0]._month;
    int indexFin = (date2.year - obsDataM[0]._year)*12 + date2.month-obsDataM[0]._month;

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
}


bool Crit3DMeteoPoint::isDateLoadedH(const Crit3DDate& myDate)
{
    if (nrObsDataDaysH == 0)
        return false;
    else if (myDate < _obsDataH[0].date || myDate > _obsDataH[nrObsDataDaysH - 1].date)
        return false;
    else
        return true;
}


bool Crit3DMeteoPoint::isDateTimeLoadedH(const Crit3DTime& myDateTime)
{
    if (nrObsDataDaysH == 0)
        return false;
    else if (myDateTime < Crit3DTime(_obsDataH[0].date,1) || myDateTime >= Crit3DTime(_obsDataH[nrObsDataDaysH - 1].date,1))
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedH(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (nrObsDataDaysH == 0)
        return false;
    else if (date1 > date2)
        return false;
    else if (date1 < _obsDataH[0].date || date2 > _obsDataH[nrObsDataDaysH - 1].date)
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateLoadedD(const Crit3DDate& myDate)
{
    if (nrObsDataDaysD == 0)
        return false;
    else if (myDate < obsDataD[0].date || myDate > obsDataD[unsigned(nrObsDataDaysD-1)].date)
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateLoadedM(const Crit3DDate& myDate)
{
    if (nrObsDataDaysM == 0)
        return false;
    else if ( (myDate.year < obsDataM[0]._year ) || (myDate.year > obsDataM[unsigned(nrObsDataDaysM-1)]._year))
        return false;
    else if (myDate.year == obsDataM[0]._year && myDate.month < obsDataM[0]._month)
        return false;
    else if (myDate.year == obsDataM[unsigned(nrObsDataDaysM-1)]._year && myDate.month > obsDataM[unsigned(nrObsDataDaysM-1)]._month)
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (nrObsDataDaysD == 0)
        return false;
    else if (date1 > date2)
        return false;
    else if (date1 < obsDataD[0].date || date2 > obsDataD[unsigned(nrObsDataDaysD-1)].date)
        return false;
    else
        return (true);
}

bool Crit3DMeteoPoint::isDateIntervalLoadedM(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (nrObsDataDaysM == 0)
        return false;
    else if (date1 > date2)
        return false;
    else if ( (date1.year < obsDataM[0]._year ) || (date2.year > obsDataM[unsigned(nrObsDataDaysM-1)]._year))
        return false;
    else if (date1.year == obsDataM[0]._year && date1.month < obsDataM[0]._month)
        return false;
    else if (date2.year == obsDataM[unsigned(nrObsDataDaysM-1)]._year && date2.month > obsDataM[unsigned(nrObsDataDaysM-1)]._month)
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateIntervalLoadedH(const Crit3DTime& timeIni, const Crit3DTime& timeFin)
{
    if (nrObsDataDaysH == 0)
        return false;
    else if (timeIni > timeFin)
        return false;
    else if (_obsDataH == nullptr)
        return false;
    else if (timeIni.date < _obsDataH[0].date || timeFin.date > (_obsDataH[0].date.addDays(nrObsDataDaysH - 1)))
        return (false);
    else
        return (true);
}

float Crit3DMeteoPoint::obsDataConsistencyH(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin)
{
    if (nrObsDataDaysH == 0)
        return 0.0;
    else if (timeIni > timeFin)
        return 0.0;
    else if (_obsDataH == nullptr)
        return 0.0;
    else if (timeFin.date < _obsDataH[0].date || timeIni.date > (_obsDataH[0].date.addDays(nrObsDataDaysH - 1)))
        return 0.0;
    else
    {
        Crit3DTime myTime = timeIni;
        float myValue;
        int deltaSeconds = 3600 / hourlyFraction;
        int counter=0, counterAll=0;
        while (myTime <= timeFin)
        {
            myValue = getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar);
            if (! isEqual(myValue, NODATA) )
                counter++;

            counterAll++;
            myTime = myTime.addSeconds(deltaSeconds);
        }
        return (float(counter)/float(counterAll));
    }

}

void Crit3DMeteoPoint::cleanObsDataH()
{
    if (_obsDataH != nullptr && nrObsDataDaysH > 0)
    {
        for (int i = 0; i < nrObsDataDaysH; i++)
        {
            delete [] _obsDataH[i].tAir;
            delete [] _obsDataH[i].prec;
            delete [] _obsDataH[i].rhAir;
            delete [] _obsDataH[i].tDew;
            delete [] _obsDataH[i].irradiance;
            delete [] _obsDataH[i].netIrradiance;
            delete [] _obsDataH[i].windScalInt;
            delete [] _obsDataH[i].windVecX;
            delete [] _obsDataH[i].windVecY;
            delete [] _obsDataH[i].windVecInt;
            delete [] _obsDataH[i].windVecDir;
            delete [] _obsDataH[i].leafW;
            delete [] _obsDataH[i].transmissivity;
            delete [] _obsDataH[i].et0;
            delete [] _obsDataH[i].pressure;
        }
        delete [] _obsDataH;
    }

    nrObsDataDaysH = 0;
    _obsDataH = nullptr;
}


void Crit3DMeteoPoint::cleanAllData()
{
    cleanObsDataH();

    obsDataD.clear();
    nrObsDataDaysD = 0;

    obsDataM.clear();
    nrObsDataDaysM = 0;

    quality = quality::missing_data;
}


bool Crit3DMeteoPoint::setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue)
{
    if (myVar == noMeteoVar || _obsDataH == nullptr || nrObsDataDaysH <= 0)
        return false;

    // day index
    int iDay = _obsDataH[0].date.daysTo(myDate);

    // out of range (accept +1 day exceed)
    if (iDay < 0 || iDay > nrObsDataDaysH)
        return false;

    // sub-hourly index: integer ceil(myMinutes / minutesPerStep)
    const int minutesPerStep = 60 / hourlyFraction;
    const int subH = (myMinutes + minutesPerStep - 1) / minutesPerStep;

    // if +1 day exceed, accept only 00:00
    if (iDay == nrObsDataDaysH && (myHour != 0 || subH != 0))
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

    TObsDataH& obs = _obsDataH[iDay];

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
    if (myVar == noMeteoVar || nrObsDataDaysD <= 0 || obsDataD.empty())
        return false;

    long index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= nrObsDataDaysD))
        return false;

    unsigned i = unsigned(index);

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
    //check
    if (myVar == noMeteoVar)
        return false;
    if (nrObsDataDaysM == 0)
        return false;

    const int index = (myDate.year - obsDataM[0]._year) * 12 + (myDate.month - obsDataM[0]._month);

    if ((index < 0) || (index >= nrObsDataDaysM))
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
    if (myVar == noMeteoVar || _obsDataH == nullptr || nrObsDataDaysH <= 0)
        return NODATA;

    // day index
    int iDay = _obsDataH[0].date.daysTo(myDate);

    // out of range (accept +1 day exceed)
    if (iDay < 0 || iDay > nrObsDataDaysH)
        return NODATA;

    // sub-hourly index: integer ceil(myMinutes / minutesPerStep)
    const int minutesPerStep = 60 / hourlyFraction;
    const int subH = (myMinutes + minutesPerStep - 1) / minutesPerStep;

    // if +1 day exceed, accept only 00:00
    if (iDay == nrObsDataDaysH && (myHour != 0 || subH != 0))
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

    const TObsDataH& obs = _obsDataH[iDay];

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
    if (_obsDataH == nullptr || index < 0 || index >= nrObsDataDaysH)
        return NO_DATE;

    return _obsDataH[index].date;
}


bool Crit3DMeteoPoint::getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH* &hourlyValues)
{
    if (_obsDataH == nullptr || nrObsDataDaysH <= 0)
        return false;

    int d = _obsDataH[0].date.daysTo(myDate);
    if (d < 0 || d >= nrObsDataDaysH)
        return false;

    hourlyValues = &(_obsDataH[d]);

    return true;
}


bool Crit3DMeteoPoint::existDailyData(const Crit3DDate& myDate)
{
    if (obsDataD.size() == 0)
        return false;

    int index = obsDataD[0].date.daysTo(myDate);

    if ((index < 0) || (index >= nrObsDataDaysD))
        return false;
    else
        return true;
}


Crit3DDate Crit3DMeteoPoint::getLastDailyData() const
{
    if (obsDataD.size() == 0)
        return NO_DATE;

    return obsDataD[nrObsDataDaysD-1].date;
}


Crit3DDate Crit3DMeteoPoint::getFirstDailyData() const
{
    if (obsDataD.size() == 0)
        return NO_DATE;

    return obsDataD[0].date;
}


float Crit3DMeteoPoint::getDailyET0_HS(const TObsDataD &dailyData, const Crit3DDate &myDate, Crit3DMeteoSettings* meteoSettings) const
{
    float et0 = dailyData.et0_hs;
    if (isEqual(et0, NODATA) && meteoSettings->getAutomaticET0HS())
    {
        if (! isEqual(dailyData.tMin, NODATA) && ! isEqual(dailyData.tMax, NODATA))
        {
            et0 = float(ET0_Hargreaves(meteoSettings->getTransSamaniCoefficient(),
                                       latitude, getDoyFromDate(myDate),
                                       dailyData.tMax, dailyData.tMin));
        }
    }

    return et0;
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings) const
{
    if (myVar == noMeteoVar)
        return NODATA;
    if (nrObsDataDaysD == 0 || obsDataD.empty())
        return NODATA;

    const int index = obsDataD[0].date.daysTo(myDate);
    if (index < 0 || index >= nrObsDataDaysD || index >= int(obsDataD.size()))
        return NODATA;

    const TObsDataD& obs = obsDataD[unsigned(index)];

    switch (myVar)
    {
    case dailyAirTemperatureMax:                return obs.tMax;
    case dailyAirTemperatureMin:                return obs.tMin;

    case dailyAirTemperatureAvg:
    {
        if (! isEqual(obs.tAvg, NODATA))
            return obs.tAvg;

        if (meteoSettings != nullptr && meteoSettings->getAutomaticTavg()
            && ! isEqual(obs.tMin, NODATA) && ! isEqual(obs.tMax, NODATA))
        {
            return (obs.tMin + obs.tMax) * 0.5f;
        }
        return NODATA;
    }

    case dailyPrecipitation:                    return obs.prec;
    case dailyAirRelHumidityMax:                return obs.rhMax;
    case dailyAirRelHumidityMin:                return obs.rhMin;
    case dailyAirRelHumidityAvg:                return obs.rhAvg;
    case dailyGlobalRadiation:                  return obs.globRad;

    case dailyReferenceEvapotranspirationPM:    return obs.et0_pm;
    case dailyHeatingDegreeDays:                return obs.dd_heating;
    case dailyCoolingDegreeDays:                return obs.dd_cooling;
    case dailyWindScalarIntensityAvg:           return obs.windScalIntAvg;
    case dailyWindScalarIntensityMax:           return obs.windScalIntMax;
    case dailyWindVectorIntensityAvg:           return obs.windVecIntAvg;
    case dailyWindVectorIntensityMax:           return obs.windVecIntMax;
    case dailyWindVectorDirectionPrevailing:    return obs.windVecDirPrev;
    case dailyLeafWetness:                      return obs.leafW;
    case dailyWaterTableDepth:                  return obs.waterTable;

    case dailyReferenceEvapotranspirationHS:
        return getDailyET0_HS(obs, myDate, meteoSettings);

    case dailyBIC:
    {
        if (! isEqual(obs.bic, NODATA))
            return obs.bic;

        if (isEqual(obs.prec, NODATA))
            return NODATA;

        const float et0 = getDailyET0_HS(obs, myDate, meteoSettings);
        if (isEqual(et0, NODATA))
            return NODATA;

        return computeDailyBIC(obs.prec, et0);
    }

    default:
        return NODATA;
    }
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar) const
{
    if (myVar == noMeteoVar)
        return NODATA;
    if (nrObsDataDaysD == 0 || obsDataD.empty())
        return NODATA;

    const int index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= nrObsDataDaysD))
        return NODATA;

    const unsigned i = unsigned(index);

    if (myVar == dailyAirTemperatureMax)
        return (obsDataD[i].tMax);
    else if (myVar == dailyAirTemperatureMin)
        return (obsDataD[i].tMin);
    else if (myVar == dailyAirTemperatureAvg)
        return obsDataD[i].tAvg;
    else if (myVar == dailyPrecipitation)
        return (obsDataD[i].prec);
    else if (myVar == dailyAirRelHumidityMax)
        return (obsDataD[i].rhMax);
    else if (myVar == dailyAirRelHumidityMin)
        return float(obsDataD[i].rhMin);
    else if (myVar == dailyAirRelHumidityAvg)
        return (obsDataD[i].rhAvg);
    else if (myVar == dailyGlobalRadiation)
        return (obsDataD[i].globRad);
    else if (myVar == dailyReferenceEvapotranspirationHS)
        return obsDataD[i].et0_hs;
    else if (myVar == dailyReferenceEvapotranspirationPM)
        return (obsDataD[i].et0_pm);
    else if (myVar == dailyBIC)
        return obsDataD[i].bic;
    else if (myVar == dailyHeatingDegreeDays)
        return obsDataD[i].dd_heating;
    else if (myVar == dailyCoolingDegreeDays)
        return obsDataD[i].dd_cooling;
    else if (myVar == dailyWindScalarIntensityAvg)
        return (obsDataD[i].windScalIntAvg);
    else if (myVar == dailyWindScalarIntensityMax)
        return (obsDataD[i].windScalIntMax);
    else if (myVar == dailyWindVectorIntensityAvg)
        return (obsDataD[i].windVecIntAvg);
    else if (myVar == dailyWindVectorIntensityMax)
        return (obsDataD[i].windVecIntMax);
    else if (myVar == dailyWindVectorDirectionPrevailing)
        return (obsDataD[i].windVecDirPrev);
    else if (myVar == dailyLeafWetness)
        return (obsDataD[i].leafW);
    else if (myVar == dailyWaterTableDepth)
        return (obsDataD[i].waterTable);
    else
        return NODATA;
}

float Crit3DMeteoPoint::getMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar) const
{
    //check
    if (myVar == noMeteoVar)
        return NODATA;
    if (nrObsDataDaysM == 0)
        return NODATA;

    const int index = (myDate.year - obsDataM[0]._year) * 12 + (myDate.month - obsDataM[0]._month);

    if ((index < 0) || (index >= nrObsDataDaysM))
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
    float value;

    if (myVar == dailyReferenceEvapotranspirationHS)
    {
        value = dailyEtpHargreaves(getMeteoPointValueD(date, dailyAirTemperatureMin),
                               getMeteoPointValueD(date, dailyAirTemperatureMax),
                               date, latitude, &meteoSettings);

        setMeteoPointValueD(date, dailyReferenceEvapotranspirationHS, value);
    }

    return true;
}


bool Crit3DMeteoPoint::computeMonthlyAggregate(const Crit3DDate &firstDate, const Crit3DDate &lastDate, meteoVariable dailyMeteoVar,
                                               Crit3DMeteoSettings* meteoSettings, Crit3DQuality* qualityCheck,
                                               Crit3DClimateParameters* climateParam)
{
    if (meteoSettings == nullptr || qualityCheck == nullptr || climateParam == nullptr)
        return false;

    int currentMonth = firstDate.month;
    int nrDays = getDaysInMonth(currentMonth, firstDate.year);

    double sum = 0;
    int nrValid = 0;
    int indexMonth = 0;
    bool aggregateDailyInMonthly = false;

    for (Crit3DDate actualDate = firstDate; actualDate <= lastDate; ++actualDate)
    {
        float dailyValue = getMeteoPointValueD(actualDate, dailyMeteoVar, meteoSettings);
        quality::qualityType qualityT = qualityCheck->checkFastValueDaily_SingleValue(dailyMeteoVar, climateParam,
                                                                                      dailyValue, currentMonth, float(point.z));
        if (qualityT == quality::accepted)
        {
            sum += dailyValue;
            nrValid++;
        }
        if (actualDate.day == nrDays || actualDate == lastDate)
        {
            indexMonth++;
            const float dataPercentage = (float(nrValid) / float(nrDays)) * 100.0f;
            if (dataPercentage >= meteoSettings->getMinimumPercentage())
            {
                aggregateDailyInMonthly = true;
                if (dailyMeteoVar == dailyAirTemperatureMin || dailyMeteoVar == dailyAirTemperatureMax
                    || dailyMeteoVar == dailyAirTemperatureAvg)
                {
                    if (nrValid > 0)
                    {
                        setMeteoPointValueM(actualDate, updateMeteoVariable(dailyMeteoVar, monthly), float(sum / nrValid));
                    }
                    else
                    {
                        setMeteoPointValueM(actualDate,updateMeteoVariable(dailyMeteoVar, monthly), NODATA);
                    }
                }
                else if (dailyMeteoVar == dailyPrecipitation || dailyMeteoVar == dailyReferenceEvapotranspirationHS
                         || dailyMeteoVar == dailyReferenceEvapotranspirationPM || dailyMeteoVar == dailyGlobalRadiation
                         || dailyMeteoVar == dailyBIC)
                {
                    setMeteoPointValueM(actualDate, updateMeteoVariable(dailyMeteoVar, monthly), float(sum));
                }
                else if (updateMeteoVariable(dailyMeteoVar, monthly) != noMeteoVar)
                {
                    if (nrValid > 0)
                    {
                        // default: average
                        setMeteoPointValueM(actualDate, updateMeteoVariable(dailyMeteoVar, monthly), float(sum / nrValid));
                    }
                    else
                    {
                        setMeteoPointValueM(actualDate, updateMeteoVariable(dailyMeteoVar, monthly), NODATA);
                    }
                }
            }
            else
            {
                setMeteoPointValueM(actualDate,updateMeteoVariable(dailyMeteoVar, monthly), NODATA);
            }

            sum = 0;
            nrValid = 0;
            Crit3DDate nextDate = actualDate.addDays(1);
            currentMonth = nextDate.month;
            nrDays = getDaysInMonth(currentMonth, nextDate.year);
        }
    }

    return aggregateDailyInMonthly;
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
    for (int i = 0; i < (int)meteoPoints.size(); i++)
    {
        if (meteoPoints[i].selected)
            return true;
    }

    return false;
}
