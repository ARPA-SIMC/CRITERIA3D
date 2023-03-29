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
#include "meteoPoint.h"
#include "quality.h"


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

    // Tpoint
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

    this->nrObsDataDaysH = 0;
    this->nrObsDataDaysD = 0;
    this->nrObsDataDaysM = 0;
    this->hourlyFraction = 1;

    this->obsDataH = nullptr;

    this->currentValue = NODATA;
    this->residual = NODATA;

    this->elaboration = NODATA;
    this->anomaly = NODATA;
    this->anomalyPercentage = NODATA;
    this->climate = NODATA;

    this->active = false;
    this->selected = false;
    this->marked = false;

    this->quality = quality::missing_data;

    proxyValues.clear();
    lapseRateCode = primary;
    topographicDistance = nullptr;
}

void Crit3DMeteoPoint::setId(std::string value)
{
    this->id = value;
}

void Crit3DMeteoPoint::setName(std::string name)
{
    this->name = name;
}

void Crit3DMeteoPoint::initializeObsDataH(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate)
{
    this->cleanObsDataH();

    nrObsDataDaysH = numberOfDays;
    hourlyFraction = myHourlyFraction;
    quality = quality::missing_data;
    residual = NODATA;

    unsigned int nrDailyValues = unsigned(hourlyFraction * 24);
    obsDataH = new TObsDataH[unsigned(numberOfDays)];

    Crit3DDate myDate = firstDate;
    for (unsigned int i = 0; i < unsigned(numberOfDays); i++)
    {
        obsDataH[i].date = myDate;
        obsDataH[i].tAir = new float[nrDailyValues];
        obsDataH[i].prec = new float[nrDailyValues];
        obsDataH[i].rhAir = new float[nrDailyValues];
        obsDataH[i].tDew = new float[nrDailyValues];
        obsDataH[i].irradiance = new float[nrDailyValues];
        obsDataH[i].netIrradiance = new float[nrDailyValues];
        obsDataH[i].et0 = new float[nrDailyValues];
        obsDataH[i].windVecX = new float[nrDailyValues];
        obsDataH[i].windVecY = new float[nrDailyValues];
        obsDataH[i].windVecInt = new float[nrDailyValues];
        obsDataH[i].windVecDir = new float[nrDailyValues];
        obsDataH[i].windScalInt = new float[nrDailyValues];
        obsDataH[i].leafW = new int[nrDailyValues];
        obsDataH[i].transmissivity = new float[nrDailyValues];

        for (unsigned int j = 0; j < nrDailyValues; j++)
        {
            obsDataH[i].tAir[j] = NODATA;
            obsDataH[i].prec[j] = NODATA;
            obsDataH[i].rhAir[j] = NODATA;
            obsDataH[i].tDew[j] = NODATA;
            obsDataH[i].irradiance[j] = NODATA;
            obsDataH[i].netIrradiance[j] = NODATA;
            obsDataH[i].et0[j] = NODATA;
            obsDataH[i].windVecX[j] = NODATA;
            obsDataH[i].windVecY[j] = NODATA;
            obsDataH[i].windVecInt[j] = NODATA;
            obsDataH[i].windVecDir[j] = NODATA;
            obsDataH[i].windScalInt[j] = NODATA;
            obsDataH[i].leafW[j] = NODATA;
            obsDataH[i].transmissivity[j] = NODATA;
        }
        ++myDate;
    }
}

void Crit3DMeteoPoint::initializeObsDataHFromMp(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint mp)
{

    hourlyFraction = myHourlyFraction;
    unsigned int nrDailyValues = unsigned(hourlyFraction * 24);
    Crit3DDate myDate = firstDate;
    TObsDataH *data = mp.getObsDataH();
    for (unsigned int i = 0; i < unsigned(numberOfDays); i++)
    {
        if (i < unsigned(nrObsDataDaysH))
        {
            obsDataH[i].date = myDate;
            for (unsigned int j = 0; j < nrDailyValues; j++)
            {
                obsDataH[i].tAir[j] = data[i].tAir[j];
                obsDataH[i].prec[j] = data[i].prec[j];
                obsDataH[i].rhAir[j] = data[i].rhAir[j];
                obsDataH[i].tDew[j] = data[i].tDew[j];
                obsDataH[i].irradiance[j] = data[i].irradiance[j];
                obsDataH[i].netIrradiance[j] =data[i].netIrradiance[j];
                obsDataH[i].et0[j] = data[i].et0[j];
                obsDataH[i].windVecX[j] = data[i].windVecX[j];
                obsDataH[i].windVecY[j] = data[i].windVecY[j];
                obsDataH[i].windVecInt[j] = data[i].windVecInt[j];
                obsDataH[i].windVecDir[j] = data[i].windVecDir[j];
                obsDataH[i].windScalInt[j] = data[i].windScalInt[j];
                obsDataH[i].leafW[j] = data[i].leafW[j];
                obsDataH[i].transmissivity[j] = data[i].transmissivity[j];
            }
            ++myDate;
        }
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

void Crit3DMeteoPoint::initializeObsDataDFromMp(unsigned int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint mp)
{
    Crit3DDate myDate = firstDate;
    for (unsigned int i = 0; i < numberOfDays; i++)
    {
        obsDataD[i].date = myDate;
        obsDataD[i].tMax = mp.obsDataD[i].tMax;
        obsDataD[i].tMin = mp.obsDataD[i].tMin;
        obsDataD[i].tAvg = mp.obsDataD[i].tAvg;
        obsDataD[i].prec = mp.obsDataD[i].prec;
        obsDataD[i].rhMax = mp.obsDataD[i].rhMax;
        obsDataD[i].rhMin = mp.obsDataD[i].rhMin;
        obsDataD[i].rhAvg = mp.obsDataD[i].rhAvg;
        obsDataD[i].globRad = mp.obsDataD[i].globRad;
        obsDataD[i].et0_hs = mp.obsDataD[i].et0_hs;
        obsDataD[i].et0_pm = mp.obsDataD[i].et0_pm;
        obsDataD[i].dd_heating = mp.obsDataD[i].dd_heating;
        obsDataD[i].dd_cooling = mp.obsDataD[i].dd_cooling;
        obsDataD[i].windVecIntAvg = mp.obsDataD[i].windVecIntAvg;
        obsDataD[i].windVecIntMax = mp.obsDataD[i].windVecIntMax;
        obsDataD[i].windVecDirPrev = mp.obsDataD[i].windVecDirPrev;
        obsDataD[i].windScalIntAvg = mp.obsDataD[i].windScalIntAvg;
        obsDataD[i].windScalIntMax = mp.obsDataD[i].windScalIntMax;
        obsDataD[i].leafW = mp.obsDataD[i].leafW;
        obsDataD[i].waterTable = mp.obsDataD[i].waterTable;
        ++myDate;
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
                addYear = addYear + 1;
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

    if (i >= 0 && i < nrObsDataDaysH)
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
        }
}

void Crit3DMeteoPoint::emptyObsDataH(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int nrDayValues = hourlyFraction * 24;
    int indexIni = obsDataH[0].date.daysTo(date1);
    int indexFin = obsDataH[0].date.daysTo(date2);

    for (int i = indexIni; i <= indexFin; i++)
        for (int j = 0; j < nrDayValues; j++)
        {
            obsDataH[i].tAir[j] = NODATA;
            obsDataH[i].prec[j] = NODATA;
            obsDataH[i].rhAir[j] = NODATA;
            obsDataH[i].tDew[j] = NODATA;
            obsDataH[i].irradiance[j] = NODATA;
            obsDataH[i].netIrradiance[j] = NODATA;
            obsDataH[i].windScalInt[j] = NODATA;
            obsDataH[i].windVecX[j] = NODATA;
            obsDataH[i].windVecY[j] = NODATA;
            obsDataH[i].windVecInt[j] = NODATA;
            obsDataH[i].windVecDir[j] = NODATA;
            obsDataH[i].leafW[j] = NODATA;
            obsDataH[i].transmissivity[j] = NODATA;
            obsDataH[i].et0[j] = NODATA;
        }
}

void Crit3DMeteoPoint::emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int indexIni = obsDataD[0].date.daysTo(date1);
    int indexFin = obsDataD[0].date.daysTo(date2);
    residual = NODATA;

    for (unsigned int i = indexIni; i <= unsigned(indexFin); i++)
        if (myVar == dailyAirTemperatureMax)
            obsDataD[i].tMax = NODATA;
        else if (myVar == dailyAirTemperatureMin)
            obsDataD[i].tMin = NODATA;
        else if (myVar == dailyAirTemperatureAvg)
            obsDataD[i].tAvg = NODATA;
        else if (myVar == dailyPrecipitation)
            obsDataD[i].prec = NODATA;
        else if (myVar == dailyAirRelHumidityMax)
            obsDataD[i].rhMax = NODATA;
        else if (myVar == dailyAirRelHumidityMin)
            obsDataD[i].rhMin = NODATA;
        else if (myVar == dailyAirRelHumidityAvg)
            obsDataD[i].rhAvg = NODATA;
        else if (myVar == dailyGlobalRadiation)
            obsDataD[i].globRad = NODATA;
        else if (myVar == dailyWindScalarIntensityAvg)
            obsDataD[i].windScalIntAvg = NODATA;
        else if (myVar == dailyWindScalarIntensityMax)
            obsDataD[i].windScalIntMax = NODATA;
        else if (myVar == dailyWindVectorIntensityAvg)
            obsDataD[i].windVecIntAvg = NODATA;
        else if (myVar == dailyWindVectorIntensityMax)
            obsDataD[i].windVecIntMax = NODATA;
        else if (myVar == dailyWindVectorDirectionPrevailing)
            obsDataD[i].windVecDirPrev = NODATA;
        else if (myVar == dailyReferenceEvapotranspirationHS)
            obsDataD[i].et0_hs = NODATA;
        else if (myVar == dailyReferenceEvapotranspirationPM)
            obsDataD[i].et0_pm = NODATA;
        else if (myVar == dailyLeafWetness)
            obsDataD[i].leafW = NODATA;
        else if (myVar == dailyHeatingDegreeDays)
            obsDataD[i].dd_heating = NODATA;
        else if (myVar == dailyCoolingDegreeDays)
            obsDataD[i].dd_cooling = NODATA;
}

void Crit3DMeteoPoint::emptyObsDataD(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int indexIni = obsDataH[0].date.daysTo(date1);
    int indexFin = obsDataH[0].date.daysTo(date2);

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
        obsDataD[i].dd_heating = NODATA;
        obsDataD[i].dd_cooling = NODATA;
        obsDataD[i].leafW = NODATA;
    }
}

void Crit3DMeteoPoint::emptyObsDataM(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedM(date1, date2)) return;

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
    else if (myDate < obsDataH[0].date || myDate > obsDataH[nrObsDataDaysH - 1].date)
        return false;
    else
        return true;
}

bool Crit3DMeteoPoint::isDateTimeLoadedH(const Crit3DTime& myDateTime)
{
    if (nrObsDataDaysH == 0)
        return false;
    else if (myDateTime < Crit3DTime(obsDataH[0].date,1) || myDateTime >= Crit3DTime(obsDataH[nrObsDataDaysH - 1].date,1))
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
    else if (date1 < obsDataH[0].date || date2 > obsDataH[nrObsDataDaysH - 1].date)
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
    else if (obsDataH == nullptr)
        return false;
    else if (timeIni.date < obsDataH[0].date || timeFin.date > (obsDataH[0].date.addDays(nrObsDataDaysH - 1)))
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
    else if (obsDataH == nullptr)
        return 0.0;
    else if (timeFin.date < obsDataH[0].date || timeIni.date > (obsDataH[0].date.addDays(nrObsDataDaysH - 1)))
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
            if (int(myValue) != int(NODATA))
                counter++;

            counterAll++;
            myTime = myTime.addSeconds(deltaSeconds);
        }
        return (float(counter)/float(counterAll));
    }

}

void Crit3DMeteoPoint::cleanObsDataH()
{
    quality = quality::missing_data;

    if (nrObsDataDaysH > 0)
    {
        for (int i = 0; i < nrObsDataDaysH; i++)
        {
            delete [] obsDataH[i].tAir;
            delete [] obsDataH[i].prec;
            delete [] obsDataH[i].rhAir;
            delete [] obsDataH[i].tDew;
            delete [] obsDataH[i].irradiance;
            delete [] obsDataH[i].netIrradiance;
            delete [] obsDataH[i].windScalInt;
            delete [] obsDataH[i].windVecX;
            delete [] obsDataH[i].windVecY;
            delete [] obsDataH[i].windVecInt;
            delete [] obsDataH[i].windVecDir;
            delete [] obsDataH[i].leafW;
            delete [] obsDataH[i].transmissivity;
        }
        delete [] obsDataH;
    }

    nrObsDataDaysH = 0;
}


void Crit3DMeteoPoint::cleanObsDataD()
{
    quality = quality::missing_data;

    obsDataD.clear();
}

void Crit3DMeteoPoint::cleanObsDataM()
{
    quality = quality::missing_data;

    obsDataM.clear();
}



bool Crit3DMeteoPoint::setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue)
{
    //check
    if (myVar == noMeteoVar || obsDataH == nullptr)
    {
        return false;
    }

    // day index
    int i = obsDataH[0].date.daysTo(myDate);

    //check if out of range (accept +1 date exceed)
    if (i < 0 || i > nrObsDataDaysH) return false;

    // sub hourly index
    int subH = int(ceil(float(myMinutes) / float(60 / hourlyFraction)));

    //if +1 date exceed accept only hour 00:00
    if (i == nrObsDataDaysH && (myHour != 0 || subH != 0)) return false;

    // hour 0 becomes hour 24 of the previous day
    if (myHour == 0 && subH == 0)
    {
        myHour = 24;
        i--;
        if (i < 0) return false;
    }

    // (sub)hour index
    int j = hourlyFraction * myHour + subH - 1;

    if (j < 0 || j >= hourlyFraction * 24) return false;

    if (myVar == airTemperature)
        obsDataH[i].tAir[j] = myValue;
    else if (myVar == precipitation)
        obsDataH[i].prec[j] = myValue;
    else if (myVar == airRelHumidity)
        obsDataH[i].rhAir[j] = myValue;
    else if (myVar == airDewTemperature)
        obsDataH[i].tDew[j] = myValue;
    else if (myVar == globalIrradiance)
        obsDataH[i].irradiance[j] = myValue;
    else if (myVar == netIrradiance)
        obsDataH[i].netIrradiance[j] = myValue;
    else if (myVar == referenceEvapotranspiration)
        obsDataH[i].et0[j] = myValue;
    else if (myVar == windScalarIntensity)
        obsDataH[i].windScalInt[j] = myValue;
    else if (myVar == windVectorX)
    {
        obsDataH[i].windVecX[j] = myValue;
        float intensity = NODATA, direction = NODATA;
        computeWindPolar(obsDataH[i].windVecX[j], obsDataH[i].windVecY[j], &intensity, &direction);
        obsDataH[i].windVecInt[j] = intensity;
        obsDataH[i].windVecDir[j] = direction;
    }
    else if (myVar == windVectorY)
    {
        obsDataH[i].windVecY[j] = myValue;
        float intensity = NODATA, direction = NODATA;
        computeWindPolar(obsDataH[i].windVecX[j], obsDataH[i].windVecY[j], &intensity, &direction);
        obsDataH[i].windVecInt[j] = intensity;
        obsDataH[i].windVecDir[j] = direction;
    }
    else if (myVar == windVectorIntensity)
    {
        obsDataH[i].windVecInt[j] = myValue;
        float u = NODATA, v = NODATA;
        computeWindCartesian(obsDataH[i].windVecInt[j], obsDataH[i].windVecDir[j], &u, &v);
        obsDataH[i].windVecX[j] = u;
        obsDataH[i].windVecY[j] = v;
    }
    else if (myVar == windVectorDirection)
    {
        obsDataH[i].windVecDir[j] = myValue;
        float u = NODATA, v = NODATA;
        computeWindCartesian(obsDataH[i].windVecInt[j], obsDataH[i].windVecDir[j], &u, &v);
        obsDataH[i].windVecX[j] = u;
        obsDataH[i].windVecY[j] = v;
    }
    else if (myVar == leafWetness)
        obsDataH[i].leafW[j] = int(myValue);
    else if (myVar == atmTransmissivity)
        obsDataH[i].transmissivity[j] = myValue;
    else
        return false;

    return true;
}

bool Crit3DMeteoPoint::setMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, float myValue)
{
    long index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= nrObsDataDaysD)) return false;

    unsigned i = unsigned(index);

    if (myVar == dailyAirTemperatureMax)
        obsDataD[i].tMax = myValue;
    else if (myVar == dailyAirTemperatureMin)
        obsDataD[i].tMin = myValue;
    else if (myVar == dailyAirTemperatureAvg)
        obsDataD[i].tAvg = myValue;
    else if (myVar == dailyPrecipitation)
        obsDataD[i].prec = myValue;
    else if (myVar == dailyAirRelHumidityMax)
        obsDataD[i].rhMax = myValue;
    else if (myVar == dailyAirRelHumidityMin)
        obsDataD[i].rhMin = myValue;
    else if (myVar == dailyAirRelHumidityAvg)
        obsDataD[i].rhAvg = myValue;
    else if (myVar == dailyGlobalRadiation)
        obsDataD[i].globRad = myValue;
    else if (myVar == dailyReferenceEvapotranspirationHS)
         obsDataD[i].et0_hs = myValue;
    else if (myVar == dailyReferenceEvapotranspirationPM)
         obsDataD[i].et0_pm = myValue;
    else if (myVar == dailyHeatingDegreeDays)
         obsDataD[i].dd_heating = myValue;
    else if (myVar == dailyCoolingDegreeDays)
         obsDataD[i].dd_cooling = myValue;
    else if (myVar == dailyWindScalarIntensityAvg)
        obsDataD[i].windScalIntAvg = myValue;
    else if (myVar == dailyWindScalarIntensityMax)
        obsDataD[i].windScalIntMax = myValue;
    else if (myVar == dailyWindVectorIntensityAvg)
        obsDataD[i].windVecIntAvg = myValue;
    else if (myVar == dailyWindVectorIntensityMax)
        obsDataD[i].windVecIntMax = myValue;
    else if (myVar == dailyWindVectorDirectionPrevailing)
        obsDataD[i].windVecDirPrev = myValue;
    else if (myVar == dailyLeafWetness)
        obsDataD[i].leafW = myValue;					
    else if (myVar == dailyWaterTableDepth)
        obsDataD[i].waterTable = myValue;
    else
        return false;

    return true;
}

bool Crit3DMeteoPoint::setMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar, float myValue)
{
    //check
    if (myVar == noMeteoVar) return false;
    if (nrObsDataDaysM == 0) return false;

    int index;
    if (myDate.year == obsDataM[0]._year)
    {
        // same year of first data
        index = myDate.month-obsDataM[0]._month;
    }
    else if (myDate.year == obsDataM[0]._year+1)
    {
        // second year
        index = 12-obsDataM[0]._month + myDate.month;
    }
    else
    {
        // other years
        index = (myDate.year - obsDataM[0]._year -1)*12+(12-obsDataM[0]._month) + myDate.month;
    }
    if ((index < 0) || (index >= nrObsDataDaysM)) return false;

    unsigned i = unsigned(index);

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

float Crit3DMeteoPoint::getMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar)
{
    //check
    if (myVar == noMeteoVar)
    {
        return NODATA;
    }
    if (obsDataH == nullptr)
    {
        return NODATA;
    }

    // day index
    int i = obsDataH[0].date.daysTo(myDate);

    //check if out of range (accept +1 date exceed)
    if (i < 0 || i > nrObsDataDaysH) return NODATA;

    // sub hourly index
    int subH = int(ceil(float(myMinutes) / float(60 / hourlyFraction)));

    //if +1 date exceed accept only hour 00:00
    if (i == nrObsDataDaysH && (myHour != 0 || subH != 0)) return NODATA;

    // hour 0 becomes hour 24 of the previous day
    if (myHour == 0 && subH == 0)
    {
        myHour = 24;
        i--;
        if (i < 0) return NODATA;
    }

    // (sub)hour index
    int j = hourlyFraction * myHour + subH - 1;
    if (j < 0 || j >= hourlyFraction * 24) return false;

    if (myVar == airTemperature)
        return (obsDataH[i].tAir[j]);
    else if (myVar == precipitation)
        return (obsDataH[i].prec[j]);
    else if (myVar == airRelHumidity)
        return (obsDataH[i].rhAir[j]);
    else if (myVar == airDewTemperature)
    {
        if (int(obsDataH[i].tDew[j]) != int(NODATA))
            return obsDataH[i].tDew[j];
        else
            return tDewFromRelHum(obsDataH[i].rhAir[j], obsDataH[i].tAir[j]);
    }
    else if (myVar == globalIrradiance)
        return (obsDataH[i].irradiance[j]);
    else if (myVar == netIrradiance)
        return (obsDataH[i].netIrradiance[j]);
    else if (myVar == referenceEvapotranspiration)
        return (obsDataH[i].et0[j]);
    else if (myVar == windScalarIntensity)
        return (obsDataH[i].windScalInt[j]);
    else if (myVar == windVectorX)
        return (obsDataH[i].windVecX[j]);
    else if (myVar == windVectorY)
        return (obsDataH[i].windVecY[j]);
    else if (myVar == windVectorIntensity)
        return (obsDataH[i].windVecInt[j]);
    else if (myVar == windVectorDirection)
        return (obsDataH[i].windVecDir[j]);
    else if (myVar == leafWetness)
        return float(obsDataH[i].leafW[j]);
    else if (myVar == atmTransmissivity)
        return (obsDataH[i].transmissivity[j]);
    else
    {
        return (NODATA);
    }
}

Crit3DDate Crit3DMeteoPoint::getMeteoPointHourlyValuesDate(int index)
{

    if (index < 0 || index >= nrObsDataDaysH) return NO_DATE;
    return obsDataH[index].date;
}

bool Crit3DMeteoPoint::getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH* &hourlyValues)
{
    int d = obsDataH[0].date.daysTo(myDate);
    if (d < 0 || d >= nrObsDataDaysH) return false;
    hourlyValues = &(obsDataH[d]);
    return true;
}


bool Crit3DMeteoPoint::existDailyData(const Crit3DDate& myDate)
{
    if (obsDataD.size() == 0) return false;

    int index = obsDataD[0].date.daysTo(myDate);

    if ((index < 0) || (index >= nrObsDataDaysD))
        return false;
    else
        return true;
}


Crit3DDate Crit3DMeteoPoint::getLastDailyData()
{
    if (obsDataD.size() == 0)
        return NO_DATE;

    return obsDataD[nrObsDataDaysD-1].date;
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings)
{
    //check
    if (myVar == noMeteoVar) return NODATA;
    if (nrObsDataDaysD == 0) return NODATA;

    int index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= nrObsDataDaysD)) return NODATA;

    unsigned i = unsigned(index);

    if (myVar == dailyAirTemperatureMax)
        return (obsDataD[i].tMax);
    else if (myVar == dailyAirTemperatureMin)
        return (obsDataD[i].tMin);
    else if (myVar == dailyAirTemperatureAvg)
    {
        if (! isEqual(obsDataD[i].tAvg, NODATA))
            return obsDataD[i].tAvg;
        else if (meteoSettings->getAutomaticTavg() && !isEqual(obsDataD[i].tMin, NODATA) && !isEqual(obsDataD[i].tMax, NODATA))
            return ((obsDataD[i].tMin + obsDataD[i].tMax) / 2);
        else
            return NODATA;
    }
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
    {
        if (! isEqual(obsDataD[i].et0_hs, NODATA))
            return obsDataD[i].et0_hs;
        else if (meteoSettings->getAutomaticET0HS() && !isEqual(obsDataD[i].tMin, NODATA) && !isEqual(obsDataD[i].tMax, NODATA))
            return float(ET0_Hargreaves(meteoSettings->getTransSamaniCoefficient(), latitude,
                                        getDoyFromDate(myDate), obsDataD[i].tMax, obsDataD[i].tMin));
        else
            return NODATA;
    }
    else if (myVar == dailyReferenceEvapotranspirationPM)
        return (obsDataD[i].et0_pm);
    else if (myVar == dailyHeatingDegreeDays)
        return (obsDataD[i].dd_heating);
    else if (myVar == dailyCoolingDegreeDays)
        return (obsDataD[i].dd_cooling);
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
        return (NODATA);
}


float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate &myDate, meteoVariable myVar)
{
    //check
    if (myVar == noMeteoVar) return NODATA;
    if (nrObsDataDaysD == 0) return NODATA;

    int index = obsDataD[0].date.daysTo(myDate);
    if ((index < 0) || (index >= nrObsDataDaysD)) return NODATA;

    unsigned i = unsigned(index);

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
        return (NODATA);
}

float Crit3DMeteoPoint::getMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar)
{
    //check
    if (myVar == noMeteoVar) return NODATA;
    if (nrObsDataDaysM == 0) return NODATA;

    int index;
    if (myDate.year == obsDataM[0]._year)
    {
        // same year of first data
        index = myDate.month-obsDataM[0]._month;
    }
    else if (myDate.year == obsDataM[0]._year+1)
    {
        // second year
        index = 12-obsDataM[0]._month + myDate.month;
    }
    else
    {
        // other years
        index = (myDate.year - obsDataM[0]._year -1)*12+(12-obsDataM[0]._month) + myDate.month;
    }
    if ((index < 0) || (index >= nrObsDataDaysM)) return NODATA;

    unsigned i = unsigned(index);

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
        return (NODATA);
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

std::vector <float> Crit3DMeteoPoint::getProxyValues()
{
    std::vector <float> myValues;
    for (unsigned int i=0; i < proxyValues.size(); i++)
        myValues.push_back(getProxyValue(i));

    return myValues;
}

bool Crit3DMeteoPoint::computeDerivedVariables(Crit3DTime dateTime)
{
    short leafW;
    float temperature, windSpeed, height, netRadiation;

    Crit3DDate myDate = dateTime.date;
    int myHour = dateTime.getHour();

    bool leafWres = false;
    bool et0res = false;

    double relHumidity = double(getMeteoPointValueH(myDate, myHour, 0, airRelHumidity));
    double prec = double(getMeteoPointValueH(myDate, myHour, 0, precipitation));

    if (computeLeafWetness(prec, relHumidity, &leafW))
        leafWres = setMeteoPointValueH(myDate, myHour, 0, leafWetness, leafW);

    temperature = getMeteoPointValueH(myDate, myHour, 0, airTemperature);
    windSpeed = getMeteoPointValueH(myDate, myHour, 0, windScalarIntensity);
    netRadiation = getMeteoPointValueH(myDate, myHour, 0, netIrradiance);
    height = float(this->point.z);
    float et0;

    if (! isEqual(temperature, NODATA) && ! isEqual(relHumidity, NODATA) && ! isEqual(windSpeed, NODATA))
    {
        et0 = float(ET0_Penman_hourly_net_rad(double(height), double(netRadiation),
                          double(temperature), double(relHumidity), double(windSpeed)));
        et0res = setMeteoPointValueH(myDate, myHour, 0, referenceEvapotranspiration, et0);
    }
    return (leafWres && et0res);
}

bool Crit3DMeteoPoint::computeMonthlyAggregate(Crit3DDate firstDate, Crit3DDate lastDate, meteoVariable dailyMeteoVar,
                                               Crit3DMeteoSettings* meteoSettings, Crit3DQuality* qualityCheck,
                                               Crit3DClimateParameters* climateParam)
{

    int currentMonth = firstDate.month;
    int nrDays = getDaysInMonth(currentMonth, firstDate.year);

    float sum = 0;
    int nrValid = 0;
    int indexMonth = 0;
    bool aggregateDailyInMonthly = false;

    for (Crit3DDate actualDate = firstDate; actualDate<=lastDate; actualDate=actualDate.addDays(1))
    {
        float myDailyValue = getMeteoPointValueD(actualDate, dailyMeteoVar, meteoSettings);
        quality::qualityType qualityT = qualityCheck->checkFastValueDaily_SingleValue(dailyMeteoVar, climateParam,
                                                                                      myDailyValue, currentMonth, float(point.z));
        if (qualityT == quality::accepted)
        {
            sum = sum + myDailyValue;
            nrValid = nrValid + 1;
        }
        if (actualDate.day == nrDays || actualDate == lastDate)
        {
            indexMonth = indexMonth + 1;
            if ((float(nrValid)/float(nrDays)*100) >= meteoSettings->getMinimumPercentage())
            {
                aggregateDailyInMonthly = true;
                if (dailyMeteoVar == dailyAirTemperatureMin || dailyMeteoVar == dailyAirTemperatureMax || dailyMeteoVar == dailyAirTemperatureAvg)
                {
                    if (nrValid != 0)
                    {
                        setMeteoPointValueM(actualDate, updateMeteoVariable(dailyMeteoVar, monthly), sum/float(nrValid));
                    }
                    else
                    {
                        setMeteoPointValueM(actualDate,updateMeteoVariable(dailyMeteoVar, monthly), NODATA);
                    }
                }
                else if (dailyMeteoVar == dailyPrecipitation || dailyMeteoVar == dailyReferenceEvapotranspirationHS
                         || dailyMeteoVar == dailyReferenceEvapotranspirationPM || dailyMeteoVar == dailyGlobalRadiation)
                {
                    setMeteoPointValueM(actualDate,updateMeteoVariable(dailyMeteoVar, monthly), sum);
                }
            }
            else
            {
                setMeteoPointValueM(actualDate,updateMeteoVariable(dailyMeteoVar, monthly),NODATA);
            }
            sum = 0;
            nrValid = 0;
            currentMonth = actualDate.addDays(1).month;
            nrDays = getDaysInMonth(currentMonth, actualDate.year);
        }

    }
    return aggregateDailyInMonthly;
}

TObsDataH *Crit3DMeteoPoint::getObsDataH() const
{
    return obsDataH;
}

// ---- end class

bool isSelectionPointsActive(Crit3DMeteoPoint* meteoPoints,int nrMeteoPoints)
{
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].selected)
            return true;
    }

    return false;
}
