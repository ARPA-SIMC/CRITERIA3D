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


Crit3DMeteoPoint::Crit3DMeteoPoint()
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
    this->obsDataD = nullptr;
    this->obsDataM = nullptr;

    this->currentValue = NODATA;
    this->residual = NODATA;

    this->elaboration = NODATA;
    this->anomaly = NODATA;
    this->anomalyPercentage = NODATA;
    this->climate = NODATA;

    this->active = false;
    this->selected = false;

    this->quality = quality::missing_data;

    proxyValues.clear();
    lapseRateCode = primary;
    topographicDistance = nullptr;
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
    for (int i = 0; i < numberOfDays; i++)
    {
        obsDataH[i].date = myDate;
        obsDataH[i].tAir = new float[nrDailyValues];
        obsDataH[i].prec = new float[nrDailyValues];
        obsDataH[i].rhAir = new float[nrDailyValues];
        obsDataH[i].tDew = new float[nrDailyValues];
        obsDataH[i].irradiance = new float[nrDailyValues];
        obsDataH[i].et0 = new float[nrDailyValues];
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
            obsDataH[i].et0[j] = NODATA;
            obsDataH[i].windVecInt[j] = NODATA;
            obsDataH[i].windVecDir[j] = NODATA;
            obsDataH[i].windScalInt[j] = NODATA;
            obsDataH[i].leafW[j] = NODATA;
            obsDataH[i].transmissivity[j] = NODATA;
        }
        ++myDate;
    }
}


void Crit3DMeteoPoint::initializeObsDataD(int numberOfDays, const Crit3DDate& firstDate)
{
    this->cleanObsDataD();

    nrObsDataDaysD = numberOfDays;
    obsDataD = new TObsDataD[unsigned(numberOfDays)];

    quality = quality::missing_data;
    residual = NODATA;

    Crit3DDate myDate = firstDate;
    for (int i = 0; i < numberOfDays; i++)
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
        obsDataD[i].windVecIntAvg = NODATA;
        obsDataD[i].windVecIntMax = NODATA;
        obsDataD[i].windVecDirPrev = NODATA;
        obsDataD[i].windScalIntAvg = NODATA;
        obsDataD[i].windScalIntMax = NODATA;
        obsDataD[i].waterTable = NODATA;
        ++myDate;
    }
}

void Crit3DMeteoPoint::initializeObsDataM(int numberOfMonths, int month, int year)
{
    this->cleanObsDataM();

    nrObsDataDaysM = numberOfMonths;
    obsDataM = new TObsDataM[unsigned(numberOfMonths)];

    quality = quality::missing_data;
    residual = NODATA;
    int addYear = -1;

    for (int i = month; i <= numberOfMonths; i++)
    {
        if (i < 12)
        {
            obsDataM[i]._month = i;
            obsDataM[i]._year = year;
        }
        else if (i%12 == 0)
        {
            addYear = addYear+1;
            obsDataM[i]._month = 12;
            obsDataM[i]._year = year + addYear;
        }
        else
        {
            obsDataM[i]._month = i%12;
            obsDataM[i]._year = year + addYear + 1;
        }

        obsDataM[i].tMax = NODATA;
        obsDataM[i].tMin = NODATA;
        obsDataM[i].tAvg = NODATA;
        obsDataM[i].prec = NODATA;
        obsDataM[i].et0 = NODATA;
        obsDataM[i].globRad = NODATA;
    }
}


void Crit3DMeteoPoint::emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& myDate)
{
    if (! isDateLoadedH(myDate)) return;

    int nrDayValues = hourlyFraction * 24;
    int i = obsDataH[0].date.daysTo(myDate);
    residual = NODATA;

    if (i>=0 && i<nrObsDataDaysH)
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
                else if (myVar == windScalarIntensity)
                    obsDataH[i].windScalInt[j] = NODATA;
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
            else if (myVar == windScalarIntensity)
                obsDataH[i].windScalInt[j] = NODATA;
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

void Crit3DMeteoPoint::emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (! isDateIntervalLoadedH(date1, date2)) return;

    int indexIni = obsDataD[0].date.daysTo(date1);
    int indexFin = obsDataD[0].date.daysTo(date2);
    residual = NODATA;

    for (int i = indexIni; i <= indexFin; i++)
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
        return (false);
    else if (myDate < obsDataD->date || myDate > (obsDataD->date.addDays(nrObsDataDaysD - 1)))
        return (false);
    else
        return (true);
}

bool Crit3DMeteoPoint::isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2)
{
    if (nrObsDataDaysD == 0)
        return false;
    else if (date1 > date2)
        return false;
    else if (date1 < obsDataD->date || date2 > (obsDataD->date.addDays(nrObsDataDaysD - 1)))
        return (false);
    else
        return (true);
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
            delete [] obsDataH[i].windScalInt;
            delete [] obsDataH[i].windVecInt;
            delete [] obsDataH[i].windVecDir;
            delete [] obsDataH[i].leafW;
            delete [] obsDataH[i].transmissivity;
        }
        delete [] obsDataH;
    }
}


void Crit3DMeteoPoint::cleanObsDataD()
{
    quality = quality::missing_data;

    if (nrObsDataDaysD > 0)
        delete [] obsDataD;
}

void Crit3DMeteoPoint::cleanObsDataM()
{
    quality = quality::missing_data;

    if (nrObsDataDaysM > 0)
        delete [] obsDataM;
}

bool Crit3DMeteoPoint::setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue)
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
    if (i < 0 || i > nrObsDataDaysH) return false;

    // sub hourly index
    int subH = int(ceil(float(myMinutes) / float(60 / hourlyFraction)));

    //if +1 date exceed accept only hour 00:00
    if (i == nrObsDataDaysH && myHour != 0 && subH == 0) return false;

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
    else if (myVar == referenceEvapotranspiration)
        obsDataH[i].et0[j] = myValue;
    else if (myVar == windScalarIntensity)
    {
        obsDataH[i].windScalInt[j] = myValue;
        obsDataH[i].windVecInt[j] = myValue;
    }
    else if (myVar == windVectorIntensity)
    {
        obsDataH[i].windScalInt[j] = myValue;
        obsDataH[i].windVecInt[j] = myValue;
    }
    else if (myVar == windVectorDirection)
        obsDataH[i].windVecDir[j] = myValue;
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

    long i = obsDataD[0].date.daysTo(myDate);
    if ((i <0) || (i >= nrObsDataDaysD)) return false;

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
    if (i == nrObsDataDaysH && myHour != 0 && subH == 0) return NODATA;

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
    else if (myVar == referenceEvapotranspiration)
        return (obsDataH[i].et0[j]);
    else if (myVar == windScalarIntensity)
        return (obsDataH[i].windScalInt[j]);
    else if (myVar == windVectorIntensity)
        return (obsDataH[i].windVecInt[j]);
    else if (myVar == windVectorDirection)
        return (obsDataH[i].windVecDir[j]);
    else if (myVar == windVectorX)
    {
        float u = NODATA, v = NODATA;
        computeWindCartesian(obsDataH[i].windVecInt[j], obsDataH[i].windVecDir[j], &u, &v);
        return u;
    }
    else if (myVar == windVectorY)
    {
        float u = NODATA, v = NODATA;
        computeWindCartesian(obsDataH[i].windVecInt[j], obsDataH[i].windVecDir[j], &u, &v);
        return v;
    }
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

float Crit3DMeteoPoint::getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar)
{
    //check
    if (myVar == noMeteoVar) return NODATA;
    if (obsDataD == nullptr) return NODATA;

    int i = obsDataD[0].date.daysTo(myDate);
    if ((i < 0) || (i >= nrObsDataDaysD)) return NODATA;

    if (myVar == dailyAirTemperatureMax)
        return (obsDataD[i].tMax);
    else if (myVar == dailyAirTemperatureMin)
        return (obsDataD[i].tMin);
    else if (myVar == dailyAirTemperatureAvg)
    {   /* todo
        if (isEqual(obsDataD[i].tAvg, NODATA) && autoTavg && (! isEqual(obsDataD[i].tMin, NODATA && ! isEqual(obsDataD[i].tMax, NODATA))))
            return ((obsDataD[i].tMin / obsDataD[i].tMax) / 2);
        else*/
        return obsDataD[i].tAvg;
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
        return (obsDataD[i].et0_hs);
    else if (myVar == dailyReferenceEvapotranspirationPM)
        return (obsDataD[i].et0_pm);
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


float Crit3DMeteoPoint::getMeteoPointValue(const Crit3DTime& myTime, meteoVariable myVar)
{
    frequencyType frequency = getVarFrequency(myVar);
    if (frequency == hourly)
        return getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), myVar);
    else if (frequency == daily)
        return getMeteoPointValueD(myTime.date, myVar);
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
