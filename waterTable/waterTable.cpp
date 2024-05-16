#include "waterTable.h"
#include "commonConstants.h"
#include "weatherGenerator.h"

WaterTable::WaterTable(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DMeteoGrid* meteoGrid, bool isMeteoGridLoaded, Crit3DMeteoSettings meteoSettings, gis::Crit3DGisSettings gisSettings)
    : meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), meteoGrid(meteoGrid), isMeteoGridLoaded(isMeteoGridLoaded), meteoSettings(meteoSettings), gisSettings(gisSettings)
{

}

QDate WaterTable::getFirstDateWell()
{
    firstDateWell = well.getFirstDate();
    return firstDateWell;
}

QDate WaterTable::getLastDateWell()
{
    lastDateWell = well.getLastDate();
    return lastDateWell;
}

QString WaterTable::getError() const
{
    return error;
}

void WaterTable::initializeWaterTable(Well myWell)
{

    this->well = myWell;
    getFirstDateWell();
    getLastDateWell();
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        WTClimateMonthly[myMonthIndex] = NODATA;
    }

    isMeteoPointLinked = false;
    isCWBEquationReady = false;
    isClimateReady = false;

    alpha = NODATA;
    h0 = NODATA;
    R2 = NODATA;
    nrDaysPeriod = NODATA;
    nrObsData = 0;
    EF = NODATA;
    NASH = NODATA;
    RMSE = NODATA;
    avgDailyCWB = NODATA;

}

bool WaterTable::computeWaterTable(Well myWell, int maxNrDays, int doy1, int doy2)
{
    if (myWell.getDepthNr() == 0)
    {
        error = "No WaterTable data loaded.";
        return false;
    }

    initializeWaterTable(myWell);
    isClimateReady = computeWTClimate();
    isMeteoPointLinked = assignNearestMeteoPoint();
    if (isMeteoPointLinked == false)
    {
        return false;
    }

    if (!computeETP_allSeries())
    {
        return false;
    }

    isCWBEquationReady = computeCWBCorrelation(maxNrDays);
    if (!isCWBEquationReady)
    {
        return false;
    }
    // LC non c'è alcun controllo sul valore di ritorno di computeWaterTableIndices
    computeWaterTableIndices (doy1, doy2);
    return true;
}

bool WaterTable::computeWTClimate()
{
    if (well.getDepthNr() < 3)
    {
        error = "Missing data";
        return false;
    }

    std::vector<float> H_sum;
    std::vector<float> H_num;
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        H_sum.push_back(0);
        H_num.push_back(0);
    }

    QMap<QDate, int> myDepths = well.getDepths();
    QMapIterator<QDate, int> it(myDepths);
    while (it.hasNext())
    {
        QDate myDate = it.key();
        int myValue = it.value();
        int myMonth = myDate.month();
        int myMonthIndex = myMonth - 1;
        H_sum[myMonthIndex] = H_sum[myMonthIndex] + myValue;
        H_num[myMonthIndex] = H_num[myMonthIndex] + 1;
    }

    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        if (H_num[myMonthIndex] < 2)
        {
            error = "Missing watertable data: month " + QString::number(myMonthIndex+1);
            return false;
        }
        WTClimateMonthly[myMonthIndex] = H_sum[myMonthIndex] / H_num[myMonthIndex];
        isClimateReady = true;
        cubicSplineYearInterpolate(WTClimateMonthly, WTClimateDaily);
    }
    return true;
}

bool WaterTable::assignNearestMeteoPoint()
{
    float minimumDistance = NODATA;
    bool assignNearestMeteoPoint = false;
    if (isMeteoGridLoaded)
    {
        int zoneNumber;
        for (unsigned row = 0; row < unsigned(meteoGrid->gridStructure().header().nrRows); row++)
        {
            for (unsigned col = 0; col < unsigned(meteoGrid->gridStructure().header().nrCols); col++)
            {
                double utmX = meteoGrid->meteoPointPointer(row,col)->point.utm.x;
                double utmY = meteoGrid->meteoPointPointer(row,col)->point.utm.y;
                if (utmX == NODATA || utmY == NODATA)
                {
                    double lat = meteoGrid->meteoPointPointer(row,col)->latitude;
                    double lon = meteoGrid->meteoPointPointer(row,col)->longitude;
                    gis::latLonToUtm(lat, lon, &utmX, &utmY, &zoneNumber);
                }
                float myDistance = gis::computeDistance(well.getUtmX(), well.getUtmY(), utmX, utmY);
                if (myDistance < MAXWELLDISTANCE )
                {
                    if (myDistance < minimumDistance || minimumDistance == NODATA)
                    {
                        if (assignWTMeteoData(*meteoGrid->meteoPointPointer(row,col) ))
                        {
                            minimumDistance = myDistance;
                            assignNearestMeteoPoint = true;
                            linkedMeteoPoint = (*meteoGrid->meteoPointPointer(row,col));
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {

            double utmX = meteoPoints[i].point.utm.x;
            double utmY = meteoPoints[i].point.utm.y;
            float myDistance = gis::computeDistance(well.getUtmX(), well.getUtmY(), utmX, utmY);
            if (myDistance < MAXWELLDISTANCE )
            {
                if (myDistance < minimumDistance || minimumDistance == NODATA)
                {
                    if (assignWTMeteoData(meteoPoints[i]))
                    {
                        minimumDistance = myDistance;
                        assignNearestMeteoPoint = true;
                        linkedMeteoPoint = meteoPoints[i];
                    }
                }
            }
        }
    }
    return assignNearestMeteoPoint;
}

bool WaterTable::assignWTMeteoData(Crit3DMeteoPoint point)
{
    firstMeteoDate = firstDateWell.addDays(-730); // necessari 24 mesi di dati meteo precedenti il primo dato di falda
    lastMeteoDate.setDate(point.getLastDailyData().year, point.getLastDailyData().month, point.getLastDailyData().day); // ultimo dato disponibile
    float precPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyPrecipitation);
    float tMinPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMin);
    float tMaxPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMax);

    float minPercentage = meteoSettings.getMinimumPercentage();
    if (precPerc > minPercentage/100 && tMinPerc > minPercentage/100 && tMaxPerc > minPercentage/100)
    {
        return true;
    }
    else
    {
        error = "Not enough meteo data to analyze watertable period. Try to decrease the required percentage";
        return false;
    }
}

bool WaterTable::computeETP_allSeries()
{
    etpValues.clear();
    precValues.clear();
    double myLat;
    double myLon;
    gis::getLatLonFromUtm(gisSettings, well.getUtmX(), well.getUtmY(), &myLat, &myLon);
    float sumCWB = 0;
    int nrValidDays = 0;
    for (QDate myDate = firstMeteoDate; myDate<=lastMeteoDate; myDate=myDate.addDays(1))
    {
        Crit3DDate date(myDate.day(), myDate.month(), myDate.year());
        float Tmin = linkedMeteoPoint.getMeteoPointValueD(date, dailyAirTemperatureMin);
        float Tmax = linkedMeteoPoint.getMeteoPointValueD(date, dailyAirTemperatureMax);
        float prec = linkedMeteoPoint.getMeteoPointValueD(date, dailyPrecipitation);
        float etp = dailyEtpHargreaves(Tmin, Tmax, date, myLat,&meteoSettings);
        etpValues.push_back(etp);
        precValues.push_back(prec);
        if (etp != NODATA && prec != NODATA)
        {
            sumCWB = sumCWB + (prec - etp);
            nrValidDays = nrValidDays + 1;
        }
    }

    if (nrValidDays > 0)
    {
        avgDailyCWB = sumCWB / nrValidDays;
    }
    else
    {
        error = "Missing data: " + QString::fromStdString(linkedMeteoPoint.name);
        return false;
    }

    return true;
}

// Ricerca del periodo di correlazione migliore
bool WaterTable::computeCWBCorrelation(int maxNrDays)
{
    float bestR2 = 0;
    float bestH0;
    float bestAlfaCoeff;
    int bestNrDays = NODATA;
    QMap<QDate, int> myDepths = well.getDepths();
    std::vector<float> myCWBSum;
    std::vector<float> myObsWT;
    float a;
    float b;
    float myR2;

    for (int nrDays = 90; nrDays <= maxNrDays; nrDays=nrDays+10)
    {
        QMapIterator<QDate, int> it(myDepths);
        while (it.hasNext())
        {
            QDate myDate = it.key();
            int myValue = it.value();
            float myCWBValue = computeCWB(myDate, nrDays);  // [cm]
            if (myCWBValue != NODATA)
            {
                myCWBSum.push_back(myCWBValue);
                myObsWT.push_back(myValue);
            }
        }
        statistics::linearRegression(myCWBSum, myObsWT, myCWBSum.size(), false, &a, &b, &myR2);
        if (myR2 > bestR2)
        {
            bestR2 = myR2;
            bestNrDays = nrDays;
            bestH0 = a;
            bestAlfaCoeff = b;
        }
    }
    if (bestR2 > 0)
    {
        nrObsData = myObsWT.size();
        nrDaysPeriod = bestNrDays;
        h0 = bestH0;
        alpha = bestAlfaCoeff;
        R2 = bestR2;
        isCWBEquationReady = true;
        return true;
    }
    else
    {
        return false;
    }
}

// Climatic WaterBalance (CWB) on a nrDaysPeriod
float WaterTable::computeCWB(QDate myDate, int nrDays)
{
    float sumCWB = 0;
    int nrValidDays = 0;
    QDate actualDate;
    float currentCWB;
    float weight;
    for (int shift = 1; shift<=nrDays; shift++)
    {
        actualDate = myDate.addDays(-shift);
        int index = firstMeteoDate.daysTo(actualDate) + 1; // LC CONTROLLARE
        if (index > 0 && index < precValues.size())
        {
            float etp = etpValues[index];
            float prec = precValues[index];
            if ( etp != NODATA &&  prec != NODATA)
            {
                currentCWB = prec - etp;
                weight = 1 - shift; // nrDaysPeriod
                sumCWB = sumCWB + currentCWB * weight;
                nrValidDays = nrValidDays + 1;
            }
        }
    }

    if (nrValidDays < (nrDaysPeriod * meteoSettings.getMinimumPercentage() / 100))
    {
        error = "Few Data";
        return NODATA;
    }
    // Climate
    float climateCWB = avgDailyCWB * nrDaysPeriod * 0.5;

    // conversion: from [mm] to [cm]
    float computeCWB = (sumCWB - climateCWB) * 0.1;

    return computeCWB;
}

// function to compute several statistical indices for watertable depth
bool WaterTable::computeWaterTableIndices(int doy1, int doy2)
{
    QMap<QDate, int> myDepths = well.getDepths();
    QMapIterator<QDate, int> it(myDepths);
    std::vector<float> myObs;
    std::vector<float> myComputed;
    std::vector<float> myClimate;
    float myIntercept;
    float myCoeff;
    while (it.hasNext())
    {
        QDate myDate = it.key();
        int myValue = it.value();
        int myDoy = myDate.dayOfYear();
        // doesn't work for interannual period
        if ((myDoy >= doy1) && (myDoy <= doy2))
        {
            float computedValue = getWaterTableDaily(myDate);
            if (computedValue != NODATA)
            {
                myObs.push_back(myValue);
                myComputed.push_back(computedValue);
                myClimate.push_back(getWaterTableClimate(myDate));
            }
        }
    }
    statistics::linearRegression(myObs, myComputed, myObs.size(), false, &myIntercept, &myCoeff, &R2);
    float mySum = 0;
    float mySumError = 0;
    float mySumDiffClimate = 0;
    float mySumDiffAvg = 0;
    float myErr = 0;
    float myErrAvg = 0;
    float myErrClimate = 0;

    nrObsData = myObs.size();
    for (int i=0; i<nrObsData; i++)
    {
        mySum = mySum + myObs[i];
    }
    float myObsAvg = mySum / nrObsData;
    for (int i=0; i<nrObsData; i++)
    {
        myErr = myComputed[i] - myObs[i];
        mySumError = mySumError + myErr * myErr;
        myErrAvg = myObs[i] - myObsAvg;
        mySumDiffAvg = mySumDiffAvg + myErrAvg * myErrAvg;
        if (isClimateReady)
        {
            myErrClimate = myObs[i] - myClimate[i];
            mySumDiffClimate = mySumDiffClimate + myErrClimate * myErrClimate;
        }
    }
    RMSE = sqrt(mySumError / nrObsData);
    NASH = 1 - mySumError / mySumDiffAvg;

    if (isClimateReady)
    {
        EF = 1 - mySumError / mySumDiffClimate;
    }
    else
    {
        EF = NODATA;
    }
    return true;
}

// restituisce il valore stimato di falda
float WaterTable::getWaterTableDaily(QDate myDate)
{
    float getWaterTableDaily = NODATA;
    bool isComputed = false;

    if (isMeteoPointLinked && isCWBEquationReady)
    {
        float myCWB = computeCWB(myDate, nrDaysPeriod);
        if (myCWB != NODATA)
        {
            float myH = h0 + alpha * myCWB;
            getWaterTableDaily = myH;
            isComputed = true;
        }
    }



    // No data: climatic value
    if (!isComputed && isClimateReady)
    {
        getWaterTableDaily = getWaterTableClimate(myDate);
    }
    return getWaterTableDaily;
}

float WaterTable::getWaterTableClimate(QDate myDate)
{

    float getWaterTableClimate = NODATA;

    if (!isClimateReady)
    {
        return getWaterTableClimate;
    }

    int myDoy = myDate.dayOfYear();
    getWaterTableClimate = WTClimateDaily[myDoy];
    return getWaterTableClimate;
}

