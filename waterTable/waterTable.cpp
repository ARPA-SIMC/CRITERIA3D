#include "waterTable.h"
#include "commonConstants.h"
#include "weatherGenerator.h"

WaterTable::WaterTable(Crit3DMeteoPoint *linkedMeteoPoint, Crit3DMeteoSettings meteoSettings, gis::Crit3DGisSettings gisSettings)
    : linkedMeteoPoint(linkedMeteoPoint), meteoSettings(meteoSettings), gisSettings(gisSettings)
{
    firstMeteoDate = QDate(linkedMeteoPoint->getFirstDailyData().year, linkedMeteoPoint->getFirstDailyData().month, linkedMeteoPoint->getFirstDailyData().day);
    lastMeteoDate = QDate(linkedMeteoPoint->getLastDailyData().year, linkedMeteoPoint->getLastDailyData().month, linkedMeteoPoint->getLastDailyData().day);
}

QString WaterTable::getIdWell() const
{
    return well.getId();
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

QMap<QDate, int> WaterTable::getDepths()
{
    return well.getDepths();
}

QString WaterTable::getError() const
{
    return error;
}

float WaterTable::getAlpha() const
{
    return alpha;
}

float WaterTable::getH0() const
{
    return h0;
}

int WaterTable::getNrDaysPeriod() const
{
    return nrDaysPeriod;
}

float WaterTable::getR2() const
{
    return R2;
}

float WaterTable::getRMSE() const
{
    return RMSE;
}

float WaterTable::getNASH() const
{
    return NASH;
}

float WaterTable::getEF() const
{
    return EF;
}

int WaterTable::getNrObsData() const
{
    return nrObsData;
}

std::vector<QDate> WaterTable::getMyDates() const
{
    return myDates;
}

std::vector<float> WaterTable::getMyHindcastSeries() const
{
    return myHindcastSeries;
}

std::vector<float> WaterTable::getMyInterpolateSeries() const
{
    return myInterpolateSeries;
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

bool WaterTable::computeWaterTable(Well myWell, int maxNrDays)
{
    if (myWell.getDepthNr() == 0)
    {
        error = "No WaterTable data loaded.";
        return false;
    }

    initializeWaterTable(myWell);
    isClimateReady = computeWTClimate();

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
    computeWaterTableIndices();
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
        it.next();
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
    }
    isClimateReady = true;
    cubicSplineYearInterpolate(WTClimateMonthly, WTClimateDaily);
    return true;
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
        float Tmin = linkedMeteoPoint->getMeteoPointValueD(date, dailyAirTemperatureMin);
        float Tmax = linkedMeteoPoint->getMeteoPointValueD(date, dailyAirTemperatureMax);
        float prec = linkedMeteoPoint->getMeteoPointValueD(date, dailyPrecipitation);
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
        error = "Missing data: " + QString::fromStdString(linkedMeteoPoint->name);
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
        myCWBSum.clear();
        myObsWT.clear();
        QMapIterator<QDate, int> it(myDepths);
        while (it.hasNext())
        {
            it.next();
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
        int index = firstMeteoDate.daysTo(actualDate);
        if (index >= 0 && index < precValues.size())
        {
            float etp = etpValues[index];
            float prec = precValues[index];
            if ( etp != NODATA &&  prec != NODATA)
            {
                currentCWB = prec - etp;
                weight = 1 - (float)shift/nrDays;
                sumCWB = sumCWB + currentCWB * weight;
                nrValidDays = nrValidDays + 1;
            }
        }
    }

    if (nrValidDays < (nrDays * meteoSettings.getMinimumPercentage() / 100))
    {
        error = "Few Data";
        return NODATA;
    }
    // Climate
    float climateCWB = avgDailyCWB * nrDays * 0.5;

    // conversion: from [mm] to [cm]
    float computeCWB = (sumCWB - climateCWB) * 0.1;

    return computeCWB;
}

// function to compute several statistical indices for watertable depth
bool WaterTable::computeWaterTableIndices()
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
        it.next();
        QDate myDate = it.key();
        int myValue = it.value();
        float computedValue = getWaterTableDaily(myDate);
        if (computedValue != NODATA)
        {
            myObs.push_back(myValue);
            myComputed.push_back(computedValue);
            myClimate.push_back(getWaterTableClimate(myDate));
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

    int nrObs = myObs.size();
    for (int i=0; i<nrObs; i++)
    {
        mySum = mySum + myObs[i];
    }
    float myObsAvg = mySum / nrObs;
    for (int i=0; i<nrObs; i++)
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
    RMSE = sqrt(mySumError / nrObs);
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

    if (isCWBEquationReady)
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
    getWaterTableClimate = WTClimateDaily[myDoy-1];  // start from 0
    return getWaterTableClimate;
}

bool WaterTable::computeWaterTableClimate(QDate currentDate, int yearFrom, int yearTo, float* myValue)
{

    *myValue = NODATA;

    int nrYears = yearTo - yearFrom + 1;
    float sumDepth = 0;
    int nrValidYears = 0;
    float myDepth;
    float myDelta;
    int myDeltaDays;

    for (int myYear = yearFrom; myYear <= yearTo; myYear++)
    {
        QDate myDate(myYear, currentDate.month(), currentDate.day());
        if (getWaterTableHindcast(myDate, &myDepth, &myDelta, &myDeltaDays))
        {
            nrValidYears = nrValidYears + 1;
            sumDepth = sumDepth + myDepth;
        }
    }

    if ( (nrValidYears / nrYears) >= meteoSettings.getMinimumPercentage() )
    {
        *myValue = sumDepth / nrValidYears;
        return true;
    }
    else
    {
        return false;
    }
}

// restituisce il dato interpolato considerando i dati osservati
bool WaterTable::getWaterTableHindcast(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays)
{
    *myValue = NODATA;
    *myDelta = NODATA;
    bool getWaterTableHindcast = false;
    if (!myDate.isValid())
    {
        error = "Wrong date";
        return getWaterTableHindcast;
    }
    if (!isCWBEquationReady)
    {
        return getWaterTableHindcast;
    }
    // first assessment
    float myWT_computation = getWaterTableDaily(myDate);
    if (myWT_computation == NODATA)
    {
        return getWaterTableHindcast;
    }
    // da qui in avanti è true (ha almeno il dato di stima)
    getWaterTableHindcast = true;
    float myWT = NODATA;
    float previousDz = NODATA;
    float nextDz = NODATA;
    float previosValue = NODATA;
    float nextValue = NODATA;
    int indexPrev = NODATA;
    int indexNext = NODATA;
    int diffWithNext = NODATA;
    int diffWithPrev = NODATA;
    QDate previousDate;
    QDate nextDate;
    int dT;

    // previuos and next observation
    QMap<QDate, int> myDepths = well.getDepths();
    QList<QDate> keys = myDepths.keys();
    int i = keys.indexOf(myDate);
    if (i != -1) // exact data found
    {
        if (i>0)
        {
            indexPrev = i - 1;
            previousDate = keys[indexPrev];
            previosValue = myDepths[previousDate];
        }
        if (i < keys.size()-1)
        {
            indexNext = i + 1;
            nextDate = keys[indexNext];
            nextValue = myDepths[nextDate];
        }
    }
    else
    {
        for (int i = 0; i<keys.size()-1; i++)
        {
            if (i == 0 && keys[i] > myDate)
            {
                indexNext = i;
                nextDate = keys[indexNext];
                nextValue = myDepths[nextDate];
                break;
            }
            else if (keys[i] < myDate && keys[i+1] > myDate)
            {
                indexPrev = i;
                previousDate = keys[indexPrev];
                previosValue = myDepths[previousDate];
                indexNext = i + 1;
                nextDate = keys[indexNext];
                nextValue = myDepths[nextDate];
                break;
            }
        }
    }
    if (indexPrev != NODATA)
    {
        myWT = getWaterTableDaily(previousDate);
        if (myWT != NODATA)
        {
            previousDz = previosValue - myWT;
        }
        diffWithPrev = previousDate.daysTo(myDate);
    }
    if (indexNext != NODATA)
    {
        myWT = getWaterTableDaily(nextDate);
        if (myWT != NODATA)
        {
            nextDz = nextValue - myWT;
        }
        diffWithNext = myDate.daysTo(nextDate);
    }

    // check lenght of missing data period
    if (previousDz != NODATA && nextDz != NODATA)
    {
        dT =  previousDate.daysTo(nextDate);
        if (dT > WATERTABLE_MAXDELTADAYS * 2)
        {
            if ( diffWithPrev <= diffWithNext)
            {
                nextDz = NODATA;
            }
            else
            {
                previousDz = NODATA;
            }
        }
    }
    if (previousDz != NODATA && nextDz != NODATA)
    {
        dT = previousDate.daysTo(nextDate);
        *myDelta = previousDz * (1 - (diffWithPrev / dT)) + nextDz * (1 - (diffWithNext / dT));
        *myDeltaDays = std::min(diffWithPrev, diffWithNext);
    }
    else if ( previousDz!= NODATA)
    {
        dT = diffWithPrev;
        *myDelta = previousDz * std::max((1 - (dT / WATERTABLE_MAXDELTADAYS)), 0);
        *myDeltaDays = dT;
    }
    else if ( nextDz!= NODATA)
    {
        dT = diffWithNext;
        *myDelta = nextDz * std::max((1 - (dT / WATERTABLE_MAXDELTADAYS)), 0);
        *myDeltaDays = dT;
    }
    else
    {
        // no observed value
        *myDelta = 0;
        *myDeltaDays = NODATA;
    }

    *myValue = myWT_computation + *myDelta;
    return getWaterTableHindcast;
}

void WaterTable::viewWaterTableSeries()
{
    QMap<QDate, int> myDepths = well.getDepths();
    QMapIterator<QDate, int> it(myDepths);

    myDates.clear();
    myHindcastSeries.clear();
    myInterpolateSeries.clear();
    float myDepth;
    float myDelta;
    int myDeltaDays;

    int numValues = well.getFirstDate().daysTo(lastMeteoDate) + 1;
    for (int i = 0; i< numValues; i++)
    {
        QDate myDate = well.getFirstDate().addDays(i);
        myDates.push_back(myDate);
        float computedValue = getWaterTableDaily(myDate.addDays(-1));
        myHindcastSeries.push_back(computedValue);
        getWaterTableHindcast(myDate.addDays(-1), &myDepth, &myDelta, &myDeltaDays);
        myInterpolateSeries.push_back(myDepth);
    }
}
