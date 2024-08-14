#include "waterTable.h"
#include "commonConstants.h"
#include "furtherMathFunctions.h"
#include <math.h>


WaterTable::WaterTable(std::vector<float> &inputTMin, std::vector<float> &inputTMax, std::vector<float> &inputPrec,
                       QDate firstMeteoDate, QDate lastMeteoDate, Crit3DMeteoSettings meteoSettings)
    : inputTMin(inputTMin), inputTMax(inputTMax), inputPrec(inputPrec), firstMeteoDate(firstMeteoDate), lastMeteoDate(lastMeteoDate), meteoSettings(meteoSettings)
{
}


void WaterTable::initializeWaterTable(Well myWell)
{
    well = myWell;

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
    RMSE = NODATA;
    avgDailyCWB = NODATA;
    errorStr = "";
}


void WaterTable::setInputTMin(const std::vector<float> &newInputTMin)
{
    inputTMin = newInputTMin;
}

void WaterTable::setInputTMax(const std::vector<float> &newInputTMax)
{
    inputTMax = newInputTMax;
}

void WaterTable::setInputPrec(const std::vector<float> &newInputPrec)
{
    inputPrec = newInputPrec;
}

void WaterTable::cleanAllMeteoVector()
{
    inputTMin.clear();
    inputTMax.clear();
    inputPrec.clear();
    etpValues.clear();
    precValues.clear();

    firstMeteoDate = QDate();
    lastMeteoDate = QDate();
}


bool WaterTable::computeWaterTableParameters(Well myWell, int stepDays)
{
    if (myWell.getObsDepthNr() == 0)
    {
        errorStr = "No WaterTable data loaded.";
        return false;
    }

    initializeWaterTable(myWell);
    isClimateReady = computeWTClimate();

    if (! computeETP_allSeries(true))
    {
        return false;
    }

    isCWBEquationReady = computeCWBCorrelation(stepDays);
    if (! isCWBEquationReady)
    {
        return false;
    }

    return computeWaterTableIndices();
}


bool WaterTable::computeWTClimate()
{
    if (well.getObsDepthNr() < 3)
    {
        errorStr = "Missing data";
        return false;
    }

    std::vector<float> H_sum;
    std::vector<float> H_num;
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        H_sum.push_back(0);
        H_num.push_back(0);
    }

    QMapIterator<QDate, float> it(well.depths);
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
            errorStr = "Missing watertable data: month " + QString::number(myMonthIndex+1);
            return false;
        }
        WTClimateMonthly[myMonthIndex] = H_sum[myMonthIndex] / H_num[myMonthIndex];
    }

    interpolation::cubicSplineYearInterpolate(WTClimateMonthly, WTClimateDaily);
    isClimateReady = true;

    return true;
}


bool WaterTable::setMeteoData(QDate myDate, float tmin, float tmax, float prec)
{
    int index = firstMeteoDate.daysTo(myDate);

    if (index < int(etpValues.size()) && index < int(precValues.size()))
    {
        Crit3DDate date = Crit3DDate(myDate.day(), myDate.month(), myDate.year());
        etpValues[index] = dailyEtpHargreaves(tmin, tmax, date, well.getLatitude(), &meteoSettings);
        precValues[index] = prec;
        return true;
    }
    else
    {
        return false;
    }
}


bool WaterTable::computeETP_allSeries(bool isUpdateAvgCWB)
{
    etpValues.clear();
    precValues.clear();

    if (inputTMin.size() != inputTMax.size() || inputTMin.size() != inputPrec.size())
    {
        errorStr = "Wrong weather size";
        return false;
    }

    float Tmin = NODATA;
    float Tmax = NODATA;
    float prec = NODATA;
    float etp = NODATA;
    double lat =  well.getLatitude();

    float sumCWB = 0;
    int nrValidDays = 0;
    int nrOfData = (int)inputTMin.size();
    for (int i = 0; i < nrOfData; i++)
    {
        QDate currentDate = firstMeteoDate.addDays(i);
        Crit3DDate myDate = Crit3DDate(currentDate.day(), currentDate.month(), currentDate.year());

        Tmin = inputTMin[i];
        Tmax = inputTMax[i];
        prec = inputPrec[i];
        etp = dailyEtpHargreaves(Tmin, Tmax, myDate, lat, &meteoSettings);

        etpValues.push_back(etp);
        precValues.push_back(prec);

        if (etp != NODATA && prec != NODATA)
        {
            sumCWB += prec - etp;
            nrValidDays++;
        }
    }

    if (isUpdateAvgCWB)
    {
        if (nrValidDays > 0)
        {
            avgDailyCWB = sumCWB / nrValidDays;
        }
        else
        {
            errorStr = "Missing data";
            return false;
        }
    }

    return true;
}


// Ricerca del periodo di correlazione migliore
bool WaterTable::computeCWBCorrelation(int stepDays)
{
    std::vector<float> myCWBSum;
    std::vector<float> myObsWT;
    float a = NODATA;
    float b = NODATA;
    float currentR2 = NODATA;
    float bestR2 = 0;
    float bestH0 = NODATA;
    float bestAlfaCoeff = NODATA;
    int bestNrDays = NODATA;

    int maxNrDays = 730;    // two years
    for (int nrDays = 90; nrDays <= maxNrDays; nrDays += stepDays)
    {
        myCWBSum.clear();
        myObsWT.clear();
        QMapIterator<QDate, float> it(well.depths);

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

        statistics::linearRegression(myCWBSum, myObsWT, int(myCWBSum.size()), false, &a, &b, &currentR2);
        if (currentR2 > bestR2)
        {
            bestR2 = currentR2;
            bestNrDays = nrDays;
            bestH0 = a;
            bestAlfaCoeff = b;
        }
    }

    if (bestR2 < 0.1)
    {
        return false;
    }

    nrObsData = int(myObsWT.size());
    nrDaysPeriod = bestNrDays;
    h0 = bestH0;
    alpha = bestAlfaCoeff;
    R2 = bestR2;
    isCWBEquationReady = true;

    return true;
}


// Climatic WaterBalance (CWB) on a nrDaysPeriod
double WaterTable::computeCWB(QDate myDate, int nrDays)
{
    double sumCWB = 0;
    int nrValidDays = 0;
    QDate actualDate;
    for (int shift = 1; shift <= nrDays; shift++)
    {
        actualDate = myDate.addDays(-shift);
        int index = firstMeteoDate.daysTo(actualDate);
        if (index >= 0 && index < int(precValues.size()))
        {
            float etp = etpValues[index];
            float prec = precValues[index];
            if ( etp != NODATA &&  prec != NODATA)
            {
                double currentCWB = double(prec - etp);
                double weight = 1 - double(shift-1) / double(nrDays);
                sumCWB += currentCWB * weight;
                nrValidDays++;
            }
        }
    }

    if (nrValidDays < (nrDays * meteoSettings.getMinimumPercentage() / 100))
    {
        errorStr = "Few Data";
        return NODATA;
    }

    // Climate
    double climateCWB = avgDailyCWB * nrDays * 0.5;

    // conversion: from [mm] to [cm]
    return (sumCWB - climateCWB) * 0.1;
}


// function to compute several statistical indices for watertable depth
bool WaterTable::computeWaterTableIndices()
{
    QMapIterator<QDate, float> it(well.depths);
    std::vector<float> myObs;
    std::vector<float> myComputed;
    std::vector<float> myClimate;
    float myIntercept, myCoeff;

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

    statistics::linearRegression(myObs, myComputed, int(myObs.size()), false, &myIntercept, &myCoeff, &R2);

    float mySum = 0;
    float mySumerrorStr = 0;
    float mySumDiffClimate = 0;
    float mySumDiffAvg = 0;
    float myErr = 0;
    float myErrAvg = 0;
    float myErrClimate = 0;

    int nrObs = int(myObs.size());
    for (int i=0; i<nrObs; i++)
    {
        mySum = mySum + myObs[i];
    }
    float myObsAvg = mySum / nrObs;

    for (int i=0; i<nrObs; i++)
    {
        myErr = myComputed[i] - myObs[i];
        mySumerrorStr = mySumerrorStr + myErr * myErr;
        myErrAvg = myObs[i] - myObsAvg;
        mySumDiffAvg = mySumDiffAvg + myErrAvg * myErrAvg;
        if (isClimateReady)
        {
            myErrClimate = myObs[i] - myClimate[i];
            mySumDiffClimate = mySumDiffClimate + myErrClimate * myErrClimate;
        }
    }

    RMSE = sqrt(mySumerrorStr / nrObs);

    if (isClimateReady)
    {
        EF = 1 - mySumerrorStr / mySumDiffClimate;
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
        if (getWaterTableInterpolation(myDate, &myDepth, &myDelta, &myDeltaDays))
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


// restituisce il dato interpolato di profondità considerando i dati osservati
// nella stessa unità di misura degli osservati (default: cm)
bool WaterTable::getWaterTableInterpolation(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays)
{
    *myValue = NODATA;
    *myDelta = NODATA;

    if (! myDate.isValid())
    {
        errorStr = "Wrong date";
        return false;
    }
    if (! isCWBEquationReady)
    {
        return false;
    }

    // first assessment
    float myWT_computation = getWaterTableDaily(myDate);
    if (myWT_computation == NODATA)
    {
        return false;
    }

    // da qui in avanti è true (ha almeno il dato di stima)
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

    QList<QDate> keys = well.depths.keys();

    // check previuos and next observed data
    int lastIndex = keys.size() - 1;
    int i = keys.indexOf(myDate);
    if (i != -1) // exact data found
    {
        indexPrev = i;
        previousDate = keys[indexPrev];
        previosValue = well.depths[previousDate];
        indexNext = i;
        nextDate = keys[indexNext];
        nextValue = well.depths[nextDate];
    }
    else
    {
        if (keys[0] > myDate)
        {
            indexNext = 0;
            nextDate = keys[indexNext];
            nextValue = well.depths[nextDate];
        }
        else if (keys[lastIndex] < myDate)
        {
            indexPrev = lastIndex;
            previousDate = keys[indexPrev];
            previosValue = well.depths[previousDate];
        }
        else
        {
            for (int i = 0; i < lastIndex; i++)
                if (keys[i] < myDate && keys[i+1] > myDate)
                {
                    indexPrev = i;
                    previousDate = keys[indexPrev];
                    previosValue = well.depths[previousDate];
                    indexNext = i + 1;
                    nextDate = keys[indexNext];
                    nextValue = well.depths[nextDate];
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
            if (diffWithPrev <= diffWithNext)
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
        if (dT == 0)
        {
            *myDelta = previousDz;
            *myDeltaDays = 0;
        }
        else
        {
            *myDelta = previousDz * (1.0 - (float(diffWithPrev) / float(dT))) + nextDz * (1.0 - (float(diffWithNext) / float(dT)));
            *myDeltaDays = std::min(diffWithPrev, diffWithNext);
        }
    }
    else if (previousDz != NODATA)
    {
        dT = diffWithPrev;
        *myDelta = previousDz * std::max((1.f - (float(dT) / float(WATERTABLE_MAXDELTADAYS))), 0.f);
        *myDeltaDays = dT;
    }
    else if (nextDz != NODATA)
    {
        dT = diffWithNext;
        *myDelta = nextDz * std::max((1.f - (float(dT) / float(WATERTABLE_MAXDELTADAYS))), 0.f);
        *myDeltaDays = dT;
    }
    else
    {
        // no observed value
        *myDelta = 0;
        *myDeltaDays = NODATA;
    }

    *myValue = myWT_computation + *myDelta;
    return true;
}


void WaterTable::computeWaterTableSeries()
{
    hindcastSeries.clear();
    interpolationSeries.clear();

    float myDelta;
    int myDeltaDays;

    firstDate = std::min(well.getFirstDate(), firstMeteoDate);
    int numValues = firstDate.daysTo(lastMeteoDate) + 1;

    for (int i = 0; i < numValues; i++)
    {
        QDate currentDate = firstDate.addDays(i);

        float currentDepth = getWaterTableDaily(currentDate);
        hindcastSeries.push_back(currentDepth);

        if (getWaterTableInterpolation(currentDate, &currentDepth, &myDelta, &myDeltaDays))
        {
            interpolationSeries.push_back(currentDepth);
        }
        else
        {
            interpolationSeries.push_back(NODATA);
        }
    }
}
