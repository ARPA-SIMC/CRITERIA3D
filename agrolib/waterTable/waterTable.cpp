#include "waterTable.h"
#include "commonConstants.h"
#include "furtherMathFunctions.h"
#include "basicMath.h"
#include "crit3dDate.h"

#include <math.h>


WaterTable::WaterTable()
{
    initializeWaterTable();
}


WaterTable::WaterTable(const std::vector<float> &inputTMin, const std::vector<float> &inputTMax, const std::vector<float> &inputPrec,
                       const QDate &firstMeteoDate, const QDate &lastMeteoDate, const Crit3DMeteoSettings &meteoSettings)
    : _meteoSettings(meteoSettings), _firstMeteoDate(firstMeteoDate), _lastMeteoDate(lastMeteoDate),
    _inputTMin(inputTMin), _inputTMax(inputTMax), _inputPrec(inputPrec)
{
    initializeWaterTable();
}


void WaterTable::initializeWaterTable()
{
    _isCWBEquationReady = false;
    _isClimateReady = false;

    _alpha = NODATA;
    _h0 = NODATA;
    _R2 = NODATA;
    _nrDaysPeriod = NODATA;
    _nrObsData = 0;
    _EF = NODATA;
    _RMSE = NODATA;
    _avgDailyCWB = NODATA;

    _errorStr = "";
}


void WaterTable::initializeWaterTable(const Well &myWell)
{
    _well = myWell;
    _well.updateDates();

    initializeWaterTable();
}


void WaterTable::cleanAllVectors()
{   
    _inputTMin.clear();
    _inputTMax.clear();
    _inputPrec.clear();
    _etpValues.clear();

    _hindcastSeries.clear();
    _interpolationSeries.clear();
}


bool WaterTable::initializeMeteoData(const QDate &firstDate, const QDate &lastDate)
{
    if (lastDate <= firstDate)
        return false;

    cleanAllVectors();

    _firstMeteoDate = firstDate;
    _lastMeteoDate = lastDate;
    int nrData = firstDate.daysTo(lastDate) +1;

    _inputTMin.resize(nrData, NODATA);
    _inputTMax.resize(nrData, NODATA);
    _inputPrec.resize(nrData, NODATA);
    _etpValues.resize(nrData, NODATA);

    return true;
}


bool WaterTable::setMeteoData(const QDate &date, float tmin, float tmax, float prec)
{
    int index = _firstMeteoDate.daysTo(date);

    if (index < int(_etpValues.size()) && index < int(_inputPrec.size()))
    {
        Crit3DDate myDate = Crit3DDate(date.day(), date.month(), date.year());
        _etpValues[index] = dailyEtpHargreaves(tmin, tmax, myDate, _well.getLatitude(), &_meteoSettings);
        _inputPrec[index] = prec;
        return true;
    }

    return false;
}


void WaterTable::setParameters(int nrDaysPeriod, double alpha, double h0, double avgDailyCWB)
{
    _nrDaysPeriod = nrDaysPeriod;
    _alpha = alpha;
    _h0 = h0;
    _avgDailyCWB = avgDailyCWB;
    _isCWBEquationReady = true;
}


bool WaterTable::computeWaterTableParameters(const Well &myWell, int stepDays)
{
    if (myWell.getObsDepthNr() == 0)
    {
        _errorStr = "No WaterTable data loaded.";
        return false;
    }

    initializeWaterTable(myWell);
    _isClimateReady = computeWTClimate();

    if (! computeWholeSeriesETP(true))
    {
        return false;
    }

    if (! computeCWBCorrelation(stepDays))
    {
        return false;
    }

    return computeWaterTableIndices();
}


bool WaterTable::computeWTClimate()
{
    if (_well.getObsDepthNr() < 12)
    {
        _errorStr = "Missing data";
        return false;
    }

    std::vector<float> H_sum;
    std::vector<float> H_num;
    for (int m = 0; m < 12; m++)
    {
        H_sum.push_back(0);
        H_num.push_back(0);
    }

    QMapIterator<QDate, float> it(_well.depths);
    while (it.hasNext())
    {
        it.next();
        QDate myDate = it.key();
        int myValue = it.value();
        int myMonth = myDate.month();
        int monthIndex = myMonth - 1;
        H_sum[monthIndex] += myValue;
        H_num[monthIndex]++;
    }

    for (int m = 0; m < 12; m++)
    {
        if (H_num[m] < 2)
        {
            _errorStr = "Missing watertable data: month " + QString::number(m+1);
            return false;
        }
        _WTClimateMonthly[m] = H_sum[m] / H_num[m];
    }

    interpolation::cubicSplineYearInterpolate(_WTClimateMonthly, _WTClimateDaily);
    _isClimateReady = true;

    return true;
}


float WaterTable::getHindcast(int index) const
{
    if (index >= (int)_hindcastSeries.size())
    {
        return NODATA;
    }

    return _hindcastSeries[index];
}


float WaterTable::getInterpolatedData(int index) const
{
    if (index >= (int)_interpolationSeries.size())
    {
        return NODATA;
    }

    return _interpolationSeries[index];
}


bool WaterTable::computeWholeSeriesETP(bool isUpdateAvgCWB)
{
    _etpValues.clear();

    if (_inputTMin.size() != _inputTMax.size() || _inputTMin.size() != _inputPrec.size())
    {
        _errorStr = "Meteo series have different sizes.";
        return false;
    }

    float Tmin = NODATA;
    float Tmax = NODATA;
    float prec = NODATA;
    float etp = NODATA;
    double lat =  _well.getLatitude();

    float sumCWB = 0;
    int nrValidDays = 0;
    int nrOfData = (int)_inputPrec.size();
    for (int i = 0; i < nrOfData; i++)
    {
        QDate myCurrentDate = _firstMeteoDate.addDays(i);
        Crit3DDate myDate = Crit3DDate(myCurrentDate.day(), myCurrentDate.month(), myCurrentDate.year());

        Tmin = _inputTMin[i];
        Tmax = _inputTMax[i];
        prec = _inputPrec[i];
        etp = dailyEtpHargreaves(Tmin, Tmax, myDate, lat, &_meteoSettings);

        _etpValues.push_back(etp);

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
            _avgDailyCWB = sumCWB / nrValidDays;
        }
        else
        {
            _errorStr = "Missing data to compute CWB.";
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

    int maxNrDays = 730;                // [days] two years
    for (int nrDays = 90; nrDays <= maxNrDays; nrDays += stepDays)
    {
        myCWBSum.clear();
        myObsWT.clear();
        QMapIterator<QDate, float> it(_well.depths);

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

    _nrObsData = int(myObsWT.size());
    _nrDaysPeriod = bestNrDays;
    _h0 = bestH0;
    _alpha = bestAlfaCoeff;
    _R2 = bestR2;
    _isCWBEquationReady = true;

    return true;
}


// compute Climatic Water Balance (CWB) on a nrDaysPeriod
// expressed as anomaly in [cm] with average value
double WaterTable::computeCWB(const QDate &myDate, int nrDays)
{
    double sumCWB = 0;
    int nrValidDays = 0;
    QDate actualDate;
    for (int shift = 1; shift <= nrDays; shift++)
    {
        actualDate = myDate.addDays(-shift);
        int index = _firstMeteoDate.daysTo(actualDate);
        if (index >= 0 && index < int(_inputPrec.size()))
        {
            float etp = _etpValues[index];
            float prec = _inputPrec[index];
            if (! isEqual(etp, NODATA) && ! isEqual(prec, NODATA))
            {
                double currentCWB = double(prec - etp);
                double weight = 1 - double(shift-1) / double(nrDays);
                sumCWB += currentCWB * weight;
                nrValidDays++;
            }
        }
    }

    if (nrValidDays < (nrDays * _meteoSettings.getMinimumPercentage() / 100))
    {
        _errorStr = "Not enough data";
        return NODATA;
    }

    // Climate
    double climateCWB = _avgDailyCWB * nrDays * 0.5;

    // conversion: from [mm] to [cm]
    return (sumCWB - climateCWB) * 0.1;
}


// function to compute several statistical indices for watertable depth
bool WaterTable::computeWaterTableIndices()
{
    QMapIterator<QDate, float> it(_well.depths);
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

    statistics::linearRegression(myObs, myComputed, int(myObs.size()), false, &myIntercept, &myCoeff, &_R2);

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
        if (_isClimateReady)
        {
            myErrClimate = myObs[i] - myClimate[i];
            mySumDiffClimate = mySumDiffClimate + myErrClimate * myErrClimate;
        }
    }

    _RMSE = sqrt(mySumerrorStr / nrObs);

    if (_isClimateReady)
    {
        _EF = 1 - mySumerrorStr / mySumDiffClimate;
    }
    else
    {
        _EF = NODATA;
    }
    return true;
}


// return assessement value of watertable depth [cm]
double WaterTable::getWaterTableDaily(const QDate &myDate)
{
    if (_isCWBEquationReady)
    {
        double deltaCWB = computeCWB(myDate, _nrDaysPeriod);
        if (deltaCWB != NODATA)
        {
            return std::max(0., _h0 + _alpha * deltaCWB);            // [cm]
        }
    }

    // climatic value
    if (_isClimateReady)
    {
        return getWaterTableClimate(myDate);
    }

    // missing data
    return NODATA;
}


float WaterTable::getObsDepth(const QDate &myDate) const
{
    if (! _well.depths.contains(myDate))
        return NODATA;

    return _well.depths[myDate];
}


float WaterTable::getWaterTableClimate(const QDate &myDate) const
{
    if (! _isClimateReady)
    {
        return NODATA;
    }

    int doy = myDate.dayOfYear();
    return _WTClimateDaily[doy-1];     // start from 0
}


bool WaterTable::computeWaterTableClimate(const QDate &currentDate, int yearFrom, int yearTo, float &myValue)
{
    myValue = NODATA;

    int nrRequestedYears = yearTo - yearFrom + 1;
    float sumDepth = 0;
    int nrValidYears = 0;
    float myDepth, myDelta;
    int myDeltaDays;

    for (int myYear = yearFrom; myYear <= yearTo; myYear++)
    {
        QDate myDate(myYear, currentDate.month(), currentDate.day());
        if (getWaterTableInterpolation(myDate, myDepth, myDelta, myDeltaDays))
        {
            nrValidYears = nrValidYears + 1;
            sumDepth = sumDepth + myDepth;
        }
    }

    // check nr of data
    if ((nrValidYears / nrRequestedYears) < _meteoSettings.getMinimumPercentage())
        return false;

    myValue = sumDepth / nrValidYears;
    return true;
}


// restituisce il dato interpolato di profondità considerando i dati osservati
// nella stessa unità di misura degli osservati (default: cm)
bool WaterTable::getWaterTableInterpolation(const QDate &myDate, float &myValue, float &myDelta, int &deltaDays)
{
    myValue = NODATA;
    myDelta = NODATA;
    deltaDays = NODATA;

    if (! myDate.isValid())
    {
        _errorStr = "Wrong date";
        return false;
    }
    if (! _isCWBEquationReady)
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

    QList<QDate> keys = _well.depths.keys();

    // check previuos and next observed data
    int lastIndex = keys.size() - 1;
    int i = keys.indexOf(myDate);
    if (i != -1) // exact data found
    {
        indexPrev = i;
        previousDate = keys[indexPrev];
        previosValue = _well.depths[previousDate];
        indexNext = i;
        nextDate = keys[indexNext];
        nextValue = _well.depths[nextDate];
    }
    else
    {
        if (keys[0] > myDate)
        {
            indexNext = 0;
            nextDate = keys[indexNext];
            nextValue = _well.depths[nextDate];
        }
        else if (keys[lastIndex] < myDate)
        {
            indexPrev = lastIndex;
            previousDate = keys[indexPrev];
            previosValue = _well.depths[previousDate];
        }
        else
        {
            for (int i = 0; i < lastIndex; i++)
                if (keys[i] < myDate && keys[i+1] > myDate)
                {
                    indexPrev = i;
                    previousDate = keys[indexPrev];
                    previosValue = _well.depths[previousDate];
                    indexNext = i + 1;
                    nextDate = keys[indexNext];
                    nextValue = _well.depths[nextDate];
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
        int dT =  previousDate.daysTo(nextDate);
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
        int dT = previousDate.daysTo(nextDate);
        if (dT == 0)
        {
            myDelta = previousDz;
            deltaDays = 0;
        }
        else
        {
            myDelta = previousDz * (1.0 - (float(diffWithPrev) / float(dT))) + nextDz * (1.0 - (float(diffWithNext) / float(dT)));
            deltaDays = std::min(diffWithPrev, diffWithNext);
        }
    }
    else if (previousDz != NODATA)
    {
        int dT = diffWithPrev;
        myDelta = previousDz * std::max((1.f - (float(dT) / float(WATERTABLE_MAXDELTADAYS))), 0.f);
        deltaDays = dT;
    }
    else if (nextDz != NODATA)
    {
        int dT = diffWithNext;
        myDelta = nextDz * std::max((1.f - (float(dT) / float(WATERTABLE_MAXDELTADAYS))), 0.f);
        deltaDays = dT;
    }
    else
    {
        // no observed value
        myDelta = 0;
        deltaDays = NODATA;
    }

    myValue = myWT_computation + myDelta;
    return true;
}


void WaterTable::computeWaterTableSeries()
{
    _hindcastSeries.clear();
    _interpolationSeries.clear();

    QDate firstDate = std::min(_well.getFirstObsDate(), _firstMeteoDate);
    int numValues = firstDate.daysTo(_lastMeteoDate) + 1;

    for (int i = 0; i < numValues; i++)
    {
        QDate currentDate = firstDate.addDays(i);

        float currentDepth = getWaterTableDaily(currentDate);
        _hindcastSeries.push_back(currentDepth);

        int deltaDays;
        float interpolationDepth, deltaDepth;
        if (getWaterTableInterpolation(currentDate, interpolationDepth, deltaDepth, deltaDays))
        {
            _interpolationSeries.push_back(interpolationDepth);
        }
        else
        {
            _interpolationSeries.push_back(NODATA);
        }
    }
}
