#include "drought.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "gammaFunction.h"
#include "meteo.h"

#include <algorithm>


Drought::Drought(droughtIndex index, int firstYear, int lastYear, Crit3DDate date, Crit3DMeteoPoint* meteoPoint, Crit3DMeteoSettings* meteoSettings)
{
    this->index = index;
    this->firstYear = firstYear;
    this->lastYear = lastYear;
    this->date = date;
    this->meteoPoint = meteoPoint;
    this->meteoSettings = meteoSettings;
    timeScale = 3; //default
    computeAll = false;  //default
    myVar = monthlyPrecipitation;  //default
    gammaStruct.beta = NODATA;
    gammaStruct.gamma = NODATA;
    gammaStruct.pzero = NODATA;
    logLogisticStruct.alpha = NODATA;
    logLogisticStruct.beta = NODATA;
    logLogisticStruct.gamma = NODATA;
    for (int i = 0; i<12; i++)
    {
        currentGamma.push_back(gammaStruct);
        currentLogLogistic.push_back(logLogisticStruct);
    }
    currentPercentileValue = NODATA;
}

droughtIndex Drought::getIndex() const
{
    return index;
}

void Drought::setIndex(const droughtIndex &value)
{
    index = value;
}

int Drought::getTimeScale() const
{
    return timeScale;
}

void Drought::setTimeScale(int value)
{
    timeScale = value;
}

int Drought::getFirstYear() const
{
    return firstYear;
}

void Drought::setFirstYear(int value)
{
    firstYear = value;
}

int Drought::getLastYear() const
{
    return lastYear;
}

void Drought::setLastYear(int value)
{
    lastYear = value;
}

bool Drought::getComputeAll() const
{
    return computeAll;
}

void Drought::setComputeAll(bool value)
{
    computeAll = value;
}


float Drought::computeDroughtIndex()
{
    timeScale = timeScale - 1; // index start from 0
    if (index == INDEX_SPI)
    {
        if (! computeSpiParameters())
        {
            return NODATA;
        }
    }
    else if (index == INDEX_SPEI)
    {
        if (! computeSpeiParameters())
        {
            return NODATA;
        }
    }

    int start, end;
    std::vector<float> mySum(meteoPoint->nrObsDataDaysM);
    for (int i = 0; i < meteoPoint->nrObsDataDaysM; i++)
    {
        droughtResults.push_back(NODATA);
    }

    if (computeAll)
    {
        start = timeScale;
        end = meteoPoint->nrObsDataDaysM;
        for (int i = 0; i <= timeScale; i++)
        {
            mySum[i] = NODATA;
        }
    }
    else
    {
        int currentYear = date.year;
        int currentMonth = date.month;
        end = (currentYear - meteoPoint->obsDataM[0]._year)*12 + currentMonth-meteoPoint->obsDataM[0]._month; // starts from 0
        start = end; // parte da 0
        if (end >= meteoPoint->nrObsDataDaysM)
        {
            return NODATA;
        }
        for (int i = 0; i < start-timeScale; i++)
        {
            mySum[i] = NODATA;
        }
    }

    for (int j = start; j <= end; j++)
    {
        mySum[j] = 0;
        for(int i = 0; i<=timeScale; i++)
        {
            if ((j-i)>=0 && j < meteoPoint->nrObsDataDaysM)
            {
                if (index == INDEX_SPI)
                {
                    if (meteoPoint->obsDataM[j-i].prec != NODATA)
                    {
                        mySum[j] = mySum[j] + meteoPoint->obsDataM[j-i].prec;
                    }
                    else
                    {
                        mySum[j] = NODATA;
                        break;
                    }
                }
                else if(index == INDEX_SPEI)
                {
                    if (meteoPoint->obsDataM[j-i].prec != NODATA && meteoPoint->obsDataM[j-i].et0_hs != NODATA)
                    {
                        mySum[j] = mySum[j] + meteoPoint->obsDataM[j-i].prec - meteoPoint->obsDataM[j-i].et0_hs;
                    }
                    else
                    {
                        mySum[j] = NODATA;
                        break;
                    }
                }
            }
            else
            {
                mySum[j] = NODATA;
            }
        }
    }

    for (int j = start; j <= end; j++)
    {
        int myMonthIndex = (j % 12)+1;  //start from 1

        if (mySum[j] != NODATA)
        {
            if (index == INDEX_SPI)
            {
                float gammaCDFRes = generalizedGammaCDF(mySum[j], currentGamma[myMonthIndex-1].beta, currentGamma[myMonthIndex-1].gamma, currentGamma[myMonthIndex-1].pzero);
                if (gammaCDFRes > 0 && gammaCDFRes < 1)
                {
                    droughtResults[j] = float(standardGaussianInvCDF(gammaCDFRes));
                }
            }
            else if(index == INDEX_SPEI)
            {
                float logLogisticRes = logLogisticCDF(mySum[j], currentLogLogistic[myMonthIndex-1].alpha, currentLogLogistic[myMonthIndex-1].beta, currentLogLogistic[myMonthIndex-1].gamma);
                if (logLogisticRes > 0 && logLogisticRes < 1)
                {
                    droughtResults[j] = float(standardGaussianInvCDF(logLogisticRes));
                }
            }
        }
    }

    return droughtResults[end];
}


bool Drought::computeSpiParameters()
{
    if (meteoPoint->nrObsDataDaysM == 0)
    {
        return false;
    }

    if (meteoPoint->obsDataM[0]._year > lastYear || meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year < firstYear)
    {
        return false;
    }
    int indexStart  = (firstYear - meteoPoint->obsDataM[0]._year)*12;
    if (indexStart < timeScale)
    {
        indexStart = timeScale;
    }
    if (meteoPoint->obsDataM[indexStart]._year > lastYear)
    {
        return false;
    }

    int lastYearStation = std::min(meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year, lastYear);

    int n = 0;
    float count = 0;
    int nTot = 0;
    std::vector<float> mySums;
    std::vector<float> monthSeries;
    float minPerc = meteoSettings->getMinimumPercentage();

    for (int j = indexStart; j<meteoPoint->nrObsDataDaysM; j++)
    {
        if (meteoPoint->obsDataM[j]._year > lastYearStation)
        {
            break;
        }
        count = 0;
        nTot = 0;
        mySums.push_back(0);
        for(int i = 0; i<= timeScale; i++)
        {
            nTot = nTot + 1;
            if (meteoPoint->obsDataM[j-i].prec != NODATA)
            {
                mySums[n] = mySums[n] + meteoPoint->obsDataM[j-i].prec;
                count = count + 1;
            }
            else
            {
                    mySums[n] = NODATA;
                    count = 0;
                    break;
            }
        }
        if ( (float)count / nTot < (minPerc / 100) )
        {
            mySums[n] = NODATA;
        }
        n = n + 1;
    }

    for (int i = 0; i<12; i++)
    {
        int myMonth = ((meteoPoint->obsDataM[indexStart]._month + i -1) % 12)+1;  //start from 1
        n = 0;

        monthSeries.clear();
        for (int j=i; j<mySums.size(); j=j+12)
        {
            if (mySums[j] != NODATA)
            {
                monthSeries.push_back(mySums[j]);
                n = n + 1;
            }
        }

        if ((float)n / (mySums.size()/12) >= minPerc / 100)
        {
            generalizedGammaFitting(monthSeries, n, &(currentGamma[myMonth-1].beta), &(currentGamma[myMonth-1].gamma),  &(currentGamma[myMonth-1].pzero));
        }
    }
    return true;
}

bool Drought::computeSpeiParameters()
{
    if (meteoPoint->nrObsDataDaysM == 0)
    {
        return false;
    }

    if (meteoPoint->obsDataM[0]._year > lastYear || meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year < firstYear)
    {
        return false;
    }

    int indexStart  = (firstYear - meteoPoint->obsDataM[0]._year)*12;
    if (indexStart < timeScale)
    {
        indexStart = timeScale;
    }
    if (meteoPoint->obsDataM[indexStart]._year > lastYear)
    {
        return false;
    }

    int lastYearStation = std::min(meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year, lastYear);

    int n = 0;
    float count = 0;
    int nTot = 0;
    std::vector<float> mySums;
    std::vector<float> monthSeries;
    std::vector<float> pwm(3);
    float minPerc = meteoSettings->getMinimumPercentage();

    for (int j = indexStart; j<meteoPoint->nrObsDataDaysM; j++)
    {
        if (meteoPoint->obsDataM[j]._year > lastYearStation)
        {
            break;
        }
        count = 0;
        nTot = 0;
        mySums.push_back(0);
        for(int i = 0; i<=timeScale; i++)
        {
            nTot = nTot + 1;
            if (meteoPoint->obsDataM[j-i].prec != NODATA && meteoPoint->obsDataM[j-i].et0_hs != NODATA)
            {
                mySums[n] = mySums[n] + meteoPoint->obsDataM[j-i].prec - meteoPoint->obsDataM[j-i].et0_hs;
                count = count + 1;
            }
            else
            {
                    mySums[n] = NODATA;
                    count = 0;
                    break;
            }
        }
        if ( (float)count / nTot < (minPerc / 100))
        {
            mySums[n] = NODATA;
        }
        n = n + 1;
    }

    for (int i = 0; i<12; i++)
    {

        int myMonth = ((meteoPoint->obsDataM[indexStart]._month + i -1) % 12)+1;  //start from 1
        n = 0;
        monthSeries.clear();
        for (int j=i; j<mySums.size(); j=j+12)
        {
            if (mySums[j] != NODATA)
            {
                monthSeries.push_back(mySums[j]);
                n = n + 1;
            }
        }

        if ((float)n / (mySums.size()/12) >= minPerc / 100)
        {
            // Sort values
            sorting::quicksortAscendingFloat(monthSeries, 0, monthSeries.size()-1);
            // Compute probability weighted moments
            probabilityWeightedMoments(monthSeries, n, pwm, 0, 0, false);
            // Fit a Log Logistic probability function
            logLogisticFitting(pwm, &currentLogLogistic[myMonth-1].alpha, &currentLogLogistic[myMonth-1].beta, &currentLogLogistic[myMonth-1].gamma);
        }
    }
    return true;
}

bool Drought::computePercentileValuesCurrentDay()
{
    if (myVar == noMeteoVar)
    {
        return false;
    }
    if (meteoPoint->obsDataM[0]._year > lastYear || meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year < firstYear)
    {
        return false;
    }
    int indexStart  = (firstYear - meteoPoint->obsDataM[0]._year)*12;
    if (meteoPoint->obsDataM[indexStart]._year > lastYear)
    {
        return false;
    }
    int lastYearStation = std::min(meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year, lastYear);
    int myMonth = meteoPoint->obsDataM[indexStart + date.month - 1]._month;
    std::vector<float> myValues;
    int nValid = 0;
    int nTot = 0;
    float minPerc = meteoSettings->getMinimumPercentage();

    for (int j = indexStart+myMonth-1; j < meteoPoint->nrObsDataDaysM ; j=j+12)
    {
        if (meteoPoint->obsDataM[j]._year > lastYearStation)
        {
            break;
        }
        Crit3DDate mydate(1,meteoPoint->obsDataM[j]._month,meteoPoint->obsDataM[j]._year);
        float myValue = meteoPoint->getMeteoPointValueM(mydate, myVar);
        if (myValue != NODATA)
        {
            myValues.push_back(myValue);
            nValid = nValid + 1;
            nTot = nTot + 1;
        }
        else
        {
            nTot = nTot + 1;
        }
    }
    if (nTot > 0)
    {
        if ((float)nValid/nTot >= minPerc / 100)
        {
            int index = (date.year - meteoPoint->obsDataM[0]._year)*12 + date.month -meteoPoint->obsDataM[0]._month; // starts from 0
            if (index < meteoPoint->nrObsDataDaysM)
            {
                Crit3DDate mydate(1,meteoPoint->obsDataM[index]._month,meteoPoint->obsDataM[index]._year);
                float myValue = meteoPoint->getMeteoPointValueM(mydate, myVar);
                if (myValue != NODATA)
                {
                    currentPercentileValue = sorting::percentileRank(myValues, myValue, true);
                }
            }
        }
    }
    return true;
}

void Drought::setMeteoPoint(Crit3DMeteoPoint *value)
{
    meteoPoint = value;
}

Crit3DMeteoSettings *Drought::getMeteoSettings() const
{
    return meteoSettings;
}

Crit3DDate Drought::getDate() const
{
    return date;
}

void Drought::setDate(const Crit3DDate &value)
{
    date = value;
}

float Drought::getCurrentPercentileValue() const
{
    return currentPercentileValue;
}

void Drought::setMyVar(const meteoVariable &value)
{
    myVar = value;
}

