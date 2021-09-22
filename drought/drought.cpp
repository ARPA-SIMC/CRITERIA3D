#include "drought.h"
#include "commonConstants.h"
#include <ctime>

Drought::Drought()
{

}

meteoVariable Drought::getVar() const
{
    return var;
}

void Drought::setVar(const meteoVariable &value)
{
    var = value;
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
    if (index == INDEX_SPI)
    {
        if (!computeSpiParameters())
        {
            return NODATA;
        }
    }
    else if (index == INDEX_SPEI)
    {
        if (!computeSpeiParameters())
        {
            return NODATA;
        }
    }
    int start;
    int end;
    std::vector<float> mySum(meteoPoint->nrObsDataDaysM);
    std::vector<float> myResults(meteoPoint->nrObsDataDaysM);
    if (computeAll)
    {
        start = timeScale;
        end = meteoPoint->nrObsDataDaysM;
        for (int i = 0; i < timeScale; i++)
        {
            mySum[i] = NODATA;
            myResults[i] = NODATA;
        }
    }
    else
    {
        time_t now = time(0);
        tm *ltm = localtime(&now);
        int currentYear = 1900 + ltm->tm_year;
        int currentMonth = 1 + ltm->tm_mon;
        end = (currentYear - meteoPoint->obsDataM[0]._year)*12 + currentMonth-meteoPoint->obsDataM[0]._month;
        start = end;
        if (end > meteoPoint->nrObsDataDaysM)
        {
            return NODATA;
        }
        for (int i = 0; i < start-timeScale; i++)
        {
            mySum[i] = NODATA;
            myResults[i] = NODATA;
        }
    }

    for (int j = start; j <= end; j++)
    {
        mySum[j] = 0;
        for(int i = 0; i<timeScale; i++)
        {
            if ((j-i)>0 && j<=meteoPoint->nrObsDataDaysM)
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
        int myMonthIndex = ((j - 1) % 12);
        myResults[j] = NODATA;

        if (mySum[j] != NODATA)
        {
            if (index == INDEX_SPI)
            {
                // TO DO
                // myResults[j] = standardGaussianInvCDF(math.gammaCDF(mySum(j), currentGamma(myMonthIndex)))
            }
            else if(index == INDEX_SPEI)
            {
                // TO DO
                // myResults[j] = standardGaussianInvCDF(math.logLogisticCDF(mySum(j), currentLogLogistic(myMonthIndex)))
            }
        }
    }

    return myResults[end];
}

bool Drought::computeSpiParameters()
{
    if (meteoPoint->obsDataM[0]._year > lastYear || meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year < firstYear)
    {
        return false;
    }
    int indexStart;
    if (firstYear == meteoPoint->obsDataM[0]._year)
    {
        indexStart = timeScale;
    }
    else
    {
        indexStart = (firstYear - meteoPoint->obsDataM[0]._year)*12 - (meteoPoint->obsDataM[0]._month-1);
        if (indexStart < timeScale)
        {
            indexStart = timeScale;
        }
    }
    if (meteoPoint->obsDataM[indexStart]._year > lastYear)
    {
        return false;
    }

    // int firstYearStation = std::max(meteoPoint->obsDataM[indexStart]._year, firstYear); // LC non viene mai usata nel codice vb
    int lastYearStation = std::min(meteoPoint->obsDataM[meteoPoint->nrObsDataDaysM-1]._year, lastYear);

    int n = 0;
    float count = 0;
    int nTot = 0;
    std::vector<float> mySums;
    std::vector<float> monthSeries;

    for (int j = indexStart; j<meteoPoint->nrObsDataDaysM; j++)
    {
        if (meteoPoint->obsDataM[j]._year <= lastYearStation)
        {
            count = 0;
            nTot = 0;
            mySums.push_back(0);
            for(int i = 0; i<timeScale; i++)
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
            if (count / nTot < meteoSettings->getMinimumPercentage() / 100)
            {
                mySums[n] = NODATA;
            }
            n = n + 1;
        }
        else
        {
            break;
        }
    }

    for (int i = 0; i<12; i++)
    {
        int myMonth = (meteoPoint->obsDataM[indexStart]._month + i) % 12;
        /*
         * TO DO
        currentGamma[myMonth].Beta = NODATA;
        currentGamma[myMonth].Gamma = NODATA;
        currentGamma[myMonth].Pzero = NODATA;
        LC non serve che siano array, usa solo il dato singolo da passare alla gammaFitting
        */
        float beta = NODATA;
        float gamma = NODATA;
        float pZero = NODATA;

        for (int j=i; j<mySums.size(); j=j+12)
        {
            if (mySums[j] != NODATA)
            {
                monthSeries.push_back(mySums[j]);
            }
        }

        if (monthSeries.size() / (mySums.size()/12) >= meteoSettings->getMinimumPercentage() / 100)
        {
            // TO DO
            // gammaFitting monthSeries, n, currentGamma(myMonth), average
            // gammaFitting monthSeries, n, beta, gamma, pZero average
        }
    }
    return true;
}

bool Drought::computeSpeiParameters()
{
    // TO DO
    return true;
}
