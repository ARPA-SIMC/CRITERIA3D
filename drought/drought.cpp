#include "drought.h"
#include "commonConstants.h"

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
    // TO DO
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
        */
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
        }
    }
    return true;
}

bool Drought::computeSpeiParameters()
{
    // TO DO
    return true;
}
