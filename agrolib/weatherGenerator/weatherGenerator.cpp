/*!
======================================================================================================
    \name weatherGenerator.cpp
======================================================================================================
    \copyright
    2016 Fausto Tomei, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

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
    ftomei@arpae.it

======================================================================================================
    \brief
    Generate weather data from monthly mean and standard deviation information.

    Based on:
    Richardson, C. W. and D. A. Wright
    WGEN: A model for generating daily weather variables.
    USDA, ARS, publication ARS-8, 1984.

    Modified by:
    G. S. Campbell, Dept. of Crop and Soil Sciences,
    Washington State University, Pullman, WA 99164-6420.
    last modified Nov 1991.
======================================================================================================
    \details
    Richardson uses a 1-term Fourier series to model temperatures, while this model
    uses quadratic spline interpolation, set up so that monthly averages are correct.
    This model also uses the quadratic spline to interpolate between monthly values of other variables.
    This model uses a Weibull distribution to generate rain amounts
    which requires only knowing the mean rainfall on wet days.
    (see Selker and Haith, Water Resour. Res. 26:2733, 1990)
    Transition probabilities for WD and WW days are computed from fraction of wet days
    (see Geng et al., Agric. Forest Meteorol. 36:363, 1986)
======================================================================================================
    \param
    (Tmax) monthly maximum temp.  (C)
    (Tmin) monthly minimum temp.  (C)
    (Prcp) total monthly precip        (mm)
    (fwet) fraction of wet days        (must be > 0)
    (Td-Tw)difference between maximum temperatures on dry and wet days (C)
    (Txsd) maximum temperature standard deviation   (C)
    (Tnsd) minimum temperature standard deviation   (C)
======================================================================================================
*/

#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>
#include <float.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#include <QDebug>
#include <QFile>

#include "crit3dDate.h"
#include "weatherGenerator.h"
#include "commonConstants.h"
#include "timeUtility.h"
#include "fileUtility.h"
#include "furtherMathFunctions.h"


float getTMax(int dayOfYear, float precThreshold, TweatherGenClimate* wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen->state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return wGen->state.currentTmax;
}


float getTMin(int dayOfYear, float precThreshold, TweatherGenClimate* wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen->state.currentDay)
        newDay(dayOfYear, precThreshold,  wGen);

    return wGen->state.currentTmin;
}


float getTAverage(int dayOfYear, float precThreshold, TweatherGenClimate* wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen->state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return 0.5f * (wGen->state.currentTmax + wGen->state.currentTmin);
}


float getPrecip(int dayOfYear, float precThreshold, TweatherGenClimate* wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen->state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return wGen->state.currentPrec;
}


// main function
void newDay(int dayOfYear, float precThreshold, TweatherGenClimate* wGen)
{
    float meanTMax, meanTMin, stdTMax, stdTMin;

    // daily structure is 0-365
    dayOfYear = dayOfYear - 1;

    //Precipitation
    bool isWetDay = markov(wGen->daily.pwd[dayOfYear], wGen->daily.pww[dayOfYear], wGen->state.wetPreviousDay);

    if (isWetDay)
    {
        meanTMax = wGen->daily.meanWetTMax[dayOfYear];
        wGen->state.currentPrec = weibull(wGen->daily.meanPrecip[dayOfYear], precThreshold);
    }
    else
    {
        meanTMax = wGen->daily.meanDryTMax[dayOfYear];
        wGen->state.currentPrec = 0;
    }

    //store information
    wGen->state.wetPreviousDay = isWetDay;

    //temperature
    meanTMin = wGen->daily.meanTMin[dayOfYear];
    stdTMax = wGen->daily.maxTempStd[dayOfYear];
    stdTMin = wGen->daily.minTempStd[dayOfYear];

    genTemps(&wGen->state.currentTmax, &wGen->state.currentTmin, meanTMax, meanTMin, stdTMax, stdTMin, &(wGen->state.resTMaxPrev), &(wGen->state.resTMinPrev));

    wGen->state.currentDay = dayOfYear;
}


void initializeWeather(TweatherGenClimate* wGen)
{
    float mpww[12];
    float mpwd[12];
    float mMeanPrecip[12];
    float fWetDays[12];
    float mMeanDryTMax[12];
    float mMeanWetTMax[12];
    float mMeanTMax[12];
    float mMeanTMin[12];
    float mMeanDiff[12];
    float mMaxTempStd[12];
    float mMinTempStd[12];
    float aveTMax = 0;
    float aveTMin = 0;
    float sumPrecip = 0;
    int daysInMonth;
    int m;

    wGen->state.currentDay = NODATA;
    wGen->state.currentTmax = NODATA;
    wGen->state.currentTmin = NODATA;
    wGen->state.currentPrec = NODATA;
    // TODO: pass residual data of last observed day
    wGen->state.resTMaxPrev = 0;
    wGen->state.resTMinPrev = 0;
    wGen->state.wetPreviousDay = false;

    for (int i = 0; i < 366; i++)
    {
        wGen->daily.maxTempStd[i] = 0;
        wGen->daily.meanDryTMax[i] = 0;
        wGen->daily.meanTMin[i] = 0;
        wGen->daily.meanPrecip[i] = 0;
        wGen->daily.meanWetTMax[i] = 0;
        wGen->daily.minTempStd[i] = 0;
        wGen->daily.pwd[i] = 0;
        wGen->daily.pww[i] = 0;
    }

    for (m = 0; m < 12; m++)
    {
        mMeanTMax[m] = wGen->monthly.monthlyTmax[m];
        mMeanTMin[m] = wGen->monthly.monthlyTmin[m];
        mMeanPrecip[m] = wGen->monthly.sumPrec[m];
        fWetDays[m] = wGen->monthly.fractionWetDays[m];
        mpww[m] = wGen->monthly.probabilityWetWet[m];
        mMeanDiff[m] = wGen->monthly.dw_Tmax[m];
        mMaxTempStd[m] = wGen->monthly.stDevTmax[m];
        mMinTempStd[m] = wGen->monthly.stDevTmin[m];
        aveTMax = aveTMax + mMeanTMax[m];
        aveTMin = aveTMin + mMeanTMin[m];
        sumPrecip = sumPrecip + mMeanPrecip[m];
    }

    aveTMax = aveTMax / 12;
    aveTMin = aveTMin / 12;

    for (m = 0; m < 12; m++)
    {
        mMeanDryTMax[m] = mMeanTMax[m] + fWetDays[m] * mMeanDiff[m];
        mMeanWetTMax[m] = mMeanDryTMax[m] - mMeanDiff[m];

        mpwd[m] = (1.f - mpww[m]) * (fWetDays[m] / (1.f - fWetDays[m]));

        daysInMonth = getDaysInMonth(m+1,2001); // year = 2001 is to avoid leap year

        // convert from total mm/month to average mm/wet day
        mMeanPrecip[m] = mMeanPrecip[m] / (fWetDays[m] * daysInMonth);
    }

    cubicSplineYearInterpolate(mpww, wGen->daily.pww);
    cubicSplineYearInterpolate(mpwd, wGen->daily.pwd);
    cubicSplineYearInterpolate(mMeanDryTMax, wGen->daily.meanDryTMax);
    cubicSplineYearInterpolate(mMeanPrecip, wGen->daily.meanPrecip);
    cubicSplineYearInterpolate(mMeanWetTMax, wGen->daily.meanWetTMax);
    cubicSplineYearInterpolate(mMeanTMin, wGen->daily.meanTMin);
    cubicSplineYearInterpolate(mMinTempStd, wGen->daily.minTempStd);
    cubicSplineYearInterpolate(mMaxTempStd, wGen->daily.maxTempStd);

}


/*!
  * \brief Generate two standard normally-distributed random numbers
  * \cite  Numerical Recipes in Pascal, W. H. Press et al. 1989, p. 225
*/
void normalRandom(float *rnd_1, float *rnd_2)
{
    double rnd, factor, r, v1, v2;

    do
    {
        rnd = double(rand()) / double(RAND_MAX);
        v1 = 2.0 * rnd - 1.0;
        rnd = double(rand()) / double(RAND_MAX);
        v2 = 2.0 * rnd - 1.0;
        r = v1 * v1 + v2 * v2;
    }
    while ((r <= 0) || (r >= 1)); // see if they are in the unit circle, and if they are not, try again.

    // Box-Muller transformation to get two normal deviates
    factor = sqrt(-2.0 * log(r) / r);

    // Gaussian random deviates
    *rnd_1 = float(v1 * factor);
    *rnd_2 = float(v2 * factor);
}


/*!
 * \brief dry/wet markov chain
 * \param pwd     probability wet-dry
 * \param pww     probability wet-wet
 * \param isWetPreviousDay  true if the previous day has been a wet day, false otherwise
 * \return true if the day is wet, false otherwise
 */
bool markov(float pwd,float pww, bool isWetPreviousDay)
{
    double c;

    if (isWetPreviousDay)
        c = double(rand()) / double(RAND_MAX) - double(pww);

    else
        c = double(rand()) / double(RAND_MAX) - double(pwd);


    if (c <= 0)
        return true;  // wet
    else
        return false; // dry
}


/*!
  * \brief weibull distribution uses only avg precipitation (computed on wet days)
  * \returns precipitation [mm]
*/
float weibull (float dailyAvgPrec, float precThreshold)
{
    double r = 0;
    while (r < EPSILON)
    {
        r = double(rand()) / double(RAND_MAX);
    }

    double w = 0.84 * double(dailyAvgPrec) * pow(-log(r), 1.3333);

    if (w > double(precThreshold))
        return float(w);
    else
        return precThreshold;
}



/*!
  * \brief Computes daily values starting from monthly mean
  * using cubic spline
*/
void cubicSplineYearInterpolate(float *meanY, float *dayVal)
{
    double monthMid [16] = {-61, - 31, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365, 396};

    for (int iMonth=0; iMonth<16; iMonth++)
    {
        monthMid[iMonth] += 15;
    }

    double* averageMonthlyAmountPrecLarger = new double[16];
    for (int iMonth = 0; iMonth < 12; iMonth++)
    {
        averageMonthlyAmountPrecLarger[iMonth+2] = double(meanY[iMonth]);
    }

    averageMonthlyAmountPrecLarger[0] = double(meanY[10]);
    averageMonthlyAmountPrecLarger[1] = double(meanY[11]);
    averageMonthlyAmountPrecLarger[14] = double(meanY[0]);
    averageMonthlyAmountPrecLarger[15] = double(meanY[1]);

    for (int jjj=0; jjj<365; jjj++)
    {
        dayVal[jjj] = interpolation::cubicSpline(jjj*1.0, monthMid, averageMonthlyAmountPrecLarger, 16);
    }

    delete [] averageMonthlyAmountPrecLarger;
}


/*!
  * \brief Computes daily values starting from monthly mean
  * using quadratic spline
  * original Campbell function
  * it has a discontinuity between end and start of the year
*/
void quadrSplineYearInterpolate(float *meanY, float *dayVal)
{
    float a[13] = {0};
    float b[14] = {0};
    float c[13] = {0};;
    float aa[13] = {0};
    float bb[13] = {0};
    float cc[13] = {0};
    float d[14] = {0};
    float h[13] = {0};
    float t = 0;

    int i,j;

    int monthends [13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};

    d[1] = meanY[0] - meanY[11];
    h[0] = 30;

    for (i = 1; i<=12; i++)
    {
        if (i == 12)
            d[i + 1] = meanY[0] - meanY[i-1];
        else
            d[i + 1] = meanY[i] - meanY[i-1];

        h[i] = monthends[i] - monthends[i - 1] - 1;
        aa[i] = h[i - 1] / 6;
        bb[i] = (h[i - 1] + h[i]) / 3;
        cc[i] = h[i] / 6;
    }

    for (i = 1; i<= 11; i++)
    {
        cc[i] = cc[i] / bb[i];
        d[i] = d[i] / bb[i];
        bb[i + 1] = bb[i + 1] - aa[i + 1] * cc[i];
        d[i + 1] = d[i + 1] - aa[i + 1] * d[i];
    }

    b[12] = d[12] / bb[12];
    for (i = 11; i>=1; i--)
        b[i] = d[i] - cc[i] * b[i + 1];

    for (i = 1; i<=12; i++)
    {
        a[i] = (b[i + 1] - b[i]) / (2 * h[i]);
        c[i] = meanY[i-1] - (b[i + 1] + 2 * b[i]) * h[i] / 6;
    }

    j = 0;
    for (i = 1; i<=365; i++)
    {
        if (monthends[j] < i)
            j = j + 1;
        t = i - monthends[j - 1] - 1;

        dayVal[i-1] = c[j] + b[j] * t + a[j] * t * t;

    }

    dayVal[365] = dayVal[0];
}


/*!
  * \brief generates maximum and minimum temperature
*/
void genTemps(float *tMax, float *tMin, float meanTMax, float meanTMin, float stdMax, float stdMin, float *resTMaxPrev, float *resTMinPrev)
{
    // matrix of serial correlation coefficients.
    float serialCorrelation[2][2]=
    {
        {0.567f, 0.086f},
        {0.253f, 0.504f}
    };

    // matrix of cross correlation coefficients.
    float crossCorrelation[2][2]=
    {
        {0.781f, 0.0f},
        {0.328f, 0.637f}
    };

    // standard normal random value for TMax and TMin
    float NorTMin, NorTMax;
    normalRandom(&NorTMin, &NorTMax);

    float resTMaxCurr, resTMinCurr;
    resTMaxCurr = crossCorrelation[0][0] * NorTMax + serialCorrelation[0][0] * (*resTMaxPrev) + serialCorrelation[0][1] * (*resTMinPrev);
    resTMinCurr = crossCorrelation[1][0] * NorTMax + crossCorrelation[1][1] * NorTMin + serialCorrelation[1][0] * (*resTMaxPrev) + serialCorrelation[1][1] * (*resTMinPrev);

    // residual tmax for previous day
    *resTMaxPrev = resTMaxCurr;
    // residual tmin for previous day
    *resTMinPrev = resTMinCurr;

    *tMax = resTMaxCurr * stdMax + meanTMax;
    *tMin = resTMinCurr * stdMin + meanTMin;

    if (*tMin > *tMax)
    {
        NorTMax = *tMin;
        *tMin = *tMax;
        *tMax = NorTMax;
    }

    // minimum deltaT (TODO improve)
    if (*tMax - *tMin < 1)
        *tMin = *tMax - 1;
}


bool assignXMLAnomaly(TXMLSeasonalAnomaly* XMLAnomaly, int modelIndex, int anomalyMonth1, int anomalyMonth2, TweatherGenClimate* wGenNoAnomaly, TweatherGenClimate* wGen)
{
    unsigned int i = 0;
    QString myVar;
    float myValue = 0.0;

    bool result;

    // loop for all XMLValuesList (Tmin, Tmax, TminVar, TmaxVar, Prec3M, Wetdays)
    for (i = 0; i < XMLAnomaly->forecast.size(); i++)
    {
        if (XMLAnomaly->forecast[i].attribute.toUpper() == "ANOMALY")
        {
            myVar = XMLAnomaly->forecast[i].type.toUpper();
            result = false;

            if (XMLAnomaly->forecast[i].value[modelIndex] != nullptr)
                myValue = XMLAnomaly->forecast[i].value[modelIndex].toFloat();
            else
                myValue = NODATA;

            if (int(myValue) != int(NODATA))
            {
                if ( (myVar == "TMIN") || (myVar == "AVGTMIN") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.monthlyTmin, wGen->monthly.monthlyTmin);
                else if ( (myVar == "TMAX") || (myVar == "AVGTMAX") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.monthlyTmax, wGen->monthly.monthlyTmax);
                else if ( (myVar == "PREC3M") || (myVar == "PREC") )
                    result = assignAnomalyPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.sumPrec, wGen->monthly.sumPrec);
                else if ( (myVar == "WETDAYSFREQUENCY") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.fractionWetDays, wGen->monthly.fractionWetDays);
                else if ( (myVar == "WETWETDAYSFREQUENCY") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.probabilityWetWet, wGen->monthly.probabilityWetWet);
                else if ( (myVar == "DELTATMAXDRYWET") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly->monthly.dw_Tmax, wGen->monthly.dw_Tmax);
            }
            else
            {
                // not critical variables
                if ((myVar == "DELTATMAXDRYWET") || (myVar == "WETWETDAYSFREQUENCY"))
                result = true;
            }

            if (result == false)
            {
                qDebug() << "wrong anomaly: " + myVar;
                return false;
            }
        }
    }

    /* DEBUG
    QString anomaly="anomaly.txt";
    QFile file(anomaly);
    file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    QTextStream stream( &file );
    for (int m = 0; m < 12; m++)
    {
        stream << "month = " << m +1 << endl;
        stream << "wGen->monthly.monthlyTmin = " << wGen->monthly.monthlyTmin[m] << endl;
        stream << "wGen->monthly.monthlyTmax = " << wGen->monthly.monthlyTmax[m] << endl;
        stream << "wGen->monthly.sumPrec = " << wGen->monthly.sumPrec[m] << endl;
        stream << "wGen->monthly.stDevTmin[m] = " << wGen->monthly.stDevTmin[m] << endl;
        stream << "wGen->monthly.stDevTmax = " << wGen->monthly.stDevTmax[m] << endl;
        stream << "wGen->monthly.fractionWetDays[m] = " << wGen->monthly.fractionWetDays[m] << endl;
        stream << "wGen->monthly.probabilityWetWet[m] = " << wGen->monthly.probabilityWetWet[m] << endl;
        stream << "wGen->monthly.dw_Tmax[m] = " << wGen->monthly.dw_Tmax[m] << endl;
        stream << "-------------------------------------------" << endl;
    }
    */

    return true;
}


bool assignAnomalyNoPrec(float myAnomaly, int anomalyMonth1, int anomalyMonth2, float* myWGMonthlyVarNoAnomaly, float* myWGMonthlyVar )
{
    int myMonth = 0;

    if (anomalyMonth2 >= anomalyMonth1)
    {
        // regular period
        for (myMonth = anomalyMonth1; myMonth <= anomalyMonth2; myMonth++)
            myWGMonthlyVar[myMonth-1] = myWGMonthlyVarNoAnomaly[myMonth-1] + myAnomaly;

    }
    else
    {
        // irregular period (between years)
        for (myMonth = anomalyMonth1; myMonth <= 12; myMonth++)
            myWGMonthlyVar[myMonth-1] = myWGMonthlyVarNoAnomaly[myMonth-1] + myAnomaly;

        for (myMonth = 1; myMonth <=anomalyMonth2; myMonth++)
            myWGMonthlyVar[myMonth-1] = myWGMonthlyVarNoAnomaly[myMonth-1] + myAnomaly;
    }

    return true;
}


bool assignAnomalyPrec(float myAnomaly, int anomalyMonth1, int anomalyMonth2, float* myWGMonthlyVarNoAnomaly, float* myWGMonthlyVar)
{

    int myMonth = 0;
    float mySumClimatePrec;
    float myNewSumPrec = 0;
    float myFraction = 0;

    int myNumMonths = numMonthsInPeriod(anomalyMonth1, anomalyMonth2);

    // compute sum of precipitation
    mySumClimatePrec = 0;
    if (anomalyMonth2 >= anomalyMonth1)
    {
        // regular period
        for (myMonth = anomalyMonth1; myMonth <= anomalyMonth2; myMonth++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[myMonth-1];

    }
    else
    {
        // irregular period (between years)
        for (myMonth = anomalyMonth1; myMonth <= 12; myMonth++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[myMonth-1];

        for (myMonth = 1; myMonth <= anomalyMonth2; myMonth++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[myMonth-1];
    }

    myNewSumPrec = std::max(mySumClimatePrec + myAnomaly, 0.f);

    if (mySumClimatePrec > 0)
        myFraction = myNewSumPrec / mySumClimatePrec;

    if (anomalyMonth2 >= anomalyMonth1)
    {
        for (myMonth = anomalyMonth1; myMonth <= anomalyMonth2; myMonth++)
        {
            if (mySumClimatePrec > 0)
                myWGMonthlyVar[myMonth-1] = myWGMonthlyVarNoAnomaly[myMonth-1] * myFraction;
            else
                myWGMonthlyVar[myMonth-1] = myNewSumPrec / myNumMonths;
        }
    }
    else
    {
        for (myMonth = 1; myMonth<= 12; myMonth++)
        {
            if (myMonth <= anomalyMonth2 || myMonth >= anomalyMonth1)
            {
                if (mySumClimatePrec > 0)
                    myWGMonthlyVar[myMonth-1] = myWGMonthlyVarNoAnomaly[myMonth-1] * myFraction;
                else
                    myWGMonthlyVar[myMonth-1] = myNewSumPrec / myNumMonths;
            }
        }
    }

    return true;
}


/*!
  * \name makeSeasonalForecast
  * \brief Generates a time series of daily data (Tmin, Tmax, Prec)
  * for a period of nrYears = numMembers * nrRepetitions
  * Different members of anomalies loaded by xml files are added to the climate
  * Output is written on outputFileName (csv)
*/
bool makeSeasonalForecast(QString outputFileName, char separator, TXMLSeasonalAnomaly* XMLAnomaly,
                          TweatherGenClimate wGenClimate, TinputObsData* lastYearDailyObsData,
                          int nrRepetitions, int myPredictionYear, int wgDoy1, int wgDoy2, float rainfallThreshold)
{
    TweatherGenClimate wGen;
    ToutputDailyMeteo* myDailyPredictions;
    Crit3DDate myFirstDatePrediction;
    Crit3DDate seasonFirstDate;
    Crit3DDate seasonLastDate;

    unsigned int nrMembers;         // number of models into xml anomaly file
    unsigned int nrYears;           // number of years of the output series. It is the length of the virtual period where all the previsions (one for each model) are given one after another
    unsigned int nrValues;          // number of days between the first and the last prediction year
    int firstYear, lastYear, myYear;
    unsigned int obsIndex;
    unsigned int addday = 0;
    bool isLastMember = false;

    // it checks if observed data includes the last 9 months before wgDoy1
    int nrDaysBeforeWgDoy1;
    if (! checkLastYearDate(lastYearDailyObsData->inputFirstDate, lastYearDailyObsData->inputLastDate,
                            lastYearDailyObsData->dataLenght, myPredictionYear, &wgDoy1, &nrDaysBeforeWgDoy1))
    {
        qDebug() << "ERROR: observed data should include at least 9 months before wgDoy1";
        return false;
    }

    nrMembers = 0;
    for (int i = 0; i<XMLAnomaly->modelMember.size(); i++)
    {
        nrMembers += XMLAnomaly->modelMember[i].toUInt();
    }

    nrYears = nrMembers * unsigned(nrRepetitions);

    firstYear = myPredictionYear;

    // wgDoy1 within myPredictionYear, wgDoy2 within myPredictionYear+1
    if (wgDoy1 < wgDoy2)
        lastYear = firstYear + signed(nrYears) - 1;
    else
        lastYear = firstYear + signed(nrYears);

    seasonFirstDate = getDateFromDoy (myPredictionYear, wgDoy1);
    if (wgDoy1 < wgDoy2)
        seasonLastDate = getDateFromDoy (myPredictionYear, wgDoy2);
    else
        seasonLastDate = getDateFromDoy (myPredictionYear+1, wgDoy2);

    myFirstDatePrediction = seasonFirstDate.addDays(-nrDaysBeforeWgDoy1);

    for (int i = myPredictionYear; i <= lastYear; i++)
    {
        if (isLeapYear(i))
            addday++;
    }

    nrValues = nrYears * 365 + addday +1;
    if (nrValues <= 0)
    {
        qDebug() << "ERROR: wrong date";
        return false;
    }

    myDailyPredictions = new ToutputDailyMeteo[nrValues];

    // copy the last 9 months before wgDoy1
    float lastTmax = NODATA;
    float lastTmin = NODATA;
    Crit3DDate myDate = myFirstDatePrediction;
    for (int tmp = 0; tmp < nrDaysBeforeWgDoy1; tmp++)
    {
        myDailyPredictions[tmp].date = myDate;
        obsIndex = difference(lastYearDailyObsData->inputFirstDate, myDailyPredictions[tmp].date);
        myDailyPredictions[tmp].minTemp = lastYearDailyObsData->inputTMin[obsIndex];
        myDailyPredictions[tmp].maxTemp = lastYearDailyObsData->inputTMax[obsIndex];
        myDailyPredictions[tmp].prec = lastYearDailyObsData->inputPrecip[obsIndex];

        if ((int(myDailyPredictions[tmp].maxTemp) == int(NODATA))
                || (int(myDailyPredictions[tmp].minTemp) == int(NODATA))
                || (int(myDailyPredictions[tmp].prec) == int(NODATA)))
        {
            if (tmp == 0)
            {
                qDebug() << "ERROR: Missing data:" << QString::fromStdString(myDailyPredictions[tmp].date.toStdString());
                return false;
            }
            else
            {
                qDebug() << "WARNING: Missing data:" << QString::fromStdString(myDailyPredictions[tmp].date.toStdString());

                if (int(myDailyPredictions[tmp].maxTemp) == int(NODATA))
                    myDailyPredictions[tmp].maxTemp = lastTmax;

                if (int(myDailyPredictions[tmp].minTemp) == int(NODATA))
                    myDailyPredictions[tmp].minTemp = lastTmin;

                if (int(myDailyPredictions[tmp].prec) == int(NODATA))
                    myDailyPredictions[tmp].prec = 0;
            }
        }
        else
        {
            lastTmax = myDailyPredictions[tmp].maxTemp;
            lastTmin = myDailyPredictions[tmp].minTemp;
        }
        ++myDate;
    }

    qDebug() << "Observed OK";
    int outputDataLenght = nrDaysBeforeWgDoy1;

    // store the climate without anomalies
    wGen = wGenClimate;
    myYear = firstYear;

    // first month of my season
    int anomalyMonth1 = seasonFirstDate.month;
    // last month of my season
    int anomalyMonth2 = seasonLastDate.month;

    for (unsigned int modelIndex = 0; modelIndex < nrMembers; modelIndex++)
    {
        // assign anomaly
        if ( !assignXMLAnomaly(XMLAnomaly, modelIndex, anomalyMonth1, anomalyMonth2, &wGenClimate, &wGen))
        {
                    qDebug() << "Error in Scenario: assignXMLAnomaly returns false";
                    return false;
        }

        if (modelIndex == nrMembers-1 )
        {
            isLastMember = true;
        }
        // compute seasonal prediction
        if (!computeSeasonalPredictions(lastYearDailyObsData, &wGen,
                                        myPredictionYear, myYear, nrRepetitions,
                                        wgDoy1, wgDoy2, rainfallThreshold, isLastMember,
                                        myDailyPredictions, &outputDataLenght ))
        {
            qDebug() << "Error in computeSeasonalPredictions";
            return false;
        }

        // next model
        myYear = myYear + nrRepetitions;
    }

    qDebug() << "\n>>> output:" << outputFileName;

    writeMeteoDataCsv (outputFileName, separator, myDailyPredictions, outputDataLenght);

    free(myDailyPredictions);

    return true;
}


/*!
  \name computeSeasonalPredictions
  \brief Generates a time series of daily data (Tmin, Tmax, Prec)
    The lenght is equals to nrRepetitions years, starting from firstYear
    Period between wgDoy1 and wgDoy2 is produced by the WG
    Others data are a copy of the observed data of predictionYear
    Weather generator climate is stored in wgClimate
    Observed data (Tmin, Tmax, Prec) are in lastYearDailyObsData
  \return outputDailyData
*/
bool computeSeasonalPredictions(TinputObsData *lastYearDailyObsData, TweatherGenClimate* wgClimate,
                                int predictionYear, int firstYear, int nrRepetitions,
                                int wgDoy1, int wgDoy2, float rainfallThreshold, bool isLastMember,
                                ToutputDailyMeteo* outputDailyData, int *outputDataLenght)

{
    Crit3DDate myDate, obsDate;
    Crit3DDate firstDate, lastDate;
    int lastYear, myDoy;
    int obsIndex, currentIndex;
    int fixwgDoy1 = wgDoy1;
    int fixwgDoy2 = wgDoy2;

    // TODO etp e falda

    currentIndex = *outputDataLenght;

    firstDate = outputDailyData[currentIndex-1].date.addDays(1);

    if (wgDoy1 < wgDoy2)
    {
        lastYear = firstYear + nrRepetitions - 1;

        if (isLastMember)
        {
            if ( (!isLeapYear(predictionYear) && !isLeapYear(lastYear)) || (isLeapYear(predictionYear) && isLeapYear(lastYear)))
                lastDate = getDateFromDoy(lastYear,wgDoy2);
            else
            {
                if(isLeapYear(predictionYear) && wgDoy2 >= 60 )
                    lastDate = getDateFromDoy(lastYear,wgDoy2-1);
                if(isLeapYear(lastYear) && wgDoy2 >= 59 )
                    lastDate = getDateFromDoy(lastYear,wgDoy2+1);
            }
        }
        else
        {
            lastDate = outputDailyData[currentIndex-1].date;
            lastDate.year = lastYear;
        }
    }
    else
    {
        lastYear = firstYear + nrRepetitions;

        if (isLastMember)
        {
            if ( (!isLeapYear(predictionYear+1) && !isLeapYear(lastYear)) || (isLeapYear(predictionYear+1) && isLeapYear(lastYear)))
                lastDate = getDateFromDoy(lastYear,wgDoy2);
            else
            {
                if(isLeapYear(predictionYear+1) && wgDoy2 >= 60)
                    lastDate = getDateFromDoy(lastYear,wgDoy2-1);
                if(isLeapYear(lastYear) && wgDoy2 >= 59 )
                    lastDate = getDateFromDoy(lastYear,wgDoy2+1);
            }
        }
        else
        {
            lastDate = outputDailyData[currentIndex-1].date;
            lastDate.year = lastYear;
        }

    }

    // initialize WG
    initializeWeather(wgClimate);

    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {

        fixWgDoy(wgDoy1, wgDoy2, predictionYear, myDate.year, &fixwgDoy1, &fixwgDoy2);
        myDoy = getDoyFromDate(myDate);

        // fill mydailyData.date
        initializeDailyDataBasic (&outputDailyData[currentIndex], myDate);

        if ( isWGDate(myDate, fixwgDoy1, fixwgDoy2) )
        {
            outputDailyData[currentIndex].maxTemp = getTMax(myDoy, rainfallThreshold, wgClimate);
            outputDailyData[currentIndex].minTemp = getTMin(myDoy, rainfallThreshold, wgClimate);
            outputDailyData[currentIndex].prec = getPrecip(myDoy, rainfallThreshold, wgClimate);
        }
        else
        {

            obsDate.day = myDate.day;
            obsDate.month = myDate.month;

            if (myDoy < fixwgDoy1)
                obsDate.year = predictionYear;
            else if (myDoy > fixwgDoy2)
                obsDate.year = predictionYear-1;

            obsIndex = difference(lastYearDailyObsData->inputFirstDate, obsDate);

            if ( obsIndex >= 0 && obsIndex <= lastYearDailyObsData->dataLenght )
            {
                outputDailyData[currentIndex].maxTemp = lastYearDailyObsData->inputTMax[obsIndex];
                outputDailyData[currentIndex].minTemp = lastYearDailyObsData->inputTMin[obsIndex];
                outputDailyData[currentIndex].prec = lastYearDailyObsData->inputPrecip[obsIndex];
            }
            else
            {
                qDebug() << "Error: wrong date in computeSeasonalPredictions";
                return false;
            }

        }
        currentIndex++;
     }

     *outputDataLenght = currentIndex;
     return true;
}


void initializeDailyDataBasic(ToutputDailyMeteo* mydailyData, Crit3DDate myDate)
{
    mydailyData->date = myDate;
    mydailyData->maxTemp = NODATA;
    mydailyData->minTemp = NODATA;
    mydailyData->prec = NODATA;

}


bool isWGDate(Crit3DDate myDate, int wgDoy1, int wgDoy2)
{
    bool isWGDate = false;
    int myDoy = getDoyFromDate(myDate);

    if (wgDoy2 >= wgDoy1)
    {
        if ( (myDoy >= wgDoy1) && (myDoy <= wgDoy2) )
                isWGDate = true;
    }
    else
    {
        if ( (myDoy >= wgDoy1) || (myDoy <= wgDoy2) )
                isWGDate = true;
    }

    return isWGDate;
}


void clearInputData(TinputObsData* myData)
{
    myData->inputTMin.clear();
    myData->inputTMax.clear();
    myData->inputPrecip.clear();
}
