/*!
======================================================================================================
    \name weatherGenerator.cpp
======================================================================================================
    \copyright
    2016 Fausto Tomei, Laura Costantini

    This file is part of agrolib distribution .
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


float getTMax(int dayOfYear, float precThreshold, TweatherGenClimate& wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen.state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return wGen.state.currentTmax;
}


float getTMin(int dayOfYear, float precThreshold, TweatherGenClimate& wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen.state.currentDay)
        newDay(dayOfYear, precThreshold,  wGen);

    return wGen.state.currentTmin;
}


float getTAverage(int dayOfYear, float precThreshold, TweatherGenClimate &wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen.state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return 0.5f * (wGen.state.currentTmax + wGen.state.currentTmin);
}


float getPrecip(int dayOfYear, float precThreshold, TweatherGenClimate& wGen)
{
    dayOfYear = dayOfYear % 365;
    if (dayOfYear != wGen.state.currentDay)
        newDay(dayOfYear, precThreshold, wGen);

    return wGen.state.currentPrec;
}


// main function
void newDay(int dayOfYear, float precThreshold, TweatherGenClimate& wGen)
{
    float meanTMax, meanTMin, stdTMax, stdTMin;

    // daily structure is 0-365
    dayOfYear = dayOfYear - 1;

    //Precipitation
    bool isWetDay = markov(wGen.daily.pwd[dayOfYear], wGen.daily.pww[dayOfYear], wGen.state.wetPreviousDay);

    if (isWetDay)
    {
        meanTMax = wGen.daily.meanWetTMax[dayOfYear];
        wGen.state.currentPrec = weibull(wGen.daily.meanPrecip[dayOfYear], precThreshold);
    }
    else
    {
        meanTMax = wGen.daily.meanDryTMax[dayOfYear];
        wGen.state.currentPrec = 0;
    }

    //store information
    wGen.state.wetPreviousDay = isWetDay;

    //temperature
    meanTMin = wGen.daily.meanTMin[dayOfYear];
    stdTMax = wGen.daily.maxTempStd[dayOfYear];
    stdTMin = wGen.daily.minTempStd[dayOfYear];

    genTemps(&wGen.state.currentTmax, &wGen.state.currentTmin, meanTMax, meanTMin, stdTMax, stdTMin, &(wGen.state.resTMaxPrev), &(wGen.state.resTMinPrev));

    wGen.state.currentDay = dayOfYear;
}


void initializeDailyDataBasic(ToutputDailyMeteo* dailyData, Crit3DDate myDate)
{
    dailyData->date = myDate;
    dailyData->maxTemp = NODATA;
    dailyData->minTemp = NODATA;
    dailyData->prec = NODATA;
    dailyData->waterTableDepth = NODATA;
}


void initializeWeather(TweatherGenClimate &wGen)
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
    float sumPrecip = 0;
    int daysInMonth;
    int m;

    wGen.state.currentDay = NODATA;
    wGen.state.currentTmax = NODATA;
    wGen.state.currentTmin = NODATA;
    wGen.state.currentPrec = NODATA;

    // TODO: pass residual data of last observed day
    wGen.state.resTMaxPrev = 0;
    wGen.state.resTMinPrev = 0;
    wGen.state.wetPreviousDay = false;

    for (int i = 0; i < 366; i++)
    {
        wGen.daily.maxTempStd[i] = 0;
        wGen.daily.meanDryTMax[i] = 0;
        wGen.daily.meanTMin[i] = 0;
        wGen.daily.meanPrecip[i] = 0;
        wGen.daily.meanWetTMax[i] = 0;
        wGen.daily.minTempStd[i] = 0;
        wGen.daily.pwd[i] = 0;
        wGen.daily.pww[i] = 0;
    }

    for (m = 0; m < 12; m++)
    {
        mMeanTMax[m] = wGen.monthly.monthlyTmax[m];
        mMeanTMin[m] = wGen.monthly.monthlyTmin[m];
        mMeanPrecip[m] = wGen.monthly.sumPrec[m];
        fWetDays[m] = wGen.monthly.fractionWetDays[m];
        mpww[m] = wGen.monthly.probabilityWetWet[m];
        mMeanDiff[m] = wGen.monthly.dw_Tmax[m];
        mMaxTempStd[m] = wGen.monthly.stDevTmax[m];
        mMinTempStd[m] = wGen.monthly.stDevTmin[m];
        sumPrecip = sumPrecip + mMeanPrecip[m];
    }

    for (m = 0; m < 12; m++)
    {
        mMeanDryTMax[m] = mMeanTMax[m] + fWetDays[m] * mMeanDiff[m];
        mMeanWetTMax[m] = mMeanDryTMax[m] - mMeanDiff[m];

        mpwd[m] = (1.f - mpww[m]) * (fWetDays[m] / (1.f - fWetDays[m]));

        daysInMonth = getDaysInMonth(m+1, 2001); // year = 2001 is to avoid leap year

        // convert from total mm/month to average mm/wet day
        mMeanPrecip[m] = mMeanPrecip[m] / (fWetDays[m] * float(daysInMonth));
    }

    interpolation::cubicSplineYearInterpolate(mpww, wGen.daily.pww);
    interpolation::cubicSplineYearInterpolate(mpwd, wGen.daily.pwd);
    interpolation::cubicSplineYearInterpolate(mMeanDryTMax, wGen.daily.meanDryTMax);
    interpolation::cubicSplineYearInterpolate(mMeanPrecip, wGen.daily.meanPrecip);
    interpolation::cubicSplineYearInterpolate(mMeanWetTMax, wGen.daily.meanWetTMax);
    interpolation::cubicSplineYearInterpolate(mMeanTMin, wGen.daily.meanTMin);
    interpolation::cubicSplineYearInterpolate(mMinTempStd, wGen.daily.minTempStd);
    interpolation::cubicSplineYearInterpolate(mMaxTempStd, wGen.daily.maxTempStd);
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


void clearInputData(TinputObsData &myData)
{
    myData.inputTMin.clear();
    myData.inputTMax.clear();
    myData.inputPrecip.clear();
}


bool assignXMLAnomaly(XMLSeasonalAnomaly* XMLAnomaly, int modelIndex, int anomalyMonth1, int anomalyMonth2, TweatherGenClimate& wGenNoAnomaly, TweatherGenClimate &wGen)
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
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.monthlyTmin, wGen.monthly.monthlyTmin);
                else if ( (myVar == "TMAX") || (myVar == "AVGTMAX") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.monthlyTmax, wGen.monthly.monthlyTmax);
                else if ( (myVar == "PREC3M") || (myVar == "PREC") )
                    result = assignAnomalyPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.sumPrec, wGen.monthly.sumPrec);
                else if ( (myVar == "WETDAYSFREQUENCY") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.fractionWetDays, wGen.monthly.fractionWetDays);
                else if ( (myVar == "WETWETDAYSFREQUENCY") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.probabilityWetWet, wGen.monthly.probabilityWetWet);
                else if ( (myVar == "DELTATMAXDRYWET") )
                    result = assignAnomalyNoPrec(myValue, anomalyMonth1, anomalyMonth2, wGenNoAnomaly.monthly.dw_Tmax, wGen.monthly.dw_Tmax);
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
        stream << "wGen.monthly.monthlyTmin = " << wGen.monthly.monthlyTmin[m] << endl;
        stream << "wGen.monthly.monthlyTmax = " << wGen.monthly.monthlyTmax[m] << endl;
        stream << "wGen.monthly.sumPrec = " << wGen.monthly.sumPrec[m] << endl;
        stream << "wGen.monthly.stDevTmin[m] = " << wGen.monthly.stDevTmin[m] << endl;
        stream << "wGen.monthly.stDevTmax = " << wGen.monthly.stDevTmax[m] << endl;
        stream << "wGen.monthly.fractionWetDays[m] = " << wGen.monthly.fractionWetDays[m] << endl;
        stream << "wGen.monthly.probabilityWetWet[m] = " << wGen.monthly.probabilityWetWet[m] << endl;
        stream << "wGen.monthly.dw_Tmax[m] = " << wGen.monthly.dw_Tmax[m] << endl;
        stream << "-------------------------------------------" << endl;
    }
    */

    return true;
}


bool assignAnomalyNoPrec(float myAnomaly, int anomalyMonth1, int anomalyMonth2,
                         float* myWGMonthlyVarNoAnomaly, float* myWGMonthlyVar)
{
    int month = 0;

    if (anomalyMonth2 >= anomalyMonth1)
    {
        // regular period
        for (month = anomalyMonth1; month <= anomalyMonth2; month++)
            myWGMonthlyVar[month-1] = myWGMonthlyVarNoAnomaly[month-1] + myAnomaly;

    }
    else
    {
        // irregular period (between years)
        for (month = anomalyMonth1; month <= 12; month++)
            myWGMonthlyVar[month-1] = myWGMonthlyVarNoAnomaly[month-1] + myAnomaly;

        for (month = 1; month <=anomalyMonth2; month++)
            myWGMonthlyVar[month-1] = myWGMonthlyVarNoAnomaly[month-1] + myAnomaly;
    }

    return true;
}


bool assignAnomalyPrec(float myAnomaly, int anomalyMonth1, int anomalyMonth2,
                       float* myWGMonthlyVarNoAnomaly, float* myWGMonthlyVar)
{

    int month;
    float mySumClimatePrec;
    float myNewSumPrec = 0;
    float myFraction = 0;

    int nrMonths = getMonthsInPeriod(anomalyMonth1, anomalyMonth2);

    // compute sum of precipitation
    mySumClimatePrec = 0;
    if (anomalyMonth2 >= anomalyMonth1)
    {
        // regular period
        for (month = anomalyMonth1; month <= anomalyMonth2; month++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[month-1];

    }
    else
    {
        // irregular period (between years)
        for (month = anomalyMonth1; month <= 12; month++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[month-1];

        for (month = 1; month <= anomalyMonth2; month++)
            mySumClimatePrec = mySumClimatePrec + myWGMonthlyVarNoAnomaly[month-1];
    }

    myNewSumPrec = std::max(mySumClimatePrec + myAnomaly, 0.f);

    if (mySumClimatePrec > 0)
        myFraction = myNewSumPrec / mySumClimatePrec;

    if (anomalyMonth2 >= anomalyMonth1)
    {
        for (month = anomalyMonth1; month <= anomalyMonth2; month++)
        {
            if (mySumClimatePrec > 0)
                myWGMonthlyVar[month-1] = myWGMonthlyVarNoAnomaly[month-1] * myFraction;
            else
                myWGMonthlyVar[month-1] = myNewSumPrec / float(nrMonths);
        }
    }
    else
    {
        for (month = 1; month<= 12; month++)
        {
            if (month <= anomalyMonth2 || month >= anomalyMonth1)
            {
                if (mySumClimatePrec > 0)
                    myWGMonthlyVar[month-1] = myWGMonthlyVarNoAnomaly[month-1] * myFraction;
                else
                    myWGMonthlyVar[month-1] = myNewSumPrec / float(nrMonths);
            }
        }
    }

    return true;
}


bool assignXMLAnomalyScenario(XMLScenarioAnomaly* XMLAnomaly,int modelIndex, int* anomalyMonth1, int* anomalyMonth2, TweatherGenClimate& wGenNoAnomaly, TweatherGenClimate &wGen)
{
    //unsigned int i = 0;
    QString myVar;
    float myValue = 0.0;

    bool result;

    // loop for all XMLValuesList (Tmin, Tmax, TminVar, TmaxVar, Prec3M, Wetdays)
    for (int iSeason=0;iSeason<4;iSeason++)
    {
        for (unsigned int iWeatherVariable = 0; iWeatherVariable < 4; iWeatherVariable++)
        {
            if (XMLAnomaly->period[iSeason].seasonalScenarios[iWeatherVariable].attribute.toUpper() == "ANOMALY")
            {
                myVar = XMLAnomaly->period[iSeason].seasonalScenarios[iWeatherVariable].type.toUpper();
                result = false;
                //if (XMLAnomaly->forecast[i].value[modelIndex] != nullptr)
                    myValue = XMLAnomaly->period[iSeason].seasonalScenarios[iWeatherVariable].value[modelIndex].toFloat();
                //else
                    //myValue = NODATA;

                if (int(myValue) != int(NODATA))
                {
                    if ( (myVar == "TMIN") || (myVar == "AVGTMIN") )
                        result = assignAnomalyNoPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.monthlyTmin, wGen.monthly.monthlyTmin);
                    else if ( (myVar == "TMAX") || (myVar == "AVGTMAX") )
                        result = assignAnomalyNoPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.monthlyTmax, wGen.monthly.monthlyTmax);
                    else if ( (myVar == "PREC3M") || (myVar == "PREC") )
                        result = assignAnomalyPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.sumPrec, wGen.monthly.sumPrec);
                    else if ( (myVar == "WETDAYSFREQUENCY") )
                        result = assignAnomalyNoPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.fractionWetDays, wGen.monthly.fractionWetDays);
                    else if ( (myVar == "WETWETDAYSFREQUENCY") )
                        result = assignAnomalyNoPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.probabilityWetWet, wGen.monthly.probabilityWetWet);
                    else if ( (myVar == "DELTATMAXDRYWET") )
                        result = assignAnomalyNoPrec(myValue, anomalyMonth1[iSeason], anomalyMonth2[iSeason], wGenNoAnomaly.monthly.dw_Tmax, wGen.monthly.dw_Tmax);

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
        // move to the next season
        //anomalyMonth1 = (anomalyMonth1 + 3)%12;
        //if (anomalyMonth1 == 0) anomalyMonth1 +=12;
        //anomalyMonth2 = (anomalyMonth2 + 3)%12;
        //if (anomalyMonth2 == 0) anomalyMonth2 +=12;
    }
    /* DEBUG
    QString anomaly="anomaly.txt";
    QFile file(anomaly);
    file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    QTextStream stream( &file );
    for (int m = 0; m < 12; m++)
    {
        stream << "month = " << m +1 << endl;
        stream << "wGen.monthly.monthlyTmin = " << wGen.monthly.monthlyTmin[m] << endl;
        stream << "wGen.monthly.monthlyTmax = " << wGen.monthly.monthlyTmax[m] << endl;
        stream << "wGen.monthly.sumPrec = " << wGen.monthly.sumPrec[m] << endl;
        stream << "wGen.monthly.stDevTmin[m] = " << wGen.monthly.stDevTmin[m] << endl;
        stream << "wGen.monthly.stDevTmax = " << wGen.monthly.stDevTmax[m] << endl;
        stream << "wGen.monthly.fractionWetDays[m] = " << wGen.monthly.fractionWetDays[m] << endl;
        stream << "wGen.monthly.probabilityWetWet[m] = " << wGen.monthly.probabilityWetWet[m] << endl;
        stream << "wGen.monthly.dw_Tmax[m] = " << wGen.monthly.dw_Tmax[m] << endl;
        stream << "-------------------------------------------" << endl;
    }
    */

    return true;
}

/*!
  * \name makeSeasonalForecast
  * \brief Generates a time series of daily data (Tmin, Tmax, Prec)
  * for a period of nrYears = numMembers * nrRepetitions
  * Different members of anomalies loaded by xml files are added to the climate
  * Output is written on outputFileName (csv)
*/
bool makeSeasonalForecast(QString outputFileName, char separator, XMLSeasonalAnomaly* XMLAnomaly,
                          TweatherGenClimate& wGenClimate, TinputObsData* dailyObsData,
                          int nrRepetitions, int myPredictionYear, int wgDoy1, int wgDoy2,
                          float rainfallThreshold)
{
    TweatherGenClimate wGen;
    std::vector<ToutputDailyMeteo> dailyPredictions;

    Crit3DDate myFirstDatePrediction, seasonFirstDate, seasonLastDate;

    unsigned int nrMembers;         // number of models into xml anomaly file
    unsigned int nrYears;           // number of years of the output series. It is the length of the virtual period where all the previsions (one for each model) are given one after another
    unsigned int nrValues;          // number of days between the first and the last prediction year
    int firstYear, lastYear, myYear;
    unsigned int obsIndex;
    unsigned int addday = 0;
    bool isLastMember = false;

    // it checks if observed data includes the last 9 months before wgDoy1
    int nrDaysBeforeWgDoy1;
    if (! checkLastYearDate(dailyObsData->inputFirstDate, dailyObsData->inputLastDate,
                            dailyObsData->dataLength, myPredictionYear, wgDoy1, nrDaysBeforeWgDoy1))
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

    dailyPredictions.resize(nrValues);

    // copy the last 9 months before wgDoy1
    float lastTmax = NODATA;
    float lastTmin = NODATA;
    Crit3DDate myDate = myFirstDatePrediction;
    for (int tmp = 0; tmp < nrDaysBeforeWgDoy1; tmp++)
    {
        dailyPredictions[tmp].date = myDate;
        obsIndex = difference(dailyObsData->inputFirstDate, dailyPredictions[tmp].date);
        dailyPredictions[tmp].minTemp = dailyObsData->inputTMin[obsIndex];
        dailyPredictions[tmp].maxTemp = dailyObsData->inputTMax[obsIndex];
        dailyPredictions[tmp].prec = dailyObsData->inputPrecip[obsIndex];

        if ((int(dailyPredictions[tmp].maxTemp) == int(NODATA))
                || (int(dailyPredictions[tmp].minTemp) == int(NODATA))
                || (int(dailyPredictions[tmp].prec) == int(NODATA)))
        {
            if (tmp == 0)
            {
                qDebug() << "ERROR: Missing data:" << QString::fromStdString(dailyPredictions[tmp].date.toISOString());
                return false;
            }
            else
            {
                qDebug() << "WARNING: Missing data:" << QString::fromStdString(dailyPredictions[tmp].date.toISOString());

                if (int(dailyPredictions[tmp].maxTemp) == int(NODATA))
                    dailyPredictions[tmp].maxTemp = lastTmax;

                if (int(dailyPredictions[tmp].minTemp) == int(NODATA))
                    dailyPredictions[tmp].minTemp = lastTmin;

                if (int(dailyPredictions[tmp].prec) == int(NODATA))
                    dailyPredictions[tmp].prec = 0;
            }
        }
        else
        {
            lastTmax = dailyPredictions[tmp].maxTemp;
            lastTmin = dailyPredictions[tmp].minTemp;
        }
        ++myDate;
    }

    qDebug() << "Observed OK";
    int outputDataLength = nrDaysBeforeWgDoy1;

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
        if ( !assignXMLAnomaly(XMLAnomaly, modelIndex, anomalyMonth1, anomalyMonth2, wGenClimate, wGen))
        {
            qDebug() << "Error in Scenario: assignXMLAnomaly returns false";
            return false;
        }

        if (modelIndex == nrMembers-1 )
        {
            isLastMember = true;
        }
        // compute seasonal prediction
        std::vector<int> indexWg;
        if (! computeSeasonalPredictions(dailyObsData, wGen,
                                        myPredictionYear, myYear, nrRepetitions,
                                        wgDoy1, wgDoy2, rainfallThreshold, isLastMember,
                                        dailyPredictions, &outputDataLength, indexWg))
        {
            qDebug() << "Error in computeSeasonalPredictions";
            return false;
        }

        // next model
        myYear = myYear + nrRepetitions;
    }

    qDebug() << "\n>>> output:" << outputFileName;

    writeMeteoDataCsv (outputFileName, separator, dailyPredictions, false);

    dailyPredictions.clear();

    return true;
}


bool initializeWaterTableData(TinputObsData* dailyObsData, WaterTable *waterTable,
                              int predictionYear, int wgDoy1, int nrDaysBeforeWgDoy1, int daysWg)
{
    Crit3DDate seasonFirstDate = getDateFromDoy(predictionYear, wgDoy1);
    Crit3DDate outputFirstDate = seasonFirstDate.addDays(-nrDaysBeforeWgDoy1);

    int firstIndex = difference(dailyObsData->inputFirstDate, outputFirstDate);
    int totDays = firstIndex + nrDaysBeforeWgDoy1 + daysWg;

    std::vector<float> inputTMin, inputTMax, inputPrec;
    for (int i = 0; i < totDays; i++)
    {
        if (i < (firstIndex + nrDaysBeforeWgDoy1))
        {
            inputTMin.push_back(dailyObsData->inputTMin[i]);
            inputTMax.push_back(dailyObsData->inputTMax[i]);
            inputPrec.push_back(dailyObsData->inputPrecip[i]);
        }
        else
        {
            // aggiungo giorni (vuoti) a watertable
            inputTMin.push_back(NODATA);
            inputTMax.push_back(NODATA);
            inputPrec.push_back(NODATA);
        }
    }

    waterTable->setInputTMin(inputTMin);
    waterTable->setInputTMax(inputTMax);
    waterTable->setInputPrec(inputPrec);

    QDate firstDate = QDate(dailyObsData->inputFirstDate.year, dailyObsData->inputFirstDate.month, dailyObsData->inputFirstDate.day);
    QDate lastDate = firstDate.addDays(totDays-1);

    waterTable->setFirstMeteoDate(firstDate);
    waterTable->setLastMeteoDate(lastDate);

    waterTable->computeETP_allSeries(false);

    return true;
}


/*!
  * \name makeSeasonalForecastWaterTable
  * \brief Generates a time series of daily data (Tmin, Tmax, Prec, WaterTable depth)
  * for a period of nrYears = numMembers * nrRepetitions
  * Different members of anomalies loaded by xml files are added to the climate
  * Output is written on outputFileName (csv)
*/
bool makeSeasonalForecastWaterTable(QString outputFileName, char separator, XMLSeasonalAnomaly* XMLAnomaly,
                          TweatherGenClimate& wGenClimate, TinputObsData* dailyObsData, WaterTable *waterTable,
                          int nrRepetitions, int predictionYear, int wgDoy1, int wgDoy2,
                          float rainfallThreshold)
{
    // it checks if observed data includes the last 9 months before wgDoy1
    int nrDaysBeforeWgDoy1;
    if (! checkLastYearDate(dailyObsData->inputFirstDate, dailyObsData->inputLastDate,
                           dailyObsData->dataLength, predictionYear, wgDoy1, nrDaysBeforeWgDoy1))
    {
        qDebug() << "ERROR: observed data should include at least 9 months before wgDoy1";
        return false;
    }

    Crit3DDate seasonLastDate;
    int daysWg;
    Crit3DDate seasonFirstDate = getDateFromDoy (predictionYear, wgDoy1);
    if (wgDoy1 < wgDoy2)
    {
        seasonLastDate = getDateFromDoy (predictionYear, wgDoy2);
        daysWg = wgDoy2 - wgDoy1 + 1;
    }
    else
    {
        seasonLastDate = getDateFromDoy (predictionYear+1, wgDoy2);
        if (isLeapYear(predictionYear))
        {
            daysWg = (366 - wgDoy1) + wgDoy2 + 1;
        }
        else
        {
            daysWg = (365 - wgDoy1) + wgDoy2 + 1;
        }
    }

    if (! initializeWaterTableData(dailyObsData, waterTable, predictionYear, wgDoy1, nrDaysBeforeWgDoy1, daysWg))
    {
        qDebug() << "ERROR in initializeWaterTableData";
        return false;
    }

    TweatherGenClimate wGen;
    std::vector<ToutputDailyMeteo> dailyPredictions;

    unsigned int nrMembers;         // number of models into xml anomaly file
    unsigned int nrYears;           // number of years of the output series. It is the length of the virtual period where all the previsions (one for each model) are given one after another
    unsigned int nrValues;          // number of days between the first and the last prediction year

    nrMembers = 0;
    for (int i = 0; i<XMLAnomaly->modelMember.size(); i++)
    {
        nrMembers += XMLAnomaly->modelMember[i].toUInt();
    }

    nrYears = nrMembers * unsigned(nrRepetitions);
    int lastYear = predictionYear + signed(nrYears) - 1;
    if (wgDoy2 < wgDoy1) lastYear++;

    Crit3DDate firstDatePrediction = seasonFirstDate.addDays(-nrDaysBeforeWgDoy1);

    unsigned int addday = 0;
    for (int i = predictionYear; i <= lastYear; i++)
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

    dailyPredictions.resize(nrValues);

    float lastTmax = NODATA;
    float lastTmin = NODATA;

    float wtDepth;
    float myDelta;
    int myDeltaDays;

    Crit3DDate myDate = firstDatePrediction;
    for (int tmp = 0; tmp < nrDaysBeforeWgDoy1; tmp++)
    {
        dailyPredictions[tmp].date = myDate;
        int obsIndex = difference(dailyObsData->inputFirstDate, dailyPredictions[tmp].date);
        dailyPredictions[tmp].minTemp = dailyObsData->inputTMin[obsIndex];
        dailyPredictions[tmp].maxTemp = dailyObsData->inputTMax[obsIndex];
        dailyPredictions[tmp].prec = dailyObsData->inputPrecip[obsIndex];
        if (waterTable->getWaterTableInterpolation(QDate(myDate.year, myDate.month, myDate.day), &wtDepth, &myDelta, &myDeltaDays))
        {
            dailyPredictions[tmp].waterTableDepth = wtDepth;
        }

        if ((int(dailyPredictions[tmp].maxTemp) == int(NODATA))
            || (int(dailyPredictions[tmp].minTemp) == int(NODATA))
            || (int(dailyPredictions[tmp].prec) == int(NODATA)))
        {
            if (tmp == 0)
            {
                qDebug() << "ERROR: Missing data:" << QString::fromStdString(dailyPredictions[tmp].date.toISOString());
                return false;
            }
            else
            {
                qDebug() << "WARNING: Missing data:" << QString::fromStdString(dailyPredictions[tmp].date.toISOString());

                if (int(dailyPredictions[tmp].maxTemp) == int(NODATA))
                    dailyPredictions[tmp].maxTemp = lastTmax;

                if (int(dailyPredictions[tmp].minTemp) == int(NODATA))
                    dailyPredictions[tmp].minTemp = lastTmin;

                if (int(dailyPredictions[tmp].prec) == int(NODATA))
                    dailyPredictions[tmp].prec = 0;
            }
        }
        else
        {
            lastTmax = dailyPredictions[tmp].maxTemp;
            lastTmin = dailyPredictions[tmp].minTemp;
        }
        ++myDate;
    }
    qDebug() << "Observed OK";
    int outputDataLength = nrDaysBeforeWgDoy1;

    // store the climate without anomalies
    wGen = wGenClimate;
    int myYear = predictionYear;
    bool isLastMember = false;

    int anomalyMonth1 = seasonFirstDate.month;
    int anomalyMonth2 = seasonLastDate.month;

    for (unsigned int modelIndex = 0; modelIndex < nrMembers; modelIndex++)
    {
        // assign anomaly
        if (! assignXMLAnomaly(XMLAnomaly, modelIndex, anomalyMonth1, anomalyMonth2, wGenClimate, wGen))
        {
            qDebug() << "Error in Scenario: assignXMLAnomaly returns false";
            return false;
        }

        if (modelIndex == nrMembers-1 )
        {
            isLastMember = true;
        }
        // compute seasonal prediction
        std::vector<int> indexWg;
        if (! computeSeasonalPredictions(dailyObsData, wGen,
                                        predictionYear, myYear, nrRepetitions,
                                        wgDoy1, wgDoy2, rainfallThreshold, isLastMember,
                                        dailyPredictions, &outputDataLength, indexWg))
        {
            qDebug() << "Error in computeSeasonalPredictions";
            return false;
        }
        if (indexWg.size() != 0)
        {
            QDate myDate(seasonFirstDate.year, seasonFirstDate.month, seasonFirstDate.day);
            QDate lastDate(seasonLastDate.year, seasonLastDate.month, seasonLastDate.day);
            for (int currentIndex = indexWg[0]; currentIndex <= indexWg[indexWg.size()-1]; currentIndex++)
            {
                float tmin = dailyPredictions[currentIndex].minTemp;
                float tmax = dailyPredictions[currentIndex].maxTemp;
                float prec = dailyPredictions[currentIndex].prec;
                if (isLastMember && myDate>lastDate)
                {
                    myDate.setDate(myDate.year()-1, myDate.month(), myDate.day());   // l'ultimo membro può prendere 2 periodi di wg
                }
                if (waterTable->setMeteoData(myDate, tmin, tmax, prec))
                {
                    if (waterTable->getWaterTableInterpolation(myDate, &wtDepth, &myDelta, &myDeltaDays))
                    {
                        dailyPredictions[currentIndex].waterTableDepth = wtDepth;
                    }
                }
                myDate = myDate.addDays(1);
            }
        }
        // next model
        myYear = myYear + nrRepetitions;
    }

    qDebug() << "\n>>> output:" << outputFileName;

    // copy all waterTableDepth outside wg period
    int fixWgDoy1 = wgDoy1;
    int fixWgDoy2 = wgDoy2;
    int index = 0;
    QDate firstDate(dailyPredictions[0].date.year, dailyPredictions[0].date.month, dailyPredictions[0].date.day);
    QDate lastDate = firstDate.addDays(nrValues);

    for (QDate myDate = firstDate; myDate < lastDate; myDate=myDate.addDays(1))
    {
        setCorrectWgDoy(wgDoy1, wgDoy2, predictionYear, myDate.year(), fixWgDoy1, fixWgDoy2);
        if (! isWGDate(Crit3DDate(myDate.day(), myDate.month(), myDate.year()), fixWgDoy1, fixWgDoy2)
            && dailyPredictions[index].waterTableDepth == NODATA)
        {
            for (int indexToBeCopyed = 0; indexToBeCopyed < 366; indexToBeCopyed++)
            {
                if (dailyPredictions[indexToBeCopyed].date.month == myDate.month() && dailyPredictions[indexToBeCopyed].date.day == myDate.day() && dailyPredictions[indexToBeCopyed].waterTableDepth != NODATA)
                {
                    dailyPredictions[index].waterTableDepth =  dailyPredictions[indexToBeCopyed].waterTableDepth;
                    break;
                }
            }

        }
        index++;
    }

    writeMeteoDataCsv (outputFileName, separator, dailyPredictions, true);

    dailyPredictions.clear();

    return true;
}


/*!
  \name computeSeasonalPredictions
  \brief Generates a time series of daily data (Tmin, Tmax, Prec)
    The Length is equals to nrRepetitions years, starting from firstYear
    Period between wgDoy1 and wgDoy2 is produced by the WG
    Others data are a copy of the observed data of predictionYear
    Weather generator climate is stored in wgClimate
    Observed data (Tmin, Tmax, Prec) are in dailyObsData
  \return outputDailyData
*/
bool computeSeasonalPredictions(TinputObsData *dailyObsData, TweatherGenClimate &wgClimate,
                                int predictionYear, int firstYear, int nrRepetitions,
                                int wgDoy1, int wgDoy2, float rainfallThreshold, bool isLastMember,
                                std::vector<ToutputDailyMeteo>& outputDailyData, int *outputDataLength, std::vector<int>& indexWg)

{
    Crit3DDate myDate, obsDate;
    Crit3DDate firstDate, lastDate;
    int lastYear, myDoy;
    int obsIndex, currentIndex;
    int fixWgDoy1 = wgDoy1;
    int fixWgDoy2 = wgDoy2;

    currentIndex = *outputDataLength;

    firstDate = outputDailyData[currentIndex-1].date.addDays(1);

    if (wgDoy1 < wgDoy2)
    {
        lastYear = firstYear + nrRepetitions - 1;

        if (isLastMember)
        {
            if ( (!isLeapYear(predictionYear) && !isLeapYear(lastYear)) || (isLeapYear(predictionYear) && isLeapYear(lastYear)))
            {
                lastDate = getDateFromDoy(lastYear,wgDoy2);
            }
            else
            {
                if(isLeapYear(predictionYear) && wgDoy2 >= 60 )
                    lastDate = getDateFromDoy(lastYear, wgDoy2-1);
                if(isLeapYear(lastYear) && wgDoy2 >= 59 )
                    lastDate = getDateFromDoy(lastYear, wgDoy2+1);
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
            {
                lastDate = getDateFromDoy(lastYear, wgDoy2);
            }
            else
            {
                if(isLeapYear(predictionYear+1) && wgDoy2 >= 60)
                    lastDate = getDateFromDoy(lastYear, wgDoy2-1);
                if(isLeapYear(lastYear) && wgDoy2 >= 59 )
                    lastDate = getDateFromDoy(lastYear, wgDoy2+1);
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

        setCorrectWgDoy(wgDoy1, wgDoy2, predictionYear, myDate.year, fixWgDoy1, fixWgDoy2);
        myDoy = getDoyFromDate(myDate);

        // fill mydailyData.date
        initializeDailyDataBasic (&outputDailyData[currentIndex], myDate);

        if ( isWGDate(myDate, fixWgDoy1, fixWgDoy2) )
        {
            outputDailyData[currentIndex].maxTemp = getTMax(myDoy, rainfallThreshold, wgClimate);
            outputDailyData[currentIndex].minTemp = getTMin(myDoy, rainfallThreshold, wgClimate);
            if (outputDailyData[currentIndex].maxTemp < outputDailyData[currentIndex].minTemp)
            {
                float average,diff;
                average = 0.5*(outputDailyData[currentIndex].maxTemp + outputDailyData[currentIndex].minTemp);
                diff = outputDailyData[currentIndex].minTemp - outputDailyData[currentIndex].maxTemp;
                outputDailyData[currentIndex].maxTemp = average + 0.5*diff;
                outputDailyData[currentIndex].minTemp = average - 0.5*diff;
            }
            outputDailyData[currentIndex].prec = getPrecip(myDoy, rainfallThreshold, wgClimate);
            indexWg.push_back(currentIndex);
        }
        else
        {

            obsDate.day = myDate.day;
            obsDate.month = myDate.month;

            if (myDoy < fixWgDoy1)
                obsDate.year = predictionYear;
            else if (myDoy > fixWgDoy2)
                obsDate.year = predictionYear-1;

            obsIndex = difference(dailyObsData->inputFirstDate, obsDate);

            if ( obsIndex >= 0 && obsIndex <= dailyObsData->dataLength )
            {
                outputDailyData[currentIndex].maxTemp = dailyObsData->inputTMax[obsIndex];
                outputDailyData[currentIndex].minTemp = dailyObsData->inputTMin[obsIndex];
                outputDailyData[currentIndex].prec = dailyObsData->inputPrecip[obsIndex];
            }
            else
            {
                qDebug() << "Error: wrong date in computeSeasonalPredictions";
                return false;
            }

        }
        currentIndex++;
     }

     *outputDataLength = currentIndex;
     return true;
}


/*!
  \name computeClimate
  \brief Generates a time series of daily data (Tmin, Tmax, Prec)
    The Length is equals to nrRepetitions years, starting from firstYear
    climate indexes are stored in wgClimate
  \return outputDailyData
*/
bool computeClimate(TweatherGenClimate &wgClimate, int firstYear, int nrRepetitions,
                    float rainfallThreshold, std::vector<ToutputDailyMeteo> &outputDailyData)
{
    Crit3DDate firstDate = Crit3DDate(1, 1, firstYear);
    int lastYear = firstYear + nrRepetitions - 1;
    Crit3DDate lastDate = Crit3DDate(31, 12, lastYear);

    // initialize output array
    int nrDays = firstDate.daysTo(lastDate) + 1;
    outputDailyData.resize(unsigned(nrDays));

    // initialize WG
    initializeWeather(wgClimate);

    unsigned int index = 0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        initializeDailyDataBasic (&outputDailyData[index], myDate);

        int myDoy = getDoyFromDate(myDate);

        outputDailyData[index].maxTemp = getTMax(myDoy, rainfallThreshold, wgClimate);
        outputDailyData[index].minTemp = wgClimate.state.currentTmin;
        outputDailyData[index].prec = wgClimate.state.currentPrec;

        index++;
    }

    return true;
}

