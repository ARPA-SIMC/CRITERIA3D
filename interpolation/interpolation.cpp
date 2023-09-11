/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/

#include <ostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "furtherMathFunctions.h"
#include "statistics.h"
#include "basicMath.h"
#include "meteoPoint.h"
#include "gis.h"
#include "spatialControl.h"
#include "interpolation.h"
#include <functional>

using namespace std;

float getMinHeight(const std::vector<Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode)
{
    float myZmin = NODATA;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].point->z != NODATA && myPoints[i].isActive && checkLapseRateCode(myPoints[i].lapseRateCode, useLapseRateCode, true))
            if (myZmin == NODATA || myPoints[i].point->z < myZmin)
                myZmin = float(myPoints[i].point->z);
    return myZmin;
}

float getMaxHeight(const std::vector<Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode)
{
    float zMax;
    zMax = NODATA;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].value != NODATA && myPoints[i].isActive && checkLapseRateCode(myPoints[i].lapseRateCode, useLapseRateCode, true))
            if (zMax == NODATA || (myPoints[i]).point->z > zMax)
                zMax = float(myPoints[i].point->z);

    return zMax;
}

float getZmin(const std::vector<Crit3DInterpolationDataPoint> &myPoints)
{
    float myZmin = NODATA;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].point->z != NODATA)
            if (myZmin == NODATA || myPoints[i].point->z < myZmin)
                myZmin = float(myPoints[i].point->z);
    return myZmin;
}

float getZmax(const std::vector<Crit3DInterpolationDataPoint> &myPoints)
{
    float myZmax = 0;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].point->z > myZmax)
            myZmax = float(myPoints[i].point->z);
    return myZmax;
}

float getProxyMaxValue(std::vector<Crit3DInterpolationDataPoint> &myPoints, unsigned pos)
{
    float maxValue = NODATA;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].getProxyValue(pos) != NODATA)
            if (maxValue == NODATA || myPoints[i].getProxyValue(pos) > maxValue)
                maxValue = myPoints[i].getProxyValue(pos);

    return maxValue;
}

float getProxyMinValue(std::vector<Crit3DInterpolationDataPoint> &myPoints, unsigned pos)
{
    float minValue = NODATA;

    for (unsigned i = 0; i < myPoints.size(); i++)
        if (myPoints[i].getProxyValue(pos) != NODATA)
            if (minValue == NODATA || myPoints[i].getProxyValue(pos) < minValue)
                minValue = myPoints[i].getProxyValue(pos);

    return minValue;
}

unsigned sortPointsByDistance(unsigned maxIndex, std::vector<Crit3DInterpolationDataPoint> &myPoints, std::vector<Crit3DInterpolationDataPoint> &myValidPoints)
{   
    if (myPoints.size() == 0) return 0;

    unsigned i, first, index, outIndex;
    float min_value = NODATA;
    std::vector<unsigned> indici_ordinati;
    std::vector<unsigned> indice_minimo;

    indici_ordinati.resize(maxIndex);
    indice_minimo.resize(myPoints.size());

    first = 0;
    index = 0;

    bool exit = false;

    while (index < maxIndex && !exit)
    {
        if (first == 0)
        {
            i = 0;
            while ((! myPoints[i].isActive || (isEqual(myPoints[i].distance, 0))) && (i < myPoints.size()-1))
                i++;

            if (i == (myPoints.size()-1) && ! myPoints[i].isActive)
                exit=true;
            else
            {
                first = 1;
                indice_minimo[first-1] = i;
                min_value = myPoints[i].distance;
            }
        }
        else
            min_value = myPoints[unsigned(indice_minimo[first-1])].distance;

        if (!exit)
        {
            for (i = unsigned(indice_minimo[first-1]) + 1; i < myPoints.size(); i++)
                if (isEqual(min_value, NODATA) || myPoints[i].distance < min_value)
                    if (myPoints[i].isActive)
                        if (myPoints[i].distance > 0)
                        {
                            first++;
                            min_value = myPoints[i].distance;
                            indice_minimo[first-1] = i;
                        }

            indici_ordinati[index] = indice_minimo[first-1];
            myPoints[indice_minimo[first-1]].isActive = false;
            index++;
            first--;
        }
    }

    outIndex = MINVALUE(index, maxIndex);
    myValidPoints.clear();

    for (i=0; i < outIndex; i++)
    {
        myPoints[indici_ordinati[i]].isActive = true;
        myValidPoints.push_back(myPoints[indici_ordinati[i]]);
    }

    indici_ordinati.clear();
    indice_minimo.clear();
    return outIndex;
}

void computeDistances(meteoVariable myVar, vector <Crit3DInterpolationDataPoint> &myPoints,  Crit3DInterpolationSettings* mySettings,
                      float x, float y, float z, bool excludeSupplemental)
{
    int row, col;

    for (unsigned long i = 0; i < myPoints.size() ; i++)
    {
        if (excludeSupplemental && ! checkLapseRateCode(myPoints[i].lapseRateCode, mySettings->getUseLapseRateCode(), false))
            myPoints[i].distance = 0;
        else
        {            
            myPoints[i].distance = gis::computeDistance(x, y, float((myPoints[i]).point->utm.x) , float((myPoints[i]).point->utm.y));

            if (mySettings->getUseTD() && getUseTdVar(myVar))
            {
                float topoDistance = 0.;
                int kh = mySettings->getTopoDist_Kh();
                if (kh != 0)
                {
                    topoDistance = NODATA;
                    if (myPoints[i].topographicDistance != nullptr)
                    {
                        if (! gis::isOutOfGridXY(x, y, myPoints[i].topographicDistance->header))
                        {
                            gis::getRowColFromXY(*(myPoints[i].topographicDistance->header), x, y, &row, &col);
                            topoDistance = myPoints[i].topographicDistance->value[row][col];
                        }
                    }

                    if (isEqual(topoDistance, NODATA))
                        topoDistance = topographicDistance(x, y, z, float(myPoints[i].point->utm.x),
                                                           float(myPoints[i].point->utm.y),
                                                           float(myPoints[i].point->z), myPoints[i].distance,
                                                           *(mySettings->getCurrentDEM()));
                }

                myPoints[i].distance += (kh * topoDistance);
            }
        }
    }
}


bool neighbourhoodVariability(meteoVariable myVar, std::vector <Crit3DInterpolationDataPoint> &myInterpolationPoints,
                              Crit3DInterpolationSettings* mySettings,
                              float x, float y, float z, int nMax,
                              float* devSt, float* avgDeltaZ, float* minDistance)
{
    int i, max_points;
    float* dataNeighborhood;
    float myValue;
    vector <float> deltaZ;
    vector <Crit3DInterpolationDataPoint> validPoints;

    computeDistances(myVar, myInterpolationPoints, mySettings, x, y, z, true);
    max_points = sortPointsByDistance(nMax, myInterpolationPoints, validPoints);

    if (max_points > 1)
    {
        dataNeighborhood = (float *) calloc (max_points, sizeof(float));

        for (i=0; i<max_points; i++)
        {
            myValue = validPoints[i].value;
            dataNeighborhood[i] = myValue;
        }

        *devSt = statistics::standardDeviation(dataNeighborhood, max_points);

        *minDistance = validPoints[0].distance;

        deltaZ.clear();
        if (z != NODATA)
            deltaZ.push_back(1);

        for (i=0; i<max_points;i++)
        {
            if ((validPoints[i]).point->z != NODATA)
            {
                deltaZ.push_back(float(fabs(validPoints[i].point->z - z)));
            }
        }

        *avgDeltaZ = statistics::mean(deltaZ.data(), int(deltaZ.size()));

        return true;
    }
    else
        return false;
}


bool regressionSimple(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                      unsigned proxyPosition, bool isZeroIntercept, float* myCoeff, float* myIntercept, float* myR2)
{
    unsigned i;
    float myProxyValue;
    Crit3DInterpolationDataPoint myPoint;
    vector <float> myValues, myZ;

    *myCoeff = NODATA;
    *myIntercept = NODATA;
    *myR2 = NODATA;

    myValues.clear();
    myZ.clear();

    for (i = 0; i < myPoints.size(); i++)
    {
        myPoint = myPoints[i];
        if (myPoint.isActive)
        {
            if (proxyPosition != mySettings->getIndexHeight() || checkLapseRateCode(myPoint.lapseRateCode, mySettings->getUseLapseRateCode(), true))
            {
                myProxyValue = myPoint.getProxyValue(proxyPosition);
                if (! isEqual(myProxyValue, NODATA))
                {
                    myValues.push_back(myPoint.value);
                    myZ.push_back(myProxyValue);
                }
            }
        }
    }

    if (myValues.size() >= MIN_REGRESSION_POINTS)
    {
        statistics::linearRegression((float*)(myZ.data()), (float*)(myValues.data()), (long)(myZ.size()), isZeroIntercept,
                                     myIntercept, myCoeff, myR2);
        return true;
    }
    else
        return false;
}

bool regressionGeneric(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                       int proxyPos, bool isZeroIntercept)
{
    float q, m, r2;

    if (! regressionSimple(myPoints, mySettings, proxyPos, isZeroIntercept, &m, &q, &r2))
        return false;

    Crit3DProxy* myProxy = mySettings->getProxy(proxyPos);
    myProxy->setRegressionSlope(m);
    myProxy->setRegressionIntercept(q);
    myProxy->setRegressionR2(r2);
    myProxy->setLapseRateT0(q);

    // clean inversion (only thermal variables)
    myProxy->setInversionIsSignificative(false);
    myProxy->setInversionLapseRate(NODATA);

    return (r2 >= mySettings->getMinRegressionR2());
}


bool regressionSimpleT(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings, Crit3DClimateParameters* myClimate,
                       Crit3DTime myTime, meteoVariable myVar, unsigned orogProxyPos)
{
    float q, m, r2;

    Crit3DProxy* myProxyOrog = mySettings->getProxy(orogProxyPos);
    myProxyOrog->initializeOrography();

    if (! regressionSimple(myPoints, mySettings, orogProxyPos, false, &m, &q, &r2))
        return false;

    if (r2 < mySettings->getMinRegressionR2())
        return false;

    myProxyOrog->setRegressionSlope(m);
    myProxyOrog->setRegressionR2(r2);
    myProxyOrog->setLapseRateT0(q);

    // only pre-inversion data
    if (m > 0)
    {
        myProxyOrog->setInversionLapseRate(m);

        float maxZ = MINVALUE(getMaxHeight(myPoints, mySettings->getUseLapseRateCode()), mySettings->getMaxHeightInversion());
        myProxyOrog->setLapseRateT1(q + m * maxZ);
        myProxyOrog->setLapseRateH1(maxZ);
        myProxyOrog->setRegressionSlope(myClimate->getClimateLapseRate(myVar, myTime));
        myProxyOrog->setInversionIsSignificative(true);
    }

    return true;
}


float findHeightIntervalAvgValue(bool useLapseRateCode, std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                 float heightInf, float heightSup, float maxPointsZ)
{
    float myValue, mySum, nValues;

    mySum = 0.;
    nValues = 0;

    for (long i = 0; i < long(myPoints.size()); i++)
         if (myPoints[i].point->z != NODATA && myPoints[i].isActive && checkLapseRateCode(myPoints[i].lapseRateCode, useLapseRateCode, true))
            if (myPoints[i].point->z >= heightInf && myPoints[i].point->z <= heightSup)
            {                    
                myValue = (myPoints[i]).value;
                if (myValue != NODATA)
                {
                    mySum += myValue;
                    nValues ++;
                }
            }

    if (nValues > 1 || (nValues > 0 && heightSup >= maxPointsZ))
        return (mySum / nValues);
    else
        return NODATA;
}

bool regressionOrographyT(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings, Crit3DClimateParameters* myClimate,
                          Crit3DTime myTime, meteoVariable myVar, int orogProxyPos, bool climateExists)
{
    long i;
    float heightInf, heightSup;
    float myAvg;
    vector <float> myData1, myData2;
    vector <float> myHeights1, myHeights2;
    vector <float> myIntervalsHeight, myIntervalsHeight1, myIntervalsHeight2;
    vector <float> myIntervalsValues, myIntervalsValues1, myIntervalsValues2;
    float m, q, r2;
    float r2_values, r2_intervals;
    float q1, m1, r21, q2, m2, r22;
    float mySignificativeR2, mySignificativeR2Inv;
    float maxPointsZ, deltaZ;
    float climateLapseRate;
    float x,y;
    float DELTAZ_INI = 80.;
    float maxHeightInv = mySettings->getMaxHeightInversion();

    Crit3DProxy* myProxyOrog = mySettings->getProxy(orogProxyPos);

    mySignificativeR2 = MAXVALUE(mySettings->getMinRegressionR2(), float(0.2));
    mySignificativeR2Inv = MAXVALUE(mySettings->getMinRegressionR2(), float(0.1));

    /*! initialize */
    myProxyOrog->initializeOrography();

    if (climateExists)
        climateLapseRate = myClimate->getClimateLapseRate(myVar, myTime);
    else
        climateLapseRate = 0.;

    myProxyOrog->setRegressionSlope(climateLapseRate);

    maxPointsZ = getMaxHeight(myPoints, mySettings->getUseLapseRateCode());
    heightInf = getMinHeight(myPoints, mySettings->getUseLapseRateCode());

    /*! not enough data to define a curve (use climate) */
    if ((maxPointsZ == heightInf) || (myPoints.size() < MIN_REGRESSION_POINTS))
        return true;

    /*! find intervals averages */

    heightSup = heightInf;
    deltaZ = DELTAZ_INI;
    while (heightSup <= maxPointsZ)
    {
        myAvg = NODATA;
        while (myAvg == NODATA)
        {
            heightSup = heightSup + deltaZ;
            myAvg = findHeightIntervalAvgValue(mySettings->getUseLapseRateCode(), myPoints, heightInf, heightSup, maxPointsZ);
        }
        myIntervalsHeight.push_back((heightSup + heightInf) / float(2.));
        myIntervalsValues.push_back(myAvg);

        deltaZ = DELTAZ_INI * expf(heightInf / maxHeightInv);
        heightInf = heightSup;
    }

    /*! find inversion height */
    myProxyOrog->setLapseRateT1(myIntervalsValues[0]);
    myProxyOrog->setLapseRateH1(myIntervalsHeight[0]);
    for (i = 1; i < long(myIntervalsValues.size()); i++)
        if (myIntervalsHeight[i] <= maxHeightInv && (myIntervalsValues[i] >= myProxyOrog->getLapseRateT1()) && (myIntervalsValues[i] > (myIntervalsValues[0] + 0.001 * (myIntervalsHeight[i] - myIntervalsHeight[0]))))
        {
            myProxyOrog->setLapseRateH1(myIntervalsHeight[i]);
            myProxyOrog->setLapseRateT1(myIntervalsValues[i]);
            myProxyOrog->setInversionIsSignificative(true);
        }

    /*! no inversion: try regression with all data */
    if (! myProxyOrog->getInversionIsSignificative())
        return (regressionGeneric(myPoints, mySettings, orogProxyPos, false));

    /*! create vectors below and above inversion */
    for (i = 0; i < long(myPoints.size()); i++)
        if (myPoints[i].point->z != NODATA && checkLapseRateCode(myPoints[i].lapseRateCode, mySettings->getUseLapseRateCode(), true))
        {
            if (myPoints[i].point->z <= myProxyOrog->getLapseRateH1())
            {
                myData1.push_back(myPoints[i].value);
                myHeights1.push_back(float(myPoints[i].point->z));
            }
            else
            {
                myData2.push_back(myPoints[i].value);
                myHeights2.push_back(float(myPoints[i].point->z));
            }
        }

    /*! create vectors of height intervals below and above inversion */
    for (i = 0; i < long(myIntervalsValues.size()); i++)
        if (myIntervalsHeight[i] <= myProxyOrog->getLapseRateH1())
        {
            myIntervalsValues1.push_back(myIntervalsValues[i]);
            myIntervalsHeight1.push_back(myIntervalsHeight[i]);
        }
        else
        {
            myIntervalsValues2.push_back(myIntervalsValues[i]);
            myIntervalsHeight2.push_back(myIntervalsHeight[i]);
        }


    /*! only positive lapse rate*/
    if (myProxyOrog->getInversionIsSignificative() && myIntervalsValues1.size() == myIntervalsValues.size())
    {
        if (! regressionSimple(myPoints, mySettings, orogProxyPos, false, &m, &q, &r2))
            return false;

        if (r2 >= mySignificativeR2)
        {
            myProxyOrog->setInversionLapseRate(m);
            myProxyOrog->setRegressionR2(r2);
            myProxyOrog->setLapseRateT0(q);
            myProxyOrog->setLapseRateT1(q + m * myProxyOrog->getLapseRateH1());
        }
        else
        {
            statistics::linearRegression(myIntervalsHeight1.data(), myIntervalsValues1.data(), (long)myIntervalsHeight1.size(), false, &q, &m, &r2);

            myProxyOrog->setRegressionR2(NODATA);

            if (r2 >= mySignificativeR2)
            {
                myProxyOrog->setLapseRateT0(q);
                myProxyOrog->setInversionLapseRate(m);
                myProxyOrog->setLapseRateT1(q + m * myProxyOrog->getLapseRateH1());
            }
            else
            {
                myProxyOrog->setInversionLapseRate(0.);
                myProxyOrog->setLapseRateT0(myIntervalsValues[0]);
                myProxyOrog->setLapseRateT0(myIntervalsValues[0]);
            }
        }

        return true;
    }

    /*! check inversion significance */
    statistics::linearRegression(myHeights1.data(), myData1.data(), long(myHeights1.size()), false, &q1, &m1, &r2_values);
    if (myIntervalsValues1.size() > 2)
        statistics::linearRegression(myIntervalsHeight1.data(), myIntervalsValues1.data(), long(myIntervalsHeight1.size()), false, &q, &m, &r2_intervals);
    else
        r2_intervals = 0.;

    /*! inversion is not significant with data neither with intervals */
    if (r2_values < mySignificativeR2Inv && r2_intervals < mySignificativeR2Inv)
    {
        if (! regressionSimple(myPoints, mySettings, orogProxyPos, false, &m, &q, &r2))
            return false;

        /*! case 0: regression with all data much significant */
        if (r2 >= 0.5)
        {
            myProxyOrog->setInversionIsSignificative(false);
            myProxyOrog->setLapseRateH1(NODATA);
            myProxyOrog->setInversionLapseRate(NODATA);
            myProxyOrog->setRegressionSlope(MINVALUE(m, float(0.0)));
            myProxyOrog->setRegressionR2(r2);
            myProxyOrog->setLapseRateT0(q);
            myProxyOrog->setLapseRateT1(NODATA);
            return true;
        }

        /*! case 1: analysis only above inversion, flat lapse rate below */
        myProxyOrog->setInversionLapseRate(0.);
        statistics::linearRegression(myHeights2.data(), myData2.data(), long(myHeights2.size()), false, &q2, &m2, &r2);

        if (myData2.size() >= MIN_REGRESSION_POINTS)
        {
            if (r2 >= mySignificativeR2)
            {
                myProxyOrog->setRegressionSlope(MINVALUE(m2, float(0.0)));
                myProxyOrog->setLapseRateT0(q2 + myProxyOrog->getLapseRateH1() * myProxyOrog->getRegressionSlope());
                myProxyOrog->setLapseRateT1(myProxyOrog->getLapseRateT0());
                myProxyOrog->setRegressionR2(r2);
                return true;
            }
            else
            {
                statistics::linearRegression(myIntervalsHeight2.data(), myIntervalsValues2.data(),
                                             int(myIntervalsHeight2.size()), false, &q2, &m2, &r2);
                if (r2 >= mySignificativeR2)
                {
                    myProxyOrog->setRegressionSlope(MINVALUE(m2, float(0.0)));
                    myProxyOrog->setLapseRateT0(q2 + myProxyOrog->getLapseRateH1() * myProxyOrog->getRegressionSlope());
                    myProxyOrog->setLapseRateT1(myProxyOrog->getLapseRateT0());;
                    myProxyOrog->setRegressionR2(r2);
                    return true;
                }
            }
        }

        myProxyOrog->setInversionIsSignificative(false);
        myProxyOrog->setLapseRateH1(NODATA);
        myProxyOrog->setInversionLapseRate(NODATA);
        myProxyOrog->setLapseRateT0(q);
        myProxyOrog->setLapseRateT1(NODATA);

        /*! case 2: regression with data */
        if (! regressionSimple(myPoints, mySettings, orogProxyPos, false, &m, &q, &r2))
            return false;

        if (r2 >= mySignificativeR2)
        {
            myProxyOrog->setRegressionSlope(MINVALUE(m, 0));
            myProxyOrog->setLapseRateT0(q);
            myProxyOrog->setRegressionR2(r2);
            return true;
        }
        else
        {
            myProxyOrog->setLapseRateT0(myIntervalsValues[0]);
            if (m > 0.)
                myProxyOrog->setRegressionSlope(0.);
            else
                myProxyOrog->setRegressionSlope(climateLapseRate);

            return true;
        }

    }

    /*! significance analysis */
    statistics::linearRegression(myHeights1.data(), myData1.data(), int(myHeights1.size()), false, &q1, &m1, &r21);
    statistics::linearRegression(myHeights2.data(), myData2.data(), int(myHeights2.size()), false, &q2, &m2, &r22);

    if (m1 <= 0)
        r21 = 0;

    myProxyOrog->setRegressionR2(r22);

    if (r21 >= mySignificativeR2Inv && r22 >= mySignificativeR2)
    {
        if (myHeights2.size() < MIN_REGRESSION_POINTS && m2 > 0.)
        {
            m2 = 0.;
            q2 = myProxyOrog->getLapseRateT1();
        }
        findLinesIntersection(q1, m1, q2, m2, &x, &y);
        myProxyOrog->setLapseRateT0(q1);
        myProxyOrog->setLapseRateT1(y);
        myProxyOrog->setInversionLapseRate(m1);
        myProxyOrog->setRegressionSlope(m2);
        myProxyOrog->setLapseRateH1(x);
        if (myProxyOrog->getLapseRateH1() > maxHeightInv)
        {
            myProxyOrog->setLapseRateT1(myProxyOrog->getLapseRateT1() - (myProxyOrog->getLapseRateH1() - maxHeightInv) * myProxyOrog->getRegressionSlope());
            myProxyOrog->setLapseRateH1(maxHeightInv);
            myProxyOrog->setInversionLapseRate((myProxyOrog->getLapseRateT1() - myProxyOrog->getLapseRateT0()) / (myProxyOrog->getLapseRateH1() - myProxyOrog->getLapseRateH0()));
        }
        return true;
    }
    else if (r21 < mySignificativeR2Inv && r22 >= mySignificativeR2)
    {
        if (myHeights2.size() < MIN_REGRESSION_POINTS && m2 > 0.)
        {
            m2 = 0.;
            q2 = myProxyOrog->getLapseRateT1();
        }

        statistics::linearRegression(myIntervalsHeight1.data(), myIntervalsValues1.data(),
                                     long(myIntervalsHeight1.size()), false, &q, &m, &r2);

        myProxyOrog->setRegressionSlope(m2);
        if (r2 >= mySignificativeR2Inv)
        {
            if (findLinesIntersectionAboveThreshold(q, m, q2, m2, 40, &x, &y))
            {
                myProxyOrog->setLapseRateH1(x);
                myProxyOrog->setLapseRateT0(q);
                myProxyOrog->setLapseRateT1(y);
                myProxyOrog->setInversionLapseRate(m);
                if (myProxyOrog->getLapseRateH1() > maxHeightInv)
                {
                    myProxyOrog->setLapseRateT1(myProxyOrog->getLapseRateT1() - (myProxyOrog->getLapseRateH1() - maxHeightInv) * myProxyOrog->getRegressionSlope());
                    myProxyOrog->setLapseRateH1(maxHeightInv);
                    myProxyOrog->setInversionLapseRate((myProxyOrog->getLapseRateT1() - myProxyOrog->getLapseRateT0()) / (myProxyOrog->getLapseRateH1() - myProxyOrog->getLapseRateH0()));
                }
                return true;
            }
        }
        else
        {
            myProxyOrog->setInversionLapseRate(0.);
            myProxyOrog->setLapseRateT1(q2 + m2 * myProxyOrog->getLapseRateH1());
            myProxyOrog->setLapseRateT0(myProxyOrog->getLapseRateT1());
            return true;
        }
    }

    else if (r21 >= mySignificativeR2Inv && r22 < mySignificativeR2)
    {
        myProxyOrog->setLapseRateT0(q1);
        myProxyOrog->setInversionLapseRate(m1);

        statistics::linearRegression(myIntervalsHeight2.data(), myIntervalsValues2.data(),
                                     long(myIntervalsHeight2.size()), false, &q, &m, &r2);
        if (r2 >= mySignificativeR2)
        {
            myProxyOrog->setRegressionSlope(MINVALUE(m, float(0.)));
            findLinesIntersection(myProxyOrog->getLapseRateT0(), myProxyOrog->getInversionLapseRate(), q, myProxyOrog->getRegressionSlope(), &x, &y);
            myProxyOrog->setLapseRateH1(x);
            myProxyOrog->setLapseRateT1(y);
        }
        else
        {
            myProxyOrog->setRegressionSlope(climateLapseRate);
            findLinesIntersection(myProxyOrog->getLapseRateT0(), myProxyOrog->getInversionLapseRate(), myProxyOrog->getLapseRateT1() - myProxyOrog->getRegressionSlope()* myProxyOrog->getLapseRateH1(), myProxyOrog->getRegressionSlope(), &x, &y);
            myProxyOrog->setLapseRateH1(x);
            myProxyOrog->setLapseRateT1(y);
        }
        return true;
    }

    else if (r21 < mySignificativeR2Inv && r22 < mySignificativeR2)
    {
        statistics::linearRegression(myIntervalsHeight1.data(), myIntervalsValues1.data(),
                                     long(myIntervalsHeight1.size()), false, &q, &m, &r2);

        if (r2 >= mySignificativeR2Inv)
        {
            myProxyOrog->setLapseRateT0(q);
            myProxyOrog->setInversionLapseRate(m);
            myProxyOrog->setLapseRateT1(q + m * myProxyOrog->getLapseRateH1());
        }
        else
        {
            myProxyOrog->setInversionLapseRate(0.);
            myProxyOrog->setLapseRateT0(myIntervalsValues[0]);
            myProxyOrog->setLapseRateT1(myProxyOrog->getLapseRateT0());
        }

        statistics::linearRegression(myIntervalsHeight2.data(), myIntervalsValues2.data(),
                                     long(myIntervalsHeight2.size()), false, &q, &m, &r2);

        if (r2 >= mySignificativeR2)
        {
            myProxyOrog->setRegressionSlope(MINVALUE(m, 0));
            if (findLinesIntersectionAboveThreshold(myProxyOrog->getLapseRateT0(), myProxyOrog->getInversionLapseRate(), q, myProxyOrog->getRegressionSlope(), 40, &x, &y))
            {
                myProxyOrog->setLapseRateH1(x);
                myProxyOrog->setLapseRateT1(y);
                return true;
            }
        }
        else
        {
            myProxyOrog->setRegressionSlope(climateLapseRate);
            return true;
        }

    }

    /*! check max lapse rate (20 C / 1000 m) */
    if (myProxyOrog->getRegressionSlope() < -0.02)
        myProxyOrog->setRegressionSlope((float)-0.02);

    myProxyOrog->initializeOrography();
    return (regressionGeneric(myPoints, mySettings, orogProxyPos, false));

}

float computeShepardInitialRadius(float area, unsigned int allPointsNr, unsigned int minPointsNr)
{
    return float(sqrt((minPointsNr * area) / (float(PI) * allPointsNr)));
}

float shepardSearchNeighbour(vector <Crit3DInterpolationDataPoint> &inputPoints,
                             Crit3DInterpolationSettings* settings,
                             vector <Crit3DInterpolationDataPoint> &outputPoints)
{
    std::vector <Crit3DInterpolationDataPoint> shepardNeighbourPoints;

    unsigned int i;
    float radius;
    unsigned int nrValid = 0;
    float shepardInitialRadius = computeShepardInitialRadius(settings->getPointsBoundingBoxArea(), unsigned(inputPoints.size()), SHEPARD_AVG_NRPOINTS);

    // define a first neighborhood inside initial radius
    for (i=0; i < inputPoints.size(); i++)
        if (inputPoints[i].distance <= shepardInitialRadius &&
            inputPoints[i].distance > 0 &&
            inputPoints[i].index != settings->getIndexPointCV())
        {
            shepardNeighbourPoints.push_back(inputPoints[i]);
            nrValid++;
        }

    if (shepardNeighbourPoints.size() <= SHEPARD_MIN_NRPOINTS)
    {
        nrValid = sortPointsByDistance(SHEPARD_MIN_NRPOINTS + 1, inputPoints, outputPoints);
        if (nrValid > SHEPARD_MIN_NRPOINTS)
        {
            radius = outputPoints[SHEPARD_MIN_NRPOINTS].distance;
            outputPoints.pop_back();
        }
        else
            radius = outputPoints[nrValid-1].distance + float(EPSILON);
    }
    else if (shepardNeighbourPoints.size() > SHEPARD_MAX_NRPOINTS)
    {
        nrValid = sortPointsByDistance(SHEPARD_MAX_NRPOINTS + 1, shepardNeighbourPoints, outputPoints);
        radius = outputPoints[SHEPARD_MAX_NRPOINTS].distance;
        outputPoints.pop_back();
    }
    else
    {
        outputPoints = shepardNeighbourPoints;
        radius = shepardInitialRadius;
    }

    return radius;
}

float shepardIdw(vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* settings, float X, float Y)
{
    std::vector <Crit3DInterpolationDataPoint> shepardValidPoints;

    float radius = shepardSearchNeighbour(myPoints, settings, shepardValidPoints);

    unsigned int i, j;
    float weightSum, radius_27_4, radius_3, tmp, cosine, result;
    std::vector <float> weight, t, S;

    unsigned int nrValid = unsigned(shepardValidPoints.size());

    weight.resize(nrValid);
    t.resize(nrValid);
    S.resize(nrValid);

    weightSum = 0;
    radius_3 = radius / 3;
    radius_27_4 = (27 / 4) / radius;
    for (i=0; i < nrValid; i++)
        if (shepardValidPoints[i].distance > 0)
        {
            if (shepardValidPoints[i].distance <= radius_3)
                S[i] = 1 / (shepardValidPoints[i].distance);
            else if (shepardValidPoints[i].distance <= radius)
            {
                tmp = (shepardValidPoints[i].distance / radius) - 1;
                S[i] = radius_27_4 * tmp * tmp;
            }
            else
                S[i] = 0;

            weightSum = weightSum + S[i];
        }

    if (weightSum == 0)
        return NODATA;

    // including direction
    for (i=0; i < nrValid; i++)
    {
        t[i] = 0;
        for (j=0; j < nrValid; j++)
            if (i != j)
            {
                cosine = ((X - (float)shepardValidPoints[i].point->utm.x) * (X - (float)shepardValidPoints[j].point->utm.x) + (Y - (float)shepardValidPoints[i].point->utm.y) * (Y - (float)shepardValidPoints[j].point->utm.y)) / (shepardValidPoints[i].distance * shepardValidPoints[j].distance);
                t[i] = t[i] + S[j] * (1 - cosine);
            }

        if (weightSum != 0)
            t[i] /= weightSum;
    }

    // weights
    weightSum = 0;
    for (i=0; i < nrValid; i++)
    {
       weight[i] = S[i] * S[i] * (1 + t[i]);
       weightSum += weight[i];
    }
    for (i=0; i < nrValid; i++)
        weight[i] /= weightSum;

    result = 0;
    for (i=0; i < nrValid; i++)
        result += weight[i] * shepardValidPoints[i].value;

    return result;
}


float modifiedShepardIdw(vector <Crit3DInterpolationDataPoint> &myPoints,
                         Crit3DInterpolationSettings* settings, float radius, float X, float Y)
{
    unsigned int i;

    unsigned int j;
    float weightSum, cosine, result;
    std::vector <float> weight, t, S;
    std::vector <Crit3DInterpolationDataPoint> validPoints;

    if (radius == NODATA)
        radius = shepardSearchNeighbour(myPoints, settings, validPoints);
    else
        validPoints = myPoints;

    weight.resize(validPoints.size());
    t.resize(validPoints.size());
    S.resize(validPoints.size());

    weightSum = 0;
    for (i=0; i < validPoints.size(); i++)
        if (validPoints[i].distance > 0)
        {
            if (validPoints[i].distance <= radius)
                S[i] = (radius - validPoints[i].distance) / (radius * validPoints[i].distance);
            else
                S[i] = 0;

            weightSum = weightSum + S[i];
        }

    if (weightSum == 0)
        return NODATA;

    // including direction
    for (i=0; i < validPoints.size(); i++)
    {
        t[i] = 0;
        for (j=0; j < validPoints.size(); j++)
            if (i != j && S[i] > 0)
            {
                cosine = ((X - (float)validPoints[i].point->utm.x) * (X - (float)validPoints[j].point->utm.x) + (Y - (float)validPoints[i].point->utm.y) * (Y - (float)validPoints[j].point->utm.y)) / (validPoints[i].distance * validPoints[j].distance);
                t[i] = t[i] + S[j] * (1 - cosine);
            }

        if (weightSum != 0)
            t[i] /= weightSum;
    }

    // weights
    weightSum = 0;
    for (i=0; i < validPoints.size(); i++)
    {
        weight[i] = S[i] * S[i] * (1 + t[i]);
        weightSum += weight[i];
    }
    for (i=0; i < validPoints.size(); i++)
        weight[i] /= weightSum;

    result = 0;
    for (i=0; i < validPoints.size(); i++)
        result += weight[i] * validPoints[i].value;

    return result;
}


float inverseDistanceWeighted(vector <Crit3DInterpolationDataPoint> &myPointList)
{
    double sum, sumWeights, weight;

    sum = 0 ;
    sumWeights = 0 ;
    for (unsigned int i = 0 ; i < myPointList.size(); i++)
    {
        if (myPointList[i].distance > 0.f)
        {
            weight = double(myPointList[i].distance) / 10000.;
            weight = fabs(1 / (weight * weight * weight));
            sumWeights += weight;
            sum += double(myPointList[i].value) * weight;
        }
    }

    if (sumWeights > 0.0)
        return float(sum / sumWeights);
    else
        return NODATA;
}

/*
float gaussWeighted(vector <Crit3DInterpolationDataPoint> &myPointList)
{
    double sum, sumWeights, weight;
    double distance, deltaZ;
    double Rd=10;
    double Rz=1;

    sum = 0 ;
    sumWeights = 0 ;
    for (int i = 0 ; i < (int)(myPointList.size()); i++)
    {
        Crit3DInterpolationDataPoint myPoint = myPointList[i];
        distance = myPoint.distance / 1000.;
        deltaZ = myPoint.deltaZ / 1000.;
        if (myPoint.distance > 0.)
        {
            weight = 1 - exp(-(distance*distance)/(Rd*Rd)) * exp(-(deltaZ*deltaZ)/(Rz*Rz));
            weight = fabs(1 / (weight * weight * weight));
            sumWeights += weight;
            sum += myPoint.value * weight;
        }
    }

    if (sumWeights > 0.0)
        return float(sum / sumWeights);
    else
        return NODATA;
}
*/

// TODO elevation std dev?
void localSelection(vector <Crit3DInterpolationDataPoint> &inputPoints, vector <Crit3DInterpolationDataPoint> &selectedPoints,
                       float x, float y, Crit3DInterpolationSettings& mySettings)
{
    // search more stations to assure min points with all valid proxies
    float ratioMinPoints = float(1.2);
    unsigned minPoints = unsigned(mySettings.getMinPointsLocalDetrending() * ratioMinPoints);
    if (inputPoints.size() <= minPoints)
    {
        selectedPoints = inputPoints;
        mySettings.setLocalRadius(computeShepardInitialRadius(mySettings.getPointsBoundingBoxArea(),
                                                              unsigned(inputPoints.size()), minPoints));
    }

    for (unsigned long i = 0; i < inputPoints.size() ; i++)
        inputPoints[i].distance = gis::computeDistance(x, y, float((inputPoints[i]).point->utm.x), float((inputPoints[i]).point->utm.y));

    unsigned int nrValid = 0;
    float stepRadius = 10000;           // [m]
    float r0 = 0;                       // [m]
    float r1 = stepRadius;              // [m]
    unsigned int i;

    while (nrValid < minPoints)
    {
        for (i=0; i < inputPoints.size(); i++)
            if (inputPoints[i].distance != NODATA && inputPoints[i].distance > r0 && inputPoints[i].distance <= r1)
            {
                selectedPoints.push_back(inputPoints[i]);
                nrValid++;
            }
        r0 = r1;
        r1 += stepRadius;
    }

    for (i=0; i < selectedPoints.size(); i++)
        selectedPoints[i].regressionWeight = (1 - selectedPoints[i].distance / r1);

    mySettings.setLocalRadius(r1);
}

bool checkPrecipitationZero(std::vector <Crit3DInterpolationDataPoint> &myPoints, float precThreshold, int* nrPrecNotNull, bool* flatPrecipitation)
{
    *flatPrecipitation = true;
    *nrPrecNotNull = 0;
    float myValue = NODATA;

    for (unsigned int i = 0; i < myPoints.size(); i++)
        if (myPoints[i].isActive)
            if (int(myPoints[i].value) != int(NODATA))
                if (myPoints[i].value >= float(precThreshold))
                {
                    if (*nrPrecNotNull > 0 && myPoints[i].value != myValue)
                        *flatPrecipitation = false;

                    myValue = myPoints[i].value;
                    (*nrPrecNotNull)++;
                }

    return (*nrPrecNotNull == 0);
}


// predisposta per eventuale aggiunta wind al detrend
bool isThermal(meteoVariable myVar)
{
    if ( myVar == airTemperature ||
            myVar == airDewTemperature ||
            myVar == dailyAirTemperatureAvg ||
            myVar == dailyAirTemperatureMax ||
            myVar == dailyAirTemperatureMin ||
            myVar == elaboration )
        return true;
    else
        return false;
}


bool getUseDetrendingVar(meteoVariable myVar)
{
    if ( myVar == airTemperature ||
            myVar == airDewTemperature ||
            myVar == dailyAirTemperatureAvg ||
            myVar == dailyAirTemperatureMax ||
            myVar == dailyAirTemperatureMin ||
            myVar == elaboration )

        return true;
    else
        return false;
}

bool getUseTdVar(meteoVariable myVar)
{
    //exclude large scale variables
    if (myVar == precipitation ||
            myVar == dailyPrecipitation ||
            myVar == globalIrradiance ||
            myVar == atmTransmissivity ||
            myVar == dailyGlobalRadiation ||
            myVar == atmPressure ||
            myVar == dailyWaterTableDepth)

        return false;
    else
        return true;
}

void detrendPoints(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                   meteoVariable myVar, unsigned pos)
{
    float detrendValue, proxyValue;
    unsigned myIndex;
    Crit3DInterpolationDataPoint* myPoint;
    Crit3DProxy* myProxy;

    if (! getUseDetrendingVar(myVar)) return;

    myProxy = mySettings->getProxy(pos);

    for (myIndex = 0; myIndex < myPoints.size(); myIndex++)
    {
        detrendValue = 0;
        myPoint = &(myPoints[myIndex]);

        proxyValue = myPoint->getProxyValue(pos);

        if (getProxyPragaName(myProxy->getName()) == height)
        {
            if (proxyValue != NODATA)
            {
                float LR_above = myProxy->getRegressionSlope();
                if (myProxy->getInversionIsSignificative())
                {
                    float LR_H1 = myProxy->getLapseRateH1();
                    float LR_H0 = myProxy->getLapseRateH0();
                    float LR_below = myProxy->getInversionLapseRate();

                    if (proxyValue <= LR_H1)
                        detrendValue = MAXVALUE(proxyValue - LR_H0, 0) * LR_below;
                    else
                        detrendValue = ((LR_H1 - LR_H0) * LR_below) + (proxyValue - LR_H1) * LR_above;
                }
                else
                    detrendValue = MAXVALUE(proxyValue, 0) * LR_above;
            }
        }

        else
        {
            if (proxyValue != NODATA)
                if (myProxy->getRegressionR2() >= mySettings->getMinRegressionR2())
                    detrendValue = proxyValue * myProxy->getRegressionSlope();
        }

        myPoint->value -= detrendValue;
    }
}

float retrend(meteoVariable myVar, vector<float> myProxyValues, Crit3DInterpolationSettings* mySettings)
{

    if (! getUseDetrendingVar(myVar)) return 0.;

    float retrendValue = 0.;
    float myProxyValue;
    Crit3DProxy* myProxy;
    Crit3DProxyCombination myCombination = mySettings->getCurrentCombination();
    float proxySlope;

    for (int pos=0; pos < int(mySettings->getProxyNr()); pos++)
    {
        myProxy = mySettings->getProxy(pos);

        if (myCombination.getValue(pos) && myProxy->getIsSignificant())
        {
            myProxyValue = mySettings->getProxyValue(pos, myProxyValues);

            if (myProxyValue != NODATA)
            {
                if (mySettings->getUseMultipleDetrending())
                {
                    proxySlope = myProxy->getRegressionSlope();
                    if (proxySlope != NODATA)
                        retrendValue += (myProxyValue - myProxy->getAvg()) / myProxy->getStdDev() * proxySlope;
                }
                else
                {
                    proxySlope = myProxy->getRegressionSlope();

                    if (getProxyPragaName(myProxy->getName()) == height)
                    {
                        if (mySettings->getUseThermalInversion() && myProxy->getInversionIsSignificative())
                        {
                            float LR_H0 = myProxy->getLapseRateH0();
                            float LR_H1 = myProxy->getLapseRateH1();
                            float LR_Below = myProxy->getInversionLapseRate();
                            if (myProxyValue <= LR_H1)
                                retrendValue += (MAXVALUE(myProxyValue - LR_H0, 0) * LR_Below);
                            else
                                retrendValue += ((LR_H1 - LR_H0) * LR_Below) + (myProxyValue - LR_H1) * proxySlope;
                        }
                        else
                            retrendValue += MAXVALUE(myProxyValue, 0) * proxySlope;
                    }
                    else
                        retrendValue += myProxyValue * proxySlope;
                }
            }
        }
    }

    return retrendValue;
}


bool regressionOrography(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                         Crit3DProxyCombination myCombination, Crit3DInterpolationSettings* mySettings, Crit3DClimateParameters* myClimate,
                         Crit3DTime myTime, meteoVariable myVar, int orogProxyPos)
{
    if (getUseDetrendingVar(myVar))
    {
        if (isThermal(myVar))
        {
            if (myCombination.getUseThermalInversion())
                return regressionOrographyT(myPoints, mySettings, myClimate, myTime, myVar, orogProxyPos, true);
            else
                return regressionSimpleT(myPoints, mySettings, myClimate, myTime, myVar, orogProxyPos);
        }
        else
        {
            return regressionGeneric(myPoints, mySettings, orogProxyPos, false);
        }
    }
    else
    {
        return false;
    }
}

void detrending(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                Crit3DProxyCombination myCombination, Crit3DInterpolationSettings* mySettings, Crit3DClimateParameters* myClimate,
                meteoVariable myVar, Crit3DTime myTime)
{
    if (! getUseDetrendingVar(myVar)) return;

    Crit3DProxy* myProxy;

    for (int pos=0; pos < int(mySettings->getProxyNr()); pos++)
    {
        if (myCombination.getValue(pos))
        {
            myProxy = mySettings->getProxy(pos);
            myProxy->setIsSignificant(false);

            if (getProxyPragaName(myProxy->getName()) == height)
            {
                if (regressionOrography(myPoints, myCombination, mySettings, myClimate, myTime, myVar, pos))
                {
                    myProxy->setIsSignificant(true);
                    detrendPoints(myPoints, mySettings, myVar, pos);
                }
            }
            else
            {
                if (regressionGeneric(myPoints, mySettings, pos, false))
                {
                    myProxy->setIsSignificant(true);
                    detrendPoints(myPoints, mySettings, myVar, pos);
                }
            }
        }
    }
}

bool proxyValidity(std::vector <Crit3DInterpolationDataPoint> &myPoints, int proxyPos, float stdDevThreshold, float* avg, float* stdDev)
{
    std::vector <float> proxyValues;
    float myValue, sum = 0;
    const int MIN_NR = 10;
    unsigned i;

    *avg = NODATA;
    *stdDev = NODATA;

    for (i = 0; i < myPoints.size(); i++)
    {
        if (myPoints[i].isActive)
        {
            myValue = myPoints[i].getProxyValue(proxyPos);
            if (myValue != NODATA)
            {
                proxyValues.push_back(myValue);
                sum += myValue;
            }
        }
    }

    if (proxyValues.size() < MIN_NR) return false;

    *avg = sum / proxyValues.size();

    sum = 0;
    for (i=0; i < proxyValues.size(); i++)
        sum += (proxyValues[i] - *avg) * (proxyValues[i] - *avg);

    *stdDev = float(sqrt(sum / (proxyValues.size() - 1)));

    if (stdDevThreshold != NODATA)
        return (*stdDev > stdDevThreshold);
    else
        return true;
}

double functionTemperatureVsHeight(std::vector <double>& x, std::vector <double>& par)
{
    double y;
    y = par[0] + par[1]*x[0] + par[2]*(1/(1+exp(-par[3]*(x[0] - par[4]))));
    return y;
}

double linear(std::vector <double>& x, std::vector <double>& par)
{
    return par[0] * x[0];
}

double functionSum(const std::vector<std::function<double(std::vector<double>, std::vector<double>)>>& functions, std::vector<double> x, std::vector<double> par)
{
    double result = 0.0;
    for (const auto& function : functions)
        result += function(x,par);

    return result;
}

std::vector<std::function<double(std::vector<double>&, std::vector<double>&)>> combineFunction(Crit3DProxyCombination myCombination, Crit3DInterpolationSettings mySettings)
{
    std::vector<std::function<double(std::vector<double>&, std::vector<double>&)>> myFunc;
    for (unsigned i=0; i<myCombination.getIsActive().size(); i++)
        if (myCombination.getValue(i))
        {
            if (getProxyPragaName(mySettings.getProxy(i)->getName()) == height)
                myFunc.push_back(functionTemperatureVsHeight);
            else
                myFunc.push_back(linear);
        }

    return myFunc;
}

Crit3DProxyCombination multipleDetrending(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                        Crit3DProxyCombination myCombination, Crit3DInterpolationSettings* mySettings, meteoVariable myVar)
{
    if (! getUseDetrendingVar(myVar)) return myCombination;

    unsigned nrPredictors = 0;
    std::vector <unsigned int> proxyIndex;
    for (int pos=0; pos < int(mySettings->getProxyNr()); pos++)
    {
        if (myCombination.getValue(pos))
        {
            mySettings->getProxy(pos)->setIsSignificant(false);
            proxyIndex.push_back(pos);
            nrPredictors++;
        }
    }

    if (nrPredictors == 0) return myCombination;    

    // proxy spatial variability (1st step)
    float avg, stdDev;
    unsigned validNr;
    Crit3DProxyCombination outCombination = myCombination;
    unsigned pos;
    validNr = 0;

    for (pos=0; pos < int(mySettings->getProxyNr()); pos++)
    {
        if (myCombination.getValue(pos) && proxyValidity(myPoints, pos, mySettings->getProxy(pos)->getStdDevThreshold(), &avg, &stdDev))
        {
            outCombination.setValue(pos, true);
            validNr++;
        }
        else
            outCombination.setValue(pos, false);
    }

    if (validNr == 0) return outCombination;

    // exclude points with incomplete proxies
    unsigned i;
    std::vector <Crit3DInterpolationDataPoint> finalPoints;
    bool isValid;
    float proxyValue;

    for (i=0; i < myPoints.size(); i++)
    {
        isValid = true;
        for (pos=0; pos < mySettings->getProxyNr(); pos++)
            if (outCombination.getValue(pos))
            {
                proxyValue = myPoints[i].getProxyValue(pos);
                if (proxyValue == NODATA)
                {
                    isValid = false;
                    break;
                }
            }

        if (isValid) finalPoints.push_back(myPoints[i]);
    }

    // proxy spatial variability (2nd step)
    std::vector <float> avgs;
    std::vector <float> stdDevs;

    for (pos=0; pos < int(mySettings->getProxyNr()); pos++)
    {
        if (myCombination.getValue(pos) && proxyValidity(finalPoints, pos, mySettings->getProxy(pos)->getStdDevThreshold(), &avg, &stdDev))
        {
            avgs.push_back(avg);
            stdDevs.push_back(stdDev);
            outCombination.setValue(pos, true);
            validNr++;
        }
        else
            outCombination.setValue(pos, false);
    }

    if (validNr == 0) return outCombination;

    // set all proxies unsignificant if not enough points
    if (finalPoints.size() < mySettings->getMinPointsLocalDetrending()) return outCombination;
    {
        for (pos = 0; pos < mySettings->getProxyNr(); pos++)
            mySettings->getProxy(pos)->setIsSignificant(false);

        return outCombination;
    }

    // z-score normalization
    std::vector <float> rowPredictors;
    std::vector <std::vector <float>> predictorsNorm;
    std::vector <float> predictands;
    std::vector <float> weights;
    unsigned index = 0;

    for (i=0; i < finalPoints.size(); i++)
    {
        rowPredictors.clear();
        index = 0;
        for (pos=0; pos < mySettings->getProxyNr(); pos++)
            if (outCombination.getValue(pos))
            {
                proxyValue = finalPoints[i].getProxyValue(pos);
                rowPredictors.push_back((proxyValue - avgs[index]) / stdDevs[index]);
                index++;
            }

        predictorsNorm.push_back(rowPredictors);
        predictands.push_back(finalPoints[i].value);
        weights.push_back(finalPoints[i].regressionWeight);
    }

    if (finalPoints.size() < mySettings->getMinPointsLocalDetrending()) return outCombination;
    {
        for (pos = 0; pos < mySettings->getProxyNr(); pos++)
            mySettings->getProxy(pos)->setIsSignificant(false);

        return outCombination;
    }


    float q;
    std::vector <float> slopes(avgs.size());
    //statistics::weightedMultiRegressionLinear(predictorsNorm, predictands, weights, long(predictorsNorm.size()), &q, slopes, int(predictorsNorm[0].size()));
    /*bestFittingMarquardt_nDimension(double (*func)(std::vector<double>&, std::vector<double>&), int nrTrials, int nrMinima,
                                        std::vector<double>& parametersMin, std::vector<double>& parametersMax,
                                        std::vector<double>& parameters, std::vector<double> &parametersDelta,
                                        int maxIterationsNr, double myEpsilon,
                                        double** x, double* y, int nrData, int xDim, bool isWeighted, double* weights);
    */

    Crit3DProxy* myProxy;
    index=0;
    for (pos = 0; pos < mySettings->getProxyNr(); pos++)
        if (outCombination.getValue(pos))
        {
            myProxy = mySettings->getProxy(pos);
            myProxy->setRegressionSlope(slopes[index]);
            myProxy->setAvg(avgs[index]);
            myProxy->setStdDev(stdDevs[index]);
            mySettings->getProxy(pos)->setIsSignificant(true);
            index++;
        }

    // detrending
    float detrendValue;
    for (i = 0; i < myPoints.size(); i++)
    {
        index = 0;
        for (pos=0; pos < mySettings->getProxyNr(); pos++)
        {
            detrendValue = 0;

            if (outCombination.getValue(pos))
            {
                proxyValue = myPoints[i].getProxyValue(pos);

                if (proxyValue != NODATA)
                    detrendValue = float((proxyValue - avgs[index]) / stdDevs[index] * slopes[index]);

                index++;

                myPoints[i].value -= detrendValue;
            }
        }
    }

    return outCombination;
}

void topographicDistanceOptimize(meteoVariable myVar,
                                 Crit3DMeteoPoint* &myMeteoPoints,
                                 int nrMeteoPoints,
                                 std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                 Crit3DInterpolationSettings* mySettings,
                                 Crit3DMeteoSettings* meteoSettings,
                                 const Crit3DTime &myTime)
{
    float avgError;

    mySettings->initializeKhSeries();

    int kh = 0;
    int bestKh = kh;
    float bestError = NODATA;
    while (kh <= mySettings->getTopoDist_maxKh())
    {
        mySettings->setTopoDist_Kh(kh);
        if (computeResiduals(myVar, myMeteoPoints, nrMeteoPoints, interpolationPoints, mySettings, meteoSettings, true, true))
        {
            avgError = computeErrorCrossValidation(myVar, myMeteoPoints, nrMeteoPoints, myTime, meteoSettings);
            if (isEqual(bestError, NODATA) || avgError < bestError)
            {
                bestError = avgError;
                bestKh = kh;
            }

            mySettings->addToKhSeries(float(kh), avgError);
        }
        kh = ((kh == 0) ? 1 : kh*2);
    }

    mySettings->setTopoDist_Kh(bestKh);
}


void optimalDetrending(meteoVariable myVar, Crit3DMeteoPoint* &myMeteoPoints, int nrMeteoPoints,
                    std::vector <Crit3DInterpolationDataPoint> &outInterpolationPoints,
                    Crit3DInterpolationSettings* mySettings, Crit3DMeteoSettings* meteoSettings, Crit3DClimateParameters* myClimate,
                    const Crit3DTime &myTime)
{

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;
    int i, nrCombination, bestCombinationIndex;
    float avgError, minError;
    size_t proxyNr = mySettings->getProxyNr();
    Crit3DProxyCombination myCombination, bestCombination;
    myCombination = mySettings->getSelectedCombination();

    nrCombination = int(pow(2, double(proxyNr) + 1));

    minError = NODATA;
    bestCombinationIndex = 0;

    for (i=0; i < nrCombination; i++)
    {
        if (mySettings->getCombination(i, myCombination))
        {
            passDataToInterpolation(myMeteoPoints, nrMeteoPoints, interpolationPoints, mySettings);
            detrending(interpolationPoints, myCombination, mySettings, myClimate, myVar, myTime);
            mySettings->setCurrentCombination(myCombination);

            if (mySettings->getUseTD() && getUseTdVar(myVar))
                topographicDistanceOptimize(myVar, myMeteoPoints, nrMeteoPoints, interpolationPoints, mySettings, meteoSettings, myTime);

            if (computeResiduals(myVar, myMeteoPoints, nrMeteoPoints, interpolationPoints, mySettings, meteoSettings, true, true))
            {
                avgError = computeErrorCrossValidation(myVar, myMeteoPoints, nrMeteoPoints, myTime, meteoSettings);
                if (! isEqual(avgError, NODATA) && (isEqual(minError, NODATA) || avgError < minError))
                {
                    minError = avgError;
                    bestCombinationIndex = i;
                }
            }
        }
    }

    if (mySettings->getCombination(bestCombinationIndex, bestCombination))
    {
        passDataToInterpolation(myMeteoPoints, nrMeteoPoints, outInterpolationPoints, mySettings);
        detrending(outInterpolationPoints, bestCombination, mySettings, myClimate, myVar, myTime);
        mySettings->setOptimalCombination(bestCombination);
    }

    return;
}


bool preInterpolation(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings, Crit3DMeteoSettings* meteoSettings,
                      Crit3DClimateParameters* myClimate, Crit3DMeteoPoint* myMeteoPoints, int nrMeteoPoints,
                      meteoVariable myVar, Crit3DTime myTime)
{
    if (myVar == precipitation || myVar == dailyPrecipitation)
    {
        int nrPrecNotNull;
        bool isFlatPrecipitation;
        if (checkPrecipitationZero(myPoints, meteoSettings->getRainfallThreshold(), &nrPrecNotNull, &isFlatPrecipitation))
        {
            mySettings->setPrecipitationAllZero(true);
            return true;
        }
        else
            mySettings->setPrecipitationAllZero(false);
    }

    if (getUseDetrendingVar(myVar))
    {
        if (mySettings->getUseMultipleDetrending())
            mySettings->setCurrentCombination(multipleDetrending(myPoints, mySettings->getSelectedCombination(), mySettings, myVar));
        else
        {
            if (mySettings->getUseBestDetrending())
            {
                optimalDetrending(myVar, myMeteoPoints, nrMeteoPoints, myPoints, mySettings, meteoSettings, myClimate, myTime);
                mySettings->setCurrentCombination(mySettings->getOptimalCombination());
            }
            else
            {
                detrending(myPoints, mySettings->getSelectedCombination(), mySettings, myClimate, myVar, myTime);
                mySettings->setCurrentCombination(mySettings->getSelectedCombination());
            }
        }
    }

    if (mySettings->getUseTD() && getUseTdVar(myVar))
        topographicDistanceOptimize(myVar, myMeteoPoints, nrMeteoPoints, myPoints, mySettings, meteoSettings, myTime);

    return (true);
}


float interpolate(vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings, Crit3DMeteoSettings* meteoSettings,
                  meteoVariable myVar, float myX, float myY, float myZ, std::vector <float> myProxyValues,
                  bool excludeSupplemental)

{
    if ((myVar == precipitation || myVar == dailyPrecipitation) && mySettings->getPrecipitationAllZero())
        return 0.;

    float myResult = NODATA;

    computeDistances(myVar, myPoints, mySettings, myX, myY, myZ, excludeSupplemental);

    if (mySettings->getInterpolationMethod() == idw)
        myResult = inverseDistanceWeighted(myPoints);
    //else if (mySettings->getInterpolationMethod() == kriging)
    //    myResult = NODATA;  //TODO
    else if (mySettings->getInterpolationMethod() == shepard)
        myResult = shepardIdw(myPoints, mySettings, myX, myY);
    else if (mySettings->getInterpolationMethod() == shepard_modified)
    {
        float radius = NODATA;
        if (mySettings->getUseLocalDetrending()) radius = mySettings->getLocalRadius();
        myResult = modifiedShepardIdw(myPoints, mySettings, radius, myX, myY);
    }

    if (int(myResult) != int(NODATA))
        myResult += retrend(myVar, myProxyValues, mySettings);
    else
        return NODATA;

    if (myVar == precipitation || myVar == dailyPrecipitation)
    {
        if (myResult < meteoSettings->getRainfallThreshold())
            return 0.;
    }
    else if (myVar == airRelHumidity || myVar == dailyAirRelHumidityAvg
             || myVar == dailyAirRelHumidityMax || myVar == dailyAirRelHumidityMin)
        myResult = MAXVALUE(MINVALUE(myResult, 100), 0);
    else if (myVar == dailyAirTemperatureRange || myVar == leafWetness || myVar == dailyLeafWetness
             || myVar == globalIrradiance || myVar == dailyGlobalRadiation || myVar == atmTransmissivity
             || myVar == windScalarIntensity || myVar == windVectorIntensity || myVar == dailyWindScalarIntensityAvg || myVar == dailyWindScalarIntensityMax || myVar == dailyWindVectorIntensityAvg || myVar == dailyWindVectorIntensityMax
             || myVar == atmPressure)
        myResult = MAXVALUE(myResult, 0);

    return myResult;

}

void getProxyValuesXY(float x, float y, Crit3DInterpolationSettings* mySettings, std::vector<float> &myValues)
{
    float myValue;
    gis::Crit3DRasterGrid* proxyGrid;

    Crit3DProxyCombination myCombination = mySettings->getCurrentCombination();

    for (unsigned int i=0; i < mySettings->getProxyNr(); i++)
    {
        myValues[i] = NODATA;

        if (myCombination.getValue(i))
        {
            proxyGrid = mySettings->getProxy(i)->getGrid();
            if (proxyGrid != nullptr && proxyGrid->isLoaded)
            {
                myValue = gis::getValueFromXY(*proxyGrid, x, y);
                if (myValue != proxyGrid->header->flag)
                    myValues[i] = myValue;
            }
        }
    }
}

float getFirstIntervalHeightValue(std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode)
{
    float maxPointsZ = getMaxHeight(myPoints, useLapseRateCode);
    float lowerHeight = getZmin(myPoints);
    float higherHeight = lowerHeight;
    float getFirstIntervalHeightValue = NODATA;

    while (getFirstIntervalHeightValue == NODATA && higherHeight < maxPointsZ)
    {
        higherHeight = std::min(higherHeight + 50, maxPointsZ);
        getFirstIntervalHeightValue = findHeightIntervalAvgValue(useLapseRateCode, myPoints,
                                                                 lowerHeight, higherHeight, maxPointsZ);
    }
    return getFirstIntervalHeightValue;
}

