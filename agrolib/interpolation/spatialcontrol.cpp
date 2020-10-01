#include <stdio.h>
#include <cmath>

#include "commonConstants.h"
#include "spatialControl.h"
#include "interpolation.h"
#include "statistics.h"

float findThreshold(meteoVariable myVar, float value, float stdDev, float nrStdDev, float stdDevZ, float minDistance)
{
    float zWeight, distWeight, threshold;

    if (   myVar == precipitation
        || myVar == dailyPrecipitation)
    {
        distWeight = MAXVALUE(1.f, minDistance / 2000.f);
        if (value < float(PREC_THRESHOLD))
            threshold = MAXVALUE(5.f, distWeight + stdDev * (nrStdDev + 1));
        else
            return 800.f;
    }
    else if (   myVar == airTemperature
             || myVar == airDewTemperature
             || myVar == dailyAirTemperatureMax
             || myVar == dailyAirTemperatureMin
             || myVar == dailyAirTemperatureAvg )
    {
        threshold = 1.f;
        zWeight = stdDevZ / 100.f;
        distWeight = minDistance / 5000.f;

        threshold = MINVALUE(MINVALUE(distWeight + threshold + zWeight, 12.f) + stdDev * nrStdDev, 15.f);
    }
    else if (   myVar == airRelHumidity
             || myVar == dailyAirRelHumidityMax
             || myVar == dailyAirRelHumidityMin
             || myVar == dailyAirRelHumidityAvg )
    {
        threshold = 8.f;
        zWeight = stdDevZ / 100.f;
        distWeight = minDistance / 1000.f;
        threshold += zWeight + distWeight + stdDev * nrStdDev;
    }
    else if (   myVar == windScalarIntensity
             || myVar == windVectorIntensity
             || myVar == dailyWindScalarIntensityAvg
             || myVar == dailyWindScalarIntensityMax
             || myVar == dailyWindVectorIntensityAvg
             || myVar == dailyWindVectorIntensityMax)
    {
        threshold = 1.f;
        zWeight = stdDevZ / 50.f;
        distWeight = minDistance / 2000.f;
        threshold += zWeight + distWeight + stdDev * nrStdDev;
    }
    else if (   myVar == globalIrradiance
             || myVar == dailyGlobalRadiation )
        threshold = MAXVALUE(stdDev * (nrStdDev + 1.f), 5.f);

    else if (myVar == atmTransmissivity)
        threshold = MAXVALUE(stdDev * nrStdDev, 0.5f);
    else
        threshold = stdDev * nrStdDev;

    return threshold;
}


bool computeResiduals(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                      std::vector <Crit3DInterpolationDataPoint> &interpolationPoints, Crit3DInterpolationSettings* settings,
                      bool excludeOutsideDem, bool excludeSupplemental)
{

    if (myVar == noMeteoVar) return false;

    float myValue, interpolatedValue;
    interpolatedValue = NODATA;
    myValue = NODATA;
    std::vector <float> myProxyValues;
    bool isValid;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        myProxyValues = meteoPoints[i].getProxyValues();

        meteoPoints[i].residual = NODATA;

        isValid = (! excludeSupplemental || checkLapseRateCode(meteoPoints[i].lapseRateCode, settings->getUseLapseRateCode(), false));
        isValid = (isValid && (! excludeOutsideDem || meteoPoints[i].isInsideDem));

        if (isValid && meteoPoints[i].quality == quality::accepted)
        {
            myValue = meteoPoints[i].currentValue;

            interpolatedValue = interpolate(interpolationPoints, settings, myVar, float(meteoPoints[i].point.utm.x),
                                            float(meteoPoints[i].point.utm.y),
                                            float(meteoPoints[i].point.z),
                                            myProxyValues,
                                            false);

            if (  myVar == precipitation
               || myVar == dailyPrecipitation)
            {
                if (myValue != NODATA)
                    if (myValue < PREC_THRESHOLD) myValue=0.;

                if (interpolatedValue != NODATA)
                    if (interpolatedValue < PREC_THRESHOLD) interpolatedValue=0.;
            }

            // TODO derived var

            if ((interpolatedValue != NODATA) && (myValue != NODATA))
                meteoPoints[i].residual = interpolatedValue - myValue;
        }
    }

    return true;
}

float computeErrorCrossValidation(meteoVariable myVar, Crit3DMeteoPoint* myPoints, int nrMeteoPoints, const Crit3DTime& myTime)
{
    std::vector <float> obsValues, estValues;
    float myValue, myEstimate, myResidual;

    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (myPoints[i].active)
        {
            myValue = myPoints[i].getMeteoPointValue(myTime, myVar);
            myResidual = myPoints[i].residual;

            if (myValue != NODATA && myResidual != NODATA)
            {
                myEstimate = myValue + myResidual;
                obsValues.push_back(myValue);
                estValues.push_back(myEstimate);
            }
        }
    }

    if (obsValues.size() > 0)
        return statistics::meanAbsoluteError(obsValues, estValues);
    else return NODATA;
}

void spatialQualityControl(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                           Crit3DInterpolationSettings *settings, Crit3DClimateParameters* myClimate, Crit3DTime myTime)
{
    int i;
    float stdDev, stdDevZ, minDist, myValue, myResidual;
    std::vector <int> listIndex;
    std::vector <float> listResiduals;
    std::vector <Crit3DInterpolationDataPoint> myInterpolationPoints;

    if (passDataToInterpolation(meteoPoints, nrMeteoPoints, myInterpolationPoints, settings))
    {
        // detrend
        if (! preInterpolation(myInterpolationPoints, settings, myClimate, meteoPoints, nrMeteoPoints, myVar, myTime))
            return;

        // compute residuals
        if (! computeResiduals(myVar, meteoPoints, nrMeteoPoints, myInterpolationPoints, settings, false, false))
            return;

        for (i = 0; i < nrMeteoPoints; i++)
            if (meteoPoints[i].quality == quality::accepted)
            {
                if (neighbourhoodVariability(myVar, myInterpolationPoints, settings, float(meteoPoints[i].point.utm.x),
                         float(meteoPoints[i].point.utm.y),float(meteoPoints[i].point.z),
                         10, &stdDev, &stdDevZ, &minDist))
                {
                    myValue = meteoPoints[i].currentValue;
                    myResidual = meteoPoints[i].residual;
                    stdDev = MAXVALUE(stdDev, myValue/100.f);
                    if (fabs(myResidual) > findThreshold(myVar, myValue, stdDev, 2, stdDevZ, minDist))
                    {
                        listIndex.push_back(i);
                        meteoPoints[i].quality = quality::wrong_spatial;
                    }
                }
            }

        if (listIndex.size() > 0)
        {
            if (passDataToInterpolation(meteoPoints, nrMeteoPoints, myInterpolationPoints, settings))
            {
                preInterpolation(myInterpolationPoints, settings, myClimate, meteoPoints, nrMeteoPoints, myVar, myTime);

                float interpolatedValue;
                for (i=0; i < int(listIndex.size()); i++)
                {
                    interpolatedValue = interpolate(myInterpolationPoints, settings, myVar,
                                            float(meteoPoints[listIndex[i]].point.utm.x),
                                            float(meteoPoints[listIndex[i]].point.utm.y),
                                            float(meteoPoints[listIndex[i]].point.z),
                                            meteoPoints[listIndex[i]].getProxyValues(), false);

                    myValue = meteoPoints[listIndex[i]].currentValue;

                    listResiduals.push_back(interpolatedValue - myValue);
                }

                for (i=0; i < int(listIndex.size()); i++)
                {
                    if (neighbourhoodVariability(myVar, myInterpolationPoints, settings, float(meteoPoints[listIndex[i]].point.utm.x),
                             float(meteoPoints[listIndex[i]].point.utm.y),
                             float(meteoPoints[listIndex[i]].point.z),
                             10, &stdDev, &stdDevZ, &minDist))
                    {
                        myResidual = listResiduals[i];

                        myValue = meteoPoints[listIndex[i]].currentValue;

                        if (fabs(myResidual) > findThreshold(myVar, myValue, stdDev, 3, stdDevZ, minDist))
                            meteoPoints[listIndex[i]].quality = quality::wrong_spatial;
                        else
                            meteoPoints[listIndex[i]].quality = quality::accepted;
                    }
                    else
                        meteoPoints[listIndex[i]].quality = quality::accepted;
                }
            }
        }
    }
}

bool checkData(Crit3DQuality* myQuality, meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                              Crit3DTime myTime, Crit3DInterpolationSettings* spatialQualityInterpolationSettings, Crit3DClimateParameters* myClimate, bool checkSpatial)
{
    if (nrMeteoPoints == 0)
        return false;

    // assign data
    for (int i = 0; i < nrMeteoPoints; i++)
        meteoPoints[i].currentValue = meteoPoints[i].getMeteoPointValue(myTime, myVar);

    // quality control - syntactic
    myQuality->syntacticQualityControl(myVar, meteoPoints, nrMeteoPoints);

    // quality control - spatial
    if (checkSpatial && myVar != precipitation && myVar != dailyPrecipitation)
    {
        spatialQualityControl(myVar, meteoPoints, nrMeteoPoints, spatialQualityInterpolationSettings, myClimate, myTime);
    }

    return true;
}

// check quality and pass good data to interpolation
bool checkAndPassDataToInterpolation(Crit3DQuality* myQuality, meteoVariable myVar,
                                      Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                                      Crit3DTime myTime, Crit3DInterpolationSettings* SQinterpolationSettings,
                                      Crit3DInterpolationSettings* interpolationSettings,
                                      Crit3DClimateParameters* myClimate,
                                      std::vector <Crit3DInterpolationDataPoint> &myInterpolationPoints,
                                      bool checkSpatial)
{
    if (! checkData(myQuality, myVar, meteoPoints, nrMeteoPoints, myTime, SQinterpolationSettings, myClimate, checkSpatial)) return false;

    // return true if at least one valid data
    return passDataToInterpolation(meteoPoints, nrMeteoPoints, myInterpolationPoints, interpolationSettings);
}

bool passDataToInterpolation(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                            std::vector <Crit3DInterpolationDataPoint> &myInterpolationPoints,
                            Crit3DInterpolationSettings* mySettings)
{
    int nrValid = 0;
    float xMin=NODATA, xMax, yMin, yMax;

    myInterpolationPoints.clear();

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active && meteoPoints[i].quality == quality::accepted)
        {
            Crit3DInterpolationDataPoint myPoint;

            myPoint.index = i;
            myPoint.value = meteoPoints[i].currentValue;
            myPoint.point->utm.x = meteoPoints[i].point.utm.x;
            myPoint.point->utm.y = meteoPoints[i].point.utm.y;
            myPoint.point->z = meteoPoints[i].point.z;
            myPoint.lapseRateCode = meteoPoints[i].lapseRateCode;
            myPoint.proxyValues = meteoPoints[i].proxyValues;
            myPoint.topographicDistance = meteoPoints[i].topographicDistance;
            myPoint.isActive = true;

            if (int(xMin) == int(NODATA))
            {
                xMin = float(myPoint.point->utm.x);
                xMax = float(myPoint.point->utm.x);
                yMin = float(myPoint.point->utm.y);
                yMax = float(myPoint.point->utm.y);
            }
            else
            {
                xMin = MINVALUE(xMin, (float)myPoint.point->utm.x);
                xMax = MAXVALUE(xMax, (float)myPoint.point->utm.x);
                yMin = MINVALUE(yMin, (float)myPoint.point->utm.y);
                yMax = MAXVALUE(yMax, (float)myPoint.point->utm.y);
            }

            myInterpolationPoints.push_back(myPoint);

            if (checkLapseRateCode(myPoint.lapseRateCode, mySettings->getUseLapseRateCode(), false))
                nrValid++;
        }
    }

    if (nrValid > 0)
    {
        mySettings->computeShepardInitialRadius((xMax - xMin)*(yMax-yMin), nrValid);
        return true;
    }
    else
        return false;
}
