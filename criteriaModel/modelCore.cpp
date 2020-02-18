/*!
    \copyright 2018 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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
    gantolini@arpae.it
*/

#include <iostream>
#include <QSqlQuery>
#include <QSqlError>
#include <QString>
#include <math.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "criteriaModel.h"
#include "croppingSystem.h"
#include "cropDbTools.h"
#include "water1D.h"
#include "modelCore.h"


bool runModel(CriteriaModel* myCase, CriteriaUnit *myUnit, QString *myError)
{
    myCase->idCase = myUnit->idCase;

    if (! myCase->setSoil(myUnit->idSoil, myError))
        return false;

    if (! myCase->loadMeteo(myUnit->idMeteo, myUnit->idForecast, myError))
        return false;

    if (! loadCropParameters(myUnit->idCrop, &(myCase->myCrop), &(myCase->dbCrop), myError))
        return false;

    if (! myCase->isSeasonalForecast)
    {
        if (! myCase->createOutputTable(myError))
            return false;
    }

    // set computation period (all meteo data)
    Crit3DDate firstDate, lastDate;
    long lastIndex = myCase->meteoPoint.nrObsDataDaysD-1;
    firstDate = myCase->meteoPoint.obsDataD[0].date;
    lastDate = myCase->meteoPoint.obsDataD[lastIndex].date;

    if (myCase->isSeasonalForecast)
        myCase->initializeSeasonalForecast(firstDate, lastDate);

    return computeModel(myCase, firstDate, lastDate, myError);
}


bool computeModel(CriteriaModel* myCase, const Crit3DDate& firstDate, const Crit3DDate& lastDate, QString *myError)
{
    Crit3DDate myDate;
    long myIndex;
    int doy;
    float tmin, tmax;                               // [°C]
    float prec, tomorrowPrec;                       // [mm]
    double et0;                                     // [mm]
    double irrigation, irrigationPrec;              // [mm]
    float waterTableDepth;                          // [m]
    bool isFirstDay = true;
    int indexSeasonalForecast = NODATA;
    bool isInsideSeason;

    if (int(myCase->meteoPoint.latitude) == int(NODATA))
    {
        *myError = "Latitude is missing";
        return false;
    }

    initializeWater(myCase);

    initializeCrop(myCase, getDoyFromDate(firstDate));

    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        // Initialize
        myCase->output.initializeDaily();
        doy = getDoyFromDate(myDate);

        // daily meteo
        myIndex = myCase->meteoPoint.obsDataD[0].date.daysTo(myDate);
        if ((myIndex < 0) || (myIndex >= myCase->meteoPoint.nrObsDataDaysD))
        {
            *myError = "Missing weather data: " + QString::fromStdString(myDate.toStdString());
            return false;
        }

        prec = myCase->meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation);
        tmin = myCase->meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = myCase->meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (int(prec) == int(NODATA) || int(tmin) == int(NODATA) || int(tmax) == int(NODATA))
        {
            *myError = "Missing weather data: " + QString::fromStdString(myDate.toStdString());
            return false;
        }

        // check on wrong data
        if (prec < 0) prec = 0;
        myCase->output.dailyPrec = double(prec);

        // WATERTABLE
        waterTableDepth = myCase->meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth);

        myCase->output.dailyWaterTable = double(waterTableDepth);
        if (myDate < lastDate)
            tomorrowPrec = myCase->meteoPoint.getMeteoPointValueD(myDate.addDays(1), dailyPrecipitation);
        else
            tomorrowPrec = 0;

        // ET0
        et0 = double(myCase->meteoPoint.getMeteoPointValueD(myDate, dailyReferenceEvapotranspirationHS));
        if (isEqual(et0, NODATA) || et0 <= 0)
            et0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, myCase->meteoPoint.latitude, doy, double(tmax), double(tmin));

        myCase->output.dailyEt0 = et0;

        // CROP
        if (! updateCrop(myCase, myDate, tmin, tmax, double(waterTableDepth), myError))
            return false;

        // Evaporation / transpiration
        myCase->output.dailyMaxEvaporation = myCase->myCrop.getMaxEvaporation(myCase->output.dailyEt0);
        myCase->output.dailyMaxTranspiration = myCase->myCrop.getMaxTranspiration(myCase->output.dailyEt0);

        // WATERTABLE (if available)
        computeCapillaryRise(myCase, double(waterTableDepth));

        // IRRIGATION
        irrigation = myCase->myCrop.getIrrigationDemand(doy, prec, tomorrowPrec, myCase->output.dailyMaxTranspiration, myCase->layers);
        if (myCase->optimizeIrrigation)
            irrigation = MINVALUE(myCase->myCrop.getCropWaterDeficit(myCase->layers), irrigation);

        myCase->output.dailyIrrigation = irrigation;
        irrigationPrec = 0.0;

        if (irrigation > 0)
        {
            if (myCase->optimizeIrrigation)
                optimalIrrigation(myCase, irrigation);
            else
                irrigationPrec = irrigation;
        }

        // INFILTRATION
        if (! computeInfiltration(myCase, prec, irrigationPrec))
            return false;

        // LATERAL DRAINAGE
        if (! computeLateralDrainage(myCase))
            return false;

        // EVAPORATION
        if (! computeEvaporation(myCase))
            return false;

        // RUNOFF (after evaporation)
        if (! computeSurfaceRunoff(myCase))
            return false;

        // Adjust irrigation losses
        if (! myCase->optimizeIrrigation)
        {
            if ((myCase->output.dailySurfaceRunoff > 1) && (myCase->output.dailyIrrigation > 0))
            {
                myCase->output.dailyIrrigation -= floor(myCase->output.dailySurfaceRunoff);
                myCase->output.dailySurfaceRunoff -= floor(myCase->output.dailySurfaceRunoff);
            }
        }

        // TRANSPIRATION
        std::vector<double> layerTranspiration;
        myCase->output.dailyTranspiration = myCase->myCrop.computeTranspiration(myCase->output.dailyMaxTranspiration,
                                                                                myCase->layers, &layerTranspiration, false);
        // assign transpiration
        if (myCase->output.dailyTranspiration > 0)
        {
            for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
            {
                myCase->layers[i].waterContent -= layerTranspiration[i];
            }
        }

        // Output variables
        myCase->output.dailySurfaceWaterContent = myCase->layers[0].waterContent;
        myCase->output.dailySoilWaterContent = getSoilWaterContent(myCase);
        myCase->output.dailyCropAvailableWater = getCropReadilyAvailableWater(myCase);
        myCase->output.dailyWaterDeficit = getSoilWaterDeficit(myCase);

        if (! myCase->isSeasonalForecast)
        {
            myCase->prepareOutput(myDate, isFirstDay);
            isFirstDay = false;
        }

        // seasonal forecast: update values of annual irrigation
        if (myCase->isSeasonalForecast)
        {
            isInsideSeason = false;
            // normal seasons
            if (myCase->firstSeasonMonth < 11)
            {
                if (myDate.month >= myCase->firstSeasonMonth && myDate.month <= myCase->firstSeasonMonth+2)
                    isInsideSeason = true;
            }
            // NDJ or DJF
            else
            {
                int lastMonth = (myCase->firstSeasonMonth + 2) % 12;
                if (myDate.month >= myCase->firstSeasonMonth || myDate.month <= lastMonth)
                   isInsideSeason = true;
            }

            if (isInsideSeason)
            {
                // first date of season
                if (myDate.day == 1 && myDate.month == myCase->firstSeasonMonth)
                {
                    if (indexSeasonalForecast == NODATA)
                        indexSeasonalForecast = 0;
                    else
                        indexSeasonalForecast++;
                }

                // sum of irrigations
                if (indexSeasonalForecast != NODATA)
                {
                    if (int(myCase->seasonalForecasts[indexSeasonalForecast]) == int(NODATA))
                        myCase->seasonalForecasts[indexSeasonalForecast] = myCase->output.dailyIrrigation;
                    else
                        myCase->seasonalForecasts[indexSeasonalForecast] += myCase->output.dailyIrrigation;
                }
            }
        }
    }

    if (myCase->isSeasonalForecast)
        return true;
    else
        return myCase->saveOutput(myError);
}

