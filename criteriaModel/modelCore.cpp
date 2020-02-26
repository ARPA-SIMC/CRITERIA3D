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
    Crit3DDate myDate, firstDate, lastDate;
    long lastIndex = myCase->meteoPoint.nrObsDataDaysD-1;
    firstDate = myCase->meteoPoint.obsDataD[0].date;
    lastDate = myCase->meteoPoint.obsDataD[lastIndex].date;

    if (myCase->isSeasonalForecast)
        myCase->initializeSeasonalForecast(firstDate, lastDate);

    return computeModel(myCase, firstDate, lastDate, myError);


    /*
     * // initialize crop and soil water
    initializeWater(myCase);
    myCase->myCrop.initialize(myCase->meteoPoint.latitude, myCase->nrLayers, myCase->mySoil.totalDepth, getDoyFromDate(firstDate));

    std::string errorString;
    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! computeDailyModel(myDate, &(myCase->meteoPoint), &(myCase->myCrop), &(myCase->soilLayers), &(myCase->output), &errorString))
        {
            *myError = QString::fromStdString(errorString);
            return false;
        }
    }

    if (! myCase->isSeasonalForecast)
        if(! myCase->saveOutput(myError))
            return false;

    return true;
    */
}


bool computeModel(CriteriaModel* myCase, const Crit3DDate& firstDate, const Crit3DDate& lastDate, QString *myError)
{
    Crit3DDate myDate;
    int doy;
    float tmin, tmax;                               // [°C]
    double prec, tomorrowPrec;                      // [mm]
    double et0;                                     // [mm]
    double irrigation, irrigationPrec;              // [mm]
    double waterTableDepth;                         // [m]
    bool isFirstDay = true;
    int indexSeasonalForecast = NODATA;
    bool isInsideSeason;

    if (int(myCase->meteoPoint.latitude) == int(NODATA))
    {
        *myError = "Latitude is missing";
        return false;
    }

    initializeWater(myCase);

    myCase->myCrop.initialize(myCase->meteoPoint.latitude, myCase->nrLayers, myCase->mySoil.totalDepth, getDoyFromDate(firstDate));

    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        // Initialize
        myCase->output.initializeDaily();
        doy = getDoyFromDate(myDate);

        // check daily meteo data
        if (! myCase->meteoPoint.existDailyData(myDate))
        {
            *myError = "Missing weather data: " + QString::fromStdString(myDate.toStdString());
            return false;
        }

        prec = double(myCase->meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
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
        waterTableDepth = double(myCase->meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));

        myCase->output.dailyWaterTable = double(waterTableDepth);
        if (myDate < lastDate)
            tomorrowPrec = double(myCase->meteoPoint.getMeteoPointValueD(myDate.addDays(1), dailyPrecipitation));
        else
            tomorrowPrec = 0;

        // ET0
        et0 = double(myCase->meteoPoint.getMeteoPointValueD(myDate, dailyReferenceEvapotranspirationHS));
        if (isEqual(et0, NODATA) || et0 <= 0)
            et0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, myCase->meteoPoint.latitude, doy, double(tmax), double(tmin));

        myCase->output.dailyEt0 = et0;

        // CROP
        if (! updateCrop(myCase, myDate, tmin, tmax, waterTableDepth, myError))
            return false;

        // Evaporation / transpiration
        myCase->output.dailyMaxEvaporation = myCase->myCrop.getMaxEvaporation(myCase->output.dailyEt0);
        myCase->output.dailyMaxTranspiration = myCase->myCrop.getMaxTranspiration(myCase->output.dailyEt0);

        // WATERTABLE (if available)
        myCase->output.dailyCapillaryRise = computeCapillaryRise(&(myCase->soilLayers), waterTableDepth);

        // IRRIGATION
        irrigation = myCase->myCrop.getIrrigationDemand(doy, prec, tomorrowPrec, myCase->output.dailyMaxTranspiration, myCase->soilLayers);
        if (myCase->optimizeIrrigation)
            irrigation = MINVALUE(myCase->myCrop.getCropWaterDeficit(myCase->soilLayers), irrigation);

        // assign irrigation (optimal or equal to precipitation)
        if (irrigation > 0)
        {
            if (myCase->optimizeIrrigation)
            {
                myCase->output.dailyIrrigation = computeOptimalIrrigation(&(myCase->soilLayers), irrigation);
                irrigationPrec = 0;
            }
            else
            {
                irrigationPrec = irrigation;
                myCase->output.dailyIrrigation = irrigation;
            }
        }
        else
        {
            myCase->output.dailyIrrigation = 0;
            irrigationPrec = 0;
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
        double waterStress;
        myCase->output.dailyTranspiration = myCase->myCrop.computeTranspiration(myCase->output.dailyMaxTranspiration, myCase->soilLayers, &waterStress);

        // assign transpiration
        if (myCase->output.dailyTranspiration > 0)
        {
            for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
            {
                myCase->soilLayers[i].waterContent -= myCase->myCrop.layerTranspiration[i];
            }
        }

        // Output variables
        myCase->output.dailySurfaceWaterContent = myCase->soilLayers[0].waterContent;
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


bool computeDailyModel(Crit3DDate myDate, Crit3DMeteoPoint* meteoPoint, Crit3DCrop* myCrop,
                       std::vector<soil::Crit3DLayer>* soilLayers, CriteriaModelOutput* myOutput,
                       std::string *myError)
{
    bool optimizeIrrigation = false;

    // Initialize output
    myOutput->initializeDaily();
    int doy = getDoyFromDate(myDate);

    // check daily meteo data
    if (! meteoPoint->existDailyData(myDate))
    {
        *myError = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    double prec = double(meteoPoint->getMeteoPointValueD(myDate, dailyPrecipitation));
    double tmin = double(meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin));
    double tmax = double(meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax));

    if (isEqual(prec, NODATA) || isEqual(tmin, NODATA) || isEqual(tmax, NODATA))
    {
        *myError = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    // check on wrong data
    if (prec < 0) prec = 0;
    myOutput->dailyPrec = prec;

    // water table
    myOutput->dailyWaterTable = double(meteoPoint->getMeteoPointValueD(myDate, dailyWaterTableDepth));

    // prec forecast
    double precTomorrow = double(meteoPoint->getMeteoPointValueD(myDate.addDays(1), dailyPrecipitation));
    if (isEqual(precTomorrow, NODATA)) precTomorrow = 0;

    // ET0
    myOutput->dailyEt0 = double(meteoPoint->getMeteoPointValueD(myDate, dailyReferenceEvapotranspirationHS));
    if (isEqual(myOutput->dailyEt0, NODATA) || myOutput->dailyEt0 <= 0)
        myOutput->dailyEt0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, meteoPoint->latitude, doy, tmax, tmin);

    // update LAI and root depth
    if (! myCrop->dailyUpdate(myDate, meteoPoint->latitude, *soilLayers, tmin, tmax, myOutput->dailyWaterTable, myError))
        return false;

    // Evaporation / transpiration
    myOutput->dailyMaxEvaporation = myCrop->getMaxEvaporation(myOutput->dailyEt0);
    myOutput->dailyMaxTranspiration = myCrop->getMaxTranspiration(myOutput->dailyEt0);

    // WATERTABLE (if available)
    myOutput->dailyCapillaryRise = computeCapillaryRise(soilLayers, myOutput->dailyWaterTable);

    // IRRIGATION
    double irrigation = myCrop->getIrrigationDemand(doy, prec, precTomorrow, myOutput->dailyMaxTranspiration, *soilLayers);
    if (optimizeIrrigation) irrigation = MINVALUE(irrigation, myCrop->getCropWaterDeficit(*soilLayers));

    // assign irrigation
    double sprayIrrigation;
    if (irrigation > 0)
    {
        if (optimizeIrrigation)
        {
            myOutput->dailyIrrigation = computeOptimalIrrigation(soilLayers, irrigation);
            sprayIrrigation = 0;
        }
        else
        {
            myOutput->dailyIrrigation = irrigation;
            sprayIrrigation = irrigation;
        }
    }

    // INFILTRATION
    /*if (! computeInfiltration(myCase, prec, sprayIrrigation))
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
        */

    // Adjust irrigation losses
    if (! optimizeIrrigation)
    {
        if ((myOutput->dailySurfaceRunoff > 1) && (myOutput->dailyIrrigation > 0))
        {
            myOutput->dailyIrrigation -= floor(myOutput->dailySurfaceRunoff);
            myOutput->dailySurfaceRunoff -= floor(myOutput->dailySurfaceRunoff);
        }
    }

    // TRANSPIRATION
    double waterStress;
    myOutput->dailyTranspiration = myCrop->computeTranspiration(myOutput->dailyMaxTranspiration, *soilLayers, &waterStress);

    // assign transpiration
    if (myOutput->dailyTranspiration > 0)
    {
        for (unsigned int i = unsigned(myCrop->roots.firstRootLayer); i <= unsigned(myCrop->roots.lastRootLayer); i++)
        {
            (*soilLayers)[i].waterContent -= myCrop->layerTranspiration[i];
        }
    }

    return true;
}


bool updateCrop(CriteriaModel* myCase, Crit3DDate myDate, float tmin, float tmax, double waterTableDepth, QString *myError)
{
    std::string errorString;

    if ( !myCase->myCrop.dailyUpdate(myDate, myCase->meteoPoint.latitude, myCase->soilLayers, tmin, tmax, waterTableDepth, &errorString))
    {
        *myError = QString::fromStdString(errorString);
        return false;
    }

    return true;
}


/*!
 * \brief getCropReadilyAvailableWater
 * \return sum of readily available water (mm) in the rooting zone
 */
double getCropReadilyAvailableWater(CriteriaModel* myCase)
{
    if (! myCase->myCrop.isLiving) return 0.;
    if (myCase->myCrop.roots.rootDepth <= myCase->myCrop.roots.rootDepthMin) return 0.;
    if (myCase->myCrop.roots.firstRootLayer == NODATA) return 0.;

    double sumRAW = 0.0;
    for (unsigned int i = unsigned(myCase->myCrop.roots.firstRootLayer); i <= unsigned(myCase->myCrop.roots.lastRootLayer); i++)
    {
        double thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(myCase->myCrop.psiLeaf), myCase->soilLayers[i].horizon);
        // [mm]
        double cropWP = thetaWP * myCase->soilLayers[i].thickness * myCase->soilLayers[i].soilFraction * 1000.0;
        // [mm]
        double threshold = myCase->soilLayers[i].FC - myCase->myCrop.fRAW * (myCase->soilLayers[i].FC - cropWP);

        double layerRAW = (myCase->soilLayers[i].waterContent - threshold);

        double layerMaxDepth = myCase->soilLayers[i].depth + myCase->soilLayers[i].thickness / 2.0;
        if (myCase->myCrop.roots.rootDepth < layerMaxDepth)
        {
                layerRAW *= (myCase->myCrop.roots.rootDepth - layerMaxDepth) / myCase->soilLayers[i].thickness;
        }

        sumRAW += layerRAW;
    }

    return sumRAW;
}


/*!
 * \brief getSoilWaterDeficit
 * \param myCase
 * \return sum of water deficit (mm) in the first meter of soil
 */
double getSoilWaterDeficit(CriteriaModel* myCase)
{
    // surface water content
    double waterDeficit = -myCase->soilLayers[0].waterContent;

    for (unsigned int i = 1; i <= myCase->nrLayers; i++)
    {
        if (myCase->soilLayers[i].depth > 1)
            return waterDeficit;

        waterDeficit += myCase->soilLayers[unsigned(i)].FC - myCase->soilLayers[unsigned(i)].waterContent;
    }

    return waterDeficit;
}


