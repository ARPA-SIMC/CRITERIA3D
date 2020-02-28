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

#include <QString>
#include <math.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "criteriaModel.h"
#include "cropDbTools.h"
#include "water1D.h"
#include "modelCore.h"


bool runModel(Crit1DIrrigationForecast *irrForecast, const Crit1DUnit& myUnit, QString *myError)
{
    irrForecast->myCase.idCase = myUnit.idCase;

    if (! irrForecast->setSoil(myUnit.idSoil, myError))
        return false;

    if (! irrForecast->loadMeteo(myUnit.idMeteo, myUnit.idForecast, myError))
        return false;

    if (! loadCropParameters(myUnit.idCrop, &(irrForecast->myCase.myCrop), &(irrForecast->dbCrop), myError))
        return false;

    if (! irrForecast->isSeasonalForecast)
    {
        if (! irrForecast->createOutputTable(myError))
            return false;
    }

    // set computation period (all meteo data)
    Crit3DDate myDate, firstDate, lastDate;
    long lastIndex = irrForecast->myCase.meteoPoint.nrObsDataDaysD-1;
    firstDate = irrForecast->myCase.meteoPoint.obsDataD[0].date;
    lastDate = irrForecast->myCase.meteoPoint.obsDataD[lastIndex].date;

    if (irrForecast->isSeasonalForecast)
        irrForecast->initializeSeasonalForecast(firstDate, lastDate);

    //return computeModel(myCase, firstDate, lastDate, myError);

    // initialize soil moisture
    initializeWater(&(irrForecast->myCase.soilLayers));

    // initialize crop
    unsigned int nrLayers = irrForecast->myCase.soilLayers.size();
    irrForecast->myCase.myCrop.initialize(irrForecast->myCase.meteoPoint.latitude, nrLayers,
                                          irrForecast->myCase.mySoil.totalDepth, getDoyFromDate(firstDate));

    std::string errorString;
    bool isFirstDay = true;
    int indexSeasonalForecast = NODATA;
    for (myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! computeDailyModel(myDate, &(irrForecast->myCase), &errorString))
        {
            *myError = QString::fromStdString(errorString);
            return false;
        }

        // output
        if (! irrForecast->isSeasonalForecast)
        {
            irrForecast->prepareOutput(myDate, isFirstDay);
            isFirstDay = false;
        }

        // seasonal forecast: update values of annual irrigation
        if (irrForecast->isSeasonalForecast)
        {
            bool isInsideSeason = false;
            // normal seasons
            if (irrForecast->firstSeasonMonth < 11)
            {
                if (myDate.month >= irrForecast->firstSeasonMonth && myDate.month <= irrForecast->firstSeasonMonth+2)
                    isInsideSeason = true;
            }
            // NDJ or DJF
            else
            {
                int lastMonth = (irrForecast->firstSeasonMonth + 2) % 12;
                if (myDate.month >= irrForecast->firstSeasonMonth || myDate.month <= lastMonth)
                   isInsideSeason = true;
            }

            if (isInsideSeason)
            {
                // first date of season
                if (myDate.day == 1 && myDate.month == irrForecast->firstSeasonMonth)
                {
                    if (indexSeasonalForecast == NODATA)
                        indexSeasonalForecast = 0;
                    else
                        indexSeasonalForecast++;
                }

                // sum of irrigations
                if (indexSeasonalForecast != NODATA)
                {
                    if (int(irrForecast->seasonalForecasts[indexSeasonalForecast]) == int(NODATA))
                        irrForecast->seasonalForecasts[indexSeasonalForecast] = irrForecast->myCase.output.dailyIrrigation;
                    else
                        irrForecast->seasonalForecasts[indexSeasonalForecast] += irrForecast->myCase.output.dailyIrrigation;
                }
            }
        }
    }

    if (irrForecast->isSeasonalForecast)
        return true;
    else
        return irrForecast->saveOutput(myError);

}


bool computeDailyModel(Crit3DDate myDate, Crit1DCase* myCase, std::string* myError)
{
    return computeDailyModel(myDate, &(myCase->meteoPoint), &(myCase->myCrop), &(myCase->soilLayers),
                             &(myCase->output), myCase->optimizeIrrigation, myError);
}


bool computeDailyModel(Crit3DDate myDate, Crit3DMeteoPoint* meteoPoint, Crit3DCrop* myCrop,
                       std::vector<soil::Crit3DLayer>* soilLayers, Crit1DOutput* myOutput,
                       bool optimizeIrrigation, std::string *myError)
{
    double ploughedSoilDepth = 0.5;     /*!< [m] depth of ploughed soil (working layer) */

    // Initialize output
    myOutput->initialize();
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

    // assign irrigation: optimal (subirrigation) or add to precipitation (sprinkler/drop)
    double waterInput = prec;
    myOutput->dailyIrrigation = 0;

    if (irrigation > 0)
    {
        if (optimizeIrrigation)
        {
            myOutput->dailyIrrigation = computeOptimalIrrigation(soilLayers, irrigation);
        }
        else
        {
            myOutput->dailyIrrigation = irrigation;
            waterInput += irrigation;
        }
    }

    // INFILTRATION
    myOutput->dailyDrainage = computeInfiltration(soilLayers, waterInput, ploughedSoilDepth);

    // LATERAL DRAINAGE
    myOutput->dailyLateralDrainage = computeLateralDrainage(soilLayers);

    // EVAPORATION
    myOutput->dailyEvaporation = computeEvaporation(soilLayers, myOutput->dailyMaxEvaporation);

    // RUNOFF (after evaporation)
    myOutput->dailySurfaceRunoff = computeSurfaceRunoff(*myCrop, soilLayers);

    // adjust irrigation losses
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

    // output variables
    myOutput->dailySurfaceWaterContent = (*soilLayers)[0].waterContent;
    myOutput->dailySoilWaterContent = getSoilWaterContent(*soilLayers);
    myOutput->dailyWaterDeficit = getSoilWaterDeficit(*soilLayers);
    myOutput->dailyCropAvailableWater = getCropReadilyAvailableWater(*myCrop, *soilLayers);

    return true;
}


