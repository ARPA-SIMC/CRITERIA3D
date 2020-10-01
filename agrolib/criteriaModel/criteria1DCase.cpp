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
#include "cropDbTools.h"
#include "water1D.h"
#include "criteria1DCase.h"


Crit1DOutput::Crit1DOutput()
{
    this->initialize();
}

void Crit1DOutput::initialize()
{
    this->dailyPrec = NODATA;
    this->dailyDrainage = NODATA;
    this->dailySurfaceRunoff = NODATA;
    this->dailyLateralDrainage = NODATA;
    this->dailyIrrigation = NODATA;
    this->dailySoilWaterContent = NODATA;
    this->dailySurfaceWaterContent = NODATA;
    this->dailyEt0 = NODATA;
    this->dailyEvaporation = NODATA;
    this->dailyMaxTranspiration = NODATA;
    this->dailyMaxEvaporation = NODATA;
    this->dailyTranspiration = NODATA;
    this->dailyAvailableWater = NODATA;
    this->dailyFractionAW = NODATA;
    this->dailyReadilyAW = NODATA;
    this->dailyWaterDeficit = NODATA;
    this->dailyWaterDeficit_25 = NODATA;
    this->dailyCapillaryRise = NODATA;
    this->dailyWaterTable = NODATA;
}


Crit1DCase::Crit1DCase()
{
    idCase = "";

    minLayerThickness = 0.02;           /*!< [m] default thickness = 2 cm  */
    geometricFactor = 1.2;              /*!< [-] default factor for geometric progression  */
    isGeometricLayers = false;
    optimizeIrrigation = false;

    soilLayers.clear();
}


bool Crit1DCase::initializeSoil(std::string &myError)
{
    soilLayers.clear();

    double factor = 1.0;
    if (isGeometricLayers) factor = geometricFactor;

    if (! mySoil.setSoilLayers(minLayerThickness, factor, soilLayers, myError))
        return false;

    initializeWater(soilLayers);

    return true;
}


bool Crit1DCase::computeDailyModel(Crit3DDate myDate, std::string &myError)
{
    return dailyModel(myDate, meteoPoint, myCrop, soilLayers, output, optimizeIrrigation, myError);
}


bool dailyModel(Crit3DDate myDate, Crit3DMeteoPoint &meteoPoint, Crit3DCrop &myCrop,
                       std::vector<soil::Crit3DLayer> &soilLayers, Crit1DOutput &myOutput,
                       bool optimizeIrrigation, std::string &myError)
{
    double ploughedSoilDepth = 0.5;     /*!< [m] depth of ploughed soil (working layer) */

    // Initialize output
    myOutput.initialize();
    int doy = getDoyFromDate(myDate);

    // check daily meteo data
    if (! meteoPoint.existDailyData(myDate))
    {
        myError = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    double prec = double(meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
    double tmin = double(meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
    double tmax = double(meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));

    if (isEqual(prec, NODATA) || isEqual(tmin, NODATA) || isEqual(tmax, NODATA))
    {
        myError = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    // check on wrong data
    if (prec < 0) prec = 0;
    myOutput.dailyPrec = prec;

    // water table
    myOutput.dailyWaterTable = double(meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));

    // prec forecast
    double precTomorrow = double(meteoPoint.getMeteoPointValueD(myDate.addDays(1), dailyPrecipitation));
    if (isEqual(precTomorrow, NODATA)) precTomorrow = 0;

    // ET0
    myOutput.dailyEt0 = double(meteoPoint.getMeteoPointValueD(myDate, dailyReferenceEvapotranspirationHS));
    if (isEqual(myOutput.dailyEt0, NODATA) || myOutput.dailyEt0 <= 0)
        myOutput.dailyEt0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, meteoPoint.latitude, doy, tmax, tmin);

    // update LAI and root depth
    if (! myCrop.dailyUpdate(myDate, meteoPoint.latitude, soilLayers, tmin, tmax, myOutput.dailyWaterTable, myError))
        return false;

    // Evaporation / transpiration
    myOutput.dailyMaxEvaporation = myCrop.getMaxEvaporation(myOutput.dailyEt0);
    myOutput.dailyMaxTranspiration = myCrop.getMaxTranspiration(myOutput.dailyEt0);

    // WATERTABLE (if available)
    myOutput.dailyCapillaryRise = computeCapillaryRise(soilLayers, myOutput.dailyWaterTable);

    // IRRIGATION
    double irrigation = myCrop.getIrrigationDemand(doy, prec, precTomorrow, myOutput.dailyMaxTranspiration, soilLayers);
    if (optimizeIrrigation) irrigation = MINVALUE(irrigation, myCrop.getCropWaterDeficit(soilLayers));

    // assign irrigation: optimal (subirrigation) or add to precipitation (sprinkler/drop)
    double waterInput = prec;
    myOutput.dailyIrrigation = 0;

    if (irrigation > 0)
    {
        if (optimizeIrrigation)
        {
            myOutput.dailyIrrigation = computeOptimalIrrigation(soilLayers, irrigation);
        }
        else
        {
            myOutput.dailyIrrigation = irrigation;
            waterInput += irrigation;
        }
    }

    // INFILTRATION
    myOutput.dailyDrainage = computeInfiltration(soilLayers, waterInput, ploughedSoilDepth);

    // LATERAL DRAINAGE
    myOutput.dailyLateralDrainage = computeLateralDrainage(soilLayers);

    // EVAPORATION
    myOutput.dailyEvaporation = computeEvaporation(soilLayers, myOutput.dailyMaxEvaporation);

    // RUNOFF (after evaporation)
    myOutput.dailySurfaceRunoff = computeSurfaceRunoff(myCrop, soilLayers);

    // adjust irrigation losses
    if (! optimizeIrrigation)
    {
        if ((myOutput.dailySurfaceRunoff > 1) && (myOutput.dailyIrrigation > 0))
        {
            myOutput.dailyIrrigation -= floor(myOutput.dailySurfaceRunoff);
            myOutput.dailySurfaceRunoff -= floor(myOutput.dailySurfaceRunoff);
        }
    }

    // TRANSPIRATION
    double waterStress;
    myOutput.dailyTranspiration = myCrop.computeTranspiration(myOutput.dailyMaxTranspiration, soilLayers, &waterStress);

    // assign transpiration
    if (myOutput.dailyTranspiration > 0)
    {
        for (unsigned int i = unsigned(myCrop.roots.firstRootLayer); i <= unsigned(myCrop.roots.lastRootLayer); i++)
        {
            soilLayers[i].waterContent -= myCrop.layerTranspiration[i];
        }
    }

    // output variables
    myOutput.dailySurfaceWaterContent = soilLayers[0].waterContent;
    myOutput.dailySoilWaterContent = getSoilWaterContent(soilLayers, 1.0);
    myOutput.dailyWaterDeficit = getSoilWaterDeficit(soilLayers, 1.0);
    myOutput.dailyWaterDeficit_25 = getSoilWaterDeficit(soilLayers, 0.25);
    myOutput.dailyAvailableWater = getSoilAvailableWater(soilLayers, 1.0);
    myOutput.dailyFractionAW = getSoilFractionAW(soilLayers, 1.0);
    myOutput.dailyReadilyAW = getReadilyAvailableWater(myCrop, soilLayers);

    return true;
}


