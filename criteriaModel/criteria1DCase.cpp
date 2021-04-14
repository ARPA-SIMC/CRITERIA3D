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
    along with CRITERIA3D.  if not, see <http://www.gnu.org/licenses/>.

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
#include "soilFluxes3D.h"


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
    this->dailyCapillaryRise = NODATA;
    this->dailyWaterTable = NODATA;
}


Crit1DCase::Crit1DCase()
{
    minLayerThickness = 0.02;           /*!< [m] layer thickness (default = 2 cm)  */
    geometricFactor = 1.2;              /*!< [-] factor for geometric progression of thickness  */
    ploughedSoilDepth = 0.5;            /*!< [m] depth of ploughed soil (working layer) */
    fieldArea = 4000;                   /*!< [m2]   */
    fieldSlope = 0.002;                 /*!< [m2 m-1]   */

    soilLayers.clear();
    prevWaterContent.clear();
}


bool Crit1DCase::initializeSoil(std::string &error)
{
    soilLayers.clear();

    double factor = 1.0;
    if (unit.isGeometricLayers) factor = geometricFactor;

    if (! mySoil.setSoilLayers(minLayerThickness, factor, soilLayers, error))
        return false;

    if (unit.isNumericalInfiltration)
    {
        if (! initializeNumericalFluxes(error))
            return false;
    }

    initializeWater(soilLayers);

    return true;
}


/*!
 * \brief initalize structures for numerical solution of water fluxes
 * (soilFluxes3D library)
 */
bool Crit1DCase::initializeNumericalFluxes(std::string &error)
{
    int nrLayers = soilLayers.size();
    int lastLayer = nrLayers-1;
    int nrlateralLinks = 0;

    int result = soilFluxes3D::initialize(nrLayers, nrLayers, nrlateralLinks, true, false, false);
    if (result != CRIT3D_OK)
    {
        error = "Error in initialize numerical fluxes";
        return false;
    }

    float horizontalConductivityRatio = 10.0;
    soilFluxes3D::setHydraulicProperties(fittingOptions.waterRetentionCurve, MEAN_LOGARITHMIC, horizontalConductivityRatio);
    soilFluxes3D::setNumericalParameters(1, 3600, 100, 10, 12, 3);

    // set soil properties (units of measurement: MKS)
    int soilIndex = 0;
    for (unsigned int horizonIndex = 0; horizonIndex < mySoil.nrHorizons; horizonIndex++)
    {
        soil::Crit3DHorizon horizon = mySoil.horizon[horizonIndex];
        float soilFraction = (1.0 - horizon.coarseFragments);
        result = soilFluxes3D::setSoilProperties(soilIndex, horizonIndex,
                            horizon.vanGenuchten.alpha * GRAVITY,
                            horizon.vanGenuchten.n, horizon.vanGenuchten.m,
                            horizon.vanGenuchten.he / GRAVITY,
                            horizon.vanGenuchten.thetaR * soilFraction,
                            horizon.vanGenuchten.thetaS * soilFraction,
                            horizon.waterConductivity.kSat / 100.0 / DAY_SECONDS,
                            horizon.waterConductivity.l,
                            horizon.organicMatter, horizon.texture.clay / 100.0);
        if (result != CRIT3D_OK)
        {
            error = "Error in setSoilProperties, horizon nr: " + std::to_string(horizonIndex);
            return false;
        }
    }

    // set surface properties
    double maxSurfaceWater = crop.getSurfaceWaterPonding();     // [mm]
    maxSurfaceWater /= 1000.0;                                  // [m]
    double roughnessManning = 0.024;                            // [s m^-0.33]
    int surfaceIndex = 0;
    soilFluxes3D::setSurfaceProperties(surfaceIndex, roughnessManning, maxSurfaceWater);

    // field structure (baulatura)
    float x0 = 0;
    float y0 = 0;
    double z0 = 0;
    double lx = 25;                 // [m]
    double ly = 200;                // [m]
    fieldArea = lx * ly;          // [m^2]
    fieldSlope = 0.002;                  // [m m^-1]

    // set surface (node 0)
    bool isSurface = true;
    int nodeIndex = 0;
    soilFluxes3D::setNode(nodeIndex, x0, y0, z0, fieldArea, isSurface, true, BOUNDARY_RUNOFF, fieldSlope, ly);
    soilFluxes3D::setNodeSurface(nodeIndex, surfaceIndex);
    soilFluxes3D::setNodeLink(nodeIndex, nodeIndex + 1, DOWN, fieldArea);

    // set nodes
    isSurface = false;
    for (int i = 1; i < nrLayers; i++)
    {
        double volume = fieldArea * soilLayers[i].thickness;      // [m^3]
        double z = z0 - soilLayers[i].depth;                        // [m]
        if (i == lastLayer)
        {
            if (unit.useWaterTableData)
                soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_PRESCRIBEDTOTALPOTENTIAL, fieldSlope, fieldArea);
            else
                soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_FREEDRAINAGE, fieldSlope, fieldArea);
        }
        else
        {
            double boundaryArea = ly * soilLayers[i].thickness;
            soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_FREELATERALDRAINAGE, fieldSlope, boundaryArea);
        }

        // set soil
        int horizonIndex = mySoil.getHorizonIndex(soilLayers[i].depth);
        soilFluxes3D::setNodeSoil(i, soilIndex, horizonIndex);

        // set links
        soilFluxes3D::setNodeLink(i, i-1, UP, fieldArea);
        if (i != lastLayer)
        {
            soilFluxes3D::setNodeLink(i, i+1, DOWN, fieldArea);
        }
    }

    return true;
}


/*!
 * \brief numerical solution of soil water fluxes (soilFluxes3D library)
 * \note units of measurement are MKS
 */
bool Crit1DCase::computeNumericalFluxes(double dailyWaterInput, std::string &error)
{
    int nrLayers = soilLayers.size();
    int lastLayer = nrLayers - 1;

    // set bottom boundary conditions (water table)
    if (unit.useWaterTableData)
    {
        double totalPotential;                  // [m]
        if (output.dailyWaterTable != NODATA)
        {
            totalPotential = output.dailyWaterTable;
        }
        else
        {
            // total potential = depth of the last layer + water potential at field capacity
            // this condition is equal to empirical model
            double fieldCapacity = soil::getFieldCapacity(soilLayers[lastLayer].horizon, soil::METER);
            totalPotential = soilLayers[lastLayer].depth + fieldCapacity;
        }
        soilFluxes3D::setPrescribedTotalPotential(lastLayer, -totalPotential);
    }

    // set surface
    int surfaceIndex = 0;
    soilFluxes3D::setWaterContent(surfaceIndex, soilLayers[surfaceIndex].waterContent * 0.001);       // [m]

    // set soil profile
    for (int i=1; i < nrLayers; i++)
    {
        soilFluxes3D::setMatricPotential(i, soilLayers[i].getWaterPotential());
    }

    soilFluxes3D::initializeBalance();

    // daily cycle
    double flux = (dailyWaterInput * 0.001 * fieldSlope) / DAY_SECONDS;  // [m3 s-1]
    for (int hour=1; hour <= 24; hour++)
    {
        soilFluxes3D::setWaterSinkSource(surfaceIndex, flux);
        soilFluxes3D::computePeriod(HOUR_SECONDS);
    }

    // mass balance error
    double massBalanceError = soilFluxes3D::getWaterMBR() - 1;

    /* WATER: OUTPUT
    'restituzione dei valori di U(L) e flux(L)
    'unita' di misura: da [m] a [mm]
    U(0) = Criteria3D.GetWaterContent(0) * 1000#
    Flux(0) = -Criteria3D.GetWaterSumFlux(0, CRIT3D.DOWN) * 1000 / Criteria3D.CRIT3DSurface
    For L = 1 To nrLayers
        U(L) = Criteria3D.GetWaterContent(L) * (suolo(L).spess * 10#)
        If (L = nrLayers) Then
            Flux(L) = Criteria3D.GetBoundaryWaterFlux(L) * 1000 / Criteria3D.CRIT3DSurface
            fluxDown = Criteria3D.GetBoundaryWaterFlux(L)

        Else
            fluxDown = -Criteria3D.GetWaterSumFlux(L, CRIT3D.DOWN)
            fluxUp = Criteria3D.GetWaterSumFlux(L, CRIT3D.UP)
            Flux(L) = fluxDown * 1000 / Criteria3D.CRIT3DSurface
        End If
    Next L

    // drenaggio o risalita capillare
    if Flux(nrLayers) >= 0
        RcTot = Flux(nrLayers)
        DrenGG = 0
    else
        DrenGG = -Flux(nrLayers)
        RcTot = 0
    } */

    return true;
}


void Crit1DCase::saveWaterContent()
{
    prevWaterContent.clear();
    prevWaterContent.resize(soilLayers.size());
    for (unsigned int i = 0; i < soilLayers.size(); i++)
    {
        prevWaterContent[i] = soilLayers[i].waterContent;
    }
}


void Crit1DCase::restoreWaterContent()
{
    for (unsigned int i = 0; i < soilLayers.size(); i++)
    {
        soilLayers[i].waterContent = prevWaterContent[i];
    }
}


/*!
 * \brief compute water fluxes
 * \param dailyWaterInput [mm] sum of precipitation and irrigation
 */
bool Crit1DCase::computeWaterFluxes(double dailyWaterInput, std::string &error)
{
    if (unit.isNumericalInfiltration)
    {
        return computeNumericalFluxes(dailyWaterInput, error);
    }
    else
    {
        // WATERTABLE
        output.dailyCapillaryRise = 0;
        if (unit.useWaterTableData)
        {
            output.dailyCapillaryRise = computeCapillaryRise(soilLayers, output.dailyWaterTable);
        }

        // INFILTRATION
        output.dailyDrainage = computeInfiltration(soilLayers, dailyWaterInput, ploughedSoilDepth);

        // LATERAL DRAINAGE
        output.dailyLateralDrainage = computeLateralDrainage(soilLayers);
    }

    return true;
}


double Crit1DCase::checkIrrigationDemand(int doy, double currentPrec, double nextPrec, double maxTranspiration)
{
    // update days since last irrigation
    if (crop.daysSinceIrrigation != NODATA)
        crop.daysSinceIrrigation++;

    // check irrigated crop
    if (crop.idCrop == "" || ! crop.isLiving || isEqual(crop.irrigationVolume, NODATA) || isEqual(crop.irrigationVolume, 0))
        return 0;

    // check irrigation period
    if (crop.doyStartIrrigation != NODATA && crop.doyEndIrrigation != NODATA)
    {
        if (doy < crop.doyStartIrrigation || doy > crop.doyEndIrrigation)
            return 0;
    }
    if (crop.degreeDaysStartIrrigation != NODATA && crop.degreeDaysEndIrrigation != NODATA)
    {
        if (crop.degreeDays < crop.degreeDaysStartIrrigation || crop.degreeDays > crop.degreeDaysEndIrrigation)
            return 0;
    }

    // check forecast (today and tomorrow)
    double waterNeeds = crop.irrigationVolume / crop.irrigationShift;
    double todayWater = currentPrec + soilLayers[0].waterContent;
    double twoDaysWater = todayWater + nextPrec;
    if (todayWater >= waterNeeds) return 0;
    if (twoDaysWater >= 2*waterNeeds) return 0;

    // check water stress (before infiltration)
    double threshold = 1. - crop.stressTolerance;

    double waterStress = 0;
    crop.computeTranspiration(maxTranspiration, soilLayers, waterStress);
    if (waterStress <= threshold)
        return 0;

    // check irrigation shift
    if (crop.daysSinceIrrigation != NODATA)
    {
        // stress too high -> forced irrigation
        if ((crop.daysSinceIrrigation < crop.irrigationShift) && (waterStress < (threshold + 0.1)))
            return 0;
    }

    // check irrigation quantity
    double irrigation = crop.irrigationVolume;
    if (crop.irrigationShift > 1)
        irrigation -= floor(twoDaysWater);

    if (unit.isOptimalIrrigation)
        irrigation = MINVALUE(irrigation, crop.getCropWaterDeficit(soilLayers));

    // reset irrigation shift
    crop.daysSinceIrrigation = 0;
    return irrigation;
}



/*!
 * \brief run model (daily cycle)
 * \param myDate
 */
bool Crit1DCase::computeDailyModel(Crit3DDate myDate, std::string &error)
{
    output.initialize();

    int doy = getDoyFromDate(myDate);

    // check daily meteo data
    if (! meteoPoint.existDailyData(myDate))
    {
        error = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    double prec = double(meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
    double tmin = double(meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
    double tmax = double(meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));

    if (isEqual(prec, NODATA) || isEqual(tmin, NODATA) || isEqual(tmax, NODATA))
    {
        error = "Missing weather data: " + myDate.toStdString();
        return false;
    }

    // check on wrong data
    if (prec < 0) prec = 0;
    output.dailyPrec = prec;

    // water table
    output.dailyWaterTable = double(meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
    // check
    if (output.dailyWaterTable != NODATA)
        output.dailyWaterTable = MAXVALUE(output.dailyWaterTable, 0.01);

    // prec forecast
    double precTomorrow = double(meteoPoint.getMeteoPointValueD(myDate.addDays(1), dailyPrecipitation));
    if (isEqual(precTomorrow, NODATA)) precTomorrow = 0;

    // ET0
    output.dailyEt0 = double(meteoPoint.getMeteoPointValueD(myDate, dailyReferenceEvapotranspirationHS));
    if (isEqual(output.dailyEt0, NODATA) || output.dailyEt0 <= 0)
        output.dailyEt0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, meteoPoint.latitude, doy, tmax, tmin);

    // update LAI and root depth
    if (! crop.dailyUpdate(myDate, meteoPoint.latitude, soilLayers, tmin, tmax, output.dailyWaterTable, error))
        return false;

    // Evaporation / transpiration
    output.dailyMaxEvaporation = crop.getMaxEvaporation(output.dailyEt0);
    output.dailyMaxTranspiration = crop.getMaxTranspiration(output.dailyEt0);

    // WATER FLUXES
    saveWaterContent();
    if (! computeWaterFluxes(output.dailyPrec, error)) return false;

    // IRRIGATION
    output.dailyIrrigation = 0;
    double irrigation = checkIrrigationDemand(doy, prec, precTomorrow, output.dailyMaxTranspiration);

    // assign irrigation: optimal (subirrigation) or add to precipitation (sprinkler)
    if (irrigation > 0)
    {
        restoreWaterContent();
        double totalWaterInput;
        if (unit.isOptimalIrrigation)
        {
            output.dailyIrrigation = assignOptimalIrrigation(soilLayers, irrigation);
            totalWaterInput = prec;
        }
        else
        {
            output.dailyIrrigation = irrigation;
            totalWaterInput = prec + irrigation;
        }
        // recompute water fluxes
        if (! computeWaterFluxes(totalWaterInput, error)) return false;
    }

    // EVAPORATION
    output.dailyEvaporation = computeEvaporation(soilLayers, output.dailyMaxEvaporation);

    // RUNOFF (after evaporation)
    output.dailySurfaceRunoff = computeSurfaceRunoff(crop, soilLayers);

    // adjust irrigation losses
    if (! unit.isOptimalIrrigation)
    {
        if ((output.dailySurfaceRunoff > 1) && (output.dailyIrrigation > 0))
        {
            output.dailyIrrigation -= floor(output.dailySurfaceRunoff);
            output.dailySurfaceRunoff -= floor(output.dailySurfaceRunoff);
        }
    }

    // TRANSPIRATION
    double waterStress = 0;
    output.dailyTranspiration = crop.computeTranspiration(output.dailyMaxTranspiration, soilLayers, waterStress);

    // assign transpiration
    if (output.dailyTranspiration > 0)
    {
        for (unsigned int i = unsigned(crop.roots.firstRootLayer); i <= unsigned(crop.roots.lastRootLayer); i++)
        {
            soilLayers[i].waterContent -= crop.layerTranspiration[i];
        }
    }

    // output variables
    output.dailySurfaceWaterContent = soilLayers[0].waterContent;
    output.dailySoilWaterContent = getSoilWaterContent(soilLayers, 1.0);
    output.dailyAvailableWater = getSoilAvailableWater(soilLayers, 1.0);
    output.dailyFractionAW = getSoilFractionAW(soilLayers, 1.0);
    output.dailyReadilyAW = getReadilyAvailableWater(crop, soilLayers);

    return true;
}


/*!
 * \brief get volumetric water content at specific depth
 * \param depth = computation soil depth  [cm]
 * \return volumetric water content [-]
 */
double Crit1DCase::getWaterContent(double depth)
{
    depth /= 100;                                   // [cm] --> [m]
    if (depth <= 0 || depth > mySoil.totalDepth)
        return NODATA;

    double upperDepth, lowerDepth;
    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        upperDepth = soilLayers[i].depth - soilLayers[i].thickness * 0.5;
        lowerDepth = soilLayers[i].depth + soilLayers[i].thickness * 0.5;
        if (depth >= upperDepth && depth <= lowerDepth)
        {
            return soilLayers[i].waterContent / (soilLayers[i].thickness * 1000);
        }
    }

    return NODATA;
}


/*!
 * \brief get water potential at specific depth
 * \param depth = computation soil depth  [cm]
 * \return water potential [kPa]
 */
double Crit1DCase::getWaterPotential(double depth)
{
    depth /= 100;                                   // [cm] --> [m]
    if (depth <= 0 || depth > mySoil.totalDepth)
        return NODATA;

    double upperDepth, lowerDepth;
    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        upperDepth = soilLayers[i].depth - soilLayers[i].thickness * 0.5;
        lowerDepth = soilLayers[i].depth + soilLayers[i].thickness * 0.5;
        if (depth >= upperDepth && depth <= lowerDepth)
        {
            return soilLayers[i].getWaterPotential();
        }
    }

    return NODATA;
}


/*!
 * \brief getSoilWaterDeficit
 * \param depth = computation soil depth  [cm]
 * \return sum of water deficit from zero to depth (mm)
 */
double Crit1DCase::getSoilWaterDeficit(double depth)
{
    depth /= 100;                           // [cm] --> [m]
    double lowerDepth, upperDepth;          // [m]
    double waterDeficitSum = 0;             // [mm]

    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        lowerDepth = soilLayers[i].depth + soilLayers[i].thickness * 0.5;

        if (lowerDepth < depth)
        {
            waterDeficitSum += soilLayers[i].FC - soilLayers[i].waterContent;
        }
        else
        {
            // fraction of last layer
            upperDepth = soilLayers[i].depth - soilLayers[i].thickness * 0.5;
            double layerDeficit = soilLayers[i].FC - soilLayers[i].waterContent;
            double depthFraction = (depth - upperDepth) / soilLayers[i].thickness;
            return waterDeficitSum + layerDeficit * depthFraction;
        }
    }

    return waterDeficitSum;
}


