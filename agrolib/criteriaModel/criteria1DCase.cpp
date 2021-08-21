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
    // deafult values
    minLayerThickness = 0.02;           /*!< [m] layer thickness (default = 2 cm)  */
    geometricFactor = 1.2;              /*!< [-] factor for geometric progression of thickness  */
    ploughedSoilDepth = 0.5;            /*!< [m] depth of ploughed soil (working layer) */
    lx = 2;                             /*!< [m]   */
    ly = 2;                             /*!< [m]   */
    area = lx * ly;                     /*!< [m2]  */

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


// meteoPoint has to be loaded
void Crit1DCase::initializeWaterContent(Crit3DDate myDate)
{
    initializeWater(soilLayers);

    // water table
    if (unit.useWaterTableData)
    {
        float waterTable = meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth);
        computeCapillaryRise(soilLayers, waterTable);
    }
}


/*!
 * \brief initalize structures for numerical solution of water fluxes
 * (soilFluxes3D library)
 */
bool Crit1DCase::initializeNumericalFluxes(std::string &error)
{
    unsigned nrLayers = unsigned(soilLayers.size());
    if (nrLayers < 1)
    {
        error = "Missing soil layers";
        return false;
    }

    unsigned lastLayer = nrLayers-1;
    int nrlateralLinks = 0;

    int result = soilFluxes3D::initialize(nrLayers, nrLayers, nrlateralLinks, true, false, false);
    if (result != CRIT3D_OK)
    {
        error = "Error in initialize numerical fluxes";
        return false;
    }

    float horizontalConductivityRatio = 10.0;
    soilFluxes3D::setHydraulicProperties(fittingOptions.waterRetentionCurve, MEAN_LOGARITHMIC, horizontalConductivityRatio);
    soilFluxes3D::setNumericalParameters(60, 3600, 100, 10, 12, 3);

    // set soil properties (units of measurement: MKS)
    int soilIndex = 0;
    for (unsigned int horizonIndex = 0; horizonIndex < mySoil.nrHorizons; horizonIndex++)
    {
        soil::Crit3DHorizon horizon = mySoil.horizon[horizonIndex];
        double soilFraction = (1.0 - horizon.coarseFragments);
        result = soilFluxes3D::setSoilProperties(soilIndex, horizonIndex,
                            horizon.vanGenuchten.alpha * GRAVITY,
                            horizon.vanGenuchten.n, horizon.vanGenuchten.m,
                            horizon.vanGenuchten.he / GRAVITY,
                            horizon.vanGenuchten.thetaR * soilFraction,
                            horizon.vanGenuchten.thetaS * soilFraction,
                            (horizon.waterConductivity.kSat * 0.01) / DAY_SECONDS,
                            horizon.waterConductivity.l,
                            horizon.organicMatter, horizon.texture.clay * 0.01);
        if (result != CRIT3D_OK)
        {
            error = "Error in setSoilProperties, horizon nr: " + std::to_string(horizonIndex);
            return false;
        }
    }

    // set surface properties
    double maxSurfaceWater = crop.getSurfaceWaterPonding() * 0.001;     // [m]
    double roughnessManning = 0.024;                                    // [s m^-0.33]
    int surfaceIndex = 0;
    soilFluxes3D::setSurfaceProperties(surfaceIndex, roughnessManning, maxSurfaceWater);

    // center
    float x0 = 0;
    float y0 = 0;
    double z0 = 0;

    // set surface (node 0)
    bool isSurface = true;
    int nodeIndex = 0;
    soilFluxes3D::setNode(nodeIndex, x0, y0, z0, area, isSurface, true, BOUNDARY_RUNOFF, unit.slope, ly);
    soilFluxes3D::setNodeSurface(nodeIndex, surfaceIndex);
    soilFluxes3D::setNodeLink(nodeIndex, nodeIndex + 1, DOWN, area);

    // set nodes
    isSurface = false;
    for (unsigned int i = 1; i < nrLayers; i++)
    {
        double volume = area * soilLayers[i].thickness;             // [m^3]
        double z = z0 - soilLayers[i].depth;                        // [m]
        if (i == lastLayer)
        {
            if (unit.useWaterTableData)
                soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_PRESCRIBEDTOTALPOTENTIAL, unit.slope, area);
            else
                soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_FREEDRAINAGE, unit.slope, area);
        }
        else
        {
            double boundaryArea = ly * soilLayers[i].thickness;
            soilFluxes3D::setNode(i, x0, y0, z, volume, isSurface, true, BOUNDARY_FREELATERALDRAINAGE, unit.slope, boundaryArea);
        }

        // set soil
        int horizonIndex = mySoil.getHorizonIndex(soilLayers[i].depth);
        soilFluxes3D::setNodeSoil(i, soilIndex, horizonIndex);

        // set links
        soilFluxes3D::setNodeLink(i, i-1, UP, area);
        if (i != lastLayer)
        {
            soilFluxes3D::setNodeLink(i, i+1, DOWN, area);
        }
    }

    return true;
}


/*!
 * \brief numerical solution of soil water fluxes (soilFluxes3D library)
 * \note units of measurement are MKS
 */
bool Crit1DCase::computeNumericalFluxes(const Crit3DDate &myDate, std::string &error)
{
    int nrLayers = int(soilLayers.size());
    int lastLayer = nrLayers - 1;
    error = "";

    // set bottom boundary conditions (water table)
    if (unit.useWaterTableData)
    {
        double totalPotential;                          // [m]
        double boundaryZ = 1.0;                         // [m]
        if (output.dailyWaterTable != NODATA)
        {
            totalPotential = output.dailyWaterTable;    // [m]
        }
        else
        {
            // boundary total potential = depth of the last layer + boundaryZ + field capacity
            double fieldCapacity = -soil::getFieldCapacity(soilLayers[lastLayer].horizon, soil::METER);     // [m]
            double waterPotential = soilLayers[lastLayer].getWaterPotential() / GRAVITY;                    // [m]
            totalPotential = soilLayers[lastLayer].depth + boundaryZ;                                       // [m]
            totalPotential += MINVALUE(fieldCapacity, waterPotential);
        }
        soilFluxes3D::setPrescribedTotalPotential(lastLayer, -totalPotential);
    }

    // set surface
    int surfaceIndex = 0;
    soilFluxes3D::setWaterContent(surfaceIndex, soilLayers[surfaceIndex].waterContent * 0.001);   // [m]

    // set soil profile
    for (int i=1; i < nrLayers; i++)
    {
        double waterPotential = soilLayers[i].getWaterPotential() / GRAVITY;   // [m]
        soilFluxes3D::setMatricPotential(i, -waterPotential);
    }

    soilFluxes3D::initializeBalance();

    // precipitation
    // TODO improve lat < 0
    int duration = 24;                              // [hours] winter
    if (myDate.month >= 5 && myDate.month <= 9)
    {
        duration = 12;                               // [hours] summer
    }
    int precH0 = 13 - duration * 0.5;
    int precH1 = precH0 + duration -1;
    double precFlux = (area * output.dailyPrec * 0.001) / (HOUR_SECONDS * duration);  // [m3 s-1]

    // irrigation
    int irrH0 = 0;
    int irrH1 = 0;
    double irrFlux = 0;
    if (! unit.isOptimalIrrigation && output.dailyIrrigation > 0)
    {
        duration = int(output.dailyIrrigation / 3);     // [hours]
        irrH0 = 6;                                      // morning
        irrH1 = irrH0 + duration -1;
        irrFlux = (area * output.dailyIrrigation * 0.001) / (HOUR_SECONDS * duration);  // [m3 s-1]
    }

    // daily cycle
    for (int hour=1; hour <= 24; hour++)
    {
        double flux = 0;                            // [m3 s-1]
        if (hour >= precH0 && hour <= precH1 && precFlux > 0)
            flux += precFlux;

        if (hour >= irrH0 && hour <= irrH1 && irrFlux > 0)
            flux += irrFlux;

        soilFluxes3D::setWaterSinkSource(surfaceIndex, flux);
        soilFluxes3D::computePeriod(HOUR_SECONDS);
    }

    // mass balance error
    //double massBalanceError = soilFluxes3D::getWaterMBR() - 1;

    // output (from [m] to [mm])
    soilLayers[surfaceIndex].waterContent = soilFluxes3D::getWaterContent(surfaceIndex) * 1000;
    for (int i=1; i < nrLayers; i++)
    {
        soilLayers[i].waterContent = soilFluxes3D::getWaterContent(i) * soilLayers[i].thickness * 1000;
    }

    output.dailySurfaceRunoff = -(soilFluxes3D::getBoundaryWaterFlow(surfaceIndex) / area) * 1000;
    output.dailyLateralDrainage = -(soilFluxes3D::getBoundaryWaterSumFlow(BOUNDARY_FREELATERALDRAINAGE) / area) * 1000;

    // drainage / capillary rise
    double fluxBottom = (soilFluxes3D::getBoundaryWaterFlow(lastLayer) / area) * 1000;
    if (fluxBottom > 0)
    {
        output.dailyCapillaryRise = fluxBottom;
        output.dailyDrainage = 0;
    }
    else
    {
        output.dailyCapillaryRise = 0;
        output.dailyDrainage = -fluxBottom;
    }

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
bool Crit1DCase::computeWaterFluxes(const Crit3DDate &myDate, std::string &error)
{
    if (unit.isNumericalInfiltration)
    {
        return computeNumericalFluxes(myDate, error);
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
        double waterInput = output.dailyPrec;
        if (! unit.isOptimalIrrigation)
            waterInput += output.dailyIrrigation;
        output.dailyDrainage = computeInfiltration(soilLayers, waterInput, ploughedSoilDepth);

        // RUNOFF
        output.dailySurfaceRunoff = computeSurfaceRunoff(crop, soilLayers);

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
    double dailyWaterNeeds = crop.irrigationVolume / crop.irrigationShift;
    double todayWater = currentPrec + soilLayers[0].waterContent;
    double twoDaysWater = todayWater + nextPrec;
    if (todayWater >= dailyWaterNeeds) return 0;
    if (twoDaysWater >= 2*dailyWaterNeeds) return 0;

    // check water stress (before infiltration)
    double threshold = 1. - crop.stressTolerance;

    double waterStress = 0;
    crop.computeTranspiration(maxTranspiration, soilLayers, waterStress);
    if (waterStress <= threshold)
        return 0;

    // check irrigation shift
    if (crop.daysSinceIrrigation != NODATA)
    {
        if (crop.daysSinceIrrigation < crop.irrigationShift)
            return 0;
    }

    // Irrigation scheduled!

    // irrigation quantity
    double irrigation = crop.irrigationVolume;

    // reset irrigation shift
    crop.daysSinceIrrigation = 0;

    return irrigation;
}


/*!
 * \brief run model (daily cycle)
 * \param myDate
 */
bool Crit1DCase::computeDailyModel(Crit3DDate &myDate, std::string &error)
{
    output.initialize();
    double previousWC = getTotalWaterContent();

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

    // Maximum evaporation and transpiration
    output.dailyMaxEvaporation = crop.getMaxEvaporation(output.dailyEt0);
    output.dailyMaxTranspiration = crop.getMaxTranspiration(output.dailyEt0);

    output.dailyIrrigation = 0;
    // Water fluxes (first computation)
    saveWaterContent();
    if (! computeWaterFluxes(myDate, error)) return false;
    // Irrigation
    double irrigation = checkIrrigationDemand(doy, prec, precTomorrow, output.dailyMaxTranspiration);

    // Assign irrigation: optimal (subirrigation) or add to precipitation (sprinkler)
    // and recompute water fluxes
    if (irrigation > 0)
    {
        restoreWaterContent();
        if (! unit.isOptimalIrrigation)
            output.dailyIrrigation = irrigation;
        else
            output.dailyIrrigation = assignOptimalIrrigation(soilLayers, crop.roots.lastRootLayer, irrigation);

        if (! computeWaterFluxes(myDate, error)) return false;
    }

    // adjust irrigation losses
    if (! unit.isOptimalIrrigation)
    {
        if ((output.dailySurfaceRunoff > 1) && (output.dailyIrrigation > 0))
        {
            output.dailyIrrigation -= floor(output.dailySurfaceRunoff);
            output.dailySurfaceRunoff -= floor(output.dailySurfaceRunoff);
        }
    }

    // Evaporation
    output.dailyEvaporation = computeEvaporation(soilLayers, output.dailyMaxEvaporation);

    // Transpiration
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

    // Water balance
    double currentWC = getTotalWaterContent();
    output.dailyBalance = currentWC - (previousWC + output.dailyPrec + output.dailyIrrigation + output.dailyCapillaryRise
                                - output.dailyTranspiration - output.dailyEvaporation - output.dailySurfaceRunoff
                                - output.dailyLateralDrainage - output.dailyDrainage);
    if (fabs(output.dailyBalance) < EPSILON)
        output.dailyBalance = 0;

    // output variables
    output.dailySurfaceWaterContent = soilLayers[0].waterContent;
    output.dailySoilWaterContent = getSoilWaterContent(soilLayers, 1.0);
    output.dailyAvailableWater = getSoilAvailableWater(soilLayers, 1.0);
    output.dailyFractionAW = getSoilFractionAW(soilLayers, 1.0);
    output.dailyReadilyAW = getReadilyAvailableWater(crop, soilLayers);

    return true;
}


/*!
 * \brief getTotalWaterContent
 * \return sum of water content on the profile [mm]
 */
double Crit1DCase::getTotalWaterContent()
{
    double sumWC = 0;
    for (unsigned int i=0; i < soilLayers.size(); i++)
    {
        sumWC += soilLayers[i].waterContent;
    }

    return sumWC;
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
            return -soilLayers[i].getWaterPotential();
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


/*!
 * \brief getAvailableWaterCapacity
 * \param depth = computation soil depth  [cm]
 * \return sum of available water capacity (FC-WP) from zero to depth (mm)
 */
double Crit1DCase::getAvailableWaterCapacity(double depth)
{
    depth /= 100;                           // [cm] --> [m]
    double lowerDepth, upperDepth;          // [m]
    double awc = 0;                         // [mm]

    for (unsigned int i = 1; i < soilLayers.size(); i++)
    {
        lowerDepth = soilLayers[i].depth + soilLayers[i].thickness * 0.5;

        if (lowerDepth < depth)
        {
            awc += soilLayers[i].FC - soilLayers[i].WP;
        }
        else
        {
            // fraction of last layer
            upperDepth = soilLayers[i].depth - soilLayers[i].thickness * 0.5;
            double layerAWC = soilLayers[i].FC - soilLayers[i].WP;
            double depthFraction = (depth - upperDepth) / soilLayers[i].thickness;
            return awc + layerAWC * depthFraction;
        }
    }

    return awc;
}


