/*!
    \name Solar Radiation
    \copyright 2026 Gabriele Antolini, Fausto Tomei
    \note  This library uses G_calc_solar_position() by Markus Neteler

    This library is part of CRITERIA3D.
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

    \authors
    Gabriele Antolini gantolini@arpae.it
    Fausto Tomei ftomei@arpae.it
*/

#include "commonConstants.h"
#include "basicMath.h"
#include "gis.h"
#include "meteoPoint.h"
#include "sunPosition.h"
#include "solarRadiation.h"
#include "physics.h"

#include <omp.h>
#include <math.h>
#include <algorithm>


Crit3DRadiationMaps::Crit3DRadiationMaps()
{
    latMap = new gis::Crit3DRasterGrid;
    lonMap = new gis::Crit3DRasterGrid;
    slopeMap = new gis::Crit3DRasterGrid;
    aspectMap = new gis::Crit3DRasterGrid;
    transmissivityMap = new gis::Crit3DRasterGrid;
    globalRadiationMap = new gis::Crit3DRasterGrid;
    beamRadiationMap = new gis::Crit3DRasterGrid;
    diffuseRadiationMap = new gis::Crit3DRasterGrid;
    reflectedRadiationMap = new gis::Crit3DRasterGrid;
    sunElevationMap = new gis::Crit3DRasterGrid;

    isComputed = false;

}

Crit3DRadiationMaps::Crit3DRadiationMaps(const gis::Crit3DRasterGrid& dem, const gis::Crit3DGisSettings& gisSettings)
{
    latMap = new gis::Crit3DRasterGrid;
    lonMap = new gis::Crit3DRasterGrid;
    gis::computeLatLonMaps(dem, latMap, lonMap, gisSettings);

    slopeMap = new gis::Crit3DRasterGrid;
    aspectMap = new gis::Crit3DRasterGrid;
    gis::computeSlopeAspectMaps(dem, slopeMap, aspectMap);

    transmissivityMap = new gis::Crit3DRasterGrid;
    transmissivityMap->initializeGrid(dem, CLEAR_SKY_TRANSMISSIVITY_DEFAULT);

    globalRadiationMap = new gis::Crit3DRasterGrid;
    globalRadiationMap->initializeGrid(dem);

    beamRadiationMap = new gis::Crit3DRasterGrid;
    beamRadiationMap->initializeGrid(dem);

    diffuseRadiationMap = new gis::Crit3DRasterGrid;
    diffuseRadiationMap->initializeGrid(dem);

    reflectedRadiationMap = new gis::Crit3DRasterGrid;
    reflectedRadiationMap->initializeGrid(dem);

    sunElevationMap = new gis::Crit3DRasterGrid;
    sunElevationMap->initializeGrid(dem);

    isComputed = false;
}


Crit3DRadiationMaps::~Crit3DRadiationMaps()
{
    // release raster data, then the objects
    clear();

    delete latMap;
    delete lonMap;
    delete slopeMap;
    delete aspectMap;
    delete transmissivityMap;
    delete globalRadiationMap;
    delete beamRadiationMap;
    delete diffuseRadiationMap;
    delete reflectedRadiationMap;
    delete sunElevationMap;
}


// release the raster data: the map objects remain valid and reusable
void Crit3DRadiationMaps::clear()
{
    latMap->clear();
    lonMap->clear();
    slopeMap->clear();
    aspectMap->clear();
    transmissivityMap->clear();
    globalRadiationMap->clear();
    beamRadiationMap->clear();
    diffuseRadiationMap->clear();
    reflectedRadiationMap->clear();
    sunElevationMap->clear();

    isComputed = false;
}


void Crit3DRadiationMaps::initialize()
{
    transmissivityMap->emptyGrid();
    globalRadiationMap->emptyGrid();
    beamRadiationMap->emptyGrid();
    diffuseRadiationMap->emptyGrid();
    reflectedRadiationMap->emptyGrid();
    sunElevationMap->emptyGrid();

    isComputed = false;
}


namespace radiation
{
/*
    double linkeMountain[13] = {1.5, 1.6, 1.8, 1.9, 2.0, 2.3, 2.3, 2.3, 2.1, 1.8, 1.6, 1.5, 1.9};
    double linkeRural[13] = {2.1, 2.2, 2.5, 2.9, 3.2, 3.4, 3.5, 3.3, 2.9, 2.6, 2.3, 2.2, 2.75};
    double linkeCity[13] = {3.1, 3.2, 3.5, 4.0, 4.2, 4.3, 4.4, 4.3, 4.0, 3.6, 3.3, 3.1, 3.75};
    double linkeIndustrial[13] = {4.1, 4.3, 4.7, 5.3, 5.5, 5.7, 5.8, 5.7, 5.3, 4.9, 4.5, 4.2, 5.0};
*/

    float readAlbedo(const Crit3DRadiationSettings* radSettings)
    {
        float output = NODATA;
        switch(radSettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                 output = NODATA;
                 break;

            default:
                output = radSettings->getAlbedo();
        }
        return output;
    }

    float readAlbedo(const Crit3DRadiationSettings* radSettings, int row, int col)
    {
        float output = NODATA;
        switch (radSettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                output = radSettings->getAlbedo(row, col);
                break;

            default:
                output = radSettings->getAlbedo(row, col);
        }
        return output;
    }

    float readAlbedo(const Crit3DRadiationSettings *radSettings, const gis::Crit3DPoint& point)
    {
        float output = NODATA;
        switch(radSettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                output = radSettings->getAlbedo(point);
                break;

            default:
                output = radSettings->getAlbedo(point);
        }
        return output;
    }


    float readLinke(const Crit3DRadiationSettings* radSettings)
    {
        float output = NODATA;
        switch(radSettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getLinkeDefault();
                break;

            case PARAM_MODE_MAP:
                 output = NODATA;
                 break;

            default:
                output = radSettings->getLinkeDefault();
        }
        return output;
    }


    float readLinke(const Crit3DRadiationSettings* radSettings, int row, int col)
    {
        float output = NODATA;
        switch(radSettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getLinkeDefault();
                break;

            case PARAM_MODE_MAP:
                output = radSettings->getLinke(row, col);
                break;

            default:
                output = radSettings->getLinke(row, col);
        }
        return output;
    }


    float readLinke(const Crit3DRadiationSettings *radSettings, const gis::Crit3DPoint& point)
    {
        float output = NODATA;
        switch(radSettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getLinkeDefault();
                break;

            case PARAM_MODE_MAP:
                output = radSettings->getLinke(point);
                break;

            default:
                output = radSettings->getLinke(point);
        }
        return output;
    }

    float readAspect(const Crit3DRadiationSettings* radSettings)
    {
        float output = NODATA;

        switch (radSettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = radSettings->getAspect();
                break;
            case TILT_TYPE_DEM:
                output = NODATA;
                break;
        }
        return output;
    }

    float readAspect(const Crit3DRadiationSettings* radSettings, gis::Crit3DRasterGrid* aspectMap, int row,int col)
    {
        float output = NODATA;

        switch (radSettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = radSettings->getAspect();
                break;
            case TILT_TYPE_DEM:
                output = aspectMap->value[row][col];
                break;
        }
        return output;
    }

    float readSlope(const Crit3DRadiationSettings* radSettings)
    {
        float output = NODATA;
        switch (radSettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = radSettings->getTilt();
                break;
            case TILT_TYPE_DEM:
                output = NODATA;
                break;
        }
        return output;
    }

    float readSlope(const Crit3DRadiationSettings* radSettings, gis::Crit3DRasterGrid* slopeMap, int row, int col)
    {
        float output = NODATA;

        switch (radSettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = radSettings->getTilt();
                break;
            case TILT_TYPE_DEM:
                output = slopeMap->value[row][col];
                break;
        }
        return output;
    }


    /*!
     * \brief Clear sky beam irradiance on a horizontal surface [W m-2]
     * \brief (Rigollier et al. 2000)
     * \param linke:  Linke turbidity factor [-]
     */
    double clearSkyBeamHorizontal(double linke, const TsunPosition& sunPosition)
    {       
        // Rayleigh optical thickness (Kasten, 1996)
        double rayleighThickness;
        // relative optical air mass corrected for pressure
        double airMass = double(sunPosition.relOptAirMassCorr);

        if (airMass <= 20)
            rayleighThickness = 1. / (6.6296 + 1.7513 * airMass - 0.1202 * airMass * airMass
                                       + 0.0065 * pow(airMass, 3) - 0.00013 * pow(airMass, 4));
        else
            rayleighThickness = 1. / (10.4 + 0.718 * airMass);

        double horizontalBeam = double(sunPosition.extraIrradianceNormal) * std::sin(sunPosition.elevationRefr * DEG_TO_RAD)
                * exp(-0.8662 * linke * airMass * rayleighThickness);

        return horizontalBeam;
    }


    /*!
     * \brief Diffuse irradiance on a horizontal surface [W m-2]
     * \brief (Rigollier et al. 2000)
     * \param linke:  Linke turbidity factor [-]
     */
    double clearSkyDiffuseHorizontal(double linke, const TsunPosition& sunPosition)
    {
        double Fd;          /*!< [-] solar altitude function            */
        double Trd;         /*!< [-] atmospheric transmittance          */

        if (sunPosition.elevationRefr <= 1e-3)
            return 0;

        Trd = -0.015843 + linke * (0.030543 + 0.0003797 * linke);
        // physical check (may become negative with low linke)
        Trd = std::max(Trd, 1e-6);

        double sinElev = std::max(std::sin(sunPosition.elevationRefr * DEG_TO_RAD), 1e-5);

        double A0 = 0.26463 + linke * (-0.061581 + 0.0031408 * linke);
        // empirical patch to force stability
        if ((A0 * Trd) < 0.0022)
            A0 = 0.0022 / Trd;

        double A1 = 2.0402 + linke * (0.018945 - 0.011161 * linke);
        double A2 = -1.3025 + linke * (0.039231 + 0.0085079 * linke);
        Fd = A0 + A1 * sinElev + A2 * sinElev * sinElev;

        return sunPosition.extraIrradianceNormal * Fd * Trd;
    }


    /*!
     * \brief Beam irradiance on an inclined surface                [W m-2]
     * beam inclined = beam horizontal * geometric ratio
     * \param Bh: beam irradiance on a horizontal surface           [W m-2]
     */
    double getBeamInclined(double Bh, const TsunPosition& sunPosition)
    {
        double sinElev = std::max(std::sin(sunPosition.elevationRefr * DEG_TO_RAD), 1e-6);
        double sinIncidence = std::max(std::sin(sunPosition.incidence * DEG_TO_RAD), 0.0);

        return Bh * (sinIncidence / sinElev);
    }


    /*!
     * \brief Diffuse irradiance on an inclined surface  (Muneer, 1990) [W m-2]
     * Fx = (n * Fg + r_sky) * (1 - Kb) + Kb * F_beam
     * \param Bh   beam irradiance on a horizontal surface              [W m-2]
     * \param Dh   diffuse irradiance on a horizontal surface           [W m-2]
     * \ref Muneer, T. (1990). Solar radiation model for Europe. Building Services Engineering Research and Technology, 11(4), 153–163
     */
    double getDiffuseInclined_Muneer(double Bh, double Dh,
                                     const TsunPosition& sun,
                                     const TradPoint& p)
    {
        if (sun.elevationRefr < 1e-6)
            return 0.0;

        const double slopeRad = p.slope * DEG_TO_RAD;
        const double aspectRad = p.aspect * DEG_TO_RAD;
        const double elevationRad = sun.elevationRefr * DEG_TO_RAD;

        const double sinElev = std::max(std::sin(elevationRad), 1e-6);
        const double sinSlope = std::sin(slopeRad);
        const double cosSlope = std::cos(slopeRad);

        // amount of beam irradiance available [-] (slight physical overshoot allowed)
        double Kb = Bh / (sun.extraIrradianceNormal * sinElev);
        Kb = std::clamp(Kb, 0.0, 1.2);

        const double r_sky = (1.0 + cosSlope) / 2.0;

        // slope-dependent geometric term [-]
        const double sinHalfSlope = std::sin(slopeRad * 0.5);
        const double Fg = sinSlope - slopeRad * cosSlope - PI * sinHalfSlope * sinHalfSlope;

        // sun.incidence is the sun altitude over the tilted surface [degree]
        const double incidenceRad = sun.incidence * DEG_TO_RAD;

        // threshold for very low solar elevation above the inclined surface [rad]
        constexpr double MUNEER_LOW_SUN_RAD = 0.1;       // ~5.73 degrees

        double Fx;      // diffuse irradiance tilt factor [-]
        if (sun.shadow || incidenceRad <= 0.0)
        {
            // 2*b / [PI * (3 + 2 * MUNEER_LOW_SUN_RAD)] = 0.25227049.
            Fx = r_sky + Fg * 0.25227049;
        }
        else
        {
            const double n = 0.00263 - Kb * (0.712 + 0.6883 * Kb);

            double beamTerm;
            if (elevationRad >= MUNEER_LOW_SUN_RAD)
            {
                // normal Muneer formulation
                beamTerm = std::sin(incidenceRad) / sinElev;
            }
            else
            {
                // low-sun approximation used by r.sun/Muneer formulation
                const double azimuthDiff = sun.azimuth * DEG_TO_RAD - aspectRad;
                beamTerm = sinSlope * std::cos(azimuthDiff)
                           / (MUNEER_LOW_SUN_RAD - 0.008 * elevationRad);
            }

            Fx = (n * Fg + r_sky) * (1.0 - Kb) + Kb * beamTerm;
        }

        return Dh * Fx;
    }


    double getReflectedIrradiance(double Bh, double Dh, double albedo, double slope)
    {
        if (slope < 1e-6)
            return 0.;

        // Tariq Muneer (1997)
        albedo = std::clamp(albedo, 0.0, 1.0);
        return (albedo * (Bh + Dh) * (1. - std::cos(slope * DEG_TO_RAD)) / 2.);
    }


    bool isIlluminated(float myTime, float riseTime, float setTime, float sunElevation)
    {
        bool output = false;

        if ((riseTime != NODATA) && (setTime != NODATA) && (sunElevation != NODATA))
            output = ((myTime >= riseTime) && (myTime <= setTime) && (sunElevation > 0));

        return output;
    }


    bool computeShadow(const TradPoint& radPoint, const TsunPosition& sunPosition, const gis::Crit3DRasterGrid& myDem)
    {
        /* INPUT
        supponiamo di avere gia' controllato che siamo dopo l'alba e prima del tramonto
        inizializzazione a sole visibile
        */

        const double x0 = radPoint.x;
        const double y0 = radPoint.y;
        const double z0 = radPoint.height;

        const double cellSize = myDem.header->cellSize;

        const double sinAz = std::sin(sunPosition.azimuth * DEG_TO_RAD);
        const double cosAz = std::cos(sunPosition.azimuth * DEG_TO_RAD);

        const double sinElev = std::sin(sunPosition.elevationRefr * DEG_TO_RAD);
        const double cosElev = std::cos(sunPosition.elevationRefr * DEG_TO_RAD);

        // avoid divsion by zero
        const double tgElev = sinElev / std::max(cosElev, 1e-6);

        const double stepX = SHADOW_FACTOR * sinAz * cellSize;        // [m]
        const double stepY = SHADOW_FACTOR * cosAz * cellSize;        // [m]
        const double stepZ = SHADOW_FACTOR * cellSize * tgElev;       // [m]

        const double maxDeltaH = cellSize * SHADOW_FACTOR * 2.0;

        double maxDistCount;
        if (std::abs(stepZ) < 1e-6)
            maxDistCount = (myDem.maximum - z0) / EPSILON;
        else
            maxDistCount = (myDem.maximum - z0) / stepZ;

        double stepCount = 0.0;
        double step = 1.0;

        while (stepCount < maxDistCount)
        {
            stepCount += step;

            const double x = x0 + stepX * stepCount;
            const double y = y0 + stepY * stepCount;
            const double z = z0 + stepZ * stepCount;

            int row, col;
            myDem.getRowCol(x, y, row, col);

            if (gis::isOutOfGridRowCol(row, col, myDem))
                return false; // out of grid = not shaded

            double zDEM = myDem.value[row][col];

            if (zDEM != myDem.header->flag)
            {
                if ((zDEM - z) > 0.5)
                {
                    return true; // shaded
                }
                else
                {
                    step = (z - zDEM) / maxDeltaH;
                    if (step < 1.0)
                        step = 1.0;
                }
            }
        }

        // not shaded
        return false;
    }


    /*!
     * \brief Diffuse fraction on a horizontal surface (Reindl et al., 1990)
     * \param Kt        clearness index Gh / G0h [-]
     * \param sinElev   sine of solar elevation  [-]
     * \return kd       diffuse fraction [-]
     */
    double diffuseFractionReindl(double Kt, double sinElev)
    {
        Kt = std::clamp(Kt, 0.0, 1.0);
        sinElev = std::clamp(sinElev, 0.0, 1.0);

        double Kd;
        if (Kt <= 0.30)
            Kd = std::min(1.020 - 0.254 * Kt + 0.0123 * sinElev, 1.0);
        else if (Kt < 0.78)
            Kd = std::clamp(1.400 - 1.749 * Kt + 0.177 * sinElev, 0.1, 0.97);
        else
            Kd = std::max(0.486 * Kt - 0.182 * sinElev, 0.1);

        return Kd;
    }


    /*!
     * \brief Diffuse fraction D/G on a horizontal surface (Erbs et al., 1982)
     * \param Kt    clearness index = Gh / G0h [-]
     * \return kd  diffuse fraction [-]
     */
    double diffuseFractionErbs(double Kt)
    {
        Kt = std::clamp(Kt, 0.0, 1.0);

        double Kd;
        if (Kt <= 0.22)
            Kd = 1.0 - 0.09 * Kt;
        else if (Kt <= 0.80)
            // Horner form of equation
            Kd = 0.9511 + Kt * (-0.1604 + Kt * (4.388 + Kt * (-16.638 + Kt * 12.336)));
        else
            Kd = 0.165;

        return std::clamp(Kd, 0.0, 1.0);
    }


    /*!
     * \brief Diffuse irradiance on a horizontal surface from global [W m-2]
     * \param Gh        measured horizontal global irradiance [W/m²]
     * \param G0h       extraterrestrial horizontal solar irradiance [W/m²]
     * \return diffuse irradiance [W/m²]
     */
    double splitDiffuseHorizontal(double Gh, double G0h)
    {
        if (Gh <= 0.)
            return 0.;

        if (G0h <= 1e-6)
            return Gh;

        // diffuse fraction [-]
        //const double kd = diffuseFractionReindl(Gh / G0h, sinElev);
        const double kd = diffuseFractionErbs(Gh / G0h);

        return std::min(Gh * kd, Gh);
    }


    bool computeRadiationRsun(const Crit3DRadiationSettings* radSettings, double temperature, const Crit3DTime& myTime,
                              double linke, double albedo, double clearSkyTransmissivity, double transmissivity,
                              TsunPosition &sunPosition, TradPoint& radPoint, const gis::Crit3DRasterGrid& dem)
    {
        int myYear, myMonth, myDay;
        int myHour, myMinute, mySecond;
        double Bhc, Bh;
        double Dhc, dH;
        double Ghc, Gh;
        bool isPointIlluminated;

        Crit3DTime localTime;
        localTime = myTime;
        if (radSettings->gisSettings->isUTC)
        {
            localTime = myTime.addSeconds(radSettings->gisSettings->timeZone * 3600);
        }

        myYear = localTime.date.year;
        myMonth =  localTime.date.month;
        myDay =  localTime.date.day;
        myHour = localTime.getHour();
        myMinute = localTime.getMinutes();
        mySecond = int(localTime.getSeconds());

        /*! Ambient default dry-bulb temperature (degrees C) (used for refraction correction) */
        if (isEqual(temperature, NODATA))
            temperature = TEMPERATURE_DEFAULT;

        /*! Surface pressure (millibars = hPa) (used for refraction correction and optical air mass) */
        double pressure = pressureFromAltitude(radPoint.height) * 0.01;

        /*! Sun position */
        if (! computeSunPosition(float(radPoint.lon), float(radPoint.lat), radSettings->gisSettings->timeZone,
                                myYear, myMonth, myDay, myHour, myMinute, mySecond,
                                (float)temperature, (float)pressure, (float)radPoint.aspect, (float)radPoint.slope, sunPosition))
            return false;

        /*! Shadowing */
        isPointIlluminated = isIlluminated(float(localTime.time), sunPosition.rise, sunPosition.set, sunPosition.elevationRefr);
        sunPosition.shadow = ! isPointIlluminated;

        if (radSettings->getShadowing())
        {
            if (isPointIlluminated && dem.isLoaded && ! gis::isOutOfGridXY(radPoint.x, radPoint.y, dem.header))
                sunPosition.shadow = computeShadow(radPoint, sunPosition, dem);
        }

        if (! isPointIlluminated)
        {
            radPoint.beam = 0;
            radPoint.diffuse = 0;
            radPoint.reflected = 0;
            radPoint.global = 0;

            return true;
        }

        if (radSettings->getRealSky() && isEqual(transmissivity, NODATA))
            return false;

        // extraterrestrial irradiance on a horizontal surface [W m-2]
        const double G0h = double(sunPosition.extraIrradianceHorizontal);

        // global and diffuse irradiance on a horizontal surface
        if (radSettings->getRealSkyAlgorithm() == RADIATION_REALSKY_TOTALTRANSMISSIVITY)
        {
            // atmospheric transmissivity (Gh / G0h)
            Gh = G0h * transmissivity;
            if (! radSettings->getRealSky())
                Gh = G0h * clearSkyTransmissivity;

            dH = splitDiffuseHorizontal(Gh, G0h);
        }
        else
        {
            Bhc = clearSkyBeamHorizontal(linke, sunPosition);
            Dhc = clearSkyDiffuseHorizontal(linke, sunPosition);
            Ghc = Dhc + Bhc;

            if (radSettings->getRealSky())
            {
                if (clearSkyTransmissivity <= 1e-6)
                    return false;

                // clear-sky index applied to the ESRA clear-sky global
                // ESRA = European Solar Radiation Atlas
                Gh = Ghc * transmissivity / clearSkyTransmissivity;

                // diffuse fraction from the clearness index Kt = Gh / G0h
                dH = splitDiffuseHorizontal(Gh, G0h);
            }
            else
            {
                Gh = Ghc;
                dH = Dhc;       // ESRA diffuse
            }
        }

        // shadowing
        if (! sunPosition.shadow && sunPosition.incidence > 0.)
        {
            Bh = Gh - dH;
        }
        else
        {
            Bh = 0;
            Gh = dH;    // approximation (portion of shadowed sky should be considered)
        }

        // inclined
        if (radPoint.slope == 0)
        {
            radPoint.beam = Bh;
            radPoint.diffuse = dH;
            radPoint.reflected = 0;
            radPoint.global = Gh;
        }
        else
        {
            if (! sunPosition.shadow && sunPosition.incidence > 0.)
                radPoint.beam = getBeamInclined(Bh, sunPosition);
            else
                radPoint.beam = 0;

            radPoint.diffuse = getDiffuseInclined_Muneer(Bh, dH, sunPosition, radPoint);
            radPoint.reflected = getReflectedIrradiance(Bh, dH, albedo, float(radPoint.slope));
            radPoint.global = radPoint.beam + radPoint.diffuse + radPoint.reflected;
        }

        return true;
    }


    /*!
     * \brief Solar noon as local time [s from midnight]
     * (midpoint between sunrise and sunset computed by SOLPOS)
     * Fallback: 12:00 clock time (polar day/night or computation failure)
     */
    double getSolarNoonLocalSeconds(const Crit3DRadiationSettings* radSettings,
                                    const Crit3DDate& localDate, const TradPoint& radPoint)
    {
        const double CLOCK_NOON = 12 * 3600;

        const double pressure = pressureFromAltitude(radPoint.height) * 0.01;

        TsunPosition sun;
        if (! computeSunPosition(float(radPoint.lon), float(radPoint.lat),
                                radSettings->gisSettings->timeZone,
                                localDate.year, localDate.month, localDate.day, 12, 0, 0,
                                float(TEMPERATURE_DEFAULT), float(pressure),
                                float(radPoint.aspect), float(radPoint.slope), sun))
            return CLOCK_NOON;

        if (isEqual(sun.rise, NODATA) || isEqual(sun.set, NODATA))
            return CLOCK_NOON;

        // valid sunrise/sunset within the day
        if (sun.rise < 0 || (sun.set > 24 * 3600) || sun.set <= sun.rise)
            return CLOCK_NOON;

        return 0.5 * (double(sun.rise) + double(sun.set));
    }


    int estimateTransmissivityWindow(const Crit3DRadiationSettings *radSettings, const gis::Crit3DPoint& myPoint,
                                     const Crit3DTime &myTime, const gis::Crit3DRasterGrid& myDem, int timeStepSecond)
    {
        const int MAX_STEPS = 23;           // [hours] 1 day
        const double MIN_ELEV = 3.0;        // [degree] low-sun filter
        const int MIN_WINDOW = 1;           // [hours] minimum transmissivity window

        TradPoint radPoint;
        TsunPosition sunPosition;
        Crit3DTime backwardTime;
        Crit3DTime forwardTime;

        // geometry
        radPoint.x = myPoint.utm.x;
        radPoint.y = myPoint.utm.y;

        if (myPoint.z != NODATA)
            radPoint.height = myPoint.z;
        else
            radPoint.height = (double)myDem.getValueFromXY(radPoint.x, radPoint.y);

        // default: horizontal
        radPoint.aspect = 180;
        radPoint.slope = 0;

        gis::getLatLonFromUtm(*(radSettings->gisSettings), radPoint.x, radPoint.y, &(radPoint.lat), &(radPoint.lon));

        double linke;
        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, myPoint);

        double albedo = readAlbedo(radSettings, myPoint);
        double clearSkyTransmissivity = radSettings->getClearSky();

        // local time (SOLPOS works in local time)
        Crit3DTime localTime = myTime;
        if (radSettings->gisSettings->isUTC)
        {
            localTime = myTime.addSeconds(radSettings->gisSettings->timeZone * 3600);
        }

        // solar noon
        Crit3DTime noonTime;
        noonTime.date = localTime.date;
        noonTime.time = int(std::round(getSolarNoonLocalSeconds(radSettings, localTime.date, radPoint)));
        if (radSettings->gisSettings->isUTC)
        {
            noonTime = noonTime.addSeconds(-radSettings->gisSettings->timeZone * 3600);
        }

        // clear sky radiation at solar noon
        double threshold = 50.;
        if (computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, noonTime, linke, albedo,
                                 clearSkyTransmissivity, clearSkyTransmissivity,
                                 sunPosition, radPoint, myDem))
        {
            // threshold: a quarter of potential radiation at noon
            threshold = std::max(50., radPoint.global * 0.25);
        }

        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime, linke, albedo,
                                  clearSkyTransmissivity, clearSkyTransmissivity,
                                  sunPosition, radPoint, myDem))
            return MIN_WINDOW;

        double sumPotentialRad = radPoint.global;

        int backwardTimeStep,forwardTimeStep;
        backwardTimeStep = forwardTimeStep = 0;

        int windowSteps = 1;
        backwardTime = forwardTime = myTime;

        while (sumPotentialRad < threshold && windowSteps < MAX_STEPS)
        {
            windowSteps += 2;

            backwardTimeStep -= timeStepSecond ;
            forwardTimeStep += timeStepSecond;
            backwardTime = myTime.addSeconds(backwardTimeStep);
            forwardTime = myTime.addSeconds(forwardTimeStep);

            computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, backwardTime,
                                      linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                      sunPosition, radPoint, myDem);
            // avoid noise
            if (sunPosition.elevationRefr > MIN_ELEV)
                sumPotentialRad += radPoint.global;

            computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, forwardTime,
                                      linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                      sunPosition, radPoint, myDem);
            // avoid noise
            if (sunPosition.elevationRefr > MIN_ELEV)
                sumPotentialRad += radPoint.global;
        }

        return windowSteps;
    }


    bool isGridPointComputable(const Crit3DRadiationSettings* radSettings, int row, int col,
                               const gis::Crit3DRasterGrid& dem, Crit3DRadiationMaps* radiationMaps)
    {
        if (gis::isOutOfGridRowCol(row, col, dem.header))
            return false;

        if (dem.value[row][col] == dem.header->flag)
            return false;

        const float latLonFlag = radiationMaps->latMap->header->flag;
        if ( isEqual(radiationMaps->latMap->value[row][col], latLonFlag)
             || isEqual(radiationMaps->lonMap->value[row][col], latLonFlag) )
            return false;

        const float slopeAspectFlag = radiationMaps->slopeMap->header->flag;
        float slope = readSlope(radSettings, radiationMaps->slopeMap, row, col);
        float aspect = readAspect(radSettings, radiationMaps->aspectMap, row, col);

        if ( isEqual(slope, slopeAspectFlag) || isEqual(slope, NODATA)
             || isEqual(aspect, slopeAspectFlag) || isEqual(aspect, NODATA) )
            return false;

        return true;
    }


    bool computeRadiationDemPoint(const Crit3DRadiationSettings* radSettings, Crit3DRadiationMaps* radiationMaps,
                                  const gis::Crit3DRasterGrid& dem, const Crit3DTime& myTime,
                                  int row, int col, double height)
    {
        // initialize
        const float flag = radiationMaps->globalRadiationMap->header->flag;
        radiationMaps->sunElevationMap->value[row][col] = flag;
        radiationMaps->globalRadiationMap->value[row][col] = flag;
        radiationMaps->beamRadiationMap->value[row][col] = flag;
        radiationMaps->diffuseRadiationMap->value[row][col] = flag;
        radiationMaps->reflectedRadiationMap->value[row][col] = flag;

        TradPoint radPoint;
        radPoint.height = height;
        dem.getXY(row, col, radPoint.x, radPoint.y);
        radPoint.lat = radiationMaps->latMap->value[row][col];
        radPoint.lon = radiationMaps->lonMap->value[row][col];
        radPoint.slope = readSlope(radSettings, radiationMaps->slopeMap, row, col);
        radPoint.aspect = readAspect(radSettings, radiationMaps->aspectMap, row, col);

        float linke;
        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, row, col);

        const float albedo = readAlbedo(radSettings, row, col);

        const float transmissivity = radiationMaps->transmissivityMap->value[row][col];

        TsunPosition sunPosition;
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime,
                                  linke, albedo, radSettings->getClearSky(), transmissivity,
                                  sunPosition, radPoint, dem))
            return false;

        // set values
        radiationMaps->sunElevationMap->value[row][col] = sunPosition.elevationRefr;
        radiationMaps->globalRadiationMap->value[row][col] = float(radPoint.global);
        radiationMaps->beamRadiationMap->value[row][col] = float(radPoint.beam);
        radiationMaps->diffuseRadiationMap->value[row][col] = float(radPoint.diffuse);
        radiationMaps->reflectedRadiationMap->value[row][col] = float(radPoint.reflected);

        return true;
    }


    bool computeRadiationPotentialRSunMeteoPoint(const Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& dem,
                                                 Crit3DMeteoPoint* myMeteoPoint, float slope, float aspect,
                                                 const Crit3DTime& myTime, TradPoint* radPoint)
    {
        radPoint->lat = myMeteoPoint->latitude;
        radPoint->lon = myMeteoPoint->longitude;
        radPoint->x = myMeteoPoint->point.utm.x;
        radPoint->y = myMeteoPoint->point.utm.y;
        radPoint->height = myMeteoPoint->point.z;
        radPoint->slope = slope;
        radPoint->aspect = aspect;

        gis::Crit3DPoint myPoint = myMeteoPoint->point;

        float linke;

        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, myPoint);

        float albedo = readAlbedo(radSettings, myPoint);

        TsunPosition sunPosition;
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime,
            linke, albedo, radSettings->getClearSky(), radSettings->getClearSky(), sunPosition, *radPoint, dem))
            return false;

        return true;
    }


    bool computeRadiationRSunMeteoPoint(const Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& dem,
                                        Crit3DMeteoPoint* myMeteoPoint, TradPoint& radPoint, const Crit3DTime& myTime)
    {
        radPoint.lat = myMeteoPoint->latitude;
        radPoint.lon = myMeteoPoint->longitude;
        radPoint.x = myMeteoPoint->point.utm.x;
        radPoint.y = myMeteoPoint->point.utm.y;
        radPoint.height = myMeteoPoint->point.z;
        radPoint.slope = readSlope(radSettings);
        radPoint.aspect = readAspect(radSettings);

        if (isEqual(radPoint.slope, NODATA) || isEqual(radPoint.aspect, NODATA))
            return false;

        gis::Crit3DPoint myPoint = myMeteoPoint->point;

        float linke;

        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, myPoint);

        float albedo = readAlbedo(radSettings, myPoint);

        float transmissivity = myMeteoPoint->getMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(), atmTransmissivity);

        TsunPosition sunPosition;
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime,
            linke, albedo, radSettings->getClearSky(), transmissivity, sunPosition, radPoint, dem))
            return false;

        return true;
    }


    bool computeRadiationDEM(const Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& dem,
                             Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myTime, bool isParallelComputing)
    {
        if (radSettings->getAlgorithm() != RADIATION_ALGORITHM_RSUN)
            return false;

        #pragma omp parallel for if (isParallelComputing)
        for (int row = 0; row < dem.header->nrRows; ++row)
        {
            for (int col = 0; col < dem.header->nrCols; ++col)
            {
                if (isGridPointComputable(radSettings, row, col, dem, radiationMaps))
                {
                    double height = double(dem.value[row][col]);
                    computeRadiationDemPoint(radSettings, radiationMaps, dem, myTime, row, col, height);
                }
            }
        }

        updateRadiationMaps(radiationMaps, myTime);

        return true;
    }


    void updateRadiationMaps(Crit3DRadiationMaps* radiationMaps, const Crit3DTime &myTime)
    {
        gis::updateMinMaxRasterGrid(radiationMaps->sunElevationMap);
        gis::updateMinMaxRasterGrid(radiationMaps->transmissivityMap);
        gis::updateMinMaxRasterGrid(radiationMaps->beamRadiationMap);
        gis::updateMinMaxRasterGrid(radiationMaps->diffuseRadiationMap);
        gis::updateMinMaxRasterGrid(radiationMaps->reflectedRadiationMap);
        gis::updateMinMaxRasterGrid(radiationMaps->globalRadiationMap);

        radiationMaps->sunElevationMap->setMapTime(myTime);
        radiationMaps->transmissivityMap->setMapTime(myTime);
        radiationMaps->beamRadiationMap->setMapTime(myTime);
        radiationMaps->diffuseRadiationMap->setMapTime(myTime);
        radiationMaps->reflectedRadiationMap->setMapTime(myTime);
        radiationMaps->globalRadiationMap->setMapTime(myTime);

        radiationMaps->setComputed(true);
    }


    bool computeSunPosition(float lon, float lat, int timeZone,
                            int myYear,int myMonth, int myDay,
                            int myHour, int myMinute, int mySecond,
                            float temp, float pressure, float aspect, float slope, TsunPosition &sunPosition)
    {
        //float etrTilt;              /*!<  Extraterrestrial (top-of-atmosphere) global irradiance on a tilted surface (W m-2) */
        //float cosZen;               /*!<  Cosine of refraction corrected solar zenith angle */
        //float zenRef;               /*!<  Solar zenith angle, deg. from zenith, refracted */
        float sunCosIncidenceCompl;     /*!<  cosine of (90 - incidence) */
        float sunRiseMinutes;           /*!<  sunrise time [minutes from midnight] */
        float sunSetMinutes;            /*!<  sunset time [minutes from midnight] */

        SolPosData solarPosition;

        int chk = RSUN_compute_solar_position(solarPosition, lon, lat, timeZone, myYear, myMonth,
                                              myDay, myHour, myMinute, mySecond, temp, pressure, aspect, slope,
                                              float(SBWID), float(SBRAD), float(SBSKY));
        if (chk > 0)
        {
            // todo: report error
            return false;
        }

        sunPosition.relOptAirMass       = solarPosition.amass;
        sunPosition.relOptAirMassCorr   = solarPosition.ampress;
        sunPosition.azimuth             = solarPosition.azim;
        sunPosition.elevation           = solarPosition.elevetr;
        sunPosition.elevationRefr       = solarPosition.elevref;
        sunPosition.extraIrradianceHorizontal   = solarPosition.etr;
        sunPosition.extraIrradianceNormal		= solarPosition.etrn;
        sunCosIncidenceCompl            = solarPosition.cosinc;
        sunRiseMinutes                  = solarPosition.sretr;
        sunSetMinutes                   = solarPosition.ssetr;

        // Solar elevation angle on Panel [deg]
        const float cosInc = std::clamp(sunCosIncidenceCompl, -1.f, 1.f);
        sunPosition.incidence = float(std::max(0., RAD_TO_DEG * std::asin(double(cosInc))));

        sunPosition.rise = sunRiseMinutes * 60.f;
        sunPosition.set = sunSetMinutes * 60.f;

        return true;
    }


    // old code (snow model)
    bool computeRadiationPointBrooks(Crit3DRadiationSettings* radSettings, TradPoint* radPoint, Crit3DDate* myDate,
                                     float currentSolarTime, float clearSkyTransmissivity, float transmissivity,
                                     float albedo = 0.2f)
    {
        const int myDoy = getDoyFromDate(*myDate);

        // local copies: do not modify the caller's radPoint
        const double latRad = radPoint->lat * DEG_TO_RAD;
        const double aspectRad = radPoint->aspect * DEG_TO_RAD;
        const double slopeRad = radPoint->slope * DEG_TO_RAD;

        const double timeAdjustment = (279.575 + 0.9856 * myDoy) * DEG_TO_RAD;     // [rad]

        // equation of time [hours]
        const double timeEq = (-104.7 * sin(timeAdjustment) + 596.2 * sin(2 * timeAdjustment)
                               + 4.3 * sin(3 * timeAdjustment) - 12.7 * sin(4 * timeAdjustment)
                               - 429.3 * cos(timeAdjustment) - 2 * cos(2 * timeAdjustment)
                               + 19.3 * cos(3 * timeAdjustment)) / 3600.0;

        const double solarDeclination = 0.4102 * sin(2.0 * PI / 365.0 * (myDoy - 80));

        // longitude correction [hours]: positive for locations west of the standard meridian
        const double correctionLong = ((radSettings->gisSettings->timeZone * 15) - radPoint->lon) / 15.0;

        double solarTime = currentSolarTime - correctionLong + timeEq;
        if (solarTime > 24)
            solarTime -= 24;
        else if (solarTime < 0)
            solarTime += 24;

        double sinElev = sin(latRad) * sin(solarDeclination)
                         + cos(latRad) * cos(solarDeclination) * cos((PI / 12) * (solarTime - 12));
        sinElev = std::clamp(sinElev, -1.0, 1.0);

        // sun below the horizon
        if (sinElev <= 1e-6)
        {
            radPoint->beam = 0;
            radPoint->diffuse = 0;
            radPoint->reflected = 0;
            radPoint->global = 0;
            return true;
        }

        const double elevationAngle = asin(sinElev);
        const double cosElev = std::max(cos(elevationAngle), 1e-6);

        // extraterrestrial irradiance on a horizontal surface [W m-2]
        const double extraTerrestrialRad = SOLAR_CONSTANT * sinElev;

        // azimuth from north, clockwise
        const double cosLat = std::max(cos(latRad), 1e-6);
        const double cosAzSouth = std::clamp((sinElev * sin(latRad) - sin(solarDeclination)) / (cosElev * cosLat),
                                             -1.0, 1.0);
        const double azimuthSouth = acos(cosAzSouth);
        const double azimuthNorth = (solarTime > 12) ? PI + azimuthSouth : PI - azimuthSouth;

        // sine of the sun altitude over the inclined surface
        const double sinIncidenceCompl = sin(slopeRad) * cosElev * cos(azimuthNorth - aspectRad)
                                         + cos(slopeRad) * sinElev;

        // atmospheric transmissivity G / G0h
        double Tt = clearSkyTransmissivity;
        if (radSettings->getRealSky() && ! isEqual(transmissivity, NODATA))
            Tt = transmissivity;

        // diffuse fraction (Erbs)
        Tt = std::clamp(Tt, 0.0, 1.0);
        const double td = Tt * diffuseFractionErbs(Tt);
        const double coeffBH = std::max(0., Tt - td);

        const double radDiffuseHor = extraTerrestrialRad * td;

        double radBeam, radDiffuse, radReflected;
        if (radPoint->slope == 0)
        {
            radBeam = extraTerrestrialRad * coeffBH;
            radDiffuse = radDiffuseHor;
            radReflected = 0;
        }
        else
        {
            radBeam = extraTerrestrialRad * coeffBH * std::max(0., sinIncidenceCompl) / sinElev;
            radDiffuse = radDiffuseHor;     // isotropic simplification, as in the original
            // Muneer (1997)
            radReflected = extraTerrestrialRad * Tt * albedo * (1.0 - cos(slopeRad)) / 2.0;
        }

        radPoint->beam = radBeam;
        radPoint->diffuse = radDiffuse;
        radPoint->reflected = radReflected;
        radPoint->global = radDiffuse + radBeam + radReflected;

        return true;
    }


    bool computeRadiationOutputPoints(Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& dem,
                                         Crit3DRadiationMaps* radiationMaps, std::vector<gis::Crit3DOutputPoint> &outputPoints,
                                         const Crit3DTime& myTime)
    {
        if (radSettings->getAlgorithm() != RADIATION_ALGORITHM_RSUN)
            return false;

        int row, col;
        for (unsigned int i = 0; i < outputPoints.size(); i++)
        {
            if (outputPoints[i].active)
            {
                dem.getRowCol(outputPoints[i].utm.x, outputPoints[i].utm.y, row, col);
                if(isGridPointComputable(radSettings, row, col, dem, radiationMaps))
                {
                    if (! computeRadiationDemPoint(radSettings, radiationMaps, dem, myTime, row, col, outputPoints[i].z))
                        return false;
                }
            }
        }

        updateRadiationMaps(radiationMaps, myTime);

        return true;
    }


    float computePointTransmissivity(const Crit3DRadiationSettings *radSettings, const gis::Crit3DPoint& point, const Crit3DTime myTime,
                                     float* measuredRad, int windowWidth, int timeStepSecond, const gis::Crit3DRasterGrid& dem)
    {
        if (windowWidth % 2 != 1)
            return NODATA;

        int intervalCenter = (windowWidth - 1) / 2;

        if (isEqual(measuredRad[intervalCenter], NODATA))
            return NODATA;

        // assign topographic height and coordinates
        TradPoint radPoint;
        radPoint.x = point.utm.x;
        radPoint.y = point.utm.y;
        radPoint.height = (! isEqual(point.z,NODATA))
                              ? point.z
                              : double(gis::getValueFromXY(dem, radPoint.x, radPoint.y));

        // suppose horizontal
        radPoint.aspect = 180.;
        radPoint.slope = 0.;

        gis::getLatLonFromUtm(*(radSettings->gisSettings), point.utm.x, point.utm.y,
                              &(radPoint.lat), &(radPoint.lon));

        const float linke = (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
                          ? radSettings->getLinke(myTime.date.month-1)
                          : readLinke(radSettings, point);

        const float albedo = readAlbedo(radSettings, point);

        const float clearSkyTransmissivity = radSettings->getClearSky();

        // center value
        double sumGHI = measuredRad[intervalCenter];

        // center value
        TsunPosition sunPosition;
        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime, linke, albedo,
                             clearSkyTransmissivity, clearSkyTransmissivity,
                             sunPosition, radPoint, dem);

        double sumClearSky = radPoint.global;

        // -----------------------------
        // Window integration
        // -----------------------------
        Crit3DTime tBack = myTime;
        Crit3DTime tFwd = myTime;

        int dtBack = 0;
        int dtFwd = 0;

        for (int i = intervalCenter - 1; i >= 0; --i)
        {
            dtBack -= timeStepSecond;
            dtFwd += timeStepSecond;

            tBack = myTime.addSeconds(dtBack);
            tFwd  = myTime.addSeconds(dtFwd);

            // backward
            if (! isEqual(measuredRad[i], NODATA))
            {
                sumGHI += measuredRad[i];

                computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, tBack, linke, albedo,
                                     clearSkyTransmissivity, clearSkyTransmissivity,
                                     sunPosition, radPoint, dem);

                sumClearSky += radPoint.global;
            }

            // forward
            int j = windowWidth - i - 1;
            if (! isEqual(measuredRad[j], NODATA))
            {
                sumGHI += measuredRad[j];

                computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, tFwd, linke, albedo,
                                     clearSkyTransmissivity, clearSkyTransmissivity,
                                     sunPosition, radPoint, dem);

                sumClearSky += radPoint.global;
            }
        }

        // -----------------------------
        // Clearness index (PVGIS style)
        // -----------------------------
        double Kt = (sumClearSky > 1e-6) ? (sumGHI / sumClearSky) : 0.0;

        // Physical limits
        Kt = std::clamp(Kt, 0.0, 1.2);

        // -----------------------------
        // Convert to transmissivity
        // -----------------------------
        const double transmissivity = Kt * clearSkyTransmissivity;

        return float(transmissivity);
    }
}

