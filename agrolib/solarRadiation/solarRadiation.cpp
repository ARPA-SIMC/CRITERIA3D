/*!
    \name Solar Radiation
    \copyright 2011 Gabriele Antolini, Fausto Tomei
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
    this->clear();
}


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


bool Crit3DRadiationMaps::getComputed()
{
    return isComputed;
}

void Crit3DRadiationMaps::setComputed(bool value)
{
    isComputed = value;
}

namespace radiation
{
/*
    double linkeMountain[13] = {1.5, 1.6, 1.8, 1.9, 2.0, 2.3, 2.3, 2.3, 2.1, 1.8, 1.6, 1.5, 1.9};
    double linkeRural[13] = {2.1, 2.2, 2.5, 2.9, 3.2, 3.4, 3.5, 3.3, 2.9, 2.6, 2.3, 2.2, 2.75};
    double linkeCity[13] = {3.1, 3.2, 3.5, 4.0, 4.2, 4.3, 4.4, 4.3, 4.0, 3.6, 3.3, 3.1, 3.75};
    double linkeIndustrial[13] = {4.1, 4.3, 4.7, 5.3, 5.5, 5.7, 5.8, 5.7, 5.3, 4.9, 4.5, 4.2, 5.0};
*/

    float readAlbedo(Crit3DRadiationSettings* radSettings)
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

    float readAlbedo(Crit3DRadiationSettings* radSettings, int row, int col)
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

    float readAlbedo(Crit3DRadiationSettings* radSettings, const gis::Crit3DPoint& point)
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


    float readLinke(Crit3DRadiationSettings* radSettings)
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


    float readLinke(Crit3DRadiationSettings* radSettings, int row, int col)
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


    float readLinke(Crit3DRadiationSettings* radSettings, const gis::Crit3DPoint& point)
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
                radSettings->getLinke(point);
        }
        return output;
    }

    float readAspect(Crit3DRadiationSettings* radSettings)
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

    float readAspect(Crit3DRadiationSettings* radSettings, gis::Crit3DRasterGrid* aspectMap, int row,int col)
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

    float readSlope(Crit3DRadiationSettings* radSettings)
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

    float readSlope(Crit3DRadiationSettings* radSettings, gis::Crit3DRasterGrid* slopeMap, int row, int col)
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
            A0 = 0.002 / Trd;

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
     * \brief Diffuse irradiance on an inclined surface (Muneer, 1990)               [W m-2]
     * \param beamIrradianceHor
     * \param diffuseIrradianceHor
     * \param sunPosition a pointer to a TsunPosition
     * \param radPoint
     */
    float clearSkyDiffuseInclined_OLD(float beamIrradianceHor, float diffuseIrradianceHor,
                                  const TsunPosition& sunPosition, const TradPoint& radPoint)
    {
        //Bh                     beam irradiance on a horizontal surface                                     [W m-2]
        //Dh                     diffuse irradiance on a horizontal surface

        double cosSlope, sinSlope;
        double slopeRad, aspectRad, elevRad, azimRad;
        double sinElev;    
        double Fg, r_sky, Fx, Aln;
        double n;
        sinElev = MAXVALUE(getSinDecimalDegree(sunPosition.elevation), 1e-6);
        cosSlope = getCosDecimalDegree(radPoint.slope);
        sinSlope = getSinDecimalDegree(radPoint.slope);
        slopeRad = radPoint.slope * DEG_TO_RAD;
        aspectRad = radPoint.aspect * DEG_TO_RAD;
        elevRad = sunPosition.elevation * DEG_TO_RAD;
        azimRad = sunPosition.azimuth * DEG_TO_RAD;

        /*!< amount of beam irradiance available [-] */
        double denom = sunPosition.extraIrradianceNormal * sinElev;
        double Kb = (denom > 1e-6) ? beamIrradianceHor / denom : 0.0;
        Kb = std::max(0.0, Kb);

        Fg = sinSlope - slopeRad * cosSlope - PI * getSinDecimalDegree(radPoint.slope * 0.5)
                                                 * getSinDecimalDegree(radPoint.slope * 0.5);
        r_sky = (1.0 + cosSlope) / 2.0;
        if ((sunPosition.shadow || (sunPosition.incidence * DEG_TO_RAD) <= 0.1) && elevRad >= 0.0)
        {
            n = 0.252271;
            Fx = r_sky + Fg * n;
        }
        else
        {
            n = 0.00263 - Kb * (0.712 + 0.6883 * Kb);
            // attenzione: crea discontinuita'
            if (elevRad >= 0.05)
                Fx = (n * Fg + r_sky) * (1 - Kb) + Kb * getSinDecimalDegree(sunPosition.incidence) / sinElev;
            else
            {
                Aln = azimRad - aspectRad;
                if (Aln > (2.0 * PI))
                    Aln -= (float)(2.0 * PI);
                else if (Aln < 0.0)
                    Aln += (float)(2.0 * PI);
                Fx = (n * Fg + r_sky) * (1.0 - Kb) + Kb * sinSlope * getCosDecimalDegree(float(Aln * RAD_TO_DEG)) / (0.1 - 0.008 * elevRad);
            }
        }

        return float(diffuseIrradianceHor * Fx);
    }


    /*!
     * \brief Diffuse irradiance on an inclined surface (Muneer, 1990)  [W m-2]
     * Fx = (1 - Kb) × F_iso + Kb × F_aniso
     * \param Bh   beam irradiance on a horizontal surface              [W m-2]
     * \param Dh   diffuse irradiance on a horizontal surface           [W m-2]
     */
    double getDiffuseInclined_Muneer(double Bh, double Dh,
                                    const TsunPosition& sun,
                                    const TradPoint& p)
    {
        if (sun.elevationRefr < 1e-6)
            return 0.0f;

        double slopeRad = p.slope * DEG_TO_RAD;
        double aspectRad = p.aspect * DEG_TO_RAD;
        double elevationRad = sun.elevationRefr * DEG_TO_RAD;

        double sinElev = std::max(sin(elevationRad), 1e-6);
        double sinSlope = std::sin(slopeRad);
        double cosSlope = std::cos(slopeRad);

        // Amount of beam irradiance available [-]
        // allow slight physical overshoot (1.2)
        double Kb = Bh / (sun.extraIrradianceNormal * sinElev);
        Kb = std::clamp(Kb, 0.0, 1.2);

        double r_sky = (1.0 + cosSlope) / 2.0;

        // Slope-dependent geometric term [-]
        double Fg = sinSlope - slopeRad * cosSlope
                    - PI * std::pow(std::sin(p.slope * 0.5 * DEG_TO_RAD), 2);

        bool lowSun = (sun.elevationRefr < 3.0);    // degree, smooth proxy

        double Fx;                                  // diffuse irradiance tilt factor [-]
        if (sun.shadow || sun.incidence <= 0.1)
        {
            Fx = r_sky + Fg * 0.252271;
        }
        else
        {
            double n = 0.00263 - Kb * (0.712 + 0.6883 * Kb);
            double termBeam = std::sin(sun.incidence * DEG_TO_RAD) / sinElev;

            if (! lowSun)
            {
                Fx = (n * Fg + r_sky) * (1.0 - Kb) + Kb * termBeam;
            }
            else
            {
                double azimuthLocalDiff = std::fmod(sun.azimuth * DEG_TO_RAD - aspectRad + 2*PI, 2*PI);

                double denom2 = std::max(0.05, 0.1 - 0.008 * elevationRad);
                Fx = (n * Fg + r_sky) * (1.0 - Kb) + Kb * sinSlope * std::cos(azimuthLocalDiff) / denom2;
            }
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
        bool output = false ;
        if ((riseTime != NODATA) && (setTime != NODATA) && (sunElevation != NODATA))
            output =  ((myTime >= riseTime) && (myTime <= setTime) && (sunElevation > 0));
        return output;
    }


    bool computeShadow(const TradPoint& radPoint, const TsunPosition& sunPosition, const gis::Crit3DRasterGrid& myDem)
    {
        /* INPUT
        supponiamo di avere gia' controllato che siamo dopo l'alba e prima del tramonto
        inizializzazione a sole visibile
        */

        double x0 = radPoint.x;
        double y0 = radPoint.y;
        double z0 = radPoint.height;

        const double cellSize = myDem.header->cellSize;

        double sinAz = std::sin(sunPosition.azimuth * DEG_TO_RAD);
        double cosAz = std::cos(sunPosition.azimuth * DEG_TO_RAD);

        double sinElev = std::sin(sunPosition.elevationRefr * DEG_TO_RAD);
        double cosElev = std::cos(sunPosition.elevationRefr * DEG_TO_RAD);

        // evita divisione per zero
        double tgElev = sinElev / std::max(cosElev, 1e-6);

        double stepX = SHADOW_FACTOR * sinAz * cellSize;        // [m]
        double stepY = SHADOW_FACTOR * cosAz * cellSize;        // [m]
        double stepZ = SHADOW_FACTOR * cellSize * tgElev;       // [m]

        double maxDeltaH = cellSize * SHADOW_FACTOR * 2.0;

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

            double x = x0 + stepX * stepCount;
            double y = y0 + stepY * stepCount;
            double z = z0 + stepZ * stepCount;

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


    void separateTransmissivity_old(double clearSkyTransmissivity, double transmissivity, double &td, double &Tt)
    {
        double maximumDiffuseTransmissivity;

        // in attesa di studi mirati (Bristow and Campbell, 1985)
        maximumDiffuseTransmissivity = 0.6 / (clearSkyTransmissivity - 0.4);

        Tt = std::max(std::min(transmissivity, clearSkyTransmissivity), 1e-5);

        td = Tt * (1.0 - exp(maximumDiffuseTransmissivity - (maximumDiffuseTransmissivity * clearSkyTransmissivity) / Tt));

        // FT 0.12 stimato da Settefonti agosto 2007
        if (Tt > 0.6)
            td = std::max(td, 0.12);
    }


    // S.A.Erbs (1982) Peter Reindl (1990)
    void separateTransmissivity_Erbs_Reindl(double clearSkyTransmissivity, double transmissivity,
                                            double sunElevationDeg, double &td, double &Tt)
    {
        // ----------------------------
        // 1. Safety bounds
        // ----------------------------
        Tt = std::clamp(transmissivity, 1e-6, clearSkyTransmissivity);

        if (clearSkyTransmissivity <= 1e-6)
        {
            td = 0.0;
            return;
        }

        // ----------------------------
        // 2. Clearness index
        // ----------------------------
        double Kt = Tt / clearSkyTransmissivity;
        Kt = std::clamp(Kt, 0.0, 1.2);

        double elevRad = sunElevationDeg * DEG_TO_RAD;
        double sinElev = std::max(std::sin(elevRad), 1e-4);

        // ----------------------------
        // 3. Erbs correlation (daily/hourly adapted)
        // ----------------------------
        double Kd;

        if (Kt <= 0.22)
        {
            Kd = 1.0 - 0.09 * Kt;
        }
        else if (Kt <= 0.80)
        {
            Kd = 0.9511 - 0.1604 * Kt + 4.388 * Kt * Kt
                 - 16.638 * Kt * Kt * Kt
                 + 12.336 * Kt * Kt * Kt * Kt;
        }
        else
        {
            Kd = 0.165;
        }

        // ----------------------------
        // 4. Reindl correction (sun elevation dependency)
        // ----------------------------
        double Kd_reindl = Kd;

        if (sunElevationDeg > 0.0)
        {
            Kd_reindl = Kd + (0.10 + 0.12 * sunElevationDeg / 90.0) * (1.0 - std::exp(-1.0 / sinElev));
        }

        Kd_reindl = std::clamp(Kd_reindl, 0.0, 1.0);

        // ----------------------------
        // 5. Final split
        // ----------------------------
        td = Tt * Kd_reindl;
    }


    bool computeRadiationRsun(Crit3DRadiationSettings* radSettings, double temperature, const Crit3DTime& myTime,
                              double linke, double albedo, double clearSkyTransmissivity, double transmissivity,
                              TsunPosition &sunPosition, TradPoint& radPoint, const gis::Crit3DRasterGrid& dem)
    {
        int myYear, myMonth, myDay;
        int myHour, myMinute, mySecond;
        double Bhc, Bh;
        double Dhc, dH;
        double Ghc, Gh;
        double globalTransmittance;  /*!<   real sky global irradiation coefficient (global transmittance) */
        double diffuseTransmittance; /*!<   real sky radPoint.diffuse irradiation coefficient (radPoint.diffuse transmittance) */
        double dhsOverGhs;           /*!<  ratio horizontal radPoint.diffuse over horizontal global */
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
        // should be passed
        if (isEqual(temperature, NODATA))
            temperature = TEMPERATURE_DEFAULT;

        /*! Surface pressure at sea level (millibars = hPa) (used for refraction correction and optical air mass) */
        double pressure = pressureFromAltitude(radPoint.height) * 0.01;

        /*! Sun position */
        if (! computeSunPosition(float(radPoint.lon), float(radPoint.lat), radSettings->gisSettings->timeZone,
                                myYear, myMonth, myDay, myHour, myMinute, mySecond,
                                (float)temperature, (float)pressure, (float)radPoint.aspect, (float)radPoint.slope, sunPosition))
            return false;

        /*! Shadowing */
        isPointIlluminated = isIlluminated(float(localTime.time), sunPosition.rise, sunPosition.set, sunPosition.elevationRefr);
        if (radSettings->getShadowing())
        {
            sunPosition.shadow = ! isPointIlluminated;

            if (isPointIlluminated && dem.isLoaded && ! gis::isOutOfGridXY(radPoint.x, radPoint.y, dem.header))
                sunPosition.shadow = computeShadow(radPoint, sunPosition, dem);
        }

        /*! Radiation */
        if (! isPointIlluminated)
        {
            radPoint.beam = 0;
            radPoint.diffuse = 0;
            radPoint.reflected = 0;
            radPoint.global = 0;

            return true;
        }

        if (radSettings->getRealSky() && transmissivity == NODATA)
            return false;

        // real sky horizontal
        if (radSettings->getRealSkyAlgorithm() == RADIATION_REALSKY_TOTALTRANSMISSIVITY)
        {
            if (! radSettings->getRealSky())
                transmissivity = clearSkyTransmissivity;

            separateTransmissivity_Erbs_Reindl (clearSkyTransmissivity, transmissivity, sunPosition.elevationRefr,
                                               diffuseTransmittance, globalTransmittance);

            Gh = sunPosition.extraIrradianceHorizontal * transmissivity;
            dH = sunPosition.extraIrradianceHorizontal * diffuseTransmittance;
        }
        else
        {
            Bhc = clearSkyBeamHorizontal(linke, sunPosition);
            Dhc = clearSkyDiffuseHorizontal(linke, sunPosition);
            Ghc = Dhc + Bhc;

            if (radSettings->getRealSky())
            {
                Gh = Ghc * transmissivity / clearSkyTransmissivity;

                // todo: trovare un metodo migliore (che non usi la clearSkyTransmissivity, non coerente con l'utilizzo di Linke)
                separateTransmissivity_Erbs_Reindl (clearSkyTransmissivity, transmissivity, sunPosition.elevationRefr, diffuseTransmittance, globalTransmittance);
                dhsOverGhs = diffuseTransmittance / globalTransmittance;
                dH = dhsOverGhs * Gh;
            }
            else
            {
                Gh = Ghc;
                dH = Dhc;
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


    int estimateTransmissivityWindow(Crit3DRadiationSettings* radSettings, const gis::Crit3DPoint& myPoint,
                                     const Crit3DTime &myTime, const gis::Crit3DRasterGrid& myDem, int timeStepSecond)
    {
        const int MAX_STEPS = 23;           // 1 day
        const double MIN_ELEV = 3.0;        // [degree] low-sun filter

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

        // noon
        Crit3DTime noonTime;
        noonTime.date = myTime.date;
        noonTime.time = 12*3600;
        if (radSettings->gisSettings->isUTC)
        {
            noonTime = noonTime.addSeconds(-radSettings->gisSettings->timeZone * 3600);
        }

        // clear sky radiation at noon
        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, noonTime, linke, albedo,
                                  clearSkyTransmissivity, clearSkyTransmissivity, sunPosition, radPoint, myDem);

        // threshold: a quarter of potential radiation at noon
        double threshold = std::max(50.f, float(radPoint.global * 0.25));

        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime, linke, albedo,
                                  clearSkyTransmissivity, clearSkyTransmissivity, sunPosition, radPoint, myDem);

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


    bool isGridPointComputable(Crit3DRadiationSettings* radSettings, int row, int col,
                               const gis::Crit3DRasterGrid& dem, Crit3DRadiationMaps* radiationMaps)
    {
        if (gis::isOutOfGridRowCol(row, col, dem.header))
            return false;

        if (dem.value[row][col] == dem.header->flag)
            return false;

        if ((radiationMaps->latMap->value[row][col] == radiationMaps->latMap->header->flag)
            || (radiationMaps->lonMap->value[row][col] == radiationMaps->lonMap->header->flag))
            return false;

        float slope = readSlope(radSettings, radiationMaps->slopeMap, row, col);
        float aspect = readAspect(radSettings, radiationMaps->aspectMap, row, col);
        if ((slope == NODATA) || (aspect == NODATA)) return false;

        return true;
    }


    bool computeRadiationDemPoint(Crit3DRadiationSettings* radSettings, Crit3DRadiationMaps* radiationMaps,
                                  const gis::Crit3DRasterGrid& dem, const Crit3DTime& myTime, int row, int col, double height)
    {
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

        float albedo = readAlbedo(radSettings, row, col);

        float transmissivity = radiationMaps->transmissivityMap->value[row][col];

        TsunPosition sunPosition;
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, myTime,
                                  linke, albedo, radSettings->getClearSky(), transmissivity, sunPosition, radPoint, dem))
            return false;

        radiationMaps->sunElevationMap->value[row][col] = sunPosition.elevationRefr;
        radiationMaps->globalRadiationMap->value[row][col] = float(radPoint.global);
        radiationMaps->beamRadiationMap->value[row][col] = float(radPoint.beam);
        radiationMaps->diffuseRadiationMap->value[row][col] = float(radPoint.diffuse);
        radiationMaps->reflectedRadiationMap->value[row][col] = float(radPoint.reflected);

        return true;
    }


    bool computeRadiationPotentialRSunMeteoPoint(Crit3DRadiationSettings* radSettings, const gis::Crit3DRasterGrid& dem,
                              Crit3DMeteoPoint* myMeteoPoint, float slope, float aspect, const Crit3DTime& myTime, TradPoint* radPoint)
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


    bool computeRadiationRSunMeteoPoint(Crit3DRadiationSettings* radSettings, const gis::Crit3DRasterGrid& dem,
                              Crit3DMeteoPoint* myMeteoPoint, TradPoint radPoint, const Crit3DTime& myTime)
    {
        radPoint.lat = myMeteoPoint->latitude;
        radPoint.lon = myMeteoPoint->longitude;
        radPoint.slope = readSlope(radSettings);
        radPoint.aspect = readAspect(radSettings);

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


    bool computeRadiationDEM(Crit3DRadiationSettings* radSettings, const gis::Crit3DRasterGrid& dem,
                              Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myTime, bool isParallelComputing)
    {
        if (radSettings->getAlgorithm() != RADIATION_ALGORITHM_RSUN)
            return false;

        bool isOk = true;

        #pragma omp parallel for if (isParallelComputing) shared(isOk)
        for (int row = 0; row < dem.header->nrRows; row++ )
        {
            if (! isOk) continue;

            for (int col = 0; col < dem.header->nrCols; col++)
            {
                if (! isOk) continue;

                float height = dem.value[row][col];
                if (! isEqual(height, dem.header->flag))
                {
                    if (! computeRadiationDemPoint(radSettings, radiationMaps, dem, myTime, row, col, height))
                        isOk = false;
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

        sunPosition.incidence = float(std::max(0., RAD_TO_DEG * ((PI / 2.0) - acos(sunCosIncidenceCompl))));
        sunPosition.rise = sunRiseMinutes * 60.f;
        sunPosition.set = sunSetMinutes * 60.f;

        return true;
    }


    bool computeRadiationPointBrooks(Crit3DRadiationSettings* radSettings, TradPoint* radPoint, Crit3DDate* myDate,
                                     float currentSolarTime, float clearSkyTransmissivity, float transmissivity)
    {
        int myDoy;
        double timeAdjustment;   /*!<  hours */
        double timeEq;           /*!<  hours */
        double solarTime;        /*!<  hours */
        double correctionLong;   /*!<  hours */

        double solarDeclination; /*!<  radians */
        double elevationAngle;   /*!<  radians */
        double incidenceAngle;   /*!<  radians */
        double azimuthSouth;     /*!<  radians */
        double azimuthNorth;     /*!<  radians */

        double coeffBH; /*!<  [-] */

        double extraTerrestrialRad; /*!<  [W/m2] */
        double radDiffuse;          /*!<  [W/m2] */
        double radBeam;             /*!<  [W/m2] */
        double radReflected;        /*!<  [W/m2] */
        double radTotal;            /*!<  [W/m2] */

        myDoy = getDoyFromDate(*myDate);

        /*! conversione in radianti per il calcolo */
        radPoint->aspect *= DEG_TO_RAD;
        radPoint->lat *= DEG_TO_RAD;

        timeAdjustment = (279.575 + 0.986 * myDoy) * DEG_TO_RAD;

        timeEq = (-104.7 * sin(timeAdjustment) + 596.2 * sin(2 * timeAdjustment)
                  + 4.3 * sin(3 * timeAdjustment) - 12.7 * sin(4 * timeAdjustment)
                  - 429.3 * cos(timeAdjustment) - 2 * cos(2 * timeAdjustment)
                  + 19.3 * cos(3 * timeAdjustment)) / 3600.0 ;

        solarDeclination = 0.4102 * sin(2.0 * PI / 365.0 * (myDoy - 80));

        // controllare i segni:
        correctionLong = ((radSettings->gisSettings->timeZone * 15) - radPoint->lon) / 15.0;

        solarTime = currentSolarTime - correctionLong + timeEq;
        if (solarTime > 24)
        {
            solarTime -= 24;
        }
        else if (solarTime < 0)
        {
            solarTime += 24;
        }

        elevationAngle = asin(sin(radPoint->lat) * sin(solarDeclination)
                                 + cos(radPoint->lat) * cos(solarDeclination)
                                 * cos((PI / 12) * (solarTime - 12)));

        extraTerrestrialRad = std::max(0., SOLAR_CONSTANT * sin(elevationAngle));

        azimuthSouth = acos((sin(elevationAngle)
                                * sin(radPoint->lat) - sin(solarDeclination))
                               / (cos(elevationAngle) * cos(radPoint->lat)));

        azimuthNorth = (solarTime>12) ? PI + azimuthSouth : PI - azimuthSouth;
        incidenceAngle = std::max(0., asin(getSinDecimalDegree(float(radPoint->slope)) *
                                        cos(elevationAngle) * cos(azimuthNorth - float(radPoint->aspect))
                                        + getCosDecimalDegree(float(radPoint->slope)) * sin(elevationAngle)));

        double Tt = clearSkyTransmissivity;
        double td = 0.1f;

        if (radSettings->getRealSky())
        {
            if (transmissivity != NODATA)
            {
                separateTransmissivity_old(clearSkyTransmissivity, transmissivity, td, Tt);
            }
        }

        coeffBH = std::max(0., Tt - td);

        radDiffuse = extraTerrestrialRad * td;

        if (radSettings->getTilt() == 0)
        {
            radBeam = extraTerrestrialRad * coeffBH;
            radReflected = 0;
        }
        else
        {
            radBeam = extraTerrestrialRad * coeffBH * std::max(0., sin(incidenceAngle) / sin(elevationAngle));
            //aggiungere Snow albedo!
            //Muneer 1997
            radReflected = extraTerrestrialRad * Tt * 0.2 * (1.0 - getCosDecimalDegree(float(radPoint->slope))) / 2.0;
        }

        radTotal = radDiffuse + radBeam + radReflected;
        radPoint->global = radTotal;
        radPoint->beam = radBeam;
        radPoint->diffuse = radDiffuse;
        radPoint->reflected = radReflected;

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


    float computePointTransmissivity(Crit3DRadiationSettings* radSettings, const gis::Crit3DPoint& point, const Crit3DTime myTime,
                                     float* measuredRad, int windowWidth, int timeStepSecond, const gis::Crit3DRasterGrid& dem)
    {
        if (windowWidth % 2 != 1)
            return NODATA;

        int intervalCenter = (windowWidth - 1) / 2;

        if (measuredRad[intervalCenter] == NODATA)
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

        float linke = (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
                          ? radSettings->getLinke(myTime.date.month-1)
                          : readLinke(radSettings, point);

        float albedo = readAlbedo(radSettings, point);

        float clearSkyTransmissivity = radSettings->getClearSky();

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
        double transmissivity = Kt * clearSkyTransmissivity;

        return float(transmissivity);
    }
}

