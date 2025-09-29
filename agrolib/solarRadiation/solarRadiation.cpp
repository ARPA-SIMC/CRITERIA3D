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

#include <math.h>
#include <omp.h>


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
                output = radSettings->getLinke();
                break;

            case PARAM_MODE_MAP:
                 output = NODATA;
                 break;

            default:
                output = radSettings->getLinke();
        }
        return output;
    }

    void readLinke(Crit3DRadiationSettings* radSettings, std::vector<float> &linkeMonthly)
    {
        switch(radSettings->getLinkeMode())
        {

        case PARAM_MODE_MONTHLY:
            linkeMonthly = radSettings->getLinkeMonthly();

            break;
        case PARAM_MODE_FIXED:
        case PARAM_MODE_MAP:
            break;
        }
        return;
    }

    float readLinke(Crit3DRadiationSettings* radSettings, int row, int col)
    {
        float output = NODATA;
        switch(radSettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = radSettings->getLinke();
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
                output = radSettings->getLinke();
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
     * \param linke:  Linke turbidity factor for an air mass equal to 2 ()
     * \param sunPosition:  a pointer to a TsunPosition
     * \return result
     */
    float clearSkyBeamHorizontal(float linke, const TsunPosition& sunPosition)
    {       
        // Rayleigh optical thickness (Kasten, 1996)
        float rayleighThickness;
        // relative optical air mass corrected for pressure
        float airMass = sunPosition.relOptAirMassCorr;

        if (airMass <= 20)
            rayleighThickness = 1.f / (6.6296f + 1.7513f * airMass - 0.1202f * float(pow(airMass, 2))
                                       + 0.0065f * float(pow(airMass, 3)) - 0.00013f * float(pow(airMass, 4)));
        else
            rayleighThickness = 1.f / (10.4f + 0.718f * airMass);

        return sunPosition.extraIrradianceNormal * getSinDecimalDegree(sunPosition.elevation)
                * float(exp(-0.8662f * linke * airMass * rayleighThickness));
    }


    /*!
     * \brief Diffuse irradiance on a horizontal surface [W m-2]
     * \brief (Rigollier et al. 2000)
     * \param linke
     * \param sunPosition a pointer to a TsunPosition
     * \return result
     */
    float clearSkyDiffuseHorizontal(float linke, const TsunPosition& sunPosition)
    {
        double Fd;          /*!< [-] diffuse solar altitude function     */
        double Trd;         /*!< [-] transmission function               */

        Trd = -0.015843 + linke * (0.030543 + 0.0003797 * linke);

        double sinElev = std::max(getSinDecimalDegree(double(sunPosition.elevation)), 0.);
        double A0 = 0.26463 + linke * (-0.061581 + 0.0031408 * linke);
        if ((A0 * Trd) < 0.0022)
        {
            A0 = 0.002 / Trd;
        }
        double A1 = 2.0402 + linke * (0.018945 - 0.011161 * linke);
        double A2 = -1.3025 + linke * (0.039231 + 0.0085079 * linke);
        Fd = A0 + A1 * sinElev + A2 * sinElev * sinElev;

        return sunPosition.extraIrradianceNormal * float(Fd * Trd);
    }


    /*!
     * \brief Beam irradiance on an inclined surface                         [W m-2]
     * \param beamIrradianceHor
     * \param sunPosition a pointer to a TsunPosition
     * \return result
     */
    float clearSkyBeamInclined(float beamIrradianceHor, const TsunPosition& sunPosition)
    {
        /*! Bh: clear sky beam irradiance on a horizontal surface */
        return (beamIrradianceHor * getSinDecimalDegree(sunPosition.incidence) / getSinDecimalDegree(sunPosition.elevationRefr)) ;
    }


    /*!
     * \brief Diffuse irradiance on an inclined surface (Muneer, 1990)               [W m-2]
     * \param beamIrradianceHor
     * \param diffuseIrradianceHor
     * \param sunPosition a pointer to a TsunPosition
     * \param radPoint
     * \return result
     */
    float clearSkyDiffuseInclined(float beamIrradianceHor, float diffuseIrradianceHor,
                                  const TsunPosition& sunPosition, const TradPoint& radPoint)
    {
        //Bh                     beam irradiance on a horizontal surface                                     [W m-2]
        //Dh                     diffuse irradiance on a horizontal surface

        double cosSlope, sinSlope;
        double slopeRad, aspectRad, elevRad, azimRad;
        double sinElev;
        double Kb;        /*!< amount of beam irradiance available [] */
        double Fg, r_sky, Fx, Aln;
        double n;
        sinElev = MAXVALUE(getSinDecimalDegree(sunPosition.elevation), 0.001);
        cosSlope = getCosDecimalDegree(radPoint.slope);
        sinSlope = getSinDecimalDegree(radPoint.slope);
        slopeRad = radPoint.slope * DEG_TO_RAD;
        aspectRad = radPoint.aspect * DEG_TO_RAD;
        elevRad = sunPosition.elevation * DEG_TO_RAD;
        azimRad = sunPosition.azimuth * DEG_TO_RAD;

        Kb = beamIrradianceHor / (sunPosition.extraIrradianceNormal * sinElev);
        Fg = sinSlope - slopeRad * cosSlope - PI * getSinDecimalDegree(radPoint.slope * 0.5)
                                                 * getSinDecimalDegree(radPoint.slope * 0.5);
        r_sky = (1.0 + cosSlope) / 2.0;
        if ((((sunPosition.shadow) || ((sunPosition).incidence * DEG_TO_RAD) <= 0.1)) && (elevRad >= 0.0))
        {
            (n = 0.252271) ;
            Fx = r_sky + Fg * n ;
        }
        else
        {
            n = 0.00263 - Kb * (0.712 + 0.6883 * Kb);
            //FT attenzione: crea discontinuita'
            if (elevRad >= 0.1) (Fx = (n * Fg + r_sky) * (1 - Kb) + Kb * getSinDecimalDegree(sunPosition.incidence) / sinElev);
            //elevRad < 0.1
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


    float getReflectedIrradiance(float beamIrradianceHor, float diffuseIrradianceHor, float albedo, float slope)
    {
        if (slope > 0)
            //Muneer 1997
            return (float)(albedo * (beamIrradianceHor + diffuseIrradianceHor) * (1 - getCosDecimalDegree(slope)) / 2.);
        else
            return 0;
    }


    bool isIlluminated(float myTime, float riseTime, float setTime, float sunElevation)
    {
        bool output = false ;
        if ((riseTime != NODATA) && (setTime != NODATA) && (sunElevation != NODATA))
            output =  ((myTime >= riseTime) && (myTime <= setTime) && (sunElevation > 0));
        return output;
    }


    bool computeShadow(const TradPoint& radPoint, const TsunPosition& sunPosition, const gis::Crit3DRasterGrid& dem)
    {
        double sunMaskStepX, sunMaskStepY;
        double sunMaskStepZ, maxDeltaH;
        double x, y, z, x0, y0, z0;
        double cosElev, sinElev, tgElev;
        double step, stepCount, maxDistCount;
        double zDEM;
        int row, col;

        /* INPUT
        azimuth
        elevationRefr
        supponiamo di avere gia' controllato se siamo dopo l'alba e prima del tramonto
        inizializzazione a sole visibile
        */

        x0 = radPoint.x;
        y0 = radPoint.y;
        z0 = radPoint.height;

        sunMaskStepX = SHADOW_FACTOR * getSinDecimalDegree(sunPosition.azimuth) * dem.header->cellSize;
        sunMaskStepY = SHADOW_FACTOR * getCosDecimalDegree(sunPosition.azimuth) * dem.header->cellSize;
        cosElev = getCosDecimalDegree(sunPosition.elevation);
        sinElev = getSinDecimalDegree(sunPosition.elevation);
        tgElev = sinElev / cosElev;
        sunMaskStepZ = dem.header->cellSize * SHADOW_FACTOR * tgElev;

        maxDeltaH = dem.header->cellSize * SHADOW_FACTOR * 2;

        if (sunMaskStepZ == 0)
            maxDistCount = dem.maximum - z0 / EPSILON;
        else
            maxDistCount = (dem.maximum - z0) / sunMaskStepZ;

        stepCount = 0;
        step = 1;
        do
        {
            stepCount += step;
            x = x0 + sunMaskStepX * stepCount;
            y = y0 + sunMaskStepY * stepCount;
            z = z0 + sunMaskStepZ * stepCount;

            dem.getRowCol(x, y, row, col);
            if (gis::isOutOfGridRowCol(row, col, dem))
            {
                // not shadowed - exit
                return false ;
            }

            zDEM = dem.value[row][col];
            if (zDEM != dem.header->flag)
            {
                if ((zDEM - z) > 0.5)
                {
                    // shadowed - exit
                    return true ;
                }
                else
                {
                    step = (z - zDEM) / maxDeltaH;
                    if (step < 1) step = 1;
                }
            }

        } while(stepCount < maxDistCount);

        return false;
    }


    void separateTransmissivity(float clearSkyTransmissivity, float transmissivity, float *td, float *Tt)
    {
        float maximumDiffuseTransmissivity;

        //in attesa di studi mirati (Bristow and Campbell, 1985)
        maximumDiffuseTransmissivity = 0.6f / (clearSkyTransmissivity - 0.4f);
        *Tt = MAXVALUE(MINVALUE(transmissivity, clearSkyTransmissivity), 0.00001f);
        *td = (*Tt) * (1 - expf(maximumDiffuseTransmissivity - (maximumDiffuseTransmissivity * clearSkyTransmissivity) / (*Tt)));

        /*! FT 0.12 stimato da Settefonti agosto 2007 */
        if ((*Tt) > 0.6f) *td = MAXVALUE(*td, 0.12f);
    }


bool computeRadiationRsun(Crit3DRadiationSettings* radSettings, float temperature, float myPressure, const Crit3DTime& myTime,
                               float linke,float albedo, float clearSkyTransmissivity, float transmissivity,
                               TsunPosition& sunPosition, TradPoint& radPoint, const gis::Crit3DRasterGrid& dem)
    {
        int myYear, myMonth, myDay;
        int myHour, myMinute, mySecond;
        float Bhc, Bh;
        float Dhc, dH;
        float Ghc, Gh;
        float globalTransmittance;  /*!<   real sky global irradiation coefficient (global transmittance) */
        float diffuseTransmittance; /*!<   real sky radPoint.diffuse irradiation coefficient (radPoint.diffuse transmittance) */
        float dhsOverGhs;           /*!<  ratio horizontal radPoint.diffuse over horizontal global */
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

        /*! Surface pressure at sea level (millibars) (used for refraction correction and optical air mass) */
        myPressure = PRESSURE_SEALEVEL * float(exp(-radPoint.height / RAYLEIGH_Z0));

        /*! Ambient default dry-bulb temperature (degrees C) (used for refraction correction) */
        //should be passed
        if (isEqual(temperature,NODATA))
            temperature = TEMPERATURE_DEFAULT;

        /*! Sun position */
        if (! computeSunPosition(float(radPoint.lon), float(radPoint.lat), radSettings->gisSettings->timeZone,
            myYear, myMonth, myDay, myHour, myMinute, mySecond,
            temperature, myPressure, float(radPoint.aspect), float(radPoint.slope), sunPosition))
            return false;

        /*! Shadowing */
        isPointIlluminated = isIlluminated(float(localTime.time), sunPosition.rise, sunPosition.set, sunPosition.elevationRefr);
        if (radSettings->getShadowing())
        {
            if (gis::isOutOfGridXY(radPoint.x, radPoint.y, dem.header))
                sunPosition.shadow = ! isPointIlluminated;
            else
            {
                if (isPointIlluminated)
                    sunPosition.shadow = computeShadow(radPoint, sunPosition, dem);
                else
                    sunPosition.shadow = true;
            }
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
            if (! radSettings->getRealSky()) transmissivity = clearSkyTransmissivity;

            Gh = sunPosition.extraIrradianceHorizontal * transmissivity;
            separateTransmissivity (clearSkyTransmissivity, transmissivity, &diffuseTransmittance, &globalTransmittance);
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
                separateTransmissivity (clearSkyTransmissivity, transmissivity, &diffuseTransmittance, &globalTransmittance);
                dhsOverGhs = diffuseTransmittance / globalTransmittance;
                dH = dhsOverGhs * Gh;
            }
            else {
                Gh = Ghc;
                dH = Dhc;
            }
        }

        // shadowing
        if (!sunPosition.shadow && sunPosition.incidence > 0.)
            Bh = Gh - dH;
        else
        {
            Bh = 0;
            Gh = dH; // approximation (portion of shadowed sky should be considered)
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
            if (!sunPosition.shadow && sunPosition.incidence > 0.)
                radPoint.beam = clearSkyBeamInclined(Bh, sunPosition);
            else
                radPoint.beam = 0;

            radPoint.diffuse = clearSkyDiffuseInclined(Bh, dH, sunPosition, radPoint);
            radPoint.reflected = getReflectedIrradiance(Bh, dH, albedo, float(radPoint.slope));
            radPoint.global = radPoint.beam + radPoint.diffuse + radPoint.reflected;
        }

        return true;
    }


    int estimateTransmissivityWindow(Crit3DRadiationSettings* radSettings, const gis::Crit3DRasterGrid& dem,
                                     const gis::Crit3DPoint& point, Crit3DTime myTime, int timeStepSecond)
    {
        double latDegrees, lonDegrees;
        TradPoint radPoint;
        TsunPosition sunPosition;
        float linke, albedo;
        float clearSkyTransmissivity;
        float sumPotentialRadThreshold = 0.;
        float sumPotentialRad = 0.;
        Crit3DTime backwardTime;
        Crit3DTime forwardTime;
        int myWindowSteps;
        int row, col;

        /*! assegna altezza e coordinate stazione */
        radPoint.x = point.utm.x;
        radPoint.y = point.utm.y;
        radPoint.height = point.z;
        if (radPoint.height == NODATA)
        {
            radPoint.height = double(gis::getValueFromXY(dem, radPoint.x, radPoint.y));
        }

        dem.getRowCol(radPoint.x, radPoint.y, row, col);
        radPoint.aspect = 0;
        radPoint.slope = 0;

        gis::getLatLonFromUtm(*(radSettings->gisSettings), radPoint.x, radPoint.y, &latDegrees, &lonDegrees);
        radPoint.lat = latDegrees;
        radPoint.lon = lonDegrees;

        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, point);

        albedo = readAlbedo(radSettings, point);
        clearSkyTransmissivity = radSettings->getClearSky();

        // noon
        Crit3DTime noonTime = myTime;
        noonTime.time = 12*3600;
        if (radSettings->gisSettings->isUTC)
        {
            noonTime = noonTime.addSeconds(-radSettings->gisSettings->timeZone * 3600);
        }

        // Threshold: half of potential radiation at noon
        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, noonTime, linke, albedo,
                                  clearSkyTransmissivity, clearSkyTransmissivity, sunPosition, radPoint, dem);
        sumPotentialRadThreshold = float(radPoint.global * 0.5);

        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime, linke, albedo,
                                  clearSkyTransmissivity, clearSkyTransmissivity, sunPosition, radPoint, dem);
        sumPotentialRad = float(radPoint.global);

        int backwardTimeStep,forwardTimeStep;
        backwardTimeStep = forwardTimeStep = 0;
        myWindowSteps = 1;
        backwardTime = forwardTime = myTime;

        while (sumPotentialRad < sumPotentialRadThreshold)
        {
            myWindowSteps += 2;

            backwardTimeStep -= timeStepSecond ;
            forwardTimeStep += timeStepSecond;
            backwardTime = myTime.addSeconds(backwardTimeStep);
            forwardTime = myTime.addSeconds(forwardTimeStep);

            computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, backwardTime,
                                      linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                      sunPosition, radPoint, dem);
            sumPotentialRad+= float(radPoint.global);

            computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, forwardTime,
                                      linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                      sunPosition, radPoint, dem);
            sumPotentialRad+= float(radPoint.global);
        }

        return myWindowSteps;
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
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime,
                                  linke, albedo, radSettings->getClearSky(), transmissivity, sunPosition, radPoint, dem))
            return false;

        radiationMaps->sunElevationMap->value[row][col] = sunPosition.elevation;
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
        if (! computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime,
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
        if (!computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime,
            linke, albedo, radSettings->getClearSky(), transmissivity, sunPosition, radPoint, dem))
            return false;

        return true;
    }


    bool computeRadiationDEM(Crit3DRadiationSettings* radSettings, const gis::Crit3DRasterGrid& dem,
                              Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myTime)
    {
        if (radSettings->getAlgorithm() != RADIATION_ALGORITHM_RSUN)
            return false;

        bool isOk = true;
        #pragma omp parallel for shared(isOk)
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

        sunPosition.incidence = float(MAXVALUE(0, RAD_TO_DEG * ((PI / 2.0) - acos(sunCosIncidenceCompl))));
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

        //controllare i segni:
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

        extraTerrestrialRad = MAXVALUE(0, SOLAR_CONSTANT * sin(elevationAngle));

        azimuthSouth = acos((sin(elevationAngle)
                                * sin(radPoint->lat) - sin(solarDeclination))
                               / (cos(elevationAngle) * cos(radPoint->lat)));

        azimuthNorth = (solarTime>12) ? PI + azimuthSouth : PI - azimuthSouth;
        incidenceAngle = MAXVALUE(0, asin(getSinDecimalDegree(float(radPoint->slope)) *
                                        cos(elevationAngle) * cos(azimuthNorth - float(radPoint->aspect))
                                        + getCosDecimalDegree(float(radPoint->slope)) * sin(elevationAngle)));

        float Tt = clearSkyTransmissivity;
        float td = 0.1f;
        if (radSettings->getRealSky())
        {
            if (transmissivity != NODATA)
            {
                separateTransmissivity (clearSkyTransmissivity, transmissivity, &td, &Tt);
            }
        }

        coeffBH = MAXVALUE(0, (Tt - td));

        radDiffuse = extraTerrestrialRad * td;

        if (radSettings->getTilt() == 0)
        {
            radBeam = extraTerrestrialRad * coeffBH;
            radReflected = 0;
        }
        else
        {
            radBeam = extraTerrestrialRad * coeffBH * MAXVALUE(0, sin(incidenceAngle) / sin(elevationAngle));
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


    float computePointTransmissivity(Crit3DRadiationSettings* radSettings, const gis::Crit3DPoint& point, Crit3DTime myTime,
                                     float* measuredRad, int windowWidth, int timeStepSecond, const gis::Crit3DRasterGrid& dem)
    {
        if (windowWidth % 2 != 1) return NODATA;

        int intervalCenter = (windowWidth-1)/2;

        if (measuredRad[intervalCenter] == NODATA) return NODATA;

        double latDegrees, lonDegrees;
        float ratioTransmissivity;
        TradPoint radPoint;
        TsunPosition sunPosition;
        float linke, albedo;
        float  clearSkyTransmissivity;
        float transmissivity;

        Crit3DTime backwardTime;
        Crit3DTime forwardTime;

        /*! assign topographic height and coordinates */
        radPoint.x = point.utm.x;
        radPoint.y = point.utm.y;
        radPoint.height = point.z;
        if (radPoint.height == NODATA)
        {
            radPoint.height = double(gis::getValueFromXY(dem, radPoint.x, radPoint.y));
        }

        /*! suppose radiometers are horizontal */
        radPoint.aspect = 0.;
        radPoint.slope = 0.;

        gis::getLatLonFromUtm(*(radSettings->gisSettings), point.utm.x, point.utm.y, &latDegrees, &lonDegrees);
        radPoint.lat = latDegrees;
        radPoint.lon = lonDegrees;

        if (radSettings->getLinkeMode() == PARAM_MODE_MONTHLY)
            linke = radSettings->getLinke(myTime.date.month-1);
        else
            linke = readLinke(radSettings, point);

        albedo = readAlbedo(radSettings, point);
        clearSkyTransmissivity = radSettings->getClearSky();

        int backwardTimeStep,forwardTimeStep;
        backwardTimeStep = forwardTimeStep = 0;
        backwardTime = forwardTime = myTime;

        float sumMeasuredRad = measuredRad[intervalCenter];

        computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime, linke, albedo,
                                clearSkyTransmissivity, clearSkyTransmissivity, sunPosition, radPoint, dem);

        float sumPotentialRad = float(radPoint.global);

        for (int windowIndex = (intervalCenter - 1); windowIndex >= 0; windowIndex--)
        {
            backwardTimeStep -= timeStepSecond;
            forwardTimeStep += timeStepSecond;
            backwardTime = myTime.addSeconds(backwardTimeStep);
            forwardTime = myTime.addSeconds(forwardTimeStep);

            if (measuredRad[windowIndex] != NODATA)
            {
                sumMeasuredRad += measuredRad[windowIndex];
                computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, backwardTime,
                                          linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                          sunPosition, radPoint, dem);
                sumPotentialRad += float(radPoint.global);
            }
            if (measuredRad[windowWidth-windowIndex-1] != NODATA)
            {
                sumMeasuredRad+= measuredRad[windowWidth-windowIndex-1];
                computeRadiationRsun(radSettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, forwardTime,
                                          linke, albedo, clearSkyTransmissivity, clearSkyTransmissivity,
                                          sunPosition, radPoint, dem);
                sumPotentialRad+= float(radPoint.global);
            }
        }

        ratioTransmissivity = MAXVALUE(sumMeasuredRad / sumPotentialRad, float(0.0));
        transmissivity = ratioTransmissivity * clearSkyTransmissivity;

        /*! transmissivity can't be over 0.85 */
        return MINVALUE(transmissivity, float(0.85));
    }
}

