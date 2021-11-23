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
#include "sunPosition.h"
#include "solarRadiation.h"
#include "gis.h"
#include <math.h>


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

    /*
    linkeMap = new gis::Crit3DRasterGrid;
    albedoMap = new gis::Crit3DRasterGrid;
    sunAzimuthMap = new gis::Crit3DRasterGrid;
    sunIncidenceMap = new gis::Crit3DRasterGrid;
    sunShadowMap = new gis::Crit3DRasterGrid;
    */

    isComputed = false;

}

Crit3DRadiationMaps::Crit3DRadiationMaps(const gis::Crit3DRasterGrid& myDEM, const gis::Crit3DGisSettings& myGisSettings)
{
    latMap = new gis::Crit3DRasterGrid;
    lonMap = new gis::Crit3DRasterGrid;
    gis::computeLatLonMaps(myDEM, latMap, lonMap, myGisSettings);

    slopeMap = new gis::Crit3DRasterGrid;
    aspectMap = new gis::Crit3DRasterGrid;
    gis::computeSlopeAspectMaps(myDEM, slopeMap, aspectMap);

    transmissivityMap = new gis::Crit3DRasterGrid;
    transmissivityMap->initializeGrid(myDEM, CLEAR_SKY_TRANSMISSIVITY_DEFAULT);

    globalRadiationMap = new gis::Crit3DRasterGrid;
    globalRadiationMap->initializeGrid(myDEM);

    beamRadiationMap = new gis::Crit3DRasterGrid;
    beamRadiationMap->initializeGrid(myDEM);

    diffuseRadiationMap = new gis::Crit3DRasterGrid;
    diffuseRadiationMap->initializeGrid(myDEM);

    reflectedRadiationMap = new gis::Crit3DRasterGrid;
    reflectedRadiationMap->initializeGrid(myDEM);

    sunElevationMap = new gis::Crit3DRasterGrid;
    sunElevationMap->initializeGrid(myDEM);

    /*
    albedoMap = new gis::Crit3DRasterGrid;
    linkeMap = new gis::Crit3DRasterGrid;
    sunAzimuthMap = new gis::Crit3DRasterGrid;
    sunIncidenceMap = new gis::Crit3DRasterGrid;
    sunShadowMap = new gis::Crit3DRasterGrid;

    linkeMap->initializeGrid(myDEM);
    albedoMap->initializeGrid(myDEM);
    sunAzimuthMap->initializeGrid(myDEM);
    sunIncidenceMap->initializeGrid(myDEM);
    sunShadowMap->initializeGrid(myDEM);
    */

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

    /*
    albedoMap->clear();
    linkeMap->clear();
    sunAzimuthMap->clear();
    sunIncidenceMap->clear();
    sunShadowMap->clear();
    */

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

    /*
    delete albedoMap;
    delete linkeMap;
    delete sunAzimuthMap;
    delete sunIncidenceMap;
    delete sunShadowMap;
    */

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

    float readAlbedo(Crit3DRadiationSettings* mySettings)
    {
        float output = NODATA;
        switch(mySettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                 output = NODATA;
                 break;

            default:
                output = mySettings->getAlbedo();
        }
        return output;
    }

    float readAlbedo(Crit3DRadiationSettings* mySettings, int myRow, int myCol)
    {
        float output = NODATA;
        switch (mySettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                output = mySettings->getAlbedo(myRow, myCol);
                break;

            default:
                output = mySettings->getAlbedo(myRow, myCol);
        }
        return output;
    }

    float readAlbedo(Crit3DRadiationSettings* mySettings, const gis::Crit3DPoint& myPoint)
    {
        float output = NODATA;
        switch(mySettings->getAlbedoMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getAlbedo();
                break;

            case PARAM_MODE_MAP:
                output = mySettings->getAlbedo(myPoint);
                break;

            default:
                output = mySettings->getAlbedo(myPoint);
        }
        return output;
    }

    float readLinke(Crit3DRadiationSettings* mySettings)
    {
        float output = NODATA;
        switch(mySettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getLinke();
                break;

            case PARAM_MODE_MAP:
                 output = NODATA;
                 break;

            default:
                output = mySettings->getLinke();
        }
        return output;
    }

    float readLinke(Crit3DRadiationSettings* mySettings, int myRow, int myCol)
    {
        float output = NODATA;
        switch(mySettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getLinke();
                break;

            case PARAM_MODE_MAP:
                output = mySettings->getLinke(myRow, myCol);
                break;

            default:
                output = mySettings->getLinke(myRow, myCol);
        }
        return output;
    }

    float readLinke(Crit3DRadiationSettings* mySettings, const gis::Crit3DPoint& myPoint)
    {
        float output = NODATA;
        switch(mySettings->getLinkeMode())
        {
            case PARAM_MODE_FIXED:
                output = mySettings->getLinke();
                break;

            case PARAM_MODE_MAP:
                output = mySettings->getLinke(myPoint);
                break;

            default:
                mySettings->getLinke(myPoint);
        }
        return output;
    }

    float readAspect(Crit3DRadiationSettings* mySettings)
    {
        float output = NODATA;

        switch (mySettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = mySettings->getAspect();
                break;
            case TILT_TYPE_DEM:
                output = NODATA;
                break;
        }
        return output;
    }

    float readAspect(Crit3DRadiationSettings* mySettings, gis::Crit3DRasterGrid* aspectMap, int myRow,int myCol)
    {
        float output = NODATA;

        switch (mySettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = mySettings->getAspect();
                break;
            case TILT_TYPE_DEM:
                output = aspectMap->value[myRow][myCol];
                break;
        }
        return output;
    }

    float readSlope(Crit3DRadiationSettings* mySettings)
    {
        float output = NODATA;
        switch (mySettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = mySettings->getTilt();
                break;
            case TILT_TYPE_DEM:
                output = NODATA;
                break;
        }
        return output;
    }

    float readSlope(Crit3DRadiationSettings* mySettings, gis::Crit3DRasterGrid* slopeMap, int myRow, int myCol)
    {
        float output = NODATA;

        switch (mySettings->getTiltMode())
        {
            case TILT_TYPE_FIXED:
                output = mySettings->getTilt();
                break;
            case TILT_TYPE_DEM:
                output = slopeMap->value[myRow][myCol];
                break;
        }
        return output;
    }

    /*!
     * \brief Clear sky beam irradiance on a horizontal surface [W m-2]
     * \brief (Rigollier et al. 2000)
     * \param myLinke
     * \param mySunPosition a pointer to a TsunPosition
     * \return result
     */
    float clearSkyBeamHorizontal(float myLinke, TsunPosition* mySunPosition)
    {
        //myLinke:   Linke turbidity factor for an air mass equal to 2 ()
        //rayl:      Rayleigh optical thickness (Kasten, 1996) ()
        //am:        relative optical air mass corrected for pressure ()

        float rayleighThickness;
        float airMass = mySunPosition->relOptAirMassCorr ;
        if (airMass <= 20)
            rayleighThickness = 1.f / (6.6296f + 1.7513f * airMass - 0.1202f * pow(airMass, 2)
                                       + 0.0065f * pow(airMass, 3) - 0.00013f * pow(airMass, 4));
        else
            rayleighThickness = 1.f / (10.4f + 0.718f * airMass);

        return mySunPosition->extraIrradianceNormal * getSinDecimalDegree(mySunPosition->elevation)
                * float(exp(-0.8662f * myLinke * airMass * rayleighThickness));
    }

    /*!
     * \brief Diffuse irradiance on a horizontal surface [W m-2]
     * \brief (Rigollier et al. 2000)
     * \param myLinke
     * \param mySunPosition a pointer to a TsunPosition
     * \return result
     */
    float clearSkyDiffuseHorizontal(float myLinke, TsunPosition* mySunPosition)
    {
        double Fd;          /*!< diffuse solar altitude function [] */
        double Trd;         /*!< transmission function [] */
        double linke = double(myLinke);

        Trd = -0.015843 + linke * (0.030543 + 0.0003797 * linke);

        double sinElev = MAXVALUE(double(getSinDecimalDegree(mySunPosition->elevation)), 0);
        double A0 = 0.26463 + linke * (-0.061581 + 0.0031408 * linke);
        if ((A0 * Trd) < 0.0022)
            A0 = 0.002 / Trd;
        double A1 = 2.0402 + linke * (0.018945 - 0.011161 * linke);
        double A2 = -1.3025 + linke * (0.039231 + 0.0085079 * linke);
        Fd = A0 + A1 * sinElev + A2 * sinElev * sinElev;

        return mySunPosition->extraIrradianceNormal * float(Fd * Trd);
    }

    /*!
     * \brief Beam irradiance on an inclined surface                         [W m-2]
     * \param beamIrradianceHor
     * \param mySunPosition a pointer to a TsunPosition
     * \return result
     */
    float clearSkyBeamInclined(float beamIrradianceHor, TsunPosition* mySunPosition )
    {
        /*! Bh: clear sky beam irradiance on a horizontal surface */
        return (beamIrradianceHor * getSinDecimalDegree(mySunPosition->incidence) / getSinDecimalDegree(mySunPosition->elevationRefr)) ;
    }

    /*!
     * \brief Diffuse irradiance on an inclined surface (Muneer, 1990)               [W m-2]
     * \param beamIrradianceHor
     * \param diffuseIrradianceHor
     * \param mySunPosition a pointer to a TsunPosition
     * \param myPoint
     * \return result
     */
    float clearSkyDiffuseInclined(float beamIrradianceHor, float diffuseIrradianceHor, TsunPosition* mySunPosition, TradPoint* myPoint)
    {
        //Bh                     beam irradiance on a horizontal surface                                     [W m-2]
        //Dh                     diffuse irradiance on a horizontal surface

        double cosSlope, sinSlope;
        double slopeRad, aspectRad, elevRad, azimRad;
        double sinElev;
        double Kb;        /*!< amount of beam irradiance available [] */
        double Fg, r_sky, Fx, Aln;
        double n;
        sinElev = MAXVALUE(getSinDecimalDegree(mySunPosition->elevation), 0.001);
        cosSlope = getCosDecimalDegree(float(myPoint->slope));
        sinSlope = getSinDecimalDegree(float(myPoint->slope));
        slopeRad = myPoint->slope * DEG_TO_RAD;
        aspectRad = myPoint->aspect * DEG_TO_RAD;
        elevRad = mySunPosition->elevation * DEG_TO_RAD;
        azimRad = mySunPosition->azimuth * DEG_TO_RAD;

        Kb = beamIrradianceHor / (mySunPosition->extraIrradianceNormal * sinElev);
        Fg = sinSlope - slopeRad * cosSlope - PI * getSinDecimalDegree(float(myPoint->slope / 2.0))
                                                 * getSinDecimalDegree(float(myPoint->slope / 2.0));
        r_sky = (1.0 + cosSlope) / 2.0;
        if ((((mySunPosition->shadow) || ((mySunPosition)->incidence * DEG_TO_RAD) <= 0.1)) && (elevRad >= 0.0))
        {
            (n = 0.252271) ;
            Fx = r_sky + Fg * n ;
        }
        else
        {
            n = 0.00263 - Kb * (0.712 + 0.6883 * Kb);
            //FT attenzione: crea discontinuita'
            if (elevRad >= 0.1) (Fx = (n * Fg + r_sky) * (1 - Kb) + Kb * getSinDecimalDegree(mySunPosition->incidence) / sinElev);
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


    float getReflectedIrradiance(float beamIrradianceHor, float diffuseIrradianceHor, float myAlbedo, float mySlope)
    {
        if (mySlope > 0)
            //Muneer 1997
            return (float)(myAlbedo * (beamIrradianceHor + diffuseIrradianceHor) * (1 - getCosDecimalDegree(mySlope)) / 2.);
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


    bool computeShadow(TradPoint* myPoint, TsunPosition* mySunPosition, const gis::Crit3DRasterGrid& myDEM)
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

        x0 = myPoint->x;
        y0 = myPoint->y;
        z0 = myPoint->height;

        sunMaskStepX = SHADOW_FACTOR * getSinDecimalDegree(mySunPosition->azimuth) * myDEM.header->cellSize;
        sunMaskStepY = SHADOW_FACTOR * getCosDecimalDegree(mySunPosition->azimuth) * myDEM.header->cellSize;
        cosElev = getCosDecimalDegree(mySunPosition->elevation);
        sinElev = getSinDecimalDegree(mySunPosition->elevation);
        tgElev = sinElev / cosElev;
        sunMaskStepZ = myDEM.header->cellSize * SHADOW_FACTOR * tgElev;

        maxDeltaH = myDEM.header->cellSize * SHADOW_FACTOR * 2;

        if (sunMaskStepZ == 0)
            maxDistCount = myDEM.maximum - z0 / EPSILON;
        else
            maxDistCount = (myDEM.maximum - z0) / sunMaskStepZ;

        stepCount = 0;
        step = 1;
        do
        {
            stepCount += step;
            x = x0 + sunMaskStepX * stepCount;
            y = y0 + sunMaskStepY * stepCount;
            z = z0 + sunMaskStepZ * stepCount;

            gis::getRowColFromXY(myDEM, x, y, &row, &col);
            if (gis::isOutOfGridRowCol(row, col, myDEM))
            {
                // not shadowed - exit
                return false ;
            }

            zDEM = myDEM.value[row][col];
            if (zDEM != myDEM.header->flag)
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


    void separateTransmissivity(float myClearSkyTransmissivity, float transmissivity, float *td, float *Tt)
    {
        float maximumDiffuseTransmissivity;

        //in attesa di studi mirati (Bristow and Campbell, 1985)
        maximumDiffuseTransmissivity = 0.6f / (myClearSkyTransmissivity - 0.4f);
        *Tt = MAXVALUE(MINVALUE(transmissivity, myClearSkyTransmissivity), 0.00001f);
        *td = (*Tt) * (1.f - exp(maximumDiffuseTransmissivity - (maximumDiffuseTransmissivity * myClearSkyTransmissivity) / (*Tt)));

        /*! FT 0.12 stimato da Settefonti agosto 2007 */
        if ((*Tt) > 0.6f) *td = MAXVALUE(*td, 0.1f);
    }


bool computeRadiationPointRsun(Crit3DRadiationSettings* mySettings, float myTemperature, float myPressure, Crit3DTime myTime,
                               float myLinke,float myAlbedo, float myClearSkyTransmissivity, float myTransmissivity,
                               TsunPosition* mySunPosition, TradPoint* myPoint, const gis::Crit3DRasterGrid& myDEM)
    {
        int myYear, myMonth, myDay;
        int myHour, myMinute, mySecond;
        float Bhc, Bh;
        float Dhc, dH;
        float Ghc, Gh;
        //float td, Tt;
        float globalTransmittance;  /*!<   real sky global irradiation coefficient (global transmittance) */
        float diffuseTransmittance; /*!<   real sky mypoint.diffuse irradiation coefficient (mypoint.diffuse transmittance) */
        float dhsOverGhs;           /*!<  ratio horizontal mypoint.diffuse over horizontal global */
        bool isPointIlluminated;

        Crit3DTime localTime;
        localTime = myTime;
        if (mySettings->gisSettings->isUTC)
        {
            localTime = myTime.addSeconds(mySettings->gisSettings->timeZone * 3600);
        }

        myYear = localTime.date.year;
        myMonth =  localTime.date.month;
        myDay =  localTime.date.day;
        myHour = localTime.getHour();
        myMinute = localTime.getMinutes();
        mySecond = int(localTime.getSeconds());

        /*! Surface pressure at sea level (millibars) (used for refraction correction and optical air mass) */
        myPressure = PRESSURE_SEALEVEL * float(exp(-myPoint->height / RAYLEIGH_Z0));

        /*! Ambient default dry-bulb temperature (degrees C) (used for refraction correction) */
        //should be passed
        if (myTemperature == NODATA) myTemperature = TEMPERATURE_DEFAULT;

        /*! Sun position */
        if (! computeSunPosition(float(myPoint->lon), float(myPoint->lat), mySettings->gisSettings->timeZone,
            myYear, myMonth, myDay, myHour, myMinute, mySecond,
            myTemperature, myPressure, float(myPoint->aspect), float(myPoint->slope), mySunPosition))
            return false;

        /*! Shadowing */
        isPointIlluminated = isIlluminated(float(localTime.time), (*mySunPosition).rise, (*mySunPosition).set, (*mySunPosition).elevationRefr);
        if (mySettings->getShadowing())
        {
            if (gis::isOutOfGridXY(myPoint->x, myPoint->y, myDEM.header))
                (*mySunPosition).shadow = ! isPointIlluminated;
            else
            {
                if (isPointIlluminated)
                    (*mySunPosition).shadow = computeShadow(myPoint, mySunPosition, myDEM);
                else
                    (*mySunPosition).shadow = true;
            }
        }

        /*! Radiation */
        if (! isPointIlluminated)
        {
            myPoint->beam = 0;
            myPoint->diffuse = 0;
            myPoint->reflected = 0;
            myPoint->global = 0;

            return true;
        }

        if (mySettings->getRealSky() && myTransmissivity == NODATA)
            return false;

        // real sky horizontal
        if (mySettings->getRealSkyAlgorithm() == RADIATION_REALSKY_TOTALTRANSMISSIVITY)
        {
            if (! mySettings->getRealSky()) myTransmissivity = myClearSkyTransmissivity;

            Gh = mySunPosition->extraIrradianceHorizontal * myTransmissivity;
            separateTransmissivity (myClearSkyTransmissivity, myTransmissivity, &diffuseTransmittance, &globalTransmittance);
            dH = mySunPosition->extraIrradianceHorizontal * diffuseTransmittance;
        }
        else
        {
            Bhc = clearSkyBeamHorizontal(myLinke, mySunPosition);
            Dhc = clearSkyDiffuseHorizontal(myLinke, mySunPosition);
            Ghc = Dhc + Bhc;

            if (mySettings->getRealSky())
            {
                Gh = Ghc * myTransmissivity / myClearSkyTransmissivity;
                // todo: trovare un metodo migliore (che non usi la clearSkyTransmissivity, non coerente con l'utilizzo di Linke)
                separateTransmissivity (myClearSkyTransmissivity, myTransmissivity, &diffuseTransmittance, &globalTransmittance);
                dhsOverGhs = diffuseTransmittance / globalTransmittance;
                dH = dhsOverGhs * Gh;
            }
            else {
                Gh = Ghc;
                dH = Dhc;
            }
        }

        // shadowing
        if ((!(*mySunPosition).shadow) && ((*mySunPosition).incidence > 0.))
            Bh = Gh - dH;
        else
        {
            Bh = 0;
            Gh = dH; // approximation (portion of shadowed sky should be considered)
        }

        // inclined
        if (myPoint->slope == 0)
        {
            (*myPoint).beam = Bh;
            (*myPoint).diffuse = dH;
            (*myPoint).reflected = 0;
            (*myPoint).global = Gh;
        }
        else
        {
            if ((!(*mySunPosition).shadow) && ((*mySunPosition).incidence > 0.))
                myPoint->beam = clearSkyBeamInclined(Bh, mySunPosition);
            else
                myPoint->beam = 0;

            myPoint->diffuse = clearSkyDiffuseInclined(Bh, dH, mySunPosition, myPoint);
            myPoint->reflected = getReflectedIrradiance(Bh, dH, myAlbedo, float(myPoint->slope));
            myPoint->global = myPoint->beam + myPoint->diffuse + myPoint->reflected;
        }

        return true;
    }


    int estimateTransmissivityWindow(Crit3DRadiationSettings* mySettings, const gis::Crit3DRasterGrid& myDEM,
                                     const gis::Crit3DPoint& myPoint, Crit3DTime myTime, int timeStepSecond)
    {
        double latDegrees, lonDegrees;
        TradPoint myRadPoint;
        TsunPosition mySunPosition;
        float myLinke, myAlbedo;
        float  myClearSkyTransmissivity;
        float sumPotentialRadThreshold = 0.;
        float sumPotentialRad = 0.;
        Crit3DTime backwardTime;
        Crit3DTime forwardTime;
        int myWindowSteps;
        int myRow, myCol;

        /*! assegna altezza e coordinate stazione */
        myRadPoint.x = myPoint.utm.x;
        myRadPoint.y = myPoint.utm.y;
        myRadPoint.height = myPoint.z;

        if (myPoint.z == NODATA)
            myRadPoint.height = double(gis::getValueFromXY(myDEM, myRadPoint.x, myRadPoint.y));

        gis::getRowColFromXY(myDEM, myRadPoint.x, myRadPoint.y, &myRow, &myCol);
        myRadPoint.aspect = 0;
        myRadPoint.slope = 0;

        gis::getLatLonFromUtm(*(mySettings->gisSettings), myPoint.utm.x, myPoint.utm.y, &latDegrees, &lonDegrees);
        myRadPoint.lat = latDegrees;
        myRadPoint.lon = lonDegrees;

        myLinke = readLinke(mySettings, myPoint);
        myAlbedo = readAlbedo(mySettings, myPoint);
        myClearSkyTransmissivity = mySettings->getClearSky();

        // noon
        Crit3DTime noonTime = myTime;
        noonTime.time = 12*3600;
        if (mySettings->gisSettings->isUTC)
        {
            noonTime = noonTime.addSeconds(-mySettings->gisSettings->timeZone * 3600);
        }

        // Threshold: half of potential radiation at noon
        computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, noonTime, myLinke, myAlbedo,
                                  myClearSkyTransmissivity, myClearSkyTransmissivity, &mySunPosition, &myRadPoint, myDEM);
        sumPotentialRadThreshold = float(myRadPoint.global * 0.5);

        computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime, myLinke, myAlbedo,
                                  myClearSkyTransmissivity, myClearSkyTransmissivity, &mySunPosition, &myRadPoint, myDEM);
        sumPotentialRad = float(myRadPoint.global);

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

            computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, backwardTime,
                                      myLinke, myAlbedo, myClearSkyTransmissivity, myClearSkyTransmissivity,
                                      &mySunPosition, &myRadPoint, myDEM);
            sumPotentialRad+= float(myRadPoint.global);

            computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, forwardTime,
                                      myLinke, myAlbedo, myClearSkyTransmissivity, myClearSkyTransmissivity,
                                      &mySunPosition, &myRadPoint, myDEM);
            sumPotentialRad+= float(myRadPoint.global);
        }

        return myWindowSteps;
    }


    bool isGridPointComputable(Crit3DRadiationSettings* mySettings, int row, int col, const gis::Crit3DRasterGrid& myDEM, Crit3DRadiationMaps* radiationMaps)
    {
        float mySlope, myAspect;
        bool output = false;
        if (myDEM.value[row][col] != myDEM.header->flag)
        {
            if ((radiationMaps->latMap->value[row][col] != radiationMaps->latMap->header->flag)
                    && (radiationMaps->lonMap->value[row][col] != radiationMaps->lonMap->header->flag))
            {
                mySlope = readSlope(mySettings, radiationMaps->slopeMap, row, col);
                myAspect = readAspect(mySettings, radiationMaps->aspectMap, row, col);
                if ((mySlope != NODATA) && (myAspect != NODATA))  output = true;
            }
        }
        return output;
    }


   bool computeRadiationGridRsun(Crit3DRadiationSettings* mySettings, const gis::Crit3DRasterGrid& myDEM,
                                 Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myTime)

    {
        int myRow, myCol;
        TsunPosition mySunPosition;
        TradPoint myRadPoint;

        for (myRow=0;myRow< myDEM.header->nrRows ; myRow++ )
        {
            for (myCol=0;myCol < myDEM.header->nrCols; myCol++)
            {
                if(isGridPointComputable(mySettings, myRow, myCol, myDEM, radiationMaps))
                {
                    gis::getUtmXYFromRowCol(myDEM, myRow, myCol, &(myRadPoint.x), &(myRadPoint.y));
                    myRadPoint.height = myDEM.value[myRow][myCol];
                    myRadPoint.lat = radiationMaps->latMap->value[myRow][myCol];
                    myRadPoint.lon = radiationMaps->lonMap->value[myRow][myCol];
                    myRadPoint.slope = readSlope(mySettings, radiationMaps->slopeMap, myRow, myCol);
                    myRadPoint.aspect = readAspect(mySettings, radiationMaps->aspectMap, myRow, myCol);

                    float linke = readLinke(mySettings, myRow, myCol);
                    float albedo = readAlbedo(mySettings, myRow, myCol);

                    float transmissivity = radiationMaps->transmissivityMap->value[myRow][myCol];

                    //CHIAMATA A SINGLE POINT
                    if (!computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime,
                        linke, albedo, mySettings->getClearSky(), transmissivity, &mySunPosition, &myRadPoint, myDEM))
                        return false;

                    /*
                    radiationMaps->sunAzimuthMap->value[myRow][myCol] = mySunPosition.azimuth;
                    radiationMaps->sunIncidenceMap->value[myRow][myCol] = mySunPosition.incidence;
                    radiationMaps->sunShadowMap->value[myRow][myCol] = float((mySunPosition.shadow) ?  0 : 1);
                    radiationMaps->reflectedRadiationMap->value[myRow][myCol] = float(myRadPoint.reflected);
                    */

                    radiationMaps->sunElevationMap->value[myRow][myCol] = mySunPosition.elevation;
                    radiationMaps->globalRadiationMap->value[myRow][myCol] = float(myRadPoint.global);
                    radiationMaps->beamRadiationMap->value[myRow][myCol] = float(myRadPoint.beam);
                    radiationMaps->diffuseRadiationMap->value[myRow][myCol] = float(myRadPoint.diffuse);
                    radiationMaps->reflectedRadiationMap->value[myRow][myCol] = float(myRadPoint.reflected);

                }
            }
        }

        /*
        gis::updateMinMaxRasterGrid(radiationMaps->sunAzimuthMap);
        gis::updateMinMaxRasterGrid(radiationMaps->sunIncidenceMap);
        gis::updateMinMaxRasterGrid(radiationMaps->sunShadowMap);
        */

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
        return true;
    }


    bool computeSunPosition(float lon, float lat, int myTimezone,
                            int myYear,int myMonth, int myDay,
                            int myHour, int myMinute, int mySecond,
                            float temp, float pressure, float aspect, float slope, TsunPosition *mySunPosition)
    {
        float etrTilt;  /*!<  Extraterrestrial (top-of-atmosphere) global irradiance on a tilted surface (W m-2) */
        float cosZen;   /*!<  Cosine of refraction corrected solar zenith angle */
        float sbcf;     /*!<  Shadow-band correction factor */
        float prime;    /*!<  Factor that normalizes Kt, Kn, etc. */
        float unPrime;  /*!<  Factor that denormalizes Kt', Kn', etc. */
        float  zenRef;  /*!<  Solar zenith angle, deg. from zenith, refracted */
        int chk;
        float sunCosIncidenceCompl; /*!<  cosine of (90 - incidence) */
        float sunRiseMinutes;       /*!<  sunrise time [minutes from midnight] */
        float sunSetMinutes;        /*!<  sunset time [minutes from midnight] */

        chk = RSUN_compute_solar_position(lon, lat, myTimezone, myYear, myMonth, myDay, myHour, myMinute, mySecond, temp, pressure, aspect, slope, float(SBWID), float(SBRAD), float(SBSKY));
        if (chk > 0)
        {
           //setErrorMsg
            return false;
        }

        RSUN_get_results(&((*mySunPosition).relOptAirMass), &((*mySunPosition).relOptAirMassCorr), &((*mySunPosition).azimuth), &sunCosIncidenceCompl, &cosZen, &((*mySunPosition).elevation), &((*mySunPosition).elevationRefr), &((*mySunPosition).extraIrradianceHorizontal), &((*mySunPosition).extraIrradianceNormal), &etrTilt, &prime, &sbcf, &sunRiseMinutes, &sunSetMinutes, &unPrime, &zenRef);

        (*mySunPosition).incidence = float(MAXVALUE(0, RAD_TO_DEG * ((PI / 2.0) - acos(sunCosIncidenceCompl))));
        (*mySunPosition).rise = sunRiseMinutes * 60.f;
        (*mySunPosition).set = sunSetMinutes * 60.f;
        return true;
    }


    bool computeRadiationPointBrooks(Crit3DRadiationSettings* mySettings, TradPoint* myPoint, Crit3DDate* myDate,
                                     float currentSolarTime, float myClearSkyTransmissivity, float myTransmissivity)
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
        myPoint->aspect *= DEG_TO_RAD;
        myPoint->lat *= DEG_TO_RAD;

        timeAdjustment = (279.575 + 0.986 * myDoy) * DEG_TO_RAD;

        timeEq = (-104.7 * sin(timeAdjustment) + 596.2 * sin(2 * timeAdjustment)
                  + 4.3 * sin(3 * timeAdjustment) - 12.7 * sin(4 * timeAdjustment)
                  - 429.3 * cos(timeAdjustment) - 2 * cos(2 * timeAdjustment)
                  + 19.3 * cos(3 * timeAdjustment)) / 3600.0 ;

        solarDeclination = 0.4102 * sin(2.0 * PI / 365.0 * (myDoy - 80));

        //controllare i segni:
        correctionLong = ((mySettings->gisSettings->timeZone * 15) - myPoint->lon) / 15.0;

        solarTime = currentSolarTime - correctionLong + timeEq;
        if (solarTime > 24)
        {
            solarTime -= 24;
        }
        else if (solarTime < 0)
        {
            solarTime += 24;
        }

        elevationAngle = asin(sin(myPoint->lat) * sin(solarDeclination)
                                 + cos(myPoint->lat) * cos(solarDeclination)
                                 * cos((PI / 12) * (solarTime - 12)));

        extraTerrestrialRad = MAXVALUE(0, SOLAR_CONSTANT * sin(elevationAngle));

        azimuthSouth = acos((sin(elevationAngle)
                                * sin(myPoint->lat) - sin(solarDeclination))
                               / (cos(elevationAngle) * cos(myPoint->lat)));

        azimuthNorth = (solarTime>12) ? PI + azimuthSouth : PI - azimuthSouth;
        incidenceAngle = MAXVALUE(0, asin(getSinDecimalDegree(float(myPoint->slope)) *
                                        cos(elevationAngle) * cos(azimuthNorth - float(myPoint->aspect))
                                        + getCosDecimalDegree(float(myPoint->slope)) * sin(elevationAngle)));

        float Tt = myClearSkyTransmissivity;
        float td = 0.1f;
        if (mySettings->getRealSky())
        {
            if (myTransmissivity != NODATA)
            {
                separateTransmissivity (myClearSkyTransmissivity, myTransmissivity, &td, &Tt);
            }
        }

        coeffBH = MAXVALUE(0, (Tt - td));

        radDiffuse = extraTerrestrialRad * td;

        if (mySettings->getTilt() == 0)
        {
            radBeam = extraTerrestrialRad * coeffBH;
            radReflected = 0;
        }
        else
        {
            radBeam = extraTerrestrialRad * coeffBH * MAXVALUE(0, sin(incidenceAngle) / sin(elevationAngle));
            //aggiungere Snow albedo!
            //Muneer 1997
            radReflected = extraTerrestrialRad * Tt * 0.2 * (1.0 - getCosDecimalDegree(float(myPoint->slope))) / 2.0;
        }

        radTotal = radDiffuse + radBeam + radReflected;
        myPoint->global = radTotal;
        myPoint->beam = radBeam;
        myPoint->diffuse = radDiffuse;
        myPoint->reflected = radReflected;

        return true;
    }


    bool preConditionsRadiationGrid(Crit3DRadiationMaps* radiationMaps)
    {
        if (! radiationMaps->latMap->isLoaded || ! radiationMaps->lonMap->isLoaded) return false;

        if (! radiationMaps->slopeMap->isLoaded || ! radiationMaps->aspectMap->isLoaded) return false;

        return true;
    }


    bool computeRadiationGridPresentTime(Crit3DRadiationSettings* mySettings, const gis::Crit3DRasterGrid& myDEM,
                                         Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myCrit3DTime)
    {
        if (! preConditionsRadiationGrid(radiationMaps))
            return false;        

        if (mySettings->getAlgorithm() == RADIATION_ALGORITHM_RSUN)
        {
            return computeRadiationGridRsun(mySettings, myDEM, radiationMaps, myCrit3DTime);
        }
        /*else if (mySettings->getAlgorithm() == RADIATION_ALGORITHM_BROOKS)
        {
            // to do
            return false;
        }*/
        else
            return false;
    }


    float computePointTransmissivity(Crit3DRadiationSettings* mySettings, const gis::Crit3DPoint& myPoint, Crit3DTime myTime,
                                     float* measuredRad, int windowWidth, int timeStepSecond, const gis::Crit3DRasterGrid& myDEM)
    {
        if (windowWidth % 2 != 1) return NODATA;

        int intervalCenter = (windowWidth-1)/2;

        if (measuredRad[intervalCenter] == NODATA) return NODATA;

        double latDegrees, lonDegrees;
        float ratioTransmissivity;
        TradPoint myRadPoint;
        TsunPosition mySunPosition;
        float myLinke, myAlbedo;
        float  myClearSkyTransmissivity;
        float myTransmissivity;

        Crit3DTime backwardTime;
        Crit3DTime forwardTime;

        /*! assign topographic height and coordinates */
        myRadPoint.x = myPoint.utm.x;
        myRadPoint.y = myPoint.utm.y;
        myRadPoint.height = myPoint.z;
        if (myPoint.z == NODATA && myPoint.utm.isInsideGrid(*(myDEM.header)))
            myRadPoint.height = double(gis::getValueFromXY(myDEM, myRadPoint.x, myRadPoint.y));

        /*! suppose radiometers are horizontal */
        myRadPoint.aspect = 0.;
        myRadPoint.slope = 0.;

        gis::getLatLonFromUtm(*(mySettings->gisSettings), myPoint.utm.x, myPoint.utm.y, &latDegrees, &lonDegrees);
        myRadPoint.lat = latDegrees;
        myRadPoint.lon = lonDegrees;

        myLinke = readLinke(mySettings, myPoint);
        myAlbedo = readAlbedo(mySettings, myPoint);
        myClearSkyTransmissivity = mySettings->getClearSky();

        int backwardTimeStep,forwardTimeStep;
        backwardTimeStep = forwardTimeStep = 0;
        backwardTime = forwardTime = myTime;

        float sumMeasuredRad = measuredRad[intervalCenter];

        computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, myTime, myLinke, myAlbedo,
                                myClearSkyTransmissivity, myClearSkyTransmissivity, &mySunPosition, &myRadPoint, myDEM);

        float sumPotentialRad = float(myRadPoint.global);

        for (int windowIndex = (intervalCenter - 1); windowIndex >= 0; windowIndex--)
        {
            backwardTimeStep -= timeStepSecond;
            forwardTimeStep += timeStepSecond;
            backwardTime = myTime.addSeconds(backwardTimeStep);
            forwardTime = myTime.addSeconds(forwardTimeStep);

            if (measuredRad[windowIndex] != NODATA)
            {
                sumMeasuredRad += measuredRad[windowIndex];
                computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, backwardTime,
                                          myLinke, myAlbedo, myClearSkyTransmissivity, myClearSkyTransmissivity,
                                          &mySunPosition, &myRadPoint, myDEM);
                sumPotentialRad += float(myRadPoint.global);
            }
            if (measuredRad[windowWidth-windowIndex-1] != NODATA)
            {
                sumMeasuredRad+= measuredRad[windowWidth-windowIndex-1];
                computeRadiationPointRsun(mySettings, TEMPERATURE_DEFAULT, PRESSURE_SEALEVEL, forwardTime,
                                          myLinke, myAlbedo, myClearSkyTransmissivity, myClearSkyTransmissivity,
                                          &mySunPosition, &myRadPoint, myDEM);
                sumPotentialRad+= float(myRadPoint.global);
            }
        }

        ratioTransmissivity = MAXVALUE(sumMeasuredRad/sumPotentialRad, float(0.0));
        myTransmissivity = ratioTransmissivity * myClearSkyTransmissivity;

        /*! transmissivity can't be over 0.85 */
        return MINVALUE(myTransmissivity, float(0.85));
    }
}

