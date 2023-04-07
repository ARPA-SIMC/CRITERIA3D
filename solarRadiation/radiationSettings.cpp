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
#include "radiationSettings.h"


Crit3DRadiationSettings::Crit3DRadiationSettings()
{
    initialize();
}

void Crit3DRadiationSettings::initialize()
{
    gisSettings = new gis::Crit3DGisSettings();
    realSky = true;
    shadowing = true;
    linkeMode = PARAM_MODE_FIXED;
    linke = 4.f;
    albedoMode = PARAM_MODE_FIXED;
    albedo = 0.2f;
    tiltMode = TILT_TYPE_DEM;
    tilt = 0;
    aspect = 0;

    algorithm = RADIATION_ALGORITHM_RSUN;
    realSkyAlgorithm = RADIATION_REALSKY_LINKE;
    clearSky = CLEAR_SKY_TRANSMISSIVITY_DEFAULT;

    LinkeMonthly.resize(12);
    AlbedoMonthly.resize(12);

    for (unsigned int i=0; i<12; i++)
    {
        LinkeMonthly[i] = NODATA;
        AlbedoMonthly[i] = NODATA;
    }

    linkeMapName = "";
    albedoMapName = "";
    linkeMap = nullptr;
    albedoMap = nullptr;
}

TradiationRealSkyAlgorithm Crit3DRadiationSettings::getRealSkyAlgorithm() const
{
    return realSkyAlgorithm;
}

void Crit3DRadiationSettings::setRealSkyAlgorithm(const TradiationRealSkyAlgorithm &value)
{
    realSkyAlgorithm = value;
}

float Crit3DRadiationSettings::getClearSky() const
{
    return clearSky;
}

void Crit3DRadiationSettings::setClearSky(float value)
{
    clearSky = value;
}

bool Crit3DRadiationSettings::getRealSky() const
{
    return realSky;
}

void Crit3DRadiationSettings::setRealSky(bool value)
{
    realSky = value;
}

bool Crit3DRadiationSettings::getShadowing() const
{
    return shadowing;
}

void Crit3DRadiationSettings::setShadowing(bool value)
{
    shadowing = value;
}

float Crit3DRadiationSettings::getLinke() const
{
    return linke;
}

float Crit3DRadiationSettings::getLinke(int row, int col) const
{
    if (linkeMode == PARAM_MODE_FIXED) return linke;

    float myLinke = NODATA;

    if (linkeMap != nullptr && linkeMap->isLoaded)
        if (gis::isOutOfGridRowCol(row, col, *linkeMap))
            myLinke = linkeMap->value[row][col];

    return myLinke;
}

float Crit3DRadiationSettings::getLinke(const gis::Crit3DPoint& myPoint) const
{
    if (linkeMode == PARAM_MODE_FIXED) return linke;

    float myLinke = NODATA;

    if (linkeMap != nullptr && linkeMap->isLoaded)
    {
        int row, col;
        linkeMap->getRowCol(myPoint.utm.x, myPoint.utm.y, row, col);
        if (gis::isOutOfGridRowCol(row, col, *linkeMap))
            myLinke = linkeMap->value[row][col];
    }

    return myLinke;
}

void Crit3DRadiationSettings::setLinke(float value)
{
    linke = value;
}

float Crit3DRadiationSettings::getAlbedo() const
{
    return albedo;
}

float Crit3DRadiationSettings::getAlbedo(int row, int col) const
{
    if (albedoMode == PARAM_MODE_FIXED) return albedo;

    float myAlbedo = NODATA;

    if (albedoMap != nullptr && albedoMap->isLoaded)
        if (gis::isOutOfGridRowCol(row, col, *albedoMap))
            myAlbedo = albedoMap->value[row][col];

    return myAlbedo;
}

float Crit3DRadiationSettings::getAlbedo(const gis::Crit3DPoint& myPoint) const
{
    if (albedoMode == PARAM_MODE_FIXED) return albedo;

    float myAlbedo = NODATA;

    if (albedoMap != nullptr && albedoMap->isLoaded)
    {
        int row, col;
        albedoMap->getRowCol(myPoint.utm.x, myPoint.utm.y, row, col);
        if (gis::isOutOfGridRowCol(row, col, *albedoMap))
            myAlbedo = albedoMap->value[row][col];
    }

    return myAlbedo;
}

void Crit3DRadiationSettings::setAlbedo(float value)
{
    albedo = value;
}

float Crit3DRadiationSettings::getTilt() const
{
    return tilt;
}

void Crit3DRadiationSettings::setTilt(float value)
{
    tilt = value;
}

float Crit3DRadiationSettings::getAspect() const
{
    return aspect;
}

void Crit3DRadiationSettings::setAspect(float value)
{
    aspect = value;
}

TradiationAlgorithm Crit3DRadiationSettings::getAlgorithm() const
{
    return algorithm;
}

void Crit3DRadiationSettings::setAlgorithm(const TradiationAlgorithm &value)
{
    algorithm = value;
}

TparameterMode Crit3DRadiationSettings::getLinkeMode() const
{
    return linkeMode;
}

void Crit3DRadiationSettings::setLinkeMode(const TparameterMode &value)
{
    linkeMode = value;
}

TparameterMode Crit3DRadiationSettings::getAlbedoMode() const
{
    return albedoMode;
}

void Crit3DRadiationSettings::setAlbedoMode(const TparameterMode &value)
{
    albedoMode = value;
}

TtiltMode Crit3DRadiationSettings::getTiltMode() const
{
    return tiltMode;
}

void Crit3DRadiationSettings::setTiltMode(const TtiltMode &value)
{
    tiltMode = value;
}

gis::Crit3DRasterGrid *Crit3DRadiationSettings::getLinkeMap() const
{
    return linkeMap;
}

void Crit3DRadiationSettings::setLinkeMap(gis::Crit3DRasterGrid *value)
{
    linkeMap = value;
}

gis::Crit3DRasterGrid *Crit3DRadiationSettings::getAlbedoMap() const
{
    return albedoMap;
}

void Crit3DRadiationSettings::setAlbedoMap(gis::Crit3DRasterGrid *value)
{
    albedoMap = value;
}

std::string Crit3DRadiationSettings::getLinkeMapName() const
{
    return linkeMapName;
}

void Crit3DRadiationSettings::setLinkeMapName(const std::string &value)
{
    linkeMapName = value;
}

std::string Crit3DRadiationSettings::getAlbedoMapName() const
{
    return albedoMapName;
}

void Crit3DRadiationSettings::setAlbedoMapName(const std::string &value)
{
    albedoMapName = value;
}

void Crit3DRadiationSettings::setLinkeMonthly(std::vector<float> myLinke)
{
    LinkeMonthly = myLinke;
}

void Crit3DRadiationSettings::setAlbedoMonthly(std::vector<float> myAlbedo)
{
    AlbedoMonthly = myAlbedo;
}

std::vector<float> Crit3DRadiationSettings::getLinkeMonthly() const
{
    return LinkeMonthly;
}

void Crit3DRadiationSettings::setGisSettings(const gis::Crit3DGisSettings* mySettings)
{
    gisSettings = new gis::Crit3DGisSettings();
    gisSettings->utmZone = mySettings->utmZone;
    gisSettings->timeZone = mySettings->timeZone;
    gisSettings->isUTC = mySettings->isUTC;
}

std::string getKeyStringRadAlgorithm(TradiationAlgorithm value)
{
    std::map<std::string, TradiationAlgorithm>::const_iterator it;
    std::string key = "";

    for (it = radAlgorithmToString.begin(); it != radAlgorithmToString.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string getKeyStringParamMode(TparameterMode value)
{
    std::map<std::string, TparameterMode>::const_iterator it;
    std::string key = "";

    for (it = paramModeToString.begin(); it != paramModeToString.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string getKeyStringTiltMode(TtiltMode value)
{
    std::map<std::string, TtiltMode>::const_iterator it;
    std::string key = "";

    for (it = tiltModeToString.begin(); it != tiltModeToString.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string getKeyStringRealSky(TradiationRealSkyAlgorithm value)
{
    std::map<std::string, TradiationRealSkyAlgorithm>::const_iterator it;
    std::string key = "";

    for (it = realSkyAlgorithmToString.begin(); it != realSkyAlgorithmToString.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

