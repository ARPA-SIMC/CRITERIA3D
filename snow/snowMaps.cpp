/*!
    \copyright Fausto Tomei, Gabriele Antolini,
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
    ftomei@arpae.it
*/

#include "commonConstants.h"
#include "basicMath.h"
#include "snowMaps.h"
#include "gis.h"


Crit3DSnowMaps::Crit3DSnowMaps()
{
    _snowWaterEquivalentMap = new gis::Crit3DRasterGrid;
    _iceContentMap = new gis::Crit3DRasterGrid;
    _liquidWaterContentMap = new gis::Crit3DRasterGrid;
    _internalEnergyMap = new gis::Crit3DRasterGrid;
    _surfaceEnergyMap = new gis::Crit3DRasterGrid;
    _snowSurfaceTempMap = new gis::Crit3DRasterGrid;
    _ageOfSnowMap = new gis::Crit3DRasterGrid;

    _snowFallMap = new gis::Crit3DRasterGrid;
    _snowMeltMap = new gis::Crit3DRasterGrid;
    _sensibleHeatMap = new gis::Crit3DRasterGrid;
    _latentHeatMap = new gis::Crit3DRasterGrid;

    _initSoilPackTemp = NODATA;
    _initSnowSurfaceTemp = NODATA;

    isInitialized = false;
}


Crit3DSnowMaps::~Crit3DSnowMaps()
{
    this->clear();
}


void Crit3DSnowMaps::clear()
{
    _snowWaterEquivalentMap->clear();
    _iceContentMap->clear();
    _liquidWaterContentMap->clear();
    _internalEnergyMap->clear();
    _surfaceEnergyMap->clear();
    _snowSurfaceTempMap->clear();
    _ageOfSnowMap->clear();

    _snowFallMap->clear();
    _snowMeltMap->clear();
    _sensibleHeatMap->clear();
    _latentHeatMap->clear();

    _initSoilPackTemp = NODATA;
    _initSnowSurfaceTemp = NODATA;

    isInitialized = false;
}


void Crit3DSnowMaps::initializeSnowMaps(const gis::Crit3DRasterGrid &dtm, double skinThickness)
{
    _snowFallMap->initializeGrid(dtm);
    _snowMeltMap->initializeGrid(dtm);
    _sensibleHeatMap->initializeGrid(dtm);
    _latentHeatMap->initializeGrid(dtm);

    _iceContentMap->initializeGrid(dtm);
    _liquidWaterContentMap->initializeGrid(dtm);
    _internalEnergyMap->initializeGrid(dtm);
    _surfaceEnergyMap->initializeGrid(dtm);
    _snowSurfaceTempMap->initializeGrid(dtm);
    _ageOfSnowMap->initializeGrid(dtm);

    // TODO: pass initial temperature
    _initSoilPackTemp = 3.4;
    _initSnowSurfaceTemp = 5.0;

    _snowWaterEquivalentMap->initializeGrid(dtm);
    // initialize with zero values
    _snowWaterEquivalentMap->setConstantValueWithBase(0, dtm);

    resetSnowModel(skinThickness);

    isInitialized = true;
}


void Crit3DSnowMaps::updateMapRowCol(Crit3DSnow &snowPoint, int row, int col)
{
    _snowWaterEquivalentMap->value[row][col] = float(snowPoint.getSnowWaterEquivalent());
    _iceContentMap->value[row][col] = float(snowPoint.getIceContent());
    _liquidWaterContentMap->value[row][col] = float(snowPoint.getLiquidWaterContent());
    _internalEnergyMap->value[row][col] = float(snowPoint.getInternalEnergy());
    _surfaceEnergyMap->value[row][col] = float(snowPoint.getSurfaceEnergy());
    _snowSurfaceTempMap->value[row][col] = float(snowPoint.getSnowSurfaceTemp());
    _ageOfSnowMap->value[row][col] = float(snowPoint.getAgeOfSnow());

    _snowFallMap->value[row][col] = float(snowPoint.getSnowFall());
    _snowMeltMap->value[row][col] = float(snowPoint.getSnowMelt());
    _sensibleHeatMap->value[row][col] = float(snowPoint.getSensibleHeat());
    _latentHeatMap->value[row][col] = float(snowPoint.getLatentHeat());
}


void Crit3DSnowMaps::flagMapRowCol(int row, int col)
{
    _snowWaterEquivalentMap->value[row][col] = _snowWaterEquivalentMap->header->flag;
    _iceContentMap->value[row][col] = _iceContentMap->header->flag;
    _liquidWaterContentMap->value[row][col] = _liquidWaterContentMap->header->flag;
    _internalEnergyMap->value[row][col] = _internalEnergyMap->header->flag;
    _surfaceEnergyMap->value[row][col] = _surfaceEnergyMap->header->flag;
    _snowSurfaceTempMap->value[row][col] = _snowSurfaceTempMap->header->flag;
    _ageOfSnowMap->value[row][col] = _ageOfSnowMap->header->flag;

    _snowFallMap->value[row][col] = _snowFallMap->header->flag;
    _snowMeltMap->value[row][col] = _snowMeltMap->header->flag;
    _sensibleHeatMap->value[row][col] = _sensibleHeatMap->header->flag;
    _latentHeatMap->value[row][col] = _latentHeatMap->header->flag;
}


void Crit3DSnowMaps::updateRangeMaps()
{
    gis::updateMinMaxRasterGrid(_snowWaterEquivalentMap);
    gis::updateMinMaxRasterGrid(_iceContentMap);
    gis::updateMinMaxRasterGrid(_liquidWaterContentMap);
    gis::updateMinMaxRasterGrid(_internalEnergyMap);
    gis::updateMinMaxRasterGrid(_surfaceEnergyMap);
    gis::updateMinMaxRasterGrid(_snowSurfaceTempMap);
    gis::updateMinMaxRasterGrid(_ageOfSnowMap);

    gis::updateMinMaxRasterGrid(_snowFallMap);
    gis::updateMinMaxRasterGrid(_snowMeltMap);
    gis::updateMinMaxRasterGrid(_sensibleHeatMap);
    gis::updateMinMaxRasterGrid(_latentHeatMap);
}


void Crit3DSnowMaps::setPoint(Crit3DSnow &snowPoint, int row, int col)
{
    snowPoint.setSnowWaterEquivalent(_snowWaterEquivalentMap->value[row][col]);
    snowPoint.setIceContent(_iceContentMap->value[row][col]);
    snowPoint.setLiquidWaterContent(_liquidWaterContentMap->value[row][col]);
    snowPoint.setInternalEnergy(_internalEnergyMap->value[row][col]);
    snowPoint.setSurfaceEnergy(_surfaceEnergyMap->value[row][col]);
    snowPoint.setSnowSurfaceTemp(_snowSurfaceTempMap->value[row][col]);
    snowPoint.setAgeOfSnow(_ageOfSnowMap->value[row][col]);
}


void Crit3DSnowMaps::resetSnowModel(double skinThickness)
{
    float initSWE;                  //  [mm]
    int surfaceBulkDensity;         //  [kg m-3]

    // TODO pass real bulk density
    surfaceBulkDensity = DEFAULT_BULK_DENSITY;

    for (long row = 0; row < _snowWaterEquivalentMap->header->nrRows; row++)
    {
        for (long col = 0; col < _snowWaterEquivalentMap->header->nrCols; col++)
        {
            initSWE = _snowWaterEquivalentMap->value[row][col];
            if (! isEqual(initSWE, _snowWaterEquivalentMap->header->flag))
            {
                _iceContentMap->value[row][col] = initSWE;
                _liquidWaterContentMap->value[row][col] = 0;
                _ageOfSnowMap->value[row][col] = NODATA;

                _snowSurfaceTempMap->value[row][col] = float(_initSnowSurfaceTemp);

                if (initSWE > 0)
                {
                    _surfaceEnergyMap->value[row][col] = float(computeSurfaceEnergySnow(_initSnowSurfaceTemp, skinThickness));
                }
                else
                {
                    _surfaceEnergyMap->value[row][col] = float(computeSurfaceEnergySoil(_initSnowSurfaceTemp, skinThickness));
                }

                _internalEnergyMap->value[row][col] = float(computeInternalEnergy(_initSoilPackTemp, surfaceBulkDensity, initSWE/1000.));

                // output
                _snowFallMap->value[row][col] = 0;
                _snowMeltMap->value[row][col] = 0;
                _sensibleHeatMap->value[row][col] = 0;
                _latentHeatMap->value[row][col] = 0;
            }
        }
    }
}


// --------------------------- output ----------------------------

gis::Crit3DRasterGrid* Crit3DSnowMaps::getSnowWaterEquivalentMap()
{
    return _snowWaterEquivalentMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getIceContentMap()
{
    return _iceContentMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getLWContentMap()
{
    return _liquidWaterContentMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getInternalEnergyMap()
{
    return _internalEnergyMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getSurfaceEnergyMap()
{
    return _surfaceEnergyMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getSnowSurfaceTempMap()
{
    return _snowSurfaceTempMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getAgeOfSnowMap()
{
    return _ageOfSnowMap;
}


gis::Crit3DRasterGrid* Crit3DSnowMaps::getSnowFallMap()
{
    return _snowFallMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getSnowMeltMap()
{
    return _snowMeltMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getSensibleHeatMap()
{
    return _sensibleHeatMap;
}

gis::Crit3DRasterGrid* Crit3DSnowMaps::getLatentHeatMap()
{
    return _latentHeatMap;
}

