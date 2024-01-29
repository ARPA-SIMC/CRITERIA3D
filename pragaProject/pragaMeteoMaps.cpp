#include "basicMath.h"
#include "gis.h"
#include "meteo.h"
#include "pragaMeteoMaps.h"


PragaHourlyMeteoMaps::PragaHourlyMeteoMaps(const gis::Crit3DRasterGrid& DEM)
{
    mapHourlyWindVectorInt = new gis::Crit3DRasterGrid;
    mapHourlyWindVectorDir = new gis::Crit3DRasterGrid;
    mapHourlyWindVectorX = new gis::Crit3DRasterGrid;
    mapHourlyWindVectorY = new gis::Crit3DRasterGrid;

    mapHourlyWindVectorInt->initializeGrid(DEM);
    mapHourlyWindVectorDir->initializeGrid(DEM);
    mapHourlyWindVectorX->initializeGrid(DEM);
    mapHourlyWindVectorY->initializeGrid(DEM);
}


PragaHourlyMeteoMaps::~PragaHourlyMeteoMaps()
{
    this->clear();
}


void PragaHourlyMeteoMaps::clear()
{
    mapHourlyWindVectorInt->clear();
    mapHourlyWindVectorDir->clear();
    mapHourlyWindVectorX->clear();
    mapHourlyWindVectorY->clear();

    delete mapHourlyWindVectorInt;
    delete mapHourlyWindVectorDir;
    delete mapHourlyWindVectorX;
    delete mapHourlyWindVectorY;
}


void PragaHourlyMeteoMaps::initialize()
{
    mapHourlyWindVectorInt->emptyGrid();
    mapHourlyWindVectorDir->emptyGrid();
    mapHourlyWindVectorX->emptyGrid();
    mapHourlyWindVectorY->emptyGrid();
}


gis::Crit3DRasterGrid* PragaHourlyMeteoMaps::getMapFromVar(meteoVariable myVar)
{
    if (myVar == windVectorIntensity)
        return mapHourlyWindVectorInt;
    else if (myVar == windVectorDirection)
        return mapHourlyWindVectorDir;
    else if (myVar == windVectorX)
        return mapHourlyWindVectorX;
    else if (myVar == windVectorY)
        return mapHourlyWindVectorY;
    else
        return nullptr;
}

bool PragaHourlyMeteoMaps::computeWindVector()
{
    if (! mapHourlyWindVectorX->isLoaded || ! mapHourlyWindVectorY->isLoaded) return false;

    float intensity, direction;

    for (long row = 0; row < mapHourlyWindVectorX->header->nrRows; row++)
        for (long col = 0; col < mapHourlyWindVectorX->header->nrCols; col++)
        {
            mapHourlyWindVectorInt->value[row][col] = mapHourlyWindVectorInt->header->flag;
            mapHourlyWindVectorDir->value[row][col] = mapHourlyWindVectorDir->header->flag;

            if (! isEqual(mapHourlyWindVectorX->value[row][col], mapHourlyWindVectorX->header->flag)
                    && ! isEqual(mapHourlyWindVectorY->value[row][col], mapHourlyWindVectorY->header->flag))
            {
                if (computeWindPolar(mapHourlyWindVectorX->value[row][col], mapHourlyWindVectorX->value[row][col], &intensity, &direction))
                {
                    mapHourlyWindVectorInt->value[row][col] = intensity;
                    mapHourlyWindVectorDir->value[row][col] = direction;
                }
            }
        }

    return (gis::updateMinMaxRasterGrid(mapHourlyWindVectorInt) && gis::updateMinMaxRasterGrid(mapHourlyWindVectorDir));
}
