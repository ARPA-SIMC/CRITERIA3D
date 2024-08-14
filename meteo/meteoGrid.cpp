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
*/

#include "commonConstants.h"
#include "basicMath.h"
#include "meteoGrid.h"
#include "statistics.h"
#include "math.h"

Crit3DMeteoGridStructure::Crit3DMeteoGridStructure()
{    
}

std::string Crit3DMeteoGridStructure::name() const
{
    return _name;
}

void Crit3DMeteoGridStructure::setName(const std::string &name)
{
    _name = name;
}

gis::Crit3DLatLonHeader Crit3DMeteoGridStructure::header() const
{
    return _header;
}

void Crit3DMeteoGridStructure::setHeader(const gis::Crit3DLatLonHeader &header)
{
    _header = header;
}

int Crit3DMeteoGridStructure::dataType() const
{
    return _dataType;
}

void Crit3DMeteoGridStructure::setDataType(int dataType)
{
    _dataType = dataType;
}

bool Crit3DMeteoGridStructure::isRegular() const
{
    return _isRegular;
}

void Crit3DMeteoGridStructure::setIsRegular(bool isRegular)
{
    _isRegular = isRegular;
}

bool Crit3DMeteoGridStructure::isTIN() const
{
    return _isTIN;
}

void Crit3DMeteoGridStructure::setIsTIN(bool isTIN)
{
    _isTIN = isTIN;
}

bool Crit3DMeteoGridStructure::isUTM() const
{
    return _isUTM;
}

void Crit3DMeteoGridStructure::setIsUTM(bool isUTM)
{
    _isUTM = isUTM;
}

bool Crit3DMeteoGridStructure::isLoaded() const
{
    return _isLoaded;
}

void Crit3DMeteoGridStructure::setIsLoaded(bool isLoaded)
{
    _isLoaded = isLoaded;
}

bool Crit3DMeteoGridStructure::isFixedFields() const
{
    return _isFixedFields;
}

void Crit3DMeteoGridStructure::setIsFixedFields(bool isFixedFields)
{
    _isFixedFields = isFixedFields;
}

bool Crit3DMeteoGridStructure::isHourlyDataAvailable() const
{
    return _isHourlyDataAvailable;
}

void Crit3DMeteoGridStructure::setIsHourlyDataAvailable(bool isHourlyDataAvailable)
{
    _isHourlyDataAvailable = isHourlyDataAvailable;
}

bool Crit3DMeteoGridStructure::isDailyDataAvailable() const
{
    return _isDailyDataAvailable;
}

void Crit3DMeteoGridStructure::setIsDailyDataAvailable(bool isDailyDataAvailable)
{
    _isDailyDataAvailable = isDailyDataAvailable;
}

bool Crit3DMeteoGridStructure::isEnsemble() const
{
    return _isEnsemble;
}

void Crit3DMeteoGridStructure::setIsEnsemble(bool isEnsemble)
{
    _isEnsemble = isEnsemble;
}

int Crit3DMeteoGridStructure::nrMembers() const
{
    return _nrMembers;
}

int Crit3DMeteoGridStructure::nrRow() const
{
    return _header.nrRows;
}

int Crit3DMeteoGridStructure::nrCol() const
{
    return _header.nrCols;
}

void Crit3DMeteoGridStructure::setNrMembers(int nrMembers)
{
    _nrMembers = nrMembers;
}

Crit3DMeteoGridStructure Crit3DMeteoGrid::gridStructure() const
{
    return _gridStructure;
}

Crit3DMeteoGrid::Crit3DMeteoGrid()
{
    _isAggregationDefined = false;
    _gisSettings.utmZone = 32;
    _isElabValue = false;
    _firstDate = Crit3DDate(1,1,1800);
    _lastDate = Crit3DDate(1,1,1800);
}


Crit3DMeteoGrid::~Crit3DMeteoGrid()
{
    for (unsigned int i = 0; i < _meteoPoints.size(); i++)
    {
        std::vector<Crit3DMeteoPoint*> meteoPointVector = _meteoPoints[i];

        for (unsigned int j = 0; j < meteoPointVector.size(); j++)
        {
            free(meteoPointVector[j]);
        }
    }

    dataMeteoGrid.clear();
}


bool Crit3DMeteoGrid::createRasterGrid()
{
    if (_gridStructure.isUTM())
    {
        dataMeteoGrid.header->cellSize = _gridStructure.header().dx;
        dataMeteoGrid.header->llCorner.x = _gridStructure.header().llCorner.longitude;
        dataMeteoGrid.header->llCorner.y = _gridStructure.header().llCorner.latitude;
    }
    else
    {
        dataMeteoGrid.header->cellSize = NODATA;
        dataMeteoGrid.header->llCorner.x = NODATA;
        dataMeteoGrid.header->llCorner.y = NODATA;
    }

    dataMeteoGrid.header->nrCols = _gridStructure.header().nrCols;
    dataMeteoGrid.header->nrRows = _gridStructure.header().nrRows;
    dataMeteoGrid.header->flag = NO_ACTIVE;

    dataMeteoGrid.initializeGrid(NODATA);

    // not active cells
    for (size_t i=0; i < size_t(_gridStructure.header().nrRows); i++)
        for (size_t j=0; j < size_t(_gridStructure.header().nrCols); j++)
             if (! _meteoPoints[i][j]->active)
                 dataMeteoGrid.value[size_t(_gridStructure.header().nrRows)-1-i][j] = NO_ACTIVE;

    dataMeteoGrid.isLoaded = true;
    return true;
}


std::vector<std::vector<Crit3DMeteoPoint *> > Crit3DMeteoGrid::meteoPoints() const
{
    return _meteoPoints;
}

Crit3DMeteoPoint Crit3DMeteoGrid::meteoPoint(unsigned row, unsigned col)
{
    return *(_meteoPoints[row][col]);
}

Crit3DMeteoPoint* Crit3DMeteoGrid::meteoPointPointer(unsigned row, unsigned col)
{
    return _meteoPoints[row][col];
}

void Crit3DMeteoGrid::setMeteoPoints(const std::vector<std::vector<Crit3DMeteoPoint *> > &meteoPoints)
{
    _meteoPoints = meteoPoints;
}

void Crit3DMeteoGrid::initMeteoPoints(int nRow, int nCol)
{

    _meteoPoints.reserve(nRow);

    for (int i = 0; i < nRow; i++)
    {
        std::vector<Crit3DMeteoPoint*> meteoPointVector;
        meteoPointVector.reserve(nCol);

        for (int j = 0; j < nCol; j++)
        {
            Crit3DMeteoPoint* meteoPoint = new Crit3DMeteoPoint;
            meteoPoint->active = false;
            meteoPoint->selected = false;
            meteoPointVector.push_back(meteoPoint);
        }

        _meteoPoints.push_back(meteoPointVector);
    }
}

void Crit3DMeteoGrid::setActive(unsigned int row,unsigned int col, bool active)
{
    _meteoPoints[row][col]->active = active;
}


// for visualization raster
void Crit3DMeteoGrid::fillMeteoPoint(unsigned int row, unsigned int col, const std::string& code, const std::string& name, int height, bool active)
{

    _meteoPoints[row][col]->id = code;
    _meteoPoints[row][col]->name = name;
    _meteoPoints[row][col]->point.z = height;
    _meteoPoints[row][col]->active = active;

    if (_gridStructure.isRegular())
    {
        if (_gridStructure.isUTM())
        {
            _meteoPoints[row][col]->point.utm.x = _gridStructure.header().llCorner.longitude + _gridStructure.header().dx * (col + 0.5);
            _meteoPoints[row][col]->point.utm.y = _gridStructure.header().llCorner.latitude + _gridStructure.header().dy * (row + 0.5);
            gis::utmToLatLon(_gisSettings.utmZone, _gisSettings.startLocation.latitude, _meteoPoints[row][col]->point.utm.x, _meteoPoints[row][col]->point.utm.y, &(_meteoPoints[row][col]->latitude), &(_meteoPoints[row][col]->longitude));
        }
        else
        {
            _meteoPoints[row][col]->longitude = _gridStructure.header().llCorner.longitude + _gridStructure.header().dx * (col + 0.5);
            _meteoPoints[row][col]->latitude = _gridStructure.header().llCorner.latitude + _gridStructure.header().dy * (row + 0.5);

            gis::Crit3DUtmPoint utmPoint;
            gis::Crit3DGeoPoint geoPoint(_meteoPoints[row][col]->latitude, _meteoPoints[row][col]->longitude);
            gis::getUtmFromLatLon(_gisSettings.utmZone, geoPoint, &utmPoint);
            _meteoPoints[row][col]->point.utm.x = utmPoint.x;
            _meteoPoints[row][col]->point.utm.y = utmPoint.y;
        }
    }
}

void Crit3DMeteoGrid::fillCurrentDailyValue(Crit3DDate date, meteoVariable variable, Crit3DMeteoSettings* meteoSettings)
{
    for (unsigned row = 0; row < unsigned(_gridStructure.header().nrRows); row++)
        for (unsigned col = 0; col < unsigned(_gridStructure.header().nrCols); col++)
            _meteoPoints[row][col]->currentValue = _meteoPoints[row][col]->getMeteoPointValueD(date, variable, meteoSettings);
}

void Crit3DMeteoGrid::fillCurrentHourlyValue(Crit3DDate date, int hour, int minute, meteoVariable variable)
{
    for (int row = 0; row < _gridStructure.header().nrRows; row++)
        for(int col = 0; col < _gridStructure.header().nrCols; col++)
            _meteoPoints[row][col]->currentValue = _meteoPoints[row][col]->getMeteoPointValueH(date, hour, minute, variable);
}

void Crit3DMeteoGrid::fillCurrentMonthlyValue(Crit3DDate date, meteoVariable variable)
{
    for (int row = 0; row < _gridStructure.header().nrRows; row++)
        for(int col = 0; col < _gridStructure.header().nrCols; col++)
            _meteoPoints[row][col]->currentValue = _meteoPoints[row][col]->getMeteoPointValueM(date, variable);
}

void Crit3DMeteoGrid::fillMeteoRaster()
{
    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = _meteoPoints[i][j]->currentValue;
             }
        }
    }
}

void Crit3DMeteoGrid::fillMeteoRasterNoData()
{
    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = NODATA;
             }
        }
    }

}

void Crit3DMeteoGrid::fillMeteoRasterElabValue()
{

    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = _meteoPoints[i][j]->elaboration;
             }
        }
    }

}

void Crit3DMeteoGrid::fillMeteoRasterAnomalyValue()
{

    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = _meteoPoints[i][j]->anomaly;

             }
        }
    }

}

void Crit3DMeteoGrid::fillMeteoRasterAnomalyPercValue()
{

    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = _meteoPoints[i][j]->anomalyPercentage;
             }
        }
    }

}

void Crit3DMeteoGrid::fillMeteoRasterClimateValue()
{

    for (int i = 0; i < dataMeteoGrid.header->nrRows; i++)
    {
        for (int j = 0; j < dataMeteoGrid.header->nrCols; j++)
        {
             if (_meteoPoints[i][j]->active)
             {
                 dataMeteoGrid.value[_gridStructure.header().nrRows-1-i][j] = _meteoPoints[i][j]->climate;
             }
        }
    }

}

gis::Crit3DGisSettings Crit3DMeteoGrid::getGisSettings() const
{
    return _gisSettings;
}

void Crit3DMeteoGrid::setGisSettings(const gis::Crit3DGisSettings &gisSettings)
{
    _gisSettings = gisSettings;
}


bool Crit3DMeteoGrid::findMeteoPointFromId(unsigned* row, unsigned* col, const std::string& id)
{
    unsigned i,j;

    for (i = 0; i < unsigned(_gridStructure.header().nrRows); i++)
    {
        for (j = 0; j < unsigned(_gridStructure.header().nrCols); j++)
        {
            if (_meteoPoints[i][j]->id == id)
            {
                *row = i;
                *col = j;
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::existsMeteoPointFromId(const std::string& id)
{
    unsigned i,j;

    for (i = 0; i < unsigned(_gridStructure.header().nrRows); i++)
    {
        for (j = 0; j < unsigned(_gridStructure.header().nrCols); j++)
        {
            if (_meteoPoints[i][j]->id == id)
            {
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::getLatLonFromId(std::string id, double* lat, double* lon)
{
    unsigned i,j;

    for (i = 0; i < unsigned(_gridStructure.header().nrRows); i++)
    {
        for (j = 0; j < unsigned(_gridStructure.header().nrCols); j++)
        {
            if (_meteoPoints[i][j]->id == id)
            {
                *lat = _meteoPoints[i][j]->latitude;
                *lon = _meteoPoints[i][j]->longitude;
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::getLatFromId(std::string id, double* lat)
{
    unsigned i,j;

    for (i = 0; i < unsigned(_gridStructure.header().nrRows); i++)
    {
        for (j = 0; j < unsigned(_gridStructure.header().nrCols); j++)
        {
            if (_meteoPoints[i][j]->id == id)
            {
                *lat = _meteoPoints[i][j]->latitude;
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::getXYZFromId(std::string id, double* x, double* y, double* z)
{
    unsigned i,j;

    for (i = 0; i < unsigned(_gridStructure.header().nrRows); i++)
    {
        for (j = 0; j < unsigned(_gridStructure.header().nrCols); j++)
        {
            if (_meteoPoints[i][j]->id == id)
            {
                *x = _meteoPoints[i][j]->point.utm.x;
                *y = _meteoPoints[i][j]->point.utm.y;
                *z = _meteoPoints[i][j]->point.z;
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::getIdFromLatLon(double lat, double lon, std::string* id)
{

    double dx = _gridStructure.header().dx;
    double dy = _gridStructure.header().dy;
    double latitude, longitude;
    double diffLat, diffLon;
    if (_gridStructure.isRegular())
    {
        if (_gridStructure.isUTM())
        {
            for (unsigned int row = 0; row < unsigned(_gridStructure.header().nrRows); row++)
            {
                for (unsigned int col = 0; col < unsigned(_gridStructure.header().nrCols); col++)
                {
                    double utmEasting;
                    double utmNorthing;
                    gis::latLonToUtmForceZone(_gisSettings.utmZone, lat, lon, &utmEasting, &utmNorthing);
                    latitude = _meteoPoints[row][col]->point.utm.y;
                    longitude = _meteoPoints[row][col]->point.utm.x;
                    diffLat = fabs(utmNorthing-latitude);
                    diffLon = fabs(utmEasting-longitude);
                    if ( diffLat<(0.5*dy) && diffLon<(0.5*dx))
                    {
                        *id = _meteoPoints[row][col]->id;
                        return true;
                    }
                }
            }
        }
        else
        {
            for (unsigned int row = 0; row < unsigned(_gridStructure.header().nrRows); row++)
            {
                for (unsigned int col = 0; col < unsigned(_gridStructure.header().nrCols); col++)
                {
                    latitude = _meteoPoints[row][col]->latitude;
                    longitude = _meteoPoints[row][col]->longitude;
                    diffLat = fabs(lat-latitude);
                    diffLon = fabs(lon-longitude);
                    if ( diffLat<(0.5*dy) && diffLon<(0.5*dx))
                    {
                        *id = _meteoPoints[row][col]->id;
                        return true;
                    }
                }
            }
        }
    }
    else
    {
        // TO DO
    }
    return false;
}

bool Crit3DMeteoGrid::getMeteoPointActiveId(int row, int col, std::string* id)
{
    if (row < _gridStructure.header().nrRows && col < _gridStructure.header().nrCols)
    {
        if (_meteoPoints[row][col]->active)
        {
            *id = _meteoPoints[row][col]->id;
            return true;
        }

    }
    return false;
}

bool Crit3DMeteoGrid::isActiveMeteoPointFromId(const std::string& id)
{

    for (int row = 0; row < _gridStructure.header().nrRows; row++)
    {
        for (int col = 0; col < _gridStructure.header().nrCols; col++)
        {
            if (_meteoPoints[row][col]->active && _meteoPoints[row][col]->id == id)
            {
                return true;
            }
        }
    }
    return false;
}

bool Crit3DMeteoGrid::findFirstActiveMeteoPoint(std::string* id, int* row, int* col)
{
    int firstRow = *row;
    int firstCol = *col;

    for (int i = firstRow; i < _gridStructure.header().nrRows; i++)
    {
        for (int j = firstCol; j < _gridStructure.header().nrCols; j++)
        {
            if (_meteoPoints[i][j]->active)
            {
                *row = i;
                *col = j;
                *id = _meteoPoints[i][j]->id;
                return true;
            }
        }
        firstCol = 0;
    }

    return false;
}

void Crit3DMeteoGrid::findGridAggregationPoints(gis::Crit3DRasterGrid* myDEM)
{
    bool excludeNoData = false;

    for (unsigned row = 0; row < unsigned(_gridStructure.header().nrRows); row++)
        for (unsigned col = 0; col < unsigned(_gridStructure.header().nrCols); col++)
            if (_meteoPoints[row][col]->active)
                assignCellAggregationPoints(row, col, myDEM, excludeNoData);

    _isAggregationDefined = true;
}

void Crit3DMeteoGrid::assignCellAggregationPoints(unsigned row, unsigned col, gis::Crit3DRasterGrid* myDEM, bool excludeNoData)
{

    gis::Crit3DUtmPoint utmLL, utmUR;
    gis::Crit3DUtmPoint utmPoint;
    gis::Crit3DPoint point;
    gis::Crit3DUtmPoint v[4];

    if (_gridStructure.isTIN())
    {
        //TO DO
    }
    else
    {
        if (_gridStructure.isUTM())
        {

            _meteoPoints[row][col]->aggregationPoints.clear();

            utmLL.x = _meteoPoints[row][col]->point.utm.x - (_gridStructure.header().dx / 2) + (myDEM->header->cellSize / 2);
            utmUR.x = _meteoPoints[row][col]->point.utm.x + (_gridStructure.header().dx / 2);
            utmLL.y = _meteoPoints[row][col]->point.utm.y - (_gridStructure.header().dy / 2) + (myDEM->header->cellSize / 2);
            utmUR.y = _meteoPoints[row][col]->point.utm.y + (_gridStructure.header().dy / 2);

            _meteoPoints[row][col]->aggregationPointsMaxNr = 0;

            for (double x = utmLL.x; x < utmUR.x; x=x+myDEM->header->cellSize)
            {
                for (double y = utmLL.y; x < utmUR.y; y=y+myDEM->header->cellSize)
                {
                    _meteoPoints[row][col]->aggregationPointsMaxNr = _meteoPoints[row][col]->aggregationPointsMaxNr + 1;
                    if (!excludeNoData || gis::getValueFromXY(*myDEM, x, y) != myDEM->header->flag )
                    {
                         utmPoint.x = x;
                         utmPoint.y = y;
                         point.utm = utmPoint;
                         point.z = NODATA;
                        _meteoPoints[row][col]->aggregationPoints.push_back(point);
                    }
                }
            }
        }
        else
        {

            gis::Crit3DGeoPoint pointLatLon0;
            gis::Crit3DGeoPoint pointLatLon;
            gis::Crit3DLatLonHeader latLonHeader;
            double utmX, utmY;

            pointLatLon0.latitude = _gridStructure.header().llCorner.latitude + row * _gridStructure.header().dy;
            pointLatLon0.longitude = _gridStructure.header().llCorner.longitude + col * _gridStructure.header().dx;
            gis::getUtmFromLatLon(_gisSettings.utmZone, pointLatLon0, &utmPoint);
            v[0] = utmPoint;

            pointLatLon.latitude = pointLatLon0.latitude + _gridStructure.header().dy;
            pointLatLon.longitude = pointLatLon0.longitude;
            gis::getUtmFromLatLon(_gisSettings.utmZone, pointLatLon, &utmPoint);
            v[1] = utmPoint;

            pointLatLon.latitude = pointLatLon0.latitude + _gridStructure.header().dy;
            pointLatLon.longitude = pointLatLon0.longitude + _gridStructure.header().dx;
            gis::getUtmFromLatLon(_gisSettings.utmZone, pointLatLon, &utmPoint);
            v[2] = utmPoint;

            pointLatLon.latitude = pointLatLon0.latitude;
            pointLatLon.longitude = pointLatLon0.longitude + _gridStructure.header().dx;
            gis::getUtmFromLatLon(_gisSettings.utmZone, pointLatLon, &utmPoint);
            v[3] = utmPoint;

            utmLL.x = MINVALUE(v[0].x, v[1].x);
            utmLL.y = MINVALUE(v[0].y, v[3].y);
            utmUR.x = MAXVALUE(v[2].x, v[3].x);
            utmUR.y = MAXVALUE(v[1].y, v[2].y);

            latLonHeader.llCorner.latitude = pointLatLon0.latitude;
            latLonHeader.llCorner.longitude = pointLatLon0.longitude;
            latLonHeader.dx = _gridStructure.header().dx;
            latLonHeader.dy = _gridStructure.header().dy;
            latLonHeader.nrRows = 1;
            latLonHeader.nrCols = 1;

            gis::Crit3DRasterCell demLL, demUR;

            myDEM->getRowCol( utmLL.x, utmLL.y, demLL.row, demLL.col);
            myDEM->getRowCol(utmUR.x, utmUR.y, demUR.row, demUR.col);
            _meteoPoints[row][col]->aggregationPoints.clear();
            _meteoPoints[row][col]->aggregationPointsMaxNr = 0;

            if ( ((demUR.row >= 0) && (demUR.row < myDEM->header->nrRows)) || ((demLL.row >= 0) && (demLL.row < myDEM->header->nrRows))
                 || ((demUR.col >= 0) && (demUR.col < myDEM->header->nrCols)) || ((demLL.col >= 0) && ( demLL.col < myDEM->header->nrCols)))
            {
                for (int demRow = demUR.row; demRow <= demLL.row; demRow++)
                {
                    for (int demCol = demLL.col; demCol <= demUR.col; demCol++)
                    {
                        gis::getUtmXYFromRowCol(*(myDEM->header), demRow, demCol, &utmX, &utmY);
                        gis::getLatLonFromUtm(_gisSettings, utmX, utmY, &pointLatLon.latitude, &pointLatLon.longitude);

                        if (pointLatLon.isInsideGrid(latLonHeader))
                        {
                            _meteoPoints[row][col]->aggregationPointsMaxNr = _meteoPoints[row][col]->aggregationPointsMaxNr + 1;
                            if (!excludeNoData || myDEM->getValueFromRowCol(demRow, demCol) != myDEM->header->flag )
                            {
                                 gis::getUtmXYFromRowCol(*(myDEM->header), demRow, demCol, &utmX, &utmY);
                                 utmPoint.x = utmX;
                                 utmPoint.y = utmY;
                                 point.utm = utmPoint;
                                 point.z = NODATA;
                                _meteoPoints[row][col]->aggregationPoints.push_back(point);
                            }
                        }
                    }
                }
            }

        }
    }
}

void Crit3DMeteoGrid::initializeData(Crit3DDate dateIni, Crit3DDate dateFin, bool isHourly, bool isDaily, bool isMonthly)
{
    int nrDays = dateIni.daysTo(dateFin) + 1;
    int nrMonths = (dateFin.year-dateIni.year)*12+dateFin.month-(dateIni.month-1);

    for (unsigned row = 0; row < unsigned(gridStructure().header().nrRows); row++)
        for (unsigned col = 0; col < unsigned(gridStructure().header().nrCols); col++)
            if (_meteoPoints[row][col]->active)
            {
                if (isHourly) _meteoPoints[row][col]->initializeObsDataH(1, nrDays, dateIni);
                if (isDaily) _meteoPoints[row][col]->initializeObsDataD(nrDays, dateIni);
                if (isMonthly) _meteoPoints[row][col]->initializeObsDataM(nrMonths, dateIni.month, dateIni.year);
            }
}

void Crit3DMeteoGrid::emptyGridData(Crit3DDate dateIni, Crit3DDate dateFin)
{
    for (unsigned row = 0; row < unsigned(gridStructure().header().nrRows); row++)
        for (unsigned col = 0; col < unsigned(gridStructure().header().nrCols); col++)
        {
            _meteoPoints[row][col]->emptyObsDataH(dateIni, dateFin);
            _meteoPoints[row][col]->emptyObsDataD(dateIni, dateFin);
            _meteoPoints[row][col]->emptyObsDataM(dateIni, dateFin);
        }
}

void Crit3DMeteoGrid::computeWindVectorHourly(const Crit3DDate myDate, const int myHour)
{
    float intensity = NODATA, direction = NODATA;
    float u,v;

    for (unsigned row = 0; row < unsigned(gridStructure().header().nrRows); row++)
        for (unsigned col = 0; col < unsigned(gridStructure().header().nrCols); col++)
        {
            u = _meteoPoints[row][col]->getMeteoPointValueH(myDate, myHour, 0, windVectorX);
            v = _meteoPoints[row][col]->getMeteoPointValueH(myDate, myHour, 0, windVectorY);

            if (! isEqual(u, NODATA) && ! isEqual(v, NODATA))
            {
                if (computeWindPolar(u, v, &intensity, &direction))
                {
                    _meteoPoints[row][col]->setMeteoPointValueH(myDate, myHour, 0, windVectorIntensity, intensity);
                    _meteoPoints[row][col]->setMeteoPointValueH(myDate, myHour, 0, windVectorDirection, direction);
                }
            }
        }
}

void Crit3DMeteoGrid::spatialAggregateMeteoGrid(meteoVariable myVar, frequencyType freq, Crit3DDate date, int  hour, int minute,
                                         gis::Crit3DRasterGrid* myDEM, gis::Crit3DRasterGrid *myRaster, aggregationMethod elab)
{
    int numberOfDays = 1;

    if (!_isAggregationDefined)
    {
        findGridAggregationPoints(myDEM);
    }

    for (unsigned col = 0; col < unsigned(_gridStructure.header().nrCols); col++)
    {
        for (unsigned row = 0; row < unsigned(_gridStructure.header().nrRows); row++)
        {
            if (_meteoPoints[row][col]->active)
            {
                double validValues = 0;
                for (unsigned int i = 0; i < _meteoPoints[row][col]->aggregationPoints.size(); i++)
                {
                    double x = _meteoPoints[row][col]->aggregationPoints[i].utm.x;
                    double y = _meteoPoints[row][col]->aggregationPoints[i].utm.y;
                    float interpolatedValue = gis::getValueFromXY(*myRaster, x, y);
                    if (isEqual(interpolatedValue, myRaster->header->flag) == false)
                    {
                        _meteoPoints[row][col]->aggregationPoints[i].z = double(interpolatedValue);
                        validValues = validValues + 1;
                    }
                }

                if (!_meteoPoints[row][col]->aggregationPoints.empty())
                {

                    if ( (validValues / _meteoPoints[row][col]->aggregationPointsMaxNr) > ( GRID_MIN_COVERAGE / 100 ) )
                    {

                        double myValue = spatialAggregateMeteoGridPoint(*(_meteoPoints[row][col]), elab);

                        if (freq == hourly)
                        {
                            if (_meteoPoints[row][col]->nrObsDataDaysH == 0)
                                _meteoPoints[row][col]->initializeObsDataH(1, numberOfDays, date);

                            _meteoPoints[row][col]->setMeteoPointValueH(date, hour, minute, myVar, float(myValue));
                            _meteoPoints[row][col]->currentValue = float(myValue);
                        }
                        else if (freq == daily)
                        {
                            if (_meteoPoints[row][col]->nrObsDataDaysD == 0)
                                _meteoPoints[row][col]->initializeObsDataD(numberOfDays, date);

                            _meteoPoints[row][col]->setMeteoPointValueD(date, myVar, float(myValue));
                            _meteoPoints[row][col]->currentValue = float(myValue);

                        }
                    }
                }

            }
        }
    }
}


double Crit3DMeteoGrid::spatialAggregateMeteoGridPoint(Crit3DMeteoPoint myPoint, aggregationMethod elab)
{

    std::vector <float> validValues;


    for (unsigned int i = 0; i < myPoint.aggregationPoints.size(); i++)
    {
        if (myPoint.aggregationPoints[i].z != NODATA)
        {
            validValues.push_back(float(myPoint.aggregationPoints[i].z));
        }
    }

    if (validValues.empty())
    {
        return NODATA;
    }

    if ( (static_cast<double>(validValues.size()) / myPoint.aggregationPointsMaxNr) < ( GRID_MIN_COVERAGE / 100.0) )
    {
        return NODATA;
    }

    if (elab == aggregationMethod::aggrAverage)
    {
        return statistics::mean(validValues.data(), int(validValues.size()));
    }
    else if (elab == aggregationMethod::aggrMedian)
    {
        int size = int(validValues.size());
        return sorting::percentile(validValues, size, 50.0, true);
    }
    else if (elab == aggregationMethod::aggrStdDeviation)
    {
        return statistics::standardDeviation(validValues.data(), int(validValues.size()));
    }
    else
    {
        return NODATA;
    }


}

bool Crit3DMeteoGrid::getIsElabValue() const
{
    return _isElabValue;
}

void Crit3DMeteoGrid::setIsElabValue(bool isElabValue)
{
    _isElabValue = isElabValue;
}


void Crit3DMeteoGrid::setGridStructure(const Crit3DMeteoGridStructure &gridStructure)
{
    _gridStructure = gridStructure;
}


bool Crit3DMeteoGrid::isAggregationDefined() const
{
    return _isAggregationDefined;
}

void Crit3DMeteoGrid::setIsAggregationDefined(bool isAggregationDefined)
{
    _isAggregationDefined = isAggregationDefined;
}

Crit3DDate Crit3DMeteoGrid::firstDate() const
{
    return _firstDate;
}

void Crit3DMeteoGrid::setFirstDate(const Crit3DDate &firstDate)
{
    _firstDate = firstDate;
}

Crit3DDate Crit3DMeteoGrid::lastDate() const
{
    return _lastDate;
}

void Crit3DMeteoGrid::setLastDate(const Crit3DDate &lastDate)
{
    _lastDate = lastDate;
}

void Crit3DMeteoGrid::saveRowColfromZone(gis::Crit3DRasterGrid* zoneGrid, std::vector<std::vector<int> > &meteoGridRow, std::vector<std::vector<int> > &meteoGridCol)
{
    float value;
    double x, y;
    int myRow, myCol;
    for (int row = 0; row < zoneGrid->header->nrRows; row++)
    {

        for (int col = 0; col < zoneGrid->header->nrCols; col++)
        {
            value = zoneGrid->value[row][col];
            if (value != zoneGrid->header->flag)
            {
                zoneGrid->getXY(row, col, x, y);
                if (!_gridStructure.isUTM())
                {
                    double utmX = x;
                    double utmY = y;
                    gis::getLatLonFromUtm(_gisSettings, utmX, utmY, &y, &x);
                    gis::getGridRowColFromXY(_gridStructure.header(), x, y, &myRow, &myCol);
                }
                else
                {
                    dataMeteoGrid.getRowCol(x, y, myRow, myCol);
                }

                if (myRow >= 0 && myCol >= 0 && myRow < _gridStructure.header().nrRows && myCol < _gridStructure.header().nrCols)
                {
                    if (_meteoPoints[myRow][myCol]->active == true)
                    {
                        meteoGridRow[row][col] = myRow;
                        meteoGridCol[row][col] = myCol;
                    }
                }

            }
        }
    }
}

void Crit3DMeteoGrid::computeHourlyDerivedVariables(Crit3DTime dateTime)
{

    for (unsigned row = 0; row < unsigned(gridStructure().header().nrRows); row++)
    {
        for (unsigned col = 0; col < unsigned(gridStructure().header().nrCols); col++)
        {
            if (_meteoPoints[row][col]->active)
            {
                _meteoPoints[row][col]->computeDerivedVariables(dateTime);
            }
        }
    }
}

