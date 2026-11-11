/*!
    \file gis.cpp

    \abstract Gis structures and functions

    \copyright
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed by ARPAE Emilia-Romagna.

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
    Fausto Tomei ftomei@arpae.it
    Gabriele Antolini gantolini@arpae.it
*/

#include <math.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <limits>

#include "commonConstants.h"
#include "basicMath.h"
#include "statistics.h"
#include "gis.h"

namespace gis
{
    Crit3DEllipsoid::Crit3DEllipsoid()
    {
        /*!  WGS84 */
        this->equatorialRadius = 6378137.0 ;
        this->eccentricitySquared = 6.69438000426083E-03;
    }

    void Crit3DGisSettings::initialize()
    {
        startLocation.latitude = 44.501;
        startLocation.longitude = 11.329;
        utmZone = 32;
        timeZone = 1;
        isUTC = true;
    }

    Crit3DGisSettings::Crit3DGisSettings()
    {
        initialize();
    }

    Crit3DPixel::Crit3DPixel()
    {
        this->x = 0;
        this->y = 0;
    }

    Crit3DPixel::Crit3DPixel(int x0, int y0)
    {
        this->x = x0;
        this->y = y0;
    }

    Crit3DRasterCell::Crit3DRasterCell()
    {
        this->row = NODATA;
        this->col = NODATA;
    }

    Crit3DUtmPoint::Crit3DUtmPoint()
    {
        this->initialize();
    }

    void Crit3DUtmPoint::initialize()
    {
        this->x = NODATA;
        this->y = NODATA;
    }


    Crit3DOutputPoint::Crit3DOutputPoint()
    {
        this->id = "";
        this->latitude = NODATA;
        this->longitude = NODATA;
        this->active = false;
        this->selected = false;
        this->currentValue = NODATA;
    }

    void Crit3DOutputPoint::initialize(const std::string& _id, bool isActive, double _latitude, double _longitude,
                                  double _z, int zoneNumber)
    {
        this->id = _id;
        this->latitude = _latitude;
        this->longitude = _longitude;
        this->z = _z;
        this->active = isActive;
        gis::latLonToUtmForceZone(zoneNumber, latitude, longitude, &(utm.x), &(utm.y));
    }


    Crit3DUtmPoint::Crit3DUtmPoint(double _x, double _y)
    {
        this->x = _x;
        this->y = _y;
    }

    Crit3DGeoPoint::Crit3DGeoPoint()
    {
        this->latitude = NODATA;
        this->longitude = NODATA;
    }

    Crit3DGeoPoint::Crit3DGeoPoint(double lat, double lon)
    {
        this->latitude = lat;
        this->longitude = lon;
    }

    bool Crit3DUtmPoint::isInsideGrid(const Crit3DRasterHeader& myGridHeader) const
    {
        return (x >= myGridHeader.llCorner.x
                && x < myGridHeader.llCorner.x + (myGridHeader.nrCols * myGridHeader.cellSize)
                && y >= myGridHeader.llCorner.y
                && y < myGridHeader.llCorner.y + (myGridHeader.nrRows * myGridHeader.cellSize));
    }

    bool Crit3DGeoPoint::isInsideGrid(const Crit3DLatLonHeader& latLonHeader) const
    {
        return (longitude >= latLonHeader.llCorner.longitude
                && longitude <= latLonHeader.llCorner.longitude + (latLonHeader.nrCols * latLonHeader.dx)
                && latitude >= latLonHeader.llCorner.latitude
                && latitude <= latLonHeader.llCorner.latitude + (latLonHeader.nrRows * latLonHeader.dy));
    }

    Crit3DPoint::Crit3DPoint()
    {
        this->utm.x = NODATA;
        this->utm.y = NODATA;
        this->z = NODATA;
    }

    Crit3DPoint::Crit3DPoint(double utmX, double utmY, double _z)
    {
        this->utm.x = utmX;
        this->utm.y = utmY;
        this->z = _z;
    }

    Crit3DRasterHeader::Crit3DRasterHeader()
    {
        nrRows = 0;
        nrCols = 0;
        nrBytes = 4;
        cellSize = NODATA;
        invCellSize = NODATA;
        flag = NODATA;
    }

    Crit3DLatLonHeader::Crit3DLatLonHeader()
    {
        nrRows = 0;
        nrCols = 0;
        nrBytes = 4;
        dx = NODATA;
        dy = NODATA;
        flag = NODATA;
    }

    bool operator == (const Crit3DRasterHeader& header1, const Crit3DRasterHeader& header2)
    {
        return (isEqual(header1.cellSize, header2.cellSize) && isEqual(header1.flag, header2.flag)
                && (fabs(header1.llCorner.x - header2.llCorner.x) < 0.01)
                && (fabs(header1.llCorner.y - header2.llCorner.y) < 0.01)
                && (header1.nrCols == header2.nrCols)
                && (header1.nrRows == header2.nrRows));
    }

    bool Crit3DRasterHeader::isEqualTo(const Crit3DRasterHeader& header)
    {
        return (isEqual(cellSize, header.cellSize) && isEqual(flag, header.flag)
                && (fabs(llCorner.x - header.llCorner.x) < 0.01)
                && (fabs(llCorner.y - header.llCorner.y) < 0.01)
                && (nrCols == header.nrCols)
                && (nrRows == header.nrRows));
    }


    /******************     INDEX GRID     ********************/

    long Crit3DIndexGrid::getValueFromRowCol(int row, int col) const
    {
        if (isOutOfGrid(row, col))
            return long(header.flag);
        else
            return value[row][col];
    }

    bool Crit3DIndexGrid::isOutOfGrid(int row, int col) const
    {
        return (row < 0 || row > (header.nrRows - 1) || col < 0 || col > header.nrCols - 1);
    }

    void Crit3DIndexGrid::initializeGrid()
    {
        value.resize(header.nrRows);
        for (int row = 0; row < header.nrRows; row++)
        {
            value[row].resize(header.nrCols);
            for (int col = 0; col < header.nrCols; col++)
            {
                value[row][col] = long(header.flag);
            }
        } 
    }

    void Crit3DIndexGrid::initializeGrid(const Crit3DRasterHeader& initHeader)
    {
        value.clear();
        header = initHeader;
        initializeGrid();
    }


    /******************     RASTER GRID     ********************/

    Crit3DRasterGrid::Crit3DRasterGrid()
    {
        value = nullptr;

        initialize();
        mapTime.setNullTime();
    }

    void Crit3DRasterGrid::initialize()
    {
        minimum = NODATA;
        maximum = NODATA;

        header.nrRows = 0;
        header.nrCols = 0;
        header.nrBytes = 4;
        header.cellSize = NODATA;
        header.invCellSize = NODATA;
        header.llCorner.initialize();

        isLoaded = false;
    }


    Crit3DRasterGrid::~Crit3DRasterGrid()
    {
        clear();
    }


    void Crit3DRasterGrid::clear()
    {
        if (value != nullptr && header.nrRows > 0)
        {
            for (int row = 0; row < header.nrRows; row++)
                if (value[row] != nullptr)
                    delete [] value[row];

            delete [] value;
            value = nullptr;
        }

        initialize();
    }


    // clean the data (set all NO DATA)
    void Crit3DRasterGrid::emptyGrid()
    {
        const float flag = header.flag;

        for (int row = 0; row < header.nrRows; ++row)
            for (int col = 0; col < header.nrCols; ++col)
                value[row][col] = flag;

        minimum = NODATA;
        maximum = NODATA;
    }


    bool Crit3DRasterGrid::initializeGrid(float initValue)
    {
        const int nrRows = header.nrRows;
        const int nrCols = header.nrCols;

        if (nrRows <= 0 || nrCols <= 0)
            return false;

        try
        {
            value = new float*[nrRows]();       // () zero-initializes

            for (int row = 0; row < nrRows; ++row)
            {
                value[row] = new float[nrCols];

                // set init value
                for (int col = 0; col < nrCols; ++col)
                    value[row][col] = initValue;
            }
        }
        catch (const std::bad_alloc&)
        {
            clear();                            // Memory error: file too big
            return false;
        }

        minimum = initValue;
        maximum = initValue;
        isLoaded = true;

        return true;
    }


    void Crit3DRasterGrid::setConstantValue(float myValue)
    {
        const int nrRows = header.nrRows;
        const int nrCols = header.nrCols;

        for (int row = 0; row < nrRows; ++row)
        {
            for (int col = 0; col < nrCols; ++col)
                value[row][col] = myValue;
        }

        minimum = myValue;
        maximum = myValue;
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterHeader& initHeader)
    {
        clear();

        header = initHeader;
        return initializeGrid(header.flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DLatLonHeader& latLonHeader)
    {
        clear();

        header.nrRows = latLonHeader.nrRows;
        header.nrCols = latLonHeader.nrCols;
        header.flag = latLonHeader.flag;
        header.llCorner.y = latLonHeader.llCorner.latitude;
        header.llCorner.x = latLonHeader.llCorner.longitude;

        // avg value (usually not used, this is a raster value container for a lat lon grid)
        header.cellSize = (latLonHeader.dx + latLonHeader.dy) * 0.5;
        header.invCellSize = 1.0 / header.cellSize;

        return initializeGrid(header.flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid)
    {
        clear();
        header = initGrid.header;

        return initializeGrid(header.flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid, float initValue)
    {
        clear();
        header = initGrid.header;
        return initializeGrid(initValue);
    }


    bool Crit3DRasterGrid::copyGrid(const Crit3DRasterGrid& initGrid)
    {
        if (! initGrid.isLoaded)
            return false;

        if (&initGrid == this)
            return true;

        clear();

        header = initGrid.header;
        colorScale = initGrid.colorScale;
        mapTime = initGrid.mapTime;

        initializeGrid(header.flag);

        for (int row = 0; row < header.nrRows; row++)
            for (int col = 0; col < header.nrCols; col++)
                    value[row][col] = initGrid.value[row][col];

        gis::updateMinMaxRasterGrid(this);

        isLoaded = true;
        return true;
    }


    bool Crit3DRasterGrid::setConstantValueWithBase(float initValue, const Crit3DRasterGrid& initGrid)
    {
        if (! isLoaded)
            return false;
        if (! compareGrids(*(this), initGrid))
            return false;

        minimum = initValue;
        maximum = initValue;

        for (int row = 0; row < header.nrRows; row++)
            for (int col = 0; col < header.nrCols; col++)
                if (! isEqual(initGrid.value[row][col], initGrid.header.flag))
                    value[row][col] = initValue;

        return gis::updateMinMaxRasterGrid(this);
    }


    Crit3DPoint Crit3DRasterGrid::getCenter()
    {
        Crit3DPoint center;
        int row, col;
        center.utm.x = (header.llCorner.x + (header.nrCols * header.cellSize)/2.);
        center.utm.y = (header.llCorner.y + (header.nrRows * header.cellSize)/2.);
        getRowCol(center.utm.x, center.utm.y, row, col);
        center.z = double(value[row][col]);

        return center;
    }


    Crit3DGeoPoint Crit3DRasterGrid::getCenterLatLon(const Crit3DGisSettings &gisSettings)
    {
        Crit3DUtmPoint utmCenter = this->getCenter().utm;
        Crit3DGeoPoint geoCenter;
        getLatLonFromUtm(gisSettings, utmCenter, geoCenter);

        return geoCenter;
    }


    /*!
     * \brief return X,Y of cell center
     * \param row
     * \param col
     * \return Crit3DUtmPoint pointer
     */
    Crit3DTime Crit3DRasterGrid::getMapTime() const
    {
        return mapTime;
    }

    void Crit3DRasterGrid::setMapTime(const Crit3DTime &value)
    {
        mapTime = value;
    }

    Crit3DUtmPoint Crit3DRasterGrid::utmPoint(int row, int col) const
    {
        const double x = header.llCorner.x + header.cellSize * (double(col) + 0.5);
        const double y = header.llCorner.y + header.cellSize * (double(header.nrRows - row) - 0.5);

        return Crit3DUtmPoint(x, y);
    }


    void Crit3DRasterGrid::getXY(int row, int col, double& x, double& y) const
    {
        x = header.llCorner.x + header.cellSize * (double(col) + 0.5);
        y = header.llCorner.y + header.cellSize * (double(header.nrRows - row) - 0.5);
    }

    // faster
    void Crit3DRasterGrid::getRowCol(double x, double y, int& row, int& col) const
    {
        const int r = static_cast<int>(floor((y - header.llCorner.y) * header.invCellSize));
        row = (header.nrRows - 1) - r;
        col = static_cast<int>(floor((x - header.llCorner.x) * header.invCellSize));
    }

    // faster
    bool Crit3DRasterGrid::isOutOfGrid(int row, int col) const
    {
        // it works because if row is < 0 it becomes a huge unsigned → out of range
        return (unsigned(row) >= unsigned(header.nrRows) ||
                unsigned(col) >= unsigned(header.nrCols));
    }

    bool Crit3DRasterGrid::isFlag(int row, int col) const
    {
        if (isOutOfGrid(row, col))
            return true;
        else
            return (isEqual(value[row][col], header.flag));
    }

    float Crit3DRasterGrid::getValueFromRowCol(int row, int col) const
    {
        if (unsigned(row) >= unsigned(header.nrRows) ||
            unsigned(col) >= unsigned(header.nrCols))
            return header.flag;

        return value[row][col];
    }

    float Crit3DRasterGrid::getValueFromXY(double x, double y) const
    {       
        const int r = static_cast<int>(floor((y - header.llCorner.y) * header.invCellSize));
        const int row = (header.nrRows - 1) - r;
        const int col = static_cast<int>(floor((x - header.llCorner.x) * header.invCellSize));

        if (unsigned(row) >= unsigned(header.nrRows) ||
            unsigned(col) >= unsigned(header.nrCols))
            return header.flag;

        return value[row][col];
    }


    void convertFlagToNodata(Crit3DRasterGrid& myGrid)
    {
        if (myGrid.header.flag == NODATA)
            return;

        for (int row = 0; row < myGrid.header.nrRows; row++)
        {
            for (int col = 0; col < myGrid.header.nrCols; col++)
            {
                if (isEqual(myGrid.value[row][col], myGrid.header.flag))
                {
                    myGrid.value[row][col] = NODATA;
                }
            }
        }

        myGrid.header.flag = NODATA;
    }


    bool updateMinMaxRasterGrid(Crit3DRasterGrid* myGrid)
    {
        int nRows = myGrid->header.nrRows;
        int nCols = myGrid->header.nrCols;
        float flag = myGrid->header.flag;

        bool isFirstValue = true;
        float minimum = NODATA;
        float maximum = NODATA;

        for (int row = 0; row < nRows; row++)
        {
            float* rowPtr = myGrid->value[row];
            for (int col = 0; col < nCols; col++)
            {
                float value = rowPtr[col];
                if (!isEqual(value, flag) && !isEqual(value, NODATA))
                {
                    if (isFirstValue)
                    {
                        minimum = maximum = value;
                        isFirstValue = false;
                    }
                    else
                    {
                        if (value < minimum) minimum = value;
                        else if (value > maximum) maximum = value;
                    }
                }
            }
        }

        /*!  no values */
        if (isFirstValue)
            return false;

        myGrid->maximum = maximum;
        myGrid->minimum = minimum;

        if (! myGrid->colorScale.isFixedRange())
        {
            myGrid->colorScale.setRange(minimum, maximum);
        }

        return true;
    }


    bool updateColorScale(Crit3DRasterGrid* myGrid, int row0, int col0, int row1, int col1)
    {
        float myValue;
        bool isFirstValue = true;
        float minimum = NODATA;
        float maximum = NODATA;

        if (row0 > row1)
            std::swap(row0, row1);

        if (col0 > col1)
            std::swap(col0, col1);

        row0 = std::max(row0, int(0));
        col0 = std::max(col0, int(0));
        row1 = std::min(row1, myGrid->header.nrRows-1);
        col1 = std::min(col1, myGrid->header.nrCols-1);

        for (int row = row0; row <= row1; row++)
            for (int col = col0; col <= col1; col++)
            {
                myValue = myGrid->value[row][col];
                if (! isEqual(myValue, myGrid->header.flag) && ! isEqual(myValue, NODATA))
                {
                    if (isFirstValue)
                    {
                        minimum = myValue;
                        maximum = myValue;
                        isFirstValue = false;
                    }
                    else
                    {
                        if (myValue < minimum) minimum = myValue;
                        else if (myValue > maximum) maximum = myValue;
                    }
                }
            }

        //  no values
        if (isFirstValue)
        {
            myGrid->colorScale.setRange(NODATA, NODATA);
            return false;
        }

        myGrid->colorScale.setRange(minimum, maximum);

        return true;
    }


    double computeDistancePoint(Crit3DUtmPoint* p0, Crit3DUtmPoint *p1)
    {
        const double dx = p1->x - p0->x;
        const double dy = p1->y - p0->y;

        return sqrt(dx * dx + dy * dy);
    }


    float computeDistance(int x1, int y1, int x2, int y2)
    {
        const float dx = float(x2 - x1);
        const float dy = float(y2 - y1);

        return sqrtf(dx * dx + dy * dy);
    }

    float computeDistance(float x1, float y1, float x2, float y2)
    {
        const float dx = x2 - x1;
        const float dy = y2 - y1;

        return sqrtf(dx * dx + dy * dy);
    }

    double computeDistance(double x1, double y1, double x2, double y2)
    {
        const double dx = x2 - x1;
        const double dy = y2 - y1;

        return sqrt(dx * dx + dy * dy);
    }


    std::vector<float> computeEuclideanDistanceStation2Area(const std::vector<std::vector<int> > &cells,
                                                            const std::vector<std::vector<int> > &stations)
    {
        // è possibile sapere in quale cella (row,col) si trova la stazione?
        const int nrStations = (int)stations.size();
        std::vector<float> distanceList(nrStations, NODATA);

        if (cells.empty())
            return distanceList;

        for (int i=0; i < nrStations; i++)
        {
            float minDistance = std::numeric_limits<float>::max();

            for (int j=0; j < (int)cells.size(); ++j)
            {
                const float dx = float(stations[i][0] - cells[j][0]);
                const float dy = float(stations[i][1] - cells[j][1]);
                minDistance = std::min(minDistance, dx*dx + dy*dy);
            }

            distanceList[i] = float(sqrt(minDistance));
        }

        return distanceList;
    }


    std::vector<int> computeManhattanDistanceStation2Area(const std::vector<std::vector<int> > &cells,
                                                           const std::vector<std::vector<int> > &stations)
    {
        // è possibile sapere in quale cella (row,col) si trova la stazione?
        const int nrStations = (int)stations.size();
        std::vector<int> distanceList(nrStations, NODATA);

        if (cells.empty())
            return distanceList;

        for (int i=0; i < nrStations; i++)
        {
            distanceList[i] = abs(stations[i][0] - cells[0][0]) + abs(stations[i][1] - cells[0][1]);
            for (int j=1; j < (int)cells.size(); j++)
            {
                distanceList[i] = std::min(distanceList[i], abs(stations[i][0] - cells[j][0]) + abs(stations[i][1] - cells[j][1]));
            }
        }

        return distanceList;
    }


    void getRowColFromXY(const Crit3DRasterHeader& myHeader, double x, double y, int *row, int *col)
    {
        *row = (myHeader.nrRows - 1) - int(floor((y - myHeader.llCorner.y) * myHeader.invCellSize));
        *col = int(floor((x - myHeader.llCorner.x) * myHeader.invCellSize));
    }

    void getRowColFromXY(const Crit3DRasterHeader& myHeader, double x, double y, int& row, int& col)
    {
        row = (myHeader.nrRows - 1) - int(floor((y - myHeader.llCorner.y) * myHeader.invCellSize));
        col = int(floor((x - myHeader.llCorner.x) * myHeader.invCellSize));
    }

    void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, int *row, int *col)
    {
        *row = (myHeader.nrRows - 1) - int(floor((p.y - myHeader.llCorner.y) * myHeader.invCellSize));
        *col = int(floor((p.x - myHeader.llCorner.x) * myHeader.invCellSize));
    }

    void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, Crit3DRasterCell* v)
    {
        v->row = (myHeader.nrRows - 1) - int(floor((p.y - myHeader.llCorner.y) * myHeader.invCellSize));
        v->col = int(floor((p.x - myHeader.llCorner.x) * myHeader.invCellSize));
    }

    void getRowColFromLonLat_zeroSouth(const Crit3DLatLonHeader& myHeader, double lon, double lat, int *row, int *col)
    {
        *row = int(floor((lat - myHeader.llCorner.latitude) / myHeader.dy));
        *col = int(floor((lon - myHeader.llCorner.longitude) / myHeader.dx));
    }

    void getRowColFromLatLon(const Crit3DLatLonHeader& latLonHeader, const Crit3DGeoPoint& p, int* row, int* col)
    {
        *row = (latLonHeader.nrRows - 1) - int(floor((p.latitude - latLonHeader.llCorner.latitude) / latLonHeader.dy));
        *col = int(floor((p.longitude - latLonHeader.llCorner.longitude) / latLonHeader.dx));
    }

    bool isOutOfGridRowCol(int row, int col, const Crit3DRasterGrid& myGrid)
    {
        // it works because if row is < 0 it becomes a huge unsigned → out of range
        return (unsigned(row) >= unsigned(myGrid.header.nrRows) ||
                unsigned(col) >= unsigned(myGrid.header.nrCols) );
    }

    bool isOutOfGridRowCol(int row, int col, const Crit3DRasterHeader& header)
    {
        // it works because if row is < 0 it becomes a huge unsigned → out of range
        return ( unsigned(row) >= unsigned(header.nrRows) ||
                 unsigned(col) >= unsigned(header.nrCols) );
    }

    bool isOutOfGridRowCol(int row, int col, const Crit3DLatLonHeader& header)
    {
        // it works because if row is < 0 it becomes a huge unsigned → out of range
        return ( unsigned(row) >= unsigned(header.nrRows) ||
                 unsigned(col) >= unsigned(header.nrCols) );
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterGrid& myGrid,
                                            int row, int col, float* myX, float* myY)
    {
        *myX = float(myGrid.header.llCorner.x + myGrid.header.cellSize * (col + 0.5));
        *myY = float(myGrid.header.llCorner.y + myGrid.header.cellSize * (myGrid.header.nrRows - row - 0.5));
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterHeader& myHeader,
                                           int row, int col, float* myX, float* myY)
    {
        *myX = float(myHeader.llCorner.x + myHeader.cellSize * (col + 0.5));
        *myY = float(myHeader.llCorner.y + myHeader.cellSize * (myHeader.nrRows - row - 0.5));
    }

    void getUtmXYFromRowCol(const Crit3DRasterHeader& myHeader, int row, int col, double* myX, double* myY)
    {
            *myX = myHeader.llCorner.x + myHeader.cellSize * (col + 0.5);
            *myY = myHeader.llCorner.y + myHeader.cellSize * (myHeader.nrRows - row - 0.5);
    }

    void getUtmXYFromRowCol(Crit3DRasterHeader *myHeader, int row, int col, double* myX, double* myY)
    {
        *myX = myHeader->llCorner.x + myHeader->cellSize * (col + 0.5);
        *myY = myHeader->llCorner.y + myHeader->cellSize * (myHeader->nrRows - row - 0.5);
    }

    void getLatLonFromRowCol(const Crit3DLatLonHeader& latLonHeader, int row, int col, double* lat, double* lon)
    {
            *lon = latLonHeader.llCorner.longitude + latLonHeader.dx * (col + 0.5);
            *lat = latLonHeader.llCorner.latitude + latLonHeader.dy * (latLonHeader.nrRows - row - 0.5);
    }

    void getLatLonFromRowCol(const Crit3DLatLonHeader& latLonHeader, const Crit3DRasterCell& v, Crit3DGeoPoint* p)
    {
            p->longitude = latLonHeader.llCorner.longitude + latLonHeader.dx * (v.col + 0.5);
            p->latitude = latLonHeader.llCorner.latitude + latLonHeader.dy * (latLonHeader.nrRows - v.row - 0.5);
    }

    float getValueFromUTMPoint(const Crit3DRasterGrid& myGrid, const Crit3DUtmPoint& utmPoint)
    {
        return myGrid.getValueFromXY(utmPoint.x, utmPoint.y);
    }

    float getValueFromXY(const Crit3DRasterGrid& myGrid, double x, double y)
    {
        return myGrid.getValueFromXY(x, y);
    }

    bool isOutOfGridXY(double x, double y, const Crit3DRasterHeader& header)
    {
        return ((x < header.llCorner.x) || (y < header.llCorner.y) ||
                (x >= (header.llCorner.x + header.nrCols * header.cellSize)) ||
                (y >= (header.llCorner.y + header.nrRows * header.cellSize)));
    }

    void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, double utmX, double utmY, double *myLat, double *myLon)
    {
        gis::utmToLatLon(gisSettings.utmZone, gisSettings.startLocation.latitude, utmX, utmY, myLat, myLon);
    }

    void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, const Crit3DUtmPoint& utmPoint, Crit3DGeoPoint& geoPoint)
    {
        gis::utmToLatLon(gisSettings.utmZone, gisSettings.startLocation.latitude, utmPoint.x, utmPoint.y, &(geoPoint.latitude), &(geoPoint.longitude));
    }


    /*!
     * \brief Converts lat/int to UTM coords.  Equations from USGS Bulletin 1532.
     * Source:
     *      Defense Mapping Agency. 1987b. DMA Technical Report:
     *      Supplement to Department of Defense World Geodetic System.
     *      1984 Technical Report. Part I and II. Washington, DC: Defense Mapping Agency
     * \param lat in decimal degrees
     * \param lon in decimal degrees
     * \param utmEasting: East Longitudes are positive, West longitudes are negative.
     * \param utmNorthing: North latitudes are positive, South latitudes are negative.
     * \param zoneNumber
     */
    void latLonToUtm(double lat, double lon, double *utmEasting, double *utmNorthing, int *zoneNumber)
    {
        Crit3DEllipsoid referenceEllipsoid;
        const double ae = referenceEllipsoid.equatorialRadius;
        const double eccSquared = referenceEllipsoid.eccentricitySquared;

        //!< Make sure the longitude is between -180.00 .. 179.9: */
        const double lonTemp = (lon + 180.) - floor((lon + 180.) / 360.) * 360. - 180.;

        const double latRad = lat * DEG_TO_RAD;
        const double lonRad = lonTemp * DEG_TO_RAD;
        *zoneNumber = int(floor((lonTemp + 180.) / 6.)) + 1;

        //!<  Special zones for Norway: */
        if ((lat >= 56.0) && (lat < 64.0) && (lonTemp >= 3.0) && (lonTemp < 12.0))
            (*zoneNumber) = 32 ;

        //!<  Special zones for Svalbard: */
        if ((lat >= 72.0)&&(lat < 84.0))
        {
            if ((lonTemp >= 0) && (lonTemp < 9.0))
                (*zoneNumber) = 31;
            else if ((lonTemp >= 9.0)&& (lonTemp < 21.0))
                (*zoneNumber) = 33;
            else if ((lonTemp >= 21.0)&& ( lonTemp < 33.0))
                (*zoneNumber) = 35;
            else if ((lonTemp >= 33.0)&& (lonTemp < 42.0))
                (*zoneNumber) = 37;
        }

        //!<  puts origin in middle of zone */
        const double lonOrigin = ((*zoneNumber) - 1.) * 6. - 180. + 3.;
        const double lonOriginRad = lonOrigin * DEG_TO_RAD;

        const double eccPrimeSquared = eccSquared / (1. - eccSquared);

        const double n = ae / sqrt(1.0 - eccSquared * sin(latRad) * sin(latRad));
        const double t = tan(latRad) * tan(latRad);
        const double c = eccPrimeSquared * cos(latRad) * cos(latRad);
        const double a = cos(latRad) * (lonRad - lonOriginRad);

       const double  m = ae * ((1. - eccSquared / 4. - 3. * eccSquared * eccSquared / 64.
          - 5. * eccSquared * eccSquared * eccSquared / 256.) * latRad
          - (3. * eccSquared / 8 + 3 * eccSquared * eccSquared / 32.
          + 45. * eccSquared * eccSquared * eccSquared / 1024.) * sin(2. * latRad)
          + (15. * eccSquared * eccSquared / 256.
          + 45. * eccSquared * eccSquared * eccSquared / 1024.) * sin(4. * latRad)
          - (35. * eccSquared * eccSquared * eccSquared / 3072.) * sin(6. * latRad));

        constexpr double ellipsoidK0 = 0.9996;

        *utmEasting = (ellipsoidK0 * n * (a + (1 - t + c) * a * a * a / 6.
                   + (5. - 18. * t + t * t + 72. * c
                   - 58. * eccPrimeSquared) * a * a * a * a * a / 120.)
                   + 500000.);

        *utmNorthing = (ellipsoidK0 * (m + n * tan(latRad) * (a * a / 2.
                    + (5. - t + 9. * c + 4. * c * c) * a * a * a * a / 24.
                    + (61. - 58. * t + t * t + 600. * c
                    - 330. * eccPrimeSquared) * a * a * a * a * a * a / 720.)));

        //!<  offset for southern hemisphere: */
        if (lat < 0)
            (*utmNorthing) += 10000000.;
    }


    void getUtmFromLatLon(int zoneNumber, const Crit3DGeoPoint& geoPoint, Crit3DUtmPoint* utmPoint)
    {
        latLonToUtmForceZone(zoneNumber, geoPoint.latitude, geoPoint.longitude, &(utmPoint->x), &(utmPoint->y));
    }

    void getUtmFromLatLon(const Crit3DGisSettings& gisSettings, double latitude, double longitude, double *utmX, double *utmY)
    {
        latLonToUtmForceZone(gisSettings.utmZone, latitude, longitude, utmX, utmY);
    }


    /*!
          \brief equivalent to latLonToUtm forcing UTM zone.
    */
    void latLonToUtmForceZone(int zoneNumber, double lat, double lon, double *utmEasting, double *utmNorthing)
    {
        double eccSquared, lonOrigin, eccPrimeSquared, ae, a, n , t, c,m,lonTemp, latRad,lonRad,lonOriginRad;
        static double ellipsoidK0 = 0.9996;

        Crit3DEllipsoid referenceEllipsoid;
        ae = referenceEllipsoid.equatorialRadius;
        eccSquared = referenceEllipsoid.eccentricitySquared;

        //!<  make sure the longitude is between -180.00 .. 179.9: */
        lonTemp = (lon + 180.) - floor((lon + 180.) / 360.) * 360. - 180.;

        latRad = lat * DEG_TO_RAD;
        lonRad = lonTemp * DEG_TO_RAD;

        //!<  puts origin in middle of zone */
        lonOrigin = (zoneNumber - 1.) * 6. - 180. + 3.;
        lonOriginRad = lonOrigin * DEG_TO_RAD;

        eccPrimeSquared = eccSquared / (1. - eccSquared);

        n = ae / sqrt(1. - eccSquared * sin(latRad) * sin(latRad));
        t = tan(latRad) * tan(latRad);
        c = eccPrimeSquared * cos(latRad) * cos(latRad);
        a = cos(latRad) * (lonRad - lonOriginRad);

        m = ae * ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64
          - 5 * eccSquared * eccSquared * eccSquared / 256) * latRad
          - (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32
          + 45 * eccSquared * eccSquared * eccSquared / 1024) * sin(2 * latRad)
          + (15 * eccSquared * eccSquared / 256
          + 45 * eccSquared * eccSquared * eccSquared / 1024) * sin(4 * latRad)
          - (35 * eccSquared * eccSquared * eccSquared / 3072) * sin(6 * latRad));

        *utmEasting = (ellipsoidK0 * n * (a + (1 - t + c) * a * a * a / 6
                   + (5 - 18 * t + t * t + 72 * c
                   - 58 * eccPrimeSquared) * a * a * a * a * a / 120)
                   + 500000.0);

        *utmNorthing = (ellipsoidK0 * (m + n * tan(latRad) * (a * a / 2
                    + (5 - t + 9 * c + 4 * c * c) * a * a * a * a / 24
                    + (61 - 58 * t + t * t + 600 * c
                    - 330 * eccPrimeSquared) * a * a * a * a * a * a / 720)));

        //!<  offset for southern hemisphere: */
        if (lat < 0) (*utmNorthing) += 10000000.;

    }

    /*!
     * \brief Converts UTM coords to Lat/Lng.  Equations from USGS Bulletin 1532.
     * \param zoneNumber
     * \param referenceLat: [degree] offset used for southern hemisphere
     * \param utmEasting:   [m] East Longitudes are positive, West longitudes are negative.
     * \param utmNorthing:  [m] North latitudes are positive, South latitudes are negative.
     * \param lat in decimal degrees.
     * \param lon in decimal degrees.
     */
    void utmToLatLon(int zoneNumber, double referenceLat, double utmEasting, double utmNorthing, double *lat, double *lon)
    {
        if (!lat || !lon)
            return;

        double ae, e1, eccSquared, eccPrimeSquared, n1, t1, c1, r1, d, m, x, y;
        double longOrigin , mu , phi1Rad;
        static double ellipsoidK0 = 0.9996;

        Crit3DEllipsoid referenceEllipsoid;
        ae = referenceEllipsoid.equatorialRadius;
        eccSquared = referenceEllipsoid.eccentricitySquared;

        e1 = (1. - sqrt(1. - eccSquared)) / (1. + sqrt(1. - eccSquared));

        /*! offset for longitude */
        x = utmEasting - 500000.0;
        y = utmNorthing;

        /*! offset used for southern hemisphere */
        if (referenceLat < 0)
            y -= 10000000.;

        eccPrimeSquared = (eccSquared) / (1. - eccSquared);

        m = y / ellipsoidK0;
        mu = m / (ae * (1. - eccSquared / 4. - 3. * eccSquared * eccSquared / 64.
           - 5. * eccSquared * eccSquared * eccSquared / 256.));

        phi1Rad = mu + (3.0 * e1 / 2.0 - 27.0 * e1 * e1 * e1 / 32.0) * sin(2.0 * mu)
                + (21.0 * e1 * e1 / 16.0 - 55.0 * e1 * e1 * e1 * e1 / 32.0) * sin(4.0 * mu)
                + (151.0 * e1 * e1 * e1 / 96.0) * sin(6.0 * mu);

        n1 = ae / sqrt(1.0 - eccSquared * sin(phi1Rad) * sin(phi1Rad));
        t1 = tan(phi1Rad) * tan(phi1Rad);
        c1 = eccPrimeSquared * cos(phi1Rad) * cos(phi1Rad);
        r1 = ae * (1.0 - eccSquared) / pow(1.0 - eccSquared
           * (sin(phi1Rad) * sin(phi1Rad)),1.5);
        d = x / (n1 * ellipsoidK0);

        *lat = phi1Rad - (n1 * tan(phi1Rad) / r1) * (d * d / 2.0
            - (5.0 + 3.0 * t1 + 10 * c1 - 4.0 * c1 * c1
            - 9.0 * eccPrimeSquared) * d * d * d * d / 24.0
            + (61.0 + 90.0 * t1 + 298 * c1 + 45.0 * t1 * t1
            - 252.0 * eccPrimeSquared - 3.0 * c1 * c1) * d * d * d * d * d * d / 720.0);

        *lat *= RAD_TO_DEG ;

        *lon = (d - (1.0 + 2.0 * t1 + c1) * d * d * d / 6.0
            + (5.0 - 2.0 * c1 + 28 * t1 - 3.0 * c1 * c1
            + 8.0 * eccPrimeSquared + 24.0 * t1 * t1)
            * d * d * d * d * d / 120.0) / cos(phi1Rad);

        /*! puts origin in middle of zone */
        longOrigin = double(zoneNumber - 1.) * 6. - 180. + 3.;

        *lon *= RAD_TO_DEG ;
        *lon += longOrigin ;
    }


    /*!
    * UTM zone:   [1,60]
    * Time zone:  [-12,12]
    * lon:       [-180,180]
    */
    bool isValidUtmTimeZone(int utmZone, int timeZone)
    {
        float lonUtmZone , lonTimeZone;
        lonUtmZone = float((utmZone - 1) * 6 - 180 + 3);
        lonTimeZone = float(timeZone * 15);
        if (fabs(lonTimeZone - lonUtmZone) <= 7.5f) return true;
        else return false;
    }


    bool computeLatLonMaps(const gis::Crit3DRasterGrid& myGrid,
                           gis::Crit3DRasterGrid* latMap, gis::Crit3DRasterGrid* lonMap,
                           const gis::Crit3DGisSettings& gisSettings)
    {
        if (! myGrid.isLoaded) return false;

        double latDegrees, lonDegrees;
        double utmX, utmY;

        latMap->initializeGrid(myGrid);
        lonMap->initializeGrid(myGrid);

        for (int row = 0; row < myGrid.header.nrRows; row++)
            for (int col = 0; col < myGrid.header.nrCols; col++)
                if (! isEqual(myGrid.value[row][col], myGrid.header.flag))
                {
                    myGrid.getXY(row, col, utmX, utmY);
                    getLatLonFromUtm(gisSettings, utmX, utmY, &latDegrees, &lonDegrees);

                    latMap->value[row][col] = float(latDegrees);
                    lonMap->value[row][col] = float(lonDegrees);
                }

        gis::updateMinMaxRasterGrid(latMap);
        gis::updateMinMaxRasterGrid(lonMap);

        latMap->isLoaded = true;
        lonMap->isLoaded = true;

        return true;
    }


    bool computeSlopeAspectBoundary(const gis::Crit3DRasterGrid& dem, gis::Crit3DRasterGrid* slopeMap,
                                    gis::Crit3DRasterGrid* aspectMap, float z, int row, int col)
    {
        if (!dem.isLoaded || !slopeMap || !aspectMap)
            return false;

        float flag = dem.header.flag;
        if (isEqual(z, flag))
            return false;

        const double cellSize = dem.header.cellSize;

        // compute dz/dy
        double dz = 0.0;
        double dy = 0.0;
        double dx = 0.0;
        for (int i = -1; i <= 1; ++i)
        {
            if (i != 0)
            {
                for(int j= -1; j <= 1; ++j)
                {
                    float z1 = dem.getValueFromRowCol(row+i, col+j);
                    if (! isEqual(z1, flag))
                    {
                        dz += i * (z - z1);
                        dy += cellSize;
                    }
                }
            }
        }

        dy = std::max(dy, EPSILON);
        const double dz_dy = dz / dy;

        // compute dz/dx
        dz = 0.0;
        dx = 0.0;
        for (int j=-1; j <=1; j++)
        {
            if (j != 0)
            {
                for(int i=-1; i <=1; i++)
                {
                    float z1 = dem.getValueFromRowCol(row+i, col+j);
                    if (! isEqual(z1, flag))
                    {
                        dz += j * (z - z1);
                        dx += cellSize;
                    }
                }
            }
        }

        dx = std::max(dx, EPSILON);
        double dz_dx = dz / dx;

        // flat cell: same convention as the interior cells (slope = 0, aspect = 0)
        if (std::abs(dz_dx) < EPSILON && std::abs(dz_dy) < EPSILON)
        {
            slopeMap->value[row][col] = 0.f;
            aspectMap->value[row][col] = 0.f;
            return true;
        }

        // slope in degrees
        double slope = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy)) * RAD_TO_DEG;
        slopeMap->value[row][col] = float(slope);

        // compute with zero to east
        double aspect = atan2(-dz_dy, dz_dx);

        // aspect in degrees: 0° = north, clockwise
        aspect = 90.0 - aspect * RAD_TO_DEG;
        if (aspect < 0)
            aspect += 360;

        aspectMap->value[row][col] = float(aspect);

        return true;
    }


    // Horn (3x3 standard)
    bool computeSlopeAspectMaps(const gis::Crit3DRasterGrid& dem,
                                gis::Crit3DRasterGrid* slopeMap,
                                gis::Crit3DRasterGrid* aspectMap)
    {
        if (!dem.isLoaded || !slopeMap || !aspectMap)
            return false;

        const double cellSize = dem.header.cellSize;

        slopeMap->initializeGrid(dem);
        aspectMap->initializeGrid(dem);

        const float flag = dem.header.flag;

        for (int row = 0; row < dem.header.nrRows; ++row)
        {
            for (int col = 0; col < dem.header.nrCols; ++col)
            {
                const float z = dem.value[row][col];

                if (isEqual(z, flag))
                    continue;

                if (isBoundary(dem, row, col))
                {
                    computeSlopeAspectBoundary(dem, slopeMap, aspectMap, z, row, col);
                    continue;
                }

                const double z1 = dem.value[row-1][col-1];
                const double z2 = dem.value[row-1][col];
                const double z3 = dem.value[row-1][col+1];
                const double z4 = dem.value[row][col-1];

                const double z6 = dem.value[row][col+1];
                const double z7 = dem.value[row+1][col-1];
                const double z8 = dem.value[row+1][col];
                const double z9 = dem.value[row+1][col+1];

                // Horn derivatives
                const double dzdx =
                    ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * cellSize);

                const double dzdy =
                    ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * cellSize);

                if (std::abs(dzdx) < EPSILON && std::abs(dzdy) < EPSILON)
                {
                    slopeMap->value[row][col] = 0.0;
                    aspectMap->value[row][col] = 0.0;
                    continue;
                }

                // slope
                const double slopeRad = std::atan(std::sqrt(dzdx*dzdx + dzdy*dzdy));

                slopeMap->value[row][col] = float(slopeRad * RAD_TO_DEG);

                // aspect
                double aspect = std::atan2(dzdy, -dzdx);

                // convert to GIS compass (0 = North, clockwise)
                aspect = 90.0 - aspect * RAD_TO_DEG;
                if (aspect < 0)
                    aspect += 360.0;

                aspectMap->value[row][col] = (float)aspect;
            }
        }

        gis::updateMinMaxRasterGrid(slopeMap);
        gis::updateMinMaxRasterGrid(aspectMap);

        slopeMap->isLoaded = true;
        aspectMap->isLoaded = true;

        return true;
    }


    bool mapAlgebra(gis::Crit3DRasterGrid* map1, gis::Crit3DRasterGrid* map2,
                    gis::Crit3DRasterGrid* outputMap, operationType myOperation)
    {
        if (! map1 || ! map2 || ! outputMap)
            return false;

        if (! map1->isLoaded || ! map2->isLoaded)
            return false;

        // output aliasing an input: initializeGrid would destroy it, so use a temporary raster
        if (outputMap == map1 || outputMap == map2)
        {
            Crit3DRasterGrid tmpGrid;
            if (! mapAlgebra(map1, map2, &tmpGrid, myOperation))
                    return false;

            return outputMap->copyGrid(tmpGrid);
        }

        const bool isSameHeader = compareGrids(*map1, *map2);

        if (! outputMap->initializeGrid(map1->header))
            return false;

        const float flag1 = map1->header.flag;
        const float flag2 = map2->header.flag;

        for (int row=0; row < outputMap->header.nrRows; ++row)
            for (int col=0; col < outputMap->header.nrCols; ++col)
            {
                const float x = map1->value[row][col];
                if (! isEqual(x, flag1))
                {
                    float y;
                    if (isSameHeader)
                    {
                        y = map2->value[row][col];
                    }
                    else
                    {
                        double utmx, utmy;
                        map1->getXY(row, col, utmx, utmy);
                        y = map2->getValueFromXY(utmx, utmy);
                    }

                    if (! isEqual(y, flag2))
                    {
                        if (myOperation == operationMin)
                            outputMap->value[row][col] = std::min(x, y);
                        else if (myOperation == operationMax)
                            outputMap->value[row][col] = std::max(x, y);
                        else if (myOperation == operationSum)
                            outputMap->value[row][col] = (x + y);
                        else if (myOperation == operationSubtract)
                            outputMap->value[row][col] = (x - y);
                        else if (myOperation == operationProduct)
                            outputMap->value[row][col] = (x * y);
                        else if (myOperation == operationDivide)
                        {
                            if (y != 0.f)
                                outputMap->value[row][col] = (x / y);
                        }
                    }
                }
            }

        gis::updateMinMaxRasterGrid(outputMap);
        return true;
    }


    // map1 and outputMap have to be the same header
    bool mapAlgebra(gis::Crit3DRasterGrid* map1, float myValue,
                    gis::Crit3DRasterGrid* outputMap, operationType myOperation)
    {
        if (outputMap == nullptr || map1 == nullptr)
            return false;

        if (! map1->isLoaded || ! outputMap->isLoaded)
            return false;

        if (! compareGrids(*map1, *outputMap))
            return false;

        // check before modifying the output (a failure in the middle would leave it half-written)
        if (myOperation == operationDivide && myValue == 0.f)
            return false;

        const float flag = map1->header.flag;
        const int nrRows = outputMap->header.nrRows;
        const int nrCols = outputMap->header.nrCols;

        for (int row=0; row < nrRows; row++)
        {
            for (int col=0; col < nrCols; col++)
            {
                if (! isEqual(map1->value[row][col], flag))
                {
                    if (myOperation == operationMin)
                        outputMap->value[row][col] = std::min(map1->value[row][col], myValue);
                    else if (myOperation == operationMax)
                        outputMap->value[row][col] = std::max(map1->value[row][col], myValue);
                    else if (myOperation == operationSum)
                        outputMap->value[row][col] = (map1->value[row][col] + myValue);
                    else if (myOperation == operationSubtract)
                        outputMap->value[row][col] = (map1->value[row][col] - myValue);
                    else if (myOperation == operationProduct)
                        outputMap->value[row][col] = (map1->value[row][col] * myValue);
                    else if (myOperation == operationDivide)
                        outputMap->value[row][col] = (map1->value[row][col] / myValue);
                }
            }
        }

        gis::updateMinMaxRasterGrid(outputMap);
        return true;
    }



    /*!
     * \brief return true if value(row, col) > values of all neighbours
     */
    bool isStrictMaximum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float adjZ;
        float z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header.flag))
            return false;

        for (int r = -1; r <= 1; r++)
        {
            for (int c = -1; c <= 1; c++)
            {
                if (r != 0 || c != 0)
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (! isEqual(adjZ, myGrid.header.flag))
                    {
                        if (z <= adjZ)
                            return false;
                    }
                 }
             }
        }

        return true;
    }


    /*!
     * \brief return true if value(row, col) <= all values of neighbours
     */
    bool isMinimum(const Crit3DRasterGrid& myGrid, bool isStrictMinumum, int row, int col)
    {
        float z, adjZ;
        z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header.flag))
            return false;

        for (int r=-1; r<=1; r++)
        {
            for (int c=-1; c<=1; c++)
            {
                if ((r != 0 || c != 0))
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (! isEqual(adjZ, myGrid.header.flag))
                    {
                        if (isStrictMinumum)
                        {
                            if (z >= adjZ)
                                return false;
                        }
                        else
                        {
                            if (z > adjZ)
                                return false;
                        }
                    }
                }
            }
        }

        return true;
    }


    /*!
     * \brief return true if (row, col) is a minimum, or adjacent to a minimum
     */
    bool isMinimumOrNearMinimum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z = myGrid.getValueFromRowCol(row, col);
        if (! isEqual(z, myGrid.header.flag))
        {
            for (int r=-1; r <= 1; r++)
            {
                for (int c=-1; c <= 1; c++)
                {
                    if (isMinimum(myGrid, false, row + r, col + c))
                        return true;
                }
            }
        }

        return false;
    }


    bool isBoundaryRunoff(const Crit3DIndexGrid& indexMap, const Crit3DRasterGrid& dtm, const Crit3DRasterGrid& aspectMap, int row, int col)
    {
        // is boundary?
        if (! isBoundary(dtm, row, col))
            return false;

        // check index map
        long index = indexMap.getValueFromRowCol(row, col);
        long nullIndex = long(indexMap.header.flag);
        if (index == nullIndex)
            return false;

        // is strict minimum: is boundary runoff
        if (isMinimum(dtm, true, row, col))
            return true;

        // check aspect
        float aspect = aspectMap.getValueFromRowCol(row, col);
        if (isEqual(aspect, aspectMap.header.flag))
            return false;

        int r = 0;
        int c = 0;
        if (aspect >= 135 && aspect <= 225)
            r = 1;
        else if ((aspect <= 45) || (aspect >= 315))
            r = -1;

        if (aspect >= 45 && aspect <= 135)
            c = 1;
        else if (aspect >= 225 && aspect <= 315)
            c = -1;

        long indexBoundary = indexMap.getValueFromRowCol(row + r, col + c);

        return (indexBoundary == nullIndex);
    }


    /*!
     * \brief return true if one neighbour (at least) is nodata
     */
    bool isBoundary(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header.flag))
            return false;

        for (int r = -1; r <= 1; r++)
            for (int c = -1; c <= 1; c++)
                if (r != 0 || c != 0)
                {
                    float zBoundary = myGrid.getValueFromRowCol(row + r, col + c);
                    if (isEqual(zBoundary, myGrid.header.flag))
                        return true;
                }

        return false;
    }


    float prevailingValue(const std::vector<float>& valueList)
    {
        if (valueList.empty())
            return NODATA;

        std::vector <float> values;
        std::vector <unsigned int> counters;

        values.push_back(valueList[0]);
        counters.push_back(1);
        for (size_t i = 1; i < valueList.size(); ++i)
        {
            bool isFound = false;
            size_t j = 0;
            while ((j < values.size()) && (! isFound))
            {
                if (isEqual(valueList[i], values[j]))
                {
                    isFound = true;
                    counters[j]++;
                }
                j++;
            }

            if (! isFound)
            {
                values.push_back(valueList[i]);
                counters.push_back(1);
            }
        }

        unsigned int maxCount = counters[0];
        float prevailing = values[0];
        for (size_t i = 1; i < values.size(); i++)
        {
            if (counters[i] > maxCount)
            {
                maxCount = counters[i];
                prevailing = values[i];
            }
        }

        return prevailing;
    }


    bool prevailingMap(const Crit3DRasterGrid& inputMap,  Crit3DRasterGrid *outputMap)
    {
        if (outputMap == nullptr || outputMap == &inputMap
            || ! outputMap->isLoaded || ! inputMap.isLoaded)
            return false;

        double x, y;
        int inputRow, inputCol;

        const int dim = 3;
        const double step = outputMap->header.cellSize / (2*dim+1);
        std::vector <float> valuesList;

        for (int row = 0; row < outputMap->header.nrRows ; row++)
            for (int col = 0; col < outputMap->header.nrCols; col++)
            {
                /*! center */
                outputMap->getXY(row, col, x, y);
                valuesList.resize(0);

                for (int i = -dim; i <= dim; i++)
                    for (int j = -dim; j <= dim; j++)
                        if (! gis::isOutOfGridXY(x+(i*step), y+(j*step), inputMap.header))
                        {
                            inputMap.getRowCol(x+(i*step), y+(j*step), inputRow, inputCol);
                            const float value = inputMap.value[inputRow][inputCol];
                            if (! isEqual(value, inputMap.header.flag))
                                valuesList.push_back(value);
                        }

                if (valuesList.empty())
                    outputMap->value[row][col] = outputMap->header.flag;
                else
                    outputMap->value[row][col] = prevailingValue(valuesList);
            }

        gis::updateMinMaxRasterGrid(outputMap);
        return true;
    }


    float topographicDistance(double x1, double y1, float z1, double x2, double y2, float z2, double distance,
                              const gis::Crit3DRasterGrid& dem)
    {
        if (distance < dem.header.cellSize)
            return 0;

        const int nrStep = int(distance / dem.header.cellSize);

        double Xi, Yi, Xf, Yf;
        float Zi;
        if (z1 < z2)
        {
            Xi = x1;
            Yi = y1;
            Zi = z1;
            Xf = x2;
            Yf = y2;
        }
        else
        {
            Xi = x2;
            Yi = y2;
            Zi = z2;
            Xf = x1;
            Yf = y1;
        }

        const double dx = (Xf - Xi) / double(nrStep);
        const double dy = (Yf - Yi) / double(nrStep);
        float maxDeltaZ = 0;

        for (int i=1; i <= nrStep; ++i)
        {
            double x = Xi + dx*i;
            double y = Yi + dy*i;

            const float demValue = dem.getValueFromXY(x, y);

            if (! isEqual(demValue, dem.header.flag))
                if (demValue > Zi)
                    maxDeltaZ = std::max(maxDeltaZ, demValue - Zi);
        }

        return maxDeltaZ;
    }


    bool topographicDistanceMap(const Crit3DPoint& myPoint, const gis::Crit3DRasterGrid& dem, Crit3DRasterGrid* myMap)
    {
        if (myMap == nullptr || myMap == &dem || ! dem.isLoaded)
            return false;

        if (! myMap->initializeGrid(dem))
            return false;

        double gridX, gridY;

        for (int row = 0; row < dem.header.nrRows; ++row)
            for (int col = 0; col < dem.header.nrCols; ++col)
            {
                const float demValue = dem.value[row][col];
                if (! isEqual(demValue, dem.header.flag))
                {
                    dem.getXY(row, col, gridX, gridY);

                    const double distance = computeDistance(gridX, gridY, myPoint.utm.x, myPoint.utm.y);

                    myMap->value[row][col] = topographicDistance(gridX, gridY, demValue,
                                            myPoint.utm.x, myPoint.utm.y, (float)myPoint.z, distance, dem);
                }
                else
                    myMap->value[row][col] = myMap->header.flag;
            }

        gis::updateMinMaxRasterGrid(myMap);
        return true;
    }


    double closestDistanceFromGrid(const Crit3DPoint &myPoint, const gis::Crit3DRasterGrid& dem)
    {
        float demValue = gis::getValueFromXY(dem, myPoint.utm.x, myPoint.utm.y);
        if (! isEqual(demValue, dem.header.flag))
        {
            return 0.0;
        }

        double closestDistanceFromGrid = NODATA;
        for (int row = 0; row < dem.header.nrRows; ++row)
        {
            for (int col = 0; col < dem.header.nrCols; ++col)
            {
                if (! isEqual(dem.getValueFromRowCol(row,col), dem.header.flag))
                {
                    double gridX, gridY;
                    dem.getXY(row, col, gridX, gridY);

                    const double distance = computeDistance(gridX, gridY, myPoint.utm.x, myPoint.utm.y);
                    if (isEqual(closestDistanceFromGrid, NODATA) || distance < closestDistanceFromGrid)
                    {
                        closestDistanceFromGrid = distance;
                    }
                }
            }
        }

        return closestDistanceFromGrid;
    }


    bool compareGrids(const gis::Crit3DRasterGrid& first, const gis::Crit3DRasterGrid& second)
    {
        return (first.header.nrRows == second.header.nrRows &&
                first.header.nrCols == second.header.nrCols &&
                isEqual(first.header.cellSize, second.header.cellSize) &&
                isEqual(first.header.llCorner.x, second.header.llCorner.x) &&
                isEqual(first.header.llCorner.y, second.header.llCorner.y));
    }


    /*!
     * \brief resampleGrid
     * \param threshold: minimum ratio of valid values (in the aggregation window)
     * required to compute the new value
     */
    bool resampleGrid(const gis::Crit3DRasterGrid& oldGrid, gis::Crit3DRasterGrid* newGrid,
                      const gis::Crit3DRasterHeader& newHeader, aggregationMethod elab,
                      float threshold)
    {
        if (newGrid == nullptr || newGrid == &oldGrid || ! oldGrid.isLoaded)
            return false;

        if (newHeader.cellSize <= 0. || oldGrid.header.cellSize <= 0.)
            return false;

        const double resampleFactor = newHeader.cellSize / oldGrid.header.cellSize;

        // aggregation methods are used only when the new cell is bigger than the old one
        if (resampleFactor > 1. && elab != aggrCenter && elab != aggrAverage
             && elab != aggrMedian && elab != aggrPrevailing)
            return false;

        if (! newGrid->initializeGrid(newHeader))
            return false;

        // for safety
        newGrid->header.invCellSize = 1.0 / newGrid->header.cellSize;

        const int nrRows = newGrid->header.nrRows;
        const int nrCols = newGrid->header.nrCols;
        const float newFlag = newGrid->header.flag;
        const float oldFlag = oldGrid.header.flag;

        for (int row = 0; row < nrRows; ++row)
            for (int col = 0; col < nrCols; ++col)
            {
                // initialize
                newGrid->value[row][col] = newFlag;
                float value = NODATA;

                if (resampleFactor <= 1. || elab == aggrCenter)
                {
                    double x, y;
                    newGrid->getXY(row, col, x, y);
                    int tmpRow, tmpCol;
                    oldGrid.getRowCol(x, y, tmpRow, tmpCol);

                    if (! gis::isOutOfGridRowCol(tmpRow, tmpCol, oldGrid))
                    {
                        const float tmpValue = oldGrid.value[tmpRow][tmpCol];
                        if (! isEqual(tmpValue, oldFlag))
                            value = tmpValue;
                    }
                }
                else
                {
                    const int nrStep = int(floor(resampleFactor)) + 1;
                    const double step = newGrid->header.cellSize / nrStep;
                    const double halfStep = step * 0.5;

                    double x0, y0;
                    newGrid->getXY(row, col, x0, y0);

                    const double xll = x0 - (newGrid->header.cellSize / 2) + halfStep;
                    const double yll = y0 - (newGrid->header.cellSize / 2) + halfStep;

                    std::vector<float> values;
                    int maxNrValues = 0;
                    for (int i = 0; i < nrStep; i++)
                    {
                        const double x = xll + step * i;
                        for (int j = 0; j < nrStep; j++)
                        {
                            const double y = yll + step * j;
                            float tmpValue = gis::getValueFromXY(oldGrid, x, y);

                            if (! isEqual(tmpValue, oldGrid.header.flag))
                            {
                                values.push_back(tmpValue);
                            }
                            maxNrValues++;
                        }
                    }

                    int nrValues = (int)values.size();

                    if (maxNrValues > 0)
                    {
                        if ((float(nrValues) / float(maxNrValues)) > threshold)
                        {
                            if (elab == aggrAverage)
                                value = statistics::mean(values);
                            else if (elab == aggrMedian)
                                value = sorting::percentile(values, nrValues, 50, true);
                            else if (elab == aggrPrevailing)
                            {
                                const int nrMissing = maxNrValues - nrValues;
                                if (nrMissing < nrValues)
                                {
                                    value = prevailingValue(values);
                                }
                            }
                        }
                    }
                }

                if (! isEqual(value, NODATA))
                {
                    newGrid->value[row][col] = value;
                }
            }

        gis::updateMinMaxRasterGrid(newGrid);
        newGrid->isLoaded = true;

        return true;
    }


    bool temporalYearlyInterpolation(const gis::Crit3DRasterGrid& firstGrid, const gis::Crit3DRasterGrid& secondGrid,
                                     int myYear, float minValue, float maxValue, gis::Crit3DRasterGrid* outGrid)
    {
        if (! firstGrid.isLoaded || ! secondGrid.isLoaded)
            return false;
        if (firstGrid.getMapTime().isNullTime() || secondGrid.getMapTime().isNullTime())
            return false;
        if (! compareGrids(firstGrid, secondGrid))
            return false;

        outGrid->initializeGrid(firstGrid);
        const int firstYear = firstGrid.getMapTime().date.year;
        const int secondYear = secondGrid.getMapTime().date.year;

        const float flag1 = firstGrid.header.flag;
        const float flag2 = secondGrid.header.flag;

        for (int row = 0; row < firstGrid.header.nrRows; row++)
            for (int col = 0; col < firstGrid.header.nrCols; col++)
            {
                const float firstVal = firstGrid.value[row][col];
                const float secondVal = secondGrid.value[row][col];

                if (! isEqual(firstVal, flag1) && ! isEqual(secondVal, flag2))
                {
                    outGrid->value[row][col] = statistics::linearInterpolation(float(firstYear), firstVal,
                                                                               float(secondYear), secondVal, float(myYear));

                    if (! isEqual(minValue, NODATA))
                        outGrid->value[row][col] = std::max(minValue, outGrid->value[row][col]);

                    if (! isEqual(maxValue, NODATA))
                        outGrid->value[row][col] = std::min(maxValue, outGrid->value[row][col]);
                }
            }

        updateMinMaxRasterGrid(outGrid);
        outGrid->setMapTime(Crit3DTime(Crit3DDate(1,1,myYear), 0));

        return true;
    }


    bool clipRasterWithRaster(const Crit3DRasterGrid* refRaster,
                              const Crit3DRasterGrid* maskRaster,
                              Crit3DRasterGrid* outputRaster)
    {
        if (refRaster == nullptr || maskRaster == nullptr || outputRaster == nullptr)
            return false;

        // initializeGrid / clear would destroy an input
        if (outputRaster == refRaster || outputRaster == maskRaster)
            return false;

        bool hasValidCell = false;
        const float maskFlag = maskRaster->header.flag;
        const auto& refHeader = refRaster->header;

        long firstRow = 0;
        long lastRow = 0;
        long firstCol = 0;
        long lastCol = 0;

        // Find the bounding box of the cells of the reference raster
        // that fall inside the mask.
        for (long row = 0; row < refHeader.nrRows; ++row)
        {
            for (long col = 0; col < refHeader.nrCols; ++col)
            {
                double x, y;
                gis::getUtmXYFromRowCol(refHeader, row, col, &x, &y);

                const float maskValue = maskRaster->getValueFromXY(x, y);

                if (isEqual(maskValue, maskFlag))
                    continue;

                if (! hasValidCell)
                {
                    firstRow = lastRow = row;
                    firstCol = lastCol = col;
                    hasValidCell = true;
                }
                else
                {
                    firstRow = std::min(firstRow, row);
                    lastRow  = std::max(lastRow, row);
                    firstCol = std::min(firstCol, col);
                    lastCol  = std::max(lastCol, col);
                }
            }
        }

        // No valid cells found.
        if (! hasValidCell)
        {
            outputRaster->clear();
            return false;
        }

        // Create the output header
        gis::Crit3DRasterHeader header = refHeader;

        header.nrRows = lastRow - firstRow + 1;
        header.nrCols = lastCol - firstCol + 1;

        header.llCorner.x = refHeader.llCorner.x + refHeader.cellSize * firstCol;
        header.llCorner.y = refHeader.llCorner.y + refHeader.cellSize * (refHeader.nrRows - lastRow - 1);

        // Initialize output raster.
        if (! outputRaster->initializeGrid(header))
        {
            outputRaster->clear();
            return false;
        }

        // Copy the clipped portion of the reference raster.
        // Cells outside the mask remain equal to the output flag.
        for (long row = firstRow; row <= lastRow; ++row)
        {
            const long outputRow = row - firstRow;
            for (long col = firstCol; col <= lastCol; ++col)
            {
                const long outputCol = col - firstCol;

                double x, y;
                gis::getUtmXYFromRowCol(refHeader, row, col, &x, &y);

                const float maskValue = maskRaster->getValueFromXY(x, y);

                if (! isEqual(maskValue, maskFlag))
                {
                    outputRaster->value[outputRow][outputCol] = refRaster->value[row][col];
                }
            }
        }

        gis::updateMinMaxRasterGrid(outputRaster);

        return true;
    }


    // replace the values ​​of the reference raster with the values ​​of the mask raster
    bool replaceRasterValues(const Crit3DRasterGrid *refRaster, const Crit3DRasterGrid *maskRaster, Crit3DRasterGrid *outputRaster)
    {
        if (refRaster == nullptr || maskRaster == nullptr || outputRaster == nullptr)
            return false;

        if (outputRaster == refRaster || outputRaster == maskRaster)
            return false;

        if (! outputRaster->initializeGrid(refRaster->header))
            return false;

        outputRaster->initializeGrid(refRaster->header);
        const float maskFlag = maskRaster->header.flag;

        for (long row = 0; row < refRaster->header.nrRows; row++)
        {
            for (long col = 0; col < refRaster->header.nrCols; col++)
            {
                double x, y;
                gis::getUtmXYFromRowCol(refRaster->header, row, col, &x, &y);

                const float maskValue = maskRaster->getValueFromXY(x, y);

                if (! isEqual(maskValue, maskFlag))
                    outputRaster->value[row][col] = maskValue;
                else
                    outputRaster->value[row][col] = refRaster->value[row][col];
            }
        }

        gis::updateMinMaxRasterGrid(outputRaster);

        return true;
    }


    bool deleteRangeOfValuesRaster(const gis::Crit3DRasterGrid* refRaster, float minValue, float maxValue,
                                   gis::Crit3DRasterGrid* outputRaster)
    {
        if (refRaster == nullptr || outputRaster == nullptr || outputRaster == refRaster)
            return false;

        if (! outputRaster->initializeGrid(refRaster->header))
            return false;

        const float flag = refRaster->header.flag;

        for (long row = 0; row < refRaster->header.nrRows; ++row)
        {
            for (long col = 0; col < refRaster->header.nrCols; ++col)
            {
                const float value = refRaster->value[row][col];

                if (! isEqual(value, flag) && (value < minValue || value > maxValue))
                {
                    outputRaster->value[row][col] = value;
                }
            }
        }

        gis::updateMinMaxRasterGrid(outputRaster);
        return true;
    }


    // return nr of valid cells and avg value
    bool rasterSummary(const Crit3DRasterGrid *myGrid, int &nrValids, float &avgValue, std::string &errorStr)
    {
        // initialize
        nrValids = 0;
        avgValue = NODATA;

        if ((myGrid == nullptr) || (! myGrid->isLoaded))
        {
            errorStr = "The raster is null or hasn't been loaded correctly.";
            return false;
        }

        double currentSum = 0.0;
        for (int row = 0; row < myGrid->header.nrRows; row++)
        {
            const float* rowPtr = myGrid->value[row];
            for (int col = 0; col < myGrid->header.nrCols; col++)
            {
                const float value = rowPtr[col];

                if (! isEqual(value, myGrid->header.flag)  && ! isEqual(value, NODATA))
                {
                    currentSum += (double)value;
                    nrValids++;
                }
            }
        }

        if (nrValids == 0)
        {
            errorStr = "No values.";
            return false;
        }

        avgValue = float(currentSum / nrValids);

        return true;
    }


    // assume integer values (categories)
    std::vector<int> extractUniqueValues(const Crit3DRasterGrid& raster)
    {
        std::set<int> uniqueValues;
        int flagInt = int(raster.header.flag);

        for (int row = 0; row < raster.header.nrRows; row++)
        {
            for (int col = 0; col < raster.header.nrCols; col++)
            {
                int valueInt = int(raster.value[row][col]);
                if (valueInt != flagInt)
                uniqueValues.insert(valueInt);
            }
        }

        return std::vector<int>(uniqueValues.begin(), uniqueValues.end());
    }


    /*!
     * \brief returns the minimum value of neighboring cells.
     * if no neighbors are valid, returns NODATA
     */
    float getNeighboursMinimumValue(const Crit3DRasterGrid& raster, int row, int col)
    {
        if (raster.isOutOfGrid(row, col))
            return NODATA;

        float minimum = NODATA;
        for (int r = -1; r <= 1; r++)
        {
            for (int c = -1; c <= 1; c++)
            {
                if (r != 0 || c != 0)
                {
                    float value = raster.getValueFromRowCol(row+r, col+c);

                    if (! isEqual(value, raster.header.flag))
                        if (minimum == NODATA || value < minimum)
                            minimum = value;
                }
            }
        }

        return minimum;
    }


    /*!
    * \brief crop raster using bounding box
    */
    bool cropRaster(const Crit3DRasterGrid* inputRaster, Crit3DRasterGrid* outputRaster,
                    int zoneNumber, const Crit3DGeoPoint& topLeft, const Crit3DGeoPoint& bottomRight)
    {
        if (inputRaster == nullptr || outputRaster == nullptr)
            return false;

        if (inputRaster == outputRaster)
            return false;

        const auto& inHeader = inputRaster->header;

        if (inHeader.nrRows <= 0 || inHeader.nrCols <= 0)
            return false;

        // Convert bounding-box corners from geographic to UTM coordinates
        gis::Crit3DUtmPoint p1, p2;

        gis::getUtmFromLatLon(zoneNumber, topLeft, &p1);
        gis::getUtmFromLatLon(zoneNumber, bottomRight, &p2);

        // Convert UTM coordinates to raster row/column
        int row1, col1, row2, col2;

        inputRaster->getRowCol(p1.x, p1.y, row1, col1);
        inputRaster->getRowCol(p2.x, p2.y, row2, col2);

        // Normalize row/column order
        int firstRow = std::min(row1, row2);
        int lastRow  = std::max(row1, row2);
        int firstCol = std::min(col1, col2);
        int lastCol  = std::max(col1, col2);

        // Check whether the bounding box intersects the raster
        if (lastRow < 0 || firstRow >= inHeader.nrRows ||
            lastCol < 0 || firstCol >= inHeader.nrCols)
        {
            return false;
        }

        // Limit the bounding box to the input raster
        firstRow = std::clamp(firstRow, 0, inHeader.nrRows - 1);
        lastRow  = std::clamp(lastRow,  0, inHeader.nrRows - 1);
        firstCol = std::clamp(firstCol, 0, inHeader.nrCols - 1);
        lastCol  = std::clamp(lastCol,  0, inHeader.nrCols - 1);

        const int newRows = lastRow - firstRow + 1;
        const int newCols = lastCol - firstCol + 1;

        // Create output header from the input header.
        gis::Crit3DRasterHeader outHeader = inHeader;

        outHeader.nrRows = newRows;
        outHeader.nrCols = newCols;

        // Update lower-left corner
        outHeader.llCorner.x = inHeader.llCorner.x + firstCol * inHeader.cellSize;

        outHeader.llCorner.y = inHeader.llCorner.y + (inHeader.nrRows - lastRow - 1) * inHeader.cellSize;

        // Initialize output raster
        if (! outputRaster->initializeGrid(outHeader))
            return false;

        // Copy raster values
        for (int row = 0; row < newRows; ++row)
        {
            for (int col = 0; col < newCols; ++col)
            {
                outputRaster->value[row][col] =
                    inputRaster->value[firstRow + row][firstCol + col];
            }
        }

        updateMinMaxRasterGrid(outputRaster);

        return true;
    }


    /*!
     * \brief resizeRasterCutEmptyFrame
     * remove the empty edges of a raster
     */
    bool resizeRasterCutEmptyFrame(const Crit3DRasterGrid* inputRaster,
                                   Crit3DRasterGrid* outputRaster, std::string& errorStr)
    {
        if (inputRaster == nullptr || outputRaster == nullptr)
        {
            errorStr = "Invalid raster pointer.";
            return false;
        }

        if (inputRaster == outputRaster)
        {
            errorStr = "Input and output raster must differ.";
            return false;
        }

        const auto& inHeader = inputRaster->header;

        const int rows = inHeader.nrRows;
        const int cols = inHeader.nrCols;

        if (rows <= 0 || cols <= 0)
        {
            errorStr = "Invalid input raster dimensions.";
            return false;
        }

        const float flag = inHeader.flag;

        int row0 = rows - 1;
        int row1 = 0;
        int col0 = cols - 1;
        int col1 = 0;

        bool containsData = false;

        // search valid-data bounds
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                if (! isEqual(inputRaster->value[r][c], flag))
                {
                    row0 = std::min(row0, r);
                    row1 = std::max(row1, r);

                    col0 = std::min(col0, c);
                    col1 = std::max(col1, c);

                    containsData = true;
                }
            }
        }

        if (! containsData)
        {
            errorStr = "Raster contains only empty values.";
            return false;
        }

        const int newRows = row1 - row0 + 1;
        const int newCols = col1 - col0 + 1;

        // copy entire header first
        outputRaster->clear();
        auto outHeader = inHeader;

        // set new values
        outHeader.nrRows = newRows;
        outHeader.nrCols = newCols;

        outHeader.llCorner.x = inHeader.llCorner.x + col0 * inHeader.cellSize;
        outHeader.llCorner.y = inHeader.llCorner.y + (inHeader.nrRows - row1 - 1) * inHeader.cellSize;

        outputRaster->initializeGrid(outHeader.flag);

        // copy values
        for (int r = 0; r < newRows; ++r)
        {
            for (int c = 0; c < newCols; ++c)
            {
                outputRaster->value[r][c] = inputRaster->value[r + row0][c + col0];
            }
        }

        updateMinMaxRasterGrid(outputRaster);

        return true;
    }


}
