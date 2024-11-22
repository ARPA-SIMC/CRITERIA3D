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


    Crit3DUtmPoint::Crit3DUtmPoint(double myX, double myY)
    {
        this->x = myX;
        this->y = myY;
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
                && x <= myGridHeader.llCorner.x + (myGridHeader.nrCols * myGridHeader.cellSize)
                && y >= myGridHeader.llCorner.y
                && y <= myGridHeader.llCorner.y + (myGridHeader.nrRows * myGridHeader.cellSize));
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

    Crit3DRasterGrid::Crit3DRasterGrid()
    {
        isLoaded = false;
        header = new Crit3DRasterHeader();
        colorScale = new Crit3DColorScale();
        minimum = NODATA;
        maximum = NODATA;
        value = nullptr;
    }


    void Crit3DRasterGrid::setConstantValue(float initValue)
    {
        for (int row = 0; row < this->header->nrRows; row++)
            for (int col = 0; col < header->nrCols; col++)
                value[row][col] = initValue;

        this->minimum = initValue;
        this->maximum = initValue;
    }


    bool Crit3DRasterGrid::initializeGrid()
    {
        this->value = new float*[unsigned(this->header->nrRows)];

        for (int row = 0; row < this->header->nrRows; row++)
        {
            this->value[row] = new float[unsigned(this->header->nrCols)];
            if (this->value[row] == nullptr)
            {
                // Memory error: file too big
                this->clear();
                return false;
            }
        }

        return true;
    }


    bool Crit3DRasterGrid::initializeGrid(float initValue)
    {
        if (! initializeGrid()) return false;

        setConstantValue(initValue);
        isLoaded = true;
        return true;
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterHeader& initHeader)
    {
        clear();
        *header = initHeader;
        return initializeGrid(header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DLatLonHeader& latLonHeader)
    {
        clear();

        header->nrRows = latLonHeader.nrRows;
        header->nrCols = latLonHeader.nrCols;
        header->flag = latLonHeader.flag;
        header->llCorner.y = latLonHeader.llCorner.latitude;
        header->llCorner.x = latLonHeader.llCorner.longitude;

        // avg value (usually not used, this is a raster value container for a lat lon grid)
        header->cellSize = (latLonHeader.dx + latLonHeader.dy) * 0.5;

        return initializeGrid(header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid)
    {
        clear();
        *header = *(initGrid.header);

        return initializeGrid(header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid, float initValue)
    {
        clear();
        *(header) = *(initGrid.header);
        return initializeGrid(initValue);
    }


    bool Crit3DRasterGrid::copyGrid(const Crit3DRasterGrid& initGrid)
    {
        clear();

        *(header) = *(initGrid.header);
        *(colorScale) = *(initGrid.colorScale);

        initializeGrid();

        for (int row = 0; row < header->nrRows; row++)
            for (int col = 0; col < header->nrCols; col++)
                    value[row][col] = initGrid.value[row][col];

        gis::updateMinMaxRasterGrid(this);
        isLoaded = true;
        return true;
    }


    bool Crit3DRasterGrid::setConstantValueWithBase(float initValue, const Crit3DRasterGrid& initGrid)
    {
        if (! this->isLoaded) return false;
        if (! (*(this->header) == *(initGrid.header))) return false;

        this->minimum = initValue;
        this->maximum = initValue;

        for (int row = 0; row < this->header->nrRows; row++)
            for (int col = 0; col < this->header->nrCols; col++)
                if (! isEqual(initGrid.value[row][col], initGrid.header->flag))
                    this->value[row][col] = initValue;

        return gis::updateMinMaxRasterGrid(this);
    }


    Crit3DPoint Crit3DRasterGrid::getCenter()
    {
        Crit3DPoint center;
        int row, col;
        center.utm.x = (header->llCorner.x + (header->nrCols * header->cellSize)/2.);
        center.utm.y = (header->llCorner.y + (header->nrRows * header->cellSize)/2.);
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


    void Crit3DRasterGrid::clear()
    {
        if (value != nullptr && header->nrRows > 0)
        {
            for (int row = 0; row < header->nrRows; row++)
                if (value[row] != nullptr)
                    delete [] value[row];

            delete [] value;
            value = nullptr;
        }

        mapTime.setNullTime();
        minimum = NODATA;
        maximum = NODATA;

        header->nrRows = 0;
        header->nrCols = 0;
        header->nrBytes = 4;
        header->cellSize = NODATA;
        header->llCorner.initialize();

        isLoaded = false;
    }


    // clean the grid (all NO DATA)
    void Crit3DRasterGrid::emptyGrid()
    {
        for (int row = 0; row < header->nrRows; row++)
            for (int col = 0; col < header->nrCols; col++)
                value[row][col] = header->flag;
    }


    Crit3DRasterGrid::~Crit3DRasterGrid()
    {
        clear();
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

    Crit3DUtmPoint* Crit3DRasterGrid::utmPoint(int row, int col)
    {
        double x, y;
        Crit3DUtmPoint *myPoint;

        x = header->llCorner.x + header->cellSize * (double(col) + 0.5);
        y = header->llCorner.y + header->cellSize * (double(header->nrRows - row) - 0.5);

        myPoint = new Crit3DUtmPoint(x, y);
        return myPoint;
    }


    void Crit3DRasterGrid::getXY(int row, int col, double& x, double& y) const
    {
        x = header->llCorner.x + header->cellSize * (double(col) + 0.5);
        y = header->llCorner.y + header->cellSize * (double(header->nrRows - row) - 0.5);
    }


    void Crit3DRasterGrid::getRowCol(double x, double y, int& row, int& col) const
    {
        row = (header->nrRows - 1) - int(floor((y - header->llCorner.y) / header->cellSize));
        col = int(floor((x - header->llCorner.x) / header->cellSize));
    }


    bool Crit3DRasterGrid::isOutOfGrid(int row, int col) const
    {
        return (row < 0 || row > (header->nrRows - 1) || col < 0 || col > header->nrCols - 1);
    }


    bool Crit3DRasterGrid::isFlag(int row, int col) const
    {
        if (isOutOfGrid(row, col))
            return true;
        else
            return (isEqual(value[row][col], header->flag));
    }

    float Crit3DRasterGrid::getValueFromRowCol(int row, int col) const
    {
        if (isOutOfGrid(row, col))
            return header->flag;
        else
            return value[row][col];
    }

    float Crit3DRasterGrid::getValueFromXY(double x, double y) const
    {
        int row, col;
        getRowCol(x, y, row, col);
        return getValueFromRowCol(row, col);
    }



    void convertFlagToNodata(Crit3DRasterGrid& myGrid)
    {
        if (myGrid.header->flag == NODATA)
            return;

        for (int row = 0; row < myGrid.header->nrRows; row++)
        {
            for (int col = 0; col < myGrid.header->nrCols; col++)
            {
                if (isEqual(myGrid.value[row][col], myGrid.header->flag))
                {
                    myGrid.value[row][col] = NODATA;
                }
            }
        }

        myGrid.header->flag = NODATA;
    }


    bool updateMinMaxRasterGrid(Crit3DRasterGrid* myGrid)
    {
        float myValue;
        bool isFirstValue = true;
        float minimum = NODATA;
        float maximum = NODATA;

        for (int row = 0; row < myGrid->header->nrRows; row++)
            for (int col = 0; col < myGrid->header->nrCols; col++)
            {
                myValue = myGrid->value[row][col];
                if (! isEqual(myValue, myGrid->header->flag)  && ! isEqual(myValue, NODATA))
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

        /*!  no values */
        if (isFirstValue)
            return false;

        myGrid->maximum = maximum;
        myGrid->minimum = minimum;

        if (! myGrid->colorScale->isFixedRange())
        {
            myGrid->colorScale->setRange(minimum, maximum);
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
        {
            int tmp = row0;
            row0 = row1;
            row1 = tmp;
        }
        row0 = std::max(row0, int(0));
        col0 = std::max(col0, int(0));
        row1 = std::min(row1, myGrid->header->nrRows-1);
        col1 = std::min(col1, myGrid->header->nrCols-1);

        for (int row = row0; row <= row1; row++)
            for (int col = col0; col <= col1; col++)
            {
                myValue = myGrid->value[row][col];
                if (! isEqual(myValue, myGrid->header->flag) && ! isEqual(myValue, NODATA))
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
            myGrid->colorScale->setRange(NODATA, NODATA);
            return false;
        }

        myGrid->colorScale->setRange(minimum, maximum);

        return true;
    }


    double computeDistancePoint(Crit3DUtmPoint* p0, Crit3DUtmPoint *p1)
    {
            double dx, dy;

            dx = p1->x - p0->x;
            dy = p1->y - p0->y;

            return sqrt((dx * dx) + (dy * dy));
    }


    float computeDistance(float x1, float y1, float x2, float y2)
    {
            float dx = x2 - x1;
            float dy = y2 - y1;

            return sqrtf((dx * dx) + (dy * dy));
    }


    std::vector<float> computeEuclideanDistanceStation2Area(std::vector<std::vector<int>>& cells,std::vector<std::vector<int>>& stations)
    {
        // è possibile sapere in quale cella (row,col) si trova la stazione?
        std::vector<float> distance(stations.size());
        for (int i=0;i<stations.size();i++)
        {
            distance[i] = (float)(stations[i][0] - cells[0][0])*(stations[i][0] - cells[0][0])+(stations[i][1] - cells[0][1])*(stations[i][1] - cells[0][1]);
            for (int j=1;j<cells[i].size();j++)
            {
                distance[i] = MINVALUE(distance[i],(stations[i][0] - cells[j][0])*(stations[i][0] - cells[j][0])+(stations[i][1] - cells[j][1])*(stations[i][1] - cells[j][1]));
            }
            distance[i] = float(sqrt(1.*distance[i]));
        }
        return distance;
    }


    std::vector<int> computeMetropolisDistanceStation2Area(std::vector<std::vector<int>>& cells,std::vector<std::vector<int>>& stations)
    {
        // è possibile sapere in quale cella (row,col) si trova la stazione?
        std::vector<int> distance(stations.size());
        for (int i=0; i<stations.size(); i++)
        {
            distance[i] = abs(stations[i][0] - cells[0][0])+abs(stations[i][1] - cells[0][1]);
            for (int j=1;j<cells[i].size();j++)
            {
                distance[i] = MINVALUE(distance[i],abs(stations[i][0] - cells[j][0])+abs(stations[i][1] - cells[j][1]));
            }
        }
        return distance;
    }


    void getRowColFromXY(const Crit3DRasterHeader& myHeader, double myX, double myY, int *row, int *col)
    {
        *row = (myHeader.nrRows - 1) - int(floor((myY - myHeader.llCorner.y) / myHeader.cellSize));
        *col = int(floor((myX - myHeader.llCorner.x) / myHeader.cellSize));
    }

    void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, int *row, int *col)
    {
        *row = (myHeader.nrRows - 1) - int(floor((p.y - myHeader.llCorner.y) / myHeader.cellSize));
        *col = int(floor((p.x - myHeader.llCorner.x) / myHeader.cellSize));
    }

    void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, Crit3DRasterCell* v)
    {
        v->row = (myHeader.nrRows - 1) - int(floor((p.y - myHeader.llCorner.y) / myHeader.cellSize));
        v->col = int(floor((p.x - myHeader.llCorner.x) / myHeader.cellSize));
    }

    void getRowColFromLonLat(const Crit3DLatLonHeader& myHeader, double lon, double lat, int *row, int *col)
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
        if (  row < 0 || row >= myGrid.header->nrRows
           || col < 0 || col >= myGrid.header->nrCols) return true;
        else return false;
    }

    bool isOutOfGridRowCol(int row, int col, const Crit3DLatLonHeader& header)
    {
        if (  row < 0 || row >= header.nrRows
            || col < 0 || col >= header.nrCols) return true;
        else return false;
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterGrid& myGrid,
                                            int row, int col, float* myX, float* myY)
    {
        *myX = float(myGrid.header->llCorner.x + myGrid.header->cellSize * (col + 0.5));
        *myY = float(myGrid.header->llCorner.y + myGrid.header->cellSize * (myGrid.header->nrRows - row) - 0.5);
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterHeader& myHeader,
                                           int row, int col, float* myX, float* myY)
    {
        *myX = float(myHeader.llCorner.x + myHeader.cellSize * (col) + 0.5);
        *myY = float(myHeader.llCorner.y + myHeader.cellSize * (myHeader.nrRows - row) - 0.5);
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

    float getValueFromUTMPoint(const Crit3DRasterGrid& myGrid, Crit3DUtmPoint& utmPoint)
    {
        return getValueFromXY(myGrid, utmPoint.x, utmPoint.y);
    }

    float getValueFromXY(const Crit3DRasterGrid& myGrid, double x, double y)
    {
        return myGrid.getValueFromXY(x, y);
    }

    bool isOutOfGridXY(double x, double y, Crit3DRasterHeader* header)
    {
        if ((x < header->llCorner.x) || (y < header->llCorner.y)
            || (x >= (header->llCorner.x + (header->nrCols * header->cellSize)))
            || (y >= (header->llCorner.y + (header->nrRows * header->cellSize))))
            return true;

        else return false;
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
        static double ellipsoidK0 = 0.9996;
        double eccSquared, lonOrigin, eccPrimeSquared, ae, a, n;
        double t, c,m,lonTemp, latRad,lonRad,lonOriginRad;

        Crit3DEllipsoid referenceEllipsoid;

        ae = referenceEllipsoid.equatorialRadius;
        eccSquared = referenceEllipsoid.eccentricitySquared;

        //!< Make sure the longitude is between -180.00 .. 179.9: */
        lonTemp = (lon + 180.) - floor((lon + 180.) / 360.) * 360. - 180.;

        latRad = lat * DEG_TO_RAD ;
        lonRad = lonTemp * DEG_TO_RAD ;
        *zoneNumber = int(ceil((lonTemp + 180.) / 6.));

        //!<  Special zones for Norway: */
        if ((lat >= 56.0) && (lat < 64.0) && (lonTemp >= 3.0) && (lonTemp < 12.0)) (*zoneNumber) = 32 ;
        //!<  Special zones for Svalbard: */
        if ((lat >= 72.0)&&(lat < 84.0))
        {
            if ((lonTemp >= 0) && (lonTemp < 9.0)) (*zoneNumber) = 31;
            else if ((lonTemp >= 9.0)&& (lonTemp < 21.0)) (*zoneNumber) = 33;
            else if ((lonTemp >= 21.0)&& ( lonTemp < 33.0)) (*zoneNumber) = 35;
            else if ((lonTemp >= 33.0)&& (lonTemp < 42.0)) (*zoneNumber) = 3;
        }

        //!<  puts origin in middle of zone */
        lonOrigin = ((*zoneNumber) - 1.) * 6. - 180. + 3.;
        lonOriginRad = lonOrigin * DEG_TO_RAD;

        eccPrimeSquared = eccSquared / (1. - eccSquared);

        n = ae / sqrt(1.0 - eccSquared * sin(latRad) * sin(latRad));
        t = tan(latRad) * tan(latRad);
        c = eccPrimeSquared * cos(latRad) * cos(latRad);
        a = cos(latRad) * (lonRad - lonOriginRad);

        m = ae * ((1. - eccSquared / 4. - 3. * eccSquared * eccSquared / 64.
          - 5. * eccSquared * eccSquared * eccSquared / 256.) * latRad
          - (3. * eccSquared / 8 + 3 * eccSquared * eccSquared / 32.
          + 45. * eccSquared * eccSquared * eccSquared / 1024.) * sin(2. * latRad)
          + (15. * eccSquared * eccSquared / 256.
          + 45. * eccSquared * eccSquared * eccSquared / 1024.) * sin(4. * latRad)
          - (35. * eccSquared * eccSquared * eccSquared / 3072.) * sin(6. * latRad));

        *utmEasting = (ellipsoidK0 * n * (a + (1 - t + c) * a * a * a / 6.
                   + (5. - 18. * t + t * t + 72. * c
                   - 58. * eccPrimeSquared) * a * a * a * a * a / 120.)
                   + 500000.);

        *utmNorthing = (ellipsoidK0 * (m + n * tan(latRad) * (a * a / 2.
                    + (5. - t + 9. * c + 4. * c * c) * a * a * a * a / 24.
                    + (61. - 58. * t + t * t + 600. * c
                    - 330. * eccPrimeSquared) * a * a * a * a * a * a / 720.)));

        //!<  offset for southern hemisphere: */
        if (lat < 0) (*utmNorthing) += 10000000.;
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
     * \param referenceLat
     * \param utmEasting: East Longitudes are positive, West longitudes are negative.
     * \param utmNorthing: North latitudes are positive, South latitudes are negative.
     * \param lat in decimal degrees.
     * \param lon in decimal degrees.
     */
    void utmToLatLon(int zoneNumber, double referenceLat, double utmEasting, double utmNorthing, double *lat, double *lon)
    {
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

        for (int row = 0; row < myGrid.header->nrRows; row++)
            for (int col = 0; col < myGrid.header->nrCols; col++)
                if (! isEqual(myGrid.value[row][col], myGrid.header->flag))
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


    bool computeSlopeAspectMaps(const gis::Crit3DRasterGrid& dem,
                                gis::Crit3DRasterGrid* slopeMap, gis::Crit3DRasterGrid* aspectMap)
    {
        if (! dem.isLoaded) return false;

        double dz_dx, dz_dy;
        double lateral_distance = dem.header->cellSize * sqrt(2);

        slopeMap->initializeGrid(dem);
        aspectMap->initializeGrid(dem);

        for (int row = 0; row < dem.header->nrRows; row++)
            for (int col = 0; col < dem.header->nrCols; col++)
            {
                float z = dem.value[row][col];
                if (! isEqual(z, dem.header->flag))
                {
                    /*! compute dz/dy */
                    double dz = 0;
                    double dy = 0;
                    for (int i=-1; i <=1; i++)
                    {
                        if (i != 0)
                        {
                            for(int j=-1; j <=1; j++)
                            {
                                float z1 = dem.getValueFromRowCol(row+i, col+j);
                                if (! isEqual(z1, dem.header->flag))
                                {
                                    dz += i * (z - z1);
                                    if (j == 0)
                                        dy += dem.header->cellSize;
                                    else
                                        dy += lateral_distance;
                                }
                            }
                        }
                    }

                    if (dy > 0)
                        dz_dy = dz / dy;
                    else
                        dz_dy = EPSILON;

                    /*! compute dz/dx */
                    dz = 0;
                    double dx = 0;
                    for (int j=-1; j <=1; j++)
                    {
                        if (j != 0)
                        {
                            for(int i=-1; i <=1; i++)
                            {
                                float z1 = dem.getValueFromRowCol(row+i, col+j);
                                if (! isEqual(z1, dem.header->flag))
                                {
                                    dz = dz + j * (z - z1);
                                    if (i == 0)
                                        dx += dem.header->cellSize;
                                    else
                                        dx += lateral_distance;
                                }
                            }
                        }
                    }

                    if (dx > 0)
                        dz_dx = dz / dx;
                    else
                        dz_dx = EPSILON;

                    /*! slope in degrees */
                    double slope = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy)) * RAD_TO_DEG;
                    slopeMap->value[row][col] = float(slope);

                    /*! avoid arctan to infinite */
                    if (dz_dx == 0.) dz_dx = EPSILON;

                    /*! compute with zero to east */
                    double aspect = 0.0;
                    if (dz_dx > 0)
                    {
                        aspect = atan(dz_dy / dz_dx);
                    }
                    else if (dz_dx < 0)
                    {
                        aspect = PI + atan(dz_dy / dz_dx);
                    }

                    /*! convert to zero from north and to degrees */
                    aspect += (PI / 2.);
                    aspect *= RAD_TO_DEG;

                    aspectMap->value[row][col] = float(aspect);
                }
            }

        gis::updateMinMaxRasterGrid(slopeMap);
        gis::updateMinMaxRasterGrid(aspectMap);

        aspectMap->isLoaded = true;
        slopeMap->isLoaded = true;

        return true;
    }


    bool mapAlgebra(gis::Crit3DRasterGrid* map1, gis::Crit3DRasterGrid* map2,
                    gis::Crit3DRasterGrid* outputMap, operationType myOperation)
    {
        if (outputMap == nullptr || map1 == nullptr || map2 == nullptr) return false;
        if (! (*(map1->header) == *(map2->header))) return false;
        if (! (*(outputMap->header) == *(map1->header))) return false;

        for (int row=0; row<outputMap->header->nrRows; row++)
            for (int col=0; col<outputMap->header->nrCols; col++)
            {
                if (!isEqual(map1->value[row][col], map1->header->flag)
                    && !isEqual(map2->value[row][col], map2->header->flag))
                {
                    if (myOperation == operationMin)
                    {
                        outputMap->value[row][col] = MINVALUE(map1->value[row][col], map2->value[row][col]);
                    }
                    else if (myOperation == operationMax)
                    {
                        outputMap->value[row][col] = MAXVALUE(map1->value[row][col], map2->value[row][col]);
                    }
                    else if (myOperation == operationSum)
                        outputMap->value[row][col] = (map1->value[row][col] + map2->value[row][col]);
                    else if (myOperation == operationSubtract)
                        outputMap->value[row][col] = (map1->value[row][col] - map2->value[row][col]);
                    else if (myOperation == operationProduct)
                        outputMap->value[row][col] = (map1->value[row][col] * map2->value[row][col]);
                    else if (myOperation == operationDivide)
                    {
                        if (map2->value[row][col] != 0.f)
                            outputMap->value[row][col] = (map1->value[row][col] / map2->value[row][col]);
                        else
                            return false;
                    }
                }
            }

        return true;
    }

    bool mapAlgebra(gis::Crit3DRasterGrid* map1, float myValue,
                    gis::Crit3DRasterGrid* outputMap, operationType myOperation)
    {
        if (outputMap == nullptr || map1 == nullptr) return false;
        if (! (*(map1->header) == *(outputMap->header))) return false;

        for (int row=0; row<outputMap->header->nrRows; row++)
            for (int col=0; col<outputMap->header->nrCols; col++)
            {
                if (! isEqual(map1->value[row][col], map1->header->flag))
                {
                    if (myOperation == operationMin)
                        outputMap->value[row][col] = MINVALUE(map1->value[row][col], myValue);
                    else if (myOperation == operationMax)
                        outputMap->value[row][col] = MAXVALUE(map1->value[row][col], myValue);
                    else if (myOperation == operationSum)
                        outputMap->value[row][col] = (map1->value[row][col] + myValue);
                    else if (myOperation == operationSubtract)
                        outputMap->value[row][col] = (map1->value[row][col] - myValue);
                    else if (myOperation == operationProduct)
                        outputMap->value[row][col] = (map1->value[row][col] * myValue);
                    else if (myOperation == operationDivide)
                    {
                        if (myValue != 0.f)
                            outputMap->value[row][col] = (map1->value[row][col] / myValue);
                        else
                            return false;
                    }
                }
            }

        return true;
    }

    /*!
     * \brief return true if value(row, col) > values of all neighbours
     */
    bool isStrictMaximum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float adjZ;
        float z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header->flag))
            return false;

        for (int r = -1; r <= 1; r++)
        {
            for (int c = -1; c <= 1; c++)
            {
                if (r != 0 || c != 0)
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (! isEqual(adjZ, myGrid.header->flag))
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
    bool isMinimum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z, adjZ;
        z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header->flag))
            return false;

        for (int r=-1; r<=1; r++)
        {
            for (int c=-1; c<=1; c++)
            {
                if ((r != 0 || c != 0))
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (! isEqual(adjZ, myGrid.header->flag))
                        if (z > adjZ)
                            return false;
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
        if (! isEqual(z, myGrid.header->flag))
        {
            for (int r=-1; r<=1; r++)
            {
                for (int c=-1; c<=1; c++)
                {
                    if (isMinimum(myGrid, row + r, col + c))
                        return true;
                }
            }
        }

        return false;
    }


    bool isBoundaryRunoff(const Crit3DRasterGrid& rasterRef, const Crit3DRasterGrid& aspectMap, int row, int col)
    {
        float value = rasterRef.getValueFromRowCol(row,col);
        float aspect = aspectMap.getValueFromRowCol(row,col);
        if (isEqual(value, rasterRef.header->flag) || isEqual(aspect, aspectMap.header->flag))
        {
            return false;
        }

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

        float valueBoundary = rasterRef.getValueFromRowCol(row + r, col + c);
        bool isBoundary = isEqual(valueBoundary, rasterRef.header->flag);

        return isBoundary;
    }


    /*!
     * \brief return true if one neighbour (at least) is nodata
     */
    bool isBoundary(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z = myGrid.getValueFromRowCol(row, col);
        if (isEqual(z, myGrid.header->flag))
            return false;

        for (int r = -1; r <= 1; r++)
            for (int c = -1; c <= 1; c++)
                if ((r != 0 || c != 0))
                {
                    float zBoundary = myGrid.getValueFromRowCol(row + r, col + c);
                    if (isEqual(zBoundary, myGrid.header->flag))
                        return true;
                }

        return false;
    }


    float prevailingValue(const std::vector<float>& valueList)
    {
        std::vector <float> values;
        std::vector <unsigned int> counters;
        float prevailing;
        unsigned int i, j, maxCount;
        bool isFound;

        values.push_back(valueList[0]);
        counters.push_back(1);
        for (i = 1; i < valueList.size(); i++)
        {
            isFound = false;
            j = 0;
            while ((j < values.size()) && (!isFound))
            {
                if (isEqual(valueList[i], values[j]))
                {
                    isFound = true;
                    counters[j]++;
                }
                j++;
            }

            if (!isFound)
            {
                values.push_back(valueList[i]);
                counters.push_back(1);
            }
        }

        maxCount = counters[0];
        prevailing = values[0];
         for (i = 1; i < values.size(); i++)
            if (counters[i] > maxCount)
            {
                maxCount = counters[i];
                prevailing = values[i];
            }

        return prevailing;
    }


    bool prevailingMap(const Crit3DRasterGrid& inputMap,  Crit3DRasterGrid *outputMap)
    {
        int i, j;
        float value;
        double x, y;
        int inputRow, inputCol;
        int dim = 3;

        std::vector <float> valuesList;
        double step = outputMap->header->cellSize / (2*dim+1);

        for (int row = 0; row < outputMap->header->nrRows ; row++)
            for (int col = 0; col < outputMap->header->nrCols; col++)
            {
                /*! center */
                outputMap->getXY(row, col, x, y);
                valuesList.resize(0);

                for (i = -dim; i <= dim; i++)
                    for (j = -dim; j <= dim; j++)
                        if (! gis::isOutOfGridXY(x+(i*step), y+(j*step), inputMap.header))
                        {
                            inputMap.getRowCol(x+(i*step), y+(j*step), inputRow, inputCol);
                            value = inputMap.value[inputRow][inputCol];
                            if (! isEqual(value, inputMap.header->flag))
                                valuesList.push_back(value);
                        }

                if (valuesList.size() == 0)
                    outputMap->value[row][col] = outputMap->header->flag;
                else
                    outputMap->value[row][col] = prevailingValue(valuesList);
            }

        return true;
    }


    float topographicDistance(float x1, float y1, float z1, float x2, float y2, float z2, float distance,
                              const gis::Crit3DRasterGrid& dem)
    {
        float x, y;
        float Xi, Yi, Zi, Xf, Yf;
        float dx, dy;
        float demValue;
        int i, nrStep;
        float maxDeltaZ;

        float stepMeter = float(dem.header->cellSize);

        if (distance < stepMeter)
            return 0;

        nrStep = int(distance / stepMeter);

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

        dx = (Xf - Xi) / float(nrStep);
        dy = (Yf - Yi) / float(nrStep);

        x = Xi;
        y = Yi;
        maxDeltaZ = 0;

        for (i=1; i<=nrStep; i++)
        {
            x = x + dx;
            y = y + dy;
            demValue = dem.getValueFromXY(x, y);
            if (! isEqual(demValue, dem.header->flag))
                if (demValue > Zi)
                    maxDeltaZ = MAXVALUE(maxDeltaZ, demValue - Zi);
        }

        return maxDeltaZ;
    }


    bool topographicDistanceMap(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& dem, Crit3DRasterGrid* myMap)
    {
        int row, col;
        float distance;
        double gridX, gridY;
        float demValue;

        myMap->initializeGrid(dem);

        for (row = 0; row < dem.header->nrRows; row++)
            for (col = 0; col < dem.header->nrCols; col++)
            {
                demValue = dem.value[row][col];
                if (! isEqual(demValue, dem.header->flag))
                {
                    dem.getXY(row, col, gridX, gridY);
                    distance = computeDistance(float(gridX), float(gridY), float(myPoint.utm.x), float(myPoint.utm.y));
                    myMap->value[row][col] = topographicDistance(float(gridX), float(gridY), demValue,
                                            float(myPoint.utm.x), float(myPoint.utm.y), float(myPoint.z), distance, dem);
                }
                else
                    myMap->value[row][col] = myMap->header->flag;
            }

        return true;
    }

    float closestDistanceFromGrid(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& dem)
    {

        int row, col;
        float closestDistanceFromGrid;
        float distance;
        double gridX, gridY;
        float demValue;

        demValue = gis::getValueFromXY(dem, myPoint.utm.x, myPoint.utm.y);
        if (! isEqual(demValue, dem.header->flag))
        {
            return 0;
        }

        closestDistanceFromGrid = NODATA;
        for (row = 0; row < dem.header->nrRows; row++)
        {
            for (col = 0; col < dem.header->nrCols; col++)
            {

                if (!isEqual(dem.getValueFromRowCol(row,col), dem.header->flag))
                {
                    dem.getXY(row, col, gridX, gridY);
                    distance = computeDistance(float(gridX), float(gridY), float(myPoint.utm.x), float(myPoint.utm.y));
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
        return (first.header->nrRows == second.header->nrRows &&
                first.header->nrCols == second.header->nrCols &&
                isEqual(first.header->cellSize, second.header->cellSize) &&
                isEqual(first.header->llCorner.x, second.header->llCorner.x) &&
                isEqual(first.header->llCorner.y, second.header->llCorner.y));
    }


    void resampleGrid(const gis::Crit3DRasterGrid& oldGrid, gis::Crit3DRasterGrid* newGrid,
                      gis::Crit3DRasterHeader* newHeader, aggregationMethod elab, float nodataRatioThreshold)
    {
        *(newGrid->header) = *newHeader;

        double resampleFactor = newGrid->header->cellSize / oldGrid.header->cellSize;

        int row, col, tmpRow, tmpCol, nrValues, maxValues;
        gis::Crit3DPoint myLL, myUR;
        std::vector<float> values;

        newGrid->initializeGrid();

        for (row = 0; row < newGrid->header->nrRows; row++)
            for (col = 0; col < newGrid->header->nrCols; col++)
            {
                newGrid->value[row][col] = newGrid->header->flag;

                float value = NODATA;
                if (resampleFactor < 1. || elab == aggrCenter)
                {
                    double x, y;
                    newGrid->getXY(row, col, x, y);
                    oldGrid.getRowCol(x, y, tmpRow, tmpCol);
                    if (! gis::isOutOfGridRowCol(tmpRow, tmpCol, oldGrid))
                    {
                        value = oldGrid.value[tmpRow][tmpCol];
                    }
                }
                else
                {
                    double x0, y0;
                    newGrid->getXY(row, col, x0, y0);
                    myLL.utm.x = x0 - (newGrid->header->cellSize / 2);
                    myLL.utm.y = y0 - (newGrid->header->cellSize / 2);
                    myUR.utm.x = x0 + (newGrid->header->cellSize / 2);
                    myUR.utm.y = y0 + (newGrid->header->cellSize / 2);

                    double step = oldGrid.header->cellSize * 0.5;

                    values.clear();
                    maxValues = 0;

                    double x = myLL.utm.x;
                    while(x <= myUR.utm.x)
                    {
                        double y = myLL.utm.y;
                        while(y <= myUR.utm.y)
                        {
                            maxValues++;
                            float tmpValue = gis::getValueFromXY(oldGrid, x, y);
                            if (! isEqual(tmpValue, oldGrid.header->flag))
                            {
                                values.push_back(tmpValue);
                            }

                            y += step;
                        }
                        x += step;
                    }
                    nrValues = int(values.size());

                    if (maxValues > 0)
                    {
                        if ((float(nrValues) / float(maxValues)) > nodataRatioThreshold)
                        {
                            if (elab == aggrAverage)
                                value = statistics::mean(values);
                            else if (elab == aggrMedian)
                                value = sorting::percentile(values, nrValues, 50, true);
                            else if (elab == aggrPrevailing)
                                value = prevailingValue(values);
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
    }


    bool temporalYearlyInterpolation(const gis::Crit3DRasterGrid& firstGrid, const gis::Crit3DRasterGrid& secondGrid,
                                     int myYear, float minValue, float maxValue, gis::Crit3DRasterGrid* outGrid)
    {

        int row, col;
        float firstVal, secondVal;
        int firstYear, secondYear;

        if (! firstGrid.isLoaded || ! secondGrid.isLoaded) return false;
        if (firstGrid.getMapTime().isNullTime() || secondGrid.getMapTime().isNullTime()) return false;
        if (! compareGrids(firstGrid, secondGrid)) return false;

        outGrid->initializeGrid(firstGrid);
        firstYear = firstGrid.getMapTime().date.year;
        secondYear = secondGrid.getMapTime().date.year;

        for (row = 0; row < firstGrid.header->nrRows; row++)
            for (col = 0; col < firstGrid.header->nrCols; col++)
                if (! gis::isOutOfGridRowCol(row, col, secondGrid))
                {
                    firstVal = firstGrid.value[row][col];
                    secondVal = secondGrid.value[row][col];

                    if (! isEqual(firstVal, NODATA) && ! isEqual(secondVal, NODATA))
                    {
                        outGrid->value[row][col] = statistics::linearInterpolation(float(firstYear), firstVal, float(secondYear), secondVal, float(myYear));
                        if (! isEqual(minValue, NODATA)) outGrid->value[row][col] = MAXVALUE(minValue, outGrid->value[row][col]);
                        if (! isEqual(maxValue, NODATA)) outGrid->value[row][col] = MINVALUE(maxValue, outGrid->value[row][col]);
                    }
                }

        updateMinMaxRasterGrid(outGrid);
        outGrid->setMapTime(Crit3DTime(Crit3DDate(1,1,myYear), 0));

        return true;
    }


    bool clipRasterWithRaster(gis::Crit3DRasterGrid* refRaster, gis::Crit3DRasterGrid* maskRaster, gis::Crit3DRasterGrid* outputRaster)
    {
        if (refRaster == nullptr || maskRaster == nullptr || outputRaster == nullptr)
            return false;

        gis::Crit3DRasterGrid* tmpRaster = new gis::Crit3DRasterGrid();
        tmpRaster->initializeGrid(*(refRaster->header));

        bool isFirst = true;
        long firstRow, lastRow, firstCol, lastCol;
        double x, y;
        for (long row = 0; row < refRaster->header->nrRows; row++)
        {
            for (long col = 0; col < refRaster->header->nrCols; col++)
            {
                gis::getUtmXYFromRowCol(refRaster->header, row, col, &x, &y);
                if (! isEqual(maskRaster->getValueFromXY(x, y), maskRaster->header->flag))
                {
                    tmpRaster->value[row][col] = refRaster->value[row][col];
                    if (isFirst)
                    {
                        firstRow = row;
                        lastRow = row;
                        firstCol = col;
                        lastCol = col;
                        isFirst = false;
                    }
                    else
                    {
                        firstRow = std::min(firstRow, row);
                        firstCol = std::min(firstCol, col);
                        lastRow = std::max(lastRow, row);
                        lastCol = std::max(lastCol, col);
                    }
                }
            }
        }

        // check no data
        if (isFirst)
        {
            tmpRaster->clear();
            return false;
        }

        // new header
        gis::Crit3DRasterHeader header;
        header = *(refRaster->header);
        header.nrRows = lastRow - firstRow + 1;
        header.nrCols = lastCol - firstCol + 1;
        header.llCorner.x = refRaster->header->llCorner.x + refRaster->header->cellSize * firstCol;
        header.llCorner.y = refRaster->header->llCorner.y + refRaster->header->cellSize * (refRaster->header->nrRows - (lastRow +1));

        // output raster
        outputRaster->initializeGrid(header);

        for (long row = 0; row < outputRaster->header->nrRows; row++)
        {
            for (long col = 0; col < outputRaster->header->nrCols; col++)
            {
                float value = tmpRaster->value[row + firstRow][col + firstCol];
                if (! isEqual (value, tmpRaster->header->flag))
                    outputRaster->value[row][col] = value;
            }
        }

        // clean memory
        tmpRaster->clear();

        gis::updateMinMaxRasterGrid(outputRaster);
        return true;
    }

}

