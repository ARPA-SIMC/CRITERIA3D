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

    bool Crit3DGeoPoint::isInsideGrid(const Crit3DGridHeader& latLonHeader) const
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
        cellSize = NODATA;
        flag = NODATA;
    }

    void Crit3DRasterHeader::convertFromLatLon(const Crit3DGridHeader& latLonHeader)
    {
        nrRows = latLonHeader.nrRows;
        nrCols = latLonHeader.nrCols;
        flag = latLonHeader.flag;
        llCorner.y = latLonHeader.llCorner.latitude;
        llCorner.x = latLonHeader.llCorner.longitude;
        cellSize = (latLonHeader.dx + latLonHeader.dy) * 0.5;
    }

    Crit3DGridHeader::Crit3DGridHeader()
    {
        nrRows = 0;
        nrCols = 0;
        dx = NODATA;
        dy = NODATA;
        flag = NODATA;
    }

    bool operator == (const Crit3DRasterHeader& myHeader1, const Crit3DRasterHeader& myHeader2)
    {
        return ((myHeader1.cellSize == myHeader2.cellSize) &&
                (myHeader1.flag == myHeader2.flag) &&
                (fabs(myHeader1.llCorner.x - myHeader2.llCorner.x) < 0.01) &&
                (fabs(myHeader1.llCorner.y - myHeader2.llCorner.y) < 0.01) &&
                (myHeader1.nrCols == myHeader2.nrCols) &&
                (myHeader1.nrRows == myHeader2.nrRows));
    }

    bool Crit3DRasterHeader::isEqualTo(const Crit3DRasterHeader& myHeader)
    {
        return ((cellSize == myHeader.cellSize) &&
                (flag == myHeader.flag) &&
                (fabs(llCorner.x - myHeader.llCorner.x) < 0.01) &&
                (fabs(llCorner.y - myHeader.llCorner.y) < 0.01) &&
                (nrCols == myHeader.nrCols) &&
                (nrRows == myHeader.nrRows));
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
        if (! this->initializeGrid()) return false;

        this->setConstantValue(initValue);

        this->isLoaded = true;
        return true;
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterHeader& initHeader)
    {
        this->clear();

        *(this->header) = initHeader;

        return this->initializeGrid(this->header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DGridHeader& latlonHeader)
    {
        this->clear();

        Crit3DRasterHeader initHeader;
        initHeader.convertFromLatLon(latlonHeader);
        *(this->header) = initHeader;

        return this->initializeGrid(this->header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid)
    {
        clear();
        *header = *(initGrid.header);

        return initializeGrid(header->flag);
    }


    bool Crit3DRasterGrid::initializeGrid(const Crit3DRasterGrid& initGrid, float initValue)
    {
        this->clear();

        *(this->header) = *(initGrid.header);

        return this->initializeGrid(initValue);
    }


    bool Crit3DRasterGrid::copyGrid(const Crit3DRasterGrid& initGrid)
    {
        this->clear();

        *(this->header) = *(initGrid.header);
        *(this->colorScale) = *(initGrid.colorScale);

        this->initializeGrid();

        for (int row = 0; row < this->header->nrRows; row++)
            for (int col = 0; col < this->header->nrCols; col++)
                    this->value[row][col] = initGrid.value[row][col];

        gis::updateMinMaxRasterGrid(this);
        this->isLoaded = true;
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
                if (initGrid.value[row][col] != initGrid.header->flag)
                    this->value[row][col] = initValue;

        return gis::updateMinMaxRasterGrid(this);
    }


    Crit3DPoint Crit3DRasterGrid::mapCenter()
    {
        int myRow, myCol;
        Crit3DPoint myPoint;

        myPoint.utm.x = (header->llCorner.x + (header->nrCols * header->cellSize)/2.);
        myPoint.utm.y = (header->llCorner.y + (header->nrRows * header->cellSize)/2.);
        getRowColFromXY(*this, myPoint.utm.x, myPoint.utm.y, &myRow, &myCol);
        myPoint.z = double(this->value[myRow][myCol]);

        return myPoint;
    }


    void Crit3DRasterGrid::clear()
    {
        if (value != nullptr && header->nrRows > 0)
        {
            for (int myRow = 0; myRow < header->nrRows; myRow++)
                if (value[myRow] != nullptr)
                    delete [] value[myRow];

            delete [] value;
            value = nullptr;
        }

        mapTime = getNullTime();
        minimum = NODATA;
        maximum = NODATA;
        header->nrRows = 0;
        header->nrCols = 0;
        header->cellSize = NODATA;
        header->llCorner.initialize();

        isLoaded = false;
    }


    void Crit3DRasterGrid::emptyGrid()
    {
        for (int myRow = 0; myRow < header->nrRows; myRow++)
            for (int myCol = 0; myCol < header->nrCols; myCol++)
                value[myRow][myCol] = header->flag;
    }

    Crit3DRasterGrid::~Crit3DRasterGrid()
    {
        clear();
    }


    /*!
     * \brief return X,Y of cell center
     * \param myRow
     * \param myCol
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

    Crit3DUtmPoint* Crit3DRasterGrid::utmPoint(int myRow, int myCol)
    {
        double x, y;
        Crit3DUtmPoint *myPoint;

        x = header->llCorner.x + header->cellSize * (double(myCol) + 0.5);
        y = header->llCorner.y + header->cellSize * (double(header->nrRows - myRow) - 0.5);

        myPoint = new Crit3DUtmPoint(x, y);
        return myPoint;
    }


    void Crit3DRasterGrid::getXY(int myRow, int myCol, double* x, double* y)
    {
        *x = header->llCorner.x + header->cellSize * (double(myCol) + 0.5);
        *y = header->llCorner.y + header->cellSize * (double(header->nrRows - myRow) - 0.5);
    }


    void checkMinimumRange(float& minimum, float& maximum)
    {
        if (isEqual(maximum, 0) && isEqual(maximum, 0))
        {
            maximum = 0.01f;
            return;
        }

        float avg = (maximum + minimum) * 0.5f;
        float minRange = std::max(0.01f, float(fabs(avg)) * 0.01f);

        if ((maximum - minimum) < minRange)
        {
            minimum = avg - minRange * 0.5f;
            maximum = avg + minRange * 0.5f;
            return;
        }
    }


    bool updateMinMaxRasterGrid(Crit3DRasterGrid* myGrid)
    {
        float myValue;
        bool isFirstValue = true;
        float minimum = NODATA;
        float maximum = NODATA;

        for (int myRow = 0; myRow < myGrid->header->nrRows; myRow++)
            for (int myCol = 0; myCol < myGrid->header->nrCols; myCol++)
            {
                myValue = myGrid->value[myRow][myCol];
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
        if (isFirstValue) return false;

        myGrid->maximum = maximum;
        myGrid->minimum = minimum;

        checkMinimumRange(minimum, maximum);
        myGrid->colorScale->maximum = maximum;
        myGrid->colorScale->minimum = minimum;

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

        for (int myRow = row0; myRow <= row1; myRow++)
            for (int myCol = col0; myCol <= col1; myCol++)
            {
                myValue = myGrid->value[myRow][myCol];
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
            myGrid->colorScale->maximum = NODATA;
            myGrid->colorScale->minimum = NODATA;
            return false;
        }

        checkMinimumRange(minimum, maximum);
        myGrid->colorScale->maximum = maximum;
        myGrid->colorScale->minimum = minimum;

        return true;
    }


    double computeDistancePoint(Crit3DUtmPoint* p0, Crit3DUtmPoint *p1)
    {
            double dx, dy;

            dx = p1->x - p0->x;
            dy = p1->y - p0->y;

            return sqrt((dx * dx)+(dy * dy));
    }


    float computeDistance(float x1, float y1, float x2, float y2)
    {
            float dx, dy;

            dx = x2 - x1;
            dy = y2 - y1;

            return sqrt((dx * dx)+(dy * dy));
    }

    void getRowColFromXY(const Crit3DRasterGrid& myGrid, double myX, double myY, int *row, int *col)
    {
        *row = (myGrid.header->nrRows - 1) - int(floor((myY - myGrid.header->llCorner.y) / myGrid.header->cellSize));
        *col = int(floor((myX - myGrid.header->llCorner.x) / myGrid.header->cellSize));
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

    void getMeteoGridRowColFromXY (const Crit3DGridHeader& myHeader, double myX, double myY, int *row, int *col)
    {
        *row = int(floor((myY - myHeader.llCorner.latitude) / myHeader.dy));
        *col = int(floor((myX - myHeader.llCorner.longitude) / myHeader.dx));
    }

    void getRowColFromLatLon(const Crit3DGridHeader& latLonHeader, const Crit3DGeoPoint& p, int* myRow, int* myCol)
    {
        *myRow = (latLonHeader.nrRows - 1) - int(floor((p.latitude - latLonHeader.llCorner.latitude) / latLonHeader.dy));
        *myCol = int(floor((p.longitude - latLonHeader.llCorner.longitude) / latLonHeader.dx));
    }

    bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DRasterGrid& myGrid)
    {
        if (  myRow < 0 || myRow >= myGrid.header->nrRows
           || myCol < 0 || myCol >= myGrid.header->nrCols) return true;
        else return false;
    }

    bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DGridHeader& header)
    {
        if (  myRow < 0 || myRow >= header.nrRows
            || myCol < 0 || myCol >= header.nrCols) return true;
        else return false;
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterGrid& myGrid,
                                            int myRow, int myCol, float* myX, float* myY)
    {
        *myX = float(myGrid.header->llCorner.x + myGrid.header->cellSize * (myCol + 0.5));
        *myY = float(myGrid.header->llCorner.y + myGrid.header->cellSize * (myGrid.header->nrRows - myRow) - 0.5);
    }

    void getUtmXYFromRowColSinglePrecision(const Crit3DRasterHeader& myHeader,
                                           int myRow, int myCol, float* myX, float* myY)
    {
        *myX = float(myHeader.llCorner.x + myHeader.cellSize * (myCol) + 0.5);
        *myY = float(myHeader.llCorner.y + myHeader.cellSize * (myHeader.nrRows - myRow) - 0.5);
    }

    void getUtmXYFromRowCol(const Crit3DRasterGrid& myGrid, int myRow, int myCol, double* myX, double* myY)
    {
            *myX = myGrid.header->llCorner.x + myGrid.header->cellSize * (myCol + 0.5);
            *myY = myGrid.header->llCorner.y + myGrid.header->cellSize * (myGrid.header->nrRows - myRow - 0.5);
    }

    void getUtmXYFromRowCol(const Crit3DRasterHeader& myHeader, int myRow, int myCol, double* myX, double* myY)
    {
            *myX = myHeader.llCorner.x + myHeader.cellSize * (myCol + 0.5);
            *myY = myHeader.llCorner.y + myHeader.cellSize * (myHeader.nrRows - myRow - 0.5);
    }

    void getLatLonFromRowCol(const Crit3DGridHeader& latLonHeader, int myRow, int myCol, double* lat, double* lon)
    {
            *lon = latLonHeader.llCorner.longitude + latLonHeader.dx * (myCol + 0.5);
            *lat = latLonHeader.llCorner.latitude + latLonHeader.dy * (latLonHeader.nrRows - myRow - 0.5);
    }

    void getLatLonFromRowCol(const Crit3DGridHeader& latLonHeader, const Crit3DRasterCell& v, Crit3DGeoPoint* p)
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
        int myRow, myCol;

        if (gis::isOutOfGridXY(x, y, myGrid.header)) return myGrid.header->flag ;
        getRowColFromXY(myGrid, x, y, &myRow, &myCol);
        return myGrid.value[myRow][myCol];
    }

    bool Crit3DRasterGrid::isFlag(int myRow, int myCol)
    {
        if (myRow < 0 || myRow > (header->nrRows - 1) || myCol < 0 || myCol > header->nrCols - 1)
            return true;
        else
            return (isEqual(value[myRow][myCol], header->flag));
    }

    float Crit3DRasterGrid::getFastValueXY(float x, float y) const
    {
        int myRow, myCol;

        myRow = (header->nrRows-1) - int((double(y) - header->llCorner.y) / header->cellSize);
        myCol = int((double(x) - header->llCorner.x) / header->cellSize);
        return getValueFromRowCol(myRow, myCol);
    }

    float Crit3DRasterGrid::getValueFromRowCol(int myRow, int myCol) const
    {
        if (myRow < 0 || myRow > (header->nrRows - 1) || myCol < 0 || myCol > header->nrCols - 1)
            return header->flag;
        else
            return value[myRow][myCol];
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


    void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, const Crit3DUtmPoint& utmPoint, Crit3DGeoPoint *geoPoint)
    {
        gis::utmToLatLon(gisSettings.utmZone, gisSettings.startLocation.latitude, utmPoint.x, utmPoint.y, &(geoPoint->latitude), &(geoPoint->longitude));
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

    void latLonToUtmForceZone(int zoneNumber, double lat, double lon, double *utmEasting, double *utmNorthing)
    {

        /*!
          equivalent to LatLonToUTM forcing UTM zone.
        */
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

        for (int myRow = 0; myRow < myGrid.header->nrRows; myRow++)
            for (int myCol = 0; myCol < myGrid.header->nrCols; myCol++)
                if (! isEqual(myGrid.value[myRow][myCol], myGrid.header->flag))
                {
                    getUtmXYFromRowCol(myGrid, myRow, myCol, &utmX, &utmY);
                    getLatLonFromUtm(gisSettings, utmX, utmY, &latDegrees, &lonDegrees);

                    latMap->value[myRow][myCol] = float(latDegrees);
                    lonMap->value[myRow][myCol] = float(lonDegrees);
                }

        gis::updateMinMaxRasterGrid(latMap);
        gis::updateMinMaxRasterGrid(lonMap);

        latMap->isLoaded = true;
        lonMap->isLoaded = true;

        return true;
    }


    bool computeSlopeAspectMaps(const gis::Crit3DRasterGrid& myDEM,
                                gis::Crit3DRasterGrid* slopeMap, gis::Crit3DRasterGrid* aspectMap)
    {
        if (! myDEM.isLoaded) return false;

        double dz_dx, dz_dy;
        double slope, aspect;
        double z, dz;
        double zNorth, zSouth, zEast, zWest;
        int i, nr;

        slopeMap->initializeGrid(myDEM);
        aspectMap->initializeGrid(myDEM);

        for (int myRow = 0; myRow < myDEM.header->nrRows; myRow++)
            for (int myCol = 0; myCol < myDEM.header->nrCols; myCol++)
            {
                z = double(myDEM.value[myRow][myCol]);
                if (! isEqual(z, double(myDEM.header->flag)))
                {
                    /*! compute dz/dy */
                    nr = 0;
                    dz = 0;
                    for (i=-1; i <=1; i++)
                    {
                        zNorth = double(myDEM.getValueFromRowCol(myRow-1, myCol+i));
                        zSouth = double(myDEM.getValueFromRowCol(myRow+1, myCol+i));
                        if (! isEqual(zNorth, double(myDEM.header->flag)))
                        {
                            dz += zNorth - z;
                            nr++;
                        }
                        if (! isEqual(zSouth, double(myDEM.header->flag)))
                        {
                            dz += z - zSouth;
                            nr++;
                        }
                    }
                    if (nr == 0)
                        dz_dy = EPSILON;
                    else
                        dz_dy = dz / (nr * myDEM.header->cellSize);

                    /*! compute dz/dx */
                    nr = 0;
                    dz = 0;
                    for (i=-1; i <=1; i++)
                    {
                        zWest = double(myDEM.getValueFromRowCol(myRow+i, myCol-1));
                        zEast = double(myDEM.getValueFromRowCol(myRow+i, myCol+1));
                        if (! isEqual (zWest, double(myDEM.header->flag)))
                        {
                            dz += zWest - z;
                            nr++;
                        }
                        if (! isEqual(zEast, double(myDEM.header->flag)))
                        {
                            dz += z - zEast;
                            nr++;
                        }
                    }
                    if (nr == 0)
                        dz_dx = EPSILON;
                    else
                        dz_dx = dz / (nr * myDEM.header->cellSize);

                    /*! slope in degrees */
                    slope = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy)) * RAD_TO_DEG;
                    slopeMap->value[myRow][myCol] = float(slope);

                    /*! avoid arctan to infinite */
                    if (dz_dx == 0.) dz_dx = EPSILON;

                    /*! compute with zero to east */
                    aspect = 0.0;
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

                    aspectMap->value[myRow][myCol] = float(aspect);
                }
            }

        gis::updateMinMaxRasterGrid(slopeMap);
        gis::updateMinMaxRasterGrid(aspectMap);

        aspectMap->isLoaded = true;
        slopeMap->isLoaded = true;

        return true;
    }


    bool mapAlgebra(gis::Crit3DRasterGrid* myMap1, gis::Crit3DRasterGrid* myMap2,
                    gis::Crit3DRasterGrid* myMapOut, operationType myOperation)
    {
        if (myMapOut == nullptr || myMap1 == nullptr || myMap2 == nullptr) return false;
        if (! (*(myMap1->header) == *(myMap2->header))) return false;
        if (! (*(myMapOut->header) == *(myMap1->header))) return false;

        for (int myRow=0; myRow<myMapOut->header->nrRows; myRow++)
            for (int myCol=0; myCol<myMapOut->header->nrCols; myCol++)
            {
                if (!isEqual(myMap1->value[myRow][myCol], myMap1->header->flag)
                    && !isEqual(myMap2->value[myRow][myCol], myMap2->header->flag))
                {
                    if (myOperation == operationMin)
                    {
                        myMapOut->value[myRow][myCol] = MINVALUE(myMap1->value[myRow][myCol], myMap2->value[myRow][myCol]);
                    }
                    else if (myOperation == operationMax)
                    {
                        myMapOut->value[myRow][myCol] = MAXVALUE(myMap1->value[myRow][myCol], myMap2->value[myRow][myCol]);
                    }
                    else if (myOperation == operationSum)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] + myMap2->value[myRow][myCol]);
                    else if (myOperation == operationSubtract)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] - myMap2->value[myRow][myCol]);
                    else if (myOperation == operationProduct)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] * myMap2->value[myRow][myCol]);
                    else if (myOperation == operationDivide)
                    {
                        if (myMap2->value[myRow][myCol] != 0.f)
                            myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] / myMap2->value[myRow][myCol]);
                        else
                            return false;
                    }
                }
            }

        return true;
    }

    bool mapAlgebra(gis::Crit3DRasterGrid* myMap1, float myValue,
                    gis::Crit3DRasterGrid* myMapOut, operationType myOperation)
    {
        if (myMapOut == nullptr || myMap1 == nullptr) return false;
        if (! (*(myMap1->header) == *(myMapOut->header))) return false;

        for (int myRow=0; myRow<myMapOut->header->nrRows; myRow++)
            for (int myCol=0; myCol<myMapOut->header->nrCols; myCol++)
            {
                if (myMap1->value[myRow][myCol] != myMap1->header->flag)
                {
                    if (myOperation == operationMin)
                        myMapOut->value[myRow][myCol] = MINVALUE(myMap1->value[myRow][myCol], myValue);
                    else if (myOperation == operationMax)
                        myMapOut->value[myRow][myCol] = MAXVALUE(myMap1->value[myRow][myCol], myValue);
                    else if (myOperation == operationSum)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] + myValue);
                    else if (myOperation == operationSubtract)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] - myValue);
                    else if (myOperation == operationProduct)
                        myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] * myValue);
                    else if (myOperation == operationDivide)
                    {
                        if (myValue != 0.f)
                            myMapOut->value[myRow][myCol] = (myMap1->value[myRow][myCol] / myValue);
                        else
                            return false;
                    }
                }
            }

        return true;
    }

    /*!
     * \brief return true if value(row, col) > all values of neighbours
     * \param myGrid Crit3DRasterGrid
     * \param row
     * \param col
     * \return true/false
     */
    bool isStrictMaximum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z, adjZ;
        z = myGrid.getValueFromRowCol(row, col);
        if (z == myGrid.header->flag) return false;

        for (int r = -1; r <= 1; r++)
        {
            for (int c = -1; c <= 1; c++)
            {
                if (r != 0 || c != 0)
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (adjZ != myGrid.header->flag)
                    {
                        if (z <= adjZ) return (false);
                    }
                 }
             }
        }

        return true;
    }


    /*!
     * \brief return true if value(row, col) <= all values of neighbours
     * \param myGrid Crit3DRasterGrid
     * \param row
     * \param col
     * \return true/false
     */
    bool isMinimum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z, adjZ;
        z = myGrid.getValueFromRowCol(row, col);
        if (z == myGrid.header->flag) return false;

        for (int r=-1; r<=1; r++)
        {
            for (int c=-1; c<=1; c++)
            {
                if ((r != 0 || c != 0))
                {
                    adjZ = myGrid.getValueFromRowCol(row+r, col+c);
                    if (adjZ != myGrid.header->flag)
                        if (z > adjZ) return (false);
                }
            }
        }
        return true;
    }


    /*!
     * \brief return true if (row, col) is a minimum, or adjacent to a minimum
     * \param myGrid Crit3DRasterGrid&
     * \param row
     * \param col
     * \return true/false
     */
    bool isMinimumOrNearMinimum(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z = myGrid.getValueFromRowCol(row, col);
        if (z != myGrid.header->flag)
        {
            for (int r=-1; r<=1; r++)
            {
                for (int c=-1; c<=1; c++)
                {
                    if (isMinimum(myGrid, row + r, col + c)) return true;
                }
            }
        }

        return false;
    }


    /*!
     * \brief return true if one neighbour (at least) is nodata
     * \param myGrid
     * \param row
     * \param col
     * \return true/false
     */
    bool isBoundary(const Crit3DRasterGrid& myGrid, int row, int col)
    {
        float z = myGrid.getValueFromRowCol(row, col);

        if (z != myGrid.header->flag)
        {
            for (int r = -1; r <= 1; r++)
            {
                for (int c = -1; c <= 1; c++)
                {
                    if ((r != 0 || c != 0))
                    {
                        if (myGrid.getValueFromRowCol(row + r, col + c) == myGrid.header->flag)
                            return true;
                    }
                }
            }
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

        return(prevailing);
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
                getUtmXYFromRowCol(*outputMap, row, col, &x, &y);
                valuesList.resize(0);

                for (i = -dim; i <= dim; i++)
                    for (j = -dim; j <= dim; j++)
                        if (! gis::isOutOfGridXY(x+(i*step), y+(j*step), inputMap.header))
                        {
                            gis::getRowColFromXY(inputMap, x+(i*step), y+(j*step), &inputRow, &inputCol);
                            value = inputMap.value[inputRow][inputCol];
                            if (value != inputMap.header->flag)
                                valuesList.push_back(value);
                        }

                if (valuesList.size() == 0)
                    outputMap->value[row][col] = outputMap->header->flag;
                else
                    outputMap->value[row][col] = prevailingValue(valuesList);
            }

        return true;
    }

    float topographicDistance(float X1, float Y1, float Z1, float X2, float Y2, float Z2, float distance,
                              const gis::Crit3DRasterGrid& myDEM)
    {
        float x, y;
        float Xi, Yi, Zi, Xf, Yf;
        float Dx, Dy;
        float demValue;
        int i, nrStep;
        float maxDeltaZ;

        float stepMeter = float(myDEM.header->cellSize);

        if (distance < stepMeter)
            return 0;

        nrStep = int(distance / stepMeter);

        if (Z1 < Z2)
        {
            Xi = X1;
            Yi = Y1;
            Zi = Z1;
            Xf = X2;
            Yf = Y2;
        }
        else
        {
            Xi = X2;
            Yi = Y2;
            Zi = Z2;
            Xf = X1;
            Yf = Y1;
        }

        Dx = (Xf - Xi) / nrStep;
        Dy = (Yf - Yi) / nrStep;

        x = Xi;
        y = Yi;
        maxDeltaZ = 0;

        for (i=1; i<=nrStep; i++)
        {
            x = x + Dx;
            y = y + Dy;
            demValue = myDEM.getFastValueXY(x, y);
            if (demValue != myDEM.header->flag)
                if (demValue > Zi)
                    maxDeltaZ = MAXVALUE(maxDeltaZ, demValue - Zi);
        }

        return maxDeltaZ;
    }

    bool topographicDistanceMap(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& myDEM, Crit3DRasterGrid* myMap)
    {

        int row, col;
        float distance;
        double gridX, gridY;
        float demValue;

        myMap->initializeGrid(myDEM);

        for (row = 0; row < myDEM.header->nrRows; row++)
            for (col = 0; col < myDEM.header->nrCols; col++)
            {
                demValue = myDEM.value[row][col];
                if (! isEqual(demValue, myDEM.header->flag))
                {
                    gis::getUtmXYFromRowCol(myDEM, row, col, &gridX, &gridY);
                    distance = computeDistance(float(gridX), float(gridY), float(myPoint.utm.x), float(myPoint.utm.y));
                    myMap->value[row][col] = topographicDistance(float(gridX), float(gridY), demValue,
                                            float(myPoint.utm.x), float(myPoint.utm.y), float(myPoint.z), distance, myDEM);
                }
                else
                    myMap->value[row][col] = myMap->header->flag;
            }

        return true;
    }

    float closestDistanceFromGrid(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& myDEM)
    {

        int row, col;
        float closestDistanceFromGrid;
        float distance;
        double gridX, gridY;
        float demValue;

        demValue = gis::getValueFromXY(myDEM, myPoint.utm.x, myPoint.utm.y);
        if (demValue != myDEM.header->flag)
        {
            return 0;
        }

        closestDistanceFromGrid = NODATA;
        for (row = 0; row < myDEM.header->nrRows; row++)
        {
            for (col = 0; col < myDEM.header->nrCols; col++)
            {

                if (!isEqual(myDEM.getValueFromRowCol(row,col), myDEM.header->flag))
                {
                    gis::getUtmXYFromRowCol(myDEM, row, col, &gridX, &gridY);
                    distance = computeDistance(float(gridX), float(gridY), float(myPoint.utm.x), float(myPoint.utm.y));
                    if (closestDistanceFromGrid == NODATA || distance < closestDistanceFromGrid)
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
                      const gis::Crit3DRasterHeader& header, aggregationMethod elab, float nodataThreshold)
    {
        *(newGrid->header) = header;

        double factor = newGrid->header->cellSize / oldGrid.header->cellSize;
        int row, col, tmpRow, tmpCol, nrValues, maxValues;
        float value, tmpValue;
        double x, y;
        gis::Crit3DPoint myLL, myUR;
        std::vector <float> values;

        newGrid->initializeGrid();

        for (row = 0; row < newGrid->header->nrRows; row++)
            for (col = 0; col < newGrid->header->nrCols; col++)
            {
                newGrid->value[row][col] = newGrid->header->flag;

                value = NODATA;

                if (factor < 1 || elab == aggrCenter)
                {
                    gis::getUtmXYFromRowCol(*newGrid, row, col, &x, &y);
                    gis::getRowColFromXY(oldGrid, x, y, &tmpRow, &tmpCol);
                    if (! gis::isOutOfGridRowCol(tmpRow, tmpCol, oldGrid))
                        value = oldGrid.value[tmpRow][tmpCol];
                }
                else
                {
                    gis::getUtmXYFromRowCol(*newGrid, row, col, &x, &y);

                    myLL.utm.x = x - (newGrid->header->cellSize / 2);
                    myLL.utm.y = y - (newGrid->header->cellSize / 2);
                    myUR.utm.x = x + (newGrid->header->cellSize / 2);
                    myUR.utm.y = y + (newGrid->header->cellSize / 2);

                    values.clear();
                    maxValues = 0;
                    for (x = myLL.utm.x; x <= myUR.utm.x; x += oldGrid.header->cellSize / 2)
                        for (y = myLL.utm.y; y <= myUR.utm.y; y += oldGrid.header->cellSize / 2)
                        {
                            maxValues++;
                            tmpValue = gis::getValueFromXY(oldGrid, x, y);
                            if (! isEqual(tmpValue, oldGrid.header->flag))
                                values.push_back(tmpValue);
                        }

                    nrValues = int(values.size());
                    if (nrValues / maxValues > nodataThreshold)
                    {
                        if (elab == aggrAverage)
                            value = statistics::mean(values, nrValues);
                        else if (elab == aggrMedian)
                            value = sorting::percentile(values, &nrValues, 50, true);
                        else if (elab == aggrPrevailing)
                            value = prevailingValue(values);
                    }
                }

                if (! isEqual(value, NODATA)) newGrid->value[row][col] = value;
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
        if (isNullTime(firstGrid.getMapTime()) || isNullTime(secondGrid.getMapTime())) return false;
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
}

