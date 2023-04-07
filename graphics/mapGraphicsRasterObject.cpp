/*!
    \file mapGraphicsRasterObject.cpp

    \abstract draw raster in MapGraphics widget

    This file is part of CRITERIA-3D distribution.

    CRITERIA-3D has been developed by A.R.P.A.E. Emilia-Romagna.

    \copyright
    CRITERIA-3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    CRITERIA-3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA-3D.  If not, see <http://www.gnu.org/licenses/>.

    \authors
    Fausto Tomei ftomei@arpae.it
    Gabriele Antolini gantolini@arpae.it
    Laura Costantini laura.costantini0@gmail.com
*/


#include "commonConstants.h"
#include "basicMath.h"
#include "mapGraphicsRasterObject.h"
#include <math.h>
#include <QMenu>

#define MAPBORDER 10


RasterObject::RasterObject(MapGraphicsView* _view, MapGraphicsObject *parent) :
    MapGraphicsObject(true, parent)
{
    setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    setFlag(MapGraphicsObject::ObjectIsMovable, false);
    setFlag(MapGraphicsObject::ObjectIsFocusable, false);

    view = _view;
    geoMap = new gis::Crit3DGeoMap();
    this->clear();
}


void RasterObject::clear()
{
    setDrawing(false);
    setDrawBorders(false);
    freeIndexesMatrix();

    latLonHeader.nrCols = 0;
    latLonHeader.nrRows = 0;

    matrix = nullptr;
    rasterPointer = nullptr;
    colorLegendPointer = nullptr;

    isGrid = false;
    isLatLon = false;
    isNetcdf = false;

    utmZone = NODATA;
    refCenterPixel = QPointF(NODATA, NODATA);
    longitudeShift = 0;

    isLoaded = false;
}


void RasterObject::setRaster(gis::Crit3DRasterGrid* rasterPtr)
{
    rasterPointer = rasterPtr;
}

gis::Crit3DRasterGrid* RasterObject::getRaster()
{
    return rasterPointer;
}

void RasterObject::setNetCDF(bool value)
{
    isNetcdf = value;
}

bool RasterObject::isNetCDF()
{
    return isNetcdf;
}

void RasterObject::setDrawing(bool value)
{
    isDrawing = value;
}

void RasterObject::setDrawBorders(bool value)
{
    isDrawBorder = value;
}

void RasterObject::setColorLegend(ColorLegend* colorLegendPtr)
{
    colorLegendPointer = colorLegendPtr;
}


/*!
\brief convert a point in geo (lat,lon) coordinates
 in pixel (local object) coordinates
*/
QPointF RasterObject::getPixel(const QPointF &geoPoint)
{
    QPointF pixel = view->tileSource()->ll2qgs(geoPoint, view->zoomLevel());
    pixel.setX(pixel.x() - refCenterPixel.x());
    pixel.setY(refCenterPixel.y() - pixel.y());
    return pixel;
}


gis::Crit3DLatLonHeader RasterObject::getLatLonHeader() const
{
    return latLonHeader;
}


/*!
\brief If sizeIsZoomInvariant() is true, this should return the size of the
 rectangle you want in PIXELS. If false, this should return the size of the rectangle in METERS. The
 rectangle should be centered at (0,0) regardless.
*/
 QRectF RasterObject::boundingRect() const
{
    int widthPixels = view->width() - MAPBORDER*2;
    int heightPixels = view->height() - MAPBORDER*2;

    return QRectF( -widthPixels, -heightPixels, widthPixels*2, heightPixels*2);
 }


 void RasterObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
 {
     Q_UNUSED(option)
     Q_UNUSED(widget)

    if (isDrawing)
    {
        setMapExtents();

        if (rasterPointer != nullptr)
            drawRaster(rasterPointer, painter);

        if (colorLegendPointer != nullptr)
            colorLegendPointer->update();
    }
}


/*!
 * \brief RasterObject::getRasterMaxSize
 * \return max of raster width and height (decimal degree)
 */
float RasterObject::getRasterMaxSize()
{
    return float(MAXVALUE(latLonHeader.nrRows * latLonHeader.dy,
                          latLonHeader.nrCols * latLonHeader.dx));
}


/*!
 * \brief RasterObject::getRasterCenter
 * \return center of raster (lat lon)
 */
gis::Crit3DGeoPoint* RasterObject::getRasterCenter()
{
    gis::Crit3DGeoPoint* center = new(gis::Crit3DGeoPoint);
    center->latitude = latLonHeader.llCorner.latitude + (latLonHeader.nrRows * latLonHeader.dy) * 0.5;
    center->longitude = latLonHeader.llCorner.longitude + (latLonHeader.nrCols * latLonHeader.dx) * 0.5;
    return center;
}


void RasterObject::freeIndexesMatrix()
{
    if (matrix == nullptr) return;

    for (int row = 0; row < latLonHeader.nrRows; row++)
        if (matrix[row] != nullptr)
            delete [] matrix[row];

    if (latLonHeader.nrRows != 0) delete [] matrix;

    matrix = nullptr;
}


void RasterObject::initializeIndexesMatrix()
{
    matrix = new RowCol*[unsigned(latLonHeader.nrRows)];

    for (int row = 0; row < latLonHeader.nrRows; row++)
        matrix[row] = new RowCol[unsigned(latLonHeader.nrCols)];

    for (int row = 0; row < latLonHeader.nrRows; row++)
        for (int col = 0; col < latLonHeader.nrCols; col++)
        {
            matrix[row][col].row = NODATA;
            matrix[row][col].col = NODATA;
        }
}


bool RasterObject::initializeUTM(gis::Crit3DRasterGrid* myRaster, const gis::Crit3DGisSettings& gisSettings, bool isGrid_)
{
    if (myRaster == nullptr) return false;
    if (! myRaster->isLoaded) return false;

    double lat, lon, x, y;
    int utmRow, utmCol;

    isLatLon = false;

    isGrid = isGrid_;
    utmZone = gisSettings.utmZone;
    rasterPointer = myRaster;

    freeIndexesMatrix();
    gis::getGeoExtentsFromUTMHeader(gisSettings, myRaster->header, &latLonHeader);
    initializeIndexesMatrix();

    for (int row = 0; row < latLonHeader.nrRows; row++)
        for (int col = 0; col < latLonHeader.nrCols; col++)
        {
            gis::getLatLonFromRowCol(latLonHeader, row, col, &lat, &lon);
            gis::latLonToUtmForceZone(gisSettings.utmZone, lat, lon, &x, &y);
            if (! gis::isOutOfGridXY(x, y, myRaster->header))
            {
                gis::getRowColFromXY(*(myRaster->header), x, y, &utmRow, &utmCol);
                if (isGrid || ! isEqual(myRaster->getValueFromRowCol(utmRow, utmCol), myRaster->header->flag))
                {
                    matrix[row][col].row = utmRow;
                    matrix[row][col].col = utmCol;
                }
            }
        }

    setDrawing(true);
    setDrawBorders(isGrid);
    isLoaded = true;

    return true;
}


bool RasterObject::initializeLatLon(gis::Crit3DRasterGrid* myRaster, const gis::Crit3DGisSettings& gisSettings,
                                    const gis::Crit3DLatLonHeader &latLonHeader_, bool isGrid_)
{
    if (myRaster == nullptr) return false;
    if (! myRaster->isLoaded) return false;

    isLatLon = true;

    isGrid = isGrid_;
    utmZone = gisSettings.utmZone;
    rasterPointer = myRaster;

    freeIndexesMatrix();

    latLonHeader = latLonHeader_;

    // TODO improve management of 0-360 netcdf grid
    double maxLongitude = latLonHeader.llCorner.longitude + latLonHeader.nrCols * latLonHeader.dx;
    if (maxLongitude > 180)
    {
        longitudeShift = maxLongitude - 180;
        latLonHeader.llCorner.longitude -= longitudeShift;
    }

    setDrawing(true);
    setDrawBorders(isGrid_);
    isLoaded = true;

    return true;
}


bool RasterObject::getCurrentWindow(gis::Crit3DRasterWindow* window)
{
    // current view extent
    int row0, row1, col0, col1;
    gis::getRowColFromLatLon(this->latLonHeader, this->geoMap->bottomLeft, &row0, &col0);
    gis::getRowColFromLatLon(this->latLonHeader, this->geoMap->topRight, &row1, &col1);

    // check if current window is out of map
    if (((col0 < 0) && (col1 < 0))
    || ((row0 < 0) && (row1 < 0))
    || ((col0 >= this->latLonHeader.nrCols) && (col1 >= this->latLonHeader.nrCols))
    || ((row0 >= this->latLonHeader.nrRows) && (row1 >= this->latLonHeader.nrRows)))
    {
        return false;
    }

    // fix extent
    row0 = std::min(this->latLonHeader.nrRows-1, std::max(0, row0));
    row1 = std::min(this->latLonHeader.nrRows-1, std::max(0, row1));
    col0 = std::min(this->latLonHeader.nrCols-1, std::max(0, col0));
    col1 = std::min(this->latLonHeader.nrCols-1, std::max(0, col1));

    *window = gis::Crit3DRasterWindow(row0, col0, row1, col1);

    return true;
}


int RasterObject::getCurrentStep(const gis::Crit3DRasterWindow& window)
{
    // boundary pixels position
    QPointF lowerLeft, topRight, pixelLL, pixelRT;
    lowerLeft.setX(latLonHeader.llCorner.longitude + window.v[0].col * latLonHeader.dx);
    lowerLeft.setY(latLonHeader.llCorner.latitude + (latLonHeader.nrRows-1 - window.v[1].row) * latLonHeader.dy);
    topRight.setX(latLonHeader.llCorner.longitude + (window.v[1].col+1) * latLonHeader.dx);
    topRight.setY(latLonHeader.llCorner.latitude + (latLonHeader.nrRows - window.v[0].row) * latLonHeader.dy);
    pixelLL = getPixel(lowerLeft);
    pixelRT = getPixel(topRight);

    // compute step
    double dx = (pixelRT.x() - pixelLL.x() + 1) / double(window.nrCols());
    double dy = (pixelRT.y() - pixelLL.y() + 1) / double(window.nrRows());

    int step = int(round(2.0 / std::min(dx, dy)));
    return std::max(1, step);
}


float RasterObject::getValue(Position& myPos)
{
    gis::Crit3DGeoPoint geoPoint;
    geoPoint.latitude = myPos.latitude();
    geoPoint.longitude = myPos.longitude();
    return getValue(geoPoint);
}


float RasterObject::getValue(gis::Crit3DGeoPoint& geoPoint)
{
    if (rasterPointer == nullptr) return NODATA;
    if (! rasterPointer->isLoaded) return NODATA;

    float value = NODATA;
    if (isLatLon)
    {
        int row, col;
        gis::getRowColFromLatLon(latLonHeader, geoPoint, &row, &col);
        value = rasterPointer->getValueFromRowCol(row, col);

    }
    else
    {
        gis::Crit3DUtmPoint utmPoint;
        gis::getUtmFromLatLon(utmZone, geoPoint, &utmPoint);

        value = gis::getValueFromUTMPoint(*rasterPointer, utmPoint);
    }

    if (isEqual(value, rasterPointer->header->flag))
        return NODATA;
    else
        return value;
}


bool RasterObject::drawRaster(gis::Crit3DRasterGrid *myRaster, QPainter* myPainter)
{
    if (myRaster == nullptr) return false;
    if (! myRaster->isLoaded) return false;

    gis::Crit3DRasterWindow window;
    if (! getCurrentWindow(&window))
    {
        myRaster->minimum = NODATA;
        myRaster->maximum = NODATA;
        return false;
    }

    // dynamic color scale
    if (! myRaster->colorScale->isRangeBlocked())
    {
        if (this->isLatLon)
        {
            // lat lon raster
            gis::updateColorScale(myRaster, window);
        }
        else
        {
            // UTM raster
            gis::Crit3DRasterWindow* utmWindow = new gis::Crit3DRasterWindow();
            gis::getUtmWindow(this->latLonHeader, *(myRaster->header), window, utmWindow, this->utmZone);
            gis::updateColorScale(myRaster, *utmWindow);
        }
        roundColorScale(myRaster->colorScale, 4, true);
    }

    int step = getCurrentStep(window);

    QPointF lowerLeft;
    lowerLeft.setX(latLonHeader.llCorner.longitude + window.v[0].col * latLonHeader.dx);
    lowerLeft.setY(latLonHeader.llCorner.latitude + (latLonHeader.nrRows-1 - window.v[1].row) * latLonHeader.dy);

    // draw
    int x0, y0, x1, y1, lx, ly;
    float value;
    QPointF p0, p1, pixel;
    Crit3DColor* myColor;
    QColor myQColor;

    for (int row = window.v[1].row; row >= window.v[0].row; row -= step)
    {
        p0.setY(lowerLeft.y() + (window.v[1].row - row) * latLonHeader.dy);
        p1.setY(p0.y() + step * latLonHeader.dy);

        for (int col = window.v[0].col; col <= window.v[1].col; col += step)
        {
            p0.setX(lowerLeft.x() + (col - window.v[0].col) * latLonHeader.dx);
            p1.setX(p0.x() + step * latLonHeader.dx);
            if (p0.x() > 180)
            {
                p0.setX(p0.x() - 360);
                p1.setX(p1.x() - 360);
            }

            pixel = getPixel(p0);
            x0 = int(pixel.x());
            y0 = int(pixel.y());

            pixel = getPixel(p1);
            y1 = int(pixel.y());
            x1 = int(pixel.x());

            lx = x1 - x0;
            ly = y1 - y0;

            if (isLatLon)
                value = myRaster->value[row][col];
            else
            {
                value = myRaster->header->flag;
                int r = matrix[row][col].row;
                if (r != int(NODATA))
                {
                    int c = matrix[row][col].col;
                    if (! gis::isOutOfGridRowCol(r, c, *(myRaster)))
                        value = myRaster->value[r][c];
                }
            }

            if (this->isGrid && isDrawBorder && ! isEqual(value, NO_ACTIVE))
            {
                myPainter->setPen(QColor(64, 64, 64));
                myPainter->setBrush(Qt::NoBrush);
                myPainter->drawRect(x0, y0, lx, ly);
            }
            else if (! isEqual(value, myRaster->header->flag) && ! isEqual(value, NODATA) && ! isEqual(value, NO_ACTIVE))
            {
                myColor = myRaster->colorScale->getColor(value);
                myQColor = QColor(myColor->red, myColor->green, myColor->blue);
                myPainter->setBrush(myQColor);
                myPainter->fillRect(x0, y0, lx, ly, myPainter->brush());
            }
        }
    }

    return true;
}


void RasterObject::updateCenter()
{
    if (! isDrawing) return;

    QPointF newCenter;
    int widthPixels = view->width() - MAPBORDER*2;
    int heightPixels = view->height() - MAPBORDER*2;
    newCenter = view->mapToScene(QPoint(widthPixels/2, heightPixels/2));

    // reference point
    geoMap->referencePoint.longitude = newCenter.x();
    geoMap->referencePoint.latitude = newCenter.y();
    refCenterPixel = view->tileSource()->ll2qgs(newCenter, view->zoomLevel());

    if (isDrawing)
    {
        setPos(newCenter);
    }
}


Position RasterObject::getCurrentCenter()
{
    Position center;
    center.setLongitude(geoMap->referencePoint.longitude);
    center.setLatitude(geoMap->referencePoint.latitude);

    return center;
}


void RasterObject::setMapExtents()
{
    int widthPixels = view->width() - MAPBORDER*2;
    int heightPixels = view->height() - MAPBORDER*2;
    QPointF botLeft = view->mapToScene(QPoint(0, heightPixels));
    QPointF topRight = view->mapToScene(QPoint(widthPixels, 0));

    geoMap->bottomLeft.longitude = MAXVALUE(-180, botLeft.x());
    geoMap->bottomLeft.latitude = MAXVALUE(-84, botLeft.y());
    geoMap->topRight.longitude = MINVALUE(180, topRight.x());
    geoMap->topRight.latitude = MINVALUE(84, topRight.y());
}


bool RasterObject::getRowCol(gis::Crit3DGeoPoint geoPoint, int* row, int* col)
{
    // only for grid
    if (! (this->isGrid || this->isNetcdf))
        return false;

    gis::getGridRowColFromXY(this->latLonHeader, geoPoint.longitude, geoPoint.latitude, row, col);

    // check out of grid
    if (gis::isOutOfGridRowCol(*row, *col, this->latLonHeader))
    {
        return false;
    }

    // UTM -> transform in real [row, col]
    if (! this->isLatLon)
    {
        RowCol myRowCol = matrix[*row][*col];
        *row = myRowCol.row;
        *col = myRowCol.col;
    }

    return true;
}

