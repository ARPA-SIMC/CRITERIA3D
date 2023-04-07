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
#include "mapGraphicsRasterUtm.h"
#include "basicMath.h"

#include <math.h>
#include <QMenu>


RasterUtmObject::RasterUtmObject(MapGraphicsView* view, MapGraphicsObject *parent) :
    MapGraphicsObject(true, parent)
{
    setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    setFlag(MapGraphicsObject::ObjectIsMovable, false);
    setFlag(MapGraphicsObject::ObjectIsFocusable, false);

    _view = view;
    _geoMap = new gis::Crit3DGeoMap();
    this->clear();
}


void RasterUtmObject::clear()
{
    setDrawing(false);
    setDrawBorders(false);
    setIsNetCDF(false);
    setIsGrid(false);

    _rasterPointer = nullptr;
    _colorLegendPointer = nullptr;

    _utmZone = NODATA;
    _refCenterPixel = QPointF(NODATA, NODATA);

    isLoaded = false;
}


/*!
\brief If sizeIsZoomInvariant() is true, this should return the size of the
 rectangle you want in PIXELS. If false, this should return the size of the rectangle in METERS. The
 rectangle should be centered at (0,0) regardless.
*/
 QRectF RasterUtmObject::boundingRect() const
{
    int widthPixels = _view->width();
    int heightPixels = _view->height();

    return QRectF(-widthPixels, -heightPixels, widthPixels*2, heightPixels*2);
}


/*!
\brief convert a point in geo (lat,lon) coordinates
 in pixel (local object) coordinates
*/
QPointF RasterUtmObject::getPixel(const QPointF &geoPoint)
{
    QPointF mapPixel = _view->tileSource()->ll2qgs(geoPoint, _view->zoomLevel());
    QPointF pixel;
    pixel.setX(mapPixel.x() - _refCenterPixel.x());
    pixel.setY(_refCenterPixel.y() - mapPixel.y());
    return pixel;
}


float RasterUtmObject::getValue(Position& pos)
{
    gis::Crit3DGeoPoint geoPoint(pos.latitude(), pos.longitude());
    return getValue(geoPoint);
}


float RasterUtmObject::getValue(gis::Crit3DGeoPoint& geoPoint)
{
    if (_rasterPointer == nullptr)
        return NODATA;
    if (! _rasterPointer->isLoaded)
        return NODATA;

    gis::Crit3DUtmPoint utmPoint;
    gis::getUtmFromLatLon(_utmZone, geoPoint, &utmPoint);

    float value = _rasterPointer->getValueFromXY(utmPoint.x, utmPoint.y);

    if (isEqual(value, _rasterPointer->header->flag))
        return NODATA;
    else
        return value;
}


Position RasterUtmObject::getCurrentCenterGeo()
{
    Position center;
    center.setLongitude(_geoMap->referencePoint.longitude);
    center.setLatitude(_geoMap->referencePoint.latitude);

    return center;
}


bool RasterUtmObject::initialize(gis::Crit3DRasterGrid* rasterPtr, const gis::Crit3DGisSettings& gisSettings, bool isGrid)
{
    if (rasterPtr == nullptr)
        return false;
    if (! rasterPtr->isLoaded)
        return false;

    setIsGrid(isGrid);
    _utmZone = gisSettings.utmZone;
    _rasterPointer = rasterPtr;

    /*
    double lat, lon, x, y;
    int utmRow, utmCol;
    gis::getGeoExtentsFromUTMHeader(gisSettings, myRaster->header, &latLonHeader);

    for (int row = 0; row < latLonHeader.nrRows; row++)
    {
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
    }
    */

    setDrawing(true);
    setDrawBorders(isGrid);
    isLoaded = true;

    return true;
}


void RasterUtmObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

   if (_isDrawing)
   {
       setMapExtents();

       if (_rasterPointer != nullptr)
           drawRaster(_rasterPointer, painter);

       if (_colorLegendPointer != nullptr)
           _colorLegendPointer->update();
   }
}


/*
bool RasterUtmObject::getCurrentWindow(gis::Crit3DRasterWindow* window)
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
*/


/*
int RasterUtmObject::getCurrentStep(const gis::Crit3DRasterWindow& window)
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
*/


bool RasterUtmObject::drawRaster(gis::Crit3DRasterGrid *raster, QPainter* painter)
{
    if (raster == nullptr) return false;
    if (! raster->isLoaded) return false;

    gis::Crit3DRasterWindow window;
    if (! getCurrentWindow(&window))
    {
        raster->minimum = NODATA;
        raster->maximum = NODATA;
        return false;
    }

    // dynamic color scale
    if (! raster->colorScale->isRangeBlocked())
    {
        gis::Crit3DRasterWindow* utmWindow = new gis::Crit3DRasterWindow();
        //gis::getUtmWindow(this->latLonHeader, *(myRaster->header), window, utmWindow, this->utmZone);
        gis::updateColorScale(raster, *utmWindow);
        roundColorScale(raster->colorScale, 4, true);
    }

    /*
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

            value = myRaster->header->flag;
            int r = matrix[row][col].row;
            if (r != int(NODATA))
            {
                int c = matrix[row][col].col;
                if (! gis::isOutOfGridRowCol(r, c, *(myRaster)))
                    value = myRaster->value[r][c];
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
    */

    return true;
}


void RasterUtmObject::updateCenter()
{
    if (! _isDrawing) return;

    QPointF newCenter = _view->mapToScene(QPoint(_view->width() * 0.5, _view->height() * 0.5));

    // reference point
    _geoMap->referencePoint.longitude = newCenter.x();
    _geoMap->referencePoint.latitude = newCenter.y();
    _refCenterPixel = _view->tileSource()->ll2qgs(newCenter, _view->zoomLevel());

    if (_isDrawing)
    {
        setPos(newCenter);
    }
}


// set geo (lat lon) extents
void RasterUtmObject::setMapExtents()
{
    QPointF bottomLeft = _view->mapToScene(QPoint(0, _view->height()));
    QPointF topRight = _view->mapToScene(QPoint(_view->width(), 0));

    _geoMap->bottomLeft.longitude = MAXVALUE(-180, bottomLeft.x());
    _geoMap->bottomLeft.latitude = MAXVALUE(-84, bottomLeft.y());
    _geoMap->topRight.longitude = MINVALUE(180, topRight.x());
    _geoMap->topRight.latitude = MINVALUE(84, topRight.y());
}

