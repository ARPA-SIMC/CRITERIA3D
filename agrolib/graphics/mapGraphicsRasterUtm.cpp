/*!
    \file mapGraphicsRasterUtm.cpp

    \abstract draws a UTM raster in the MapGraphics widget

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
#include "color.h"

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

    _rasterPointer = nullptr;
    _colorLegendPointer = nullptr;
    isLoaded = false;

    _utmZone = NODATA;
    _refCenterPixel = QPointF(NODATA, NODATA);
}


/*!
\brief boundingRect
 If sizeIsZoomInvariant() is true, this should return the size of the
 rectangle you want in PIXELS. If false, this should return the size of the rectangle in METERS. The
 rectangle should be centered at (0,0) regardless.
*/
 QRectF RasterUtmObject::boundingRect() const
{
    int widthPixels = _view->width();
    int heightPixels = _view->height();

    return QRectF(-widthPixels, -heightPixels, widthPixels*2, heightPixels*2);
}


bool RasterUtmObject::initialize(gis::Crit3DRasterGrid* rasterPtr, const gis::Crit3DGisSettings& gisSettings)
{
    if (rasterPtr == nullptr)
        return false;
    if (! rasterPtr->isLoaded)
        return false;

    _utmZone = gisSettings.utmZone;
    _rasterPointer = rasterPtr;

    gis::getGeoExtentsFromUTMHeader(gisSettings, _rasterPointer->header, &_latLonHeader);
    gis::Crit3DRasterHeader utmHeader = *_rasterPointer->header;

    // latlon raster have one extra cell
    gis::Crit3DRasterHeader extHeader = utmHeader;
    extHeader.nrCols++;
    extHeader.nrRows++;

    _latRaster.initializeGrid(extHeader);
    _lonRaster.initializeGrid(extHeader);

    double x, y, lat, lon;
    for (int row = 0; row < extHeader.nrRows; row++)
    {
        for (int col = 0; col < extHeader.nrCols; col++)
        {
            gis::getUtmXYFromRowCol(utmHeader, row, col, &x, &y);

            // move to top left of the cell
            y += utmHeader.cellSize * 0.5;
            x -= utmHeader.cellSize * 0.5;

            gis::getLatLonFromUtm(gisSettings, x, y, &lat, &lon);

            _latRaster.value[row][col] = lat;
            _lonRaster.value[row][col] = lon;
        }
    }

    setDrawing(true);
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
           drawRaster(painter);

       if (_colorLegendPointer != nullptr)
           _colorLegendPointer->update();
   }
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


/*!
 * \brief getValue
 * \return return the raster value at a lat lon position
 */
float RasterUtmObject::getValue(Position& pos)
{
    if (_rasterPointer == nullptr)
        return NODATA;
    if (! _rasterPointer->isLoaded)
        return NODATA;

    gis::Crit3DUtmPoint utmPoint;
    gis::Crit3DGeoPoint geoPoint(pos.latitude(), pos.longitude());
    gis::getUtmFromLatLon(_utmZone, geoPoint, &utmPoint);

    float value = _rasterPointer->getValueFromXY(utmPoint.x, utmPoint.y);

    if (isEqual(value, _rasterPointer->header->flag))
        return NODATA;
    else
        return value;
}


/*!
 * \brief getCurrentCenter
 * \return the current center of mapView (lat, lon)
 */
Position RasterUtmObject::getCurrentCenter()
{
    Position center;
    center.setLatitude(_geoMap->referencePoint.latitude);
    center.setLongitude(_geoMap->referencePoint.longitude);

    return center;
}


/*!
 * \brief getRasterCenter
 * \return the center of the raster (lat, lon)
 */
Position RasterUtmObject::getRasterCenter()
{
    Position center;
    int rowCenter = _latRaster.header->nrRows * 0.5;
    int colCenter = _latRaster.header->nrCols * 0.5;
    center.setLatitude(_latRaster.value[rowCenter][colCenter]);
    center.setLongitude(_lonRaster.value[rowCenter][colCenter]);

    return center;
}


/*!
 * \brief getRasterMaxSize
 * \return the maximum size of the raster in decimal degrees (width or height)
 */
float RasterUtmObject::getRasterMaxSize()
{
    return float(MAXVALUE(_latLonHeader.nrRows * _latLonHeader.dy,
                          _latLonHeader.nrCols * _latLonHeader.dx));
}


/*!
 * \brief getCurrentWindow
 * \return the currently displayed raster window (row, col)
 */
bool RasterUtmObject::getCurrentWindow(gis::Crit3DRasterWindow* rasterWindow)
{
    // check pointer
    if (_rasterPointer == nullptr)
        return false;

    // get current view extent
    gis::Crit3DUtmPoint bottomleft, topRight;
    gis::getUtmFromLatLon(_utmZone, this->_geoMap->bottomLeft, &bottomleft);
    gis::getUtmFromLatLon(_utmZone, this->_geoMap->topRight, &topRight);

    int row0, row1, col0, col1;
    gis::Crit3DRasterHeader utmHeader = *_rasterPointer->header;
    gis::getRowColFromXY(utmHeader, bottomleft, &row0, &col0);
    gis::getRowColFromXY(utmHeader, topRight, &row1, &col1);
    col0 -= 2;
    row1--;

    // check if current window is out of map
    if (((col0 < 0) && (col1 < 0))
    || ((row0 < 0) && (row1 < 0))
    || ((col0 >= utmHeader.nrCols) && (col1 >= utmHeader.nrCols))
    || ((row0 >= utmHeader.nrRows) && (row1 >= utmHeader.nrRows)))
    {
        return false;
    }

    // fix extent
    row0 = std::min(utmHeader.nrRows-1, std::max(0, row0));
    row1 = std::min(utmHeader.nrRows-1, std::max(0, row1));
    col0 = std::min(utmHeader.nrCols-1, std::max(0, col0));
    col1 = std::min(utmHeader.nrCols-1, std::max(0, col1));

    *rasterWindow = gis::Crit3DRasterWindow(row0, col0, row1, col1);

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


int RasterUtmObject::getCurrentStep(const gis::Crit3DRasterWindow& rasterWindow)
{
    // boundary pixels position
    QPointF lowerLeft, topRight, pixelLL, pixelRT;
    lowerLeft.setX(_lonRaster.value[rasterWindow.v[0].row][rasterWindow.v[0].col]);
    lowerLeft.setY(_latRaster.value[rasterWindow.v[0].row][rasterWindow.v[0].col]);
    topRight.setX(_lonRaster.value[rasterWindow.v[1].row][rasterWindow.v[1].col]);
    topRight.setY(_latRaster.value[rasterWindow.v[1].row][rasterWindow.v[1].col]);
    pixelLL = getPixel(lowerLeft);
    pixelRT = getPixel(topRight);

    // compute step
    qreal dx = (pixelRT.x() - pixelLL.x() + 1) / qreal(rasterWindow.nrCols());
    qreal dy = (pixelLL.y() - pixelRT.y() + 1) / qreal(rasterWindow.nrRows());

    int step = int(round(1.0 / std::min(dx, dy)));
    return std::max(step, 1);
}


bool RasterUtmObject::drawRaster(QPainter* painter)
{
    if (_rasterPointer == nullptr) return false;
    if (! _rasterPointer->isLoaded) return false;

    gis::Crit3DRasterWindow rasterWindow;
    if (! getCurrentWindow(&rasterWindow))
    {
        _rasterPointer->minimum = NODATA;
        _rasterPointer->maximum = NODATA;
        return false;
    }

    // dynamic color scale
    if (! _rasterPointer->colorScale->isFixedRange())
    {
        gis::updateColorScale(_rasterPointer, rasterWindow);
        roundColorScale(_rasterPointer->colorScale, 4, true);
    }

    int step = getCurrentStep(rasterWindow);

    // draw
    painter->setPen(Qt::NoPen);
    QPointF geoPoint[4];
    QPointF pixel[4];
    Crit3DColor* myColor;
    QColor myQColor;

    for (int row1 = rasterWindow.v[0].row; row1 <= rasterWindow.v[1].row; row1 += step)
    {
        int row2 = std::min(row1 + step, _rasterPointer->header->nrRows-1);
        int rowCenter = floor((row1 + row2) * 0.5);
        if (row2 == row1)
        {
            row2++;
        }

        for (int col1 = rasterWindow.v[0].col; col1 <= rasterWindow.v[1].col; col1 += step)
        {
            int col2 = std::min(col1 + step, _rasterPointer->header->nrCols-1);
            int colCenter = floor((col1 + col2) * 0.5);
            if (col2 == col1)
            {
                col2++;
            }

            // raster value
            float value = _rasterPointer->value[rowCenter][colCenter];

            // check NODATA value (transparent)
            if (isEqual(value, _rasterPointer->header->flag) || isEqual(value, NODATA))
                continue;

            // check outliers (transparent)
            if (_rasterPointer->colorScale->isHideOutliers())
            {
                if (value <= _rasterPointer->colorScale->minimum() || value > _rasterPointer->colorScale->maximum())
                    continue;
            }

            // set color
            myColor = _rasterPointer->colorScale->getColor(value);
            myQColor = QColor(myColor->red, myColor->green, myColor->blue);
            painter->setBrush(myQColor);

            // set polygon
            geoPoint[0].setX(_lonRaster.value[row1][col1]);
            geoPoint[0].setY(_latRaster.value[row1][col1]);
            pixel[0] = getPixel(geoPoint[0]);

            geoPoint[1].setX(_lonRaster.value[row1][col2]);
            geoPoint[1].setY(_latRaster.value[row1][col2]);
            pixel[1] = getPixel(geoPoint[1]);

            geoPoint[2].setX(_lonRaster.value[row2][col2]);
            geoPoint[2].setY(_latRaster.value[row2][col2]);
            pixel[2] = getPixel(geoPoint[2]);

            geoPoint[3].setX(_lonRaster.value[row2][col1]);
            geoPoint[3].setY(_latRaster.value[row2][col1]);
            pixel[3] = getPixel(geoPoint[3]);

            painter->drawPolygon(pixel, 4);
        }
    }

    return true;
}
