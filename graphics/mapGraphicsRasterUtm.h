/*!
    \file mapGraphicsRasterObject.h

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


#ifndef MAPGRAPHICSRASTERUTM_H
#define MAPGRAPHICSRASTERUTM_H

    #include "MapGraphicsObject.h"
    #include "MapGraphicsView.h"

    #ifndef POSITION_H
        #include "Position.h"
    #endif

    #ifndef COLORLEGEND_H
        #include "colorLegend.h"
    #endif

    #ifndef GEOMAP_H
        #include "geoMap.h"
    #endif

    #include <vector>


    /*!
     * \brief The RasterObject class
     */
    class RasterUtmObject : public MapGraphicsObject
    {
        Q_OBJECT
    public:
        /*!
         * \brief RasterUtmObject constructor
         * \param view a MapGraphicsView pointer
         * \param parent MapGraphicsObject
         */
        explicit RasterUtmObject(MapGraphicsView* view, MapGraphicsObject *parent = nullptr);

        bool isLoaded;

        void clear();
        bool initialize(gis::Crit3DRasterGrid* rasterPtr, const gis::Crit3DGisSettings& gisSettings);

        void setDrawing(bool value) {_isDrawing = value;}
        void setColorLegend(ColorLegend* colorLegendPtr) {_colorLegendPointer = colorLegendPtr;}
        void setRaster(gis::Crit3DRasterGrid* rasterPtr) {_rasterPointer = rasterPtr;}

        gis::Crit3DRasterGrid* getRasterPointer() {return _rasterPointer;}

        float getValue(Position& pos);
        float getRasterMaxSize();
        Position getCurrentCenter();
        Position getRasterCenter();
        QPointF getPixel(const QPointF &geoPoint);
        gis::Crit3DLatLonHeader getLatLonHeader() {return _latLonHeader;}

        void updateCenter();

    protected:
        //virtual from MapGraphicsObject
        /*!
         * \brief paint pure-virtual from MapGraphicsObject
         * \param painter a QPainter pointer
         * \param option a QStyleOptionGraphicsItem pointer
         * \param widget a QWidget pointer
         */
        void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

        /*!
         * \brief boundingRect pure-virtual from MapGraphicsObject
         * Defines the outer bounds of the item as a rectangle; all painting must be restricted to inside an item's bounding rect.
         * \return the bounding rect QRectF
         */
        QRectF boundingRect() const;

    private:
        MapGraphicsView* _view;
        gis::Crit3DRasterGrid* _rasterPointer;
        gis::Crit3DGeoMap* _geoMap;
        ColorLegend* _colorLegendPointer;

        gis::Crit3DRasterGrid _latRaster;
        gis::Crit3DRasterGrid _lonRaster;
        gis::Crit3DLatLonHeader _latLonHeader;

        QPointF _refCenterPixel;

        bool _isDrawing;
        int _utmZone;

        void setMapExtents();
        bool getCurrentWindow(gis::Crit3DRasterWindow* rasterWindow);
        int getCurrentStep(const gis::Crit3DRasterWindow& rasterWindow);
        bool drawRaster(QPainter* painter);

    };


#endif // MAPGRAPHICSRASTERUTM_H
