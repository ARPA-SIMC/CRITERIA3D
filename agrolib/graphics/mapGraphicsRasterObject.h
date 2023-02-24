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


#ifndef MAPGRAPHICSRASTEROBJECT_H
#define MAPGRAPHICSRASTEROBJECT_H

    #include "MapGraphics_global.h"
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

    struct RowCol
    {
        int row;
        int col;
    };


    /*!
     * \brief The RasterObject class
     */
    class RasterObject : public MapGraphicsObject
    {
        Q_OBJECT
    public:
        /*!
         * \brief RasterObject constructor
         * \param view a MapGraphicsView pointer
         * \param parent MapGraphicsObject
         */
        explicit RasterObject(MapGraphicsView* view, MapGraphicsObject *parent = nullptr);

        bool isLoaded;

        void clear();
        void setDrawing(bool value);
        void setDrawBorders(bool value);
        void setColorLegend(ColorLegend* myLegend);
        void setNetCDF(bool value);
        bool isNetCDF();

        QPointF getPixel(const QPointF &latLonPoint);
        QPointF getLatLon(const QPointF &pos);

        bool initializeUTM(gis::Crit3DRasterGrid* myRaster, const gis::Crit3DGisSettings& gisSettings, bool isGrid_);
        bool initializeLatLon(gis::Crit3DRasterGrid* myRaster, const gis::Crit3DGisSettings& gisSettings,
                              const gis::Crit3DGridHeader& latLonHeader, bool isGrid_);
        float getRasterMaxSize();
        gis::Crit3DGeoPoint* getRasterCenter();
        void setRaster(gis::Crit3DRasterGrid* rasterPtr);
        gis::Crit3DRasterGrid* getRaster();
        void updateCenter();
        Position getCurrentCenter();

        gis::Crit3DGridHeader getLatLonHeader() const;
        bool getRowCol(gis::Crit3DGeoPoint geoPoint, int* row, int* col);
        float getValue(gis::Crit3DGeoPoint& geoPoint);
        float getValue(Position& myPos);

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
        MapGraphicsView* view;
        gis::Crit3DRasterGrid* rasterPointer;
        gis::Crit3DGeoMap* geoMap;
        QPointF referencePixel;
        ColorLegend* colorScaleLegend;
        bool isDrawBorder;
        RowCol **matrix;
        gis::Crit3DGridHeader latLonHeader;
        double longitudeShift;

        bool isLatLon;
        bool isDrawing;
        bool isGrid;
        bool isNetcdf;
        int utmZone;

        void freeIndexesMatrix();
        void initializeIndexesMatrix();
        void setMapExtents();
        bool getCurrentWindow(gis::Crit3DRasterWindow* window);
        int getCurrentStep(const gis::Crit3DRasterWindow& window);
        bool drawRaster(gis::Crit3DRasterGrid *myRaster, QPainter* myPainter);

    };


#endif // MAPGRAPHICSRASTEROBJECT_H
