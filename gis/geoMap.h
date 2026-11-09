#ifndef GEOMAP_H
#define GEOMAP_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <cmath>

    namespace gis
    {
        class  Crit3DRasterWindow {
        public:
            Crit3DRasterCell v[2];

            Crit3DRasterWindow() {}
            Crit3DRasterWindow(int row0, int col0, int row1, int col1);

            int nrRows() const
            { return (v[1].row - v[0].row + 1); }

            int nrCols() const
            { return (v[1].col - v[0].col + 1); }
        };

        class Crit3DUtmWindow
        {
        public:
            Crit3DUtmPoint v0;
            Crit3DUtmPoint v1;

            Crit3DUtmWindow() {}
            Crit3DUtmWindow(const Crit3DUtmPoint& v0_, const Crit3DUtmPoint& v1_)
                :v0(v0_), v1(v1_) {}

            double width() const
            { return fabs(v1.x - v0.x); }

            double height() const
            { return fabs(v1.y - v0.y); }
        };

        class Crit3DPixelWindow
        {
        public:
            Crit3DPixel v0;
            Crit3DPixel v1;

            Crit3DPixelWindow() {}
            Crit3DPixelWindow(const Crit3DPixel& v0_, const Crit3DPixel& v1_)
                :v0(v0_), v1(v1_) {}

            int width() const
            { return abs(v1.x - v0.x); }

            int height() const
            { return abs(v1.y - v0.y); }
        };

        class Crit3DGeoMap
        {
        public:
            bool isDrawing;
            bool isChanged;
            bool isSelecting;

            Crit3DGeoPoint referencePoint;
            Crit3DGeoPoint bottomLeft;
            Crit3DGeoPoint topRight;

            Crit3DGeoMap() {
                isDrawing = false;
                isChanged = false;
                isSelecting = false;
            }
        };


        bool updateColorScale(Crit3DRasterGrid* myGrid, const Crit3DRasterWindow& myWindow);
        bool getUtmWindow(const Crit3DLatLonHeader& latLonHeader, const Crit3DRasterHeader& utmHeader,
                          const Crit3DRasterWindow& latLonWindow, Crit3DRasterWindow* utmWindow, int utmZone);

    }


#endif // GEOMAP_H
