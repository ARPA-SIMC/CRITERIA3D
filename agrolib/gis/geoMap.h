#ifndef GEOMAP_H
#define GEOMAP_H

    #ifndef GIS_H
        #include "gis.h"
    #endif

    namespace gis
    {
        class  Crit3DRasterWindow {
        public:
            Crit3DRasterCell v[2];

            Crit3DRasterWindow();
            Crit3DRasterWindow(int row0, int col0, int row1, int col1);

            int nrRows() const;
            int nrCols() const;
        };

        class Crit3DUtmWindow
        {
        public:
            Crit3DUtmPoint v0;
            Crit3DUtmPoint v1;

            Crit3DUtmWindow();
            Crit3DUtmWindow(const Crit3DUtmPoint& v0, const Crit3DUtmPoint& v1);
            double width();
            double height();
        };

        class Crit3DPixelWindow
        {
        public:
            Crit3DPixel v0;
            Crit3DPixel v1;

            Crit3DPixelWindow();
            Crit3DPixelWindow(const Crit3DPixel& v0, const Crit3DPixel& v1);
            int width();
            int height();
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

            Crit3DGeoMap();
        };


        bool updateColorScale(Crit3DRasterGrid* myGrid, const Crit3DRasterWindow& myWindow);
        bool getUtmWindow(const Crit3DLatLonHeader &latLonHeader, const Crit3DRasterHeader &utmHeader,
                          const Crit3DRasterWindow &latLonWindow, Crit3DRasterWindow *utmWindow, int utmZone);

    }


#endif // GEOMAP_H
