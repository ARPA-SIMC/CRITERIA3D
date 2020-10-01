#ifndef GIS_H
#define GIS_H

    #ifndef VECTOR_H
        #include <vector>
    #endif
    #ifndef _STRING_
        #include <string>
    #endif
    #ifndef COLOR_H
        #include "color.h"
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef STATISTICS_H
        #include "statistics.h"
    #endif

    enum operationType {operationMin, operationMax, operationSum, operationSubtract, operationProduct, operationDivide};

    namespace gis
    {
        class  Crit3DPixel {
        public:
            int x;
            int y;
            Crit3DPixel();
            Crit3DPixel(int _x, int _y);
        };

        class Crit3DRasterHeader;
        class Crit3DGridHeader;

        class  Crit3DUtmPoint {
        public:
            double x;
            double y;

            Crit3DUtmPoint();
            Crit3DUtmPoint(double x, double y);

            void initialize();
            bool isInsideGrid(const Crit3DRasterHeader& myGridHeader) const;
        };

        class  Crit3DGeoPoint {
        public:
            double latitude;
            double longitude;

            Crit3DGeoPoint();
            Crit3DGeoPoint(double lat, double lon);

            bool isInsideGrid(const Crit3DGridHeader& latLonHeader) const;
        };


        class  Crit3DPoint {
        public:
            Crit3DUtmPoint utm;
            double z;

            Crit3DPoint();
            Crit3DPoint(double, double, double);
        };


        class Crit3DGridHeader
        {
        public:
            int nrRows;
            int nrCols;
            double dx, dy;
            float flag;
            Crit3DGeoPoint llCorner;

            Crit3DGridHeader();
        };

        class Crit3DRasterHeader
        {
        public:
            int nrRows;
            int nrCols;
            double cellSize;
            float flag;
            Crit3DUtmPoint llCorner;

            Crit3DRasterHeader();

            void convertFromLatLon(const Crit3DGridHeader& latLonHeader);
            bool isEqualTo(const Crit3DRasterHeader& myHeader);

            friend bool operator == (const Crit3DRasterHeader& myHeader1, const Crit3DRasterHeader& myHeader2);
        };


        class  Crit3DRasterCell {
        public:
            int row;
            int col;

            Crit3DRasterCell();
        };


        class Crit3DRasterGrid
        {
        public:
            Crit3DRasterHeader* header;
            Crit3DColorScale* colorScale;
            float** value;
            float minimum, maximum;
            bool isLoaded;
            Crit3DTime mapTime;

            Crit3DUtmPoint* utmPoint(int myRow, int myCol);
            void getXY(int myRow, int myCol, double* x, double* y);

            void clear();
            void emptyGrid();

            Crit3DRasterGrid();
            ~Crit3DRasterGrid();

            void setConstantValue(float initValue);

            bool initializeGrid();
            bool initializeGrid(float initValue);
            bool initializeGrid(const Crit3DRasterGrid& initGrid);
            bool initializeGrid(const Crit3DRasterHeader& initHeader);
            bool initializeGrid(const Crit3DRasterGrid& initGrid, float initValue);

            bool copyGrid(const Crit3DRasterGrid& initGrid);

            bool setConstantValueWithBase(float initValue, const Crit3DRasterGrid& initGrid);
            float getValueFromRowCol(int myRow, int myCol) const;
            float getFastValueXY(float x, float y) const;
            bool isFlag(int myRow, int myCol);

            Crit3DPoint mapCenter();
            Crit3DTime getMapTime() const;
            void setMapTime(const Crit3DTime &value);
        };

        class Crit3DGisSettings
        {
        public:
            Crit3DGeoPoint startLocation;
            int utmZone;
            bool isUTC;
            int timeZone;

            Crit3DGisSettings();
            void initialize();
        };

        class Crit3DEllipsoid
        {
        public:
            double equatorialRadius;
            double eccentricitySquared;

            Crit3DEllipsoid();
        };

        float computeDistance(float x1, float y1, float x2, float y2);
        double computeDistancePoint(Crit3DUtmPoint* p0, Crit3DUtmPoint *p1);
        bool updateMinMaxRasterGrid(Crit3DRasterGrid* myGrid);
        bool updateColorScale(Crit3DRasterGrid* myGrid, int row0, int col0, int row1, int col1);

        void getRowColFromXY(const Crit3DRasterGrid &myGrid, double myX, double myY, int* row, int* col);
        void getRowColFromXY(const Crit3DRasterHeader& myHeader, double myX, double myY, int *row, int *col);
        void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, int *row, int *col);
        void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, Crit3DRasterCell* v);
        void getMeteoGridRowColFromXY(const Crit3DGridHeader& myHeader, double myX, double myY, int *row, int *col);

        void getRowColFromLatLon(const Crit3DGridHeader &latLonHeader, const Crit3DGeoPoint& p, int *myRow, int *myCol);
        bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DRasterGrid &myGrid);

        void getUtmXYFromRowColSinglePrecision(const Crit3DRasterGrid& myGrid, int myRow, int myCol,float* myX,float* myY);
        void getUtmXYFromRowColSinglePrecision(const Crit3DRasterHeader& myHeader, int myRow, int myCol,float* myX,float* myY);
        void getUtmXYFromRowCol(const Crit3DRasterGrid& myGrid, int myRow, int myCol ,double* myX, double* myY);
        void getUtmXYFromRowCol(const Crit3DRasterHeader& myHeader,int myRow, int myCol, double* myX, double* myY);

        void getLatLonFromRowCol(const Crit3DGridHeader &latLonHeader, int myRow, int myCol, double* lat, double* lon);
        void getLatLonFromRowCol(const Crit3DGridHeader &latLonHeader, const Crit3DRasterCell& v, Crit3DGeoPoint* p);
        float getValueFromXY(const Crit3DRasterGrid& myGrid, double x, double y);

        bool isOutOfGridXY(double x, double y, Crit3DRasterHeader* header);
        bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DGridHeader& header);

        bool isMinimum(const Crit3DRasterGrid& myGrid, int row, int col);
        bool isMinimumOrNearMinimum(const Crit3DRasterGrid& myGrid, int row, int col);
        bool isBoundary(const Crit3DRasterGrid& myGrid, int row, int col);
        bool isStrictMaximum(const Crit3DRasterGrid& myGrid, int row, int col);

        bool getNorthernEmisphere();
        void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, double utmX,double utmY, double *myLat, double *myLon);
        void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, const Crit3DUtmPoint& utmPoint, Crit3DGeoPoint *geoPoint);
        void getUtmFromLatLon(int zoneNumber, const Crit3DGeoPoint& geoPoint, Crit3DUtmPoint* utmPoint);

        void latLonToUtm(double lat, double lon,double *utmEasting,double *utmNorthing,int *zoneNumber);
        void latLonToUtmForceZone(int zoneNumber, double lat, double lon, double *utmEasting, double *utmNorthing);
        void utmToLatLon(int zoneNumber, double referenceLat, double utmEasting, double utmNorthing, double *lat, double *lon);
        bool isValidUtmTimeZone(int utmZone, int timeZone);

        bool readEsriGrid(std::string myFileName, Crit3DRasterGrid* myGrid, std::string* myError);
        bool writeEsriGrid(std::string myFileName, Crit3DRasterGrid* myGrid, std::string* myError);

        bool mapAlgebra(Crit3DRasterGrid* myMap1, Crit3DRasterGrid* myMap2, Crit3DRasterGrid *myMapOut, operationType myOperation);
        bool mapAlgebra(Crit3DRasterGrid* myMap1, float myValue, Crit3DRasterGrid *myMapOut, operationType myOperation);
        bool prevailingMap(const Crit3DRasterGrid& inputMap,  Crit3DRasterGrid *outputMap);
        float prevailingValue(const std::vector<float> &valueList);

        bool computeLatLonMaps(const gis::Crit3DRasterGrid& myGrid,
                               gis::Crit3DRasterGrid* latMap, gis::Crit3DRasterGrid* lonMap,
                               const gis::Crit3DGisSettings& gisSettings);

        bool computeSlopeAspectMaps(const gis::Crit3DRasterGrid& myDEM,
                               gis::Crit3DRasterGrid* slopeMap, gis::Crit3DRasterGrid* aspectMap);

        bool getGeoExtentsFromUTMHeader(const Crit3DGisSettings& mySettings,
                                        Crit3DRasterHeader *utmHeader, Crit3DGridHeader *latLonHeader);

        float topographicDistance(float X1, float Y1, float Z1, float X2, float Y2, float Z2, float distance,
                                  const gis::Crit3DRasterGrid& myDEM);
        bool topographicDistanceMap(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& myDEM, Crit3DRasterGrid* myMap);
        bool compareGrids(const gis::Crit3DRasterGrid& first, const gis::Crit3DRasterGrid& second);
        void resampleGrid(const gis::Crit3DRasterGrid& oldGrid, gis::Crit3DRasterGrid* newGrid,
                          const Crit3DRasterHeader &header, aggregationMethod elab, float nodataThreshold);
        bool temporalYearlyInterpolation(const gis::Crit3DRasterGrid& firstGrid, const gis::Crit3DRasterGrid& secondGrid,
                                         int myYear, float minValue, float maxValue, gis::Crit3DRasterGrid* outGrid);
    }


#endif // GIS_H
