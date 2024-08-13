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
        class Crit3DLatLonHeader;

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

            bool isInsideGrid(const Crit3DLatLonHeader& latLonHeader) const;
        };


        class  Crit3DPoint {
        public:
            Crit3DUtmPoint utm;
            double z;

            Crit3DPoint();
            Crit3DPoint(double utmX, double utmY, double _z);
        };


        class  Crit3DOutputPoint : public gis::Crit3DPoint
        {
        public:
            Crit3DOutputPoint();

            std::string id;
            double latitude;
            double longitude;

            bool active;
            bool selected;

            float currentValue;

            void initialize(const std::string& _id, bool isActive, double _latitude, double _longitude,
                            double _z, int zoneNumber);
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


        class Crit3DLatLonHeader
        {
        public:
            int nrRows;
            int nrCols;
            int nrBytes;
            double dx, dy;
            float flag;
            Crit3DGeoPoint llCorner;

            Crit3DLatLonHeader();
        };


        class Crit3DRasterHeader
        {
        public:
            int nrRows;
            int nrCols;
            int nrBytes;
            double cellSize;
            float flag;
            Crit3DUtmPoint llCorner;

            Crit3DRasterHeader();

            bool isEqualTo(const Crit3DRasterHeader& myHeader);

        };

        bool operator == (const Crit3DRasterHeader& myHeader1, const Crit3DRasterHeader& myHeader2);


        class  Crit3DRasterCell {
        public:
            int row;
            int col;

            Crit3DRasterCell();
        };

        struct RasterGridCell {
        public:
            int row;
            int col;
            std::vector<std::vector<double>> fittingParameters;
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
            std::vector<RasterGridCell> singleCell;

            Crit3DUtmPoint* utmPoint(int myRow, int myCol);
            void getXY(int myRow, int myCol, double &x, double &y) const;
            void getRowCol(double x, double y, int& row, int& col) const;
            Crit3DPoint getCenter();
            Crit3DGeoPoint getCenterLatLon(const Crit3DGisSettings &gisSettings);

            void clear();
            void emptyGrid();

            Crit3DRasterGrid();
            ~Crit3DRasterGrid();

            void setConstantValue(float initValue);

            bool initializeGrid();
            bool initializeGrid(float initValue);
            bool initializeGrid(const Crit3DRasterGrid& initGrid);
            bool initializeGrid(const Crit3DRasterHeader& initHeader);
            bool initializeGrid(const Crit3DLatLonHeader& latLonHeader);
            bool initializeGrid(const Crit3DRasterGrid& initGrid, float initValue);

            bool initializeParameters(const Crit3DRasterHeader &initHeader);
            bool initializeParametersLatLonHeader(const Crit3DLatLonHeader& latLonHeader);

            bool copyGrid(const Crit3DRasterGrid& initGrid);

            bool setConstantValueWithBase(float initValue, const Crit3DRasterGrid& initGrid);

            bool isOutOfGrid(int row, int col) const;
            bool isFlag(int myRow, int myCol) const;
            float getValueFromRowCol(int myRow, int myCol) const;
            float getValueFromXY(double x, double y) const;
            std::vector<std::vector<double>> getParametersFromRowCol(int row, int col);
            bool setParametersForRowCol(int row, int col, std::vector<std::vector<double>> parameters);
            std::vector<std::vector<double>> prepareParameters(int row, int col, std::vector<bool> activeList);

            Crit3DTime getMapTime() const;
            void setMapTime(const Crit3DTime &value);
        };


        class Crit3DEllipsoid
        {
        public:
            double equatorialRadius;
            double eccentricitySquared;

            Crit3DEllipsoid();
        };

        float computeDistance(float x1, float y1, float x2, float y2);
        double computeDistancePoint(Crit3DUtmPoint *p0, Crit3DUtmPoint *p1);
        bool updateMinMaxRasterGrid(Crit3DRasterGrid *rasterGrid);
        void convertFlagToNodata(Crit3DRasterGrid& myGrid);
        bool updateColorScale(Crit3DRasterGrid* rasterGrid, int row0, int col0, int row1, int col1);

        void getRowColFromXY(const Crit3DRasterHeader& myHeader, double myX, double myY, int *row, int *col);
        void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, int *row, int *col);
        void getRowColFromXY(const Crit3DRasterHeader& myHeader, const Crit3DUtmPoint& p, Crit3DRasterCell* v);
        void getRowColFromLonLat(const Crit3DLatLonHeader& myHeader, double lon, double lat, int *row, int *col);

        void getRowColFromLatLon(const Crit3DLatLonHeader &latLonHeader, const Crit3DGeoPoint& p, int *myRow, int *myCol);
        bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DRasterGrid &rasterGrid);

        void getUtmXYFromRowColSinglePrecision(const Crit3DRasterGrid& rasterGrid, int myRow, int myCol,float* myX,float* myY);
        void getUtmXYFromRowColSinglePrecision(const Crit3DRasterHeader& myHeader, int myRow, int myCol,float* myX,float* myY);
        void getUtmXYFromRowCol(const Crit3DRasterHeader& myHeader,int myRow, int myCol, double* myX, double* myY);
        void getUtmXYFromRowCol(Crit3DRasterHeader *myHeader, int row, int col, double* myX, double* myY);

        void getLatLonFromRowCol(const Crit3DLatLonHeader &latLonHeader, int myRow, int myCol, double* lat, double* lon);
        void getLatLonFromRowCol(const Crit3DLatLonHeader &latLonHeader, const Crit3DRasterCell& v, Crit3DGeoPoint* p);
        float getValueFromXY(const Crit3DRasterGrid& rasterGrid, double x, double y);
        float getValueFromUTMPoint(const Crit3DRasterGrid& rasterGrid, Crit3DUtmPoint& utmPoint);

        bool isOutOfGridXY(double x, double y, Crit3DRasterHeader* header);
        bool isOutOfGridRowCol(int myRow, int myCol, const Crit3DLatLonHeader& header);

        bool isMinimum(const Crit3DRasterGrid& rasterGrid, int row, int col);
        bool isMinimumOrNearMinimum(const Crit3DRasterGrid& rasterGrid, int row, int col);
        bool isBoundary(const Crit3DRasterGrid& rasterGrid, int row, int col);
        bool isBoundaryRunoff(const Crit3DRasterGrid& rasterRef, const Crit3DRasterGrid &aspectMap, int row, int col);
        bool isStrictMaximum(const Crit3DRasterGrid& rasterGrid, int row, int col);

        bool getNorthernEmisphere();
        void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, double utmX,double utmY, double *myLat, double *myLon);
        void getLatLonFromUtm(const Crit3DGisSettings& gisSettings, const Crit3DUtmPoint& utmPoint, Crit3DGeoPoint& geoPoint);

        void getUtmFromLatLon(int zoneNumber, const Crit3DGeoPoint& geoPoint, Crit3DUtmPoint* utmPoint);
        void getUtmFromLatLon(const Crit3DGisSettings& gisSettings, double latitude, double longitude, double *utmX, double *utmY);

        void latLonToUtm(double lat, double lon,double *utmEasting,double *utmNorthing,int *zoneNumber);
        void latLonToUtmForceZone(int zoneNumber, double lat, double lon, double *utmEasting, double *utmNorthing);
        void utmToLatLon(int zoneNumber, double referenceLat, double utmEasting, double utmNorthing, double *lat, double *lon);
        bool isValidUtmTimeZone(int utmZone, int timeZone);

        bool openRaster(std::string fileName, Crit3DRasterGrid *rasterGrid, int currentUtmZone, std::string &errorStr);

        bool readEsriGrid(const std::string &fileName, Crit3DRasterGrid* rasterGrid, std::string &errorStr);
        bool writeEsriGrid(const std::string &fileName, Crit3DRasterGrid *rasterGrid, std::string &errorStr);

        bool readEnviGrid(std::string fileName, Crit3DRasterGrid* rasterGrid, int currentUtmZone, std::string &errorStr);
        bool writeEnviGrid(std::string fileName, int utmZone, Crit3DRasterGrid *rasterGrid, std::string &errorStr);

        bool mapAlgebra(Crit3DRasterGrid* myMap1, Crit3DRasterGrid* myMap2, Crit3DRasterGrid *outputMap, operationType myOperation);
        bool mapAlgebra(Crit3DRasterGrid* myMap1, float myValue, Crit3DRasterGrid *outputMap, operationType myOperation);

        bool prevailingMap(const Crit3DRasterGrid& inputMap,  Crit3DRasterGrid *outputMap);
        float prevailingValue(const std::vector<float> &valueList);

        bool clipRasterWithRaster(gis::Crit3DRasterGrid* refRaster, gis::Crit3DRasterGrid* maskRaster,
                                  gis::Crit3DRasterGrid* outputRaster);

        bool computeLatLonMaps(const gis::Crit3DRasterGrid& rasterGrid,
                               gis::Crit3DRasterGrid* latMap, gis::Crit3DRasterGrid* lonMap,
                               const gis::Crit3DGisSettings& gisSettings);

        bool computeSlopeAspectMaps(const gis::Crit3DRasterGrid& dem,
                               gis::Crit3DRasterGrid* slopeMap, gis::Crit3DRasterGrid* aspectMap);

        bool getGeoExtentsFromUTMHeader(const Crit3DGisSettings& mySettings,
                                        Crit3DRasterHeader *utmHeader, Crit3DLatLonHeader *latLonHeader);
        bool getGeoExtentsFromLatLonHeader(const Crit3DGisSettings& mySettings, double cellSize, Crit3DRasterHeader *utmHeader, Crit3DLatLonHeader *latLonHeader);
        double getGeoCellSizeFromLatLonHeader(const Crit3DGisSettings& mySettings, Crit3DLatLonHeader *latLonHeader);

        float topographicDistance(float x1, float y1, float z1, float x2, float y2, float z2, float distance,
                                  const gis::Crit3DRasterGrid& dem);
        bool topographicDistanceMap(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& dem, Crit3DRasterGrid* myMap);
        float closestDistanceFromGrid(Crit3DPoint myPoint, const gis::Crit3DRasterGrid& dem);
        bool compareGrids(const gis::Crit3DRasterGrid& first, const gis::Crit3DRasterGrid& second);
        void resampleGrid(const gis::Crit3DRasterGrid& oldGrid, gis::Crit3DRasterGrid* newGrid,
                          Crit3DRasterHeader* newHeader, aggregationMethod elab, float nodataRatioThreshold);
        bool temporalYearlyInterpolation(const gis::Crit3DRasterGrid& firstGrid, const gis::Crit3DRasterGrid& secondGrid,
                                         int myYear, float minValue, float maxValue, gis::Crit3DRasterGrid* outGrid);
    }


#endif // GIS_H
