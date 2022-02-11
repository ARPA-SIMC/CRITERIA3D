#ifndef SOLARRADIATION_H
#define SOLARRADIATION_H

    #ifndef RADIATIONSETTINGS_H
        #include "radiationSettings.h"
    #endif

    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif

    class Crit3DRadiationMaps
    {
    private:
        bool isComputed;


    public:
        gis::Crit3DRasterGrid* latMap;
        gis::Crit3DRasterGrid* lonMap;
        gis::Crit3DRasterGrid* slopeMap;
        gis::Crit3DRasterGrid* aspectMap;
        gis::Crit3DRasterGrid* beamRadiationMap;
        gis::Crit3DRasterGrid* diffuseRadiationMap;
        gis::Crit3DRasterGrid* reflectedRadiationMap;
        gis::Crit3DRasterGrid* transmissivityMap;
        gis::Crit3DRasterGrid* globalRadiationMap;

        // for vine3d
        gis::Crit3DRasterGrid* sunElevationMap;

        /*
        gis::Crit3DRasterGrid* linkeMap;
        gis::Crit3DRasterGrid* albedoMap;
        gis::Crit3DRasterGrid* sunAzimuthMap;
        gis::Crit3DRasterGrid* sunIncidenceMap;
        gis::Crit3DRasterGrid* sunShadowMap;
        */

        Crit3DRadiationMaps();
        Crit3DRadiationMaps(const gis::Crit3DRasterGrid& dem, const gis::Crit3DGisSettings& gisSettings);
        ~Crit3DRadiationMaps();

        void clear();
        void initialize();
        bool getComputed();
        void setComputed(bool value);
    };


    namespace radiation
    {

        float readAlbedo(Crit3DRadiationSettings* mySettings, const gis::Crit3DPoint& point);
        float readLinke(Crit3DRadiationSettings* mySettings, const gis::Crit3DPoint& myPoint);

        bool computeSunPosition(float lon, float lat, int myTimezone,
                                int myYear,int myMonth, int myDay,
                                int myHour, int myMinute, int mySecond,
                                float temp, float pressure, float aspect, float slope, TsunPosition *mySunPosition);

        int estimateTransmissivityWindow(Crit3DRadiationSettings* mySettings, const gis::Crit3DRasterGrid& myDEM,
                                         const gis::Crit3DPoint &myPoint, Crit3DTime myTime, int timeStepSecond);

        bool computeRadiationRsun(Crit3DRadiationSettings* mySettings, float myTemperature, float myPressure, Crit3DTime myTime,
                                       float myLinke, float myAlbedo, float myClearSkyTransmissivity, float transmissivity,
                                       TsunPosition* mySunPosition, TradPoint* myPoint, const gis::Crit3DRasterGrid& myDEM);

        bool computeRadiationRSunGridPoint(Crit3DRadiationSettings* mySettings, const gis::Crit3DRasterGrid& myDEM,
                                  Crit3DRadiationMaps* radiationMaps, TradPoint radPoint,
                                  int row, int col, const Crit3DTime& myTime);

        bool computeRadiationOutputPoints(Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& myDEM,
                                             Crit3DRadiationMaps *radiationMaps, std::vector<gis::Crit3DOutputPoint> &outputPoints,
                                             const Crit3DTime& myCrit3DTime);

        bool computeRadiationGrid(Crit3DRadiationSettings *radSettings, const gis::Crit3DRasterGrid& myDEM,
                                 Crit3DRadiationMaps* radiationMaps, const Crit3DTime& myCrit3DTime);

        float computePointTransmissivity(Crit3DRadiationSettings *mySettings, const gis::Crit3DPoint& myPoint, Crit3DTime myTime, float* measuredRad,
                                         int windowWidth, int timeStepSecond, const gis::Crit3DRasterGrid& myDEM);

        bool isGridPointComputable(Crit3DRadiationSettings* mySettings, int row, int col, const gis::Crit3DRasterGrid& myDEM, Crit3DRadiationMaps* radiationMaps);
    }


#endif // SOLARRADIATION_H
