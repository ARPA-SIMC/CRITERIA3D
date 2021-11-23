#ifndef METEOMAPS_H
#define METEOMAPS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif

    #ifndef METEO_H
        #include "meteo.h"
    #endif

    #ifndef SOLARRADIATION_H
        #include "solarRadiation.h"
    #endif

    class Crit3DDailyMeteoMaps
    {
    private:

    public:
        gis::Crit3DRasterGrid* mapDailyTAvg;
        gis::Crit3DRasterGrid* mapDailyTMin;
        gis::Crit3DRasterGrid* mapDailyTMax;
        gis::Crit3DRasterGrid* mapDailyPrec;
        gis::Crit3DRasterGrid* mapDailyRHAvg;
        gis::Crit3DRasterGrid* mapDailyRHMin;
        gis::Crit3DRasterGrid* mapDailyRHMax;
        gis::Crit3DRasterGrid* mapDailyET0HS;
        gis::Crit3DRasterGrid* mapDailyLeafW;

        Crit3DDailyMeteoMaps(const gis::Crit3DRasterGrid& DEM);
        ~Crit3DDailyMeteoMaps();

        void clear();

        gis::Crit3DRasterGrid* getMapFromVar(meteoVariable myVar);
        bool computeHSET0Map(gis::Crit3DGisSettings *gisSettings, Crit3DDate myDate);
        bool fixDailyThermalConsistency();
    };


    class Crit3DHourlyMeteoMaps
    {
    private:
        bool isComputed;

    public:
        gis::Crit3DRasterGrid* mapHourlyTair;
        gis::Crit3DRasterGrid* mapHourlyTdew;
        gis::Crit3DRasterGrid* mapHourlyPrec;
        gis::Crit3DRasterGrid* mapHourlyRelHum;
        gis::Crit3DRasterGrid* mapHourlyWindScalarInt;
        gis::Crit3DRasterGrid* mapHourlyET0;
        gis::Crit3DRasterGrid* mapHourlyLeafW;

        Crit3DHourlyMeteoMaps(const gis::Crit3DRasterGrid& DEM);
        ~Crit3DHourlyMeteoMaps();

        void clear();
        void initialize();

        gis::Crit3DRasterGrid* getMapFromVar(meteoVariable myVar);
        bool computeET0PMMap(const gis::Crit3DRasterGrid &DEM, Crit3DRadiationMaps *radMaps);
        bool computeRelativeHumidityMap(gis::Crit3DRasterGrid* myGrid);
        bool computeLeafWetnessMap();
        void setComputed(bool value);
        bool getComputed();
    };


#endif // METEOMAPS_H
