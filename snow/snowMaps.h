#ifndef SNOWMAPS_H
#define SNOWMAPS_H

    #ifndef SNOW_H
        #include "snow.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    class Crit3DSnowMaps
    {
    public:
        bool isInitialized;

        Crit3DSnowMaps();
        ~Crit3DSnowMaps();

        void clear();
        void initialize(const gis::Crit3DRasterGrid &dtm, double skinThickness);
        void resetSnowModel(double skinThickness);

        void updateMap(Crit3DSnow &snowPoint, int row, int col);
        void setPoint(Crit3DSnow &snowPoint, int row, int col);

        void updateRangeMaps();

        gis::Crit3DRasterGrid* getSnowWaterEquivalentMap();
        gis::Crit3DRasterGrid* getIceContentMap();
        gis::Crit3DRasterGrid* getLWContentMap();
        gis::Crit3DRasterGrid* getInternalEnergyMap();
        gis::Crit3DRasterGrid* getSurfaceEnergyMap();
        gis::Crit3DRasterGrid* getSnowSurfaceTempMap();
        gis::Crit3DRasterGrid* getAgeOfSnowMap();

        gis::Crit3DRasterGrid* getSnowFallMap();
        gis::Crit3DRasterGrid* getSnowMeltMap();
        gis::Crit3DRasterGrid* getSensibleHeatMap();
        gis::Crit3DRasterGrid* getLatentHeatMap();

    private:
        gis::Crit3DRasterGrid* _snowWaterEquivalentMap;
        gis::Crit3DRasterGrid* _iceContentMap;
        gis::Crit3DRasterGrid* _liquidWaterContentMap;
        gis::Crit3DRasterGrid* _internalEnergyMap;
        gis::Crit3DRasterGrid* _surfaceEnergyMap;
        gis::Crit3DRasterGrid* _snowSurfaceTempMap;
        gis::Crit3DRasterGrid* _ageOfSnowMap;

        gis::Crit3DRasterGrid* _snowFallMap;
        gis::Crit3DRasterGrid* _snowMeltMap;
        gis::Crit3DRasterGrid* _sensibleHeatMap;
        gis::Crit3DRasterGrid* _latentHeatMap;

        double _initSoilPackTemp;
        double _initSnowSurfaceTemp;
    };


#endif // SNOWMAPS_H
