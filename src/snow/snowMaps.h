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
        void initializeSnowMaps(const gis::Crit3DRasterGrid &dtm, double skinThickness);
        void resetSnowModel(double skinThickness);

        void updateMapRowCol(Crit3DSnow &snowPoint, int row, int col);
        void flagMapRowCol(int row, int col);

        void setPoint(Crit3DSnow &snowPoint, int row, int col);

        void updateRangeMaps();

        gis::Crit3DRasterGrid* getSnowWaterEquivalentMap() {return _snowWaterEquivalentMap; }
        gis::Crit3DRasterGrid* getIceContentMap() { return _iceContentMap; }
        gis::Crit3DRasterGrid* getLWContentMap() { return _liquidWaterContentMap; }
        gis::Crit3DRasterGrid* getInternalEnergyMap() { return _internalEnergyMap; }
        gis::Crit3DRasterGrid* getSurfaceEnergyMap() { return _surfaceEnergyMap; }
        gis::Crit3DRasterGrid* getSnowSurfaceTempMap() { return _snowSurfaceTempMap; }
        gis::Crit3DRasterGrid* getAgeOfSnowMap() { return _ageOfSnowMap; }

        gis::Crit3DRasterGrid* getSnowFallMap() {return _snowFallMap;}
        gis::Crit3DRasterGrid* getSnowMeltMap() {return _snowMeltMap;}
        gis::Crit3DRasterGrid* getDeltaSWEMap() {return _deltaSWEMap;}
        gis::Crit3DRasterGrid* getSensibleHeatMap() {return _sensibleHeatMap;}
        gis::Crit3DRasterGrid* getLatentHeatMap() {return _latentHeatMap;}

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
        gis::Crit3DRasterGrid* _deltaSWEMap;
        gis::Crit3DRasterGrid* _sensibleHeatMap;
        gis::Crit3DRasterGrid* _latentHeatMap;

        double _initSoilPackTemp;
        double _initSnowSurfaceTemp;
    };


#endif // SNOWMAPS_H
