#ifndef SNOWMAPS_H
#define SNOWMAPS_H

    #ifndef SNOWPOINT_H
        #include "snowPoint.h"
    #endif

    class Crit3DSnowMaps
    {
    public:
        Crit3DSnowMaps();
        ~Crit3DSnowMaps();
        Crit3DSnowMaps(const gis::Crit3DRasterGrid& dtmGrid);

        void initializeMaps();
        void resetSnowModel(gis::Crit3DRasterGrid* myGrd, Crit3DSnowPoint* snowPoint);
        void updateMap(Crit3DSnowPoint* snowPoint, int row, int col);

        gis::Crit3DRasterGrid* getSnowFallMap();
        gis::Crit3DRasterGrid* getSnowMeltMap();
        gis::Crit3DRasterGrid* getSnowWaterEquivalentMap();
        gis::Crit3DRasterGrid* getIceContentMap();
        gis::Crit3DRasterGrid* getLWContentMap();
        gis::Crit3DRasterGrid* getInternalEnergyMap();
        gis::Crit3DRasterGrid* getSurfaceInternalEnergyMap();
        gis::Crit3DRasterGrid* getSnowSurfaceTempMap();
        gis::Crit3DRasterGrid* getAgeOfSnowMap();

        static double computeSurfaceInternalEnergy(double initSnowSurfaceTemp,int bulkDensity, double initSWE, double snowSkinThickness);
        static double computeInternalEnergyMap(double initSoilPackTemp,int bulkDensity, double initSWE);

    private:
        gis::Crit3DRasterGrid* _snowFallMap;
        gis::Crit3DRasterGrid* _snowMeltMap;
        gis::Crit3DRasterGrid* _snowWaterEquivalentMap;
        gis::Crit3DRasterGrid* _iceContentMap;
        gis::Crit3DRasterGrid* _lWContentMap;
        gis::Crit3DRasterGrid* _internalEnergyMap;
        gis::Crit3DRasterGrid* _surfaceInternalEnergyMap;
        gis::Crit3DRasterGrid* _snowSurfaceTempMap;
        gis::Crit3DRasterGrid* _ageOfSnowMap;

        double _initSoilPackTemp;
        double _initSnowSurfaceTemp;
        bool _isLoaded;
    };


#endif // SNOWMAPS_H
