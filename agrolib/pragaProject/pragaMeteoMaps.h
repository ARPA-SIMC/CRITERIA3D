#ifndef PRAGAMETEOMAPS_H
#define PRAGAMETEOMAPS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif

    #ifndef METEOMAPS_H
        #include "meteoMaps.h"
    #endif

    class PragaHourlyMeteoMaps
    {

    public:
        gis::Crit3DRasterGrid* mapHourlyWindVectorInt;
        gis::Crit3DRasterGrid* mapHourlyWindVectorDir;
        gis::Crit3DRasterGrid* mapHourlyWindVectorX;
        gis::Crit3DRasterGrid* mapHourlyWindVectorY;

        PragaHourlyMeteoMaps(const gis::Crit3DRasterGrid& DEM);
        ~PragaHourlyMeteoMaps();

        void clear();
        void initialize();

        gis::Crit3DRasterGrid* getMapFromVar(meteoVariable myVar);
        bool computeWindVector();
    };

#endif // PRAGAMETEOMAPS_H
