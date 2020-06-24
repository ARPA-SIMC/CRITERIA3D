#ifndef GDALRASTERFUNCTIONS_H
#define GDALRASTERFUNCTIONS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    class QString;

    bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid* myRaster, QString* myError);


#endif // GDALRASTERFUNCTIONS_H
