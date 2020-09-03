#ifndef GDALRASTERFUNCTIONS_H
#define GDALRASTERFUNCTIONS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <gdal_priv.h>

    class QString;

    bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid *myRaster, int &utmZone, QString &error);
    bool convertGdalRaster(GDALDataset* dataset, gis::Crit3DRasterGrid *myRaster, int &utmZone, QString &error);


#endif // GDALRASTERFUNCTIONS_H
