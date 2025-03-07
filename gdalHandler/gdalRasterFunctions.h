#ifndef GDALRASTERFUNCTIONS_H
#define GDALRASTERFUNCTIONS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <gdal_priv.h>

    class QString;

    bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid *rasterPointer, int &utmZone, QString &errorStr);

    bool convertGdalRaster(GDALDataset* dataset, gis::Crit3DRasterGrid *myRaster, int &utmZone, QString &errorStr);

    bool gdalReprojection(GDALDatasetH &srcDataset, GDALDatasetH &dstDataset,
                          QString newProjection, QString projFileName, QString &errorStr);

    bool gdalExportPng(GDALDatasetH &rasterDataset, QString pngFileName, QString pngProjection, QString &errorStr);


#endif // GDALRASTERFUNCTIONS_H
