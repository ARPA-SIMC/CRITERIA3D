#ifndef GDALRASTERFUNCTIONS_H
#define GDALRASTERFUNCTIONS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <gdal_priv.h>

    class QString;

    bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid *myRaster, int &utmZone, QString &error);

    bool convertGdalRaster(GDALDataset* dataset, gis::Crit3DRasterGrid *myRaster, int &utmZone, QString &error);

    bool gdalReprojection(GDALDatasetH rasterDataset, GDALDatasetH &projDataset, char *pszProjection,
                          QString newProjection, QString projFileName, QString &errorStr);

    bool gdalExportImg(GDALDatasetH rasterDataset, QString format, QString imgFileName, QString &errorStr);


#endif // GDALRASTERFUNCTIONS_H
