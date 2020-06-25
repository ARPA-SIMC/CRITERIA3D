
#include "gdalRasterFunctions.h"
#include "commonConstants.h"

#include <iostream>
#include <gdal_priv.h>

#include <QString>
#include <QDebug>


bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid* myRaster, QString* myError)
{
    if (myRaster == nullptr) return false;
    myRaster->isLoaded = false;

    GDALDataset* dataset = (GDALDataset*) GDALOpen(fileName.toStdString().data(), GA_ReadOnly);
    if(! dataset)
    {
        *myError = "Load raster failed!";
        return false;
    }

    // Driver
    double adfGeoTransform[6];
    qDebug() << "Driver: " << dataset->GetDriver()->GetDescription() << dataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME);

    // size
    qDebug() << "Size = " << dataset->GetRasterXSize() << dataset->GetRasterYSize() << dataset->GetRasterCount();

    // projection
    if (dataset->GetProjectionRef() != nullptr)
        qDebug() << "Projection = " << dataset->GetProjectionRef();

    // TODO chek spatial reference
    OGRSpatialReference spatialReference;
    spatialReference.SetWellKnownGeogCS("WGS84");
    // TODO check utm zone

    // Origin and size
    if (dataset->GetGeoTransform(adfGeoTransform) == CE_None)
    {
        qDebug() << "Origin = " << adfGeoTransform[0] << adfGeoTransform[3];
        qDebug() << "Pixel Size = " << adfGeoTransform[1] << adfGeoTransform[5];
    }

    // TODO select band
    qDebug() << "Nr. band: " << dataset->GetRasterCount();
    GDALRasterBand* band = dataset->GetRasterBand(1);

    if(band == nullptr)
    {
        *myError = "Band 1 is void!";
        return false;
    }

    // NODATA
    if (band)
    {
        int success;
        double nodataValue = band->GetNoDataValue(&success);
        if (success)
        {
            qDebug() << "Nodata: " << QString::number(nodataValue);
            myRaster->header->flag = float(nodataValue);
        }
        else
        {
            qDebug() << "Missing NODATA";
            myRaster->header->flag = NODATA;
        }
    }

    // SIZE
    myRaster->header->nrCols = dataset->GetRasterXSize();
    myRaster->header->nrRows = dataset->GetRasterYSize();

    //myRaster->initializeGrid();

    // TODO read xll position
    // read data
    // updateMinmax

    return true;
}

