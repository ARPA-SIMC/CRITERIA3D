
#include "gdalRasterFunctions.h"
#include "commonConstants.h"

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

    // TODO chek spatial reference
    OGRSpatialReference spatialReference;
    spatialReference.SetWellKnownGeogCS("WGS84");
    const char* projRef = dataset->GetProjectionRef();
    qDebug(projRef);

    int nBands = dataset->GetRasterCount();
    qDebug() << "Nr. band: " << QString::number(nBands);

    // TODO select band
    GDALRasterBand* first = dataset->GetRasterBand(1);
    if (first)
    {
        int success;
        double nodataValue = first->GetNoDataValue(&success);
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

    // dimension
    myRaster->header->nrCols = dataset->GetRasterXSize();
    myRaster->header->nrRows = dataset->GetRasterYSize();

    // TODO read xll position
    // raster initialize
    // read data
    // updateMinmax

    return true;
}

