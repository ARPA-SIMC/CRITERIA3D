
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

    // TODO read utm zone

    int nBands = dataset->GetRasterCount();
    qDebug() << "Nr. band: " << QString::number(nBands);

    // TODO select band
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

