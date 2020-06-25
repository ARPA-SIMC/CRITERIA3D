
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
    myRaster->header->nrCols = dataset->GetRasterXSize();
    myRaster->header->nrRows = dataset->GetRasterYSize();

    // projection
    if (dataset->GetProjectionRef() != nullptr)
        qDebug() << "Projection = " << dataset->GetProjectionRef();

    // TODO chek spatial reference
    OGRSpatialReference spatialReference;
    spatialReference.SetWellKnownGeogCS("WGS84");
    // TODO check utm zone

    // Origin (top left) and size
    if (dataset->GetGeoTransform(adfGeoTransform) == CE_None)
    {
        qDebug() << "Origin = " << adfGeoTransform[0] << adfGeoTransform[3];
        qDebug() << "Pixel Size = " << adfGeoTransform[1] << adfGeoTransform[5];
    }
    if (adfGeoTransform[1] != -adfGeoTransform[5])
    {
        *myError = "Not regular pixel size!";
        return false;
    }
    myRaster->header->cellSize = adfGeoTransform[1];
    myRaster->header->llCorner.x = adfGeoTransform[0];
    myRaster->header->llCorner.y = adfGeoTransform[3] - myRaster->header->cellSize * myRaster->header->nrRows;

    // TODO select band
    qDebug() << "Nr. band: " << dataset->GetRasterCount();
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if(band == nullptr)
    {
        *myError = "Missing data!";
        return false;
    }

    // NODATA
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

    // Initialize
    myRaster->initializeGrid(1);

    // TODO read data type (float or byte)

    // read data (band 1)
    int xSize = band->GetXSize();
    int ySize = band->GetYSize();
    float* data = (float *) CPLMalloc(sizeof(float) * xSize * ySize);
    band->RasterIO(GF_Read, 0, 0, xSize, ySize, data, xSize, ySize, GDT_Float32, 0, 0);

    for (int row = 0; row < myRaster->header->nrRows; row++)
        for (int col = 0; col < myRaster->header->nrCols; col++)
            myRaster->value[row][col] = data[row*xSize+col];

    CPLFree(data);

    updateMinMaxRasterGrid(myRaster);

    return true;
}

