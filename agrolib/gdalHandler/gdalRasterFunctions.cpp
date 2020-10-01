
#include "gdalRasterFunctions.h"
#include "commonConstants.h"

#include <iostream>
#include <cmath>

#include <QString>
#include <QDebug>


/*! readGdalRaster
 * \brief open a raster file with GDAL library
 * \return GDALDataset
 */
bool readGdalRaster(QString fileName, gis::Crit3DRasterGrid* myRaster, int &utmZone, QString &error)
{
    // check
    if (myRaster == nullptr) return false;
    if (fileName == "") return false;

    GDALDataset* dataset = (GDALDataset*) GDALOpen(fileName.toStdString().data(), GA_ReadOnly);
    if(! dataset)
    {
        error = "Load raster failed!";
        return false;
    }

    bool myResult = convertGdalRaster(dataset, myRaster, utmZone, error);
    GDALClose(dataset);

    return myResult;
}


/*! convertGdalRaster
 * \brief convert a GDAL dataset in a Crit3DRasterGrid
 */
bool convertGdalRaster(GDALDataset* dataset, gis::Crit3DRasterGrid* myRaster, int &utmZone, QString &error)
{
    myRaster->isLoaded = false;

    // Driver
    double adfGeoTransform[6];
    qDebug() << "Driver =" << dataset->GetDriver()->GetDescription() << dataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME);

    // Layers and band
    qDebug() << "Nr. layers =" << dataset->GetLayerCount();
    qDebug() << "Nr. band =" << dataset->GetRasterCount();

    // size
    qDebug() << "Size (x,y) =" << dataset->GetRasterXSize() << dataset->GetRasterYSize();

    // projection
    OGRSpatialReference* spatialReference = new OGRSpatialReference();
    QString prjString = QString::fromStdString(dataset->GetProjectionRef());
    if (prjString != "")
    {
        qDebug() << "Projection =" << dataset->GetProjectionRef();
        spatialReference = new OGRSpatialReference(dataset->GetProjectionRef());

        // TODO geo projection?
        if (! spatialReference->IsProjected())
        {
            error = "Not projected data";
            return false;
        }

        // TODO: check WGS84 -> convert

        // UTM zone
        utmZone = spatialReference->GetUTMZone();
        qDebug() << "UTM zone =" << spatialReference->GetUTMZone();
    }
    else
    {
        qDebug() << "Projection is missing! It will use WGS84 UTM zone:" << utmZone;
        spatialReference->SetWellKnownGeogCS("WGS84");
    }

    // Origin (top left) and size
    if (dataset->GetGeoTransform(adfGeoTransform) == CE_None)
    {
        qDebug() << "Origin =" << adfGeoTransform[0] << adfGeoTransform[3];
        qDebug() << "Pixel Size =" << adfGeoTransform[1] << adfGeoTransform[5];
    }
    if (adfGeoTransform[1] != fabs(adfGeoTransform[5]))
    {
        error = "Not regular pixel size! Will be used x size.";
    }

    // TODO choose band
    // read band 1
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if(band == nullptr)
    {
        error = "Missing data!";
        return false;
    }

    int             bGotMin, bGotMax;
    double          adfMinMax[2];
    qDebug() << "Type =" << GDALGetDataTypeName(band->GetRasterDataType());
    qDebug() << "ColorInterpretation =" << GDALGetColorInterpretationName(band->GetColorInterpretation());

    adfMinMax[0] = band->GetMinimum( &bGotMin );
    adfMinMax[1] = band->GetMaximum( &bGotMax );

    // min e max
    if( ! (bGotMin && bGotMax) )
    {
        GDALComputeRasterMinMax(GDALRasterBandH(band), TRUE, adfMinMax);
    }
    qDebug() << "Min =" << adfMinMax[0] << " Max =" << adfMinMax[1];

    if (band->GetOverviewCount() > 0)
        qDebug() << "Band has " << band->GetOverviewCount() << " overviews";

    if (band->GetColorTable() != nullptr)
        qDebug() << "Band has a color table. Nr. of entries =" << band->GetColorTable()->GetColorEntryCount();

    // nodata value
    int noDataOk;
    double nodataValue = band->GetNoDataValue(&noDataOk);
    if (! noDataOk)
    {
        qDebug() << "Missing NODATA: will be set on minimum value.";
        nodataValue = adfMinMax[0];
    }
    qDebug() << "Nodata =" << QString::number(nodataValue);

    // initialize raster
    myRaster->header->nrCols = dataset->GetRasterXSize();
    myRaster->header->nrRows = dataset->GetRasterYSize();
    myRaster->header->cellSize = adfGeoTransform[1];
    myRaster->header->llCorner.x = adfGeoTransform[0];
    myRaster->header->llCorner.y = adfGeoTransform[3] - myRaster->header->cellSize * myRaster->header->nrRows;
    myRaster->header->flag = float(nodataValue);

    if (! myRaster->initializeGrid(myRaster->header->flag))
    {
        error = "Memory error: file too big.";
        return false;
    }

    // read data
    int xSize = band->GetXSize();
    int ySize = band->GetYSize();
    float* data = (float *) CPLMalloc(sizeof(float) * xSize * ySize);
    CPLErr errGdal = band->RasterIO(GF_Read, 0, 0, xSize, ySize, data, xSize, ySize, GDT_Float32, 0, 0);

    if (errGdal > CE_Warning)
    {
        error = "Error in RasterIO";
        CPLFree(data);
        return false;
    }

    // set data
    for (int row = 0; row < myRaster->header->nrRows; row++)
        for (int col = 0; col < myRaster->header->nrCols; col++)
            myRaster->value[row][col] = data[row*xSize+col];

    // free memory
    CPLFree(data);

    // min & max
    if (noDataOk)
    {
        myRaster->minimum = float(adfMinMax[0]);
        myRaster->maximum = float(adfMinMax[1]);
    }
    else
    {
        updateMinMaxRasterGrid(myRaster);
    }

    return true;
}

