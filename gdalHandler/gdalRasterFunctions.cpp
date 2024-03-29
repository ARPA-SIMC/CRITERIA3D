#include "gdalRasterFunctions.h"
#include "commonConstants.h"

#include <gdalwarper.h>

#include <iostream>
#include <cmath>

#include <QFileInfo>
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
    OGRSpatialReference* spatialReference;
    if (dataset->GetProjectionRef() != "")
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
        qDebug() << "UTM zone = " << spatialReference->GetUTMZone();
    }
    else
    {
        qDebug() << "Projection is missing! It will use WGS84 UTM zone:" << utmZone;
        spatialReference = new OGRSpatialReference();
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
    int xSize = dataset->GetRasterXSize();
    int ySize = dataset->GetRasterYSize();
    myRaster->header->nrCols = xSize;
    myRaster->header->nrRows = ySize;
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


bool gdalReprojection(GDALDatasetH &srcDataset, GDALDatasetH &dstDataset,
                      QString newProjection, QString projFileName, QString &errorStr)
{
    // check destination coordinate system
    OGRSpatialReference dstSpatialRef;
    int ogrError = dstSpatialRef.importFromEPSG(newProjection.toInt());
    if (ogrError != OGRERR_NONE)
    {
        errorStr = "Error in gdalReprojection. Wrong projection: " + newProjection;
        return false;
    }

    // source projection
    const char* srcWKTProjection = GDALGetProjectionRef( srcDataset ) ;

    // destination projection
    char* dstWKTProjection;
    dstSpatialRef.exportToWkt( &dstWKTProjection );

    // datatype
    GDALDataType dataType = GDALGetRasterDataType(GDALGetRasterBand(srcDataset, 1));

    // output driver (GeoTIFF format)
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    if (hDriver == nullptr)
    {
        errorStr = "Error in gdalReprojection (GDALGetDriverByName GTiff)";
        return false;
    }

    // Create a transformer that maps from source pixel/line coordinates
    // to destination georeferenced coordinates (not destination
    // pixel line).  We do that by omitting the destination dataset
    // handle (setting it to nullptr).
    void *handleTransformArg = GDALCreateGenImgProjTransformer(srcDataset, srcWKTProjection, nullptr, dstWKTProjection, FALSE, 0, 1 );

    if ( handleTransformArg == nullptr )
    {
        errorStr = "Error in gdalReprojection (GDALCreateGenImgProjTransformer)";
        return false;
    }

    // Get approximate output georeferenced bounds and resolution for file.
    int nrPixels, nrLines;
    double adfDstGeoTransform[6];

    CPLErr eErr;
    eErr = GDALSuggestedWarpOutput( srcDataset,
                                   GDALGenImgProjTransform, handleTransformArg,
                                   adfDstGeoTransform, &nrPixels, &nrLines );
    if( eErr != CE_None )
    {
        errorStr = "Error gdalReprojection (GDALSuggestedWarpOutput). Error nr: " + QString::number(eErr);
        return false;
    }

    char *createOptions[] = { strdup("COMPRESS=LZW"), nullptr };

    // Create the projected dataset
    dstDataset = GDALCreate( hDriver, strdup(projFileName.toStdString().c_str()), nrPixels, nrLines,
                             GDALGetRasterCount(srcDataset), dataType, createOptions );
    if( dstDataset == nullptr )
    {
        errorStr = "Error in gdalReprojection (GDALCreate)";
        return false;
    }

    // Write out the projection definition.
    GDALSetProjection( dstDataset, dstWKTProjection );
    GDALSetGeoTransform( dstDataset, adfDstGeoTransform );

    // Setup warp options.
    GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
    psWarpOptions->hSrcDS = srcDataset;
    psWarpOptions->hDstDS = dstDataset;
    psWarpOptions->nBandCount = MIN(GDALGetRasterCount(srcDataset),
                                    GDALGetRasterCount(dstDataset));

    psWarpOptions->panSrcBands = (int *)CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);
    psWarpOptions->panDstBands = (int *)CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);

    for( int iBand = 0; iBand < psWarpOptions->nBandCount; iBand++ )
    {
        psWarpOptions->panSrcBands[iBand] = iBand+1;
        psWarpOptions->panDstBands[iBand] = iBand+1;
    }

    /* -------------------------------------------------------------------- */
    /*      Setup no data values                                            */
    /* -------------------------------------------------------------------- */
    for (int i = 0; i < psWarpOptions->nBandCount; i++)
    {
        GDALRasterBandH rasterBand = GDALGetRasterBand( psWarpOptions->hSrcDS, psWarpOptions->panSrcBands[i] );

        int hasNoDataValue;
        double noDataValue = GDALGetRasterNoDataValue(rasterBand, &hasNoDataValue );

        if ( hasNoDataValue )
        {
            // Check if the nodata value is out of range
            int bClamped = FALSE;
            int bRounded = FALSE;
            CPL_IGNORE_RET_VAL(
                GDALAdjustValueToDataType( GDALGetRasterDataType( rasterBand ), noDataValue, &bClamped, &bRounded ));
            if (! bClamped )
            {
                GDALWarpInitNoDataReal( psWarpOptions, -1e10 );
                GDALSetRasterNoDataValue( GDALGetRasterBand(psWarpOptions->hDstDS, i+1), noDataValue);
                psWarpOptions->padfSrcNoDataReal[i] = noDataValue;
                psWarpOptions->padfDstNoDataReal[i] = noDataValue;
            }
        }
    }

    psWarpOptions->pTransformerArg = handleTransformArg;
    psWarpOptions->papszWarpOptions = CSLSetNameValue( psWarpOptions->papszWarpOptions, "INIT_DEST", "NO_DATA" );
    psWarpOptions->papszWarpOptions = CSLSetNameValue( psWarpOptions->papszWarpOptions, "WRITE_FLUSH", "YES" );
    CPLFetchBool( psWarpOptions->papszWarpOptions, "OPTIMIZE_SIZE", true );

    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    // Initialize and execute the warp operation.
    int cplError;
    cplError = GDALReprojectImage(srcDataset, srcWKTProjection,
                              dstDataset, dstWKTProjection,
                              GRA_Bilinear,
                              0.0, 0.0,
                              GDALTermProgress, nullptr,
                              psWarpOptions);

    GDALDestroyGenImgProjTransformer( handleTransformArg );
    GDALDestroyWarpOptions( psWarpOptions );

    if (cplError != CE_None)
    {
        errorStr =  CPLGetLastErrorMsg();
        return false;
    }
    else
    {
        errorStr = "";
        return true;
    }
}


// make a png copy
bool gdalExportPng(GDALDatasetH &rasterDataset, QString pngFileName, QString pngProjection, QString &errorStr)
{
    // rename old file
    QFile::remove(pngFileName);

    // tmp file
    QFileInfo outputFile(pngFileName);
    QString tmpFileName = outputFile.absolutePath() + "/tmp";

    // reprojection
    GDALDatasetH projDataset;
    if (! pngProjection.isEmpty())
    {
        if (! gdalReprojection(rasterDataset, projDataset, pngProjection, tmpFileName, errorStr))
        {
            return false;
        }
    }

    // copy png
    GDALDriverH driver = GDALGetDriverByName("PNG");
    GDALDriver *pngDriver = (GDALDriver *)driver;
    GDALDataset *pngDataset;

    if (! pngProjection.isEmpty())
    {
        pngDataset = pngDriver->CreateCopy(strdup(pngFileName.toStdString().c_str()),
                                           (GDALDataset *)projDataset, FALSE, NULL, NULL, NULL);
        GDALClose( projDataset );
        QFile::remove(tmpFileName);
    }
    else
    {
        pngDataset = pngDriver->CreateCopy(strdup(pngFileName.toStdString().c_str()),
                                           (GDALDataset *)rasterDataset, FALSE, NULL, NULL, NULL);
    }

    if (pngDataset == nullptr)
    {
        errorStr = "Error to create PNG copy.";
        return false;
    }
    else
    {
        GDALClose( pngDataset );
        return true;
    }
}


