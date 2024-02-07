#include "gdalRasterFunctions.h"
#include "commonConstants.h"

#include <gdalwarper.h>

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
    OGRSpatialReference* spatialReference;
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


bool gdalReprojection(GDALDatasetH rasterDataset, GDALDatasetH &projDataset, char *pszProjection,
                      QString newProjection, QString projFileName, QString &errorStr)
{
    // Create output with same datatype as first input band.
    GDALDataType eDT = GDALGetRasterDataType(GDALGetRasterBand(rasterDataset, 1));

    // Get output driver (GeoTIFF format)
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    if (hDriver == nullptr)
    {
        errorStr = "Error in reprojection (GDALGetDriverByName GTiff)";
        return false;
    }

    // Get source coordinate system
    char *pszDstWKT = nullptr;
    OGRSpatialReference oSRS;
    oSRS.SetWellKnownGeogCS(newProjection.toStdString().c_str());
    oSRS.exportToWkt( &pszDstWKT );

    // Create a transformer that maps from source pixel/line coordinates
    // to destination georeferenced coordinates (not destination
    // pixel line).  We do that by omitting the destination dataset
    // handle (setting it to nullptr).
    void *hTransformArg = GDALCreateGenImgProjTransformer(rasterDataset, pszProjection, nullptr, pszDstWKT, FALSE, 0, 1 );

    if ( hTransformArg == nullptr )
    {
        errorStr = "Error in reprojection (GDALCreateGenImgProjTransformer)";
        return false;
    }

    // Get approximate output georeferenced bounds and resolution for file.
    double adfDstGeoTransform[6];
    int nPixels=0, nLines=0;
    CPLErr eErr;
    eErr = GDALSuggestedWarpOutput( rasterDataset,
                                   GDALGenImgProjTransform, hTransformArg,
                                   adfDstGeoTransform, &nPixels, &nLines );
    if( eErr != CE_None )
    {
        errorStr = "Error reprojection (GDALSuggestedWarpOutput)";
        return false;
    }

    char *createOptions[] = {strdup("COMPRESS=LZW"), nullptr};

    // Create the projected dataset
    projDataset = GDALCreate( hDriver, strdup(projFileName.toStdString().c_str()), nPixels, nLines,
                             GDALGetRasterCount(rasterDataset), eDT, createOptions );

    if( projDataset == nullptr )
    {
        errorStr = "Error in reprojection (GDALCreate)";
        return false;
    }

    // Write out the projection definition.
    GDALSetProjection( projDataset, pszDstWKT );
    GDALSetGeoTransform( projDataset, adfDstGeoTransform );

    // Setup warp options.
    GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
    psWarpOptions->hSrcDS = rasterDataset;
    psWarpOptions->hDstDS = projDataset;
    psWarpOptions->nBandCount = MIN(GDALGetRasterCount(rasterDataset),
                                    GDALGetRasterCount(projDataset));

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

    psWarpOptions->pTransformerArg = hTransformArg;
    psWarpOptions->papszWarpOptions = CSLSetNameValue( psWarpOptions->papszWarpOptions, "INIT_DEST", "NO_DATA" );
    psWarpOptions->papszWarpOptions = CSLSetNameValue( psWarpOptions->papszWarpOptions, "WRITE_FLUSH", "YES" );
    CPLFetchBool( psWarpOptions->papszWarpOptions, "OPTIMIZE_SIZE", true );

    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    // Initialize and execute the warp operation.
    eErr = GDALReprojectImage(rasterDataset, pszProjection,
                              projDataset, pszDstWKT,
                              GRA_Bilinear,
                              0.0, 0.0,
                              GDALTermProgress, nullptr,
                              psWarpOptions);

    GDALDestroyGenImgProjTransformer( hTransformArg );
    GDALDestroyWarpOptions( psWarpOptions );

    if (eErr != CE_None)
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


// make a bitmap copy (tipically PNG)
bool gdalExportImg(GDALDatasetH rasterDataset, QString format, QString imgFileName, QString &errorStr)
{
    /*
    GDALDriverH driver = GDALGetDriverByName( strdup(format.toStdString().c_str()) );
    GDALDriver *imgDriver = (GDALDriver *)driver;
    GDALDataset *imgDataset = imgDriver->CreateCopy(strdup(imgFileName.toStdString().c_str()), (GDALDataset *)rasterDataset, FALSE, NULL, NULL, NULL);

    GDALClose( imgDataset );
    */

    return true;
}


