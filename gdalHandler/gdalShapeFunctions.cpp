#include "gdalShapeFunctions.h"
#include <QFileInfo>
#include <string.h>
#include <ogrsf_frmts.h>
#include "ogr_spatialref.h"
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <gdalwarper.h>


bool shapeToRaster(QString shapeFileName, QString shapeField, QString resolution, QString proj,
                   QString outputName, QString paletteFileName, QString &errorStr)
{

    int error = -1;
    GDALAllRegister();
    QFileInfo file(outputName);
    QString ext = file.completeSuffix();

    std::string formatOption;
    if (mapExtensionShortName.contains(ext))
    {
        formatOption = mapExtensionShortName.value(ext).toStdString();
    }
    else
    {
        errorStr = "Unknown output format";
        return false;
    }

    GDALDataset* shpDS;
    shpDS = (GDALDataset*)GDALOpenEx(shapeFileName.toStdString().data(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if( shpDS == nullptr )
    {
        errorStr = "Open shapefile failed";
        return false;
    }

    // projection
    char *pszProjection = nullptr;
    OGRSpatialReference srs;
    OGRSpatialReference * pOrigSrs = shpDS->GetLayer(0)->GetSpatialRef();
    if ( pOrigSrs )
    {
        srs = *pOrigSrs;
    }
    if ( srs.IsProjected() )
    {
        srs.exportToWkt( &pszProjection );
    }
    else
    {
        GDALClose(shpDS);
        errorStr = "Missing projection";
        return false;
    }

    std::string outputNoReprojStr;
    if (proj.isEmpty())
    {
        outputNoReprojStr = outputName.toStdString();   // there is no reprojection to do
    }
    else
    {
        QString fileName = file.absolutePath() + "/" + file.baseName() + "_noreproj." + ext;
        outputNoReprojStr = fileName.toStdString();
    }

    std::string res = resolution.toStdString();

    // set options
    char *options[] = {strdup("-at"), strdup("-of"), strdup(formatOption.c_str()), strdup("-a"), strdup(shapeField.toStdString().c_str()), strdup("-a_nodata"), strdup("-9999"),
                       strdup("-a_srs"), pszProjection, strdup("-tr"), strdup(res.c_str()), strdup(res.c_str()), strdup("-co"), strdup("COMPRESS=LZW"), nullptr};

    GDALRasterizeOptions *psOptions = GDALRasterizeOptionsNew(options, nullptr);
    if( psOptions == nullptr )
    {
        GDALClose(shpDS);
        errorStr = "psOptions is null";
        return false;
    }

    // rasterize
    GDALDatasetH noColorDataset = GDALRasterize(strdup(outputNoReprojStr.c_str()), nullptr, shpDS, psOptions, &error);

    GDALClose(shpDS);
    GDALRasterizeOptionsFree(psOptions);

    if (noColorDataset == nullptr || error == 1)
    {
        CPLFree( pszProjection );
        return false;
    }

    // save color map (before reprojection)
    GDALDatasetH rasterizeDS;
    rasterizeDS = GDALDEMProcessing(strdup(outputName.toStdString().c_str()), noColorDataset, "color-relief",
                                      strdup(paletteFileName.toStdString().c_str()), nullptr, &error);

    GDALClose(noColorDataset);

    if (rasterizeDS == nullptr || error == 1)
    {
        QFile::remove(outputName);
        CPLFree( pszProjection );
        return false;
    }

    // reprojection
    if (!proj.isEmpty())
    {
        GDALDatasetH hDstDS;
        GDALDriverH hDriver;
        GDALDataType eDT;
        // Create output with same datatype as first input band.
        eDT = GDALGetRasterDataType(GDALGetRasterBand(rasterizeDS,1));

        // Get output driver (GeoTIFF format)
        hDriver = GDALGetDriverByName( "GTiff" );
        if (hDriver == nullptr)
        {
            errorStr = "Error GDALGetDriverByName";
            GDALClose(rasterizeDS);
            CPLFree( pszProjection );
            return false;
        }

        // Get Source coordinate system.
        char *pszDstWKT = nullptr;
        OGRSpatialReference oSRS;
        oSRS.SetWellKnownGeogCS(proj.toStdString().c_str());
        oSRS.exportToWkt( &pszDstWKT );
        // Create a transformer that maps from source pixel/line coordinates
        // to destination georeferenced coordinates (not destination
        // pixel line).  We do that by omitting the destination dataset
        // handle (setting it to nullptr).
        void *hTransformArg;
        hTransformArg =
            GDALCreateGenImgProjTransformer( rasterizeDS, pszProjection, nullptr, pszDstWKT,
                                             FALSE, 0, 1 );
        if ( hTransformArg == nullptr )
        {
            errorStr = "Error GDALCreateGenImgProjTransformer";
            GDALClose(rasterizeDS);
            CPLFree( pszProjection );
            return false;
        }

        // Get approximate output georeferenced bounds and resolution for file.
        double adfDstGeoTransform[6];
        int nPixels=0, nLines=0;
        CPLErr eErr;
        eErr = GDALSuggestedWarpOutput( rasterizeDS,
                                        GDALGenImgProjTransform, hTransformArg,
                                        adfDstGeoTransform, &nPixels, &nLines );
        if( eErr != CE_None )
        {
            errorStr = "Error GDALSuggestedWarpOutput";
            GDALClose(rasterizeDS);
            CPLFree( pszProjection );
            return false;
        }

        char *createOptions[] = {strdup("COMPRESS=LZW"), nullptr};
        // Create the output file.
        QString fileName = file.absolutePath() + "/" + file.baseName() + "_proj." + ext;
        hDstDS = GDALCreate( hDriver, strdup(fileName.toStdString().c_str()) , nPixels, nLines,
                             GDALGetRasterCount(rasterizeDS), eDT, createOptions );

        if( hDstDS == nullptr )
        {
            errorStr = "Error GDALCreate output reprojected";
            GDALClose(rasterizeDS);
            CPLFree( pszProjection );
            return false;
        }

        // Write out the projection definition.
        GDALSetProjection( hDstDS, pszDstWKT );
        GDALSetGeoTransform( hDstDS, adfDstGeoTransform );

        // Setup warp options.
        GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
        psWarpOptions->hSrcDS = rasterizeDS;
        psWarpOptions->hDstDS = hDstDS;
        psWarpOptions->nBandCount = MIN(GDALGetRasterCount(rasterizeDS),
                                     GDALGetRasterCount(hDstDS));

        psWarpOptions->panSrcBands = (int *)
            CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);
        psWarpOptions->panDstBands = (int *)
            CPLMalloc(sizeof(int) * psWarpOptions->nBandCount);

        for( int iBand = 0; iBand < psWarpOptions->nBandCount; iBand++ )
        {
            psWarpOptions->panSrcBands[iBand] = iBand+1;
            psWarpOptions->panDstBands[iBand] = iBand+1;
        }

        /* -------------------------------------------------------------------- */
        /*      Setup no data values                                            */
        /* -------------------------------------------------------------------- */
        for ( int i = 0; i < psWarpOptions->nBandCount; i++ )
        {
          GDALRasterBandH rasterBand = GDALGetRasterBand( psWarpOptions->hSrcDS, psWarpOptions->panSrcBands[i] );

          int hasNoDataValue;
          double noDataValue = GDALGetRasterNoDataValue( rasterBand, &hasNoDataValue );

          if ( hasNoDataValue )
          {
            // Check if the nodata value is out of range
            int bClamped = FALSE;
            int bRounded = FALSE;
            CPL_IGNORE_RET_VAL(
              GDALAdjustValueToDataType( GDALGetRasterDataType( rasterBand ),
                                         noDataValue, &bClamped, &bRounded ) );
            if ( !bClamped )
            {
              GDALWarpInitNoDataReal( psWarpOptions, -1e10 );
              GDALSetRasterNoDataValue( GDALGetRasterBand(psWarpOptions->hDstDS, i+1), noDataValue);
              psWarpOptions->padfSrcNoDataReal[i] = noDataValue;
              psWarpOptions->padfDstNoDataReal[i] = noDataValue;
            }
          }
        }

        psWarpOptions->pTransformerArg = hTransformArg;
        psWarpOptions->papszWarpOptions =
                    CSLSetNameValue( psWarpOptions->papszWarpOptions,
                                    "INIT_DEST", "NO_DATA" );
        psWarpOptions->papszWarpOptions =
                    CSLSetNameValue( psWarpOptions->papszWarpOptions,
                                    "WRITE_FLUSH", "YES" );
        CPLFetchBool( psWarpOptions->papszWarpOptions, "OPTIMIZE_SIZE", true );

        psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

        // Initialize and execute the warp operation.
        eErr = GDALReprojectImage(rasterizeDS, pszProjection,
                                  hDstDS, pszDstWKT,
                                  GRA_Bilinear,
                                  0.0, 0.0,
                                  GDALTermProgress, nullptr,
                                  psWarpOptions);
        if (eErr != CE_None)
        {
            errorStr =  CPLGetLastErrorMsg();
            GDALDestroyGenImgProjTransformer( hTransformArg );
            GDALClose(rasterizeDS);
            GDALDestroyWarpOptions( psWarpOptions );
            CPLFree( pszProjection );
            return false;
        }

        GDALDestroyGenImgProjTransformer( hTransformArg );
        GDALDestroyWarpOptions( psWarpOptions );
        GDALClose( hDstDS );
        QFile::remove(QString::fromStdString(outputNoReprojStr));
    }

    GDALClose(rasterizeDS);
    CPLFree( pszProjection );
    return true;
}
