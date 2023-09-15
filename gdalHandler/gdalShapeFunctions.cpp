#include <QFileInfo>
#include <string.h>
#include <ogrsf_frmts.h>
#include "ogr_spatialref.h"
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <gdalwarper.h>

#include "gdalShapeFunctions.h"


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

    GDALDataset* shapeDataset = (GDALDataset*)GDALOpenEx(shapeFileName.toStdString().data(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if( shapeDataset == nullptr )
    {
        errorStr = "Open shapefile failed";
        return false;
    }

    // projection
    char *pszProjection = nullptr;
    OGRSpatialReference srs;
    OGRSpatialReference * pOrigSrs = shapeDataset->GetLayer(0)->GetSpatialRef();
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
        GDALClose( shapeDataset );
        errorStr = "Missing projection";
        return false;
    }

    QString fileNameRaster = file.absolutePath() + "/raster." + ext;
    QString fileNameProj = file.absolutePath() + "/proj." + ext;
    QString fileNameColor = file.absolutePath() + "/color." + ext;

    GDALDatasetH rasterDataset;
    GDALDatasetH projDataset;
    GDALDatasetH inputDataset;

    std::string res = resolution.toStdString();

    // set options
    char *options[] = {strdup("-at"), strdup("-of"), strdup(formatOption.c_str()), strdup("-a"), strdup(shapeField.toStdString().c_str()), strdup("-a_nodata"), strdup("-9999"),
                       strdup("-a_srs"), pszProjection, strdup("-tr"), strdup(res.c_str()), strdup(res.c_str()), strdup("-co"), strdup("COMPRESS=LZW"), nullptr};

    GDALRasterizeOptions *psOptions = GDALRasterizeOptionsNew(options, nullptr);
    if( psOptions == nullptr )
    {
        GDALClose(shapeDataset);
        errorStr = "psOptions is null";
        return false;
    }

    // rasterize
    rasterDataset = GDALRasterize(strdup(fileNameRaster.toStdString().c_str()), nullptr, shapeDataset, psOptions, &error);

    GDALClose(shapeDataset);
    GDALRasterizeOptionsFree(psOptions);

    if (rasterDataset == nullptr || error == 1)
    {
        CPLFree( pszProjection );
        return false;
    }

    // reprojection
    if (proj.isEmpty())
    {
        inputDataset = rasterDataset;
    }
    else
    {
        GDALDriverH hDriver;
        GDALDataType eDT;
        // Create output with same datatype as first input band.
        eDT = GDALGetRasterDataType(GDALGetRasterBand(rasterDataset,1));

        // Get output driver (GeoTIFF format)
        hDriver = GDALGetDriverByName( "GTiff" );
        if (hDriver == nullptr)
        {
            errorStr = "Error GDALGetDriverByName";
            GDALClose(rasterDataset);
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
            GDALCreateGenImgProjTransformer( rasterDataset, pszProjection, nullptr, pszDstWKT,
                                             FALSE, 0, 1 );
        if ( hTransformArg == nullptr )
        {
            errorStr = "Error GDALCreateGenImgProjTransformer";
            GDALClose(rasterDataset);
            CPLFree( pszProjection );
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
            errorStr = "Error GDALSuggestedWarpOutput";
            GDALClose(rasterDataset);
            CPLFree( pszProjection );
            return false;
        }

        char *createOptions[] = {strdup("COMPRESS=LZW"), nullptr};

        // Create the projected dataset
        projDataset = GDALCreate( hDriver, strdup(fileNameProj.toStdString().c_str()) , nPixels, nLines,
                             GDALGetRasterCount(rasterDataset), eDT, createOptions );

        if( projDataset == nullptr )
        {
            errorStr = "Error GDALCreate output reprojected";
            GDALClose(rasterDataset);
            CPLFree( pszProjection );
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
          double noDataValue = GDALGetRasterNoDataValue(rasterBand, &hasNoDataValue );

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
        eErr = GDALReprojectImage(rasterDataset, pszProjection,
                                  projDataset, pszDstWKT,
                                  GRA_Bilinear,
                                  0.0, 0.0,
                                  GDALTermProgress, nullptr,
                                  psWarpOptions);
        if (eErr != CE_None)
        {
            errorStr =  CPLGetLastErrorMsg();
            GDALDestroyGenImgProjTransformer( hTransformArg );
            GDALClose(rasterDataset);
            GDALDestroyWarpOptions( psWarpOptions );
            CPLFree( pszProjection );
            return false;
        }

        GDALDestroyGenImgProjTransformer( hTransformArg );
        GDALDestroyWarpOptions( psWarpOptions );

        inputDataset = projDataset;
        GDALClose( rasterDataset );
    }

    CPLFree( pszProjection );

    // save color map
    if (! paletteFileName.isEmpty())
    {
        GDALDatasetH colorDataset = GDALDEMProcessing(strdup(fileNameColor.toStdString().c_str()), inputDataset, "color-relief",
                                                      strdup(paletteFileName.toStdString().c_str()), nullptr, &error);

        if (colorDataset == nullptr || error == 1)
        {
            errorStr = "Error in GDALDEMProcessing.";
            GDALClose( inputDataset );
            return false;
        }

        GDALClose( colorDataset );
    }

    GDALClose( inputDataset );

    // rename output and remove temporary files
    QFile::remove(outputName);

    if (! paletteFileName.isEmpty())
    {
        QFile::rename(fileNameColor, outputName);
    }
    else if(! proj.isEmpty())
    {
        QFile::rename(fileNameProj, outputName);
    }
    else
    {
        QFile::rename(fileNameRaster, outputName);
    }

    QFile::remove(fileNameRaster);
    QFile::remove(fileNameProj);
    QFile::remove(fileNameColor);

    return true;
}
