#include <QFileInfo>
#include <string.h>
#include <ogrsf_frmts.h>
#include "ogr_spatialref.h"
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <QDebug>

#include "gdalShapeFunctions.h"
#include "gdalRasterFunctions.h"


bool gdalShapeToRaster(QString shapeFileName, QString shapeField, QString resolution,
                       QString mapProjection, QString outputName, QString paletteFileName,
                       bool isPngCopy, QString pngFileName, QString pngProjection, QString &errorStr)
{
    GDALAllRegister();
    QFileInfo outputFile(outputName);
    QString ext = outputFile.completeSuffix();

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

    QString fileNameRaster = outputFile.absolutePath() + "/raster." + ext;
    QString fileNameColor = outputFile.absolutePath() + "/color." + ext;

    GDALDatasetH rasterDataset;
    GDALDatasetH inputDataset;

    std::string res = resolution.toStdString();

    // set options
    char *options[] = {strdup("-at"), strdup("-of"), strdup(formatOption.c_str()), strdup("-a"), strdup(shapeField.toStdString().c_str()), strdup("-a_nodata"), strdup("-9999"),
                       strdup("-a_srs"), pszProjection, strdup("-tr"), strdup(res.c_str()), strdup(res.c_str()), strdup("-co"), strdup("COMPRESS=LZW"), nullptr};

    GDALRasterizeOptions *psOptions = GDALRasterizeOptionsNew(options, nullptr);
    if( psOptions == nullptr )
    {
        GDALClose(shapeDataset);
        errorStr = "Error in GDALRasterizeOptionsNew";
        return false;
    }

    // rasterize
    int error = 0;
    rasterDataset = GDALRasterize(strdup(fileNameRaster.toStdString().c_str()), nullptr, shapeDataset, psOptions, &error);

    GDALClose(shapeDataset);
    GDALRasterizeOptionsFree(psOptions);

    if (rasterDataset == nullptr || error != 0)
    {
        errorStr = "Error in GDALRasterize";
        CPLFree( pszProjection );
        return false;
    }

    // re-projection
    QString fileNameProj = outputFile.absolutePath() + "/proj." + ext;
    if (! mapProjection.isEmpty())
    {
        bool isProjOk = gdalReprojection(rasterDataset, inputDataset, mapProjection, fileNameProj, errorStr);
        GDALClose( rasterDataset );
        CPLFree( pszProjection );

        if (! isProjOk)
            return false;
    }
    else
    {
        inputDataset = rasterDataset;
        CPLFree( pszProjection );
    }

    // create color map
    if (! paletteFileName.isEmpty())
    {
        char *optionForDEM[] = {const_cast<char *>("-alpha"), nullptr};
        GDALDEMProcessingOptions *psOptions = GDALDEMProcessingOptionsNew(optionForDEM, nullptr);

        GDALDatasetH colorDataset = GDALDEMProcessing(strdup(fileNameColor.toStdString().c_str()), inputDataset, "color-relief",
                                                      strdup(paletteFileName.toStdString().c_str()), psOptions, &error);
        if (colorDataset == nullptr || error != 0)
        {
            errorStr = "Error in coloring map (GDALDEMProcessing).";
            GDALClose( inputDataset );
            return false;
        }

        GDALClose( inputDataset );
        inputDataset = colorDataset;
    }

    // PNG copy
    if (isPngCopy)
    {
        if (! gdalExportPng(inputDataset, pngFileName, pngProjection, errorStr))
        {
            qDebug() << "ERROR: failed to write" << pngFileName;
            qDebug() << errorStr;
            errorStr = "";
        }
    }

    GDALClose( inputDataset );

    // rename output and remove temporary files
    QFile::remove(outputName);

    if (! paletteFileName.isEmpty())
    {
        QFile::rename(fileNameColor, outputName);
    }
    else if(! mapProjection.isEmpty())
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
