#ifndef GDALSHAPEFUNCTIONS_H
#define GDALSHAPEFUNCTIONS_H

    #include <QString>
    #include <QMap>

    static QMap<QString, QString> mapExtensionShortName
    {
        {"vrt", "VTR"},
        {"tif", "GTiff"},
        {"tiff", "GTiff"},
        {"ntf", "NITF"},
        {"img", "HFA"},
        {"bmp", "BMP"},
        {"pix", "PCIDSK"},
        {"map", "PCRaster"},
        {"rgb", "SGI"},
        {"xml", "PDS4"},
        {"ers", "ERS"},
        {"rsw", "RMF"},
        {"rst", "RST"},
        {"pdf", "PDF"},
        {"mbtiles", "MBTiles"},
        {"mrf", "MRF"},
        {"hdr", "MFF"},
        {"kro", "KRO"},
        {"gen", "ADRG"},
        {"gpkg", "GPKG"},
        {"bil", "EHdr"}
    };

    bool gdalShapeToRaster(QString shapeFileName, QString shapeField, QString resolution, QString mapProjection,
                            QString outputName, QString paletteFileName, bool isPngCopy, QString pngFileName,
                            QString pngProjection, QString &errorStr);


#endif // GDALSHAPEFUNCTIONS_H
