#ifndef GDALSHAPEFUNCTIONS_H
#define GDALSHAPEFUNCTIONS_H

#ifndef SHAPEHANDLER_H
    #include "shapeHandler.h"
#endif
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

bool shapeToRaster(QString shapeFileName, QString shapeField, QString resolution, QString proj,
                   QString outputName, QString paletteFileName, QString &errorStr);


#endif // GDALSHAPEFUNCTIONS_H
