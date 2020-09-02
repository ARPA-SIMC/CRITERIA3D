#ifndef GDALSHAPEFUNCTIONS_H
#define GDALSHAPEFUNCTIONS_H

#ifndef SHAPEHANDLER_H
    #include "shapeHandler.h"
#endif
#include <QString>
#include <QMap>
#include <geos_c.h>

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

bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, QString ucmFileName, std::string *error, bool showInfo);


bool shapeIntersection(Crit3DShapeHandler *first, Crit3DShapeHandler *second, GEOSGeometry **inteserctionGeom);
bool getShapeFromGeom(GEOSGeometry *inteserctionGeom, Crit3DShapeHandler *ucm);
GEOSGeometry *loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler);
bool shapeToRaster(QString shapeFileName, std::string shapeField, QString resolution, QString outputName, QString &errorStr);

//GEOSGeometry * SHPObject_to_GeosPolygon_NoHoles(SHPObject *object);
//GEOSGeometry *load_shapefile_as_collection(char *pathname);
//GEOSGeometry * testIntersection();

#endif // GDALSHAPEFUNCTIONS_H
