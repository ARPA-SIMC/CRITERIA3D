#ifndef GDALSHAPEFUNCTIONS_H
#define GDALSHAPEFUNCTIONS_H

#ifndef SHAPEHANDLER_H
    #include "shapeHandler.h"
#endif
#include <QString>
#include <geos_c.h>


bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, QString ucmFileName, std::string *error, bool showInfo);


bool shapeIntersection(Crit3DShapeHandler *first, Crit3DShapeHandler *second, GEOSGeometry **inteserctionGeom);
bool getShapeFromGeom(GEOSGeometry *inteserctionGeom, Crit3DShapeHandler *ucm);
GEOSGeometry *loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler);
bool shapeToGeoTIFF(QString shapeFileName, std::string shapeField, QString resolution, QString geoTIFFName, std::string* errorStr);

//GEOSGeometry * SHPObject_to_GeosPolygon_NoHoles(SHPObject *object);
//GEOSGeometry *load_shapefile_as_collection(char *pathname);
//GEOSGeometry * testIntersection();

#endif // GDALSHAPEFUNCTIONS_H
