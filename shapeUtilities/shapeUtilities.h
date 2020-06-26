#ifndef SHAPEUTILITIES_H
#define SHAPEUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>
    #include <geos_c.h>

    QString cloneShapeFile(QString refFileName, QString newFileName);
    bool cleanShapeFile(Crit3DShapeHandler *shapeHandler);
    GEOSGeometry *loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler);
    //GEOSGeometry * SHPObject_to_GeosPolygon_NoHoles(SHPObject *object);
   // GEOSGeometry *load_shapefile_as_collection(char *pathname);
    //GEOSGeometry * testIntersection();

#endif // SHAPEUTILITIES_H
