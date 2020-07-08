#ifndef UNITCROPMAP_H
#define UNITCROPMAP_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>
    #include <geos_c.h>

    bool computeUcmPrevailing(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                        std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize, QString ucmFileName,
                        std::string *error, bool showInfo);
    bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                     std::string idCrop, std::string idSoil, std::string idMeteo,
                     QString ucmFileName, std::string *error, bool showInfo);
    bool fillIDCase(Crit3DShapeHandler *ucm, std::string idCrop, std::string idSoil, std::string idMeteo);

    bool shapeIntersection(Crit3DShapeHandler *first, Crit3DShapeHandler *second, GEOSGeometry **inteserctionGeom);

    bool getShapeFromGeom(GEOSGeometry *inteserctionGeom, Crit3DShapeHandler *ucm);


#endif // UNITCROPMAP_H
