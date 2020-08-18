#ifndef ZONALSTATISTIC_H
#define ZONALSTATISTIC_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    enum opType{MAJORITY, MIN, MAX, AVG};

    bool zonalStatisticsShape(Crit3DShapeHandler* shapeRef, Crit3DShapeHandler* shapeVal, gis::Crit3DRasterGrid* rasterRef,
                                  gis::Crit3DRasterGrid* rasterVal, std::string valField, std::string valFieldOutput,
                                  opType type, std::string *error, bool showInfo);

#endif // ZONALSTATISTIC_H
