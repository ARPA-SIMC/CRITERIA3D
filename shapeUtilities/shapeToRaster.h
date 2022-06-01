#ifndef SHAPETORASTER_H
#define SHAPETORASTER_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    bool initializeRasterFromShape(Crit3DShapeHandler &shape, gis::Crit3DRasterGrid &raster, double cellSize);
    bool fillRasterWithShapeNumber(gis::Crit3DRasterGrid &raster, Crit3DShapeHandler &shape);
    bool fillRasterWithField(gis::Crit3DRasterGrid &raster, Crit3DShapeHandler &shape, std::string valField);
    bool rasterizeShape(Crit3DShapeHandler &shape, gis::Crit3DRasterGrid &newRaster, std::string field, double cellSize);


#endif // SHAPETORASTER_H
