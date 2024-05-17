#ifndef SHAPETORASTER_H
#define SHAPETORASTER_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    bool initializeRasterFromShape(const Crit3DShapeHandler &shapeHandler, gis::Crit3DRasterGrid &raster, double cellSize);

    bool fillRasterWithShapeNumber(gis::Crit3DRasterGrid &raster, const Crit3DShapeHandler &shapeHandler);

    bool fillRasterWithField(gis::Crit3DRasterGrid &raster, Crit3DShapeHandler &shapeHandler, const std::string &valField);

    bool rasterizeShape(Crit3DShapeHandler &shapeHandler, gis::Crit3DRasterGrid &newRaster, const std::string &field, double cellSize);

    bool rasterizeShapeWithRef(const gis::Crit3DRasterGrid &refRaster, gis::Crit3DRasterGrid &newRaster,
                               Crit3DShapeHandler &shapeHandler, const std::string &fieldName);


#endif // SHAPETORASTER_H
