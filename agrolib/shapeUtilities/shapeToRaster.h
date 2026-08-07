#ifndef SHAPETORASTER_H
#define SHAPETORASTER_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    bool initializeRasterFromShape(const Crit3DShapeHandler &shapeHandler,
                               gis::Crit3DRasterGrid &newRaster, double cellSize);

    bool fillRasterWithShapeIndex(gis::Crit3DRasterGrid &raster, const Crit3DShapeHandler &shapeHandler);

    bool rasterizeShape(const gis::Crit3DRasterGrid *refRaster, gis::Crit3DRasterGrid &newRaster,
                        const Crit3DShapeHandler &shapeHandler, const std::string &fieldName, double cellSizeRef,
                        int sampleGrid, double coverageThreshold, bool useReferenceRaster);

#endif // SHAPETORASTER_H
