#ifndef SHAPETORASTER_H
#define SHAPETORASTER_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include "gis.h"

        gis::Crit3DRasterGrid *initializeRasterFromShape(Crit3DShapeHandler *shape, gis::Crit3DRasterGrid *raster, double cellSize);
        void fillRasterWithShapeNumber(gis::Crit3DRasterGrid* raster, Crit3DShapeHandler* shapePointer, bool showInfo);
        void fillRasterWithField(gis::Crit3DRasterGrid* raster, Crit3DShapeHandler* shape, std::string valField, bool showInfo);


#endif // SHAPETORASTER_H
