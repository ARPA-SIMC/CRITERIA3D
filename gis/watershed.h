#ifndef WATERSHED_H
#define WATERSHED_H

    #include "gis.h"

    struct D8Cell
    {
        int row = -1;
        int col = -1;

        bool isValid() const
        {
            return row >= 0 && col >= 0;
        }
    };

    struct Cell
    {
        int row;
        int col;
    };


    namespace gis
    {
        static D8Cell computeFlowDirectionD8(const Crit3DRasterGrid& dem, int row, int col);

        bool cleanBasin(const Crit3DRasterGrid& dem, Crit3DRasterGrid& basinMask,
                        double xClosure, double yClosure);

        void cleanBasin_simple(const Crit3DRasterGrid& dem, Crit3DRasterGrid& outputRaster,
                               double xClosure, double yClosure);

        bool extractBasin(const Crit3DRasterGrid& dem, Crit3DRasterGrid& outputRaster,
                        double xClosure, double yClosure);
    }


#endif // WATERSHED_H
