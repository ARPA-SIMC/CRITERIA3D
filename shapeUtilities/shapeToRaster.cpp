#include <float.h>
#include <math.h>
#include <limits>

#include "commonConstants.h"
#include "basicMath.h"
#include "gis.h"
#include "shapeToRaster.h"
#include "shapeHandler.h"


bool initializeRasterFromShape(const Crit3DShapeHandler &shapeHandler, gis::Crit3DRasterGrid &newRaster, double cellSize)
{
    int nrShape = shapeHandler.getShapeCount();

    // check void shapefile
    if (nrShape <= 0) 
        return false;

    // ceck cellsize
    if (cellSize <= 0 || isEqual(cellSize, NODATA))
        return false;

    gis::Crit3DRasterHeader header;
    ShapeObject object;

    double ymin =  std::numeric_limits<double>::max();
    double xmin =  std::numeric_limits<double>::max();
    double ymax =  std::numeric_limits<double>::lowest();
    double xmax =  std::numeric_limits<double>::lowest();

    for (int i = 0; i < nrShape; i++)
    {
        shapeHandler.getShape(i, object);
        const auto bounds = object.getBounds();

        xmin = std::min(xmin, bounds.xmin);
        ymin = std::min(ymin, bounds.ymin);
        xmax = std::max(xmax, bounds.xmax);
        ymax = std::max(ymax, bounds.ymax);
    }

    // ensures that the limits are included
    xmin = floor(xmin / cellSize) * cellSize;
    ymin = floor(ymin / cellSize) * cellSize;

    xmax = ceil(xmax / cellSize) * cellSize;
    ymax = ceil(ymax / cellSize) * cellSize;

    header.cellSize = cellSize;
    header.invCellSize = 1.0 / cellSize;
    header.llCorner.x = xmin;
    header.llCorner.y = ymin;
    header.nrRows = static_cast<int>(std::ceil((ymax - ymin) / cellSize));
    header.nrCols = static_cast<int>(std::ceil((xmax - xmin) / cellSize));
    header.flag = NODATA;

    return newRaster.initializeGrid(header);
}


bool fillRasterWithShapeIndex(gis::Crit3DRasterGrid &raster, const Crit3DShapeHandler &shapeHandler)
{
    int nrShapes = shapeHandler.getShapeCount();
    if (nrShapes <= 0)
        return false;

    const auto& header = *(raster.header);

    raster.emptyGrid();

    ShapeObject object;
    for (int shapeIndex = 0; shapeIndex < nrShapes; ++shapeIndex)
    {
        shapeHandler.getShape(shapeIndex, object);

        Box<double> bounds = object.getBounds();

        int r0, r1, c0, c1;
        gis::getRowColFromXY(header, bounds.xmin, bounds.ymax, &r0, &c0);
        gis::getRowColFromXY(header, bounds.xmax, bounds.ymin, &r1, &c1);

        // bounds out of raster
        if (r1 < 0 || r0 >= header.nrRows ||
            c1 < 0 || c0 >= header.nrCols)
            continue;

        r0 = MAXVALUE(r0 - 1, 0);
        r1 = MINVALUE(r1 + 1, header.nrRows - 1);
        c0 = MAXVALUE(c0 - 1, 0);
        c1 = MINVALUE(c1 + 1, header.nrCols - 1);

        for (int row = r0; row <= r1; ++row)
        {
            double y = header.llCorner.y + (header.nrRows - row - 0.5) * header.cellSize;
            double x0 = header.llCorner.x + (c0 + 0.5) * header.cellSize;

            double x = x0;
            for (int col = c0; col <= c1; ++col)
            {
                if (isEqual(raster.value[row][col], header.flag))
                {
                    if (object.pointInPolygon(x, y))
                    {
                        raster.value[row][col] = static_cast<float>(shapeIndex);
                    }
                }
                x += header.cellSize;
            }
        }
    }

    return true;
}


bool rasterizeShape(const gis::Crit3DRasterGrid *refRaster, gis::Crit3DRasterGrid &newRaster,
                    const Crit3DShapeHandler &shapeHandler, const std::string &fieldName, double cellSizeRef = NODATA,
                    int sampleGrid = 1, double coverageThreshold = 0.5, bool useReferenceRaster = true)
{
    if (useReferenceRaster)
    {
        if (! newRaster.initializeGrid(*(refRaster->header)))
            return false;
    }
    else
    {
        if (! initializeRasterFromShape(shapeHandler, newRaster, cellSizeRef))
            return false;
    }

    const auto& header = *newRaster.header;

    const int nrShape = shapeHandler.getShapeCount();
    if (nrShape <= 0)
        return false;

    const bool isShapeId = (fieldName == "Shape ID");

    int fieldIndex = -1;
    if (! isShapeId)
    {
        fieldIndex = shapeHandler.getDBFFieldIndex(fieldName.c_str());
        if (fieldIndex < 0)
            return false;
    }

    // check nr of samples (1-10) and coverage threshold (0-1)
    sampleGrid = std::max(1, std::min(10, sampleGrid));
    coverageThreshold = std::max(0.0, std::min(1.0, coverageThreshold));

    const int nrSamples = sampleGrid * sampleGrid;
    const int minPoints = std::max(1, static_cast<int>(std::ceil(coverageThreshold * nrSamples)));

    const int nrRows = header.nrRows;
    const int nrCols = header.nrCols;
    const double cellSize = header.cellSize;

    // point offset (one time)
    std::vector<std::pair<double,double>> offsets;
    offsets.resize(nrSamples);

    int k = 0;
    for (int iy = 0; iy < sampleGrid; ++iy)
    {
        for (int ix = 0; ix < sampleGrid; ++ix)
        {
            offsets[k++] =
                {
                    (-0.5 + (ix + 0.5) / sampleGrid) * cellSize,
                    ( 0.5 - (iy + 0.5) / sampleGrid) * cellSize
                };
        }
    }

    // tmp coverage raster
    std::vector<uint8_t> coverageCount(nrRows * nrCols, 0);

    for (int shapeIndex = 0; shapeIndex < nrShape; ++shapeIndex)
    {
        ShapeObject object;
        shapeHandler.getShape(shapeIndex, object);

        const double fieldValue = isShapeId ?
                                      shapeIndex :
                                      shapeHandler.getNumericValue(shapeIndex, fieldIndex);

        if (isEqual(fieldValue, NODATA))
            continue;

        int r0, r1, c0, c1;

        const Box<double> bounds = object.getBounds();

        gis::getRowColFromXY(header, bounds.xmin, bounds.ymax, &r0, &c0);
        gis::getRowColFromXY(header, bounds.xmax, bounds.ymin, &r1, &c1);

        // bounds out of raster
        if (r1 < 0 || r0 >= nrRows ||
            c1 < 0 || c0 >= nrCols)
            continue;

        r0 = MAXVALUE(r0 - 1, 0);
        r1 = MINVALUE(r1 + 1, nrRows - 1);
        c0 = MAXVALUE(c0 - 1, 0);
        c1 = MINVALUE(c1 + 1, nrCols - 1);

        for (int row = r0; row <= r1; ++row)
        {
            for (int col = c0; col <= c1; ++col)
            {
                if (useReferenceRaster)
                {
                    // skip empty cells of reference raster
                    if (isEqual(refRaster->value[row][col], header.flag))
                        continue;
                }

                double xc, yc;
                newRaster.getXY(row, col, xc, yc);

                uint8_t inside = 0;
                int remaining = nrSamples;

                for (const auto &p : offsets)
                {
                    --remaining;

                    if (object.pointInPolygon(xc + p.first,
                                              yc + p.second))
                        ++inside;

                    // skip unnecessary steps
                    if (inside + remaining < minPoints)
                        break;
                }

                // check threshold
                if (inside < minPoints)
                    continue;

                // check and store cell coverage
                const int index = row * nrCols + col;

                if (inside > coverageCount[index])
                {
                    coverageCount[index] = inside;
                    newRaster.value[row][col] = static_cast<float>(fieldValue);
                }
            }
        }
    }

    return true;
}


/*
bool rasterizeShapeWithRef_old(const gis::Crit3DRasterGrid &refRaster, gis::Crit3DRasterGrid &newRaster,
                             Crit3DShapeHandler &shapeHandler, const std::string &fieldName)
{
    newRaster.initializeGrid(*(refRaster.header));

    const int nrShape = shapeHandler.getShapeCount();

    // check void shapefile
    if (nrShape <= 0)
        return false;

    const bool isShapeId = (fieldName == "Shape ID");

    int fieldIndex = -1;
    if (! isShapeId)
    {
        fieldIndex = shapeHandler.getDBFFieldIndex(fieldName.c_str());
        // check field
        if (fieldIndex < 0)
            return false;
    }

    const int nrRows = newRaster.header->nrRows;
    const int nrCols = newRaster.header->nrCols;

    for (int shapeIndex = 0; shapeIndex < nrShape; ++shapeIndex)
    {
        ShapeObject object;
        shapeHandler.getShape(shapeIndex, object);

        const double fieldValue = isShapeId
                                ? shapeIndex
                                : shapeHandler.getNumericValue(shapeIndex, fieldIndex);

        if (! isEqual(fieldValue, NODATA))
        {
            int r0, r1, c0, c1;

            // get bounds
            Box<double> bounds = object.getBounds();

            gis::getRowColFromXY(*(newRaster.header), bounds.xmin, bounds.ymax, &r0, &c0);
            gis::getRowColFromXY(*(newRaster.header), bounds.xmax, bounds.ymin, &r1, &c1);

            // check bounds
            r0 = MAXVALUE(r0-1, 0);
            r1 = MINVALUE(r1+1, nrRows-1);
            c0 = MAXVALUE(c0-1, 0);
            c1 = MINVALUE(c1+1, nrCols-1);

            for (int row = r0; row <= r1; ++row)
            {
                for (int col = c0; col <= c1; ++col)
                {
                    if (! isEqual(refRaster.value[row][col], refRaster.header->flag))
                    {
                        double x, y;
                        newRaster.getXY(row, col, x, y);
                        if (object.pointInPolygon(x, y))
                        {
                            newRaster.value[row][col] = static_cast<float>(fieldValue);
                        }
                    }
                }
            }
        }
    }

    return true;
}
*/

/*
bool fillRasterWithField(gis::Crit3DRasterGrid &raster, Crit3DShapeHandler &shapeHandler, const std::string &fieldName)
{
    int nrShape = shapeHandler.getShapeCount();
    if (nrShape <= 0)
    {
        // void shapefile
        return false;
    }

    ShapeObject object;
    double x, y, fieldValue;
    Box<double> bounds;
    int r0, r1, c0, c1;

    int fieldIndex = shapeHandler.getDBFFieldIndex(fieldName.c_str());

    for (int shapeIndex = 0; shapeIndex < nrShape; shapeIndex++)
    {
        shapeHandler.getShape(shapeIndex, object);

        fieldValue = shapeHandler.getNumericValue(shapeIndex, fieldIndex);

        if (! isEqual(fieldValue, NODATA))
        {
            // get bounds
            bounds = object.getBounds();
            gis::getRowColFromXY(*(raster.header), bounds.xmin, bounds.ymax, &r0, &c0);
            gis::getRowColFromXY(*(raster.header), bounds.xmax, bounds.ymin, &r1, &c1);

            // check bounds
            r0 = MAXVALUE(r0-1, 0);
            r1 = MINVALUE(r1+1, raster.header->nrRows -1);
            c0 = MAXVALUE(c0-1, 0);
            c1 = MINVALUE(c1+1, raster.header->nrCols -1);

            for (int row = r0; row <= r1; row++)
            {
                for (int col = c0; col <= c1; col++)
                {
                    if (isEqual(raster.value[row][col], raster.header->flag))
                    {
                        raster.getXY(row, col, x, y);
                        if (object.pointInPolygon(x, y))
                        {
                            raster.value[row][col] = float(fieldValue);
                        }
                    }
                }
            }
        }
    }

    return true;
}


bool rasterizeShape_old(Crit3DShapeHandler &shapeHandler, gis::Crit3DRasterGrid &newRaster,
                    const std::string &field, double cellSize)
{
    if (! initializeRasterFromShape(shapeHandler, newRaster, cellSize))
        return false;

    if (field == "Shape ID")
    {
        if (! fillRasterWithShapeNumber(newRaster, shapeHandler))
            return false;
    }
    else
    {
        if (! fillRasterWithField(newRaster, shapeHandler, field))
            return false;
    }

    return true;
}
*/
