#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "gis.h"
#include "shapeToRaster.h"
#include "shapeHandler.h"


bool initializeRasterFromShape(const Crit3DShapeHandler &shapeHandler, gis::Crit3DRasterGrid &newRaster, double cellSize)
{
    const int nrShape = shapeHandler.getShapeCount();

    // check void shapefile
    if (nrShape <= 0) 
        return false;

    // ceck cellsize
    if (cellSize <= 0 || isEqual(cellSize, NODATA))
        return false;

    double xmin, xmax, ymin, ymax;
    if (! shapeHandler.getBounds(xmin, xmax, ymin, ymax))
        return false;

    // ensures that the limits are included
    const double x0 = floor(xmin / cellSize) * cellSize;
    const double y0 = floor(ymin / cellSize) * cellSize;

    const double x1 = ceil(xmax / cellSize) * cellSize;
    const double y1 = ceil(ymax / cellSize) * cellSize;

    gis::Crit3DRasterHeader header;

    header.cellSize = cellSize;
    header.invCellSize = 1.0 / cellSize;
    header.llCorner.x = x0;
    header.llCorner.y = y0;

    header.nrRows = static_cast<int>(std::round((y1 - y0) / cellSize));
    header.nrCols = static_cast<int>(std::round((x1 - x0) / cellSize));

    if (header.nrRows <= 0 || header.nrCols <= 0)
        return false;

    header.flag = NODATA;

    return newRaster.initializeGrid(header);
}


bool fillRasterWithShapeIndex(gis::Crit3DRasterGrid &raster, const Crit3DShapeHandler &shapeHandler)
{
    int nrShapes = shapeHandler.getShapeCount();
    if (nrShapes <= 0)
        return false;

    const auto& header = raster.header;

    raster.emptyGrid();

    for (int shapeIndex = 0; shapeIndex < nrShapes; ++shapeIndex)
    {
        ShapeObject object;
        shapeHandler.getShape(shapeIndex, object);

        const auto& bounds = object.getBounds();

        int r0, r1, c0, c1;
        gis::getRowColFromXY(header, bounds.xmin, bounds.ymax, &r0, &c0);
        gis::getRowColFromXY(header, bounds.xmax, bounds.ymin, &r1, &c1);

        // bounds out of raster
        if (r1 < 0 || r0 >= header.nrRows ||
            c1 < 0 || c0 >= header.nrCols)
            continue;

        r0 = std::max(r0 - 1, 0);
        r1 = std::min(r1 + 1, header.nrRows - 1);
        c0 = std::max(c0 - 1, 0);
        c1 = std::min(c1 + 1, header.nrCols - 1);

        #pragma omp parallel for if (r1 - r0 > 30)
        for (int row = r0; row <= r1; ++row)
        {
            const double y = header.llCorner.y + (header.nrRows - row - 0.5) * header.cellSize;
            const double x0 = header.llCorner.x + (c0 + 0.5) * header.cellSize;

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
        if (refRaster == nullptr)
            return false;
        if (! newRaster.initializeGrid(refRaster->header))
            return false;
    }
    else
    {
        if (! initializeRasterFromShape(shapeHandler, newRaster, cellSizeRef))
            return false;
    }

    const auto& header = newRaster.header;
    const float flag = header.flag;

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
    const int halfSamples = (nrSamples + 1) / 2;

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
    const size_t nrCells = static_cast<size_t>(nrRows) * static_cast<size_t>(nrCols);
    std::vector<uint8_t> coverageCount(nrCells, 0);

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

        const auto& bounds = object.getBounds();

        gis::getRowColFromXY(header, bounds.xmin, bounds.ymax, &r0, &c0);
        gis::getRowColFromXY(header, bounds.xmax, bounds.ymin, &r1, &c1);

        // bounds out of raster
        if (r1 < 0 || r0 >= nrRows ||
            c1 < 0 || c0 >= nrCols)
            continue;

        r0 = std::max(r0 - 1, 0);
        r1 = std::min(r1 + 1, nrRows - 1);
        c0 = std::max(c0 - 1, 0);
        c1 = std::min(c1 + 1, nrCols - 1);

        #pragma omp parallel for if (r1 - r0 > 30)
        for (int row = r0; row <= r1; ++row)
        {
            const double yc = header.llCorner.y + (header.nrRows - row - 0.5) * header.cellSize;

            for (int col = c0; col <= c1; ++col)
            {
                if (useReferenceRaster)
                {
                    // skip empty cells of reference raster
                    if (isEqual(refRaster->value[row][col], flag))
                        continue;
                }

                const size_t index = static_cast<size_t>(row) * nrCols + col;

                // just covered
                const int currentCoverage = coverageCount[index];
                if (currentCoverage >= halfSamples)
                    continue;

                const double xc = header.llCorner.x + (col + 0.5) * header.cellSize;

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

                    if (inside + remaining <= currentCoverage)
                        break;
                }

                // check threshold
                if (inside < minPoints)
                    continue;

                // check and store cell coverage
                if (inside > currentCoverage)
                {
                    coverageCount[index] = inside;
                    newRaster.value[row][col] = static_cast<float>(fieldValue);
                }
            }
        }
    }

    return true;
}

