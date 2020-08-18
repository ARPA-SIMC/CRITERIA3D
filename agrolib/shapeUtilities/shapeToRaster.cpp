#include "shapeToRaster.h"

#include "commonConstants.h"
#include <float.h>
#include <math.h>
#include "formInfo.h"
#include "gis.h"


gis::Crit3DRasterGrid* initializeRasterFromShape(Crit3DShapeHandler* shape, gis::Crit3DRasterGrid* raster, double cellSize)
{
    gis::Crit3DRasterHeader header;
    ShapeObject object;
    Box<double> bounds;

    double ymin = DBL_MAX;
    double xmin = DBL_MAX;
    double ymax = DBL_MIN;
    double xmax = DBL_MIN;

    for (int i = 0; i < shape->getShapeCount(); i++)
    {
        shape->getShape(i, object);
        bounds = object.getBounds();
        ymin = MINVALUE(ymin, bounds.ymin);
        xmin = MINVALUE(xmin, bounds.xmin);
        ymax = MAXVALUE(ymax, bounds.ymax);
        xmax = MAXVALUE(xmax, bounds.xmax);
    }

    xmin = floor(xmin);
    ymin = floor(ymin);
    header.cellSize = cellSize;
    header.llCorner.x = xmin;
    header.llCorner.y = ymin;
    header.nrRows = int(floor((ymax - ymin) / cellSize))+1;
    header.nrCols = int(floor((xmax - xmin) / cellSize))+1;

    raster->initializeGrid(header);
    return raster;
}


void fillRasterWithShapeNumber(gis::Crit3DRasterGrid* raster, Crit3DShapeHandler *shapeHandler, bool showInfo)
{
    ShapeObject object;
    FormInfo formInfo;
    double x, y;
    Box<double> bounds;
    int r0, r1, c0, c1;
    int nShape = shapeHandler->getShapeCount();

    QString fileName = QString::fromStdString(shapeHandler->getFilepath());

    if (showInfo)
    {
        formInfo.start("Rasterize shape " + fileName, nShape);
    }

    raster->emptyGrid();

    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        if (showInfo) formInfo.setValue(shapeIndex);

        shapeHandler->getShape(shapeIndex, object);

        // get bounds
        bounds = object.getBounds();
        gis::getRowColFromXY(*(raster->header), bounds.xmin, bounds.ymax, &r0, &c0);
        gis::getRowColFromXY(*(raster->header), bounds.xmax, bounds.ymin, &r1, &c1);
        // check bounds
        r0 = MAXVALUE(r0-1, 0);
        r1 = MINVALUE(r1+1, raster->header->nrRows -1);
        c0 = MAXVALUE(c0-1, 0);
        c1 = MINVALUE(c1+1, raster->header->nrCols -1);

        for (int row = r0; row <= r1; row++)
        {
            for (int col = c0; col <= c1; col++)
            {
                if (raster->value[row][col] == raster->header->flag)
                {
                    raster->getXY(row, col, &x, &y);
                    if (object.pointInPolygon(x, y))
                    {
                        raster->value[row][col] = shapeIndex;
                    }
                }
            }
        }
    }

    if (showInfo) formInfo.close();
}


void fillRasterWithField(gis::Crit3DRasterGrid* raster, Crit3DShapeHandler* shapeHandler, std::string valField, bool showInfo)
{
    ShapeObject object;
    FormInfo formInfo;
    double x, y, fieldValue;
    int fieldIndex = shapeHandler->getDBFFieldIndex(valField.c_str());
    int nShape = shapeHandler->getShapeCount();
    DBFFieldType fieldType = shapeHandler->getFieldType(fieldIndex);
    Box<double> bounds;
    int r0, r1, c0, c1;

    if (showInfo)
    {
        formInfo.start("Rasterize shape...", nShape);
    }

    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        if (showInfo) formInfo.setValue(shapeIndex);

        shapeHandler->getShape(shapeIndex, object);

        fieldValue = NODATA;
        if (fieldType == FTInteger)
        {
            fieldValue = shapeHandler->readIntAttribute(shapeIndex, fieldIndex);
        }
        else if (fieldType == FTDouble)
        {
            fieldValue = shapeHandler->readDoubleAttribute(shapeIndex, fieldIndex);
        }

        if (fieldValue != NODATA)
        {
            // get bounds
            bounds = object.getBounds();
            gis::getRowColFromXY(*(raster->header), bounds.xmin, bounds.ymax, &r0, &c0);
            gis::getRowColFromXY(*(raster->header), bounds.xmax, bounds.ymin, &r1, &c1);
            // check bounds
            r0 = MAXVALUE(r0-1, 0);
            r1 = MINVALUE(r1+1, raster->header->nrRows -1);
            c0 = MAXVALUE(c0-1, 0);
            c1 = MINVALUE(c1+1, raster->header->nrCols -1);

            for (int row = r0; row <= r1; row++)
            {
                for (int col = c0; col <= c1; col++)
                {
                    if (raster->value[row][col] == raster->header->flag)
                    {
                        raster->getXY(row, col, &x, &y);
                        if (object.pointInPolygon(x, y))
                        {
                            raster->value[row][col] = float(fieldValue);
                        }
                    }
                }
            }
        }
    }

    if (showInfo) formInfo.close();
}


// TODO funzione fillRasterWithPrevailingShapeNumber con valore prevalente (nrSubdivision, minPercentage)
// shift = cellsize / (nrSubdivision +1)
// x primo punto = utmPoint.x - cellsize/2 + shift
