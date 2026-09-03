/*******************************************************************
 * This code is based on shapeobject.cpp of Erik Svensson
 * https://github.com/blueluna/shapes
 *
 * Copyright (c) 2012 Erik Svensson
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:

 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 *******************************************************************/

#include <string.h>
#include "shapeObject.h"
#include "commonConstants.h"
#include "basicMath.h"


unsigned int ShapeObject::getPartCount() const
{
    return partCount;
}

ShapeObject::ShapeObject()
    : index(-1), type(0), vertexCount(0), vertices(nullptr) {}

ShapeObject::ShapeObject(const SHPObject* obj)
    : index(-1), type(0), vertexCount(0), vertices(nullptr)
{
    assign(obj);
}

ShapeObject::ShapeObject(const ShapeObject &other)
    : index(-1), type(0), vertexCount(0), vertices(nullptr)
{
    assign(other);
}

ShapeObject::~ShapeObject()
{
    destroy();
}

ShapeObject& ShapeObject::operator = (const ShapeObject& other)
{
    assign(other);
    return *this;
}

void ShapeObject::destroy()
{
    if (vertexCount > 0) {
        delete [] vertices;
        vertices = nullptr;
        vertexCount = 0;
        partCount = 0;
    }
    parts.clear();

    type = 0;
    index = -1;
}


void ShapeObject::assign(const SHPObject *obj)
{
    if (obj == nullptr)
        return;

    if (index >= 0)
        destroy();

    index = obj->nShapeId;
    type = obj->nSHPType;
    vertexCount = static_cast<unsigned int>(obj->nVertices);

    // Copy vertices
    if (vertexCount > 0)
    {
        vertices = new Point<double>[vertexCount];

        for (unsigned int i = 0; i < vertexCount; ++i)
        {
            vertices[i].set(obj->padfX[i], obj->padfY[i]);
        }
    }

    // Shape bounds
    bounds.xmin = obj->dfXMin;
    bounds.xmax = obj->dfXMax;
    bounds.ymin = obj->dfYMin;
    bounds.ymax = obj->dfYMax;

    // Parts
    partCount = static_cast<unsigned int>(obj->nParts);
    parts.reserve(partCount);

    for (unsigned int partIndex = 0; partIndex < partCount; ++partIndex)
    {
        Part part;

        part.type = obj->panPartType[partIndex];
        part.offset = static_cast<unsigned int>(
            obj->panPartStart[partIndex]);

        if (partIndex + 1 < partCount)
        {
            part.length = static_cast<unsigned int>(
                obj->panPartStart[partIndex + 1] -
                obj->panPartStart[partIndex]);
        }
        else
        {
            part.length = vertexCount - part.offset;
        }

        // According to the Shapefile polygon convention:
        // clockwise rings are exterior rings,
        // counter-clockwise rings are holes.
        part.hole = !isClockWise(part);

        // Initialize part bounds
        part.boundsPart.xmin = bounds.xmax;
        part.boundsPart.xmax = bounds.xmin;
        part.boundsPart.ymin = bounds.ymax;
        part.boundsPart.ymax = bounds.ymin;

        // Compute part bounds
        const unsigned int end = part.offset + part.length;

        for (unsigned int vertexIndex = part.offset;
             vertexIndex < end;
             ++vertexIndex)
        {
            part.boundsPart.xmin =
                MINVALUE(part.boundsPart.xmin, obj->padfX[vertexIndex]);

            part.boundsPart.xmax =
                MAXVALUE(part.boundsPart.xmax, obj->padfX[vertexIndex]);

            part.boundsPart.ymin =
                MINVALUE(part.boundsPart.ymin, obj->padfY[vertexIndex]);

            part.boundsPart.ymax =
                MAXVALUE(part.boundsPart.ymax, obj->padfY[vertexIndex]);
        }

        parts.push_back(part);
    }
}


/*
void ShapeObject::assign_old(const SHPObject* obj)
{
    if (obj != nullptr)
    {
        if (index >= 0)
        {
            destroy();
        }
        index = obj->nShapeId;
        type = obj->nSHPType;
        vertexCount = unsigned(obj->nVertices);
        if (vertexCount > 0)
        {
            vertices = new Point<double> [vertexCount];

            double *xptr = obj->padfX, *yptr = obj->padfY;
            Point<double> *pptr = vertices;
            Point<double> *pend = pptr + vertexCount;
            while (pptr < pend)
            {
                pptr->set(*xptr, *yptr);
                xptr++;
                yptr++;
                pptr++;
            }
        }
        bounds.ymin = obj->dfYMin;
        bounds.xmin = obj->dfXMin;
        bounds.ymax = obj->dfYMax;
        bounds.xmax = obj->dfXMax;

        partCount = unsigned(obj->nParts);
        int *ps = obj->panPartStart;
        int *pt = obj->panPartType;

        for (unsigned int n = 0; n < partCount; n++)
        {
            Part* part = new Part;
            part->type = *pt;
            part->offset = unsigned(*ps);
            if ((n+1) == partCount)
            {
                part->length = vertexCount - unsigned(*ps);
            }
            else
            {
                part->length = unsigned(*(ps+1) - *ps);
            }

            // assign if the part is an hole
            if (! isClockWise(*part))
            {
                part->hole = true;
            }
            else
            {
                part->hole = false;
            }

            // save bounds for each part
            part->boundsPart.ymin = bounds.ymax;
            part->boundsPart.xmin = bounds.xmax;
            part->boundsPart.ymax = bounds.ymin;
            part->boundsPart.xmax = bounds.xmin;

            for (unsigned int k = part->offset; k < part->offset + part->length; k++)
            {
                part->boundsPart.xmin = MINVALUE(part->boundsPart.xmin, obj->padfX[k]);
                part->boundsPart.xmax = MAXVALUE(part->boundsPart.xmax, obj->padfX[k]);
                part->boundsPart.ymin = MINVALUE(part->boundsPart.ymin, obj->padfY[k]);
                part->boundsPart.ymax = MAXVALUE(part->boundsPart.ymax, obj->padfY[k]);
            }
            // save part coordination
            //part->padfXPart = obj->padfX+part->offset;
            //part->padfYPart = obj->padfY+part->offset;
            parts.push_back(*part);
            ps++;
            pt++;
        }
    }
}
*/


void ShapeObject::assign(const ShapeObject& other)
{
    if (&other != this)
    {
        if (index >= 0) {
            destroy();
        }
        index = other.index;
        type = other.type;
        vertexCount = other.vertexCount;
        partCount = other.partCount;
        if (vertexCount > 0)
        {
            vertices = new Point<double> [vertexCount];
            memcpy(vertices, other.vertices, other.vertexCount * sizeof(Point<double>));
        }
        bounds = other.bounds;
        parts = other.parts;
    }
}


int ShapeObject::getIndex() const
{
    return index;
}

int ShapeObject::getType() const
{
    return type;
}

std::string ShapeObject::getTypeString() const
{
    return getShapeTypeAsString(type);
}

unsigned long ShapeObject::getVertexCount() const
{
    return vertexCount;
}

const Point<double>* ShapeObject::getVertices() const
{
    return vertices;
}

const Point<double>& ShapeObject::getVertex(unsigned int index) const
{
    return vertices[index];
}

const Box<double>& ShapeObject::getBounds() const
{
    return bounds;
}

const std::vector<ShapeObject::Part>& ShapeObject::getParts() const
{
    return parts;
}

const ShapeObject::Part& ShapeObject::getPart(unsigned int partIndex) const
{
    return parts[partIndex];
}


double ShapeObject::getTotalArea() const
{
    double totalArea = 0.0;

    for (unsigned int i = 0; i < parts.size(); ++i)
    {
        const Part &part = getPart(i);

        if (part.hole)
            totalArea -= std::abs(polygonArea(part));
        else
            totalArea += std::abs(polygonArea(part));
    }

    return totalArea;
}


double ShapeObject::polygonArea(const Part& part) const
{
    double area = 0.0;
    unsigned long i, j;

    const unsigned long offSet = part.offset;
    const unsigned long length = part.length;

    for (i = 0; i < length; i++)
    {
        j = (i + 1) % length;
        area += (vertices[i+offSet].x * vertices[j+offSet].y - vertices[j+offSet].x * vertices[i+offSet].y);
    }

    return (area * 0.5);
}


bool ShapeObject::isClockWise(const Part &part) const
{
    return polygonArea(part) < 0;
}


// ray casting / even-odd rule
bool ShapeObject::pointInPart(double x, double y, unsigned int partIndex) const
{
    const auto &part = parts[partIndex];

    if (x < part.boundsPart.xmin || x > part.boundsPart.xmax ||
        y < part.boundsPart.ymin || y > part.boundsPart.ymax)
    {
        return false;
    }

    // A polygon part must contain at least three vertices.
    if (part.length < 3)
        return false;

    const unsigned long offset = part.offset;
    const unsigned long last = offset + part.length - 1;

    bool inside = false;
    unsigned long j = last;

    for (unsigned long i = offset; i <= last; ++i)
    {
        const auto &a = vertices[i];
        const auto &b = vertices[j];

        if ((a.y < y && b.y >= y) ||
            (b.y < y && a.y >= y))
        {
            const double xIntersect =
                a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y);

            if (xIntersect < x)
                inside = !inside;
        }

        j = i;
    }

    return inside;
}


bool ShapeObject::pointInPolygon(double x, double y) const
{
    if (x < bounds.xmin || x > bounds.xmax
        || y < bounds.ymin || y > bounds.ymax)
    {
        return false;
    }

    bool inside = false;
    const unsigned int nParts = getPartCount();

    for (unsigned int partIndex = 0; partIndex < nParts; ++partIndex)
    {
        if (! pointInPart(x, y, partIndex))
            continue;

        if (parts[partIndex].hole)
            return false;

        inside = true;
    }

    return inside;
}


int ShapeObject::getOuterPartIndex(double x, double y) const
{
    if (x < bounds.xmin || x > bounds.xmax ||
        y < bounds.ymin || y > bounds.ymax)
    {
        return NODATA;
    }

    const unsigned int nrParts = getPartCount();

    for (unsigned int partIndex = 0; partIndex < nrParts; ++partIndex)
    {
        if (!parts[partIndex].hole && pointInPart(x, y, partIndex))
            return static_cast<int>(partIndex);
    }

    return NODATA;
}


std::string getShapeTypeAsString(int shapeType)
{
    std::string shape;
    switch (shapeType) {
    case SHPT_NULL:
        shape = "None";
        break;
    case SHPT_POINT:
        shape = "2D Point";
        break;
    case SHPT_ARC:
        shape = "2D Arc";
        break;
    case SHPT_POLYGON:
        shape = "2D Polygon";
        break;
    case SHPT_MULTIPOINT:
        shape = "2D Multi-point";
        break;
    case SHPT_POINTZ:
        shape = "3D Point";
        break;
    case SHPT_ARCZ:
        shape = "3D Arc";
        break;
    case SHPT_POLYGONZ:
        shape = "3D Polygon";
        break;
    case SHPT_MULTIPOINTZ:
        shape = "3D Multi-point";
        break;
    case SHPT_POINTM:
        shape = "2D Measure Point";
        break;
    case SHPT_ARCM:
        shape = "2D Measure Arc";
        break;
    case SHPT_POLYGONM:
        shape = "2D Measure Polygon";
        break;
    case SHPT_MULTIPOINTM:
        shape = "2D Measure Multi-point";
        break;
    case SHPT_MULTIPATCH:
        shape = "Multi-patch";
        break;
    default:
        shape = "Unknown";
    }
    return shape;
}

