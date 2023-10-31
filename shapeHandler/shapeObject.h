/*******************************************************************
 * This code is based on shapeobject.hpp of Erik Svensson
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


#ifndef SHAPEOBJECT_H
#define SHAPEOBJECT_H

    #include <string>
    #include <vector>
    #include <shapelib/shapefil.h>

    template<typename T>
    struct Point
    {
        T x;
        T y;

        Point()
            : x(static_cast<T>(0))
            , y(static_cast<T>(0))
        {
        }

        Point(T x_, T y_)
            : x(x_)
            , y(y_)
        {
        }

        void set(T x_, T y_)
        {
            x = x_;
            y = y_;
        }
    };


    template<typename T>
    struct Box
    {
        T xmin;
        T ymin;
        T xmax;
        T ymax;

        Box()
            : xmin(static_cast<T>(0))
            , ymin(static_cast<T>(0))
            , xmax(static_cast<T>(0))
            , ymax(static_cast<T>(0))
        {
        }

        T width() const
        {
            return xmax - xmin;
        }

        T height() const
        {
            return ymax - ymin;
        }

        Point<T> center() const
        {
            T xoffset = ((xmax - xmin) / static_cast<T>(2));
            T yoffset = ((ymax - ymin) / static_cast<T>(2));
            return Point<T>(xmin + xoffset, ymin + yoffset);
        }

        void grow(T delta)
        {
            xmin -= delta;
            xmax += delta;
            ymin -= delta;
            ymax += delta;
        }

        void grow(T dx, T dy)
        {
            xmin -= dx;
            xmax += dx;
            ymin -= dy;
            ymax += dy;
        }

        void move(T x, T y)
        {
            xmin += x;
            xmax += x;
            ymin += y;
            ymax += y;
        }
    };


    class ShapeObject
    {
    public:
        struct Part
        {
            int type;
            unsigned long offset;
            unsigned long length;
            Box<double> boundsPart;
            bool hole;

            Part() : type(0), offset(0), length(0), hole(false) {}
        };

    protected:
        int                 index;
        int					type;
        unsigned long		vertexCount;
        unsigned int        partCount;
        Point<double>		*vertices;
        Box<double>			bounds;
        std::vector<Part>	parts;

        void        destroy();
        void        assign(const ShapeObject& obj);

    public:      
        ShapeObject();
        ShapeObject(const SHPObject* obj);
        ShapeObject(const ShapeObject& other);
        ~ShapeObject();

        ShapeObject& operator = (const ShapeObject& other);

        void                    assign(const SHPObject* obj);
        int                     getIndex() const;
        int                     getType() const;
        std::string             getTypeString() const;

        unsigned long           getVertexCount() const;
        const Point<double>*	getVertices() const;
        Point<double>           getVertex(unsigned int index);
        Box<double>             getBounds() const;

        std::vector<Part>		getParts() const;
        ShapeObject::Part       getPart(unsigned int indexPart) const;
        unsigned int            getPartCount() const;
        bool                    isHole(unsigned int n);
        double                  polygonArea(Part* part);
        bool                    isClockWise(Part *part);
        bool                    pointInPart(double x, double y, unsigned int indexPart);
        bool                    pointInPolygon(double x, double y);
        int                     getIndexPart(double x, double y);
    };

    std::string getShapeTypeAsString(int shapeType);


#endif // SHAPEOBJECT_H
