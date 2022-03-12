#include "geometry.h"

Crit3DGeometry::Crit3DGeometry()
{
    this->clear();
}


void Crit3DGeometry::clear()
{
    m_xCenter = 0;
    m_yCenter = 0;
    m_zCenter = 0;
    m_dx = 0;
    m_dy = 0;
    m_magnify = 1;

    m_data.clear();
    m_colors.clear();
}


void Crit3DGeometry::setCenter(float x, float y, float z)
{
    m_xCenter = x;
    m_yCenter = y;
    m_zCenter = z;
}

void Crit3DGeometry::setDimension(float dx, float dy)
{
    m_dx = dx;
    m_dy = dy;
}

void Crit3DGeometry::addTriangle(const gis::Crit3DPoint &p1, const gis::Crit3DPoint &p2, const gis::Crit3DPoint &p3,
                                 const Crit3DColor &c1, const Crit3DColor &c2, const Crit3DColor &c3)
{
    addVertex(p1);
    addVertexColor(c1);
    addVertex(p2);
    addVertexColor(c2);
    addVertex(p3);
    addVertexColor(c3);
}

void Crit3DGeometry::addVertex(const gis::Crit3DPoint &v)
{
    m_data.push_back(v.utm.x - m_xCenter);
    m_data.push_back(v.utm.y - m_yCenter);
    m_data.push_back((v.z - m_zCenter) * m_magnify);

}

void Crit3DGeometry::addVertexColor(const Crit3DColor &color)
{
    m_colors.push_back(color.red);
    m_colors.push_back(color.green);
    m_colors.push_back(color.blue);
}

void Crit3DGeometry::setVertexColor(int i, const Crit3DColor &color)
{
    if (i > vertexCount()) return;

    m_colors[i*3] = color.red;
    m_colors[i*3+1] = color.green;
    m_colors[i*3+2] = color.blue;
}

