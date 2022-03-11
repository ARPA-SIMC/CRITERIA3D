#ifndef GEOMETRY_H
#define GEOMETRY_H

    #include <qopengl.h>
    #include <QVector>
    #include <QVector3D>
    #ifndef CRIT3DCOLOR_H
        #include "color.h"
    #endif

    class Crit3DGeometry
    {
    public:
        Crit3DGeometry();

        void clear();

        const GLfloat *getData() const { return m_data.constData(); }
        const GLfloat *getColors() const { return m_colors.constData(); }

        int count() const { return m_dataCount; }
        int vertexCount() const { return m_dataCount / 3; }
        float defaultDistance() const { return std::max(m_dx, m_dy); }

        void setMagnify(float magnify) { m_magnify = magnify; }
        void setCenter(float x, float y, float z);
        void setDimension(float dx, float dy);

        void addTriangle(const QVector3D &p1, const QVector3D &p2, const QVector3D &p3,
                         const Crit3DColor &c1, const Crit3DColor &c2, const Crit3DColor &c3);

        void setVertexColor(int i, const Crit3DColor &color);

    private:

        void addVertex(const QVector3D &v);
        void addVertexColor(const Crit3DColor &color);

        QVector<GLfloat> m_data;
        QVector<GLfloat> m_colors;

        int m_dataCount;
        int m_colorCount;

        float m_dx, m_dy;
        float m_xCenter, m_yCenter, m_zCenter;
        float m_magnify;
    };


#endif // GEOMETRY_H
