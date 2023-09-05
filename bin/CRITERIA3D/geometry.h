#ifndef GEOMETRY_H
#define GEOMETRY_H

    #include <qopengl.h>
    #include <vector>

    #ifndef CRIT3DCOLOR_H
        #include "color.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    class Crit3DGeometry
    {
    public:
        Crit3DGeometry();

        void clear();

        const GLfloat *getVertices() const { return m_vertices.data(); }
        const GLubyte *getColors() const { return m_colors.data(); }

        long dataCount() const { return long(m_vertices.size()); }
        long vertexCount() const { return long(m_vertices.size()) / 3; }
        float defaultDistance() const { return std::max(m_dx, m_dy); }
        float magnify() const { return m_magnify; }
        int artifactSlope() const { return m_artifactSlope; }

        void setMagnify(float magnify);
        void setArtifactSlope(int artifactSlope){ m_artifactSlope = artifactSlope; }
        void setCenter(float x, float y, float z);
        void setDimension(float dx, float dy);

        void addTriangle(const gis::Crit3DPoint &p1, const gis::Crit3DPoint &p2, const gis::Crit3DPoint &p3,
                         const Crit3DColor &c1, const Crit3DColor &c2, const Crit3DColor &c3);

        void setVertexColor(int i, const Crit3DColor &color);

    private:

        void addVertex(const gis::Crit3DPoint &v);
        void addVertexColor(const Crit3DColor &color);

        std::vector<GLfloat> m_vertices;
        std::vector<GLubyte> m_colors;

        float m_dx, m_dy;
        float m_xCenter, m_yCenter, m_zCenter;
        float m_magnify;
        int m_artifactSlope = 60;
    };


#endif // GEOMETRY_H
