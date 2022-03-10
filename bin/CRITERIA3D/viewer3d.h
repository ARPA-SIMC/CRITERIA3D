#ifndef VIEWER3D_H
#define VIEWER3D_H

    #include <QWidget>
    #include "geometry.h"

    class QSlider;
    class GLWidget;

    class Viewer3D : public QWidget
    {
        Q_OBJECT

    public:
        Viewer3D(Crit3DGeometry *geometry);
        ~Viewer3D();

    protected:

    private:
        QSlider *verticalSlider(int maximumAngle);
        QSlider *horizontalSlider(int maximumAngle);

        GLWidget *glWidget;
        QSlider *xSlider;
        QSlider *ySlider;
        QSlider *zSlider;
    };

#endif
