#ifndef VIEWER3D_H
#define VIEWER3D_H

    #include <QWidget>

    class Crit3DGeometry;
    class Crit3DOpenGLWidget;
    class QSlider;

    class Viewer3D : public QWidget
    {
        Q_OBJECT

    public:
        Viewer3D(Crit3DGeometry *geometry);

    protected:

    private:
        QSlider *verticalSlider(int maximumAngle);
        QSlider *horizontalSlider(int maximumAngle);

        Crit3DOpenGLWidget *glWidget;
        QSlider *xSlider;
        QSlider *ySlider;
        QSlider *zSlider;
    };

#endif
