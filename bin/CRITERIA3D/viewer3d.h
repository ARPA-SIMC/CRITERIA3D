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
        QSlider *verticalSlider(int minimum, int maximum, int step, int tick);
        QSlider *horizontalSlider(int minimum, int maximum, int step, int tick);

        Crit3DOpenGLWidget *glWidget;
        QSlider *turnSlider;
        QSlider *rotateSlider;
        QSlider *magnifySlider;
    };

#endif
