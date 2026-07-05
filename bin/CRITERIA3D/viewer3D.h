#ifndef VIEWER3D_H
#define VIEWER3D_H

    #include <QWidget>
    #include <QSlider>

    class Crit3DGeometry;
    class Crit3DOpenGLWidget;

    class Viewer3D : public QWidget
    {
        Q_OBJECT

    public:
        Viewer3D(Crit3DGeometry *geometry);

        Crit3DOpenGLWidget *glWidget;

        float getSlope() const {
            return slopeSlider->value();
        }

    protected:

    signals:
        void slopeChanged();

    private:
        QSlider *verticalSlider(int minimum, int maximum, int step, int tick);
        QSlider *horizontalSlider(int minimum, int maximum, int step, int tick);

        QSlider *turnSlider;
        QSlider *rotateSlider;
        QSlider *magnifySlider;
        QSlider *slopeSlider;

        void on_slopeChanged() {
            emit slopeChanged();
        }
    };

#endif
