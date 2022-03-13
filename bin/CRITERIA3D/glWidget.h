#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include "geometry.h"


QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)

#define DEGREE_MULTIPLY 16


class Crit3DOpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    Crit3DOpenGLWidget(Crit3DGeometry *m_geometry, QWidget *parent = nullptr);
    ~Crit3DOpenGLWidget() override;
    void clear();

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

public slots:
    void setXRotation(int angle);
    void setZRotation(int angle);
    void setXTraslation(float traslation);
    void setYTraslation(float traslation);
    void setZoom(float zoom);
    void setMagnify(float magnify);

signals:
    void xRotationChanged(int angle);
    void zRotationChanged(int angle);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    int m_xRotation;
    int m_zRotation;
    float m_xTraslation;
    float m_yTraslation;
    float m_zoom;

    QPoint m_lastPos;

    QOpenGLBuffer m_bufferObject;
    QOpenGLShaderProgram *m_program;
    Crit3DGeometry *m_geometry;

    int m_projMatrixLoc;
    int m_mvMatrixLoc;

    QMatrix4x4 m_proj;
    QMatrix4x4 m_camera;
    QMatrix4x4 m_world;
};

#endif
