#include "glWidget.h"
#include "viewer3D.h"

#include <QSlider>
#include <QLabel>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QHBoxLayout>


Viewer3D::Viewer3D(Crit3DGeometry *geometry)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(tr("3D view"));

    QHBoxLayout *glLayout = new QHBoxLayout;
    glWidget = new Crit3DOpenGLWidget(geometry);
    glLayout->addWidget(glWidget);
    turnSlider = verticalSlider(0, 360 * DEGREE_MULTIPLY, DEGREE_MULTIPLY, 15 * DEGREE_MULTIPLY);
    glLayout->addWidget(turnSlider);

    QHBoxLayout *rotateLayout = new QHBoxLayout;
    QLabel *rotateLabel = new QLabel("Rotation:");
    rotateLayout->addWidget(rotateLabel);
    rotateSlider = horizontalSlider(0, 360 * DEGREE_MULTIPLY, DEGREE_MULTIPLY, 15 * DEGREE_MULTIPLY);
    rotateLayout->addWidget(rotateSlider);

    QHBoxLayout *magnifyLayout = new QHBoxLayout;
    QLabel *magnifyLabel = new QLabel("Magnify: ");
    magnifyLayout->addWidget(magnifyLabel);
    magnifySlider = horizontalSlider(1, 100, 1, 5);
    magnifyLayout->addWidget(magnifySlider);

    QHBoxLayout *slopeLayout = new QHBoxLayout;
    QLabel *slopeLabel = new QLabel("Artifact slope: ");
    slopeLayout->addWidget(slopeLabel);
    slopeSlider = horizontalSlider(1, 100, 1, 5);
    slopeLayout->addWidget(slopeSlider);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addLayout(glLayout);
    mainLayout->addLayout(rotateLayout);
    mainLayout->addLayout(magnifyLayout);
    mainLayout->addLayout(slopeLayout);
    setLayout(mainLayout);

    QStatusBar *statusBar = new QStatusBar(this);
    mainLayout->addWidget(statusBar);

    connect(turnSlider, &QSlider::valueChanged, glWidget, &Crit3DOpenGLWidget::setXRotation);
    connect(glWidget, &Crit3DOpenGLWidget::xRotationChanged, turnSlider, &QSlider::setValue);
    connect(rotateSlider, &QSlider::valueChanged, glWidget, &Crit3DOpenGLWidget::setZRotation);
    connect(glWidget, &Crit3DOpenGLWidget::zRotationChanged, rotateSlider, &QSlider::setValue);
    connect(magnifySlider, &QSlider::valueChanged, glWidget, &Crit3DOpenGLWidget::setMagnify);
    connect(slopeSlider, &QSlider::valueChanged, this, &Viewer3D::on_slopeChanged);

    turnSlider->setValue(30 * DEGREE_MULTIPLY);
    rotateSlider->setValue(0 * DEGREE_MULTIPLY);
    magnifySlider->setValue(geometry->magnify() * 10);
    slopeSlider->setValue(geometry->artifactSlope());
}


QSlider* Viewer3D::verticalSlider(int minimum, int maximum, int step, int tick)
{
    QSlider *slider = new QSlider(Qt::Vertical);
    slider->setRange(minimum, maximum);
    slider->setSingleStep(step);
    slider->setPageStep(tick);
    slider->setTickInterval(tick);
    slider->setTickPosition(QSlider::TicksRight);
    return slider;
}


QSlider* Viewer3D::horizontalSlider(int minimum, int maximum, int step, int tick)
{
    QSlider *slider = new QSlider(Qt::Horizontal);
    slider->setRange(minimum, maximum);
    slider->setSingleStep(step);
    slider->setPageStep(tick);
    slider->setTickInterval(tick);
    slider->setTickPosition(QSlider::TicksBelow);
    return slider;
}


float Viewer3D::getSlope()
{
    return slopeSlider->value();
}


void Viewer3D::on_slopeChanged()
{
    emit slopeChanged();
}

