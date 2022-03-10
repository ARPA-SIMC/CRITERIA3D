
#include "glWidget.h"
#include "viewer3D.h"

#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>


Viewer3D::Viewer3D(Crit3DGeometry *geometry)
{
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(tr("3D view"));

    glWidget = new GLWidget(geometry);

    xSlider = verticalSlider(360);
    zSlider = horizontalSlider(360);

    QVBoxLayout *container = new QVBoxLayout;
    container->addWidget(glWidget);
    container->addWidget(zSlider);

    QHBoxLayout *mainLayout = new QHBoxLayout;
    mainLayout->addLayout(container);
    mainLayout->addWidget(xSlider);

    setLayout(mainLayout);

    connect(xSlider, &QSlider::valueChanged, glWidget, &GLWidget::setXRotation);
    connect(glWidget, &GLWidget::xRotationChanged, xSlider, &QSlider::setValue);
    connect(zSlider, &QSlider::valueChanged, glWidget, &GLWidget::setZRotation);
    connect(glWidget, &GLWidget::zRotationChanged, zSlider, &QSlider::setValue);

    xSlider->setValue(30 * DEGREE_MULTIPLY);
    zSlider->setValue(0 * DEGREE_MULTIPLY);
}


Viewer3D::~Viewer3D()
{
    glWidget->clear();
}


QSlider *Viewer3D::verticalSlider(int maximumAngle)
{
    QSlider *slider = new QSlider(Qt::Vertical);
    slider->setRange(0, maximumAngle * DEGREE_MULTIPLY);
    slider->setSingleStep(DEGREE_MULTIPLY);
    slider->setPageStep(15 * DEGREE_MULTIPLY);
    slider->setTickInterval(15 * DEGREE_MULTIPLY);
    slider->setTickPosition(QSlider::TicksRight);
    return slider;
}


QSlider *Viewer3D::horizontalSlider(int maximumAngle)
{
    QSlider *slider = new QSlider(Qt::Horizontal);
    slider->setRange(0, maximumAngle * DEGREE_MULTIPLY);
    slider->setSingleStep(DEGREE_MULTIPLY);
    slider->setPageStep(15 * DEGREE_MULTIPLY);
    slider->setTickInterval(15 * DEGREE_MULTIPLY);
    slider->setTickPosition(QSlider::TicksBelow);
    return slider;
}

